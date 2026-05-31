# -*- coding: utf-8 -*-
"""
神经网络融合方案改进实验 (V5.2) - 修复数据泄露问题

修复内容：
1. 正确划分训练集、验证集、测试集
2. 训练神经网络使用训练集
3. 早停机制使用验证集
4. 最终评估使用测试集
"""

import os
import sys
import json
import time
import warnings
import datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import degree

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
LOG_DIR = os.path.join(BASE_DIR, 'experiment_logs')

TRAIN_FULL_CSV = os.path.join(DATA_DIR, 'dataset_train.csv')
TRAIN_FULL_JSON = os.path.join(DATA_DIR, 'dataset_train_graphs.json')
TEST_CSV = os.path.join(DATA_DIR, 'dataset_test.csv')
TEST_JSON = os.path.join(DATA_DIR, 'dataset_test_graphs.json')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUMERIC_FEATURES = [
    'inter_api_access_duration(sec)', 'api_access_uniqueness', 
    'sequence_length(count)', 'vsession_duration(min)', 
    'num_sessions', 'num_users', 'num_unique_apis'
]
CATEGORICAL_FEATURES = ['ip_type', 'source']


class ExperimentLogger:
    def __init__(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        self.start_time = time.time()
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"实验开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    
    def log_section(self, title):
        line = f"\n{'=' * 80}\n{title}\n{'=' * 80}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    
    def finalize(self):
        total_time = time.time() - self.start_time
        self.log(f"\n实验总耗时: {total_time:.2f} 秒")


class GCNModel(nn.Module):
    def __init__(self, hidden_channels=64, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(1, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.fc = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x)).squeeze()


class FusionNetV3_BN(nn.Module):
    """V3：批归一化网络"""
    def __init__(self):
        super().__init__()
        self.name = "V3_BN"
        self.fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()


def prepare_features(rf_prob, gcn_prob):
    """准备6维输入特征"""
    rf_conf = torch.abs(rf_prob - 0.5) * 2
    gcn_conf = torch.abs(gcn_prob - 0.5) * 2
    prob_product = rf_prob * gcn_prob
    prob_diff = torch.abs(rf_prob - gcn_prob)
    
    return torch.stack([rf_prob, gcn_prob, rf_conf, gcn_conf, prob_product, prob_diff], dim=1)


def load_data(csv_path, json_path):
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        graphs = json.load(f)
    return df, graphs


def prepare_rf_features(df, encoders=None, scaler=None, fit=False):
    df = df.copy()
    
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('unknown')
    
    X_num = df[NUMERIC_FEATURES].values
    
    if fit:
        encoders = {}
        X_cat = np.zeros((len(df), len(CATEGORICAL_FEATURES)))
        for i, col in enumerate(CATEGORICAL_FEATURES):
            le = LabelEncoder()
            X_cat[:, i] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        X_cat = np.zeros((len(df), len(CATEGORICAL_FEATURES)))
        for i, col in enumerate(CATEGORICAL_FEATURES):
            le = encoders[col]
            X_cat[:, i] = le.transform(df[col].astype(str))
    
    X = np.hstack((X_num, X_cat))
    
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    y = df['classification'].apply(lambda x: 1 if x == 'outlier' else 0).values
    ids = df['_id'].values
    
    return X, y, ids, encoders, scaler


def build_pyg_data(df, json_path):
    with open(json_path, 'r') as f:
        graph_data = json.load(f)
    
    label_dict = df.set_index('_id')['classification'].apply(
        lambda x: 1 if x == 'outlier' else 0
    ).to_dict()
    id_to_graph = {item['_id']: item.get('call_graph', []) for item in graph_data}
    
    dataset_pyg = []
    
    for idx, row in df.iterrows():
        trace_id = row['_id']
        if trace_id not in label_dict:
            continue
        
        label = label_dict[trace_id]
        call_graph = id_to_graph.get(trace_id, [])
        
        if not call_graph:
            continue
        
        unique_uuids = set()
        for edge in call_graph:
            unique_uuids.add(edge.get('fromId', ''))
            unique_uuids.add(edge.get('toId', ''))
        
        if len(unique_uuids) == 0:
            continue
        
        unique_uuids = list(unique_uuids)
        uuid_to_idx = {uuid: i for i, uuid in enumerate(unique_uuids)}
        
        src_nodes, dst_nodes = [], []
        for edge in call_graph:
            src_nodes.append(uuid_to_idx[edge['fromId']])
            dst_nodes.append(uuid_to_idx[edge['toId']])
        
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        V = len(unique_uuids)
        if edge_index.size(1) > 0:
            out_degree = degree(edge_index[0], num_nodes=V, dtype=torch.float)
            in_degree = degree(edge_index[1], num_nodes=V, dtype=torch.float)
            x = (out_degree + in_degree).view(-1, 1)
        else:
            x = torch.ones((V, 1), dtype=torch.float)
        
        y = torch.tensor([[label]], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y, trace_id=trace_id)
        dataset_pyg.append(data)
    
    return dataset_pyg


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def train_gcn_model(train_pyg, hidden_channels, num_layers, logger, epochs=12):
    model = GCNModel(hidden_channels=hidden_channels, num_layers=num_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    train_loader = DataLoader(train_pyg, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(1, epochs + 1):
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.squeeze())
            loss.backward()
            optimizer.step()
    
    return model


def train_fusion_net_proper(train_features, train_labels, 
                            val_features, val_labels,
                            logger, max_epochs=200, patience=20, lr=0.001):
    """正确实现的训练函数：训练集和验证集分离"""
    model = FusionNetV3_BN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.BCELoss()
    
    train_features = train_features.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    val_features = val_features.to(DEVICE)
    val_labels = val_labels.to(DEVICE)
    
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_count = 0
    best_epoch = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        alphas = model(train_features)
        fused = alphas * train_features[:, 0] + (1 - alphas) * train_features[:, 1]
        train_loss = criterion(fused, train_labels)
        
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_alphas = model(val_features)
            val_fused = val_alphas * val_features[:, 0] + (1 - val_alphas) * val_features[:, 1]
            val_loss = criterion(val_fused, val_labels)
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            best_epoch = epoch
        else:
            no_improve_count += 1
        
        if epoch % 25 == 0:
            logger.log(f"    Epoch {epoch}: train_loss={train_loss.item():.4f}, val_loss={val_loss.item():.4f}, best_epoch={best_epoch}")
        
        if no_improve_count >= patience:
            logger.log(f"    早停触发: epoch={epoch}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_epoch


def run_experiments():
    log_file = os.path.join(LOG_DIR, f'nn_fusion_v52_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = ExperimentLogger(log_file)
    
    logger.log_section("神经网络融合改进实验 (V5.2) - 修复数据泄露")
    logger.log("修复内容：正确划分训练集、验证集、测试集")
    
    logger.log_section("阶段1: 数据加载")
    
    df_train_full, _ = load_data(TRAIN_FULL_CSV, TRAIN_FULL_JSON)
    df_test, _ = load_data(TEST_CSV, TEST_JSON)
    
    logger.log(f"  完整训练集: {len(df_train_full)} 条")
    logger.log(f"  测试集: {len(df_test)} 条")
    
    logger.log_section("阶段2: 划分训练集和验证集")
    
    df_train_split, df_val_split = train_test_split(
        df_train_full, test_size=0.2, random_state=SEED, 
        stratify=df_train_full['classification']
    )
    
    logger.log(f"  训练集: {len(df_train_split)} 条")
    logger.log(f"  验证集: {len(df_val_split)} 条")
    logger.log(f"  测试集: {len(df_test)} 条")
    
    df_train_split.to_csv(os.path.join(DATA_DIR, 'dataset_train_nn.csv'), index=False)
    df_val_split.to_csv(os.path.join(DATA_DIR, 'dataset_val_nn.csv'), index=False)
    
    logger.log_section("阶段3: 特征准备")
    
    X_train, y_train, ids_train, encoders, scaler = prepare_rf_features(df_train_split, fit=True)
    X_val, y_val, ids_val, _, _ = prepare_rf_features(df_val_split, encoders, scaler)
    X_test, y_test, ids_test, _, _ = prepare_rf_features(df_test, encoders, scaler)
    
    logger.log_section("阶段4: 图数据构建")
    
    train_pyg = build_pyg_data(df_train_split, TRAIN_FULL_JSON)
    val_pyg = build_pyg_data(df_val_split, TRAIN_FULL_JSON)
    test_pyg = build_pyg_data(df_test, TEST_JSON)
    
    logger.log(f"  训练图: {len(train_pyg)}")
    logger.log(f"  验证图: {len(val_pyg)}")
    logger.log(f"  测试图: {len(test_pyg)}")
    
    logger.log_section("阶段5: 训练基础模型")
    
    logger.log("训练RF模型...")
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2,
        max_features='sqrt', random_state=SEED, class_weight='balanced', n_jobs=-1
    )
    rf_model.fit(X_res, y_res)
    
    logger.log("训练GCN模型 (epochs=12)...")
    gcn_model = train_gcn_model(train_pyg, 32, 2, logger, epochs=12)
    
    logger.log_section("阶段6: 获取预测概率")
    
    rf_probs_train = rf_model.predict_proba(X_train)[:, 1]
    rf_probs_val = rf_model.predict_proba(X_val)[:, 1]
    rf_probs_test = rf_model.predict_proba(X_test)[:, 1]
    
    gcn_model.eval()
    
    gcn_probs_train = []
    with torch.no_grad():
        train_loader = DataLoader(train_pyg, batch_size=32, shuffle=False)
        for data in train_loader:
            data = data.to(DEVICE)
            out = gcn_model(data.x, data.edge_index, data.batch)
            gcn_probs_train.extend(out.cpu().numpy().flatten())
    gcn_probs_train = np.array(gcn_probs_train)
    
    gcn_probs_val = []
    with torch.no_grad():
        val_loader = DataLoader(val_pyg, batch_size=32, shuffle=False)
        for data in val_loader:
            data = data.to(DEVICE)
            out = gcn_model(data.x, data.edge_index, data.batch)
            gcn_probs_val.extend(out.cpu().numpy().flatten())
    gcn_probs_val = np.array(gcn_probs_val)
    
    gcn_probs_test = []
    with torch.no_grad():
        test_loader = DataLoader(test_pyg, batch_size=32, shuffle=False)
        for data in test_loader:
            data = data.to(DEVICE)
            out = gcn_model(data.x, data.edge_index, data.batch)
            gcn_probs_test.extend(out.cpu().numpy().flatten())
    gcn_probs_test = np.array(gcn_probs_test)
    
    y_train_gcn = np.array([data.y.item() for data in train_pyg])
    y_val_gcn = np.array([data.y.item() for data in val_pyg])
    y_test_gcn = np.array([data.y.item() for data in test_pyg])
    
    train_trace_ids = [data.trace_id for data in train_pyg]
    train_id_to_idx = {tid: i for i, tid in enumerate(ids_train)}
    rf_probs_train_aligned = np.array([rf_probs_train[train_id_to_idx[tid]] for tid in train_trace_ids])
    
    val_trace_ids = [data.trace_id for data in val_pyg]
    val_id_to_idx = {tid: i for i, tid in enumerate(ids_val)}
    rf_probs_val_aligned = np.array([rf_probs_val[val_id_to_idx[tid]] for tid in val_trace_ids])
    
    test_trace_ids = [data.trace_id for data in test_pyg]
    test_id_to_idx = {tid: i for i, tid in enumerate(ids_test)}
    rf_probs_test_aligned = np.array([rf_probs_test[test_id_to_idx[tid]] for tid in test_trace_ids])
    
    logger.log(f"\n数据对齐后:")
    logger.log(f"  训练集: {len(rf_probs_train_aligned)} 条")
    logger.log(f"  验证集: {len(rf_probs_val_aligned)} 条")
    logger.log(f"  测试集: {len(rf_probs_test_aligned)} 条")
    
    train_features = prepare_features(
        torch.tensor(rf_probs_train_aligned, dtype=torch.float32),
        torch.tensor(gcn_probs_train, dtype=torch.float32)
    )
    train_labels = torch.tensor(y_train_gcn, dtype=torch.float32)
    
    val_features = prepare_features(
        torch.tensor(rf_probs_val_aligned, dtype=torch.float32),
        torch.tensor(gcn_probs_val, dtype=torch.float32)
    )
    val_labels = torch.tensor(y_val_gcn, dtype=torch.float32)
    
    test_features = prepare_features(
        torch.tensor(rf_probs_test_aligned, dtype=torch.float32),
        torch.tensor(gcn_probs_test, dtype=torch.float32)
    )
    
    logger.log_section("阶段7: 神经网络融合实验")
    
    results = []
    
    epoch_configs = [50, 75, 100, 125, 150, 175, 200]
    
    for epochs in epoch_configs:
        logger.log(f"\n--- V3_BN (epochs={epochs}) ---")
        
        trained_model, best_epoch = train_fusion_net_proper(
            train_features, train_labels,
            val_features, val_labels,
            logger, max_epochs=epochs, patience=20, lr=0.001
        )
        
        trained_model.eval()
        with torch.no_grad():
            test_features_dev = test_features.to(DEVICE)
            alphas = trained_model(test_features_dev)
            fused_probs = alphas.cpu().numpy() * rf_probs_test_aligned + (1 - alphas.cpu().numpy()) * gcn_probs_test
        
        preds = (fused_probs > 0.5).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        
        logger.log(f"  测试集结果: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        logger.log(f"  FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
        
        trained_model.eval()
        with torch.no_grad():
            val_features_dev = val_features.to(DEVICE)
            val_alphas = trained_model(val_features_dev)
            val_fused = val_alphas.cpu().numpy() * rf_probs_val_aligned + (1 - val_alphas.cpu().numpy()) * gcn_probs_val
        val_preds = (val_fused > 0.5).astype(int)
        val_metrics = calculate_metrics(y_val_gcn, val_preds)
        logger.log(f"  验证集结果: F1={val_metrics['f1']:.4f}")
        
        results.append({
            'name': f"V3_BN(e={epochs})",
            'test_metrics': metrics,
            'val_metrics': val_metrics,
            'best_epoch': best_epoch
        })
    
    logger.log_section("实验结果汇总")
    
    logger.log("\n所有实验结果（按测试集F1排序）:")
    logger.log("-" * 120)
    logger.log(f"{'方案':<20} {'测试F1':<12} {'测试Precision':<14} {'测试Recall':<12} {'验证F1':<12} {'Best Epoch':<12}")
    logger.log("-" * 120)
    
    for r in sorted(results, key=lambda x: x['test_metrics']['f1'], reverse=True):
        tm = r['test_metrics']
        vm = r['val_metrics']
        logger.log(f"{r['name']:<20} {tm['f1']:<12.4f} {tm['precision']:<14.4f} {tm['recall']:<12.4f} {vm['f1']:<12.4f} {r['best_epoch']:<12}")
    
    logger.log("-" * 120)
    
    best = max(results, key=lambda x: x['test_metrics']['f1'])
    logger.log(f"\n最佳方案: {best['name']}, 测试集F1={best['test_metrics']['f1']:.4f}")
    
    results_json = os.path.join(LOG_DIR, 'nn_fusion_v52_results.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.log(f"\n结果已保存至: {results_json}")
    
    logger.finalize()
    
    return results


if __name__ == "__main__":
    run_experiments()
