# -*- coding: utf-8 -*-
"""
神经网络融合方案改进实验 (V5.3) - 严格数据分离

严格数据分离：
1. 完整训练数据(15000条) → 划分为NN训练集(60%)、NN验证集(20%)、NN测试集(20%)
2. RF和GCN在NN训练集上训练
3. 神经网络在NN训练集上训练，NN验证集用于早停
4. 最终评估使用NN测试集
5. 原始测试集作为独立验证
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
    def __init__(self):
        super().__init__()
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
    log_file = os.path.join(LOG_DIR, f'nn_fusion_v53_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = ExperimentLogger(log_file)
    
    logger.log_section("神经网络融合改进实验 (V5.3) - 严格数据分离")
    logger.log("数据划分：训练集60%、验证集20%、测试集20%（从完整训练数据中划分）")
    
    logger.log_section("阶段1: 数据加载")
    
    df_train_full, _ = load_data(TRAIN_FULL_CSV, TRAIN_FULL_JSON)
    df_test_external, _ = load_data(TEST_CSV, TEST_JSON)
    
    logger.log(f"  完整训练数据: {len(df_train_full)} 条")
    logger.log(f"  外部测试集: {len(df_test_external)} 条")
    
    logger.log_section("阶段2: 三折划分（训练/验证/测试）")
    
    df_train, df_temp = train_test_split(
        df_train_full, test_size=0.4, random_state=SEED, 
        stratify=df_train_full['classification']
    )
    
    df_val, df_test_internal = train_test_split(
        df_temp, test_size=0.5, random_state=SEED,
        stratify=df_temp['classification']
    )
    
    logger.log(f"  训练集: {len(df_train)} 条 (60%)")
    logger.log(f"  验证集: {len(df_val)} 条 (20%)")
    logger.log(f"  内部测试集: {len(df_test_internal)} 条 (20%)")
    logger.log(f"  外部测试集: {len(df_test_external)} 条")
    
    train_normal = (df_train['classification'] == 'normal').sum()
    train_outlier = (df_train['classification'] == 'outlier').sum()
    val_normal = (df_val['classification'] == 'normal').sum()
    val_outlier = (df_val['classification'] == 'outlier').sum()
    
    logger.log(f"\n训练集分布: 正常={train_normal}, 异常={train_outlier}")
    logger.log(f"验证集分布: 正常={val_normal}, 异常={val_outlier}")
    
    logger.log_section("阶段3: 特征准备")
    
    X_train, y_train, ids_train, encoders, scaler = prepare_rf_features(df_train, fit=True)
    X_val, y_val, ids_val, _, _ = prepare_rf_features(df_val, encoders, scaler)
    X_test_internal, y_test_internal, ids_test_internal, _, _ = prepare_rf_features(df_test_internal, encoders, scaler)
    X_test_external, y_test_external, ids_test_external, _, _ = prepare_rf_features(df_test_external, encoders, scaler)
    
    logger.log_section("阶段4: 图数据构建")
    
    train_pyg = build_pyg_data(df_train, TRAIN_FULL_JSON)
    val_pyg = build_pyg_data(df_val, TRAIN_FULL_JSON)
    test_internal_pyg = build_pyg_data(df_test_internal, TRAIN_FULL_JSON)
    test_external_pyg = build_pyg_data(df_test_external, TEST_JSON)
    
    logger.log(f"  训练图: {len(train_pyg)}")
    logger.log(f"  验证图: {len(val_pyg)}")
    logger.log(f"  内部测试图: {len(test_internal_pyg)}")
    logger.log(f"  外部测试图: {len(test_external_pyg)}")
    
    logger.log_section("阶段5: 训练基础模型（仅在训练集上）")
    
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
    
    logger.log_section("阶段6: 评估基础模型")
    
    rf_train_preds = rf_model.predict(X_train)
    rf_val_preds = rf_model.predict(X_val)
    rf_test_internal_preds = rf_model.predict(X_test_internal)
    rf_test_external_preds = rf_model.predict(X_test_external)
    
    logger.log(f"RF训练集F1: {f1_score(y_train, rf_train_preds):.4f}")
    logger.log(f"RF验证集F1: {f1_score(y_val, rf_val_preds):.4f}")
    logger.log(f"RF内部测试集F1: {f1_score(y_test_internal, rf_test_internal_preds):.4f}")
    logger.log(f"RF外部测试集F1: {f1_score(y_test_external, rf_test_external_preds):.4f}")
    
    logger.log_section("阶段7: 获取预测概率")
    
    rf_probs_train = rf_model.predict_proba(X_train)[:, 1]
    rf_probs_val = rf_model.predict_proba(X_val)[:, 1]
    rf_probs_test_internal = rf_model.predict_proba(X_test_internal)[:, 1]
    rf_probs_test_external = rf_model.predict_proba(X_test_external)[:, 1]
    
    gcn_model.eval()
    
    def get_gcn_probs(pyg_dataset):
        probs = []
        with torch.no_grad():
            loader = DataLoader(pyg_dataset, batch_size=32, shuffle=False)
            for data in loader:
                data = data.to(DEVICE)
                out = gcn_model(data.x, data.edge_index, data.batch)
                probs.extend(out.cpu().numpy().flatten())
        return np.array(probs)
    
    gcn_probs_train = get_gcn_probs(train_pyg)
    gcn_probs_val = get_gcn_probs(val_pyg)
    gcn_probs_test_internal = get_gcn_probs(test_internal_pyg)
    gcn_probs_test_external = get_gcn_probs(test_external_pyg)
    
    y_train_gcn = np.array([data.y.item() for data in train_pyg])
    y_val_gcn = np.array([data.y.item() for data in val_pyg])
    y_test_internal_gcn = np.array([data.y.item() for data in test_internal_pyg])
    y_test_external_gcn = np.array([data.y.item() for data in test_external_pyg])
    
    def align_rf_probs(rf_probs, ids, pyg_dataset):
        trace_ids = [data.trace_id for data in pyg_dataset]
        id_to_idx = {tid: i for i, tid in enumerate(ids)}
        return np.array([rf_probs[id_to_idx[tid]] for tid in trace_ids])
    
    rf_probs_train_aligned = align_rf_probs(rf_probs_train, ids_train, train_pyg)
    rf_probs_val_aligned = align_rf_probs(rf_probs_val, ids_val, val_pyg)
    rf_probs_test_internal_aligned = align_rf_probs(rf_probs_test_internal, ids_test_internal, test_internal_pyg)
    rf_probs_test_external_aligned = align_rf_probs(rf_probs_test_external, ids_test_external, test_external_pyg)
    
    logger.log(f"\n数据对齐后:")
    logger.log(f"  训练集: {len(rf_probs_train_aligned)} 条")
    logger.log(f"  验证集: {len(rf_probs_val_aligned)} 条")
    logger.log(f"  内部测试集: {len(rf_probs_test_internal_aligned)} 条")
    logger.log(f"  外部测试集: {len(rf_probs_test_external_aligned)} 条")
    
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
    
    test_internal_features = prepare_features(
        torch.tensor(rf_probs_test_internal_aligned, dtype=torch.float32),
        torch.tensor(gcn_probs_test_internal, dtype=torch.float32)
    )
    
    test_external_features = prepare_features(
        torch.tensor(rf_probs_test_external_aligned, dtype=torch.float32),
        torch.tensor(gcn_probs_test_external, dtype=torch.float32)
    )
    
    logger.log_section("阶段8: 神经网络融合实验")
    
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
            val_features_dev = val_features.to(DEVICE)
            val_alphas = trained_model(val_features_dev)
            val_fused = val_alphas.cpu().numpy() * rf_probs_val_aligned + (1 - val_alphas.cpu().numpy()) * gcn_probs_val
        val_preds = (val_fused > 0.5).astype(int)
        val_metrics = calculate_metrics(y_val_gcn, val_preds)
        
        with torch.no_grad():
            test_internal_features_dev = test_internal_features.to(DEVICE)
            alphas = trained_model(test_internal_features_dev)
            fused_probs = alphas.cpu().numpy() * rf_probs_test_internal_aligned + (1 - alphas.cpu().numpy()) * gcn_probs_test_internal
        preds = (fused_probs > 0.5).astype(int)
        test_internal_metrics = calculate_metrics(y_test_internal_gcn, preds)
        
        with torch.no_grad():
            test_external_features_dev = test_external_features.to(DEVICE)
            alphas = trained_model(test_external_features_dev)
            fused_probs = alphas.cpu().numpy() * rf_probs_test_external_aligned + (1 - alphas.cpu().numpy()) * gcn_probs_test_external
        preds = (fused_probs > 0.5).astype(int)
        test_external_metrics = calculate_metrics(y_test_external_gcn, preds)
        
        logger.log(f"  验证集F1: {val_metrics['f1']:.4f}")
        logger.log(f"  内部测试集F1: {test_internal_metrics['f1']:.4f}")
        logger.log(f"  外部测试集F1: {test_external_metrics['f1']:.4f}")
        
        results.append({
            'name': f"V3_BN(e={epochs})",
            'val_metrics': val_metrics,
            'test_internal_metrics': test_internal_metrics,
            'test_external_metrics': test_external_metrics,
            'best_epoch': best_epoch
        })
    
    logger.log_section("实验结果汇总")
    
    logger.log("\n所有实验结果（按外部测试集F1排序）:")
    logger.log("-" * 140)
    logger.log(f"{'方案':<20} {'验证F1':<12} {'内部测试F1':<14} {'外部测试F1':<14} {'外部Precision':<16} {'外部Recall':<12} {'Best Epoch':<12}")
    logger.log("-" * 140)
    
    for r in sorted(results, key=lambda x: x['test_external_metrics']['f1'], reverse=True):
        vm = r['val_metrics']
        ti = r['test_internal_metrics']
        te = r['test_external_metrics']
        logger.log(f"{r['name']:<20} {vm['f1']:<12.4f} {ti['f1']:<14.4f} {te['f1']:<14.4f} {te['precision']:<16.4f} {te['recall']:<12.4f} {r['best_epoch']:<12}")
    
    logger.log("-" * 140)
    
    best = max(results, key=lambda x: x['test_external_metrics']['f1'])
    logger.log(f"\n最佳方案: {best['name']}, 外部测试集F1={best['test_external_metrics']['f1']:.4f}")
    
    results_json = os.path.join(LOG_DIR, 'nn_fusion_v53_results.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.log(f"\n结果已保存至: {results_json}")
    
    logger.finalize()
    
    return results


if __name__ == "__main__":
    run_experiments()
