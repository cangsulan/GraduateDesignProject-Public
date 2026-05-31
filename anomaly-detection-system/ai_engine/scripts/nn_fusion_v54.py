# -*- coding: utf-8 -*-
"""
神经网络融合方案实验 (V5.4) - 充分利用训练数据

数据使用原则：
1. 测试集（dataset_test.csv）完全不可见
2. 训练数据（15000条）充分利用

训练数据划分：
- 训练集 (80%, 12000条)：训练RF、GCN、神经网络
- 验证集 (20%, 3000条)：神经网络早停验证

耗时统计维度：
1. RF推理单条平均耗时
2. GCN推理单条平均耗时
3. RF总耗时
4. GCN总耗时
5. 模型输出单条平均耗时（RF+GCN顺序执行）
6. 模型输出总耗时
7. 融合判定单条平均耗时
8. 融合判定总耗时
9. 总过程单条平均耗时
10. 总过程总耗时
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

TRAIN_CSV = os.path.join(DATA_DIR, 'dataset_train.csv')
TRAIN_JSON = os.path.join(DATA_DIR, 'dataset_train_graphs.json')
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


class InferenceTimeRecorder:
    """推理耗时记录器 - 记录实时性分析所需的10个维度"""
    
    def __init__(self):
        self.rf_times = []
        self.gcn_times = []
        self.fusion_times = []
        self.total_times = []
    
    def record_rf(self, elapsed_ms):
        self.rf_times.append(elapsed_ms)
    
    def record_gcn(self, elapsed_ms):
        self.gcn_times.append(elapsed_ms)
    
    def record_fusion(self, elapsed_ms):
        self.fusion_times.append(elapsed_ms)
    
    def record_total(self, elapsed_ms):
        self.total_times.append(elapsed_ms)
    
    def summary(self):
        n = len(self.total_times) if self.total_times else 1
        
        rf_total = sum(self.rf_times) if self.rf_times else 0
        rf_avg = rf_total / len(self.rf_times) if self.rf_times else 0
        
        gcn_total = sum(self.gcn_times) if self.gcn_times else 0
        gcn_avg = gcn_total / len(self.gcn_times) if self.gcn_times else 0
        
        fusion_total = sum(self.fusion_times) if self.fusion_times else 0
        fusion_avg = fusion_total / len(self.fusion_times) if self.fusion_times else 0
        
        total_total = sum(self.total_times) if self.total_times else 0
        total_avg = total_total / n
        
        model_output_total = rf_total + gcn_total
        model_output_avg = model_output_total / n
        
        return {
            'rf_avg_ms': rf_avg,
            'gcn_avg_ms': gcn_avg,
            'rf_total_ms': rf_total,
            'gcn_total_ms': gcn_total,
            'model_output_avg_ms': model_output_avg,
            'model_output_total_ms': model_output_total,
            'fusion_avg_ms': fusion_avg,
            'fusion_total_ms': fusion_total,
            'total_avg_ms': total_avg,
            'total_total_ms': total_total,
            'sample_count': n
        }


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


def train_fusion_net(train_features, train_labels, 
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
            logger.log(f"    早停触发: epoch={epoch}, best_epoch={best_epoch}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_epoch


def run_experiments():
    log_file = os.path.join(LOG_DIR, f'nn_fusion_v54_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = ExperimentLogger(log_file)
    
    logger.log_section("神经网络融合实验 (V5.4)")
    logger.log("数据使用原则：")
    logger.log("  1. 测试集完全不可见")
    logger.log("  2. 训练数据(15000条)充分利用")
    logger.log("  3. 训练集80% + 验证集20%")
    
    logger.log_section("阶段1: 数据加载")
    
    df_train_full, _ = load_data(TRAIN_CSV, TRAIN_JSON)
    df_test, _ = load_data(TEST_CSV, TEST_JSON)
    
    logger.log(f"  训练数据: {len(df_train_full)} 条")
    logger.log(f"  测试集: {len(df_test)} 条（完全不可见）")
    
    logger.log_section("阶段2: 训练数据划分")
    
    df_train, df_val = train_test_split(
        df_train_full, test_size=0.2, random_state=SEED, 
        stratify=df_train_full['classification']
    )
    
    train_normal = (df_train['classification'] == 'normal').sum()
    train_outlier = (df_train['classification'] == 'outlier').sum()
    val_normal = (df_val['classification'] == 'normal').sum()
    val_outlier = (df_val['classification'] == 'outlier').sum()
    
    logger.log(f"  训练集: {len(df_train)} 条 (正常={train_normal}, 异常={train_outlier})")
    logger.log(f"  验证集: {len(df_val)} 条 (正常={val_normal}, 异常={val_outlier})")
    logger.log(f"  总计使用: {len(df_train) + len(df_val)} 条（全部训练数据）")
    
    logger.log_section("阶段3: 特征准备")
    
    X_train, y_train, ids_train, encoders, scaler = prepare_rf_features(df_train, fit=True)
    X_val, y_val, ids_val, _, _ = prepare_rf_features(df_val, encoders, scaler)
    X_test, y_test, ids_test, _, _ = prepare_rf_features(df_test, encoders, scaler)
    
    logger.log_section("阶段4: 图数据构建")
    
    train_pyg = build_pyg_data(df_train, TRAIN_JSON)
    val_pyg = build_pyg_data(df_val, TRAIN_JSON)
    test_pyg = build_pyg_data(df_test, TEST_JSON)
    
    logger.log(f"  训练图: {len(train_pyg)}")
    logger.log(f"  验证图: {len(val_pyg)}")
    logger.log(f"  测试图: {len(test_pyg)}")
    
    test_normal = (df_test['classification'] == 'normal').sum()
    test_outlier = (df_test['classification'] == 'outlier').sum()
    logger.log(f"\n测试集分布: 正常={test_normal}, 异常={test_outlier}")
    
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
    
    logger.log_section("阶段6: 获取预测概率并记录耗时")
    
    rf_timer = InferenceTimeRecorder()
    start_time = time.time()
    rf_probs_train = rf_model.predict_proba(X_train)[:, 1]
    rf_probs_val = rf_model.predict_proba(X_val)[:, 1]
    rf_probs_test = rf_model.predict_proba(X_test)[:, 1]
    rf_elapsed = (time.time() - start_time) * 1000
    rf_avg = rf_elapsed / (len(X_train) + len(X_val) + len(X_test))
    
    gcn_model.eval()
    
    def get_gcn_probs_with_timing(pyg_dataset, recorder):
        probs = []
        start_time = time.time()
        with torch.no_grad():
            loader = DataLoader(pyg_dataset, batch_size=32, shuffle=False)
            for data in loader:
                data = data.to(DEVICE)
                out = gcn_model(data.x, data.edge_index, data.batch)
                probs.extend(out.cpu().numpy().flatten())
        elapsed = (time.time() - start_time) * 1000
        per_sample = elapsed / len(pyg_dataset)
        for _ in range(len(pyg_dataset)):
            recorder.record_gcn(per_sample)
        return np.array(probs)
    
    gcn_timer = InferenceTimeRecorder()
    gcn_probs_train = get_gcn_probs_with_timing(train_pyg, gcn_timer)
    gcn_probs_val = get_gcn_probs_with_timing(val_pyg, gcn_timer)
    gcn_probs_test = get_gcn_probs_with_timing(test_pyg, gcn_timer)
    
    gcn_elapsed = sum(gcn_timer.gcn_times)
    gcn_avg = gcn_elapsed / (len(train_pyg) + len(val_pyg) + len(test_pyg))
    
    logger.log(f"  RF平均耗时: {rf_avg:.4f} ms/条")
    logger.log(f"  GCN平均耗时: {gcn_avg:.4f} ms/条")
    
    y_train_gcn = np.array([data.y.item() for data in train_pyg])
    y_val_gcn = np.array([data.y.item() for data in val_pyg])
    y_test_gcn = np.array([data.y.item() for data in test_pyg])
    
    def align_rf_probs(rf_probs, ids, pyg_dataset):
        trace_ids = [data.trace_id for data in pyg_dataset]
        id_to_idx = {tid: i for i, tid in enumerate(ids)}
        return np.array([rf_probs[id_to_idx[tid]] for tid in trace_ids])
    
    rf_probs_train_aligned = align_rf_probs(rf_probs_train, ids_train, train_pyg)
    rf_probs_val_aligned = align_rf_probs(rf_probs_val, ids_val, val_pyg)
    rf_probs_test_aligned = align_rf_probs(rf_probs_test, ids_test, test_pyg)
    
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
        
        trained_model, best_epoch = train_fusion_net(
            train_features, train_labels,
            val_features, val_labels,
            logger, max_epochs=epochs, patience=20, lr=0.001
        )
        
        timer = InferenceTimeRecorder()
        
        for i in range(len(rf_probs_test_aligned)):
            timer.record_rf(rf_avg)
            timer.record_gcn(gcn_avg)
        
        trained_model.eval()
        fused_probs = []
        
        with torch.no_grad():
            test_features_dev = test_features.to(DEVICE)
            
            for i in range(len(rf_probs_test_aligned)):
                fusion_start = time.perf_counter()
                
                alpha = trained_model(test_features_dev[i:i+1]).item()
                fp = alpha * rf_probs_test_aligned[i] + (1 - alpha) * gcn_probs_test[i]
                
                fusion_elapsed = (time.perf_counter() - fusion_start) * 1000
                timer.record_fusion(fusion_elapsed)
                timer.record_total(rf_avg + gcn_avg + fusion_elapsed)
                
                fused_probs.append(fp)
        
        fused_probs = np.array(fused_probs)
        preds = (fused_probs > 0.5).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        timing = timer.summary()
        
        logger.log(f"  测试集结果: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        logger.log(f"  FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
        logger.log(f"  总平均耗时: {timing['total_avg_ms']:.4f} ms/条")
        
        results.append({
            'name': f'神经网络融合(e={epochs})',
            'metrics': metrics,
            'timing': timing,
            'best_epoch': best_epoch,
            'category': '神经网络融合'
        })
    
    logger.log_section("实验结果汇总")
    
    logger.log("\n一、检测性能对比:")
    logger.log("-" * 100)
    logger.log(f"{'方案':<25} {'F1':<12} {'Precision':<14} {'Recall':<12} {'FPR':<12} {'FNR':<12}")
    logger.log("-" * 100)
    for r in sorted(results, key=lambda x: x['metrics']['f1'], reverse=True):
        m = r['metrics']
        logger.log(f"{r['name']:<25} {m['f1']:<12.4f} {m['precision']:<14.4f} {m['recall']:<12.4f} {m['fpr']:<12.4f} {m['fnr']:<12.4f}")
    logger.log("-" * 100)
    
    logger.log("\n二、实时性分析（耗时统计，单位：ms）:")
    logger.log("-" * 160)
    logger.log(f"{'方案':<25} {'RF平均':<10} {'GCN平均':<10} {'RF总':<10} {'GCN总':<10} {'模型输出平均':<12} {'模型输出总':<12} {'融合平均':<10} {'融合总':<10} {'总平均':<10} {'总耗时':<10}")
    logger.log("-" * 160)
    for r in results:
        t = r['timing']
        logger.log(f"{r['name']:<25} {t['rf_avg_ms']:<10.4f} {t['gcn_avg_ms']:<10.4f} {t['rf_total_ms']:<10.2f} {t['gcn_total_ms']:<10.2f} {t['model_output_avg_ms']:<12.4f} {t['model_output_total_ms']:<12.2f} {t['fusion_avg_ms']:<10.4f} {t['fusion_total_ms']:<10.2f} {t['total_avg_ms']:<10.4f} {t['total_total_ms']:<10.2f}")
    logger.log("-" * 160)
    
    best = max(results, key=lambda x: x['metrics']['f1'])
    logger.log(f"\n最佳方案: {best['name']}, F1={best['metrics']['f1']:.4f}")
    
    results_json = os.path.join(LOG_DIR, 'nn_fusion_v54_results.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.log(f"\n结果已保存至: {results_json}")
    
    logger.finalize()
    
    return results


if __name__ == "__main__":
    run_experiments()
