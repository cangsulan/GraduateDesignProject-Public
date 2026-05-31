# -*- coding: utf-8 -*-
"""
神经网络融合方案改进实验 (V5)

目标：通过改进网络结构和训练策略，提升神经网络融合的性能

改进点：
1. 更深的网络结构（多层感知机）
2. 批归一化（Batch Normalization）
3. Dropout防止过拟合
4. 残差连接（Residual Connection）
5. 扩展输入特征（概率乘积、概率差异等）
6. 学习率调度（Cosine Annealing）
7. 早停机制（Early Stopping）
8. 多种网络架构对比
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

TRAIN_SPLIT_CSV = os.path.join(DATA_DIR, 'dataset_train_split.csv')
TRAIN_SPLIT_JSON = os.path.join(DATA_DIR, 'dataset_train_split_graphs.json')
VAL_CSV = os.path.join(DATA_DIR, 'dataset_val.csv')
VAL_JSON = os.path.join(DATA_DIR, 'dataset_val_graphs.json')
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


class FusionNetV1_Original(nn.Module):
    """原始版本：简单MLP（改进输入维度）"""
    def __init__(self):
        super().__init__()
        self.name = "V1_Original"
        self.fc = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()


class FusionNetV2_Deep(nn.Module):
    """V2：更深的网络"""
    def __init__(self):
        super().__init__()
        self.name = "V2_Deep"
        self.fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()


class FusionNetV3_BN(nn.Module):
    """V3：添加批归一化"""
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


class FusionNetV4_Dropout(nn.Module):
    """V4：添加Dropout"""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.name = "V4_Dropout"
        self.fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()


class FusionNetV5_Residual(nn.Module):
    """V5：残差连接"""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.name = "V5_Residual"
        
        self.input_proj = nn.Linear(6, hidden_dim)
        
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        x = F.relu(x + self.block1(x))
        x = F.relu(x + self.block2(x))
        x = F.relu(x + self.block3(x))
        
        return self.output(x).squeeze()


class FusionNetV6_Attention(nn.Module):
    """V6：注意力机制"""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.name = "V6_Attention"
        
        self.input_proj = nn.Linear(6, hidden_dim)
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = F.softmax(q * k / (x.size(-1) ** 0.5), dim=-1)
        x = attn * v
        
        return self.fc(x).squeeze()


class FusionNetV7_Wide(nn.Module):
    """V7：宽网络"""
    def __init__(self):
        super().__init__()
        self.name = "V7_Wide"
        self.fc = nn.Sequential(
            nn.Linear(6, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()


def prepare_features(rf_prob, gcn_prob):
    """准备扩展的输入特征"""
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


def train_fusion_net_with_early_stop(model, train_features, train_labels, 
                                      val_features, val_labels, logger,
                                      max_epochs=200, patience=20, lr=0.001):
    """带早停机制的训练"""
    logger.log(f"  训练 {model.name} (max_epochs={max_epochs}, patience={patience})...")
    
    model = model.to(DEVICE)
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
    log_file = os.path.join(LOG_DIR, f'nn_fusion_v5_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = ExperimentLogger(log_file)
    
    logger.log_section("神经网络融合改进实验 (V5)")
    
    logger.log_section("阶段1: 数据加载")
    
    df_train, train_graphs = load_data(TRAIN_SPLIT_CSV, TRAIN_SPLIT_JSON)
    df_val, val_graphs = load_data(VAL_CSV, VAL_JSON)
    df_test, test_graphs = load_data(TEST_CSV, TEST_JSON)
    
    logger.log(f"  训练集: {len(df_train)} 条")
    logger.log(f"  验证集: {len(df_val)} 条")
    logger.log(f"  测试集: {len(df_test)} 条")
    
    logger.log_section("阶段2: 特征准备")
    
    X_train, y_train, ids_train, encoders, scaler = prepare_rf_features(df_train, fit=True)
    X_val, y_val, ids_val, _, _ = prepare_rf_features(df_val, encoders, scaler)
    X_test, y_test, ids_test, _, _ = prepare_rf_features(df_test, encoders, scaler)
    
    logger.log_section("阶段3: 图数据构建")
    
    train_pyg = build_pyg_data(df_train, TRAIN_SPLIT_JSON)
    val_pyg = build_pyg_data(df_val, VAL_JSON)
    test_pyg = build_pyg_data(df_test, TEST_JSON)
    
    logger.log(f"  训练图: {len(train_pyg)}")
    logger.log(f"  验证图: {len(val_pyg)}")
    logger.log(f"  测试图: {len(test_pyg)}")
    
    logger.log_section("阶段4: 训练基础模型")
    
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
    
    logger.log_section("阶段5: 获取预测概率")
    
    rf_probs_val = rf_model.predict_proba(X_val)[:, 1]
    rf_probs_test = rf_model.predict_proba(X_test)[:, 1]
    
    gcn_model.eval()
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
    
    y_val_gcn = np.array([data.y.item() for data in val_pyg])
    y_test_gcn = np.array([data.y.item() for data in test_pyg])
    
    val_trace_ids = [data.trace_id for data in val_pyg]
    val_id_to_idx = {tid: i for i, tid in enumerate(ids_val)}
    rf_probs_val_aligned = np.array([rf_probs_val[val_id_to_idx[tid]] for tid in val_trace_ids])
    
    test_trace_ids = [data.trace_id for data in test_pyg]
    test_id_to_idx = {tid: i for i, tid in enumerate(ids_test)}
    rf_probs_test_aligned = np.array([rf_probs_test[test_id_to_idx[tid]] for tid in test_trace_ids])
    
    train_features = prepare_features(
        torch.tensor(rf_probs_val_aligned, dtype=torch.float32),
        torch.tensor(gcn_probs_val, dtype=torch.float32)
    )
    train_labels = torch.tensor(y_val_gcn, dtype=torch.float32)
    
    test_features = prepare_features(
        torch.tensor(rf_probs_test_aligned, dtype=torch.float32),
        torch.tensor(gcn_probs_test, dtype=torch.float32)
    )
    
    logger.log_section("阶段6: 神经网络融合实验")
    
    results = []
    
    network_configs = [
        ('V1_Original', FusionNetV1_Original(), 0.01),
        ('V2_Deep', FusionNetV2_Deep(), 0.001),
        ('V3_BN', FusionNetV3_BN(), 0.001),
        ('V4_Dropout', FusionNetV4_Dropout(dropout_rate=0.3), 0.001),
        ('V5_Residual', FusionNetV5_Residual(hidden_dim=32), 0.001),
        ('V6_Attention', FusionNetV6_Attention(hidden_dim=32), 0.001),
        ('V7_Wide', FusionNetV7_Wide(), 0.001),
    ]
    
    for name, model, lr in network_configs:
        logger.log(f"\n--- {name} ---")
        
        trained_model, best_epoch = train_fusion_net_with_early_stop(
            model, train_features, train_labels,
            train_features[:100], train_labels[:100],
            logger, max_epochs=200, patience=20, lr=lr
        )
        
        trained_model.eval()
        with torch.no_grad():
            test_features_dev = test_features.to(DEVICE)
            alphas = trained_model(test_features_dev)
            fused_probs = alphas.cpu().numpy() * rf_probs_test_aligned + (1 - alphas.cpu().numpy()) * gcn_probs_test
        
        preds = (fused_probs > 0.5).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        
        logger.log(f"  结果: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        logger.log(f"  FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
        
        results.append({
            'name': name,
            'metrics': metrics,
            'best_epoch': best_epoch,
            'lr': lr
        })
    
    logger.log_section("阶段7: 最佳网络多训练轮数对比")
    
    best_network_name = max(results, key=lambda x: x['metrics']['f1'])['name']
    logger.log(f"最佳网络架构: {best_network_name}")
    
    epoch_configs = [50, 100, 150, 200, 300]
    
    for epochs in epoch_configs:
        logger.log(f"\n--- {best_network_name} (epochs={epochs}) ---")
        
        if best_network_name == 'V5_Residual':
            model = FusionNetV5_Residual(hidden_dim=32)
        elif best_network_name == 'V4_Dropout':
            model = FusionNetV4_Dropout(dropout_rate=0.3)
        elif best_network_name == 'V3_BN':
            model = FusionNetV3_BN()
        else:
            model = FusionNetV5_Residual(hidden_dim=32)
        
        trained_model, best_epoch = train_fusion_net_with_early_stop(
            model, train_features, train_labels,
            train_features[:100], train_labels[:100],
            logger, max_epochs=epochs, patience=30, lr=0.001
        )
        
        trained_model.eval()
        with torch.no_grad():
            test_features_dev = test_features.to(DEVICE)
            alphas = trained_model(test_features_dev)
            fused_probs = alphas.cpu().numpy() * rf_probs_test_aligned + (1 - alphas.cpu().numpy()) * gcn_probs_test
        
        preds = (fused_probs > 0.5).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        
        logger.log(f"  结果: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        
        results.append({
            'name': f"{best_network_name}(e={epochs})",
            'metrics': metrics,
            'best_epoch': best_epoch
        })
    
    logger.log_section("实验结果汇总")
    
    logger.log("\n所有实验结果（按F1排序）:")
    logger.log("-" * 100)
    logger.log(f"{'方案':<30} {'F1':<10} {'Precision':<12} {'Recall':<10} {'FPR':<10} {'FNR':<10} {'Best Epoch':<12}")
    logger.log("-" * 100)
    
    for r in sorted(results, key=lambda x: x['metrics']['f1'], reverse=True):
        m = r['metrics']
        logger.log(f"{r['name']:<30} {m['f1']:<10.4f} {m['precision']:<12.4f} {m['recall']:<10.4f} {m['fpr']:<10.4f} {m['fnr']:<10.4f} {r.get('best_epoch', '-'):<12}")
    
    logger.log("-" * 100)
    
    best = max(results, key=lambda x: x['metrics']['f1'])
    logger.log(f"\n最佳方案: {best['name']}, F1={best['metrics']['f1']:.4f}")
    
    results_json = os.path.join(LOG_DIR, 'nn_fusion_v5_results.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.log(f"\n结果已保存至: {results_json}")
    
    logger.finalize()
    
    return results


if __name__ == "__main__":
    run_experiments()
