# -*- coding: utf-8 -*-
"""
对照实验脚本 - 微服务异常流量检测系统 (V4)

实验目的：
通过多组训练轮数对比，找到每个实验方案的最优配置

实验设计：
1. RF单模型 - 无需训练轮数
2. GCN单模型（1/2/3层）- epochs: 10, 12, 15, 18, 20
3. 固定权重融合 - epochs: 10, 12, 15, 18, 20
4. 策略实验组（单策略1/单策略2/双策略）- epochs: 10, 12, 15, 18, 20
5. 神经网络融合 - epochs: 50, 75, 100, 125, 150, 175, 200

数据配置：
- 固定权重实验组：使用完整15000条训练数据
- 策略实验组：使用12000条训练+3000条验证集

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

BEST_FIXED_ALPHA = 0.32

GCN_EPOCHS_LIST = [10, 12, 15, 18, 20]
NN_EPOCHS_LIST = [50, 75, 100, 125, 150, 175, 200]


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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x)).squeeze()


class FusionWeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rf_prob, gcn_prob, rf_conf, gcn_conf):
        x = torch.stack([rf_prob, gcn_prob, rf_conf, gcn_conf], dim=1)
        return self.fc(x).squeeze()


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


def fixed_weight_fusion(rf_prob, gcn_prob, alpha):
    fused_prob = alpha * rf_prob + (1 - alpha) * gcn_prob
    return fused_prob, 0.5


def confidence_weight_fusion(rf_prob, gcn_prob, alpha, conf_threshold, 
                             weight_high_conf, weight_low_conf):
    rf_conf = abs(rf_prob - 0.5) * 2
    gcn_conf = abs(gcn_prob - 0.5) * 2
    
    rf_pred = 1 if rf_prob > 0.5 else 0
    gcn_pred = 1 if gcn_prob > 0.5 else 0
    
    if rf_pred == gcn_pred:
        actual_alpha = alpha
    else:
        if rf_conf > conf_threshold and gcn_conf <= conf_threshold:
            actual_alpha = weight_high_conf
        elif gcn_conf > conf_threshold and rf_conf <= conf_threshold:
            actual_alpha = weight_low_conf
        elif rf_conf > conf_threshold and gcn_conf > conf_threshold:
            actual_alpha = weight_high_conf if rf_conf > gcn_conf else weight_low_conf
        else:
            actual_alpha = alpha
    
    fused_prob = actual_alpha * rf_prob + (1 - actual_alpha) * gcn_prob
    return fused_prob, 0.5


def adaptive_threshold_fusion(rf_prob, gcn_prob, alpha, 
                              threshold_adjust_rf, threshold_adjust_gcn):
    rf_pred = 1 if rf_prob > 0.5 else 0
    gcn_pred = 1 if gcn_prob > 0.5 else 0
    
    if rf_pred == gcn_pred:
        threshold = 0.5
    elif rf_pred == 1 and gcn_pred == 0:
        threshold = 0.5 + threshold_adjust_rf
    else:
        threshold = 0.5 - threshold_adjust_gcn
    
    threshold = max(0.1, min(0.9, threshold))
    fused_prob = alpha * rf_prob + (1 - alpha) * gcn_prob
    return fused_prob, threshold


def confidence_adaptive_fusion(rf_prob, gcn_prob, alpha, conf_threshold,
                               weight_high_conf, weight_low_conf,
                               threshold_adjust_rf, threshold_adjust_gcn):
    rf_conf = abs(rf_prob - 0.5) * 2
    gcn_conf = abs(gcn_prob - 0.5) * 2
    
    rf_pred = 1 if rf_prob > 0.5 else 0
    gcn_pred = 1 if gcn_prob > 0.5 else 0
    
    if rf_pred == gcn_pred:
        actual_alpha = alpha
        threshold = 0.5
    else:
        if rf_conf > conf_threshold and gcn_conf <= conf_threshold:
            actual_alpha = weight_high_conf
        elif gcn_conf > conf_threshold and rf_conf <= conf_threshold:
            actual_alpha = weight_low_conf
        elif rf_conf > conf_threshold and gcn_conf > conf_threshold:
            actual_alpha = weight_high_conf if rf_conf > gcn_conf else weight_low_conf
        else:
            actual_alpha = alpha
        
        if rf_pred == 1 and gcn_pred == 0:
            threshold = 0.5 + threshold_adjust_rf
        else:
            threshold = 0.5 - threshold_adjust_gcn
    
    threshold = max(0.1, min(0.9, threshold))
    fused_prob = actual_alpha * rf_prob + (1 - actual_alpha) * gcn_prob
    return fused_prob, threshold


def train_rf_model(X_train, y_train, logger):
    logger.log("训练 Random Forest 模型...")
    start_time = time.time()
    
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=SEED, 
        class_weight='balanced', 
        n_jobs=-1
    )
    rf_model.fit(X_res, y_res)
    
    train_time = time.time() - start_time
    logger.log(f"  RF训练完成，耗时: {train_time:.2f}秒")
    
    return rf_model


def train_gcn_model(train_pyg, hidden_channels, num_layers, logger, epochs=10, verbose=True):
    model_name = f"GCN-{num_layers}层"
    if verbose:
        logger.log(f"训练 {model_name} 模型 (epochs={epochs})...")
    start_time = time.time()
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    model = GCNModel(hidden_channels=hidden_channels, num_layers=num_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    def seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(train_pyg, batch_size=32, shuffle=True, 
                              worker_init_fn=seed_worker, generator=g)
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        
        if verbose and epoch % 5 == 0:
            avg_loss = total_loss / len(train_pyg)
            logger.log(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    train_time = time.time() - start_time
    if verbose:
        logger.log(f"  {model_name}训练完成，耗时: {train_time:.2f}秒")
    
    return model


def train_fusion_net(rf_probs_val, gcn_probs_val, y_val, logger, epochs=100, verbose=True):
    if verbose:
        logger.log(f"训练神经网络融合权重网络 (epochs={epochs})...")
    
    fusion_net = FusionWeightNet().to(DEVICE)
    optimizer = torch.optim.Adam(fusion_net.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    rf_probs_t = torch.tensor(rf_probs_val, dtype=torch.float32).to(DEVICE)
    gcn_probs_t = torch.tensor(gcn_probs_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    rf_conf = torch.abs(rf_probs_t - 0.5) * 2
    gcn_conf = torch.abs(gcn_probs_t - 0.5) * 2
    
    fusion_net.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        alphas = fusion_net(rf_probs_t, gcn_probs_t, rf_conf, gcn_conf)
        fused = alphas * rf_probs_t + (1 - alphas) * gcn_probs_t
        loss = criterion(fused, y_val_t)
        loss.backward()
        optimizer.step()
        
        if verbose and epoch % 25 == 0:
            logger.log(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    if verbose:
        logger.log(f"  训练完成")
    
    return fusion_net


def get_rf_probs_with_timing(model, X, recorder):
    start_time = time.time()
    probs = model.predict_proba(X)[:, 1]
    elapsed = (time.time() - start_time) * 1000
    
    per_sample = elapsed / len(X)
    for _ in range(len(X)):
        recorder.record_rf(per_sample)
    
    return probs


def get_gcn_probs_with_timing(model, pyg_dataset, recorder):
    model.eval()
    probs = []
    
    start_time = time.time()
    with torch.no_grad():
        loader = DataLoader(pyg_dataset, batch_size=32, shuffle=False)
        for data in loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            probs.extend(out.cpu().numpy().flatten())
    
    elapsed = (time.time() - start_time) * 1000
    
    per_sample = elapsed / len(pyg_dataset)
    for _ in range(len(pyg_dataset)):
        recorder.record_gcn(per_sample)
    
    return np.array(probs)


def run_experiments():
    log_file = os.path.join(LOG_DIR, f'experiment_log_v4_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = ExperimentLogger(log_file)
    
    logger.log_section("V4实验：多训练轮数对比实验")
    logger.log(f"GCN训练轮数列表: {GCN_EPOCHS_LIST}")
    logger.log(f"神经网络训练轮数列表: {NN_EPOCHS_LIST}")
    
    logger.log_section("阶段1: 数据加载")
    
    logger.log("\n加载完整训练数据（15000条）...")
    df_train_full, train_full_graphs = load_data(TRAIN_FULL_CSV, TRAIN_FULL_JSON)
    logger.log(f"  完整训练集: {len(df_train_full)} 条")
    
    logger.log("\n加载划分后的训练数据（12000条）和验证数据（3000条）...")
    df_train_split, train_split_graphs = load_data(TRAIN_SPLIT_CSV, TRAIN_SPLIT_JSON)
    df_val, val_graphs = load_data(VAL_CSV, VAL_JSON)
    logger.log(f"  划分训练集: {len(df_train_split)} 条")
    logger.log(f"  验证集: {len(df_val)} 条")
    
    logger.log("\n加载测试数据...")
    df_test, test_graphs = load_data(TEST_CSV, TEST_JSON)
    logger.log(f"  测试集: {len(df_test)} 条")
    
    logger.log_section("阶段2: 特征准备")
    
    logger.log("\n准备完整训练数据的RF特征...")
    X_train_full, y_train_full, ids_train_full, encoders_full, scaler_full = prepare_rf_features(df_train_full, fit=True)
    logger.log(f"  RF特征维度: {X_train_full.shape[1]}")
    
    logger.log("\n准备划分训练数据的RF特征...")
    X_train_split, y_train_split, ids_train_split, encoders_split, scaler_split = prepare_rf_features(df_train_split, fit=True)
    X_val, y_val, ids_val, _, _ = prepare_rf_features(df_val, encoders_split, scaler_split)
    
    logger.log("\n准备测试数据的RF特征...")
    X_test, y_test, ids_test, _, _ = prepare_rf_features(df_test, encoders_full, scaler_full)
    
    logger.log_section("阶段3: 图数据构建")
    
    logger.log("\n构建完整训练数据的图数据...")
    train_full_pyg = build_pyg_data(df_train_full, TRAIN_FULL_JSON)
    logger.log(f"  完整训练图数量: {len(train_full_pyg)}")
    
    logger.log("\n构建划分训练数据的图数据...")
    train_split_pyg = build_pyg_data(df_train_split, TRAIN_SPLIT_JSON)
    val_pyg = build_pyg_data(df_val, VAL_JSON)
    logger.log(f"  划分训练图数量: {len(train_split_pyg)}")
    logger.log(f"  验证图数量: {len(val_pyg)}")
    
    logger.log("\n构建测试数据的图数据...")
    test_pyg = build_pyg_data(df_test, TEST_JSON)
    logger.log(f"  测试图数量: {len(test_pyg)}")
    
    normal_test = (df_test['classification'] == 'normal').sum()
    outlier_test = (df_test['classification'] == 'outlier').sum()
    logger.log(f"\n测试集分布: 正常={normal_test}, 异常={outlier_test}")
    
    y_test_gcn = np.array([data.y.item() for data in test_pyg])
    test_trace_ids = [data.trace_id for data in test_pyg]
    test_id_to_idx = {tid: i for i, tid in enumerate(ids_test)}
    
    results = []
    
    logger.log_section("实验1: RF单模型")
    rf_model_full = train_rf_model(X_train_full, y_train_full, logger)
    
    timer_rf = InferenceTimeRecorder()
    total_start = time.time()
    rf_probs_test_full = get_rf_probs_with_timing(rf_model_full, X_test, timer_rf)
    total_elapsed = (time.time() - total_start) * 1000
    for _ in range(len(X_test)):
        timer_rf.record_total(total_elapsed / len(X_test))
    
    rf_preds_full = (rf_probs_test_full > 0.5).astype(int)
    metrics_rf = calculate_metrics(y_test, rf_preds_full)
    timing_rf = timer_rf.summary()
    
    logger.log(f"\nRF单模型 结果:")
    logger.log(f"  F1-Score: {metrics_rf['f1']:.4f}, FPR: {metrics_rf['fpr']:.4f}, FNR: {metrics_rf['fnr']:.4f}")
    
    results.append({
        'name': 'RF单模型',
        'metrics': metrics_rf,
        'timing': timing_rf,
        'epochs': '-',
        'category': '单模型'
    })
    
    rf_probs_aligned_full = np.array([rf_probs_test_full[test_id_to_idx[tid]] for tid in test_trace_ids])
    rf_avg_time = timing_rf['rf_avg_ms']
    
    logger.log_section("实验2-4: GCN单模型（多训练轮数对比）")
    
    gcn_results = {}
    for num_layers in [1, 2, 3]:
        gcn_results[num_layers] = {}
        for epochs in GCN_EPOCHS_LIST:
            logger.log(f"\n--- GCN-{num_layers}层, epochs={epochs} ---")
            model = train_gcn_model(train_full_pyg, 64, num_layers, logger, epochs=epochs, verbose=True)
            
            timer = InferenceTimeRecorder()
            total_start = time.time()
            probs = get_gcn_probs_with_timing(model, test_pyg, timer)
            total_elapsed = (time.time() - total_start) * 1000
            for _ in range(len(test_pyg)):
                timer.record_total(total_elapsed / len(test_pyg))
            
            preds = (probs > 0.5).astype(int)
            metrics = calculate_metrics(y_test_gcn, preds)
            timing = timer.summary()
            
            logger.log(f"  F1={metrics['f1']:.4f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
            
            result = {
                'name': f'GCN-{num_layers}层(e={epochs})',
                'metrics': metrics,
                'timing': timing,
                'epochs': epochs,
                'num_layers': num_layers,
                'category': 'GCN单模型'
            }
            results.append(result)
            gcn_results[num_layers][epochs] = {
                'model': model,
                'probs': probs,
                'timing': timing
            }
    
    logger.log_section("实验5: 固定权重融合（多训练轮数对比）")
    
    for epochs in GCN_EPOCHS_LIST:
        logger.log(f"\n--- 固定权重融合, epochs={epochs} ---")
        
        gcn_probs = gcn_results[2][epochs]['probs']
        gcn_timing = gcn_results[2][epochs]['timing']
        
        timer = InferenceTimeRecorder()
        
        for i in range(len(rf_probs_aligned_full)):
            timer.record_rf(rf_avg_time)
            timer.record_gcn(gcn_timing['gcn_avg_ms'])
        
        fused_probs = []
        for i in range(len(rf_probs_aligned_full)):
            fusion_start = time.perf_counter()
            fp, _ = fixed_weight_fusion(rf_probs_aligned_full[i], gcn_probs[i], alpha=BEST_FIXED_ALPHA)
            fusion_elapsed = (time.perf_counter() - fusion_start) * 1000
            timer.record_fusion(fusion_elapsed)
            fused_probs.append(fp)
        
        for i in range(len(rf_probs_aligned_full)):
            model_time = rf_avg_time + gcn_timing['gcn_avg_ms']
            fusion_time = timer.fusion_times[i] if i < len(timer.fusion_times) else 0
            timer.record_total(model_time + fusion_time)
        
        fused_probs = np.array(fused_probs)
        preds = (fused_probs > 0.5).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        timing = timer.summary()
        
        logger.log(f"  F1={metrics['f1']:.4f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
        
        results.append({
            'name': f'固定权重融合(e={epochs})',
            'metrics': metrics,
            'timing': timing,
            'epochs': epochs,
            'alpha': BEST_FIXED_ALPHA,
            'category': '固定权重融合'
        })
    
    logger.log_section("阶段4: 策略实验组准备")
    
    logger.log("\n训练调优后的RF模型...")
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_train_split, y_train_split)
    rf_model_opt = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2, 
        max_features='sqrt', random_state=SEED, class_weight='balanced', n_jobs=-1
    )
    rf_model_opt.fit(X_res, y_res)
    
    rf_probs_test_opt = rf_model_opt.predict_proba(X_test)[:, 1]
    rf_probs_aligned_opt = np.array([rf_probs_test_opt[test_id_to_idx[tid]] for tid in test_trace_ids])
    
    rf_timer_opt = InferenceTimeRecorder()
    rf_start = time.time()
    _ = rf_model_opt.predict_proba(X_test)[:, 1]
    rf_elapsed_opt = (time.time() - rf_start) * 1000
    rf_avg_opt = rf_elapsed_opt / len(X_test)
    
    val_trace_ids = [data.trace_id for data in val_pyg]
    val_id_to_idx = {tid: i for i, tid in enumerate(ids_val)}
    rf_probs_val_aligned = np.array([rf_model_opt.predict_proba(X_val)[val_id_to_idx[tid], 1] for tid in val_trace_ids])
    y_val_gcn = np.array([data.y.item() for data in val_pyg])
    
    best_s1_params = {'alpha': 0.2, 'conf_threshold': 0.4, 'weight_high_conf': 0.45, 'weight_low_conf': 0.1}
    best_s2_params = {'alpha': 0.32, 'threshold_adjust_rf': 0.05, 'threshold_adjust_gcn': 0.03}
    best_dual_params = {
        'alpha': 0.2, 'conf_threshold': 0.4, 'weight_high_conf': 0.5, 'weight_low_conf': 0.1,
        'threshold_adjust_rf': 0.05, 'threshold_adjust_gcn': 0.03
    }
    
    gcn_strategy_results = {}
    for epochs in GCN_EPOCHS_LIST:
        logger.log(f"\n--- 策略组GCN训练, epochs={epochs} ---")
        model = train_gcn_model(train_split_pyg, 32, 2, logger, epochs=epochs, verbose=True)
        
        model.eval()
        gcn_probs_test = []
        with torch.no_grad():
            test_loader = DataLoader(test_pyg, batch_size=32, shuffle=False)
            for data in test_loader:
                data = data.to(DEVICE)
                out = model(data.x, data.edge_index, data.batch)
                gcn_probs_test.extend(out.cpu().numpy().flatten())
        gcn_probs_test = np.array(gcn_probs_test)
        
        gcn_probs_val = []
        with torch.no_grad():
            val_loader = DataLoader(val_pyg, batch_size=32, shuffle=False)
            for data in val_loader:
                data = data.to(DEVICE)
                out = model(data.x, data.edge_index, data.batch)
                gcn_probs_val.extend(out.cpu().numpy().flatten())
        gcn_probs_val = np.array(gcn_probs_val)
        
        gcn_timer = InferenceTimeRecorder()
        gcn_start = time.time()
        with torch.no_grad():
            for data in DataLoader(test_pyg, batch_size=32, shuffle=False):
                data = data.to(DEVICE)
                _ = model(data.x, data.edge_index, data.batch)
        gcn_elapsed = (time.time() - gcn_start) * 1000
        gcn_avg = gcn_elapsed / len(test_pyg)
        
        gcn_strategy_results[epochs] = {
            'model': model,
            'gcn_probs_test': gcn_probs_test,
            'gcn_probs_val': gcn_probs_val,
            'gcn_avg': gcn_avg
        }
    
    logger.log_section("实验6: 自适应阈值融合(单策略2)（多训练轮数对比）")
    
    for epochs in GCN_EPOCHS_LIST:
        logger.log(f"\n--- 单策略2, epochs={epochs} ---")
        
        gcn_probs_test = gcn_strategy_results[epochs]['gcn_probs_test']
        gcn_avg = gcn_strategy_results[epochs]['gcn_avg']
        
        timer = InferenceTimeRecorder()
        fused_probs = []
        thresholds = []
        
        for i in range(len(rf_probs_aligned_opt)):
            timer.record_rf(rf_avg_opt)
            timer.record_gcn(gcn_avg)
            
            fusion_start = time.perf_counter()
            fp, th = adaptive_threshold_fusion(
                rf_probs_aligned_opt[i], gcn_probs_test[i],
                alpha=best_s2_params['alpha'],
                threshold_adjust_rf=best_s2_params['threshold_adjust_rf'],
                threshold_adjust_gcn=best_s2_params['threshold_adjust_gcn']
            )
            fusion_elapsed = (time.perf_counter() - fusion_start) * 1000
            timer.record_fusion(fusion_elapsed)
            timer.record_total(rf_avg_opt + gcn_avg + fusion_elapsed)
            
            fused_probs.append(fp)
            thresholds.append(th)
        
        preds = (np.array(fused_probs) > np.array(thresholds)).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        timing = timer.summary()
        
        logger.log(f"  F1={metrics['f1']:.4f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
        
        results.append({
            'name': f'单策略2(e={epochs})',
            'metrics': metrics,
            'timing': timing,
            'epochs': epochs,
            'params': best_s2_params,
            'category': '单策略2'
        })
    
    logger.log_section("实验7: 置信度动态权重融合(单策略1)（多训练轮数对比）")
    
    for epochs in GCN_EPOCHS_LIST:
        logger.log(f"\n--- 单策略1, epochs={epochs} ---")
        
        gcn_probs_test = gcn_strategy_results[epochs]['gcn_probs_test']
        gcn_avg = gcn_strategy_results[epochs]['gcn_avg']
        
        timer = InferenceTimeRecorder()
        fused_probs = []
        
        for i in range(len(rf_probs_aligned_opt)):
            timer.record_rf(rf_avg_opt)
            timer.record_gcn(gcn_avg)
            
            fusion_start = time.perf_counter()
            fp, _ = confidence_weight_fusion(
                rf_probs_aligned_opt[i], gcn_probs_test[i],
                alpha=best_s1_params['alpha'],
                conf_threshold=best_s1_params['conf_threshold'],
                weight_high_conf=best_s1_params['weight_high_conf'],
                weight_low_conf=best_s1_params['weight_low_conf']
            )
            fusion_elapsed = (time.perf_counter() - fusion_start) * 1000
            timer.record_fusion(fusion_elapsed)
            timer.record_total(rf_avg_opt + gcn_avg + fusion_elapsed)
            
            fused_probs.append(fp)
        
        preds = (np.array(fused_probs) > 0.5).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        timing = timer.summary()
        
        logger.log(f"  F1={metrics['f1']:.4f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
        
        results.append({
            'name': f'单策略1(e={epochs})',
            'metrics': metrics,
            'timing': timing,
            'epochs': epochs,
            'params': best_s1_params,
            'category': '单策略1'
        })
    
    logger.log_section("实验8: 置信度自适应融合(双策略)（多训练轮数对比）")
    
    for epochs in GCN_EPOCHS_LIST:
        logger.log(f"\n--- 双策略, epochs={epochs} ---")
        
        gcn_probs_test = gcn_strategy_results[epochs]['gcn_probs_test']
        gcn_avg = gcn_strategy_results[epochs]['gcn_avg']
        
        timer = InferenceTimeRecorder()
        fused_probs = []
        thresholds = []
        
        for i in range(len(rf_probs_aligned_opt)):
            timer.record_rf(rf_avg_opt)
            timer.record_gcn(gcn_avg)
            
            fusion_start = time.perf_counter()
            fp, th = confidence_adaptive_fusion(
                rf_probs_aligned_opt[i], gcn_probs_test[i],
                alpha=best_dual_params['alpha'],
                conf_threshold=best_dual_params['conf_threshold'],
                weight_high_conf=best_dual_params['weight_high_conf'],
                weight_low_conf=best_dual_params['weight_low_conf'],
                threshold_adjust_rf=best_dual_params['threshold_adjust_rf'],
                threshold_adjust_gcn=best_dual_params['threshold_adjust_gcn']
            )
            fusion_elapsed = (time.perf_counter() - fusion_start) * 1000
            timer.record_fusion(fusion_elapsed)
            timer.record_total(rf_avg_opt + gcn_avg + fusion_elapsed)
            
            fused_probs.append(fp)
            thresholds.append(th)
        
        preds = (np.array(fused_probs) > np.array(thresholds)).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        timing = timer.summary()
        
        logger.log(f"  F1={metrics['f1']:.4f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
        
        results.append({
            'name': f'双策略(e={epochs})',
            'metrics': metrics,
            'timing': timing,
            'epochs': epochs,
            'params': best_dual_params,
            'category': '双策略'
        })
    
    logger.log_section("实验9: 神经网络学习权重融合（多训练轮数对比）")
    
    for epochs in NN_EPOCHS_LIST:
        logger.log(f"\n--- 神经网络融合, epochs={epochs} ---")
        
        fusion_net = train_fusion_net(
            rf_probs_val_aligned, 
            gcn_strategy_results[15]['gcn_probs_val'],
            y_val_gcn, 
            logger, 
            epochs=epochs, 
            verbose=True
        )
        
        gcn_avg = gcn_strategy_results[15]['gcn_avg']
        gcn_probs_test = gcn_strategy_results[15]['gcn_probs_test']
        
        timer = InferenceTimeRecorder()
        fused_probs = []
        
        fusion_net.eval()
        with torch.no_grad():
            rf_t = torch.tensor(rf_probs_aligned_opt, dtype=torch.float32).to(DEVICE)
            gcn_t = torch.tensor(gcn_probs_test, dtype=torch.float32).to(DEVICE)
            rf_conf_t = torch.abs(rf_t - 0.5) * 2
            gcn_conf_t = torch.abs(gcn_t - 0.5) * 2
            
            for i in range(len(rf_probs_aligned_opt)):
                timer.record_rf(rf_avg_opt)
                timer.record_gcn(gcn_avg)
                
                fusion_start = time.perf_counter()
                alpha = fusion_net(rf_t[i:i+1], gcn_t[i:i+1], rf_conf_t[i:i+1], gcn_conf_t[i:i+1])
                fp = (alpha * rf_t[i] + (1 - alpha) * gcn_t[i]).item()
                fusion_elapsed = (time.perf_counter() - fusion_start) * 1000
                timer.record_fusion(fusion_elapsed)
                timer.record_total(rf_avg_opt + gcn_avg + fusion_elapsed)
                
                fused_probs.append(fp)
        
        preds = (np.array(fused_probs) > 0.5).astype(int)
        metrics = calculate_metrics(y_test_gcn, preds)
        timing = timer.summary()
        
        logger.log(f"  F1={metrics['f1']:.4f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
        
        results.append({
            'name': f'神经网络融合(e={epochs})',
            'metrics': metrics,
            'timing': timing,
            'epochs': epochs,
            'category': '神经网络融合'
        })
    
    logger.log_section("实验结果汇总")
    
    logger.log("\n一、各方案最佳F1分数:")
    logger.log("-" * 80)
    
    categories = ['RF单模型', 'GCN单模型', '固定权重融合', '单策略1', '单策略2', '双策略', '神经网络融合']
    for cat in categories:
        cat_results = [r for r in results if r.get('category') == cat]
        if cat_results:
            best = max(cat_results, key=lambda x: x['metrics']['f1'])
            logger.log(f"{cat:<20}: 最佳F1={best['metrics']['f1']:.4f} ({best['name']})")
    
    logger.log("\n二、详细结果表:")
    logger.log("-" * 120)
    logger.log(f"{'实验方案':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FPR':<10} {'FNR':<10}")
    logger.log("-" * 120)
    for r in sorted(results, key=lambda x: x['metrics']['f1'], reverse=True):
        m = r['metrics']
        logger.log(f"{r['name']:<30} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['fpr']:<10.4f} {m['fnr']:<10.4f}")
    logger.log("-" * 120)
    
    best_result = max(results, key=lambda x: x['metrics']['f1'])
    logger.log(f"\n最佳方案: {best_result['name']}, F1={best_result['metrics']['f1']:.4f}")
    
    results_json = os.path.join(LOG_DIR, 'experiment_results_v4.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.log(f"\n结果已保存至: {results_json}")
    
    logger.finalize()
    
    return results


if __name__ == "__main__":
    run_experiments()
