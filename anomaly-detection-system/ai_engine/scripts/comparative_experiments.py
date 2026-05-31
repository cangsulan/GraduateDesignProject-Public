# -*- coding: utf-8 -*-
"""
对照实验脚本 - 微服务异常流量检测系统

实验目的：
对比不同检测方案的性能，验证双模态融合策略的有效性

实验组：
1. RF单模型 - 使用完整训练数据(15000条)
2. 单层GCN模型 - 使用完整训练数据(15000条)
3. 双层GCN模型 - 使用完整训练数据(15000条)
3.5. 三层GCN模型 - 使用完整训练数据(15000条)
4. RF + 双层GCN + 固定权重融合 (α=BEST_FIXED_ALPHA，来自505组实验) - 使用完整训练数据
5. RF + 双层GCN + 自适应阈值融合 (单策略2：策略2) - 使用训练+验证集调参
6. RF + 双层GCN + 置信度动态权重融合 (单策略1：策略1) - 使用训练+验证集调参
7. RF + 双层GCN + 置信度自适应融合 (双策略：策略1+策略2，最终方案) - 使用训练+验证集调参
8. RF + 双层GCN + 神经网络学习权重融合 (对比方案) - 使用训练+验证集调参

策略定义：
- 策略1（置信度加权融合策略）：根据置信度动态调整权重，阈值固定0.5
- 策略2（自适应阈值策略）：权重固定，根据预测场景动态调整阈值
- 双策略：策略1 + 策略2

数据划分：
- 完整训练集 (dataset_train.csv): 15000条，用于固定权重实验
- 划分训练集 (dataset_train_split.csv): 12000条，用于策略实验
- 验证集 (dataset_val.csv): 3000条，用于策略参数调优
- 测试集 (dataset_test.csv): 独立测试集，仅用于最终评估

重要原则：
1. 测试集数据绝对不参与训练和参数选择
2. 固定权重实验不需要验证集，使用完整训练数据
3. 策略实验需要验证集进行参数调优
4. 记录详细的性能指标和耗时
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
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import degree

warnings.filterwarnings('ignore')

# ======================== 随机种子设置（确保可复现性） ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================== 路径配置 ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
LOG_DIR = os.path.join(BASE_DIR, 'experiment_logs')

# 完整训练数据路径（用于固定权重实验）
TRAIN_FULL_CSV = os.path.join(DATA_DIR, 'dataset_train.csv')
TRAIN_FULL_JSON = os.path.join(DATA_DIR, 'dataset_train_graphs.json')

# 划分后的训练/验证数据路径（用于策略实验）
TRAIN_SPLIT_CSV = os.path.join(DATA_DIR, 'dataset_train_split.csv')
TRAIN_SPLIT_JSON = os.path.join(DATA_DIR, 'dataset_train_split_graphs.json')
VAL_CSV = os.path.join(DATA_DIR, 'dataset_val.csv')
VAL_JSON = os.path.join(DATA_DIR, 'dataset_val_graphs.json')

# 测试数据路径
TEST_CSV = os.path.join(DATA_DIR, 'dataset_test.csv')
TEST_JSON = os.path.join(DATA_DIR, 'dataset_test_graphs.json')

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 特征定义
NUMERIC_FEATURES = [
    'inter_api_access_duration(sec)', 'api_access_uniqueness', 
    'sequence_length(count)', 'vsession_duration(min)', 
    'num_sessions', 'num_users', 'num_unique_apis'
]
CATEGORICAL_FEATURES = ['ip_type', 'source']

# ======================== 日志工具 ========================
class ExperimentLogger:
    """实验日志记录器"""
    
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
    
    def log_table(self, headers, rows):
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        header_line = '-' * (sum(col_widths) + 3 * len(headers) - 1)
        self.log(header_line)
        
        header_row = ' | '.join(str(h).ljust(w) for h, w in zip(headers, col_widths))
        self.log(header_row)
        
        separator = '-+-'.join('-' * w for w in col_widths)
        self.log(separator)
        
        for row in rows:
            row_line = ' | '.join(str(v).ljust(w) for v, w in zip(row, col_widths))
            self.log(row_line)
        
        self.log(header_line)
    
    def finalize(self):
        total_time = time.time() - self.start_time
        self.log(f"\n实验总耗时: {total_time:.2f} 秒")
        self.log(f"实验结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ======================== 详细耗时记录器 ========================
class DetailedTimeRecorder:
    """详细耗时记录器 - 记录各阶段耗时"""
    
    def __init__(self):
        self.records = {}
        self.current_start = None
        self.current_name = None
    
    def start(self, name):
        self.current_name = name
        self.current_start = time.time()
    
    def stop(self):
        if self.current_start is None:
            return 0
        elapsed = (time.time() - self.current_start) * 1000
        if self.current_name not in self.records:
            self.records[self.current_name] = []
        self.records[self.current_name].append(elapsed)
        self.current_start = None
        self.current_name = None
        return elapsed
    
    def record(self, name, elapsed_ms):
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(elapsed_ms)
    
    def summary(self):
        result = {}
        for name, times in self.records.items():
            result[name] = {
                'total': sum(times),
                'avg': np.mean(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0,
                'count': len(times)
            }
        return result
    
    def get_total(self, name):
        if name in self.records:
            return sum(self.records[name])
        return 0


# ======================== 模型定义 ========================
class GCNModel(nn.Module):
    """GCN模型"""
    
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
    """神经网络学习融合权重"""
    
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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, rf_prob, gcn_prob, rf_conf, gcn_conf):
        x = torch.stack([rf_prob, gcn_prob, rf_conf, gcn_conf], dim=1)
        alpha = self.fc(x)
        return alpha.squeeze()


# ======================== 数据处理 ========================
def load_data(csv_path, json_path):
    """加载数据"""
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        graphs = json.load(f)
    return df, graphs


def prepare_rf_features(df, encoders=None, scaler=None, fit=False):
    """准备RF模型特征"""
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
    """构建PyG图数据集"""
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


# ======================== 参数搜索配置 ========================
RF_PARAM_SEARCH_SPACE = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

GCN_PARAM_SEARCH_SPACE = {
    'hidden_channels': [32, 64, 128],
    'learning_rate': [0.001, 0.005, 0.01],
    'dropout': [0.0, 0.3]
}

SINGLE_STRATEGY1_PARAM_SPACE = {
    'alpha': [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50],
    'conf_threshold': [0.4, 0.5, 0.6, 0.7, 0.8],
    'weight_high_conf': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    'weight_low_conf': [0.10, 0.15, 0.20, 0.25, 0.30]
}

SINGLE_STRATEGY2_PARAM_SPACE = {
    'threshold_adjust_rf': [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    'threshold_adjust_gcn': [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
}

DUAL_STRATEGY_PARAM_SPACE = {
    'alpha': [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50],
    'conf_threshold': [0.4, 0.5, 0.6, 0.7, 0.8],
    'weight_high_conf': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    'weight_low_conf': [0.10, 0.15, 0.20, 0.25, 0.30],
    'threshold_adjust_rf': [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    'threshold_adjust_gcn': [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
}

BEST_FIXED_ALPHA = 0.32


# ======================== 融合策略函数 ========================
def fixed_weight_fusion(rf_prob, gcn_prob, alpha):
    """固定权重融合"""
    fused_prob = alpha * rf_prob + (1 - alpha) * gcn_prob
    return fused_prob, 0.5


def confidence_weight_fusion(rf_prob, gcn_prob, alpha, conf_threshold, 
                             weight_high_conf, weight_low_conf):
    """策略1：置信度加权融合"""
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
    """策略2：自适应阈值融合"""
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
    """双策略：置信度加权 + 自适应阈值"""
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


# ======================== 评估函数 ========================
def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    normal_count = tn + fp
    anomaly_count = fn + tp
    
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
        'tp': int(tp),
        'normal_count': normal_count,
        'anomaly_count': anomaly_count
    }


# ======================== 参数搜索函数 ========================
def search_rf_params(X_train, y_train, X_val, y_val, logger, time_recorder):
    """搜索RF模型最优参数"""
    logger.log("\n在验证集上搜索RF模型最优参数...")
    
    best_params = None
    best_score = 0
    best_model = None
    
    n_estimators_list = RF_PARAM_SEARCH_SPACE['n_estimators']
    max_depth_list = RF_PARAM_SEARCH_SPACE['max_depth']
    min_samples_split_list = RF_PARAM_SEARCH_SPACE['min_samples_split']
    max_features_list = RF_PARAM_SEARCH_SPACE['max_features']
    
    total = len(n_estimators_list) * len(max_depth_list) * len(min_samples_split_list) * len(max_features_list)
    logger.log(f"搜索空间: {total} 组合")
    logger.log(f"优化目标: 最大化 F1-Score")
    
    search_start = time.time()
    count = 0
    for n_est in n_estimators_list:
        for max_d in max_depth_list:
            for min_split in min_samples_split_list:
                for max_feat in max_features_list:
                    count += 1
                    
                    smote = SMOTE(random_state=SEED)
                    X_res, y_res = smote.fit_resample(X_train, y_train)
                    
                    model = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_split,
                        max_features=max_feat,
                        random_state=SEED,
                        class_weight='balanced',
                        n_jobs=-1
                    )
                    model.fit(X_res, y_res)
                    
                    probs = model.predict_proba(X_val)[:, 1]
                    preds = (probs > 0.5).astype(int)
                    
                    result = calculate_metrics(y_val, preds)
                    score = result['f1']
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'min_samples_split': min_split,
                            'max_features': max_feat
                        }
                        best_model = model
    
    search_time = time.time() - search_start
    time_recorder.record('rf_param_search', search_time * 1000)
    
    logger.log(f"\nRF最优参数:")
    logger.log(f"  n_estimators={best_params['n_estimators']}")
    logger.log(f"  max_depth={best_params['max_depth']}")
    logger.log(f"  min_samples_split={best_params['min_samples_split']}")
    logger.log(f"  max_features={best_params['max_features']}")
    logger.log(f"验证集 F1-Score: {best_score:.4f}")
    logger.log(f"参数搜索耗时: {search_time:.2f}秒")
    
    return best_params, best_model


def search_gcn_params(train_pyg, val_pyg, logger, time_recorder, epochs=15):
    """搜索GCN模型最优参数"""
    logger.log("\n在验证集上搜索GCN模型最优参数...")
    
    best_params = None
    best_score = 0
    best_model = None
    
    hidden_channels_list = GCN_PARAM_SEARCH_SPACE['hidden_channels']
    lr_list = GCN_PARAM_SEARCH_SPACE['learning_rate']
    dropout_list = GCN_PARAM_SEARCH_SPACE['dropout']
    
    total = len(hidden_channels_list) * len(lr_list) * len(dropout_list)
    logger.log(f"搜索空间: {total} 组合")
    logger.log(f"优化目标: 最大化 F1-Score")
    
    y_val = np.array([data.y.item() for data in val_pyg])
    
    search_start = time.time()
    count = 0
    for hidden_ch in hidden_channels_list:
        for lr in lr_list:
            for dropout in dropout_list:
                count += 1
                logger.log(f"  [{count}/{total}] hidden={hidden_ch}, lr={lr}, dropout={dropout}")
                
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                random.seed(SEED)
                
                model = GCNModel(hidden_channels=hidden_ch, num_layers=2).to(DEVICE)
                
                if dropout > 0:
                    model.dropout = nn.Dropout(dropout)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
                
                model.eval()
                probs = []
                with torch.no_grad():
                    val_loader = DataLoader(val_pyg, batch_size=32, shuffle=False)
                    for data in val_loader:
                        data = data.to(DEVICE)
                        out = model(data.x, data.edge_index, data.batch)
                        probs.extend(out.cpu().numpy())
                
                probs = np.array(probs)
                preds = (probs > 0.5).astype(int)
                
                result = calculate_metrics(y_val, preds)
                score = result['f1']
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'hidden_channels': hidden_ch,
                        'learning_rate': lr,
                        'dropout': dropout
                    }
                    best_model = model
    
    search_time = time.time() - search_start
    time_recorder.record('gcn_param_search', search_time * 1000)
    
    logger.log(f"\nGCN最优参数:")
    logger.log(f"  hidden_channels={best_params['hidden_channels']}")
    logger.log(f"  learning_rate={best_params['learning_rate']}")
    logger.log(f"  dropout={best_params['dropout']}")
    logger.log(f"验证集 F1-Score: {best_score:.4f}")
    logger.log(f"参数搜索耗时: {search_time:.2f}秒")
    
    return best_params, best_model


def search_single_strategy1_params(rf_probs, gcn_probs, y_true, logger, time_recorder):
    """搜索单策略1最优参数"""
    logger.log("\n在验证集上搜索单策略1（策略1：置信度加权融合）最优参数...")
    logger.log(f"搜索空间: {len(SINGLE_STRATEGY1_PARAM_SPACE['alpha']) * len(SINGLE_STRATEGY1_PARAM_SPACE['conf_threshold']) * len(SINGLE_STRATEGY1_PARAM_SPACE['weight_high_conf']) * len(SINGLE_STRATEGY1_PARAM_SPACE['weight_low_conf'])} 组合")
    logger.log(f"优化目标: 最大化 F1-Score")
    
    best_params = None
    best_score = 0
    
    search_start = time.time()
    total = 0
    
    for alpha in SINGLE_STRATEGY1_PARAM_SPACE['alpha']:
        for conf_th in SINGLE_STRATEGY1_PARAM_SPACE['conf_threshold']:
            for w_high in SINGLE_STRATEGY1_PARAM_SPACE['weight_high_conf']:
                for w_low in SINGLE_STRATEGY1_PARAM_SPACE['weight_low_conf']:
                    total += 1
                    
                    preds = []
                    for i in range(len(rf_probs)):
                        fused_prob, _ = confidence_weight_fusion(
                            rf_probs[i], gcn_probs[i], alpha, conf_th, w_high, w_low
                        )
                        preds.append(1 if fused_prob > 0.5 else 0)
                    
                    result = calculate_metrics(y_true, preds)
                    score = result['f1']
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'alpha': alpha,
                            'conf_threshold': conf_th,
                            'weight_high_conf': w_high,
                            'weight_low_conf': w_low
                        }
                    
                    if total % 120 == 0:
                        logger.log(f"  搜索进度: {total//120}0% ({total}/{len(SINGLE_STRATEGY1_PARAM_SPACE['alpha']) * len(SINGLE_STRATEGY1_PARAM_SPACE['conf_threshold']) * len(SINGLE_STRATEGY1_PARAM_SPACE['weight_high_conf']) * len(SINGLE_STRATEGY1_PARAM_SPACE['weight_low_conf'])})")
    
    search_time = time.time() - search_start
    time_recorder.record('strategy1_param_search', search_time * 1000)
    
    logger.log(f"\n搜索完成! 共评估 {total} 组合")
    logger.log(f"\n单策略1最优参数:")
    logger.log(f"  alpha={best_params['alpha']}")
    logger.log(f"  conf_threshold={best_params['conf_threshold']}")
    logger.log(f"  weight_high_conf={best_params['weight_high_conf']}")
    logger.log(f"  weight_low_conf={best_params['weight_low_conf']}")
    logger.log(f"验证集 F1-Score: {best_score:.4f}")
    logger.log(f"参数搜索耗时: {search_time:.2f}秒")
    
    return best_params


def search_single_strategy2_params(rf_probs, gcn_probs, y_true, alpha_fixed, logger, time_recorder):
    """搜索单策略2最优参数"""
    logger.log("\n在验证集上搜索单策略2（策略2：自适应阈值）最优参数...")
    logger.log(f"使用固定权重 alpha={alpha_fixed}")
    logger.log(f"搜索空间: {len(SINGLE_STRATEGY2_PARAM_SPACE['threshold_adjust_rf']) * len(SINGLE_STRATEGY2_PARAM_SPACE['threshold_adjust_gcn'])} 组合")
    logger.log(f"优化目标: 最大化 F1-Score（优先选择FNR较低的参数）")
    
    best_params = None
    best_score = 0
    best_fnr = 1.0
    
    all_results = []
    
    search_start = time.time()
    total = 0
    
    for th_rf in SINGLE_STRATEGY2_PARAM_SPACE['threshold_adjust_rf']:
        for th_gcn in SINGLE_STRATEGY2_PARAM_SPACE['threshold_adjust_gcn']:
            total += 1
            
            preds = []
            for i in range(len(rf_probs)):
                fused_prob, threshold = adaptive_threshold_fusion(
                    rf_probs[i], gcn_probs[i], alpha_fixed, th_rf, th_gcn
                )
                preds.append(1 if fused_prob > threshold else 0)
            
            result = calculate_metrics(y_true, preds)
            score = result['f1']
            fnr = result['fnr']
            
            all_results.append({
                'params': {
                    'alpha': alpha_fixed,
                    'threshold_adjust_rf': th_rf,
                    'threshold_adjust_gcn': th_gcn
                },
                'f1': score,
                'fnr': fnr,
                'fpr': result['fpr']
            })
            
            if total % 12 == 0:
                logger.log(f"  搜索进度: {total}/{len(SINGLE_STRATEGY2_PARAM_SPACE['threshold_adjust_rf']) * len(SINGLE_STRATEGY2_PARAM_SPACE['threshold_adjust_gcn'])}")
    
    all_results.sort(key=lambda x: (x['fnr'], -x['f1']))
    
    if all_results:
        best_result = all_results[0]
        best_params = best_result['params']
        best_score = best_result['f1']
        best_fnr = best_result['fnr']
    
    search_time = time.time() - search_start
    time_recorder.record('strategy2_param_search', search_time * 1000)
    
    logger.log(f"\n搜索完成! 共评估 {total} 组合")
    logger.log(f"\n单策略2最优参数（FNR最低）:")
    logger.log(f"  alpha={best_params['alpha']} (固定)")
    logger.log(f"  threshold_adjust_rf={best_params['threshold_adjust_rf']}")
    logger.log(f"  threshold_adjust_gcn={best_params['threshold_adjust_gcn']}")
    
    preds = []
    for i in range(len(rf_probs)):
        fused_prob, threshold = adaptive_threshold_fusion(
            rf_probs[i], gcn_probs[i], best_params['alpha'],
            best_params['threshold_adjust_rf'], best_params['threshold_adjust_gcn']
        )
        preds.append(1 if fused_prob > threshold else 0)
    result = calculate_metrics(y_true, preds)
    logger.log(f"\n验证集结果:")
    logger.log(f"  Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, F1={result['f1']:.4f}")
    logger.log(f"  FPR={result['fpr']:.4f}, FNR={result['fnr']:.4f}")
    logger.log(f"参数搜索耗时: {search_time:.2f}秒")
    
    return best_params


def search_dual_strategy_params(rf_probs, gcn_probs, y_true, logger, time_recorder):
    """搜索双策略最优参数"""
    logger.log("\n在验证集上搜索双策略（策略1+策略2：置信度自适应融合）最优参数...")
    
    total_combinations = (len(DUAL_STRATEGY_PARAM_SPACE['alpha']) * 
                         len(DUAL_STRATEGY_PARAM_SPACE['conf_threshold']) * 
                         len(DUAL_STRATEGY_PARAM_SPACE['weight_high_conf']) * 
                         len(DUAL_STRATEGY_PARAM_SPACE['weight_low_conf']) * 
                         len(DUAL_STRATEGY_PARAM_SPACE['threshold_adjust_rf']) * 
                         len(DUAL_STRATEGY_PARAM_SPACE['threshold_adjust_gcn']))
    
    logger.log(f"搜索空间: {total_combinations} 组合")
    logger.log(f"优化目标: 最大化 F1-Score（优先选择FNR较低的参数）")
    
    best_params = None
    best_score = 0
    best_fnr = 1.0
    
    search_start = time.time()
    total = 0
    
    for alpha in DUAL_STRATEGY_PARAM_SPACE['alpha']:
        for conf_th in DUAL_STRATEGY_PARAM_SPACE['conf_threshold']:
            for w_high in DUAL_STRATEGY_PARAM_SPACE['weight_high_conf']:
                for w_low in DUAL_STRATEGY_PARAM_SPACE['weight_low_conf']:
                    for th_rf in DUAL_STRATEGY_PARAM_SPACE['threshold_adjust_rf']:
                        for th_gcn in DUAL_STRATEGY_PARAM_SPACE['threshold_adjust_gcn']:
                            total += 1
                            
                            preds = []
                            for i in range(len(rf_probs)):
                                fused_prob, threshold = confidence_adaptive_fusion(
                                    rf_probs[i], gcn_probs[i], alpha, conf_th,
                                    w_high, w_low, th_rf, th_gcn
                                )
                                preds.append(1 if fused_prob > threshold else 0)
                            
                            result = calculate_metrics(y_true, preds)
                            score = result['f1']
                            fnr = result['fnr']
                            
                            if fnr < best_fnr or (fnr == best_fnr and score > best_score):
                                best_score = score
                                best_fnr = fnr
                                best_params = {
                                    'alpha': alpha,
                                    'conf_threshold': conf_th,
                                    'weight_high_conf': w_high,
                                    'weight_low_conf': w_low,
                                    'threshold_adjust_rf': th_rf,
                                    'threshold_adjust_gcn': th_gcn
                                }
                            
                            if total % 5760 == 0:
                                logger.log(f"  搜索进度: {total//5760}0% ({total}/{total_combinations})")
    
    search_time = time.time() - search_start
    time_recorder.record('dual_strategy_param_search', search_time * 1000)
    
    logger.log(f"\n搜索完成! 共评估 {total} 组合")
    logger.log(f"\n双策略最优参数（FNR最低={best_fnr:.4f}）:")
    logger.log(f"  alpha={best_params['alpha']}")
    logger.log(f"  conf_threshold={best_params['conf_threshold']}")
    logger.log(f"  weight_high_conf={best_params['weight_high_conf']}")
    logger.log(f"  weight_low_conf={best_params['weight_low_conf']}")
    logger.log(f"  threshold_adjust_rf={best_params['threshold_adjust_rf']}")
    logger.log(f"  threshold_adjust_gcn={best_params['threshold_adjust_gcn']}")
    
    preds = []
    for i in range(len(rf_probs)):
        fused_prob, threshold = confidence_adaptive_fusion(
            rf_probs[i], gcn_probs[i], best_params['alpha'],
            best_params['conf_threshold'], best_params['weight_high_conf'],
            best_params['weight_low_conf'], best_params['threshold_adjust_rf'],
            best_params['threshold_adjust_gcn']
        )
        preds.append(1 if fused_prob > threshold else 0)
    result = calculate_metrics(y_true, preds)
    logger.log(f"\n验证集结果:")
    logger.log(f"  Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, F1={result['f1']:.4f}")
    logger.log(f"  FPR={result['fpr']:.4f}, FNR={result['fnr']:.4f}")
    logger.log(f"参数搜索耗时: {search_time:.2f}秒")
    
    return best_params


# ======================== 训练函数 ========================
def train_rf_model(X_train, y_train, logger, time_recorder):
    """训练RF模型"""
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
    time_recorder.record('rf_train', train_time * 1000)
    logger.log(f"  RF训练完成，耗时: {train_time:.2f}秒")
    
    return rf_model, train_time


def train_gcn_model(train_pyg, hidden_channels, num_layers, logger, time_recorder, epochs=20):
    """训练GCN模型"""
    model_name = f"GCN-{num_layers}层"
    logger.log(f"训练 {model_name} 模型...")
    start_time = time.time()
    
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
        
        if epoch % 5 == 0:
            avg_loss = total_loss / len(train_pyg)
            logger.log(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    train_time = time.time() - start_time
    time_recorder.record(f'gcn{num_layers}_train', train_time * 1000)
    logger.log(f"  {model_name}训练完成，耗时: {train_time:.2f}秒")
    
    return model, train_time


def train_fusion_net(rf_probs_val, gcn_probs_val, y_val, logger, time_recorder, epochs=100):
    """训练神经网络融合权重网络"""
    logger.log("训练神经网络融合权重网络...")
    start_time = time.time()
    
    model = FusionWeightNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    rf_probs_t = torch.tensor(rf_probs_val, dtype=torch.float32).to(DEVICE)
    gcn_probs_t = torch.tensor(gcn_probs_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    rf_conf = torch.abs(rf_probs_t - 0.5) * 2
    gcn_conf = torch.abs(gcn_probs_t - 0.5) * 2
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        alpha = model(rf_probs_t, gcn_probs_t, rf_conf, gcn_conf)
        fused = alpha * rf_probs_t + (1 - alpha) * gcn_probs_t
        loss = criterion(fused, y_val_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            logger.log(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    train_time = time.time() - start_time
    time_recorder.record('fusion_net_train', train_time * 1000)
    logger.log(f"  神经网络训练完成，耗时: {train_time:.2f}秒")
    
    return model, train_time


# ======================== 推理函数 ========================
def get_rf_probs(model, X, time_recorder):
    """获取RF模型预测概率"""
    start_time = time.time()
    probs = model.predict_proba(X)[:, 1]
    elapsed = (time.time() - start_time) * 1000
    time_recorder.record('rf_inference', elapsed)
    return probs


def get_gcn_probs(model, pyg_dataset, time_recorder):
    """获取GCN模型预测概率"""
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
    time_recorder.record('gcn_inference', elapsed)
    
    return np.array(probs)


# ======================== 实验结果记录 ========================
def run_single_experiment(exp_name, y_true, y_pred, time_recorder, logger):
    """运行单个实验并记录结果"""
    metrics = calculate_metrics(y_true, y_pred)
    time_summary = time_recorder.summary()
    
    logger.log(f"\n{exp_name} 结果:")
    logger.log(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.log(f"  Precision: {metrics['precision']:.4f}")
    logger.log(f"  Recall:    {metrics['recall']:.4f}")
    logger.log(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.log(f"  FPR(误报率): {metrics['fpr']:.4f}")
    logger.log(f"  FNR(漏报率): {metrics['fnr']:.4f}")
    
    for phase, stats in time_summary.items():
        logger.log(f"  {phase}耗时: 总计{stats['total']:.2f}ms, 平均{stats['avg']:.4f}ms/条")
    
    return {
        'name': exp_name,
        'metrics': metrics,
        'time': time_summary
    }


# ======================== 主实验流程 ========================
def run_experiments():
    """运行所有对照实验"""
    
    log_file = os.path.join(LOG_DIR, f'experiment_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = ExperimentLogger(log_file)
    
    # 全局耗时记录器
    global_timer = DetailedTimeRecorder()
    
    # ======================== 阶段1: 数据加载 ========================
    logger.log_section("阶段1: 数据加载")
    
    # 加载完整训练数据（用于固定权重实验）
    logger.log("\n加载完整训练数据（15000条）...")
    load_start = time.time()
    df_train_full, train_full_graphs = load_data(TRAIN_FULL_CSV, TRAIN_FULL_JSON)
    load_time = time.time() - load_start
    global_timer.record('data_load_full', load_time * 1000)
    logger.log(f"  完整训练集: {len(df_train_full)} 条")
    
    # 加载划分后的训练数据（用于策略实验）
    logger.log("\n加载划分后的训练数据（12000条）和验证数据（3000条）...")
    load_start = time.time()
    df_train_split, train_split_graphs = load_data(TRAIN_SPLIT_CSV, TRAIN_SPLIT_JSON)
    df_val, val_graphs = load_data(VAL_CSV, VAL_JSON)
    load_time = time.time() - load_start
    global_timer.record('data_load_split', load_time * 1000)
    logger.log(f"  划分训练集: {len(df_train_split)} 条")
    logger.log(f"  验证集: {len(df_val)} 条")
    
    # 加载测试数据
    logger.log("\n加载测试数据...")
    load_start = time.time()
    df_test, test_graphs = load_data(TEST_CSV, TEST_JSON)
    load_time = time.time() - load_start
    global_timer.record('data_load_test', load_time * 1000)
    logger.log(f"  测试集: {len(df_test)} 条")
    
    # ======================== 阶段2: 特征准备 ========================
    logger.log_section("阶段2: 特征准备")
    
    # 准备完整训练数据的RF特征
    logger.log("\n准备完整训练数据的RF特征...")
    feat_start = time.time()
    X_train_full, y_train_full, ids_train_full, encoders_full, scaler_full = prepare_rf_features(df_train_full, fit=True)
    feat_time = time.time() - feat_start
    global_timer.record('feature_prep_full', feat_time * 1000)
    logger.log(f"  RF特征维度: {X_train_full.shape[1]}")
    
    # 准备划分训练数据的RF特征
    logger.log("\n准备划分训练数据的RF特征...")
    feat_start = time.time()
    X_train_split, y_train_split, ids_train_split, encoders_split, scaler_split = prepare_rf_features(df_train_split, fit=True)
    X_val, y_val, ids_val, _, _ = prepare_rf_features(df_val, encoders_split, scaler_split)
    feat_time = time.time() - feat_start
    global_timer.record('feature_prep_split', feat_time * 1000)
    
    # 准备测试数据的RF特征（使用完整训练数据的编码器）
    logger.log("\n准备测试数据的RF特征...")
    feat_start = time.time()
    X_test, y_test, ids_test, _, _ = prepare_rf_features(df_test, encoders_full, scaler_full)
    feat_time = time.time() - feat_start
    global_timer.record('feature_prep_test', feat_time * 1000)
    
    # ======================== 阶段3: 图数据构建 ========================
    logger.log_section("阶段3: 图数据构建")
    
    logger.log("\n构建完整训练数据的图数据...")
    graph_start = time.time()
    train_full_pyg = build_pyg_data(df_train_full, TRAIN_FULL_JSON)
    graph_time = time.time() - graph_start
    global_timer.record('graph_build_full', graph_time * 1000)
    logger.log(f"  完整训练图数量: {len(train_full_pyg)}")
    
    logger.log("\n构建划分训练数据的图数据...")
    graph_start = time.time()
    train_split_pyg = build_pyg_data(df_train_split, TRAIN_SPLIT_JSON)
    val_pyg = build_pyg_data(df_val, VAL_JSON)
    graph_time = time.time() - graph_start
    global_timer.record('graph_build_split', graph_time * 1000)
    logger.log(f"  划分训练图数量: {len(train_split_pyg)}")
    logger.log(f"  验证图数量: {len(val_pyg)}")
    
    logger.log("\n构建测试数据的图数据...")
    graph_start = time.time()
    test_pyg = build_pyg_data(df_test, TEST_JSON)
    graph_time = time.time() - graph_start
    global_timer.record('graph_build_test', graph_time * 1000)
    logger.log(f"  测试图数量: {len(test_pyg)}")
    
    # 数据分布统计
    normal_train_full = (df_train_full['classification'] == 'normal').sum()
    outlier_train_full = (df_train_full['classification'] == 'outlier').sum()
    logger.log(f"\n完整训练集分布: 正常={normal_train_full}, 异常={outlier_train_full}")
    
    normal_test = (df_test['classification'] == 'normal').sum()
    outlier_test = (df_test['classification'] == 'outlier').sum()
    logger.log(f"测试集分布: 正常={normal_test}, 异常={outlier_test}")
    
    results = []
    
    # ======================== 固定权重实验（使用完整训练数据15000条） ========================
    logger.log_section("=" * 40 + "\n固定权重实验组（使用完整训练数据15000条）\n" + "=" * 40)
    
    # 实验1: RF单模型
    logger.log_section("实验1: RF单模型")
    timer1 = DetailedTimeRecorder()
    rf_model_full, rf_train_time = train_rf_model(X_train_full, y_train_full, logger, timer1)
    rf_probs_test_full = get_rf_probs(rf_model_full, X_test, timer1)
    rf_preds_full = (rf_probs_test_full > 0.5).astype(int)
    result1 = run_single_experiment("RF单模型", y_test, rf_preds_full, timer1, logger)
    result1['train_time'] = rf_train_time
    result1['data_config'] = '完整训练数据(15000条)'
    results.append(result1)
    
    # 实验2: 单层GCN模型
    logger.log_section("实验2: 单层GCN模型")
    timer2 = DetailedTimeRecorder()
    gcn1_model, gcn1_train_time = train_gcn_model(train_full_pyg, 64, 1, logger, timer2)
    gcn1_probs_test = get_gcn_probs(gcn1_model, test_pyg, timer2)
    y_test_gcn1 = np.array([data.y.item() for data in test_pyg])
    gcn1_preds = (gcn1_probs_test > 0.5).astype(int)
    result2 = run_single_experiment("单层GCN", y_test_gcn1, gcn1_preds, timer2, logger)
    result2['train_time'] = gcn1_train_time
    result2['data_config'] = '完整训练数据(15000条)'
    results.append(result2)
    
    # 实验3: 双层GCN模型
    logger.log_section("实验3: 双层GCN模型")
    timer3 = DetailedTimeRecorder()
    gcn2_model, gcn2_train_time = train_gcn_model(train_full_pyg, 64, 2, logger, timer3)
    gcn2_probs_test = get_gcn_probs(gcn2_model, test_pyg, timer3)
    y_test_gcn2 = np.array([data.y.item() for data in test_pyg])
    gcn2_preds = (gcn2_probs_test > 0.5).astype(int)
    result3 = run_single_experiment("双层GCN", y_test_gcn2, gcn2_preds, timer3, logger)
    result3['train_time'] = gcn2_train_time
    result3['data_config'] = '完整训练数据(15000条)'
    results.append(result3)
    
    # 实验3.5: 三层GCN模型
    logger.log_section("实验3.5: 三层GCN模型")
    timer3_5 = DetailedTimeRecorder()
    gcn3_model, gcn3_train_time = train_gcn_model(train_full_pyg, 64, 3, logger, timer3_5)
    gcn3_probs_test = get_gcn_probs(gcn3_model, test_pyg, timer3_5)
    y_test_gcn3 = np.array([data.y.item() for data in test_pyg])
    gcn3_preds = (gcn3_probs_test > 0.5).astype(int)
    result3_5 = run_single_experiment("三层GCN", y_test_gcn3, gcn3_preds, timer3_5, logger)
    result3_5['train_time'] = gcn3_train_time
    result3_5['data_config'] = '完整训练数据(15000条)'
    results.append(result3_5)
    
    # 实验4: 固定权重融合
    logger.log_section(f"实验4: RF + 双层GCN + 固定权重融合 (α={BEST_FIXED_ALPHA})")
    
    # 对齐数据
    test_trace_ids = [data.trace_id for data in test_pyg]
    test_id_to_idx = {tid: i for i, tid in enumerate(ids_test)}
    rf_probs_aligned = np.array([rf_probs_test_full[test_id_to_idx[tid]] for tid in test_trace_ids])
    y_test_aligned = np.array([y_test[test_id_to_idx[tid]] for tid in test_trace_ids])
    
    logger.log(f"对齐后的测试数据量: {len(test_trace_ids)}")
    
    timer4 = DetailedTimeRecorder()
    timer4.start('total')
    
    fused_probs_4 = []
    for i in range(len(rf_probs_aligned)):
        timer4.start('fusion')
        fp, _ = fixed_weight_fusion(rf_probs_aligned[i], gcn2_probs_test[i], alpha=BEST_FIXED_ALPHA)
        timer4.stop()
        fused_probs_4.append(fp)
    
    fused_probs_4 = np.array(fused_probs_4)
    timer4.stop()
    
    preds_4 = (fused_probs_4 > 0.5).astype(int)
    result4 = run_single_experiment("固定权重融合", y_test_gcn2, preds_4, timer4, logger)
    result4['data_config'] = '完整训练数据(15000条)'
    result4['alpha'] = BEST_FIXED_ALPHA
    results.append(result4)
    
    # ======================== 策略实验（使用划分训练数据12000条+验证集3000条） ========================
    logger.log_section("=" * 40 + "\n策略实验组（使用划分训练数据12000条+验证集3000条）\n" + "=" * 40)
    
    # 步骤5: 参数调优
    logger.log_section("步骤5: 在验证集上调优RF和GCN模型参数")
    
    logger.log("\n>>> 调优RF模型参数...")
    rf_best_params, rf_best_model = search_rf_params(X_train_split, y_train_split, X_val, y_val, logger, global_timer)
    
    logger.log("\n>>> 调优GCN模型参数...")
    gcn_best_params, gcn_best_model = search_gcn_params(train_split_pyg, val_pyg, logger, global_timer, epochs=15)
    
    # 步骤6: 策略参数搜索
    logger.log_section("步骤6: 在验证集上搜索融合策略参数")
    
    val_trace_ids = [data.trace_id for data in val_pyg]
    val_id_to_idx = {tid: i for i, tid in enumerate(ids_val)}
    rf_probs_val_aligned = np.array([rf_best_model.predict_proba(X_val)[val_id_to_idx[tid], 1] for tid in val_trace_ids])
    gcn_probs_val = get_gcn_probs(gcn_best_model, val_pyg, global_timer)
    y_val_gcn = np.array([data.y.item() for data in val_pyg])
    
    logger.log(f"验证集数据量: {len(val_trace_ids)}")
    logger.log(f"验证集分布: 正常={int(sum(y_val_gcn == 0))}, 异常={int(sum(y_val_gcn == 1))}")
    
    logger.log(f"\n>>> 使用固定权重 alpha={BEST_FIXED_ALPHA}（来自505组实验）")
    
    logger.log("\n>>> 搜索单策略1参数...")
    single_strategy1_params = search_single_strategy1_params(rf_probs_val_aligned, gcn_probs_val, y_val_gcn, logger, global_timer)
    
    logger.log("\n>>> 搜索单策略2参数...")
    single_strategy2_params = search_single_strategy2_params(rf_probs_val_aligned, gcn_probs_val, y_val_gcn, BEST_FIXED_ALPHA, logger, global_timer)
    
    logger.log("\n>>> 搜索双策略参数...")
    dual_strategy_params = search_dual_strategy_params(rf_probs_val_aligned, gcn_probs_val, y_val_gcn, logger, global_timer)
    
    # 获取测试集预测概率
    rf_probs_test_opt = rf_best_model.predict_proba(X_test)[:, 1]
    gcn_probs_test_opt = get_gcn_probs(gcn_best_model, test_pyg, global_timer)
    rf_probs_aligned_opt = np.array([rf_probs_test_opt[test_id_to_idx[tid]] for tid in test_trace_ids])
    
    # 实验5: 自适应阈值融合(单策略2)
    logger.log_section("实验5: RF + 双层GCN + 自适应阈值融合 (单策略2, 调优后)")
    logger.log(f"使用单策略2最优参数:")
    logger.log(f"  RF: n_estimators={rf_best_params['n_estimators']}, max_depth={rf_best_params['max_depth']}")
    logger.log(f"  GCN: hidden={gcn_best_params['hidden_channels']}, lr={gcn_best_params['learning_rate']}")
    logger.log(f"  融合: alpha={single_strategy2_params['alpha']}")
    logger.log(f"        threshold_adjust_rf={single_strategy2_params['threshold_adjust_rf']}, threshold_adjust_gcn={single_strategy2_params['threshold_adjust_gcn']}")
    
    timer5 = DetailedTimeRecorder()
    timer5.start('total')
    
    fused_probs_5 = []
    thresholds_5 = []
    for i in range(len(rf_probs_aligned_opt)):
        timer5.start('fusion')
        fp, th = adaptive_threshold_fusion(
            rf_probs_aligned_opt[i], gcn_probs_test_opt[i],
            alpha=single_strategy2_params['alpha'],
            threshold_adjust_rf=single_strategy2_params['threshold_adjust_rf'],
            threshold_adjust_gcn=single_strategy2_params['threshold_adjust_gcn']
        )
        timer5.stop()
        fused_probs_5.append(fp)
        thresholds_5.append(th)
    
    fused_probs_5 = np.array(fused_probs_5)
    thresholds_5 = np.array(thresholds_5)
    timer5.stop()
    
    preds_5 = (fused_probs_5 > thresholds_5).astype(int)
    result5 = run_single_experiment("自适应阈值融合(单策略2,调优)", y_test_gcn2, preds_5, timer5, logger)
    result5['rf_params'] = rf_best_params
    result5['gcn_params'] = gcn_best_params
    result5['fusion_params'] = single_strategy2_params
    result5['data_config'] = '划分训练数据(12000条)+验证集(3000条)'
    results.append(result5)
    
    # 实验6: 置信度动态权重融合(单策略1)
    logger.log_section("实验6: RF + 双层GCN + 置信度动态权重融合 (单策略1, 调优后)")
    logger.log(f"使用单策略1最优参数:")
    logger.log(f"  RF: n_estimators={rf_best_params['n_estimators']}, max_depth={rf_best_params['max_depth']}")
    logger.log(f"  GCN: hidden={gcn_best_params['hidden_channels']}, lr={gcn_best_params['learning_rate']}")
    logger.log(f"  融合: alpha={single_strategy1_params['alpha']}, conf_threshold={single_strategy1_params['conf_threshold']}")
    logger.log(f"        weight_high={single_strategy1_params['weight_high_conf']}, weight_low={single_strategy1_params['weight_low_conf']}")
    
    timer6 = DetailedTimeRecorder()
    timer6.start('total')
    
    fused_probs_6 = []
    for i in range(len(rf_probs_aligned_opt)):
        timer6.start('fusion')
        fp, _ = confidence_weight_fusion(
            rf_probs_aligned_opt[i], gcn_probs_test_opt[i],
            alpha=single_strategy1_params['alpha'],
            conf_threshold=single_strategy1_params['conf_threshold'],
            weight_high_conf=single_strategy1_params['weight_high_conf'],
            weight_low_conf=single_strategy1_params['weight_low_conf']
        )
        timer6.stop()
        fused_probs_6.append(fp)
    
    fused_probs_6 = np.array(fused_probs_6)
    timer6.stop()
    
    preds_6 = (fused_probs_6 > 0.5).astype(int)
    result6 = run_single_experiment("置信度动态权重融合(单策略1,调优)", y_test_gcn2, preds_6, timer6, logger)
    result6['rf_params'] = rf_best_params
    result6['gcn_params'] = gcn_best_params
    result6['fusion_params'] = single_strategy1_params
    result6['data_config'] = '划分训练数据(12000条)+验证集(3000条)'
    results.append(result6)
    
    # 实验7: 置信度自适应融合(双策略)
    logger.log_section("实验7: RF + 双层GCN + 置信度自适应融合 (双策略, 最终方案, 调优后)")
    logger.log(f"使用双策略最优参数:")
    logger.log(f"  RF: n_estimators={rf_best_params['n_estimators']}, max_depth={rf_best_params['max_depth']}")
    logger.log(f"  GCN: hidden={gcn_best_params['hidden_channels']}, lr={gcn_best_params['learning_rate']}")
    logger.log(f"  融合: alpha={dual_strategy_params['alpha']}, conf_threshold={dual_strategy_params['conf_threshold']}")
    logger.log(f"        weight_high={dual_strategy_params['weight_high_conf']}, weight_low={dual_strategy_params['weight_low_conf']}")
    logger.log(f"        thresh_rf={dual_strategy_params['threshold_adjust_rf']}, thresh_gcn={dual_strategy_params['threshold_adjust_gcn']}")
    
    timer7 = DetailedTimeRecorder()
    timer7.start('total')
    
    fused_probs_7 = []
    thresholds_7 = []
    for i in range(len(rf_probs_aligned_opt)):
        timer7.start('fusion')
        fp, th = confidence_adaptive_fusion(
            rf_probs_aligned_opt[i], gcn_probs_test_opt[i],
            alpha=dual_strategy_params['alpha'],
            conf_threshold=dual_strategy_params['conf_threshold'],
            weight_high_conf=dual_strategy_params['weight_high_conf'],
            weight_low_conf=dual_strategy_params['weight_low_conf'],
            threshold_adjust_rf=dual_strategy_params['threshold_adjust_rf'],
            threshold_adjust_gcn=dual_strategy_params['threshold_adjust_gcn']
        )
        timer7.stop()
        fused_probs_7.append(fp)
        thresholds_7.append(th)
    
    fused_probs_7 = np.array(fused_probs_7)
    thresholds_7 = np.array(thresholds_7)
    timer7.stop()
    
    preds_7 = (fused_probs_7 > thresholds_7).astype(int)
    result7 = run_single_experiment("置信度自适应融合(调优)", y_test_gcn2, preds_7, timer7, logger)
    result7['rf_params'] = rf_best_params
    result7['gcn_params'] = gcn_best_params
    result7['fusion_params'] = dual_strategy_params
    result7['data_config'] = '划分训练数据(12000条)+验证集(3000条)'
    results.append(result7)
    
    # 实验8: 神经网络学习权重融合
    logger.log_section("实验8: RF + 双层GCN + 神经网络学习权重融合 (调优后)")
    
    fusion_net, fusion_train_time = train_fusion_net(
        rf_probs_val_aligned, gcn_probs_val, y_val_gcn, logger, global_timer
    )
    
    timer8 = DetailedTimeRecorder()
    timer8.start('total')
    
    fusion_net.eval()
    with torch.no_grad():
        rf_t = torch.tensor(rf_probs_aligned_opt, dtype=torch.float32).to(DEVICE)
        gcn_t = torch.tensor(gcn_probs_test_opt, dtype=torch.float32).to(DEVICE)
        rf_conf_t = torch.abs(rf_t - 0.5) * 2
        gcn_conf_t = torch.abs(gcn_t - 0.5) * 2
        
        timer8.start('fusion')
        alphas = fusion_net(rf_t, gcn_t, rf_conf_t, gcn_conf_t)
        timer8.stop()
        
        fused_probs_8 = (alphas * rf_t + (1 - alphas) * gcn_t).cpu().numpy()
    
    timer8.stop()
    
    preds_8 = (fused_probs_8 > 0.5).astype(int)
    result8 = run_single_experiment("神经网络学习权重融合(调优)", y_test_gcn2, preds_8, timer8, logger)
    result8['train_time'] = fusion_train_time
    result8['rf_params'] = rf_best_params
    result8['gcn_params'] = gcn_best_params
    result8['data_config'] = '划分训练数据(12000条)+验证集(3000条)'
    results.append(result8)
    
    # ======================== 结果汇总 ========================
    logger.log_section("实验结果汇总")
    
    headers = ["实验方案", "Accuracy", "Precision", "Recall", "F1-Score", "FPR", "FNR", "数据配置"]
    rows = []
    for r in results:
        m = r['metrics']
        rows.append([
            r['name'],
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['fpr']:.4f}",
            f"{m['fnr']:.4f}",
            r.get('data_config', '-')
        ])
    
    logger.log_table(headers, rows)
    
    best_result = max(results, key=lambda x: x['metrics']['f1'])
    logger.log(f"\n最佳方案: {best_result['name']}, F1={best_result['metrics']['f1']:.4f}")
    
    baseline_f1 = results[0]['metrics']['f1']
    improvement = (best_result['metrics']['f1'] - baseline_f1) / baseline_f1 * 100
    logger.log(f"相比RF单模型提升: {improvement:.2f}%")
    
    # ======================== 详细耗时分析 ========================
    logger.log_section("详细耗时分析汇总")
    
    # 各阶段耗时
    logger.log("\n一、数据准备阶段耗时:")
    prep_times = global_timer.summary()
    for name in ['data_load_full', 'data_load_split', 'data_load_test', 
                 'feature_prep_full', 'feature_prep_split', 'feature_prep_test',
                 'graph_build_full', 'graph_build_split', 'graph_build_test']:
        if name in prep_times:
            logger.log(f"  {name}: {prep_times[name]['total']:.2f}ms")
    
    logger.log("\n二、模型训练耗时:")
    for name in ['rf_train', 'gcn1_train', 'gcn2_train', 'gcn3_train', 'fusion_net_train']:
        if name in prep_times:
            logger.log(f"  {name}: {prep_times[name]['total']:.2f}ms")
    
    logger.log("\n三、参数搜索耗时:")
    for name in ['rf_param_search', 'gcn_param_search', 'strategy1_param_search', 
                 'strategy2_param_search', 'dual_strategy_param_search']:
        if name in prep_times:
            logger.log(f"  {name}: {prep_times[name]['total']:.2f}ms ({prep_times[name]['total']/1000:.2f}秒)")
    
    logger.log("\n四、推理耗时:")
    for name in ['rf_inference', 'gcn_inference']:
        if name in prep_times:
            logger.log(f"  {name}: {prep_times[name]['total']:.2f}ms")
    
    # 各实验耗时表格
    logger.log("\n五、各实验组耗时详情:")
    time_headers = ["实验方案", "总耗时(ms)", "平均耗时(ms/条)", "融合耗时(ms)", "数据配置"]
    time_rows = []
    for r in results:
        t = r.get('time', {})
        total = t.get('total', {}).get('total', 0)
        avg = t.get('total', {}).get('avg', 0)
        fusion = t.get('fusion', {}).get('total', 0)
        time_rows.append([r['name'], f"{total:.2f}", f"{avg:.4f}", f"{fusion:.2f}", r.get('data_config', '-')])
    
    logger.log_table(time_headers, time_rows)
    
    # 保存结果
    results_json = os.path.join(LOG_DIR, 'experiment_results.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.log(f"\n结果已保存至: {results_json}")
    
    logger.finalize()
    
    return results


if __name__ == "__main__":
    run_experiments()
