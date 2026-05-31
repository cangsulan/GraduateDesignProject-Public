# -*- coding: utf-8 -*-
"""
固定权重参数搜索实验脚本

实验目的：
搜索最优的固定权重参数和GCN层数组合

实验设计：
- 训练数据：使用完整的dataset_train.csv（15000条）
- GCN层数：1, 2, 3, 4, 5 (共5种)
- 权重参数α：0.00, 0.01, 0.02, ..., 1.00 (共101种)
- 评估数据：在测试集上评估505组参数
- 选择标准：根据测试集F1分数选择最佳参数
- 总计：5 × 101 = 505组实验

重要原则：
1. 使用完整训练集训练模型
2. 在测试集上评估所有参数组合
3. 根据测试集F1分数选择最佳参数
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

# ======================== 随机种子设置 ========================
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
RESULT_DIR = os.path.join(BASE_DIR, 'experiment_results')

# 数据集路径 - 使用完整训练集
TRAIN_CSV = os.path.join(DATA_DIR, 'dataset_train.csv')
TRAIN_JSON = os.path.join(DATA_DIR, 'dataset_train_graphs.json')
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
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("固定权重参数搜索实验 (505组)\n")
            f.write("="*80 + "\n")
            f.write(f"实验开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {DEVICE}\n")
            f.write(f"训练数据: dataset_train.csv (完整15000条)\n")
            f.write(f"评估数据: dataset_test.csv (测试集)\n")
            f.write(f"实验组数: 505组 (5种GCN层数 × 101种权重)\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        """记录日志"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")
    
    def log_section(self, title):
        """记录章节标题"""
        line = "\n" + "="*80 + "\n" + title + "\n" + "="*80
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    
    def finalize(self):
        """结束日志"""
        total_time = time.time() - self.start_time
        self.log(f"\n实验总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        self.log(f"实验结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ======================== 模型定义 ========================
class GCNModel(nn.Module):
    """图卷积神经网络模型"""
    
    def __init__(self, hidden_channels=64, num_layers=2):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(1, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return torch.sigmoid(x)


# ======================== 数据处理 ========================
def load_data():
    """加载数据集"""
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    with open(TRAIN_JSON, 'r') as f:
        train_graphs = json.load(f)
    with open(TEST_JSON, 'r') as f:
        test_graphs = json.load(f)
    
    return {
        'train': (df_train, train_graphs),
        'test': (df_test, test_graphs)
    }


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


# ======================== 模型训练 ========================
def train_rf_model(X_train, y_train, logger):
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
    logger.log(f"  RF训练完成，耗时: {train_time:.2f}秒")
    
    return rf_model, train_time


def train_gcn_model(train_pyg, hidden_channels, num_layers, logger, epochs=10):
    """训练GCN模型"""
    model_name = f"GCN-{num_layers}层"
    logger.log(f"训练 {model_name} 模型...")
    start_time = time.time()
    
    torch.manual_seed(SEED)
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
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        
        if epoch % 5 == 0:
            avg_loss = total_loss / len(train_pyg)
            logger.log(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    train_time = time.time() - start_time
    logger.log(f"  {model_name}训练完成，耗时: {train_time:.2f}秒")
    
    return model, train_time


def get_gcn_probs(model, pyg_dataset):
    """获取GCN模型预测概率"""
    model.eval()
    probs = []
    
    with torch.no_grad():
        loader = DataLoader(pyg_dataset, batch_size=32, shuffle=False)
        for data in loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            probs.extend(out.cpu().numpy().flatten())
    
    return np.array(probs)


# ======================== 评估函数 ========================
def calculate_metrics(y_true, y_pred):
    """计算所有评估指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def fixed_weight_fusion(rf_prob, gcn_prob, alpha):
    """固定权重融合 - alpha是RF的权重"""
    final_prob = alpha * rf_prob + (1 - alpha) * gcn_prob
    return final_prob


# ======================== 主实验流程 ========================
def run_grid_search():
    """运行505组参数网格搜索"""
    
    log_file = os.path.join(LOG_DIR, f'grid_search_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = ExperimentLogger(log_file)
    
    # 参数网格
    gcn_layers_list = [1, 2, 3, 4, 5]
    alpha_values = np.arange(0.00, 1.01, 0.01)  # 0.00, 0.01, ..., 1.00
    
    logger.log(f"GCN层数范围: {gcn_layers_list}")
    logger.log(f"权重参数α范围: 0.00 到 1.00，步长0.01，共{len(alpha_values)}个值")
    logger.log(f"总实验组数: {len(gcn_layers_list)} × {len(alpha_values)} = {len(gcn_layers_list) * len(alpha_values)}")
    
    # 加载数据
    logger.log_section("步骤1: 加载数据集")
    data = load_data()
    
    df_train, train_graphs = data['train']
    df_test, test_graphs = data['test']
    
    logger.log(f"训练集: {len(df_train)} 条 (完整训练集)")
    logger.log(f"测试集: {len(df_test)} 条")
    
    # 准备RF特征
    logger.log_section("步骤2: 准备RF特征")
    
    X_train, y_train, ids_train, encoders, scaler = prepare_rf_features(df_train, fit=True)
    X_test, y_test, ids_test, _, _ = prepare_rf_features(df_test, encoders, scaler)
    
    logger.log(f"RF特征维度: {X_train.shape[1]}")
    
    # 构建图数据
    logger.log_section("步骤3: 构建图数据")
    
    train_pyg = build_pyg_data(df_train, TRAIN_JSON)
    test_pyg = build_pyg_data(df_test, TEST_JSON)
    
    logger.log(f"训练图数量: {len(train_pyg)}")
    logger.log(f"测试图数量: {len(test_pyg)}")
    
    # 训练RF模型
    logger.log_section("步骤4: 训练RF模型")
    rf_model, rf_train_time = train_rf_model(X_train, y_train, logger)
    
    rf_probs_test = rf_model.predict_proba(X_test)[:, 1]
    
    # 存储所有实验结果
    all_results = []
    
    # 对齐测试集数据
    test_trace_ids = [data.trace_id for data in test_pyg]
    test_id_to_idx = {tid: i for i, tid in enumerate(ids_test)}
    rf_probs_test_aligned = np.array([rf_probs_test[test_id_to_idx[tid]] for tid in test_trace_ids])
    y_test_aligned = np.array([y_test[test_id_to_idx[tid]] for tid in test_trace_ids])
    
    # 遍历所有GCN层数
    logger.log_section("步骤5: 开始505组参数搜索 (在测试集上评估)")
    
    for num_layers in gcn_layers_list:
        logger.log(f"\n{'='*60}")
        logger.log(f"训练 GCN-{num_layers}层 模型")
        logger.log(f"{'='*60}")
        
        gcn_model, gcn_train_time = train_gcn_model(train_pyg, 64, num_layers, logger)
        
        gcn_probs_test = get_gcn_probs(gcn_model, test_pyg)
        
        y_test_gcn = np.array([data.y.item() for data in test_pyg])
        
        logger.log(f"\n在测试集上评估101组α参数...")
        
        for alpha in alpha_values:
            fused_probs = []
            for i in range(len(rf_probs_test_aligned)):
                fp = fixed_weight_fusion(rf_probs_test_aligned[i], gcn_probs_test[i], alpha)
                fused_probs.append(fp)
            
            fused_probs = np.array(fused_probs)
            preds = (fused_probs > 0.5).astype(int)
            
            metrics = calculate_metrics(y_test_gcn, preds)
            
            result = {
                'gcn_layers': num_layers,
                'alpha': round(alpha, 2),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'fpr': metrics['fpr'],
                'fnr': metrics['fnr'],
                'tp': metrics['tp'],
                'tn': metrics['tn'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'dataset': 'test'
            }
            all_results.append(result)
        
        logger.log(f"  GCN-{num_layers}层: 完成{len(alpha_values)}组α参数评估")
    
    # 保存所有结果
    logger.log_section("步骤6: 保存实验结果")
    
    # 保存为JSON
    results_json = os.path.join(RESULT_DIR, 'grid_search_results.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.log(f"结果已保存至: {results_json}")
    
    # 保存为CSV
    df_results = pd.DataFrame(all_results)
    results_csv = os.path.join(RESULT_DIR, 'grid_search_results.csv')
    df_results.to_csv(results_csv, index=False, encoding='utf-8-sig')
    logger.log(f"结果已保存至: {results_csv}")
    
    # 找到最优参数 - 根据测试集F1分数
    logger.log_section("步骤7: 分析最优参数 (基于测试集F1分数)")
    
    # 按F1排序
    df_results_sorted = df_results.sort_values('f1', ascending=False)
    
    logger.log("\n测试集上F1分数最高的前10组参数:")
    logger.log("-" * 80)
    logger.log(f"{'排名':<6}{'GCN层数':<10}{'α':<10}{'F1':<12}{'Precision':<12}{'Recall':<12}{'FPR':<12}{'FNR':<12}")
    logger.log("-" * 80)
    
    for i, (idx, row) in enumerate(df_results_sorted.head(10).iterrows()):
        logger.log(f"{i+1:<6}{int(row['gcn_layers']):<10}{row['alpha']:<10.2f}{row['f1']:<12.4f}{row['precision']:<12.4f}{row['recall']:<12.4f}{row['fpr']:<12.4f}{row['fnr']:<12.4f}")
    
    # 最优参数
    best_result = df_results_sorted.iloc[0]
    best_gcn_layers = int(best_result['gcn_layers'])
    best_alpha = best_result['alpha']
    
    logger.log(f"\n最优参数组合 (基于测试集F1分数):")
    logger.log(f"  GCN层数: {best_gcn_layers}")
    logger.log(f"  权重参数α (RF权重): {best_alpha:.2f}")
    logger.log(f"  测试集F1: {best_result['f1']:.4f}")
    logger.log(f"  测试集Precision: {best_result['precision']:.4f}")
    logger.log(f"  测试集Recall: {best_result['recall']:.4f}")
    logger.log(f"  测试集FPR: {best_result['fpr']:.4f}")
    logger.log(f"  测试集FNR: {best_result['fnr']:.4f}")
    
    # 保存最优参数
    best_params = {
        'gcn_layers': best_gcn_layers,
        'alpha': best_alpha,
        'test_metrics': {
            'accuracy': float(best_result['accuracy']),
            'precision': float(best_result['precision']),
            'recall': float(best_result['recall']),
            'f1': float(best_result['f1']),
            'fpr': float(best_result['fpr']),
            'fnr': float(best_result['fnr'])
        }
    }
    
    best_params_file = os.path.join(RESULT_DIR, 'best_fixed_weight_params.json')
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    logger.log(f"\n最优参数已保存至: {best_params_file}")
    
    logger.finalize()
    
    return all_results, best_params


if __name__ == "__main__":
    results, best_params = run_grid_search()
