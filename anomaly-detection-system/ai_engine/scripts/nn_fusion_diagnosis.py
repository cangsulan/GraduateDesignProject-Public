# -*- coding: utf-8 -*-
"""
神经网络融合诊断脚本 - 检查数据泄露和结果合理性
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import degree

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

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


class GCNModel(nn.Module):
    def __init__(self, hidden_channels=64, num_layers=2):
        super().__init__()
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


def train_gcn_model(train_pyg, hidden_channels, num_layers, epochs=12):
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


def run_diagnosis():
    print("=" * 80)
    print("神经网络融合诊断分析")
    print("=" * 80)
    
    df_train_full, _ = load_data(TRAIN_CSV, TRAIN_JSON)
    df_test, _ = load_data(TEST_CSV, TEST_JSON)
    
    df_train, df_val = train_test_split(
        df_train_full, test_size=0.2, random_state=SEED,
        stratify=df_train_full['classification']
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(df_train)} 条")
    print(f"  验证集: {len(df_val)} 条")
    print(f"  测试集: {len(df_test)} 条")
    
    X_train, y_train, ids_train, encoders, scaler = prepare_rf_features(df_train, fit=True)
    X_val, y_val, ids_val, _, _ = prepare_rf_features(df_val, encoders, scaler)
    X_test, y_test, ids_test, _, _ = prepare_rf_features(df_test, encoders, scaler)
    
    train_pyg = build_pyg_data(df_train, TRAIN_JSON)
    val_pyg = build_pyg_data(df_val, TRAIN_JSON)
    test_pyg = build_pyg_data(df_test, TEST_JSON)
    
    print(f"\n图数据:")
    print(f"  训练图: {len(train_pyg)}")
    print(f"  验证图: {len(val_pyg)}")
    print(f"  测试图: {len(test_pyg)}")
    
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2,
        max_features='sqrt', random_state=SEED, class_weight='balanced', n_jobs=-1
    )
    rf_model.fit(X_res, y_res)
    
    gcn_model = train_gcn_model(train_pyg, 32, 2, epochs=12)
    
    rf_probs_train = rf_model.predict_proba(X_train)[:, 1]
    rf_probs_val = rf_model.predict_proba(X_val)[:, 1]
    rf_probs_test = rf_model.predict_proba(X_test)[:, 1]
    
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
    gcn_probs_test = get_gcn_probs(test_pyg)
    
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
    
    print("\n" + "=" * 80)
    print("诊断1: RF和GCN在训练集上的表现（应该很高，因为见过数据）")
    print("=" * 80)
    
    rf_train_preds = (rf_probs_train_aligned > 0.5).astype(int)
    gcn_train_preds = (gcn_probs_train > 0.5).astype(int)
    
    print(f"\nRF在训练集上: F1={f1_score(y_train_gcn, rf_train_preds):.4f}")
    print(f"GCN在训练集上: F1={f1_score(y_train_gcn, gcn_train_preds):.4f}")
    
    print("\n" + "=" * 80)
    print("诊断2: RF和GCN在验证集上的表现（应该较低，未见数据）")
    print("=" * 80)
    
    rf_val_preds = (rf_probs_val_aligned > 0.5).astype(int)
    gcn_val_preds = (gcn_probs_val > 0.5).astype(int)
    
    print(f"\nRF在验证集上: F1={f1_score(y_val_gcn, rf_val_preds):.4f}")
    print(f"GCN在验证集上: F1={f1_score(y_val_gcn, gcn_val_preds):.4f}")
    
    print("\n" + "=" * 80)
    print("诊断3: RF和GCN在测试集上的表现（完全未见数据）")
    print("=" * 80)
    
    rf_test_preds = (rf_probs_test_aligned > 0.5).astype(int)
    gcn_test_preds = (gcn_probs_test > 0.5).astype(int)
    
    print(f"\nRF在测试集上: F1={f1_score(y_test_gcn, rf_test_preds):.4f}")
    print(f"GCN在测试集上: F1={f1_score(y_test_gcn, gcn_test_preds):.4f}")
    
    print("\n" + "=" * 80)
    print("诊断4: 检查训练集和测试集的预测概率分布差异")
    print("=" * 80)
    
    print(f"\nRF概率分布:")
    print(f"  训练集: mean={rf_probs_train_aligned.mean():.4f}, std={rf_probs_train_aligned.std():.4f}")
    print(f"  验证集: mean={rf_probs_val_aligned.mean():.4f}, std={rf_probs_val_aligned.std():.4f}")
    print(f"  测试集: mean={rf_probs_test_aligned.mean():.4f}, std={rf_probs_test_aligned.std():.4f}")
    
    print(f"\nGCN概率分布:")
    print(f"  训练集: mean={gcn_probs_train.mean():.4f}, std={gcn_probs_train.std():.4f}")
    print(f"  验证集: mean={gcn_probs_val.mean():.4f}, std={gcn_probs_val.std():.4f}")
    print(f"  测试集: mean={gcn_probs_test.mean():.4f}, std={gcn_probs_test.std():.4f}")
    
    print("\n" + "=" * 80)
    print("诊断5: 多次运行神经网络训练，检查结果稳定性")
    print("=" * 80)
    
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
    
    results = []
    
    for run in range(5):
        torch.manual_seed(SEED + run * 1000)
        
        model = FusionNetV3_BN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss()
        
        train_features_dev = train_features.to(DEVICE)
        train_labels_dev = train_labels.to(DEVICE)
        val_features_dev = val_features.to(DEVICE)
        val_labels_dev = val_labels.to(DEVICE)
        
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(1, 76):
            model.train()
            optimizer.zero_grad()
            alphas = model(train_features_dev)
            fused = alphas * train_features_dev[:, 0] + (1 - alphas) * train_features_dev[:, 1]
            train_loss = criterion(fused, train_labels_dev)
            train_loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_alphas = model(val_features_dev)
                val_fused = val_alphas * val_features_dev[:, 0] + (1 - val_alphas) * val_features_dev[:, 1]
                val_loss = criterion(val_fused, val_labels_dev)
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
        
        model.load_state_dict(best_model_state)
        model.eval()
        
        with torch.no_grad():
            test_features_dev = test_features.to(DEVICE)
            test_alphas = model(test_features_dev)
            test_fused = test_alphas * test_features_dev[:, 0] + (1 - test_alphas) * test_features_dev[:, 1]
            fused_probs = test_fused.cpu().numpy()
        
        preds = (fused_probs > 0.5).astype(int)
        f1 = f1_score(y_test_gcn, preds)
        
        tn, fp, fn, tp = confusion_matrix(y_test_gcn, preds).ravel()
        
        results.append({
            'run': run + 1,
            'best_epoch': best_epoch,
            'val_loss': best_val_loss,
            'test_f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })
        
        print(f"\n运行 {run+1}: best_epoch={best_epoch}, val_loss={best_val_loss:.4f}, test_F1={f1:.4f}")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    print("\n" + "=" * 80)
    print("诊断6: 结果汇总")
    print("=" * 80)
    
    f1_scores = [r['test_f1'] for r in results]
    print(f"\n5次运行的F1分数:")
    print(f"  平均值: {np.mean(f1_scores):.4f}")
    print(f"  标准差: {np.std(f1_scores):.4f}")
    print(f"  最小值: {np.min(f1_scores):.4f}")
    print(f"  最大值: {np.max(f1_scores):.4f}")
    
    if np.std(f1_scores) > 0.05:
        print("\n⚠️ 警告: 结果波动较大，可能存在不稳定性!")
    else:
        print("\n✓ 结果相对稳定")
    
    print("\n" + "=" * 80)
    print("诊断7: 检查是否存在数据泄露")
    print("=" * 80)
    
    train_ids = set([data.trace_id for data in train_pyg])
    val_ids = set([data.trace_id for data in val_pyg])
    test_ids = set([data.trace_id for data in test_pyg])
    
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    print(f"\n训练集与验证集重叠: {len(train_val_overlap)} 条")
    print(f"训练集与测试集重叠: {len(train_test_overlap)} 条")
    print(f"验证集与测试集重叠: {len(val_test_overlap)} 条")
    
    if len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
        print("\n⚠️ 警告: 存在数据泄露!")
    else:
        print("\n✓ 无数据泄露")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnosis()
