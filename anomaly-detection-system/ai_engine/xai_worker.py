import os
import json
import joblib
import torch
import pika
import time
import numpy as np
import pandas as pd
from loguru import logger
from lime.lime_tabular import LimeTabularExplainer
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.explain import Explainer, GNNExplainer

from train_models import GCN, GCN1Layer, NUMERIC_FEATURES, NUMERIC_FEATURES_BASE, CATEGORICAL_FEATURES, MODEL_DIR
import pymysql # 用于查询检测历史

# ----------------- 全局配置与模型加载 -----------------
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "admin123")

MYSQL_USER = os.getenv("SPRING_DATASOURCE_USERNAME", "root")
MYSQL_PASS = os.getenv("SPRING_DATASOURCE_PASSWORD", "root123")
MYSQL_DB = os.getenv("MYSQL_DATABASE", "anomaly_detection")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rf_model_base = None
rf_model_topo = None
gcn_model = None
gcn_1layer_model = None
scaler_base = None
scaler_topo = None
encoders_base = None
encoders_topo = None
lime_explainer_base = None
lime_explainer_topo = None

def load_models_and_init_lime():
    global rf_model_base, rf_model_topo, gcn_model, gcn_1layer_model, scaler_base, scaler_topo, encoders_base, encoders_topo, lime_explainer_base, lime_explainer_topo
    try:
        logger.info("XAI Worker: 正在加载 RF 模型与预处理器...")
        rf_model_base = joblib.load(os.path.join(MODEL_DIR, "rf_model_base.pkl"))
        rf_model_topo = joblib.load(os.path.join(MODEL_DIR, "rf_model_topo.pkl"))
        scaler_base = joblib.load(os.path.join(MODEL_DIR, "scaler_base.pkl"))
        scaler_topo = joblib.load(os.path.join(MODEL_DIR, "scaler_topo.pkl"))
        encoders_base = joblib.load(os.path.join(MODEL_DIR, "encoder_base.pkl"))
        encoders_topo = joblib.load(os.path.join(MODEL_DIR, "encoder_topo.pkl"))
        
        logger.info(f"XAI Worker: 正在加载 GCN 模型... 设备: {device}")
        gcn_model = GCN(hidden_channels=64).to(device)
        gcn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_model.pth"), map_location=device))
        gcn_model.eval()

        logger.info(f"XAI Worker: 正在加载 GCN1Layer (兜底模型)... 设备: {device}")
        gcn_1layer_model = GCN1Layer(hidden_channels=64).to(device)
        gcn_1layer_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_1layer_model.pth"), map_location=device))
        gcn_1layer_model.eval()
        
        logger.info("XAI Worker: 初始化 LIME Explainer...")
        
        num_features_base = len(NUMERIC_FEATURES_BASE) + len(CATEGORICAL_FEATURES)
        background_data_base = []
        for _ in range(100):
            point = np.random.normal(loc=scaler_base.mean_, scale=scaler_base.scale_, size=num_features_base)
            background_data_base.append(point)
        background_data_base = np.array(background_data_base)
        
        feature_names_base = NUMERIC_FEATURES_BASE + CATEGORICAL_FEATURES
        lime_explainer_base = LimeTabularExplainer(
            background_data_base,
            feature_names=feature_names_base,
            class_names=['Normal', 'Anomaly'],
            mode='classification'
        )
        
        num_features_topo = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        background_data_topo = []
        for _ in range(100):
            point = np.random.normal(loc=scaler_topo.mean_, scale=scaler_topo.scale_, size=num_features_topo)
            background_data_topo.append(point)
        background_data_topo = np.array(background_data_topo)
        
        feature_names_topo = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        lime_explainer_topo = LimeTabularExplainer(
            background_data_topo,
            feature_names=feature_names_topo,
            class_names=['Normal', 'Anomaly'],
            mode='classification'
        )
        
        logger.info("XAI Worker: 模型加载成功，LIME 初始化完毕！")
    except FileNotFoundError:
        logger.warning("XAI Worker: 模型文件缺失，等待 ai-engine 构建生成中... (5s后重试)")
        time.sleep(5)
        load_models_and_init_lime()
    except Exception as e:
        logger.error(f"XAI Worker 启动失败: {e}")
        raise e

def _warmup_models():
    """预热模型，避免第一次请求遭到冷启动长延迟惩罚"""
    logger.info("XAI Worker: 正在执行模型预热 (Warm-up)... 预计耗时数分钟，请耐心等待！")
    try:
        dummy_num_base = {col: 0.0 for col in NUMERIC_FEATURES_BASE}
        dummy_cat = {col: 'unknown' for col in CATEGORICAL_FEATURES}
        dummy_features_base = {**dummy_num_base, **dummy_cat}
        analyze_lime(dummy_features_base, use_topology=False)
        
        dummy_num_topo = {col: 0.0 for col in NUMERIC_FEATURES}
        dummy_features_topo = {**dummy_num_topo, **dummy_cat}
        analyze_lime(dummy_features_topo, use_topology=True)
        
        dummy_nodes = ["111", "222"]
        dummy_graph = [
            {"fromId": "111", "toId": "222", "importance": 0.0},
            {"fromId": "222", "toId": "111", "importance": 0.0}
        ]
        analyze_gcn(dummy_graph, use_fallback=False)
        analyze_gcn(dummy_graph, use_fallback=True)
        logger.info("XAI Worker: 模型预热完毕！完全消除首次运算的冷启动时间。")
    except Exception as e:
        logger.warning(f"XAI Worker: 模型预热过程中发生异常(可忽略): {e}")

def query_features_from_db(trace_id):
    """从数据库查询原始特征"""
    # 这里我们直连 MySQL 读取对应的检测结果，获取特征和图数据
    try:
        # 为了兼容 docker-compose 网络，优先使用 mysql (服务名), 本地测试退回 localhost
        host = "mysql" if "RABBITMQ_HOST" in os.environ else "localhost"
        connection = pymysql.connect(
            host=host,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            database=MYSQL_DB,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            sql = "SELECT features_json, call_graph_json FROM detect_result WHERE trace_id = %s"
            cursor.execute(sql, (trace_id,))
            result = cursor.fetchone()
            return result
    except Exception as e:
        logger.error(f"查询数据库失败: {e}")
        return None
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()

# ----------------- XAI 分析逻辑 -----------------
def analyze_lime(features_dict: dict, use_topology: bool = True) -> dict:
    """使用 LIME 分析 Random Forest 的特征重要性
    
    Args:
        features_dict: 特征字典
        use_topology: 是否使用拓扑特征（决定使用哪个RF模型）
    """
    if not features_dict:
        return {}
    
    if use_topology:
        rf_model = rf_model_topo
        scaler = scaler_topo
        encoders = encoders_topo
        lime_explainer = lime_explainer_topo
        numeric_features = NUMERIC_FEATURES
    else:
        rf_model = rf_model_base
        scaler = scaler_base
        encoders = encoders_base
        lime_explainer = lime_explainer_base
        numeric_features = NUMERIC_FEATURES
        
    df = pd.DataFrame([features_dict])
    for col in numeric_features:
        if col not in df.columns: df[col] = 0.0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns: df[col] = 'unknown'
        
    X_numeric = df[numeric_features].values
    X_categorical = np.zeros((1, len(CATEGORICAL_FEATURES)))
    for i, col in enumerate(CATEGORICAL_FEATURES):
        val = df[col].iloc[0]
        try:
            X_categorical[0, i] = encoders[col].transform([val])[0]
        except ValueError:
            X_categorical[0, i] = 0 
            
    X = np.hstack((X_numeric, X_categorical))
    
    def predict_fn(x_raw):
        x_scaled = scaler.transform(x_raw)
        return rf_model.predict_proba(x_scaled)

    exp = lime_explainer.explain_instance(X[0], predict_fn, num_features=5)
    
    weights = {}
    for feature_desc, weight in exp.as_list():
        weights[feature_desc] = weight
        
    return weights

def analyze_gcn(call_graph: list, use_fallback: bool = False) -> list:
    """使用 GNNExplainer 提取 GCN 中的异常边"""
    if not call_graph or len(call_graph) == 0:
        return []
        
    unique_uuids = set()
    for edge in call_graph:
        unique_uuids.add(edge['fromId'])
        unique_uuids.add(edge['toId'])
        
    unique_uuids = list(unique_uuids)
    uuid_to_idx = {uuid: idx for idx, uuid in enumerate(unique_uuids)}
    idx_to_uuid = {idx: uuid for uuid, idx in uuid_to_idx.items()}
    
    src_nodes = [uuid_to_idx[edge['fromId']] for edge in call_graph]
    dst_nodes = [uuid_to_idx[edge['toId']] for edge in call_graph]
    
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    V = len(unique_uuids)
    if edge_index.size(1) > 0:
        out_degree = degree(edge_index[0], num_nodes=V, dtype=torch.float)
        in_degree = degree(edge_index[1], num_nodes=V, dtype=torch.float)
        x = (out_degree + in_degree).view(-1, 1)
    else:
        x = torch.ones((V, 1), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    data = data.to(device)
    
    # 构建 GNNExplainer，根据用例选择主模型或兜底单层模型
    target_model = gcn_1layer_model if use_fallback else gcn_model
    explainer = Explainer(
        model=target_model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='probs',
        ),
    )
    
    # batch = torch.zeros(V, dtype=torch.long).to(device)
    
    # 解释整个图对最终分类结果的贡献 (这里将 node_index 置为 None 解释图)
    try:
        explanation = explainer(x=data.x, edge_index=data.edge_index, batch=torch.zeros(V, dtype=torch.long).to(device))
        
        # 获取 edge_mask
        edge_mask = explanation.edge_mask.cpu().numpy()
        
        # 将带有权重的边映射回原 UUID, 保留权重排名前 10% 的高危边
        threshold = np.percentile(edge_mask, 90) if len(edge_mask) > 0 else 0.0

        abnormal_edges = []
        for i in range(len(src_nodes)):
            importance = float(edge_mask[i])
            if importance >= threshold and importance > 0:
                abnormal_edges.append({
                    "fromId": idx_to_uuid[src_nodes[i]],
                    "toId": idx_to_uuid[dst_nodes[i]],
                    "importance": importance
                })
        
        # 按重要性排序
        abnormal_edges.sort(key=lambda x: x["importance"], reverse=True)
        return abnormal_edges
    except Exception as e:
        logger.error(f"GNNExplainer 异常: {e}")
        return []

# ----------------- RabbitMQ 消费者 -----------------
def start_xai_worker():
    load_models_and_init_lime()
    _warmup_models()
    
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST, 
        port=RABBITMQ_PORT, 
        credentials=credentials,
        heartbeat=0,  # 禁用心跳以防止 GNN 长时间运算导致断线
        blocked_connection_timeout=0
    )
    
    while True:
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.exchange_declare(exchange='anomaly.topic', exchange_type='topic', durable=True)
            channel.queue_declare(queue='q.xai.req', durable=True)
            channel.queue_declare(queue='q.xai.res', durable=True)
            
            channel.basic_qos(prefetch_count=5)
            logger.info("[*] XAI Worker 就绪，正在监听 `q.xai.req` 队列...")
            
            for method_frame, properties, body in channel.consume('q.xai.req'):
                try:
                    msg = json.loads(body.decode())
                    trace_id = msg.get("traceId", "")
                    logger.info(f"收到 XAI 分析请求: TraceId={trace_id}")
                    
                    db_record = query_features_from_db(trace_id)
                    if not db_record:
                        logger.warning(f"TraceId={trace_id} 的特征数据不存在，跳过分析")
                        channel.basic_ack(method_frame.delivery_tag)
                        continue
                        
                    features_json = db_record.get('features_json')
                    call_graph_json = db_record.get('call_graph_json')
                    
                    features_dict = json.loads(features_json) if features_json else {}
                    call_graph = json.loads(call_graph_json) if call_graph_json else []
                    
                    has_call_graph = len(call_graph) > 0
                    use_topology = has_call_graph
                    
                    lime_weights = analyze_lime(features_dict, use_topology=use_topology)
                    
                    is_fallback_gcn = (len(features_dict) == 0)
                    abnormal_edges = analyze_gcn(call_graph, use_fallback=is_fallback_gcn)
                    
                    xai_result = {
                        "traceId": trace_id,
                        "limeWeights": json.dumps(lime_weights),
                        "abnormalEdges": json.dumps(abnormal_edges)
                    }
                    
                    channel.basic_publish(
                        exchange='anomaly.topic',
                        routing_key='xai.res',
                        body=json.dumps(xai_result)
                    )
                    
                    logger.info(f"XAI 分析完成并回传 MQ，TraceId={trace_id}")
                    channel.basic_ack(method_frame.delivery_tag)
                except Exception as inner_e:
                    logger.error(f"分析消息时发生内部异常: {inner_e}")
                    try:
                        trace_id = json.loads(body.decode()).get("traceId", "unknown")
                        err_res = {"traceId": trace_id, "error": str(inner_e)}
                        channel.basic_publish(exchange='anomaly.topic', routing_key='xai.res', body=json.dumps(err_res))
                        channel.basic_ack(method_frame.delivery_tag)
                    except:
                        pass
                
        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ 连接断开，将在5秒后重试...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"XAI Worker 发生严重异常: {e}")
            time.sleep(5)

if __name__ == "__main__":
    start_xai_worker()
