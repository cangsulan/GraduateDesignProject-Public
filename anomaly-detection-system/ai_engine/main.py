import os
import json
import joblib
import torch
import pika
import asyncio
import numpy as np
import pandas as pd
import requests
import io
import time
from loguru import logger
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager
from torch_geometric.data import Data
from torch_geometric.utils import degree
from typing import List, Dict, Any, Optional

from train_models import GCN, GCN1Layer, NUMERIC_FEATURES_BASE, CATEGORICAL_FEATURES, MODEL_DIR, main as train_models_main

from confidence_fusion import (
    confidence_adaptive_fusion,
    fixed_weight_fusion,
    OPTIMAL_PARAMS
)

try:
    from traffic_processor import (
        process_pcap_bytes,
        TrafficProcessorError,
        PcapParseError
    )
    TRAFFIC_PROCESSOR_AVAILABLE = True
    logger.info("流量处理模块加载成功")
except ImportError as e:
    TRAFFIC_PROCESSOR_AVAILABLE = False
    logger.warning(f"流量处理模块加载失败: {e}，pcap处理功能不可用")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "admin123")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rf_model = None
gcn_model = None
gcn_1layer_model = None
scaler = None
encoders = None

def load_models():
    """生命周期启动时加载模型与预处理器"""
    global rf_model, gcn_model, gcn_1layer_model, scaler, encoders
    try:
        logger.info("正在加载 RF 模型与预处理器...")
        rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        encoders = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
        logger.info("RF 模型加载成功")
        
        logger.info(f"正在加载 GCN 模型... 设备: {device}")
        
        gcn_model = GCN(hidden_channels=64).to(device)
        gcn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_model.pth"), map_location=device))
        gcn_model.eval()
        logger.info("GCN 双层模型加载成功 (用于有特征时的双模态融合)")

        gcn_1layer_model = GCN1Layer(hidden_channels=64).to(device)
        gcn_1layer_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_1layer_model.pth"), map_location=device))
        gcn_1layer_model.eval()
        logger.info("GCN 单层模型加载成功 (用于无特征时的降级模式)")
        
        logger.info("所有模型加载完成: RF + GCN双层 + GCN单层")
        logger.info(f"融合策略: 置信度自适应融合 (F1=0.98, FPR=2.17%, FNR=0%)")
        logger.info(f"最优参数: alpha={OPTIMAL_PARAMS['alpha']}, conf_threshold={OPTIMAL_PARAMS['conf_threshold']}")
        
        _warmup_models()
        
    except FileNotFoundError:
        logger.warning("模型文件缺失，尝试自动触发离线训练构建...")
        train_models_main()
        load_models()
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e


def _warmup_models():
    """模型预热 - 执行一次虚拟推理以初始化CUDA kernel"""
    logger.info("正在进行模型预热...")
    start_time = time.time()
    
    try:
        dummy_features = {col: 0.0 for col in NUMERIC_FEATURES_BASE}
        dummy_features['ip_type'] = 'default'
        dummy_features['source'] = 'E'
        
        df = pd.DataFrame([dummy_features])
        X_numeric = df[NUMERIC_FEATURES_BASE].values
        X_categorical = np.zeros((1, len(CATEGORICAL_FEATURES)))
        X = np.hstack((X_numeric, X_categorical))
        X_scaled = scaler.transform(X)
        _ = rf_model.predict_proba(X_scaled)
        
        dummy_graph = [
            {"fromId": "node_1", "toId": "node_2"},
            {"fromId": "node_2", "toId": "node_3"}
        ]
        
        unique_uuids = ["node_1", "node_2", "node_3"]
        uuid_to_idx = {uuid: idx for idx, uuid in enumerate(unique_uuids)}
        
        src_nodes = [uuid_to_idx[edge['fromId']] for edge in dummy_graph]
        dst_nodes = [uuid_to_idx[edge['toId']] for edge in dummy_graph]
        
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        V = len(unique_uuids)
        x = torch.ones((V, 1), dtype=torch.float)
        batch = torch.zeros(V, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        data = data.to(device)
        batch = batch.to(device)
        
        with torch.no_grad():
            _ = gcn_model(data.x, data.edge_index, batch)
            _ = gcn_1layer_model(data.x, data.edge_index, batch)
        
        elapsed = time.time() - start_time
        logger.info(f"模型预热完成，耗时: {elapsed:.2f}秒")
        
    except Exception as e:
        logger.warning(f"模型预热失败（不影响正常使用）: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    
    import threading
    mq_thread = threading.Thread(target=start_rabbitmq_consumer, daemon=True)
    mq_thread.start()
    
    yield

app = FastAPI(title="Anomaly Detection AI Engine", lifespan=lifespan)

def infer_pipeline(features: dict, call_graph: list, fusion_mode: str = 'adaptive') -> dict:
    """
    双模态融合推理核心逻辑
    
    创新点：置信度自适应融合策略
    - 根据模型置信度动态调整融合权重
    - 根据预测一致性动态调整判定阈值
    - 实验结果：F1=0.98, FPR=2.17%, FNR=0%
    
    兜底机制：
    - 有call_graph时：RF + GCN双模态融合
    - 无call_graph时：仅使用RF
    """
    has_call_graph = call_graph and len(call_graph) > 0
    
    rf_prob = 0.0
    
    if features:
        df = pd.DataFrame([features])
        
        for col in NUMERIC_FEATURES_BASE:
            if col not in df.columns: df[col] = 0.0
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns: df[col] = 'unknown'
        
        X_numeric = df[NUMERIC_FEATURES_BASE].values
        
        X_categorical = np.zeros((1, len(CATEGORICAL_FEATURES)))
        for i, col in enumerate(CATEGORICAL_FEATURES):
            val = df[col].iloc[0]
            try:
                X_categorical[0, i] = encoders[col].transform([val])[0]
            except ValueError:
                X_categorical[0, i] = 0 
                
        X = np.hstack((X_numeric, X_categorical))
        X_scaled = scaler.transform(X)
        
        probs = rf_model.predict_proba(X_scaled)
        classes = list(rf_model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            rf_prob = float(probs[0, idx])
        else:
            rf_prob = float(probs[0, 1])

    gcn_prob = 0.0
    if has_call_graph:
        unique_uuids = set()
        for edge in call_graph:
            unique_uuids.add(edge['fromId'])
            unique_uuids.add(edge['toId'])
            
        unique_uuids = list(unique_uuids)
        uuid_to_idx = {uuid: idx for idx, uuid in enumerate(unique_uuids)}
        
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
        
        batch = torch.zeros(V, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        data = data.to(device)
        batch = batch.to(device)
        
        target_gcn = gcn_model if features else gcn_1layer_model
        
        with torch.no_grad():
            out = target_gcn(data.x, data.edge_index, batch)
            gcn_prob = float(out.item())
    
    if not features and has_call_graph:
        final_prob = gcn_prob
        is_anomaly = 1 if final_prob > 0.5 else 0
        fusion_info = {'mode': 'gcn_only', 'w_rf': 0.0, 'w_gcn': 1.0, 'threshold': 0.5}
    elif features and not has_call_graph:
        final_prob = rf_prob
        is_anomaly = 1 if final_prob > 0.5 else 0
        fusion_info = {'mode': 'rf_only', 'w_rf': 1.0, 'w_gcn': 0.0, 'threshold': 0.5}
    else:
        if fusion_mode == 'adaptive':
            final_prob, threshold, is_anomaly, fusion_info = confidence_adaptive_fusion(
                rf_prob, gcn_prob
            )
        else:
            final_prob, is_anomaly, fusion_info = fixed_weight_fusion(rf_prob, gcn_prob)

    return {
        "rfProb": rf_prob,
        "gcnProb": gcn_prob,
        "finalProb": float(final_prob),
        "isAnomaly": is_anomaly,
        "fusionInfo": fusion_info
    }

mq_connection = None
mq_channel = None

def get_mq_channel():
    global mq_connection, mq_channel
    if mq_connection and mq_connection.is_open:
        return mq_channel
        
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST, 
        port=RABBITMQ_PORT, 
        credentials=credentials,
        heartbeat=60
    )
    mq_connection = pika.BlockingConnection(parameters)
    mq_channel = mq_connection.channel()
    
    mq_channel.exchange_declare(exchange='anomaly.topic', exchange_type='topic', durable=True)
    mq_channel.queue_declare(queue='q.detect.req', durable=True)
    mq_channel.queue_declare(queue='q.detect.res', durable=True)
    return mq_channel

def start_rabbitmq_consumer():
    """监听 Java 发往 AI 引擎的检测请求"""
    import time
    global mq_connection, mq_channel
    while True:
        try:
            channel = get_mq_channel()
            channel.basic_qos(prefetch_count=10)
            
            logger.info(f"[*] AI 发动机启动，正在监听 `q.detect.req` 队列...")
            
            for method_frame, properties, body in channel.consume('q.detect.req'):
                try:
                    msg = json.loads(body.decode())
                    trace_id = msg.get("traceId", "unknown")
                    features = msg.get("features", {})
                    call_graph = msg.get("callGraph", [])
                    
                    result = infer_pipeline(features, call_graph)
                    
                    result["traceId"] = trace_id
                    result["sourceIp"] = msg.get("sourceIp", "")
                    result["timestamp"] = msg.get("timestamp", 0)
                    
                    result["features"] = json.dumps(features)
                    result["callGraph"] = json.dumps(call_graph)

                    res_body = json.dumps(result)
                    channel.basic_publish(
                        exchange='anomaly.topic',
                        routing_key='detect.res',
                        body=res_body
                    )
                    
                    channel.basic_ack(method_frame.delivery_tag)
                    
                except Exception as e:
                    logger.error(f"处理检测请求失败: {e}")
                    channel.basic_nack(method_frame.delivery_tag, requeue=False)
                    
        except Exception as e:
            logger.error(f"RabbitMQ 消费者线程异常退出: {e}。将在5秒后重试重连...")
            time.sleep(5)
            if mq_connection and not mq_connection.is_closed:
                try: 
                    mq_connection.close()
                except Exception:
                    pass
            mq_connection = None
            mq_channel = None

class BatchDetectRequest(BaseModel):
    batchData: List[Dict[str, Any]]

@app.post("/api/batch_detect")
async def batch_detect(request: BatchDetectRequest):
    """供文件离线检测时 Java 批量调用的 HTTP 接口"""
    results = []
    
    for item in request.batchData:
        trace_id = item.get("traceId", "")
        features = item.get("features", {})
        call_graph = item.get("callGraph", [])
        
        try:
            res = infer_pipeline(features, call_graph)
            res["traceId"] = trace_id
            results.append(res)
            
        except Exception as e:
            logger.error(f"批量推理异常: {e}")
            
    return {"code": 200, "data": results}

class TaskRequest(BaseModel):
    taskId: int
    csvUrl: str = None
    jsonUrl: str = None
    callbackUrl: str


def send_http_callback(task_id: int, callback_payload: dict, callback_url: str):
    """
    通过HTTP发送回调消息
    
    特性：
    - 重试机制：失败时重试3次
    - 超时处理：60秒超时
    - 指数退避：重试间隔递增
    """
    max_retries = 3
    base_retry_delay = 2
    
    callback_payload["timestamp"] = int(time.time() * 1000)
    callback_payload["messageId"] = f"callback-{task_id}-{callback_payload['timestamp']}"
    
    for attempt in range(max_retries):
        try:
            retry_delay = base_retry_delay * (2 ** attempt)
            
            if attempt > 0:
                logger.info(f"[Task-{task_id}] 等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            
            response = requests.post(
                callback_url, 
                json=callback_payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                logger.info(f"[Task-{task_id}] HTTP回调成功 (attempt={attempt+1})")
                return True
            else:
                logger.warning(f"[Task-{task_id}] HTTP回调失败: status={response.status_code}, response={response.text[:200]}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"[Task-{task_id}] HTTP回调超时 (attempt={attempt+1}, timeout=60s)")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[Task-{task_id}] HTTP回调连接失败 (attempt={attempt+1}): {e}")
        except Exception as e:
            logger.warning(f"[Task-{task_id}] HTTP回调异常 (attempt={attempt+1}): {e}")
    
    logger.error(f"[Task-{task_id}] HTTP回调失败，已重试 {max_retries} 次")
    return False

def validate_features(features: dict) -> tuple:
    """
    验证特征数据是否有效
    
    返回: (is_valid, error_message)
    """
    required_numeric = NUMERIC_FEATURES_BASE
    
    for col in required_numeric:
        if col not in features:
            return False, f"缺失数值特征: {col}"
        
        val = features[col]
        if val is None:
            return False, f"特征 {col} 值为None"
        
        if isinstance(val, float):
            if np.isnan(val):
                return False, f"特征 {col} 值为NaN"
            if np.isinf(val):
                return False, f"特征 {col} 值为Inf"
    
    for col in CATEGORICAL_FEATURES:
        if col not in features:
            return False, f"缺失类别特征: {col}"
        
        val = features[col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False, f"类别特征 {col} 值无效"
    
    return True, None


def clean_features(features: dict) -> dict:
    """
    清洗特征数据，处理缺失值和异常值
    """
    cleaned = {}
    
    for col in NUMERIC_FEATURES_BASE:
        val = features.get(col, 0)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            cleaned[col] = 0.0
        else:
            cleaned[col] = float(val)
    
    for col in CATEGORICAL_FEATURES:
        val = features.get(col, 'unknown')
        if val is None or (isinstance(val, float) and np.isnan(val)):
            cleaned[col] = 'unknown'
        else:
            cleaned[col] = str(val)
    
    return cleaned


def process_offline_task(req: TaskRequest):
    logger.info(f"[Task-{req.taskId}] 启动后台文件检测流水线...")
    start_ts = time.time()
    
    try:
        df = None
        graph_data = None
        
        if req.csvUrl:
            logger.info(f"[Task-{req.taskId}] 下载提取 CSV特征资源...")
            res = requests.get(req.csvUrl, timeout=60)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                logger.info(f"[Task-{req.taskId}] CSV加载成功: {len(df)} 行, {len(df.columns)} 列")
                
        if req.jsonUrl:
            logger.info(f"[Task-{req.taskId}] 下载提取 JSON调用拓扑...")
            res = requests.get(req.jsonUrl, timeout=60)
            if res.status_code == 200:
                graph_data = res.json()
                logger.info(f"[Task-{req.taskId}] JSON加载成功: {len(graph_data)} 条图数据")
                
        if df is None and graph_data is None:
            raise ValueError("双路文件全部拉取失败或均未上传！")

        valid_count = 0
        anomaly_count = 0
        skipped_count = 0
        records = []
        skipped_records = []
        
        graph_dict = {}
        if graph_data:
            for item in graph_data:
                graph_dict[str(item.get('_id'))] = item.get('call_graph', [])
        
        csv_id_set = set()
        if df is not None and '_id' in df.columns:
            csv_id_set = set(df['_id'].astype(str).tolist())
        
        json_id_set = set(graph_dict.keys())
        
        if df is not None:
            common_ids = csv_id_set & json_id_set if json_id_set else csv_id_set
            master_keys = list(common_ids) if common_ids else df['_id'].astype(str).tolist()
        else:
            master_keys = list(json_id_set)
            
        total_count = len(master_keys)
        logger.info(f"[Task-{req.taskId}] 待处理记录数: {total_count}")
        
        for str_id in master_keys:
            features = {}
            if df is not None:
                row_mask = df['_id'].astype(str) == str_id
                if row_mask.any():
                    row_dict = df[row_mask].iloc[0].to_dict()
                    features = row_dict
            
            call_graph = graph_dict.get(str_id, [])
            
            is_valid, error_msg = validate_features(features)
            
            if not is_valid:
                skipped_count += 1
                skipped_records.append({
                    "id": str_id,
                    "reason": error_msg
                })
                if skipped_count <= 10:
                    logger.warning(f"[Task-{req.taskId}] 跳过无效数据 ID={str_id}: {error_msg}")
                continue
            
            try:
                cleaned_features = clean_features(features)
                
                res_prob = infer_pipeline(cleaned_features, call_graph)
                
                valid_count += 1
                
                if res_prob["isAnomaly"] == 1:
                    anomaly_count += 1

                record_entry = {
                    "traceId": f"file-{req.taskId}-{str_id}",
                    "sourceIp": str(features.get("source_ip", features.get("src_ip", features.get("sourceIp", "")))),
                    "rfProb": res_prob.get("rfProb"),
                    "gcnProb": res_prob.get("gcnProb"),
                    "finalProb": res_prob.get("finalProb"),
                    "isAnomaly": res_prob.get("isAnomaly", 0),
                    "features": json.dumps(features, ensure_ascii=False) if features else None,
                    "callGraph": json.dumps(call_graph, ensure_ascii=False) if call_graph else None,
                }
                for k, v in record_entry.items():
                    if hasattr(v, 'item'):
                        record_entry[k] = v.item()
                records.append(record_entry)
                
            except Exception as e:
                skipped_count += 1
                if skipped_count <= 10:
                    logger.warning(f"[Task-{req.taskId}] 处理ID={str_id}时发生异常: {e}")
                skipped_records.append({
                    "id": str_id,
                    "reason": str(e)
                })
                continue
                
        duration_ms = int((time.time() - start_ts) * 1000)
        logger.info(f"[Task-{req.taskId}] 分析结算 -> 耗时 {duration_ms}ms, 有效: {valid_count}, 跳过: {skipped_count}, 异常数: {anomaly_count}")
        
        if skipped_count > 10:
            logger.warning(f"[Task-{req.taskId}] 共跳过 {skipped_count} 条无效数据（仅显示前10条）")
        
        callback_payload = {
            "taskId": req.taskId,
            "status": "COMPLETED",
            "recordCount": valid_count,
            "anomalyCount": anomaly_count,
            "skippedCount": skipped_count,
            "duration": duration_ms,
            "records": records
        }
        
        if skipped_records:
            callback_payload["skippedRecords"] = skipped_records[:100]
        
    except Exception as e:
        logger.error(f"[Task-{req.taskId}] 检测崩溃终结: {e}")
        import traceback
        logger.error(traceback.format_exc())
        callback_payload = {
            "taskId": req.taskId,
            "status": "FAILED",
            "recordCount": 0,
            "anomalyCount": 0,
            "duration": int((time.time() - start_ts) * 1000),
            "errorMessage": str(e)
        }
        
    send_http_callback(req.taskId, callback_payload, req.callbackUrl)


@app.post("/api/predict/task")
async def create_offline_task(req: TaskRequest, background_tasks: BackgroundTasks):
    """供 Java 在创建大体积文件检测时发起的非阻塞异步任务"""
    background_tasks.add_task(process_offline_task, req)
    return {"code": 200, "message": "Task Accpeted"}


def process_pcap_task(
    task_id: int,
    pcap_files: List[tuple],
    callback_url: str
):
    """
    后台处理pcap文件任务
    
    Args:
        task_id: 任务ID
        pcap_files: pcap文件列表，每个元素为 (文件名, 二进制数据) 元组
        callback_url: 回调URL
    """
    logger.info(f"[PCAP-Task-{task_id}] 启动pcap流量检测流水线...")
    start_ts = time.time()
    
    if not TRAFFIC_PROCESSOR_AVAILABLE:
        logger.error(f"[PCAP-Task-{task_id}] 流量处理模块不可用")
        callback_payload = {
            "taskId": task_id,
            "status": "FAILED",
            "recordCount": 0,
            "anomalyCount": 0,
            "duration": int((time.time() - start_ts) * 1000),
            "errorMessage": "流量处理模块不可用"
        }
        send_http_callback(task_id, callback_payload, callback_url)
        return
    
    try:
        all_records = []
        anomaly_count = 0
        total_count = 0
        
        for file_name, pcap_bytes in pcap_files:
            try:
                logger.info(f"[PCAP-Task-{task_id}] 处理文件: {file_name}")
                
                feature_records, call_graphs = process_pcap_bytes(
                    pcap_bytes, 
                    file_name=file_name
                )
                
                for i, record in enumerate(feature_records):
                    total_count += 1
                    
                    call_graph = []
                    if i < len(call_graphs):
                        call_graph = call_graphs[i].to_graph_list()
                    
                    features = record.to_features_dict()
                    
                    result = infer_pipeline(features, call_graph)
                    
                    if result['isAnomaly'] == 1:
                        anomaly_count += 1
                    
                    record_entry = {
                        "traceId": f"pcap-{task_id}-{record._id}",
                        "sourceIp": record.source_ip,
                        "rfProb": result['rfProb'],
                        "gcnProb": result['gcnProb'],
                        "finalProb": result['finalProb'],
                        "isAnomaly": result['isAnomaly'],
                        "features": features,
                        "callGraph": call_graph
                    }
                    all_records.append(record_entry)
                    
            except TrafficProcessorError as e:
                logger.error(f"[PCAP-Task-{task_id}] 处理文件 {file_name} 失败: {e}")
                continue
            except Exception as e:
                logger.error(f"[PCAP-Task-{task_id}] 处理文件 {file_name} 时发生未知错误: {e}")
                continue
        
        duration_ms = int((time.time() - start_ts) * 1000)
        logger.info(f"[PCAP-Task-{task_id}] 检测完成 -> 耗时 {duration_ms}ms, "
                   f"总记录: {total_count}, 异常数: {anomaly_count}")
        
        callback_payload = {
            "taskId": task_id,
            "status": "COMPLETED",
            "recordCount": total_count,
            "anomalyCount": anomaly_count,
            "duration": duration_ms,
            "records": all_records
        }
        
    except Exception as e:
        logger.error(f"[PCAP-Task-{task_id}] 任务崩溃: {e}")
        callback_payload = {
            "taskId": task_id,
            "status": "FAILED",
            "recordCount": 0,
            "anomalyCount": 0,
            "duration": int((time.time() - start_ts) * 1000),
            "errorMessage": str(e)
        }
    
    send_http_callback(task_id, callback_payload, callback_url)


@app.post("/api/predict/from-pcap")
async def predict_from_pcap(
    taskId: int = Form(...),
    pcapFiles: List[UploadFile] = File(...),
    callbackUrl: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    从pcap文件进行检测（内存直通模式）
    
    该接口接收pcap文件，直接在内存中进行特征提取和检测，
    无需生成中间文件，实现低耦合集成。
    
    Args:
        taskId: 任务ID
        pcapFiles: pcap文件列表（支持多文件上传）
        callbackUrl: 回调URL
        
    Returns:
        任务接受确认
    """
    if not TRAFFIC_PROCESSOR_AVAILABLE:
        return {"code": 500, "message": "流量处理模块不可用，请检查scapy是否安装"}
    
    if not pcapFiles:
        return {"code": 400, "message": "未提供pcap文件"}
    
    logger.info(f"接收pcap检测任务: taskId={taskId}, 文件数={len(pcapFiles)}")
    
    pcap_data_list = []
    for pcap_file in pcapFiles:
        try:
            content = await pcap_file.read()
            pcap_data_list.append((pcap_file.filename, content))
        except Exception as e:
            logger.error(f"读取pcap文件失败: {pcap_file.filename}, 错误: {e}")
    
    if not pcap_data_list:
        return {"code": 400, "message": "所有pcap文件读取失败"}
    
    background_tasks.add_task(
        process_pcap_task,
        taskId,
        pcap_data_list,
        callbackUrl
    )
    
    return {"code": 200, "message": "PCAP检测任务已接受，正在后台处理"}


@app.get("/api/pcap/status")
async def get_pcap_processor_status():
    """获取pcap处理器状态"""
    return {
        "code": 200,
        "data": {
            "trafficProcessorAvailable": TRAFFIC_PROCESSOR_AVAILABLE,
            "message": "流量处理模块可用" if TRAFFIC_PROCESSOR_AVAILABLE else "流量处理模块不可用，请安装scapy"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
