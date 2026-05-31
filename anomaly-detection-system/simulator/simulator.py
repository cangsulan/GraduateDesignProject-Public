import asyncio
import os
import json
import time
import logging
import aiohttp
import pandas as pd
import pika
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Simulator")

# ----------------- 全局配置 -----------------
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "admin123")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

DATA_DIR = os.path.abspath(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), 'dataset')))

class TrafficSimulator:
    def __init__(self):
        self.qps = 10
        self.is_running = False
        self.features_data = None
        self.call_graphs = {}
        self.current_index = 0
        
        # 加载数据
        self.load_data()

    def load_data(self):
        logger.info("正在加载 Remaining 数据集供模拟器使用...")
        try:
            csv_path = os.path.join(DATA_DIR, 'demo_simulator.csv')
            json_path = os.path.join(DATA_DIR, 'demo_simulator_graphs.json')
            
            if not os.path.exists(csv_path) or not os.path.exists(json_path):
                logger.error("找不到数据集文件！模拟器将无法发送真实数据。")
                return

            self.features_data = pd.read_csv(csv_path)
            
            with open(json_path, 'r') as f:
                graphs = json.load(f)
                for item in graphs:
                    self.call_graphs[item['_id']] = item.get('call_graph', [])
                    
            logger.info(f"数据加载完成: {len(self.features_data)} 条特征, {len(self.call_graphs)} 个调用图")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")

    async def _send_request(self, session: aiohttp.ClientSession, index: int):
        if self.features_data is None or index >= len(self.features_data):
            self.is_running = False
            logger.info("数据集已读完，自动停止模拟！")
            return

        row = self.features_data.iloc[index]
        trace_id = row['_id']
        
        # 构造 Payload
        payload = {
            "traceId": trace_id,
            "sourceIp": f"192.168.1.{index % 254 + 1}",  # 模拟一些随机 IP
            "timestamp": int(time.time() * 1000),
            "features": row.drop(['_id', 'behavior', 'behavior_type', 'classification'], errors='ignore').to_dict(),
            "callGraph": self.call_graphs.get(trace_id, [])
        }

        try:
            async with session.post(f"{BACKEND_URL}/api/simulator/receive", json=payload, timeout=5) as response:
                if response.status != 200:
                    logger.warning(f"请求失败，状态码: {response.status}")
        except Exception as e:
            logger.error(f"发送请求异常: {e}")

    async def traffic_loop(self):
        logger.info("进入流量发送主循环...")
        async with aiohttp.ClientSession() as session:
            while True:
                if self.is_running and self.features_data is not None:
                    tasks = []
                    # 每秒发送 QPS 个请求
                    for _ in range(self.qps):
                        if self.current_index >= len(self.features_data):
                            break
                        tasks.append(self._send_request(session, self.current_index))
                        self.current_index += 1
                        
                    if tasks:
                        await asyncio.gather(*tasks)
                    
                    # 控制发送速率（粗略控制，大约1秒）
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(0.5)

    def process_control_message(self, body: bytes):
        try:
            msg = json.loads(body.decode())
            action = msg.get("action")
            qps = msg.get("qps")
            
            if action == "start":
                self.is_running = True
                if qps: self.qps = qps
                logger.info(f"收到启动指令! 当前 QPS={self.qps}")
            elif action == "stop":
                self.is_running = False
                logger.info("收到停止指令!")
            elif action == "update" and qps:
                self.qps = qps
                logger.info(f"收到更新指令! 新 QPS={self.qps}")
                
        except Exception as e:
            logger.error(f"处理控制指令异常 {body}: {e}")

async def listen_rabbitmq(simulator: TrafficSimulator):
    """在一个独立的通过 asyncio 调度的后台任务中以轮询方式消费消息 (如果使用 pika) 
    对于全异步环境建议使用 aio_pika, 这里用 pika+asyncio 轮询妥协实现
    """
    logger.info("准备连接 RabbitMQ 控制队列...")
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST, 
        port=RABBITMQ_PORT, 
        credentials=credentials,
        heartbeat=60
    )
    
    connection = None
    channel = None
    
    while True:
        try:
            if not connection or not connection.is_open:
                connection = pika.BlockingConnection(parameters)
                channel = connection.channel()
                channel.queue_declare(queue='q.simulator.control', durable=True)
                logger.info("[*] 已连接至 RabbitMQ q.simulator.control")

            # 非阻塞获取消息
            method_frame, header_frame, body = channel.basic_get(queue='q.simulator.control')
            if method_frame:
                simulator.process_control_message(body)
                channel.basic_ack(method_frame.delivery_tag)
            
            await asyncio.sleep(0.1) # 略作休息，防止占用满 CPU
            
        except Exception as e:
            logger.error(f"RabbitMQ 监听异常: {e}. 5秒后重试...")
            connection = None
            await asyncio.sleep(5)

async def main():
    logger.info("=== Anomaly Traffic Simulator 启动 ===")
    simulator = TrafficSimulator()
    
    # 并发执行 MQ 监听与流量打榜
    await asyncio.gather(
        listen_rabbitmq(simulator),
        simulator.traffic_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
