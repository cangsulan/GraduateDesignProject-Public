# -*- coding: utf-8 -*-
"""
数据结构定义模块

定义流量处理过程中使用的所有核心数据结构，包括：
- HTTPRequest: HTTP请求数据结构
- PcapSession: pcap会话数据结构
- FeatureRecord: 特征记录数据结构（对应CSV一行）
- GraphEdge: 调用图边
- CallGraph: 调用图数据结构
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import uuid


@dataclass
class HTTPRequest:
    """
    HTTP请求数据结构
    
    从pcap文件的TCP流中提取的HTTP请求信息
    
    Attributes:
        method: HTTP方法 (GET, POST, PUT, DELETE, PATCH)
        path: API路径 (如 /identity/api/auth/login)
        host: 主机名 (如 localhost:8888)
        timestamp: 时间戳，Unix时间戳（秒）
        user_agent: User-Agent头部
        content_type: Content-Type头部
        body: 请求体内容
    """
    method: str
    path: str
    host: str
    timestamp: float
    user_agent: Optional[str] = None
    content_type: Optional[str] = None
    body: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'method': self.method,
            'path': self.path,
            'host': self.host,
            'timestamp': self.timestamp,
            'user_agent': self.user_agent,
            'content_type': self.content_type,
            'body': self.body
        }


@dataclass
class PcapSession:
    """
    pcap会话数据结构
    
    从单个pcap文件中提取的完整TCP会话信息
    
    Attributes:
        file_name: pcap文件名
        source_ip: 源IP地址
        dest_ip: 目标IP地址
        source_port: 源端口
        dest_port: 目标端口
        start_time: 会话开始时间（Unix时间戳）
        end_time: 会话结束时间（Unix时间戳）
        packet_count: 数据包总数
        http_requests: HTTP请求列表
    """
    file_name: str
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    start_time: float
    end_time: float
    packet_count: int
    http_requests: List[HTTPRequest] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """会话持续时间（秒）"""
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0
    
    @property
    def duration_minutes(self) -> float:
        """会话持续时间（分钟）"""
        return self.duration_seconds / 60.0


@dataclass
class FeatureRecord:
    """
    特征记录数据结构
    
    对应CSV文件中的一行，包含模型所需的所有特征字段
    
    Attributes:
        _id: 唯一标识符（UUID格式）
        inter_api_access_duration_sec: API访问间隔时间的平均值（秒）
        api_access_uniqueness: API访问唯一性比例（0-1）
        sequence_length_count: API调用序列长度
        vsession_duration_min: 虚拟会话持续时间（分钟）
        ip_type: IP类型 (default/datacenter/private_ip)
        num_sessions: 会话数量
        num_users: 用户数量
        num_unique_apis: 唯一API数量
        source: 数据来源标识
        classification: 分类标签（可选，normal/outlier）
        source_ip: 源IP地址（用于溯源）
        # 创新点一：拓扑感知特征
        graph_density: 图密度
        max_in_degree: 最大入度
        avg_clustering: 平均聚类系数
    """
    _id: str
    inter_api_access_duration_sec: float
    api_access_uniqueness: float
    sequence_length_count: float
    vsession_duration_min: float
    ip_type: str
    num_sessions: float
    num_users: float
    num_unique_apis: float
    source: str = 'E'
    classification: Optional[str] = None
    source_ip: str = ''
    # 创新点一：拓扑感知特征
    graph_density: float = 0.0
    max_in_degree: float = 0.0
    avg_clustering: float = 0.0
    
    @classmethod
    def create_empty(cls, _id: str = None) -> 'FeatureRecord':
        """
        创建空的特征记录
        
        当pcap文件中没有HTTP请求时使用
        
        Args:
            _id: 唯一标识符，如果不提供则自动生成
            
        Returns:
            空的FeatureRecord实例
        """
        return cls(
            _id=_id or str(uuid.uuid4()),
            inter_api_access_duration_sec=0.0,
            api_access_uniqueness=0.0,
            sequence_length_count=0.0,
            vsession_duration_min=0.0,
            ip_type='default',
            num_sessions=1.0,
            num_users=1.0,
            num_unique_apis=0.0,
            source='E',
            graph_density=0.0,
            max_in_degree=0.0,
            avg_clustering=0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        字段名与原始数据集CSV格式保持一致
        """
        return {
            '_id': self._id,
            'inter_api_access_duration(sec)': self.inter_api_access_duration_sec,
            'api_access_uniqueness': self.api_access_uniqueness,
            'sequence_length(count)': self.sequence_length_count,
            'vsession_duration(min)': self.vsession_duration_min,
            'ip_type': self.ip_type,
            'num_sessions': self.num_sessions,
            'num_users': self.num_users,
            'num_unique_apis': self.num_unique_apis,
            'source': self.source,
            'graph_density': self.graph_density,
            'max_in_degree': self.max_in_degree,
            'avg_clustering': self.avg_clustering
        }
    
    def to_features_dict(self) -> Dict[str, Any]:
        """
        转换为模型推理使用的特征字典（包含拓扑特征）
        
        返回可直接用于infer_pipeline的特征格式
        """
        return {
            'inter_api_access_duration(sec)': self.inter_api_access_duration_sec,
            'api_access_uniqueness': self.api_access_uniqueness,
            'sequence_length(count)': self.sequence_length_count,
            'vsession_duration(min)': self.vsession_duration_min,
            'ip_type': self.ip_type,
            'num_sessions': self.num_sessions,
            'num_users': self.num_users,
            'num_unique_apis': self.num_unique_apis,
            'source': self.source,
            'graph_density': self.graph_density,
            'max_in_degree': self.max_in_degree,
            'avg_clustering': self.avg_clustering
        }


@dataclass
class GraphEdge:
    """
    调用图边数据结构
    
    表示API调用拓扑图中的一条有向边
    
    Attributes:
        fromId: 调用方节点（直接使用API路径字符串，便于阅读）
        toId: 被调用方节点（直接使用API路径字符串）
    """
    fromId: str
    toId: str
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {
            'fromId': self.fromId,
            'toId': self.toId
        }


@dataclass
class CallGraph:
    """
    调用图数据结构
    
    对应JSON文件中的一个元素，包含会话ID和调用图边列表
    
    Attributes:
        _id: 与FeatureRecord._id对应的唯一标识符
        call_graph: 调用图边列表
    """
    _id: str
    call_graph: List[GraphEdge] = field(default_factory=list)
    
    @classmethod
    def create_empty(cls, _id: str) -> 'CallGraph':
        """
        创建空的调用图
        
        当pcap文件中没有足够的HTTP请求时使用
        
        Args:
            _id: 唯一标识符
            
        Returns:
            空的CallGraph实例
        """
        return cls(_id=_id, call_graph=[])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            '_id': self._id,
            'call_graph': [edge.to_dict() for edge in self.call_graph]
        }
    
    def to_graph_list(self) -> List[Dict[str, str]]:
        """
        转换为模型推理使用的图边列表格式
        
        Returns:
            可直接用于infer_pipeline的callGraph格式
        """
        return [edge.to_dict() for edge in self.call_graph]


class TrafficProcessorError(Exception):
    """流量处理基础异常"""
    pass


class PcapParseError(TrafficProcessorError):
    """pcap文件解析异常"""
    pass


class HTTPExtractionError(TrafficProcessorError):
    """HTTP请求提取异常"""
    pass


class FeatureCalculationError(TrafficProcessorError):
    """特征计算异常"""
    pass
