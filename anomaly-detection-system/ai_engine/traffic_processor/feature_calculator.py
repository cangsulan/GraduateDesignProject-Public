# -*- coding: utf-8 -*-
"""
特征计算模块

负责根据HTTP请求列表计算模型所需的特征字段：
- inter_api_access_duration(sec): API访问间隔时间的平均值
- api_access_uniqueness: API访问唯一性比例
- sequence_length(count): API调用序列长度
- vsession_duration(min): 会话持续时间
- num_sessions: 会话数量
- num_users: 用户数量
- num_unique_apis: 唯一API数量
- ip_type: IP类型
- source: 数据来源

创新点一：拓扑感知特征
- graph_density: 图密度
- max_in_degree: 最大入度
- avg_clustering: 平均聚类系数
"""

from typing import List, Optional
from loguru import logger

from .models import (
    HTTPRequest, PcapSession, FeatureRecord, 
    FeatureCalculationError
)

# 导入拓扑特征提取模块（创新点一）
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topology_features import extract_topology_features


class FeatureCalculator:
    """
    特征计算器
    
    根据pcap会话和HTTP请求列表计算特征字段
    """
    
    @staticmethod
    def calculate(
        session: PcapSession,
        http_requests: List[HTTPRequest],
        record_id: str = None,
        call_graph: List = None
    ) -> FeatureRecord:
        """
        计算特征记录
        
        Args:
            session: PcapSession对象
            http_requests: HTTP请求列表
            record_id: 记录ID，如果不提供则自动生成
            call_graph: 调用图边列表（用于创新点一：拓扑特征提取）
            
        Returns:
            FeatureRecord对象
            
        Raises:
            FeatureCalculationError: 计算失败
        """
        try:
            import uuid
            
            # 生成记录ID
            _id = record_id or str(uuid.uuid4())
            
            # 如果没有HTTP请求，返回空记录
            if not http_requests:
                logger.warning(f"会话 {session.file_name} 没有HTTP请求，返回空特征记录")
                record = FeatureRecord.create_empty(_id)
                record.source_ip = session.source_ip
                record.ip_type = FeatureCalculator._infer_ip_type(session.source_ip)
                return record
            
            # 计算各特征字段
            inter_api_duration = FeatureCalculator._calc_inter_api_duration(http_requests)
            api_uniqueness = FeatureCalculator._calc_api_uniqueness(http_requests)
            sequence_length = FeatureCalculator._calc_sequence_length(http_requests)
            session_duration = FeatureCalculator._calc_session_duration(session)
            num_unique_apis = FeatureCalculator._calc_num_unique_apis(http_requests)
            ip_type = FeatureCalculator._infer_ip_type(session.source_ip)
            
            # === 创新点一：计算拓扑特征 ===
            topo_features = {'graph_density': 0.0, 'max_in_degree': 0.0, 'avg_clustering': 0.0}
            if call_graph and len(call_graph) > 0:
                topo_features = extract_topology_features(call_graph)
            
            # 创建特征记录
            record = FeatureRecord(
                _id=_id,
                inter_api_access_duration_sec=inter_api_duration,
                api_access_uniqueness=api_uniqueness,
                sequence_length_count=sequence_length,
                vsession_duration_min=session_duration,
                ip_type=ip_type,
                num_sessions=1.0,  # 每个pcap文件是一个会话
                num_users=1.0,     # 每个pcap文件是一个用户会话
                num_unique_apis=num_unique_apis,
                source='E',
                source_ip=session.source_ip,
                # 创新点一：拓扑特征
                graph_density=topo_features['graph_density'],
                max_in_degree=topo_features['max_in_degree'],
                avg_clustering=topo_features['avg_clustering']
            )
            
            return record
            
        except Exception as e:
            logger.error(f"计算特征失败: {e}")
            raise FeatureCalculationError(f"计算特征失败: {e}")
    
    @staticmethod
    def _calc_inter_api_duration(http_requests: List[HTTPRequest]) -> float:
        """
        计算API访问间隔时间的平均值（秒）
        
        Args:
            http_requests: HTTP请求列表（已按时间戳排序）
            
        Returns:
            平均间隔时间（秒）
        """
        if len(http_requests) <= 1:
            return 0.0
        
        # 计算相邻请求的时间间隔
        intervals = []
        for i in range(1, len(http_requests)):
            interval = http_requests[i].timestamp - http_requests[i-1].timestamp
            if interval >= 0:  # 忽略负值（时间戳异常）
                intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        return sum(intervals) / len(intervals)
    
    @staticmethod
    def _calc_api_uniqueness(http_requests: List[HTTPRequest]) -> float:
        """
        计算API访问唯一性比例
        
        定义：唯一API路径数量 / 总API调用数量
        
        Args:
            http_requests: HTTP请求列表
            
        Returns:
            唯一性比例（0-1）
        """
        if not http_requests:
            return 0.0
        
        # 提取所有API路径
        paths = [req.path for req in http_requests]
        
        # 计算唯一路径数
        unique_count = len(set(paths))
        
        # 计算唯一性比例
        return unique_count / len(paths)
    
    @staticmethod
    def _calc_sequence_length(http_requests: List[HTTPRequest]) -> float:
        """
        计算API调用序列长度
        
        Args:
            http_requests: HTTP请求列表
            
        Returns:
            序列长度
        """
        return float(len(http_requests))
    
    @staticmethod
    def _calc_session_duration(session: PcapSession) -> float:
        """
        计算会话持续时间（分钟）
        
        Args:
            session: PcapSession对象
            
        Returns:
            持续时间（分钟）
        """
        return session.duration_minutes
    
    @staticmethod
    def _calc_num_unique_apis(http_requests: List[HTTPRequest]) -> float:
        """
        计算唯一API数量
        
        Args:
            http_requests: HTTP请求列表
            
        Returns:
            唯一API数量
        """
        if not http_requests:
            return 0.0
        
        paths = set(req.path for req in http_requests)
        return float(len(paths))
    
    @staticmethod
    def _infer_ip_type(source_ip: str) -> str:
        """
        根据IP地址推断IP类型
        
        Args:
            source_ip: 源IP地址字符串
            
        Returns:
            IP类型: 'private_ip' 或 'default'
        """
        if not source_ip or source_ip == 'unknown':
            return 'default'
        
        # 本地回环地址 127.0.0.0/8
        if source_ip.startswith('127.'):
            return 'private_ip'
        
        # C类私有地址 192.168.0.0/16
        if source_ip.startswith('192.168.'):
            return 'private_ip'
        
        # A类私有地址 10.0.0.0/8
        if source_ip.startswith('10.'):
            return 'private_ip'
        
        # B类私有地址 172.16.0.0/12 (172.16.x.x - 172.31.x.x)
        if source_ip.startswith('172.'):
            parts = source_ip.split('.')
            if len(parts) >= 2:
                try:
                    second_octet = int(parts[1])
                    if 16 <= second_octet <= 31:
                        return 'private_ip'
                except ValueError:
                    pass
        
        # 公网地址
        return 'default'
