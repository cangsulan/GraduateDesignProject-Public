# -*- coding: utf-8 -*-
"""
统一处理入口模块

提供pcap文件处理的统一入口，整合所有处理步骤：
1. pcap文件解析
2. HTTP请求提取
3. 特征计算
4. 调用图构建
"""

import os
import uuid
from typing import List, Tuple, Optional
from loguru import logger
from scapy.all import rdpcap

from .models import (
    PcapSession, FeatureRecord, CallGraph, GraphEdge,
    TrafficProcessorError, PcapParseError
)
from .pcap_parser import PcapParser
from .http_extractor import HTTPExtractor
from .feature_calculator import FeatureCalculator
from .call_graph_builder import CallGraphBuilder


def process_pcap_file(
    file_path: str,
    record_id: str = None
) -> Tuple[FeatureRecord, CallGraph]:
    """
    处理单个pcap文件
    
    完整的处理流程：
    1. 解析pcap文件，提取会话信息
    2. 提取HTTP请求
    3. 计算特征
    4. 构建调用图
    
    Args:
        file_path: pcap文件路径
        record_id: 记录ID，如果不提供则自动生成
        
    Returns:
        (FeatureRecord, CallGraph) 元组
        
    Raises:
        TrafficProcessorError: 处理失败
    """
    try:
        logger.info(f"开始处理pcap文件: {file_path}")
        
        # 1. 解析pcap文件
        session = PcapParser.parse_file(file_path)
        
        # 2. 读取数据包用于HTTP提取
        packets = rdpcap(file_path)
        
        # 3. 提取HTTP请求
        http_requests = HTTPExtractor.extract(session, packets)
        
        # 4. 构建调用图（先构建，用于拓扑特征提取）
        call_graph = CallGraphBuilder.build(
            http_requests, 
            record_id or str(uuid.uuid4())
        )
        
        # 5. 计算特征（传入调用图用于创新点一：拓扑特征提取）
        feature_record = FeatureCalculator.calculate(
            session, 
            http_requests, 
            record_id,
            call_graph=[edge.to_dict() for edge in call_graph.call_graph]
        )
        
        # 更新调用图的_id以匹配特征记录
        call_graph._id = feature_record._id
        
        logger.info(f"pcap文件处理完成: {file_path}, "
                   f"特征记录ID: {feature_record._id}, "
                   f"HTTP请求数: {len(http_requests)}, "
                   f"调用图边数: {len(call_graph.call_graph)}")
        
        return feature_record, call_graph
        
    except PcapParseError as e:
        logger.error(f"pcap解析失败: {e}")
        raise TrafficProcessorError(f"pcap解析失败: {e}")
    except Exception as e:
        logger.error(f"处理pcap文件失败: {file_path}, 错误: {e}")
        raise TrafficProcessorError(f"处理pcap文件失败: {e}")


def process_pcap_bytes(
    pcap_bytes: bytes,
    file_name: str = 'unknown.pcap',
    record_id: str = None
) -> Tuple[List[FeatureRecord], List[CallGraph]]:
    """
    处理pcap二进制数据
    
    Args:
        pcap_bytes: pcap文件的二进制内容
        file_name: 文件名（用于标识）
        record_id: 记录ID
        
    Returns:
        (FeatureRecord列表, CallGraph列表) 元组
        注意：当前实现每个pcap对应一个会话，所以列表长度为1
        
    Raises:
        TrafficProcessorError: 处理失败
    """
    import tempfile
    
    try:
        logger.info(f"开始处理pcap二进制数据: {file_name}")
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
            tmp.write(pcap_bytes)
            tmp_path = tmp.name
        
        try:
            # 处理临时文件
            feature_record, call_graph = process_pcap_file(tmp_path, record_id)
            
            # source_ip 已经在 process_pcap_file 中从pcap文件解析得到，不需要覆盖
            
            return [feature_record], [call_graph]
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except TrafficProcessorError:
        raise
    except Exception as e:
        logger.error(f"处理pcap二进制数据失败: {e}")
        raise TrafficProcessorError(f"处理pcap二进制数据失败: {e}")


def process_pcap_batch(
    file_paths: List[str],
    labels: Optional[List[str]] = None
) -> Tuple[List[FeatureRecord], List[CallGraph]]:
    """
    批量处理多个pcap文件
    
    Args:
        file_paths: pcap文件路径列表
        labels: 可选的分类标签列表（normal/outlier）
        
    Returns:
        (FeatureRecord列表, CallGraph列表) 元组
        
    Raises:
        TrafficProcessorError: 处理失败
    """
    feature_records = []
    call_graphs = []
    
    for i, file_path in enumerate(file_paths):
        try:
            feature_record, call_graph = process_pcap_file(file_path)
            
            # 如果提供了标签，设置分类标签
            if labels and i < len(labels):
                feature_record.classification = labels[i]
            
            feature_records.append(feature_record)
            call_graphs.append(call_graph)
            
        except TrafficProcessorError as e:
            logger.warning(f"跳过文件 {file_path}: {e}")
            continue
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时发生未知错误: {e}")
            continue
    
    logger.info(f"批量处理完成: 成功 {len(feature_records)}/{len(file_paths)} 个文件")
    return feature_records, call_graphs


def process_pcap_directory(
    directory: str,
    label: Optional[str] = None
) -> Tuple[List[FeatureRecord], List[CallGraph]]:
    """
    处理目录下的所有pcap文件
    
    Args:
        directory: 目录路径
        label: 可选的分类标签（应用于所有文件）
        
    Returns:
        (FeatureRecord列表, CallGraph列表) 元组
    """
    # 查找所有pcap文件
    pcap_files = []
    for f in os.listdir(directory):
        if f.endswith('.pcap'):
            pcap_files.append(os.path.join(directory, f))
    
    if not pcap_files:
        logger.warning(f"目录 {directory} 中没有找到pcap文件")
        return [], []
    
    # 构建标签列表
    labels = [label] * len(pcap_files) if label else None
    
    logger.info(f"在目录 {directory} 中找到 {len(pcap_files)} 个pcap文件")
    
    return process_pcap_batch(pcap_files, labels)


def export_to_csv_format(feature_records: List[FeatureRecord]) -> List[dict]:
    """
    将特征记录列表导出为CSV格式（字典列表）
    
    Args:
        feature_records: FeatureRecord列表
        
    Returns:
        字典列表，可直接用于pandas DataFrame或JSON序列化
    """
    return [record.to_dict() for record in feature_records]


def export_to_json_format(call_graphs: List[CallGraph]) -> List[dict]:
    """
    将调用图列表导出为JSON格式
    
    Args:
        call_graphs: CallGraph列表
        
    Returns:
        字典列表，可直接JSON序列化
    """
    return [graph.to_dict() for graph in call_graphs]
