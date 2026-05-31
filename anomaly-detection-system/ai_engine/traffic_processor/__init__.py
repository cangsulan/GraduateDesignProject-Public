# -*- coding: utf-8 -*-
"""
流量处理模块 - 从pcap原始流量文件提取特征数据

该模块实现了从pcap网络流量捕获文件到系统可用特征数据的完整转换流程，
支持双模态检测系统（Random Forest + GCN）的数据格式要求。
"""

from .models import (
    HTTPRequest, 
    PcapSession, 
    FeatureRecord, 
    GraphEdge, 
    CallGraph,
    TrafficProcessorError,
    PcapParseError,
    HTTPExtractionError,
    FeatureCalculationError
)
from .pcap_parser import PcapParser
from .http_extractor import HTTPExtractor
from .feature_calculator import FeatureCalculator
from .call_graph_builder import CallGraphBuilder
from .processor import (
    process_pcap_file, 
    process_pcap_bytes, 
    process_pcap_batch,
    process_pcap_directory,
    export_to_csv_format,
    export_to_json_format
)

__all__ = [
    # 数据结构
    'HTTPRequest',
    'PcapSession', 
    'FeatureRecord',
    'GraphEdge',
    'CallGraph',
    # 异常类
    'TrafficProcessorError',
    'PcapParseError',
    'HTTPExtractionError',
    'FeatureCalculationError',
    # 处理器类
    'PcapParser',
    'HTTPExtractor',
    'FeatureCalculator',
    'CallGraphBuilder',
    # 便捷函数
    'process_pcap_file',
    'process_pcap_bytes',
    'process_pcap_batch',
    'process_pcap_directory',
    'export_to_csv_format',
    'export_to_json_format'
]
