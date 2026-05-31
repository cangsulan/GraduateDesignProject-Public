# -*- coding: utf-8 -*-
"""
调用图构建模块

负责根据HTTP请求序列构建微服务调用拓扑图：
- 节点：每个API端点（直接使用API路径字符串作为标识）
- 边：相邻请求之间的调用关系
"""

from typing import List, Optional
from loguru import logger

from .models import HTTPRequest, GraphEdge, CallGraph


class CallGraphBuilder:
    """
    调用图构建器
    
    根据HTTP请求序列构建有向调用图
    """
    
    @staticmethod
    def build(http_requests: List[HTTPRequest], record_id: str) -> CallGraph:
        """
        构建调用图
        
        Args:
            http_requests: HTTP请求列表（按时间戳排序）
            record_id: 记录ID
            
        Returns:
            CallGraph对象
        """
        # 如果请求数量不足，返回空图
        if len(http_requests) < 2:
            logger.debug(f"HTTP请求数量不足({len(http_requests)})，返回空调用图")
            return CallGraph.create_empty(record_id)
        
        # 构建边列表
        edges = []
        for i in range(len(http_requests) - 1):
            from_path = http_requests[i].path
            to_path = http_requests[i + 1].path
            
            # 创建边（直接使用API路径作为节点标识）
            edge = GraphEdge(fromId=from_path, toId=to_path)
            edges.append(edge)
        
        # 创建调用图
        call_graph = CallGraph(_id=record_id, call_graph=edges)
        
        logger.debug(f"构建调用图完成: {len(edges)} 条边")
        return call_graph
    
    @staticmethod
    def build_with_dedup(http_requests: List[HTTPRequest], record_id: str) -> CallGraph:
        """
        构建调用图（去重版本）
        
        移除重复的边，只保留唯一的调用关系
        
        Args:
            http_requests: HTTP请求列表
            record_id: 记录ID
            
        Returns:
            CallGraph对象
        """
        if len(http_requests) < 2:
            return CallGraph.create_empty(record_id)
        
        # 使用集合去重
        seen_edges = set()
        edges = []
        
        for i in range(len(http_requests) - 1):
            from_path = http_requests[i].path
            to_path = http_requests[i + 1].path
            
            # 创建边的唯一标识
            edge_key = (from_path, to_path)
            
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(GraphEdge(fromId=from_path, toId=to_path))
        
        return CallGraph(_id=record_id, call_graph=edges)
    
    @staticmethod
    def extract_services(http_requests: List[HTTPRequest]) -> dict:
        """
        从HTTP请求中提取微服务信息
        
        Args:
            http_requests: HTTP请求列表
            
        Returns:
            字典: {服务名: [API路径列表]}
        """
        services = {}
        
        for req in http_requests:
            path = req.path
            
            # 从路径中提取服务名
            service = CallGraphBuilder._extract_service_name(path)
            
            if service not in services:
                services[service] = []
            
            if path not in services[service]:
                services[service].append(path)
        
        return services
    
    @staticmethod
    def _extract_service_name(path: str) -> str:
        """
        从API路径中提取服务名
        
        规则：取路径第二段（如果第一段为空则取第三段）
        
        示例:
        - /identity/api/auth/login -> identity
        - /workshop/api/shop/products -> workshop
        - /users/v1/register -> users
        
        Args:
            path: API路径
            
        Returns:
            服务名
        """
        if not path:
            return 'unknown'
        
        # 分割路径
        parts = path.split('/')
        
        # 过滤空字符串
        parts = [p for p in parts if p]
        
        if not parts:
            return 'unknown'
        
        # 返回第一段作为服务名
        return parts[0]
    
    @staticmethod
    def get_graph_statistics(call_graph: CallGraph) -> dict:
        """
        获取调用图统计信息
        
        Args:
            call_graph: CallGraph对象
            
        Returns:
            统计信息字典
        """
        if not call_graph.call_graph:
            return {
                'edge_count': 0,
                'node_count': 0,
                'unique_from': 0,
                'unique_to': 0
            }
        
        from_nodes = set()
        to_nodes = set()
        
        for edge in call_graph.call_graph:
            from_nodes.add(edge.fromId)
            to_nodes.add(edge.toId)
        
        all_nodes = from_nodes | to_nodes
        
        return {
            'edge_count': len(call_graph.call_graph),
            'node_count': len(all_nodes),
            'unique_from': len(from_nodes),
            'unique_to': len(to_nodes)
        }
