# -*- coding: utf-8 -*-
"""
HTTP请求提取模块

从TCP流中提取HTTP请求信息，包括：
- HTTP方法（GET, POST, PUT, DELETE, PATCH）
- API路径
- Host头部
- 时间戳
- 其他HTTP头部信息
"""

import re
from typing import List, Tuple, Optional
from loguru import logger

from .models import HTTPRequest, PcapSession, HTTPExtractionError
from .pcap_parser import PcapParser


class HTTPExtractor:
    """
    HTTP请求提取器
    
    从pcap会话的原始数据包中提取HTTP请求信息
    """
    
    # HTTP方法正则表达式
    HTTP_METHOD_PATTERN = re.compile(
        r'^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|CONNECT|TRACE)\s+(\S+)\s+HTTP/[\d.]+',
        re.IGNORECASE
    )
    
    # Host头部正则表达式
    HOST_PATTERN = re.compile(r'^Host:\s*(.+)$', re.IGNORECASE | re.MULTILINE)
    
    # User-Agent头部正则表达式
    USER_AGENT_PATTERN = re.compile(r'^User-Agent:\s*(.+)$', re.IGNORECASE | re.MULTILINE)
    
    # Content-Type头部正则表达式
    CONTENT_TYPE_PATTERN = re.compile(r'^Content-Type:\s*(.+)$', re.IGNORECASE | re.MULTILINE)
    
    @staticmethod
    def extract(session: PcapSession, packets) -> List[HTTPRequest]:
        """
        从pcap会话中提取HTTP请求
        
        Args:
            session: PcapSession对象
            packets: scapy数据包列表
            
        Returns:
            HTTPRequest列表，按时间戳排序
            
        Raises:
            HTTPExtractionError: 提取失败
        """
        http_requests = []
        
        try:
            # 提取原始负载
            payloads = PcapParser.extract_raw_payloads(packets)
            
            for timestamp, raw_data in payloads:
                # 尝试解码为文本
                try:
                    # 尝试多种编码
                    decoded = None
                    for encoding in ['utf-8', 'latin-1', 'ascii']:
                        try:
                            decoded = raw_data.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if decoded is None:
                        continue
                    
                    # 检查是否是HTTP请求
                    http_request = HTTPExtractor._parse_http_request(decoded, timestamp)
                    if http_request:
                        http_requests.append(http_request)
                        
                except Exception as e:
                    logger.debug(f"解析数据包负载失败: {e}")
                    continue
            
            # 按时间戳排序
            http_requests.sort(key=lambda x: x.timestamp)
            
            # 更新session的http_requests
            session.http_requests = http_requests
            
            logger.info(f"从pcap会话中提取到 {len(http_requests)} 个HTTP请求")
            return http_requests
            
        except Exception as e:
            logger.error(f"提取HTTP请求失败: {e}")
            raise HTTPExtractionError(f"提取HTTP请求失败: {e}")
    
    @staticmethod
    def _parse_http_request(raw_text: str, timestamp: float) -> Optional[HTTPRequest]:
        """
        解析单个HTTP请求
        
        Args:
            raw_text: 原始HTTP请求文本
            timestamp: 时间戳
            
        Returns:
            HTTPRequest对象，如果不是有效的HTTP请求则返回None
        """
        # 检查是否以HTTP方法开头
        if not raw_text.startswith(('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS')):
            return None
        
        # 分割行
        lines = raw_text.split('\r\n')
        if not lines:
            return None
        
        # 解析请求行
        request_line = lines[0]
        match = HTTPExtractor.HTTP_METHOD_PATTERN.match(request_line)
        
        if not match:
            return None
        
        method = match.group(1).upper()
        path = match.group(2)
        
        # 解析头部
        host = ''
        user_agent = None
        content_type = None
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                break  # 空行表示头部结束
            
            # Host
            host_match = HTTPExtractor.HOST_PATTERN.match(line)
            if host_match:
                host = host_match.group(1).strip()
                continue
            
            # User-Agent
            ua_match = HTTPExtractor.USER_AGENT_PATTERN.match(line)
            if ua_match:
                user_agent = ua_match.group(1).strip()
                continue
            
            # Content-Type
            ct_match = HTTPExtractor.CONTENT_TYPE_PATTERN.match(line)
            if ct_match:
                content_type = ct_match.group(1).strip()
                continue
        
        # 提取请求体（如果有）
        body = None
        body_start = raw_text.find('\r\n\r\n')
        if body_start != -1:
            body = raw_text[body_start + 4:].strip()
            if not body:
                body = None
        
        return HTTPRequest(
            method=method,
            path=path,
            host=host,
            timestamp=timestamp,
            user_agent=user_agent,
            content_type=content_type,
            body=body
        )
    
    @staticmethod
    def extract_from_bytes(pcap_bytes: bytes, file_name: str = 'unknown.pcap') -> Tuple[PcapSession, List[HTTPRequest]]:
        """
        从pcap二进制数据中提取HTTP请求（便捷方法）
        
        Args:
            pcap_bytes: pcap文件的二进制内容
            file_name: 文件名
            
        Returns:
            (PcapSession, List[HTTPRequest]) 元组
        """
        from scapy.all import rdpcap
        import tempfile
        import os
        
        # 解析pcap获取会话信息
        session = PcapParser.parse_bytes(pcap_bytes, file_name)
        
        # 重新读取数据包用于HTTP提取
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
            tmp.write(pcap_bytes)
            tmp_path = tmp.name
        
        try:
            packets = rdpcap(tmp_path)
            http_requests = HTTPExtractor.extract(session, packets)
            return session, http_requests
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
