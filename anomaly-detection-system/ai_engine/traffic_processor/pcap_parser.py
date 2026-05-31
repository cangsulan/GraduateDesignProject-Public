# -*- coding: utf-8 -*-
"""
pcap文件解析模块

负责解析pcap格式的网络流量捕获文件，提取：
- 网络层信息（源/目标IP）
- 传输层信息（源/目标端口）
- 时间戳信息
- 原始数据包内容

使用scapy库进行pcap文件解析
"""

import os
from typing import List, Tuple, Optional
from scapy.all import rdpcap, IP, TCP, Raw
from loguru import logger

from .models import PcapSession, HTTPRequest, PcapParseError


class PcapParser:
    """
    pcap文件解析器
    
    解析单个pcap文件，提取TCP会话的基本信息和原始数据包
    """
    
    @staticmethod
    def parse_file(file_path: str) -> PcapSession:
        """
        解析pcap文件
        
        Args:
            file_path: pcap文件路径
            
        Returns:
            PcapSession对象，包含会话的基本信息
            
        Raises:
            PcapParseError: pcap文件解析失败
        """
        try:
            # 读取pcap文件
            packets = rdpcap(file_path)
            
            if len(packets) == 0:
                raise PcapParseError(f"pcap文件为空: {file_path}")
            
            # 提取文件名
            file_name = os.path.basename(file_path)
            
            # 从文件名解析IP和端口信息
            source_ip, source_port, dest_ip, dest_port = PcapParser._parse_filename(file_name)
            
            # 如果文件名解析失败，从数据包中提取
            if source_ip is None:
                source_ip, dest_ip, source_port, dest_port = PcapParser._extract_from_packets(packets)
            
            # 提取时间戳
            timestamps = []
            for pkt in packets:
                if hasattr(pkt, 'time'):
                    timestamps.append(float(pkt.time))
            
            if not timestamps:
                timestamps = [0.0]
            
            start_time = min(timestamps)
            end_time = max(timestamps)
            
            # 创建PcapSession对象
            session = PcapSession(
                file_name=file_name,
                source_ip=source_ip or 'unknown',
                dest_ip=dest_ip or 'unknown',
                source_port=source_port or 0,
                dest_port=dest_port or 0,
                start_time=start_time,
                end_time=end_time,
                packet_count=len(packets),
                http_requests=[]
            )
            
            return session
            
        except PcapParseError:
            raise
        except Exception as e:
            logger.error(f"解析pcap文件失败: {file_path}, 错误: {e}")
            raise PcapParseError(f"解析pcap文件失败: {e}")
    
    @staticmethod
    def parse_bytes(pcap_bytes: bytes, file_name: str = 'unknown.pcap') -> PcapSession:
        """
        解析pcap二进制数据
        
        Args:
            pcap_bytes: pcap文件的二进制内容
            file_name: 文件名（用于标识）
            
        Returns:
            PcapSession对象
            
        Raises:
            PcapParseError: 解析失败
        """
        import io
        import tempfile
        
        try:
            # 将bytes写入临时文件，因为scapy需要文件路径
            with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
                tmp.write(pcap_bytes)
                tmp_path = tmp.name
            
            try:
                session = PcapParser.parse_file(tmp_path)
                session.file_name = file_name
                return session
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except PcapParseError:
            raise
        except Exception as e:
            logger.error(f"解析pcap二进制数据失败: {e}")
            raise PcapParseError(f"解析pcap二进制数据失败: {e}")
    
    @staticmethod
    def _parse_filename(file_name: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
        """
        从文件名解析IP和端口信息
        
        文件名格式: {源IP}_{源端口}_{目标IP}_{目标端口}.pcap
        示例: 127.0.0.1_49153_127.0.0.1_8888.pcap
        
        Args:
            file_name: 文件名
            
        Returns:
            (源IP, 源端口, 目标IP, 目标端口) 元组
        """
        try:
            # 移除.pcap后缀
            base_name = file_name.replace('.pcap', '')
            parts = base_name.split('_')
            
            if len(parts) >= 4:
                source_ip = parts[0]
                source_port = int(parts[1])
                dest_ip = parts[2]
                dest_port = int(parts[3])
                return source_ip, source_port, dest_ip, dest_port
            
            return None, None, None, None
            
        except Exception:
            return None, None, None, None
    
    @staticmethod
    def _extract_from_packets(packets) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
        """
        从数据包中提取IP和端口信息
        
        Args:
            packets: scapy数据包列表
            
        Returns:
            (源IP, 目标IP, 源端口, 目标端口) 元组
        """
        source_ip = None
        dest_ip = None
        source_port = None
        dest_port = None
        
        for pkt in packets:
            if IP in pkt:
                if source_ip is None:
                    source_ip = pkt[IP].src
                    dest_ip = pkt[IP].dst
            if TCP in pkt:
                if source_port is None:
                    source_port = pkt[TCP].sport
                    dest_port = pkt[TCP].dport
            
            # 如果已经提取到所有信息，提前退出
            if all([source_ip, dest_ip, source_port, dest_port]):
                break
        
        return source_ip, dest_ip, source_port, dest_port
    
    @staticmethod
    def extract_raw_payloads(packets) -> List[Tuple[float, bytes]]:
        """
        提取数据包中的原始负载
        
        Args:
            packets: scapy数据包列表
            
        Returns:
            (时间戳, 负载数据) 元组列表
        """
        payloads = []
        
        for pkt in packets:
            if Raw in pkt:
                timestamp = float(pkt.time) if hasattr(pkt, 'time') else 0.0
                raw_data = bytes(pkt[Raw].load)
                payloads.append((timestamp, raw_data))
        
        return payloads
