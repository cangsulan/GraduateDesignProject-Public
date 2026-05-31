# -*- coding: utf-8 -*-
"""
置信度自适应融合策略模块

该模块实现了基于置信度的自适应融合策略，用于微服务异常流量检测。

核心思想：
1. 利用RF和GCN的互补特性
2. 根据模型置信度动态调整融合权重
3. 根据预测一致性动态调整判定阈值

"""

import numpy as np


OPTIMAL_PARAMS = {
    'alpha': 0.4,
    'conf_threshold': 0.5,
    'weight_high_conf': 0.4,
    'weight_low_conf': 0.15,
    'threshold_adjust_rf': 0.1,
    'threshold_adjust_gcn': 0.05
}


def confidence_adaptive_fusion(rf_prob, gcn_prob, 
                               alpha=OPTIMAL_PARAMS['alpha'],
                               conf_threshold=OPTIMAL_PARAMS['conf_threshold'],
                               weight_high_conf=OPTIMAL_PARAMS['weight_high_conf'],
                               weight_low_conf=OPTIMAL_PARAMS['weight_low_conf'],
                               threshold_adjust_rf=OPTIMAL_PARAMS['threshold_adjust_rf'],
                               threshold_adjust_gcn=OPTIMAL_PARAMS['threshold_adjust_gcn']):
    """
    置信度自适应融合策略
    
    参数:
    - rf_prob: RF模型预测概率
    - gcn_prob: GCN模型预测概率
    - alpha: 基础融合权重
    - conf_threshold: 置信度阈值
    - weight_high_conf: 高置信度时RF权重
    - weight_low_conf: 低置信度时RF权重
    - threshold_adjust_rf: RF异常时阈值调整系数
    - threshold_adjust_gcn: GCN异常时阈值调整系数
    
    返回:
    - final_prob: 融合后的概率
    - threshold: 自适应阈值
    - fusion_info: 融合信息字典
    """
    rf_conf = abs(rf_prob - 0.5) * 2
    gcn_conf = abs(gcn_prob - 0.5) * 2
    
    rf_pred = 1 if rf_prob > 0.5 else 0
    gcn_pred = 1 if gcn_prob > 0.5 else 0
    
    fusion_info = {
        'rf_prob': rf_prob,
        'gcn_prob': gcn_prob,
        'rf_conf': rf_conf,
        'gcn_conf': gcn_conf,
        'rf_pred': rf_pred,
        'gcn_pred': gcn_pred
    }
    
    if rf_pred == gcn_pred:
        fused = alpha * rf_prob + (1 - alpha) * gcn_prob
        threshold = 0.5
        fusion_info['mode'] = 'consistent'
        fusion_info['weight_rf'] = alpha
        fusion_info['weight_gcn'] = 1 - alpha
        
    elif rf_pred == 1 and gcn_pred == 0:
        if rf_conf > conf_threshold:
            fused = weight_high_conf * rf_prob + (1 - weight_high_conf) * gcn_prob
            fusion_info['mode'] = 'rf_high_conf'
            fusion_info['weight_rf'] = weight_high_conf
            fusion_info['weight_gcn'] = 1 - weight_high_conf
        else:
            fused = weight_low_conf * rf_prob + (1 - weight_low_conf) * gcn_prob
            fusion_info['mode'] = 'rf_low_conf'
            fusion_info['weight_rf'] = weight_low_conf
            fusion_info['weight_gcn'] = 1 - weight_low_conf
        
        threshold = 0.5 + threshold_adjust_rf * (1 - rf_conf)
        
    else:
        if gcn_conf > conf_threshold:
            fused = (1 - weight_high_conf) * rf_prob + weight_high_conf * gcn_prob
            fusion_info['mode'] = 'gcn_high_conf'
            fusion_info['weight_rf'] = 1 - weight_high_conf
            fusion_info['weight_gcn'] = weight_high_conf
        else:
            fused = alpha * rf_prob + (1 - alpha) * gcn_prob
            fusion_info['mode'] = 'gcn_low_conf'
            fusion_info['weight_rf'] = alpha
            fusion_info['weight_gcn'] = 1 - alpha
        
        threshold = 0.5 - threshold_adjust_gcn * gcn_conf
    
    fusion_info['threshold'] = threshold
    fusion_info['fused_prob'] = fused
    
    final_prob = fused
    is_anomaly = 1 if fused > threshold else 0
    
    return final_prob, threshold, is_anomaly, fusion_info


def fixed_weight_fusion(rf_prob, gcn_prob, alpha=0.33):
    """
    固定权重融合（基线方法）
    
    参数:
    - rf_prob: RF模型预测概率
    - gcn_prob: GCN模型预测概率
    - alpha: RF权重
    
    返回:
    - final_prob: 融合后的概率
    - is_anomaly: 是否异常
    - fusion_info: 融合信息字典
    """
    final_prob = alpha * rf_prob + (1 - alpha) * gcn_prob
    is_anomaly = 1 if final_prob > 0.5 else 0
    
    fusion_info = {
        'mode': 'fixed',
        'weight_rf': alpha,
        'weight_gcn': 1 - alpha,
        'threshold': 0.5
    }
    
    return final_prob, is_anomaly, fusion_info
