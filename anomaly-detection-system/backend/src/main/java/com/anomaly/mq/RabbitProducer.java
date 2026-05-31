package com.anomaly.mq;

import com.alibaba.fastjson2.JSON;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * 消息队列生产者
 */
@Slf4j
@Component
public class RabbitProducer {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    /**
     * 发送控制指令给模拟器
     *
     * @param action 指令: start, stop, update
     * @param qps    每秒请求数
     */
    public void sendSimulatorControl(String action, Integer qps) {
        Map<String, Object> msg = new HashMap<>();
        msg.put("action", action);
        if (qps != null) {
            msg.put("qps", qps);
        }

        String jsonMsg = JSON.toJSONString(msg);
        log.info("发送模拟器控制指令: {}", jsonMsg);
        rabbitTemplate.convertAndSend(RabbitMQConfig.EXCHANGE_NAME, RabbitMQConfig.ROUTING_KEY_SIM_CTRL, jsonMsg);
    }

    /**
     * 发送检测数据到 AI 引擎 (通常由前端文件检测或模拟器HTTP请求触发此处)
     */
    public void sendDetectRequest(String jsonMsg) {
        // log.debug("发送检测请求至AI: {}", jsonMsg);
        rabbitTemplate.convertAndSend(RabbitMQConfig.EXCHANGE_NAME, RabbitMQConfig.ROUTING_KEY_DETECT_REQ, jsonMsg);
    }

    /**
     * 发送 XAI 分析请求到后端分析 Worker
     * 
     * @param traceId 链路ID
     */
    public void sendXaiRequest(String traceId) {
        Map<String, String> msg = new HashMap<>();
        msg.put("traceId", traceId);
        String jsonMsg = JSON.toJSONString(msg);
        log.info("发送 XAI 分析请求: {}", jsonMsg);
        rabbitTemplate.convertAndSend(RabbitMQConfig.EXCHANGE_NAME, RabbitMQConfig.ROUTING_KEY_XAI_REQ, jsonMsg);
    }
}
