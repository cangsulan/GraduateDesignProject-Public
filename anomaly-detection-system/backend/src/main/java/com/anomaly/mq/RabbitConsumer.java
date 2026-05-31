package com.anomaly.mq;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.anomaly.entity.DetectResult;
import com.anomaly.entity.XaiRecord;
import com.anomaly.service.DetectResultService;
import com.anomaly.service.XaiRecordService;
import com.anomaly.ws.WebSocketServer;
import com.rabbitmq.client.Channel;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.Date;

/**
 * 消息队列消费者
 */
@Slf4j
@Component
public class RabbitConsumer {

    @Autowired
    private DetectResultService detectResultService;

    @Autowired
    private XaiRecordService xaiRecordService;

    /**
     * 消费 AI 传回来的检测结果
     */
    @RabbitListener(queues = RabbitMQConfig.QUEUE_DETECT_RES)
    public void receiveDetectResult(Message message, Channel channel) throws IOException {
        String jsonMsg = new String(message.getBody());
        // log.debug("收到AI引擎返回的检测结果: {}", jsonMsg);

        try {
            JSONObject jsonObject = JSON.parseObject(jsonMsg);

            DetectResult result = new DetectResult();
            result.setTraceId(jsonObject.getString("traceId"));
            result.setRfProb(jsonObject.getFloat("rfProb"));
            result.setGcnProb(jsonObject.getFloat("gcnProb"));
            result.setFinalProb(jsonObject.getFloat("finalProb"));
            result.setIsAnomaly(jsonObject.getInteger("isAnomaly"));
            result.setDetectType(0); // 0 代表实时流检测，1 代表文件批处理

            // 为了简化，源特征直接从消息中提取，或者可以在 HTTP 接收阶段预先写入数据库
            // 我们默认 HTTP 请求收到流量包时只做转发不到数据库，由 AI 检测出结果后一并入库
            if (jsonObject.containsKey("features")) {
                result.setFeaturesJson(jsonObject.getString("features"));
            }
            if (jsonObject.containsKey("callGraph")) {
                result.setCallGraphJson(jsonObject.getString("callGraph"));
            }
            if (jsonObject.containsKey("sourceIp")) {
                result.setSourceIp(jsonObject.getString("sourceIp"));
            }
            result.setTimestamp(jsonObject.getLongValue("timestamp"));

            // 入库保存：针对重复模拟数据，改用存在则更新，不存在则插入的逻辑，避免完整性约束冲撞导致 WebSocket 中断
            try {
                DetectResult exist = detectResultService.query().eq("trace_id", result.getTraceId()).one();
                if (exist != null) {
                    result.setId(exist.getId());
                    detectResultService.updateById(result);
                } else {
                    detectResultService.save(result);
                }
            } catch (org.springframework.dao.DuplicateKeyException dke) {
                log.warn("并发发包导致检测结果重复写入 (traceId={})，主动忽略防止错误溢出。", result.getTraceId());
            }

            // 将结果通过 WebSocket Push 到前端大屏实时展示
            // 我们构造一个带 type 为 detectResult 的消息
            JSONObject wsMsg = new JSONObject();
            wsMsg.put("type", "detectResult");
            wsMsg.put("data", result);
            WebSocketServer.broadcastMessage(wsMsg);

            // 手动 Ack
            channel.basicAck(message.getMessageProperties().getDeliveryTag(), false);
        } catch (Exception e) {
            log.error("处理检测结果消息异常", e);
            // 发生异常，暂时拒绝并丢弃（防止死信循环）
            channel.basicReject(message.getMessageProperties().getDeliveryTag(), false);
        }
    }

    /**
     * 消费 AI XAI Worker 传回来的溯源解析结果
     */
    @RabbitListener(queues = RabbitMQConfig.QUEUE_XAI_RES)
    public void receiveXaiResult(Message message, Channel channel) throws IOException {
        String jsonMsg = new String(message.getBody());
        log.info("收到XAI Worker返回的解析结果: {}", jsonMsg);

        try {
            JSONObject jsonObject = JSON.parseObject(jsonMsg);
            String traceId = jsonObject.getString("traceId");

            XaiRecord record = xaiRecordService.query().eq("trace_id", traceId).one();
            if (record != null) {
                if (jsonObject.containsKey("error")) {
                    record.setStatus(2); // 标记异常失败中止
                } else {
                    record.setStatus(1); // 完成
                    record.setLimeWeightsJson(jsonObject.getString("limeWeights"));
                    record.setAbnormalEdgesJson(jsonObject.getString("abnormalEdges"));
                }
                record.setAnalysisTime(new Date());
                xaiRecordService.updateById(record);

                // 推送给前端 WebSocket
                JSONObject wsMsg = new JSONObject();
                wsMsg.put("type", "xaiResult");
                wsMsg.put("data", record);
                WebSocketServer.broadcastMessage(wsMsg);
            }

            channel.basicAck(message.getMessageProperties().getDeliveryTag(), false);
        } catch (Exception e) {
            log.error("处理XAI解析结果消息异常", e);
            channel.basicReject(message.getMessageProperties().getDeliveryTag(), false);
        }
    }
}
