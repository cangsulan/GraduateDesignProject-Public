package com.anomaly.controller;

import com.anomaly.common.Result;
import com.anomaly.entity.XaiRecord;
import com.anomaly.mq.RabbitProducer;
import com.anomaly.service.XaiRecordService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Date;

@RestController
@RequestMapping("/api/xai")
public class XaiController {

    @Autowired
    private XaiRecordService xaiRecordService;

    @Autowired
    private RabbitProducer rabbitProducer;

    /**
     * 手动触发指定流量链路 (traceId) 的深度溯源分析
     */
    @PostMapping("/analyze/{traceId}")
    public Result<String> analyze(@PathVariable String traceId) {
        XaiRecord record = xaiRecordService.getOne(new QueryWrapper<XaiRecord>().eq("trace_id", traceId));
        if (record == null) {
            record = new XaiRecord();
            record.setTraceId(traceId);
            record.setStatus(0); // 处理中
            xaiRecordService.save(record);
            rabbitProducer.sendXaiRequest(traceId);
            return Result.success("分析任务已提交，请等待处理结果");
        }

        if (record.getStatus() == 0) {
            return Result.success("分析任务正在处理中...");
        } else if (record.getStatus() == 2 || record.getStatus() == 1) {
            // 失败或已完成的均可触发重制与重新列队
            record.setStatus(0);
            xaiRecordService.updateById(record);
            rabbitProducer.sendXaiRequest(traceId);
            return Result.success("该链路分析任务已重新送入集群列队");
        }
        return Result.success("未知状态");
    }

    /**
     * 获取 XAI 溯源分析结果
     */
    @GetMapping("/result/{traceId}")
    public Result<XaiRecord> getResult(@PathVariable String traceId) {
        XaiRecord record = xaiRecordService.getOne(new QueryWrapper<XaiRecord>().eq("trace_id", traceId));
        if (record != null) {
            if (record.getStatus() == 1) {
                return Result.success(record);
            } else if (record.getStatus() == 2) {
                return Result.error(500, "底层 AI 引擎在分析这条样本时发生致命错误或中断。");
            }
        }
        return Result.error(202, "分析尚未完成或未发起");
    }
}
