package com.anomaly.controller;

import com.anomaly.common.Result;
import com.anomaly.entity.DetectTask;
import com.anomaly.mq.RabbitProducer;
import com.anomaly.service.DetectTaskService;
import com.alibaba.fastjson2.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/simulator")
public class SimulatorController {

    @Autowired
    private RabbitProducer rabbitProducer;

    @Autowired
    private DetectTaskService detectTaskService;

    private static final Long SIMULATOR_TASK_ID = 1L;

    /**
     * 获取模拟器当前状态
     */
    @GetMapping("/status")
    public Result<JSONObject> getStatus() {
        DetectTask task = detectTaskService.getById(SIMULATOR_TASK_ID);
        JSONObject status = new JSONObject();
        
        if (task != null) {
            status.put("isRunning", task.getStatus() != null && task.getStatus() == 1);
            status.put("qps", task.getQpsSetting() != null ? task.getQpsSetting() : 10);
            status.put("taskId", task.getId());
            status.put("taskName", task.getTaskName());
        } else {
            status.put("isRunning", false);
            status.put("qps", 10);
            status.put("taskId", SIMULATOR_TASK_ID);
            status.put("taskName", "流量模拟器");
        }
        
        return Result.success(status);
    }

    /**
     * 控制模拟器启停与QPS
     */
    @PostMapping("/control")
    public Result<String> control(@RequestBody Map<String, Object> params) {
        String action = (String) params.get("action");
        Integer qps = params.get("qps") != null ? Integer.valueOf(params.get("qps").toString()) : null;

        if ("start".equals(action)) {
            detectTaskService.updateTaskStatus(SIMULATOR_TASK_ID, 1, qps);
            rabbitProducer.sendSimulatorControl("start", qps);
            return Result.success("流量模拟器已启动");
        } else if ("stop".equals(action)) {
            detectTaskService.updateTaskStatus(SIMULATOR_TASK_ID, 0, null);
            rabbitProducer.sendSimulatorControl("stop", null);
            return Result.success("流量模拟器已停止");
        } else if ("update".equals(action)) {
            detectTaskService.updateTaskStatus(SIMULATOR_TASK_ID, 1, qps);
            rabbitProducer.sendSimulatorControl("update", qps);
            return Result.success("流量系统QPS已更新");
        }

        return Result.error("未知指令");
    }

    /**
     * 接收流量模拟器发送的 HTTP 伪装流量
     */
    @PostMapping("/receive")
    public Result<String> receiveTraffic(@RequestBody JSONObject body) {
        rabbitProducer.sendDetectRequest(body.toJSONString());
        return Result.success("OK");
    }
}
