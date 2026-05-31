package com.anomaly.controller;

import com.anomaly.common.Result;
import com.anomaly.entity.DetectResult;
import com.anomaly.service.DetectResultService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.alibaba.fastjson2.JSON;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@RestController
@RequestMapping("/api/detect")
public class DetectResultController {

    @Autowired
    private DetectResultService detectResultService;

    @Autowired
    private StringRedisTemplate redisTemplate;

    /**
     * 分页查询检测历史记录
     */
    @GetMapping("/history")
    public Result<Page<DetectResult>> getHistory(
            @RequestParam(defaultValue = "1") Integer current,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String traceId,
            @RequestParam(required = false) String sourceIp,
            @RequestParam(required = false) Integer isAnomaly,
            @RequestParam(required = false) Integer detectType,
            @RequestParam(required = false) Double minProb,
            @RequestParam(required = false) Double maxProb,
            @RequestParam(required = false) Long startTime,
            @RequestParam(required = false) Long endTime,
            @RequestParam(required = false) Long fileId,
            @RequestParam(required = false) String taskName) {

        QueryWrapper<DetectResult> queryWrapper = new QueryWrapper<>();
        if (traceId != null && !traceId.isBlank()) {
            queryWrapper.like("trace_id", traceId);
        }
        if (sourceIp != null && !sourceIp.isBlank()) {
            queryWrapper.like("source_ip", sourceIp);
        }
        if (isAnomaly != null) {
            queryWrapper.eq("is_anomaly", isAnomaly);
        }
        if (detectType != null) {
            queryWrapper.eq("detect_type", detectType);
        }
        if (minProb != null) {
            queryWrapper.ge("final_prob", minProb);
        }
        if (maxProb != null) {
            queryWrapper.le("final_prob", maxProb);
        }
        if (startTime != null) {
            queryWrapper.ge("timestamp", startTime);
        }
        if (endTime != null) {
            queryWrapper.le("timestamp", endTime);
        }
        if (fileId != null) {
            queryWrapper.eq("file_id", fileId);
        }
        if (taskName != null && !taskName.isBlank()) {
            queryWrapper.like("task_name", taskName);
        }
        queryWrapper.orderByDesc("timestamp");

        Page<DetectResult> page = new Page<>(current, size);
        return Result.success(detectResultService.page(page, queryWrapper));
    }

    /**
     * 获取数据汇总统计报表 (正常/异常总数)
     * 支持查询参数的时间过滤，为了应对大屏轮询加入了 10秒 级别的 Redis 缓存保护数据库
     */
    @GetMapping("/statistics")
    public Result<Map<String, Object>> getStatistics(
            @RequestParam(required = false) Long startTime,
            @RequestParam(required = false) Long endTime) {

        String cacheKey = "detect:stats:" + (startTime == null ? "all" : startTime) + "_"
                + (endTime == null ? "all" : endTime);
        String cachedValue = redisTemplate.opsForValue().get(cacheKey);
        if (cachedValue != null) {
            try {
                // Warning: Fastjson2 JSON.parseObject returns JSONObject which implements Map
                Map<String, Object> map = JSON.parseObject(cachedValue, Map.class);
                return Result.success(map);
            } catch (Exception e) {
                // ignore
            }
        }

        QueryWrapper<DetectResult> queryNormal = new QueryWrapper<DetectResult>().eq("is_anomaly", 0);
        QueryWrapper<DetectResult> queryAnomaly = new QueryWrapper<DetectResult>().eq("is_anomaly", 1);

        if (startTime != null) {
            queryNormal.ge("timestamp", startTime);
            queryAnomaly.ge("timestamp", startTime);
        }
        if (endTime != null) {
            queryNormal.le("timestamp", endTime);
            queryAnomaly.le("timestamp", endTime);
        }

        long normalCount = detectResultService.count(queryNormal);
        long anomalyCount = detectResultService.count(queryAnomaly);
        long totalCount = normalCount + anomalyCount;

        Map<String, Object> map = new HashMap<>();
        map.put("normal", normalCount);
        map.put("anomaly", anomalyCount);
        map.put("total", totalCount);

        // 写入 Redis 缓存，过期时间 10 秒
        redisTemplate.opsForValue().set(cacheKey, JSON.toJSONString(map), 10, TimeUnit.SECONDS);

        return Result.success(map);
    }
}
