package com.anomaly.controller;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.anomaly.common.Result;
import com.anomaly.entity.DetectFileTask;
import com.anomaly.entity.DetectResult;
import com.anomaly.service.DetectFileTaskService;
import com.anomaly.service.DetectResultService;
import com.anomaly.service.MinioService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/detect/file-task")
public class DetectFileTaskController {

    @Autowired
    private MinioService minioService;

    @Autowired
    private DetectFileTaskService detectFileTaskService;

    @Autowired
    private DetectResultService detectResultService;

    private final RestTemplate restTemplate = new RestTemplate();

    @PostMapping("/create")
    public Result<String> createTask(@RequestParam("taskName") String taskName,
            @RequestParam(value = "csvFile", required = false) MultipartFile csvFile,
            @RequestParam(value = "jsonFile", required = false) MultipartFile jsonFile) {
        if (!StringUtils.hasText(taskName)) {
            return Result.error("任务名称不能为空");
        }
        if ((csvFile == null || csvFile.isEmpty()) && (jsonFile == null || jsonFile.isEmpty())) {
            return Result.error("必须至少上传一个特征文件或调用链文件");
        }

        DetectFileTask task = new DetectFileTask();
        task.setTaskName(taskName);
        task.setCreateTime(new Date());
        task.setStatus("PENDING");

        try {
            if (csvFile != null && !csvFile.isEmpty()) {
                String csvUrl = minioService.uploadFile(csvFile);
                task.setCsvUrl(csvUrl);
            }
            if (jsonFile != null && !jsonFile.isEmpty()) {
                String jsonUrl = minioService.uploadFile(jsonFile);
                task.setJsonUrl(jsonUrl);
            }
            task.setUploadTime(new Date());

            detectFileTaskService.save(task);

            JSONObject payload = new JSONObject();
            payload.put("taskId", task.getId());
            payload.put("csvUrl", task.getCsvUrl());
            payload.put("jsonUrl", task.getJsonUrl());
            payload.put("callbackUrl", "http://anomaly-backend:8080/api/detect/file-task/callback");

            String pythonAiUrl = "http://anomaly-ai-engine:8000/api/predict/task";

            task.setStatus("DETECTING");
            task.setStartTime(new Date());
            detectFileTaskService.updateById(task);

            restTemplate.postForObject(pythonAiUrl, payload, String.class);

            return Result.success("任务已提交检测！流转排队中...");
        } catch (Exception e) {
            log.error("创建文件检测任务异常", e);
            if (task.getId() != null) {
                task.setStatus("FAILED");
                detectFileTaskService.updateById(task);
            }
            return Result.error("任务创建失败：" + e.getMessage());
        }
    }

    @GetMapping("/page")
    public Result<Page<DetectFileTask>> pageQuery(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String taskName,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate) {

        Page<DetectFileTask> pageParam = new Page<>(pageNum, pageSize);
        QueryWrapper<DetectFileTask> queryWrapper = new QueryWrapper<>();
        if (StringUtils.hasText(taskName)) {
            queryWrapper.like("task_name", taskName);
        }
        if (StringUtils.hasText(status)) {
            queryWrapper.eq("status", status);
        }
        if (StringUtils.hasText(startDate)) {
            queryWrapper.ge("create_time", startDate + " 00:00:00");
        }
        if (StringUtils.hasText(endDate)) {
            queryWrapper.le("create_time", endDate + " 23:59:59");
        }
        queryWrapper.orderByDesc("create_time");

        Page<DetectFileTask> resultPage = detectFileTaskService.page(pageParam, queryWrapper);
        return Result.success(resultPage);
    }

    @PostMapping("/callback")
    public Result<String> callback(@RequestBody JSONObject payload) {
        Long taskId = payload.getLong("taskId");
        String status = payload.getString("status");
        Integer recordCount = payload.getInteger("recordCount");
        Integer anomalyCount = payload.getInteger("anomalyCount");
        Long duration = payload.getLong("duration");
        Boolean fullData = payload.getBoolean("fullData");

        if (taskId == null)
            return Result.error("缺少taskId");

        DetectFileTask task = detectFileTaskService.getById(taskId);
        if (task != null) {
            task.setStatus(status);
            task.setRecordCount(recordCount);
            task.setAnomalyCount(anomalyCount);
            task.setDuration(duration);
            if (recordCount != null && recordCount > 0 && anomalyCount != null) {
                task.setAnomalyRate((float) anomalyCount / recordCount);
            }
            task.setEndTime(new Date());
            detectFileTaskService.updateById(task);
            log.info("接收到AI引擎的检测结果: taskId={}, status={}, fullData={}", taskId, status, fullData);

            JSONArray records = payload.getJSONArray("records");
            if (records != null && !records.isEmpty()) {
                long now = System.currentTimeMillis();
                List<DetectResult> resultList = new ArrayList<>();
                for (int i = 0; i < records.size(); i++) {
                    JSONObject rec = records.getJSONObject(i);
                    DetectResult dr = new DetectResult();
                    dr.setTraceId(rec.getString("traceId"));
                    dr.setSourceIp(rec.getString("sourceIp"));
                    dr.setRfProb(rec.getFloat("rfProb"));
                    dr.setGcnProb(rec.getFloat("gcnProb"));
                    dr.setFinalProb(rec.getFloat("finalProb"));
                    dr.setIsAnomaly(rec.getInteger("isAnomaly"));
                    dr.setDetectType(1);
                    dr.setFileId(taskId);
                    dr.setTaskName(task.getTaskName());
                    dr.setTimestamp(now + i);
                    
                    String featuresJson = rec.getString("features");
                    if (featuresJson != null) {
                        dr.setFeaturesJson(featuresJson);
                    }
                    String callGraphJson = rec.getString("callGraph");
                    if (callGraphJson != null) {
                        dr.setCallGraphJson(callGraphJson);
                    }
                    
                    resultList.add(dr);
                }
                detectResultService.saveBatch(resultList);
                log.info("文件检测结果批量写入 detect_result 完成: {} 条 (fullData={})", resultList.size(), fullData);
            }
        }
        return Result.success("OK");
    }

    @DeleteMapping("/{id}")
    public Result<String> delete(@PathVariable Long id) {
        detectFileTaskService.removeById(id);
        return Result.success("删除成功");
    }

    /**
     * 从pcap文件创建检测任务
     * 
     * 该接口接收pcap文件，先上传到MinIO存储，再转发给AI引擎进行特征提取和检测。
     * 
     * @param taskName 任务名称
     * @param pcapFiles pcap文件列表（支持多文件上传）
     * @return 任务创建结果
     */
    @PostMapping("/create-from-pcap")
    public Result<String> createTaskFromPcap(
            @RequestParam("taskName") String taskName,
            @RequestParam("pcapFiles") MultipartFile[] pcapFiles) {
        
        if (!StringUtils.hasText(taskName)) {
            return Result.error("任务名称不能为空");
        }
        
        if (pcapFiles == null || pcapFiles.length == 0) {
            return Result.error("必须上传至少一个pcap文件");
        }

        // 创建任务记录
        DetectFileTask task = new DetectFileTask();
        task.setTaskName(taskName);
        task.setCreateTime(new Date());
        task.setStatus("PENDING");

        try {
            // 1. 上传pcap文件到MinIO
            String pcapUrl = minioService.uploadFile(pcapFiles[0]);
            Date uploadTime = new Date();
            
            // 存储pcap文件URL到csvUrl字段（用于标识数据来源和下载链接）
            task.setCsvUrl(pcapUrl);
            task.setUploadTime(uploadTime);
            
            detectFileTaskService.save(task);
            log.info("创建pcap检测任务: taskId={}, taskName={}, fileCount={}, pcapUrl={}", 
                    task.getId(), taskName, pcapFiles.length, pcapUrl);

            // 2. 构建multipart请求发送给AI引擎
            String pythonAiUrl = "http://anomaly-ai-engine:8000/api/predict/from-pcap";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("taskId", task.getId());
            body.add("callbackUrl", "http://anomaly-backend:8080/api/detect/file-task/callback");

            // 添加所有pcap文件
            for (MultipartFile pcapFile : pcapFiles) {
                if (pcapFile != null && !pcapFile.isEmpty()) {
                    ByteArrayResource resource = new ByteArrayResource(pcapFile.getBytes()) {
                        @Override
                        public String getFilename() {
                            return pcapFile.getOriginalFilename();
                        }
                    };
                    body.add("pcapFiles", resource);
                }
            }

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            // 3. 更新任务状态
            task.setStatus("DETECTING");
            task.setStartTime(new Date());
            detectFileTaskService.updateById(task);

            // 4. 异步发送请求（使用线程避免阻塞）
            final Long taskId = task.getId();
            final MultiValueMap<String, Object> finalBody = body;
            new Thread(() -> {
                try {
                    ResponseEntity<String> response = restTemplate.exchange(
                            pythonAiUrl, 
                            HttpMethod.POST, 
                            new HttpEntity<>(finalBody, headers), 
                            String.class);
                    log.info("pcap任务已提交给AI引擎: taskId={}, response={}", 
                            taskId, response.getBody());
                } catch (Exception e) {
                    log.error("pcap任务提交失败: taskId={}, error={}", taskId, e.getMessage());
                    DetectFileTask failTask = detectFileTaskService.getById(taskId);
                    if (failTask != null) {
                        failTask.setStatus("FAILED");
                        failTask.setEndTime(new Date());
                        detectFileTaskService.updateById(failTask);
                    }
                }
            }).start();

            return Result.success("PCAP检测任务已提交！处理中...");
            
        } catch (Exception e) {
            log.error("创建pcap检测任务异常", e);
            if (task.getId() != null) {
                task.setStatus("FAILED");
                detectFileTaskService.updateById(task);
            }
            return Result.error("任务创建失败：" + e.getMessage());
        }
    }

    /**
     * 检查pcap处理功能是否可用
     */
    @GetMapping("/pcap-status")
    public Result<JSONObject> checkPcapStatus() {
        try {
            String pythonAiUrl = "http://anomaly-ai-engine:8000/api/pcap/status";
            ResponseEntity<JSONObject> response = restTemplate.exchange(
                    pythonAiUrl, 
                    HttpMethod.GET, 
                    null, 
                    JSONObject.class);
            return Result.success(response.getBody());
        } catch (Exception e) {
            log.error("检查pcap状态失败", e);
            JSONObject errorResult = new JSONObject();
            errorResult.put("trafficProcessorAvailable", false);
            errorResult.put("message", "AI引擎不可用: " + e.getMessage());
            return Result.success(errorResult);
        }
    }
}
