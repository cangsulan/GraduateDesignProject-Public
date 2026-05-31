package com.anomaly.service;

import com.anomaly.entity.DetectTask;
import com.baomidou.mybatisplus.extension.service.IService;

public interface DetectTaskService extends IService<DetectTask> {
    /**
     * 更新任务状态和QPS
     */
    void updateTaskStatus(Long taskId, Integer status, Integer qpsSetting);
}
