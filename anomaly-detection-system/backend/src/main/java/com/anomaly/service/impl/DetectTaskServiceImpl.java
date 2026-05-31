package com.anomaly.service.impl;

import com.anomaly.entity.DetectTask;
import com.anomaly.mapper.DetectTaskMapper;
import com.anomaly.service.DetectTaskService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

@Service
public class DetectTaskServiceImpl extends ServiceImpl<DetectTaskMapper, DetectTask> implements DetectTaskService {

    @Override
    public void updateTaskStatus(Long taskId, Integer status, Integer qpsSetting) {
        DetectTask task = this.getById(taskId);
        if (task != null) {
            task.setStatus(status);
            task.setQpsSetting(qpsSetting);
            this.updateById(task);
        }
    }
}
