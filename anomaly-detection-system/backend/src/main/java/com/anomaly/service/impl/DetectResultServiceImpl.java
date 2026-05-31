package com.anomaly.service.impl;

import com.anomaly.entity.DetectResult;
import com.anomaly.mapper.DetectResultMapper;
import com.anomaly.service.DetectResultService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

@Service
public class DetectResultServiceImpl extends ServiceImpl<DetectResultMapper, DetectResult>
        implements DetectResultService {
}
