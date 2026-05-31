package com.anomaly.service.impl;

import com.anomaly.entity.XaiRecord;
import com.anomaly.mapper.XaiRecordMapper;
import com.anomaly.service.XaiRecordService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

@Service
public class XaiRecordServiceImpl extends ServiceImpl<XaiRecordMapper, XaiRecord> implements XaiRecordService {
}
