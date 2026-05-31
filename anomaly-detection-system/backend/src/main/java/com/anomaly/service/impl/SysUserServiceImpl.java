package com.anomaly.service.impl;

import com.anomaly.entity.SysUser;
import com.anomaly.mapper.SysUserMapper;
import com.anomaly.service.SysUserService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

@Service
public class SysUserServiceImpl extends ServiceImpl<SysUserMapper, SysUser> implements SysUserService {
}
