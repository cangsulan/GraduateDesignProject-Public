package com.anomaly.controller;

import com.anomaly.common.Result;
import com.anomaly.entity.SysUser;
import com.anomaly.security.JwtUtils;
import com.anomaly.service.SysUserService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    @Autowired
    private SysUserService sysUserService;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private JwtUtils jwtUtils;

    @PostMapping("/login")
    public Result<Map<String, Object>> login(@RequestBody SysUser loginUser) {
        SysUser user = sysUserService.getOne(new QueryWrapper<SysUser>().eq("username", loginUser.getUsername()));

        if (user == null) {
            return Result.error("账号不存在");
        }

        if (!passwordEncoder.matches(loginUser.getPassword(), user.getPassword())) {
            return Result.error("密码错误");
        }

        if (user.getStatus() != null && user.getStatus() == 0) {
            return Result.error("账号已被冻结");
        }

        String token = jwtUtils.generateToken(user.getUsername());
        Map<String, Object> data = new HashMap<>();
        data.put("token", token);
        data.put("username", user.getUsername());
        data.put("role", user.getRole());
        data.put("email", user.getEmail());
        data.put("status", user.getStatus());

        return Result.success(data);
    }
}
