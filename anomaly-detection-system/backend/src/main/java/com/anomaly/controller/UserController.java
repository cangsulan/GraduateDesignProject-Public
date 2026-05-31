package com.anomaly.controller;

import com.anomaly.common.Result;
import com.anomaly.entity.SysUser;
import com.anomaly.security.JwtUtils;
import com.anomaly.service.SysUserService;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import jakarta.servlet.http.HttpServletRequest;
import java.util.List;
import java.util.Map;
import java.util.Date;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private SysUserService sysUserService;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private JwtUtils jwtUtils;

    /**
     * 获取当前操作用户的角色并判断是否为admin
     */
    private boolean isAdmin(HttpServletRequest request) {
        String token = request.getHeader("Authorization");
        if (token != null && token.startsWith("Bearer ")) {
            token = token.substring(7);
            String username = jwtUtils.extractUsername(token);
            SysUser user = sysUserService.getOne(new QueryWrapper<SysUser>().eq("username", username));
            return user != null && "admin".equals(user.getRole());
        }
        return false;
    }

    /**
     * 获取当前操作的用户名
     */
    private String getCurrentUsername(HttpServletRequest request) {
        String token = request.getHeader("Authorization");
        if (token != null && token.startsWith("Bearer ")) {
            token = token.substring(7);
            return jwtUtils.extractUsername(token);
        }
        return null;
    }

    /**
     * 获取所有用户列表 (仅管理员，带分页与组合查询)
     */
    @GetMapping
    public Result<?> getAllUsers(
            HttpServletRequest request,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String role,
            @RequestParam(required = false) Integer status) {
        if (!isAdmin(request)) {
            return Result.error("权限不足");
        }

        LambdaQueryWrapper<SysUser> wrapper = new LambdaQueryWrapper<>();

        if (username != null && !username.trim().isEmpty()) {
            wrapper.like(SysUser::getUsername, username.trim());
        }
        if (role != null && !role.trim().isEmpty()) {
            wrapper.eq(SysUser::getRole, role.trim());
        }
        if (status != null) {
            wrapper.eq(SysUser::getStatus, status);
        }

        // 按照 ID 升序排序
        wrapper.orderByAsc(SysUser::getId);

        Page<SysUser> pageParam = new Page<>(page, size);
        Page<SysUser> userPage = sysUserService.page(pageParam, wrapper);

        // 脱敏处理，不返回密码
        userPage.getRecords().forEach(user -> user.setPassword(null));

        return Result.success(userPage);
    }

    /**
     * 新增用户 (仅管理员)
     */
    @PostMapping
    public Result<?> addUser(@RequestBody SysUser user, HttpServletRequest request) {
        if (!isAdmin(request)) {
            return Result.error("权限不足");
        }

        SysUser existUser = sysUserService.getOne(new QueryWrapper<SysUser>().eq("username", user.getUsername()));
        if (existUser != null) {
            return Result.error("用户名已存在");
        }

        user.setPassword(passwordEncoder.encode(user.getPassword()));
        user.setCreateTime(new Date());
        if (user.getStatus() == null) {
            user.setStatus(1); // 默认启用
        }
        if (user.getRole() == null || user.getRole().isEmpty()) {
            user.setRole("user"); // 默认普通用户
        }

        boolean saved = sysUserService.save(user);
        return saved ? Result.success() : Result.error("添加用户失败");
    }

    /**
     * 更新用户信息 (仅管理员)
     * 注意：不能修改用户名
     */
    @PutMapping("/{id}")
    public Result<?> updateUser(@PathVariable Long id, @RequestBody SysUser user, HttpServletRequest request) {
        if (!isAdmin(request)) {
            return Result.error("权限不足");
        }

        SysUser existUser = sysUserService.getById(id);
        if (existUser == null) {
            return Result.error("用户不存在");
        }

        existUser.setEmail(user.getEmail());
        existUser.setRole(user.getRole());
        existUser.setStatus(user.getStatus());

        // 如果传了密码，顺便修改密码
        if (user.getPassword() != null && !user.getPassword().isEmpty()) {
            existUser.setPassword(passwordEncoder.encode(user.getPassword()));
        }

        boolean updated = sysUserService.updateById(existUser);
        return updated ? Result.success() : Result.error("更新用户失败");
    }

    /**
     * 切换用户状态：启用/冻结 (仅管理员)
     */
    @PutMapping("/{id}/status")
    public Result<?> toggleStatus(@PathVariable Long id, HttpServletRequest request) {
        if (!isAdmin(request)) {
            return Result.error("权限不足");
        }

        SysUser existUser = sysUserService.getById(id);
        if (existUser == null) {
            return Result.error("用户不存在");
        }

        if ("admin".equals(existUser.getUsername())) {
            return Result.error("系统默认管理员不能被冻结");
        }

        existUser.setStatus(existUser.getStatus() == 1 ? 0 : 1);
        boolean updated = sysUserService.updateById(existUser);
        return updated ? Result.success(existUser.getStatus()) : Result.error("操作失败");
    }

    /**
     * 获取当前登录用户信息
     */
    @GetMapping("/me")
    public Result<SysUser> getCurrentUserInfo(HttpServletRequest request) {
        String username = getCurrentUsername(request);
        if (username == null) {
            return Result.error("未认证");
        }
        SysUser user = sysUserService.getOne(new QueryWrapper<SysUser>().eq("username", username));
        if (user == null) {
            return Result.error("用户不存在");
        }
        user.setPassword(null);
        return Result.success(user);
    }

    /**
     * 当前用户修改自己的个人信息（例如邮箱）
     */
    @PutMapping("/me/profile")
    public Result<?> updateMyProfile(@RequestBody SysUser requestUser, HttpServletRequest request) {
        String username = getCurrentUsername(request);
        if (username == null) {
            return Result.error("未认证");
        }
        SysUser user = sysUserService.getOne(new QueryWrapper<SysUser>().eq("username", username));
        if (user == null) {
            return Result.error("用户不存在");
        }

        user.setEmail(requestUser.getEmail());
        boolean updated = sysUserService.updateById(user);
        return updated ? Result.success() : Result.error("更新失败");
    }

    /**
     * 当前用户修改自己的密码
     */
    @PutMapping("/me/password")
    public Result<?> updateMyPassword(@RequestBody Map<String, String> requestMap, HttpServletRequest request) {
        String oldPassword = requestMap.get("oldPassword");
        String newPassword = requestMap.get("newPassword");

        if (oldPassword == null || newPassword == null) {
            return Result.error("参数不完整");
        }

        String username = getCurrentUsername(request);
        if (username == null) {
            return Result.error("未认证");
        }

        SysUser user = sysUserService.getOne(new QueryWrapper<SysUser>().eq("username", username));
        if (user == null) {
            return Result.error("用户不存在");
        }

        if (!passwordEncoder.matches(oldPassword, user.getPassword())) {
            return Result.error("原密码错误");
        }

        user.setPassword(passwordEncoder.encode(newPassword));
        boolean updated = sysUserService.updateById(user);
        return updated ? Result.success() : Result.error("更新密码失败");
    }
}
