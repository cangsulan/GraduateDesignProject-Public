package com.anomaly.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.io.Serializable;
import java.util.Date;

/**
 * 系统用户实体
 */
@Data
@TableName("sys_user")
public class SysUser implements Serializable {
    private static final long serialVersionUID = 1L;

    @TableId(type = IdType.AUTO)
    private Long id;

    private String username;

    private String password;

    private String email;

    /**
     * 状态：0-冻结，1-启用
     */
    private Integer status;

    /**
     * 角色：admin-管理员，user-普通用户
     */
    private String role;

    private Date createTime;
}
