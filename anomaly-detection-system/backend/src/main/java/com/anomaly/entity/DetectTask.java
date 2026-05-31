package com.anomaly.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.io.Serializable;
import java.util.Date;

/**
 * 实时异常流量检测任务模拟设置
 */
@Data
@TableName("detect_task")
public class DetectTask implements Serializable {
    private static final long serialVersionUID = 1L;

    @TableId(type = IdType.AUTO)
    private Long id;

    private String taskName;

    /** 0:停止 1:运行中 */
    private Integer status;

    /** 每秒发包数 */
    private Integer qpsSetting;

    private Date createTime;
}
