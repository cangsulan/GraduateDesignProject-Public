package com.anomaly.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.io.Serializable;
import java.util.Date;

/**
 * 异步文件批量检测长时任务
 */
@Data
@TableName("detect_file_task")
public class DetectFileTask implements Serializable {
    private static final long serialVersionUID = 1L;

    @TableId(type = IdType.AUTO)
    private Long id;

    private String taskName;

    private String csvUrl;

    private String jsonUrl;

    /** PENDING, DETECTING, COMPLETED, FAILED */
    private String status;

    private Integer recordCount;

    private Integer anomalyCount;

    private Float anomalyRate;

    private Date createTime;

    private Date uploadTime;

    private Date startTime;

    private Date endTime;

    private Long duration;
}
