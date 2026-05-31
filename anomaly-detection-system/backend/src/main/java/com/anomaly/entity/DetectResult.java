package com.anomaly.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.io.Serializable;
import java.util.Date;

/**
 * 异常流量检测结果记录实体
 */
@Data
@TableName("detect_result")
public class DetectResult implements Serializable {
    private static final long serialVersionUID = 1L;

    @TableId(type = IdType.AUTO)
    private Long id;

    private String traceId;

    private String sourceIp;

    private Long timestamp;

    private Float rfProb;

    private Float gcnProb;

    private Float finalProb;

    /** 0:正常 1:异常 */
    private Integer isAnomaly;

    /** JSON结构存储的节点特征数据 */
    private String featuresJson;

    /** JSON结构存储的拓扑图结构 */
    private String callGraphJson;

    /** 0:实时检测 1:文件离线检测 */
    private Integer detectType;

    /** 关联上传的 detect_file 记录体 ID */
    private Long fileId;

    /** 关联文件检测任务名 */
    private String taskName;

    private Date createTime;
}
