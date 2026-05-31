package com.anomaly.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.io.Serializable;
import java.util.Date;

/**
 * XAI（可解释性 AI）记录实体
 */
@Data
@TableName("xai_record")
public class XaiRecord implements Serializable {
    private static final long serialVersionUID = 1L;

    @TableId(type = IdType.AUTO)
    private Long id;

    private String traceId;

    /** 0:处理中 1:完成 */
    private Integer status;

    /** JSON结构的 LIME 特征贡献权重 (针对 RF) */
    private String limeWeightsJson;

    /** JSON结构的异常边权重掩码 (针对 GCN) */
    private String abnormalEdgesJson;

    private Date analysisTime;
}
