-- ============================================================
-- 数据库初始化脚本
-- 微服务场景下的异常流量检测系统
-- ============================================================

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS `anomaly_detection`
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE `anomaly_detection`;

-- ============================================================
-- 1. 系统用户表
-- ============================================================
CREATE TABLE IF NOT EXISTS `sys_user` (
    `id`          BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `username`    VARCHAR(64)  NOT NULL                COMMENT '用户名（唯一）',
    `password`    VARCHAR(128) NOT NULL                COMMENT '密码（BCrypt加密存储）',
    `email`       VARCHAR(128) DEFAULT NULL            COMMENT '邮箱（可选）',
    `status`      TINYINT      NOT NULL DEFAULT 1      COMMENT '状态：0-冻结，1-启用',
    `role`        VARCHAR(32)  NOT NULL DEFAULT 'user' COMMENT '角色：admin-管理员，user-普通用户',
    `create_time` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统用户表';

-- 插入默认系统管理员账号，密码为 123456 的 BCrypt 加密值
INSERT INTO `sys_user` (`username`, `password`, `email`, `status`, `role`) VALUES
    ('admin', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBOsl7iKTVKIUi', NULL, 1, 'admin')
ON DUPLICATE KEY UPDATE `username` = VALUES(`username`);

-- ============================================================
-- 2. 检测任务表（实时流量模拟任务）
-- ============================================================
CREATE TABLE IF NOT EXISTS `detect_task` (
    `id`          BIGINT      NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `task_name`   VARCHAR(128) NOT NULL               COMMENT '任务名称',
    `status`      TINYINT     NOT NULL DEFAULT 0      COMMENT '任务状态：0-停止，1-运行中',
    `qps_setting` INT         NOT NULL DEFAULT 10     COMMENT 'QPS设置（每秒请求数）',
    `create_time` DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='检测任务表（实时流量模拟）';

-- 插入默认检测任务
INSERT INTO `detect_task` (`task_name`, `status`, `qps_setting`) VALUES
    ('默认实时检测任务', 0, 10);

-- ============================================================
-- 3. 检测结果表
-- ============================================================
CREATE TABLE IF NOT EXISTS `detect_result` (
    `id`              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `trace_id`        VARCHAR(64)  NOT NULL                COMMENT '链路追踪ID（唯一）',
    `source_ip`       VARCHAR(64)  DEFAULT NULL            COMMENT '来源IP地址',
    `timestamp`       BIGINT       DEFAULT NULL            COMMENT '请求时间戳（毫秒）',
    `rf_prob`         FLOAT        DEFAULT NULL            COMMENT 'Random Forest 预测概率',
    `gcn_prob`        FLOAT        DEFAULT NULL            COMMENT 'GCN 预测概率',
    `final_prob`      FLOAT        DEFAULT NULL            COMMENT '融合后最终概率',
    `is_anomaly`      TINYINT      NOT NULL DEFAULT 0      COMMENT '是否异常：0-正常，1-异常',
    `features_json`   JSON         DEFAULT NULL            COMMENT '原始特征数据（JSON格式）',
    `call_graph_json` JSON         DEFAULT NULL            COMMENT '调用链路图数据（JSON格式）',
    `detect_type`     TINYINT      NOT NULL DEFAULT 0      COMMENT '检测类型：0-实时检测，1-文件检测',
    `file_id`         BIGINT       DEFAULT NULL            COMMENT '关联文件ID（文件检测时使用）',
    `task_name`       VARCHAR(256) DEFAULT NULL            COMMENT '关联文件检测任务名',
    `create_time`     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_trace_id` (`trace_id`),
    KEY `idx_is_anomaly` (`is_anomaly`),
    KEY `idx_create_time` (`create_time`),
    KEY `idx_timestamp` (`timestamp`),
    KEY `idx_detect_type` (`detect_type`),
    KEY `idx_file_id` (`file_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='检测结果表';

-- ============================================================
-- 4. XAI 可解释性分析记录表
-- ============================================================
CREATE TABLE IF NOT EXISTS `xai_record` (
    `id`                   BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `trace_id`             VARCHAR(64)  NOT NULL                COMMENT '关联的链路追踪ID（唯一）',
    `status`               TINYINT      NOT NULL DEFAULT 0      COMMENT '分析状态：0-处理中，1-已完成',
    `lime_weights_json`    JSON         DEFAULT NULL            COMMENT 'LIME特征权重（JSON格式）',
    `abnormal_edges_json`  JSON         DEFAULT NULL            COMMENT 'GCN异常边信息（JSON格式）',
    `analysis_time`        DATETIME     DEFAULT NULL            COMMENT '分析完成时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_trace_id` (`trace_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='XAI可解释性分析记录表';

-- ============================================================
-- ============================================================
-- 5. 文件批量全量检测任务长时追踪表 (替代原有的单文件检测表)
-- ============================================================
CREATE TABLE IF NOT EXISTS `detect_file_task` (
    `id`             BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `task_name`      VARCHAR(256) NOT NULL                COMMENT '检测任务名',
    `csv_url`        VARCHAR(1024) DEFAULT NULL           COMMENT '特征表数据MinIO链接',
    `json_url`       VARCHAR(1024) DEFAULT NULL           COMMENT '调用链图数据MinIO链接',
    `status`         VARCHAR(32)  NOT NULL DEFAULT 'PENDING' COMMENT '检测状态：PENDING, DETECTING, COMPLETED, FAILED',
    `record_count`   INT          DEFAULT 0               COMMENT '总记录条数',
    `anomaly_count`  INT          DEFAULT 0               COMMENT '异常记录条数',
    `anomaly_rate`   FLOAT        DEFAULT 0               COMMENT '异常比例',
    `create_time`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '任务创建时间',
    `upload_time`    DATETIME     DEFAULT NULL            COMMENT '文件上传成功时间',
    `start_time`     DATETIME     DEFAULT NULL            COMMENT '开始检测时间',
    `end_time`       DATETIME     DEFAULT NULL            COMMENT '结束检测时间',
    `duration`       BIGINT       DEFAULT 0               COMMENT '检测耗时（毫秒）',
    PRIMARY KEY (`id`),
    KEY `idx_status` (`status`),
    KEY `idx_create_time` (`create_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='文件检测异步任务表';
