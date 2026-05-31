package com.anomaly;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 异常流量检测系统后端启动类
 */
@SpringBootApplication
@MapperScan("com.anomaly.mapper")
public class AnomalyDetectionApplication {

    public static void main(String[] args) {
        SpringApplication.run(AnomalyDetectionApplication.class, args);
    }
}
