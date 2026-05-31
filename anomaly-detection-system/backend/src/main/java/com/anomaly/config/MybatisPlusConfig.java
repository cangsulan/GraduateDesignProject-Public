package com.anomaly.config;

import com.baomidou.mybatisplus.annotation.DbType;
import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * MyBatis-Plus 核心配置类
 */
@Configuration
public class MybatisPlusConfig {

    /**
     * 添加分页插件
     * 没有这个插件，.page() 方法由于无法自动拼接 LIMIT 将触发致命全量查询降级
     */
    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor() {
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
        // 如果您的数据库层是 MySQL，配置 DbType.MYSQL 即可优化 COUNT 查询与 LIMIT 拼接
        interceptor.addInnerInterceptor(new PaginationInnerInterceptor(DbType.MYSQL));
        return interceptor;
    }
}
