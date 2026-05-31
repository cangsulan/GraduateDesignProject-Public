package com.anomaly.config;

import com.anomaly.common.BusinessException;
import com.anomaly.common.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * 全局异常处理器
 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    /**
     * 自定义业务异常拦截
     */
    @ExceptionHandler(BusinessException.class)
    public Result<?> handleBusinessException(BusinessException e) {
        log.warn("Business Exception: {}", e.getMessage());
        return Result.error(e.getCode(), e.getMessage());
    }

    /**
     * 运行时异常拦截（系统级错误）
     */
    @ExceptionHandler(Exception.class)
    public Result<?> handleException(Exception e) {
        log.error("System Exception: ", e);
        return Result.error(500, "系统内部错误：" + e.getMessage());
    }
}
