package com.anomaly.mq;

import org.springframework.amqp.core.*;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * RabbitMQ 配置类
 * 统一使用 Topic Exchange: anomaly.topic
 */
@Configuration
public class RabbitMQConfig {

    public static final String EXCHANGE_NAME = "anomaly.topic";

    // 队列名称
    public static final String QUEUE_SIMULATOR_CTRL = "q.simulator.control";
    public static final String QUEUE_DETECT_REQ = "q.detect.req";
    public static final String QUEUE_DETECT_RES = "q.detect.res";
    public static final String QUEUE_XAI_REQ = "q.xai.req";
    public static final String QUEUE_XAI_RES = "q.xai.res";

    // 路由键
    public static final String ROUTING_KEY_SIM_CTRL = "sim.ctrl";
    public static final String ROUTING_KEY_DETECT_REQ = "detect.req";
    public static final String ROUTING_KEY_DETECT_RES = "detect.res";
    public static final String ROUTING_KEY_XAI_REQ = "xai.req";
    public static final String ROUTING_KEY_XAI_RES = "xai.res";

    /**
     * 定义主题交换机
     */
    @Bean
    public TopicExchange anomalyExchange() {
        return new TopicExchange(EXCHANGE_NAME);
    }

    /** 定义队列 */
    @Bean
    public Queue simulatorControlQueue() {
        return new Queue(QUEUE_SIMULATOR_CTRL, true);
    }

    @Bean
    public Queue detectReqQueue() {
        return new Queue(QUEUE_DETECT_REQ, true);
    }

    @Bean
    public Queue detectResQueue() {
        return new Queue(QUEUE_DETECT_RES, true);
    }

    @Bean
    public Queue xaiReqQueue() {
        return new Queue(QUEUE_XAI_REQ, true);
    }

    @Bean
    public Queue xaiResQueue() {
        return new Queue(QUEUE_XAI_RES, true);
    }

    /** 绑定队列到交换机 */
    @Bean
    public Binding bindingSimulatorCtrl() {
        return BindingBuilder.bind(simulatorControlQueue()).to(anomalyExchange()).with(ROUTING_KEY_SIM_CTRL);
    }

    @Bean
    public Binding bindingDetectReq() {
        return BindingBuilder.bind(detectReqQueue()).to(anomalyExchange()).with(ROUTING_KEY_DETECT_REQ);
    }

    @Bean
    public Binding bindingDetectRes() {
        return BindingBuilder.bind(detectResQueue()).to(anomalyExchange()).with(ROUTING_KEY_DETECT_RES);
    }

    @Bean
    public Binding bindingXaiReq() {
        return BindingBuilder.bind(xaiReqQueue()).to(anomalyExchange()).with(ROUTING_KEY_XAI_REQ);
    }

    @Bean
    public Binding bindingXaiRes() {
        return BindingBuilder.bind(xaiResQueue()).to(anomalyExchange()).with(ROUTING_KEY_XAI_RES);
    }
}
