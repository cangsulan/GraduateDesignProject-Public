package com.anomaly.ws;

import com.alibaba.fastjson2.JSON;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import jakarta.websocket.*;
import jakarta.websocket.server.ServerEndpoint;
import java.io.IOException;
import java.util.concurrent.CopyOnWriteArraySet;

/**
 * 实时异常监控大屏推流 WebSocket 端点
 */
@Slf4j
@Component
@ServerEndpoint("/ws/dashboard")
public class WebSocketServer {

    // 静态变量，用来记录当前在线连接数
    private static int onlineCount = 0;
    // 线程安全 Set，用来存放每个客户端对应的 WebSocketServer 对象
    private static CopyOnWriteArraySet<WebSocketServer> webSocketSet = new CopyOnWriteArraySet<>();
    // 与某个客户端的连接会话
    private Session session;

    /**
     * 连接建立成功调用的方法
     */
    @OnOpen
    public void onOpen(Session session) {
        this.session = session;
        webSocketSet.add(this);
        addOnlineCount();
        log.info("有新大屏连接加入！当前在线大屏数为" + getOnlineCount());
    }

    /**
     * 连接关闭调用的方法
     */
    @OnClose
    public void onClose() {
        webSocketSet.remove(this);
        subOnlineCount();
        log.info("一大屏连接关闭！当前在线大屏数为" + getOnlineCount());
    }

    /**
     * 收到客户端消息后调用的方法
     */
    @OnMessage
    public void onMessage(String message, Session session) {
        log.info("收到来自大屏的消息: " + message);
        // 可选：响应心跳 pingpong
    }

    @OnError
    public void onError(Session session, Throwable error) {
        log.error("大屏WebSocket发生错误");
        error.printStackTrace();
    }

    /**
     * 服务器主动发送消息
     */
    public void sendMessage(String message) throws IOException {
        synchronized (this.session) {
            this.session.getBasicRemote().sendText(message);
        }
    }

    /**
     * 群发实时流量或结果检测数据
     */
    public static void broadcastMessage(Object obj) {
        String msg = JSON.toJSONString(obj);
        for (WebSocketServer item : webSocketSet) {
            try {
                item.sendMessage(msg);
            } catch (IOException e) {
                log.error("群发大屏消息失败");
            }
        }
    }

    public static synchronized int getOnlineCount() {
        return onlineCount;
    }

    public static synchronized void addOnlineCount() {
        WebSocketServer.onlineCount++;
    }

    public static synchronized void subOnlineCount() {
        WebSocketServer.onlineCount--;
    }
}
