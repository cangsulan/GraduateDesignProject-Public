# ============================================================
# Java Backend Dockerfile
# 多阶段构建以优化镜像体积
# ============================================================

# 第一阶段：Maven 构建
FROM maven:3.9.5-eclipse-temurin-17 AS builder
WORKDIR /app
COPY pom.xml .
COPY src ./src
# 忽略测试，并行加速构建
RUN mvn clean package -DskipTests -T 1C

# 第二阶段：运行环境
FROM eclipse-temurin:17-jre
WORKDIR /app

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 创建上传文件存放目录
RUN mkdir -p /tmp/uploads

# 复制打包好的 JAR
COPY --from=builder /app/target/anomaly-backend-1.0.0.jar app.jar

# 暴露后端端口
EXPOSE 8080

# 启动命令
ENTRYPOINT ["java", "-jar", "app.jar"]
