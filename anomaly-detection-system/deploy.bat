@echo off
chcp 65001 >nul
echo 正在停止并清理旧容器...
docker-compose down

echo 正在拉取镜像并构建服务（强制跳过缓存）...
docker-compose build --no-cache

echo 正在启动所有服务...
docker-compose up -d

echo 部署完成。系统访问地址：
echo [前端大屏 (Nginx)] http://localhost
echo [后端 API 网关]  http://localhost:8080/api
echo [RabbitMQ 管理]  http://localhost:15672 (admin/admin123)
echo [MinIO 对象存储] http://localhost:9001 (root/rootpassword)
pause
