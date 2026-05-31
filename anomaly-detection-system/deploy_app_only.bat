@echo off
chcp 65001 >nul
echo 正在仅重新构建并部署业务微服务容器 (不重启中间件)...
echo 包含: backend, frontend, xai-worker, ai-engine, simulator

docker-compose stop backend frontend xai-worker ai-engine simulator
docker-compose rm -f backend frontend xai-worker ai-engine simulator
docker-compose build --no-cache backend frontend xai-worker ai-engine simulator
docker-compose up -d backend frontend xai-worker ai-engine simulator

echo 部署完成。系统访问地址：
echo [前端大屏 (Nginx)] http://localhost
echo [后端 API 网关]  http://localhost:8080/api
echo [MinIO 对象存储] http://localhost:9001 (root/rootpassword)
pause
