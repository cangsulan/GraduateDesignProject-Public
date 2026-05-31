# ============================================================
# Frontend Dockerfile
# Vue3 编译及 Nginx 承载多阶段构建
# ============================================================

# 构建阶段
FROM node:20-alpine as build-stage
WORKDIR /app
COPY package*.json ./
RUN npm config set registry https://registry.npmmirror.com/
RUN npm install
COPY . .
RUN npm run build

# 生产环境阶段
FROM nginx:stable-alpine as production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
