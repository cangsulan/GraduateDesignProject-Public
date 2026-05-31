# ============================================================
# Simulator Dockerfile
# 基于 Python 3.10 构建流量模拟器运行环境
# ============================================================

FROM python:3.10-slim

WORKDIR /app

# 设置时区和编码
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码
COPY . .

CMD ["python", "simulator.py"]
