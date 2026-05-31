# ============================================================
# Python AI Engine Dockerfile (GPU 加速版 - RTX 5060 优化)
# 基于 PyTorch 2.7 官方 CUDA 12.8 环境，支持 Blackwell 架构 (sm_120)
# ============================================================

FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# 设置时区和编码
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖配置
COPY requirements.txt .

# 过滤掉 base img 已自带的 torch，避免冲突重装导致 GPU 失效
RUN grep -v '^torch==' requirements.txt > req.tmp && mv req.tmp requirements.txt

# 使用国内源加速安装基础包
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 为 RTX 5060 (Blackwell架构) 安装匹配 CUDA 12.8 的 PyG 扩展组件
RUN pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 复制代码与离线训练好的模型
COPY . .

# 暴露 FastAPI 端口
EXPOSE 8000

# 默认启动命令
CMD ["python", "main.py"]
