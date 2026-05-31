# 微服务异常流量检测系统

## 项目简介

本项目是一套面向微服务架构的**微服务异常流量检测平台**，采用前后端分离与微服务架构设计，融合**随机森林（RF）**与**图卷积网络（GCN）**检测模型，并提出**置信度动态融合方法**，实现了对微服务 API 流量的实时检测、文件批量检测与可解释性溯源分析，以及流量检索管理等主要功能。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + TypeScript + Element Plus + ECharts + Pinia + Vue Router |
| 后端 | Spring Boot 3+ Spring Security + MyBatis+ WebSocket |
| AI 引擎 | Python + FastAPI + PyTorch + PyTorch Geometric + scikit-learn + LIME |
| 消息队列 | RabbitMQ |
| 数据库 | MySQL 8.0 |
| 缓存 | Redis |
| 对象存储 | MinIO |
| 容器化 | Docker + Docker Compose |

---

## 项目结构

```
GraduateDesignProject/
├── anomaly-detection-system/          # 核心系统代码
│   ├── ai_engine/                     # 核心流量检测模块（Python）
│   ├── backend/                       # Java 后端管理服务
│   ├── frontend/                      # Vue 前端应用
│   ├── simulator/                     # 流量模拟器
│   ├── infrastructure/                # 基础设施配置
│   ├── docker-compose.yml             # 容器编排配置
│   ├── .env                           # 全局环境变量
│   └── deploy.bat                     # 一键部署脚本
├── 原始数据/                           # 原始 PCAP 数据集
│   ├── abnormal/                      # 异常流量样本
│   └── normal/                        # 正常流量样本
├── 数据集/                             # kaggle数据集
|   ├── ....
├── .gitignore
└── readme.md
```

---

## 各模块详细说明

### 1. AI 推理引擎（ai_engine/）

AI 引擎是系统的核心检测模块，基于 FastAPI 构建，负责模型加载、双模型融合推理与可解释性分析。

```
ai_engine/
├── main.py                    # FastAPI 主入口，推理 API 与 RabbitMQ 消费者
├── train_models.py            # 离线模型训练脚本（RF + GCN）
├── confidence_fusion.py       # 置信度自适应融合策略模块
├── xai_worker.py              # XAI 可解释性分析 Worker（LIME + GNNExplainer）
├── requirements.txt           # Python 依赖
├── Dockerfile                 # 容器构建文件
├── dataset/                   # 训练与测试数据集
├── traffic_processor/         # PCAP 流量处理模块
│   ├── __init__.py            # 模块导出
│   ├── models.py              # 数据结构定义（HTTPRequest, FeatureRecord, CallGraph 等）
│   ├── pcap_parser.py         # PCAP 文件解析器
│   ├── http_extractor.py      # HTTP 请求提取器
│   ├── feature_calculator.py  # 特征计算器（含拓扑特征提取）
│   ├── call_graph_builder.py  # 微服务调用图构建器
│   └── processor.py           # 统一处理入口
└── scripts/                   # 开发过程中产生的实验脚本（对比实验、网格搜索、融合改进等）
```

**核心功能**：

- **双模型推理流水线**（`infer_pipeline`）：接收特征字典与调用图，分别通过 RF 和 GCN 模型推理，再经融合策略输出最终判定结果
- **置信度自适应融合**（`confidence_adaptive_fusion`）：根据 RF/GCN 各自的置信度与预测一致性，动态调整融合权重与判定阈值
- **固定权重融合**（`fixed_weight_fusion`）：基线融合方法，用于对比实验
- **RabbitMQ 消费者**：监听 `q.detect.req` 队列，接收 Java 后端转发的检测请求
- **批量检测 API**（`/api/batch_detect`）：供 Java 后端批量调用的 HTTP 接口
- **离线文件检测**（`/api/predict/task`）：异步处理大体积文件检测任务，完成后 HTTP 回调
- **PCAP 直通检测**（`/api/predict/from-pcap`）：接收 PCAP 文件，内存直通特征提取与检测
- **XAI 可解释性分析**（`xai_worker.py`）：
  - **LIME**：分析 RF 模型的特征权重，解释哪些特征对异常判定贡献最大
  - **GNNExplainer**：定位 GCN 模型中的异常调用边，识别拓扑结构中的异常路径

**模型架构**：

- **RF 模型**：RandomForestClassifier，使用 7 个数值特征 + 2 个类别特征，经 SMOTE 处理数据不平衡
- **GCN 双层模型**：2 层 GCNConv + global_mean_pool + Linear，用于有特征时的双模型融合
- **GCN 单层模型**：1 层 GCNConv + global_mean_pool + Linear，用于无特征时的降级模式



---

### 2. Java 后端服务（backend/）

基于 Spring Boot 3 构建的后端服务，负责业务逻辑、数据持久化、消息路由与 WebSocket 实时推送。

```
backend/
├── pom.xml                    # Maven 依赖配置
├── Dockerfile                 # 容器构建文件
└── src/main/java/com/anomaly/
    ├── AnomalyDetectionApplication.java   # Spring Boot 启动类
    ├── common/                # 通用工具
    │   ├── Result.java        # 统一响应封装
    │   └── BusinessException.java  # 业务异常
    ├── config/                # 配置类
    │   ├── CorsConfig.java    # 跨域配置
    │   ├── GlobalExceptionHandler.java  # 全局异常处理
    │   ├── MinioConfig.java   # MinIO 配置
    │   └── MybatisPlusConfig.java  # MyBatis-Plus 分页配置
    ├── controller/            # REST 控制器
    │   ├── AuthController.java        # 认证接口（登录/注册）
    │   ├── DetectResultController.java # 检测结果查询与统计
    │   ├── DetectFileTaskController.java # 文件检测任务管理
    │   ├── SimulatorController.java   # 流量模拟器控制
    │   ├── UserController.java        # 用户管理
    │   └── XaiController.java         # XAI 可解释性分析接口
    ├── entity/                # 数据实体
    │   ├── DetectResult.java   # 检测结果实体
    │   ├── DetectTask.java     # 检测任务实体
    │   ├── DetectFileTask.java # 文件检测任务实体
    │   ├── SysUser.java        # 系统用户实体
    │   └── XaiRecord.java      # XAI 分析记录实体
    ├── mapper/                # MyBatis Mapper 接口
    ├── mq/                    # 消息队列
    │   ├── RabbitMQConfig.java  # RabbitMQ 交换机/队列/绑定配置
    │   ├── RabbitProducer.java  # 消息生产者
    │   └── RabbitConsumer.java  # 消息消费者（检测结果 + XAI 结果）
    ├── security/              # 安全模块
    │   ├── JwtUtils.java       # JWT 工具类
    │   ├── JwtAuthenticationFilter.java  # JWT 认证过滤器
    │   └── SecurityConfig.java # Spring Security 配置
    ├── service/               # 业务服务层
    │   ├── DetectResultService.java
    │   ├── DetectTaskService.java
    │   ├── DetectFileTaskService.java
    │   ├── MinioService.java   # MinIO 对象存储服务
    │   ├── SysUserService.java
    │   ├── XaiRecordService.java
    │   └── impl/              # 服务实现类
    └── ws/                    # WebSocket
        ├── WebSocketConfig.java  # WebSocket 配置
        └── WebSocketServer.java  # 实时推送端点（/ws/dashboard）
```

**核心功能**：

- **认证鉴权**：基于 JWT 的无状态认证，支持登录/注册/角色权限控制
- **实时检测流程**：模拟器 → HTTP 接收 → RabbitMQ 转发 → AI 引擎检测 → 结果回传 → WebSocket 推送前端
- **文件检测流程**：文件上传 MinIO → 调用 AI 引擎异步检测 → HTTP 回调 → 结果批量入库
- **PCAP 检测流程**：PCAP 文件上传 → 转发 AI 引擎内存直通处理 → 回调入库
- **XAI 溯源流程**：前端触发分析请求 → RabbitMQ 转发 XAI Worker → LIME/GNNExplainer 分析 → 结果回传入库
- **WebSocket 实时推送**：检测结果与 XAI 分析结果实时推送到前端大屏
- **统计缓存**：使用 Redis 缓存统计数据（10 秒过期），保护数据库免受大屏轮询压力

**RabbitMQ 消息路由**：

| 队列 | 路由键 | 方向 | 说明 |
|------|--------|------|------|
| q.detect.req | detect.req | Backend → AI Engine | 检测请求 |
| q.detect.res | detect.res | AI Engine → Backend | 检测结果 |
| q.xai.req | xai.req | Backend → XAI Worker | XAI 分析请求 |
| q.xai.res | xai.res | XAI Worker → Backend | XAI 分析结果 |
| q.simulator.control | simulator.control | Backend → Simulator | 模拟器控制指令 |

---

### 3. Vue 前端应用（frontend/）

基于 Vue 3 + TypeScript 构建的单页面应用，提供可视化监控大屏与系统管理界面。

```
frontend/
├── package.json               # NPM 依赖配置
├── vite.config.ts             # Vite 构建配置
├── index.html                 # 入口 HTML
├── nginx.conf                 # Nginx 配置（生产环境）
├── Dockerfile                 # 容器构建文件
└── src/
    ├── main.ts                # 应用入口
    ├── App.vue                # 根组件
    ├── assets/styles/         # 全局样式
    ├── components/            # 公共组件
    ├── layout/                # 布局组件
    │   ├── index.vue          # 主布局（侧边栏 + 顶栏 + 内容区）
    │   └── SimulatorControl.vue  # 流量模拟器控制条
    ├── router/index.ts        # 路由配置与权限守卫
    ├── store/                 # Pinia 状态管理
    │   ├── user.ts            # 用户状态（Token/角色）
    │   └── dashboard.ts       # 大屏状态
    ├── utils/request.ts       # Axios 请求封装
    └── views/                 # 页面视图
        ├── login/index.vue    # 登录页
        ├── dashboard/index.vue # 实时监控大屏
        ├── file/index.vue     # 文件批量检测
        ├── anomaly/index.vue  # 异常记录与溯源
        ├── history/index.vue  # 综合统计分析
        ├── user/index.vue     # 用户权限管理（仅 admin）
        └── profile/index.vue  # 个人信息
```

**页面功能**：

| 页面 | 路由 | 功能 |
|------|------|------|
| 实时监控大屏 | /dashboard | WebSocket 实时展示检测结果、异常告警、统计图表 |
| 文件批量检测 | /file-detect | 上传 CSV/JSON/PCAP 文件，创建批量检测任务 |
| 异常记录与溯源 | /anomalies | 检测结果多维度查询、XAI 可解释性分析 |
| 用户权限管理 | /users | 用户增删改查与角色管理（仅 admin 可见） |
| 个人信息 | /profile | 用户个人信息查看与修改 |

---

### 4. 流量模拟器（simulator/）

独立的 Python 服务，用于模拟微服务 API 调用流量，支持 QPS 配置与启停控制。

```
simulator/
├── simulator.py               # 模拟器主程序
├── requirements.txt           # Python 依赖
└── Dockerfile                 # 容器构建文件
```

**核心功能**：

- 从预置数据集中读取特征与调用图数据
- 按 QPS 设定向 Java 后端发送模拟流量请求
- 通过 RabbitMQ 接收启停/速率调节控制指令
- 数据集读完后自动停止模拟

---

### 5. 基础设施（infrastructure/）

```
infrastructure/
└── init.sql                   # MySQL 数据库初始化脚本
```

**数据库表结构**：

| 表名 | 说明 |
|------|------|
| sys_user | 系统用户表（BCrypt 加密存储密码） |
| detect_task | 检测任务表（实时流量模拟任务状态与 QPS 配置） |
| detect_result | 检测结果表（RF/GCN 概率、融合概率、异常标记、特征与图数据 JSON） |
| xai_record | XAI 可解释性分析记录表（LIME 特征权重、GCN 异常边信息） |
| detect_file_task | 文件批量检测任务表（任务状态追踪、MinIO 文件链接、统计信息） |

---

## 快速部署

### 环境要求

- Docker & Docker Compose
- NVIDIA GPU + nvidia-container-toolkit（用于 GPU 加速推理）

### 一键部署

```bash
cd anomaly-detection-system
deploy.bat
```
