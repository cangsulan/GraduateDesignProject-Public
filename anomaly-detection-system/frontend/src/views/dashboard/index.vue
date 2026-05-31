<template>
  <div class="dashboard">
    <!-- 顶部时间筛选项 -->
    <div class="header-controls" style="margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
      <h2 style="margin: 0; font-size: 20px; color: #303133;">实时大屏与历史回溯</h2>
      <div style="display: flex; align-items: center;">
        <el-radio-group v-model="timeRange" @change="handleTimeRangeChange">
          <el-radio-button label="all">全部</el-radio-button>
          <el-radio-button label="1h">过去1小时</el-radio-button>
          <el-radio-button label="1d">过去1天</el-radio-button>
          <el-radio-button label="1w">过去1周</el-radio-button>
          <el-radio-button label="1m">过去1月</el-radio-button>
          <el-radio-button label="custom">自定义时间</el-radio-button>
        </el-radio-group>
        <el-date-picker 
          v-if="timeRange === 'custom'"
          v-model="customDateRange"
          type="datetimerange"
          range-separator="至"
          start-placeholder="开始时间"
          end-placeholder="结束时间"
          format="YYYY-MM-DD HH:mm:ss"
          value-format="x"
          @change="handleCustomDateChange"
          style="margin-left: 15px; width: 360px;"
        />
        <el-button type="primary" style="margin-left: 15px;" @click="refreshData">
          <el-icon><RefreshRight /></el-icon>
        </el-button>
      </div>
    </div>

    <!-- 顶部数据卡片 -->
    <el-row :gutter="20" class="top-cards">
      <el-col :span="6">
        <el-card shadow="hover" class="data-card info">
          <div class="card-icon"><el-icon><Monitor /></el-icon></div>
          <div class="card-content">
            <div class="card-title">总计检测流量</div>
            <div class="card-value">{{ dashboardStore.totalDetectCount }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="data-card warning">
          <div class="card-icon"><el-icon><Warning /></el-icon></div>
          <div class="card-content">
            <div class="card-title">拦截异常攻击</div>
            <div class="card-value">{{ dashboardStore.totalAnomalyCount }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="data-card danger">
          <div class="card-icon"><el-icon><Bell /></el-icon></div>
          <div class="card-content">
            <div class="card-title">实时告警 (Alerts)</div>
            <div class="card-value">{{ dashboardStore.alertCount }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="data-card primary">
          <div class="card-icon"><el-icon><Odometer /></el-icon></div>
          <div class="card-content">
            <div class="card-title">当前系统吞吐 QPS</div>
            <div class="card-value">{{ currentQps }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 图表区域 -->
    <el-row :gutter="20" class="chart-section" style="margin-top:20px;">
      <el-col :span="16">
        <div class="chart-container">
          <div class="chart-header">吞吐量(QPS)与异常拦截趋势</div>
          <div ref="lineChartRef" class="chart-body"></div>
        </div>
      </el-col>
      <el-col :span="8">
        <div class="chart-container">
          <div class="chart-header">最新检测最终置信概率</div>
          <div ref="gaugeChartRef" class="chart-body"></div>
        </div>
      </el-col>
    </el-row>

    <!-- 实时记录表格 -->
    <div class="chart-container" style="margin-top:20px;">
      <div class="chart-header">
        <span>实时检测滚动记录 (Latest 100)</span>
        <el-button type="primary" size="small" plain @click="dashboardStore.clearRecords()">清空屏幕</el-button>
      </div>
      <el-table 
        :data="dashboardStore.latestRecords" 
        style="width: 100%; height: 350px;" 
        stripe
        :row-class-name="tableRowClassName"
      >
        <el-table-column prop="traceId" label="Trace ID" width="220" show-overflow-tooltip />
        <el-table-column prop="sourceIp" label="源 IP" width="150" />
        <el-table-column label="RF 分析概率" width="120">
          <template #default="scope">
            {{ (scope.row.rfProb * 100).toFixed(2) }}%
          </template>
        </el-table-column>
        <el-table-column label="GCN 分析概率" width="120">
          <template #default="scope">
            {{ (scope.row.gcnProb * 100).toFixed(2) }}%
          </template>
        </el-table-column>
        <el-table-column label="融合决策概率" width="120">
          <template #default="scope">
            <strong>{{ (scope.row.finalProb * 100).toFixed(2) }}%</strong>
          </template>
        </el-table-column>
        <el-table-column label="综合判决" width="100">
          <template #default="scope">
            <el-tag :type="scope.row.isAnomaly ? 'danger' : 'success'">
              {{ scope.row.isAnomaly ? '异常攻击' : '正常' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="时间" align="right">
          <template #default="scope">
            {{ formatTime(scope.row.timestamp) }}
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { useDashboardStore } from '@/store/dashboard'
import dayjs from 'dayjs'
import { RefreshRight } from '@element-plus/icons-vue'

const dashboardStore = useDashboardStore()
const lineChartRef = ref<HTMLElement | null>(null)
const gaugeChartRef = ref<HTMLElement | null>(null)

// 时间筛选相关状态
const timeRange = ref('all')
const customDateRange = ref<number[] | null>(null)

const handleTimeRangeChange = (val: string) => {
  if (val === 'custom') return // custom 只有在具体选择日期后才生效
  
  const now = Date.now()
  let startTime: number | null = null
  let endTime: number | null = null // 自动跟随当前时间的通常不设 endTime，这里我们设定好时间距
  
  if (val === '1h') startTime = now - 3600 * 1000
  if (val === '1d') startTime = now - 24 * 3600 * 1000
  if (val === '1w') startTime = now - 7 * 24 * 3600 * 1000
  if (val === '1m') startTime = now - 30 * 24 * 3600 * 1000

  dashboardStore.fetchBaseStats(startTime, endTime)
}

const handleCustomDateChange = (val: number[] | null) => {
  if (val && val.length === 2) {
    dashboardStore.fetchBaseStats(val[0], val[1])
  } else {
    dashboardStore.fetchBaseStats(null, null)
  }
}

const refreshData = () => {
  if (timeRange.value === 'custom') {
    handleCustomDateChange(customDateRange.value)
  } else {
    handleTimeRangeChange(timeRange.value)
  }
}

let ws: WebSocket | null = null
let lineChart: echarts.ECharts | null = null
let gaugeChart: echarts.ECharts | null = null

// 用于计算实时QPS的辅助变量
const currentQps = ref(0)
let qpsCounter = 0
let qpsTimer: number

// 折线图数据
const timeData: string[] = []
const qpsData: number[] = []
const anomalyData: number[] = []
let currentSecondAnomaly = 0

// 初始化 ECharts
const initCharts = () => {
  if (lineChartRef.value) {
    lineChart = echarts.init(lineChartRef.value)
    
    // 初始化 60 秒的数据
    const now = new Date()
    for(let i=60; i>0; i--) {
      timeData.push(dayjs(now.getTime() - i * 1000).format('HH:mm:ss'))
      qpsData.push(0)
      anomalyData.push(0)
    }

    lineChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['吞吐(QPS)', '拦截异常量'], bottom: 0 },
      grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: timeData },
      yAxis: { type: 'value' },
      series: [
        {
          name: '吞吐(QPS)', type: 'line', smooth: true,
          itemStyle: { color: '#409eff' },
          areaStyle: { color: 'rgba(64,158,255,0.2)' },
          data: qpsData
        },
        {
          name: '拦截异常量', type: 'line', smooth: true,
          itemStyle: { color: '#f56c6c' },
          areaStyle: { color: 'rgba(245,108,108,0.2)' },
          data: anomalyData
        }
      ]
    })
  }

  if (gaugeChartRef.value) {
    gaugeChart = echarts.init(gaugeChartRef.value)
    updateGauge(0)
  }
}

const updateGauge = (prob: number) => {
  if (!gaugeChart) return
  gaugeChart.setOption({
    series: [
      {
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        center: ['50%', '75%'],
        radius: '90%',
        min: 0,
        max: 100,
        splitNumber: 5,
        axisLine: {
          lineStyle: {
            width: 15,
            color: [
              [0.5, '#67c23a'],
              [0.8, '#e6a23c'],
              [1, '#f56c6c']
            ]
          }
        },
        pointer: {
          icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
          length: '12%',
          width: 20,
          offsetCenter: [0, '-60%'],
          itemStyle: { color: 'inherit' }
        },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        detail: {
          fontSize: 30,
          offsetCenter: [0, '-10%'],
          valueAnimation: true,
          formatter: '{value}%',
          color: 'inherit'
        },
        data: [{ value: (prob * 100).toFixed(1), name: '异常概率' }]
      }
    ]
  })
}

const updateLineChart = () => {
  currentQps.value = qpsCounter
  
  timeData.shift()
  timeData.push(dayjs().format('HH:mm:ss'))
  
  qpsData.shift()
  qpsData.push(qpsCounter)
  
  anomalyData.shift()
  anomalyData.push(currentSecondAnomaly)
  
  if (lineChart) {
    lineChart.setOption({
      xAxis: { data: timeData },
      series: [{ data: qpsData }, { data: anomalyData }]
    })
  }
  
  // 重置下一秒计数器
  qpsCounter = 0
  currentSecondAnomaly = 0
}

// 初始化 WebSocket
const initWebSocket = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  // 开发环境下使用当前 host 因为 vite 配置了 ws proxy
  const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`
  
  ws = new WebSocket(wsUrl)
  
  ws.onopen = () => {
    console.log('大屏 WebSocket 连接成功')
  }
  
  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data)
      if (msg.type === 'detectResult') {
        const data = msg.data
        dashboardStore.addRecord(data)
        qpsCounter++
        if (data.isAnomaly === 1) {
          currentSecondAnomaly++
        }
        // 更新仪表盘
        updateGauge(data.finalProb)
      } else if (msg.type === 'xaiResult') {
        // xaiResult 可以用消息通知
        console.log('接收到 XAI 分析结果', msg.data)
      }
    } catch (e) {
      console.error('解析 WebSocket 消息失败', e)
    }
  }
  
  ws.onclose = () => {
    console.log('WebSocket 断开，两秒后重连...')
    setTimeout(initWebSocket, 2000)
  }
}

const tableRowClassName = ({row}: {row: any}) => {
  if (row.isAnomaly === 1) {
    return 'danger-row'
  }
  return ''
}

const formatTime = (ts: number | null) => {
  if (!ts) return '-'
  return dayjs(ts).format('YYYY-MM-DD HH:mm:ss')
}

// 页面加载阶段
onMounted(async () => {
  initCharts()
  // 首先拉取当前统计基数，然后再连接 WebSocket 将新的叠加
  await dashboardStore.fetchBaseStats()
  initWebSocket()
  
  // 定时器：每秒刷新折线图并计算 QPS
  qpsTimer = window.setInterval(updateLineChart, 1000)
  
  window.addEventListener('resize', () => {
    lineChart?.resize()
    gaugeChart?.resize()
  })
})

onUnmounted(() => {
  clearInterval(qpsTimer)
  lineChart?.dispose()
  gaugeChart?.dispose()
  if (ws) ws.close()
})
</script>

<style scoped lang="scss">
.dashboard {
  .top-cards {
    .data-card {
      border: none;
      border-radius: 12px;
      color: #fff;
      
      :deep(.el-card__body) {
        display: flex;
        align-items: center;
        padding: 20px;
      }
      
      &.info {
        background: linear-gradient(135deg, #3b82f6, #60a5fa);
      }
      &.warning {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
      }
      &.danger {
        background: linear-gradient(135deg, #ef4444, #f87171);
      }
      &.primary {
        background: linear-gradient(135deg, #10b981, #34d399);
      }
      
      .card-icon {
        font-size: 48px;
        opacity: 0.8;
        margin-right: 20px;
      }
      
      .card-content {
        flex: 1;
        text-align: right;
        
        .card-title {
          font-size: 14px;
          opacity: 0.9;
          margin-bottom: 8px;
        }
        
        .card-value {
          font-size: 32px;
          font-weight: 700;
        }
      }
    }
  }

  .chart-header {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #333;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .chart-body {
    height: 350px;
    width: 100%;
  }
}

:deep(.danger-row) {
  --el-table-tr-bg-color: rgba(245, 108, 108, 0.08);
}
</style>
