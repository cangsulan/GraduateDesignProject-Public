<template>
  <div class="history">
    <el-card shadow="hover" class="stat-card" style="margin-bottom: 20px;">
      <el-row :gutter="20">
        <el-col :span="8">
          <div class="stat-item">
            <div class="stat-title">数据库总落盘记录</div>
            <div class="stat-value primary-color">{{ stats.total }} <span class="unit">条</span></div>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="stat-item border-left border-right">
            <div class="stat-title">常态流量基座</div>
            <div class="stat-value success-color">{{ stats.normal }} <span class="unit">条</span></div>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="stat-item">
            <div class="stat-title">历史已拦截异常</div>
            <div class="stat-value danger-color">{{ stats.anomaly }} <span class="unit">条</span></div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <div class="chart-container" style="height: 500px">
      <div class="chart-header">宏观常态/异常流量比例全景环形图</div>
      <div ref="pieChartRef" style="width: 100%; height: 400px; margin-top: 20px;"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import request from '@/utils/request'
import * as echarts from 'echarts'

const stats = ref({ total: 0, normal: 0, anomaly: 0 })
const pieChartRef = ref()
let pieChart: echarts.ECharts | null = null

const loadStats = async () => {
  try {
    const res: any = await request.get('/detect/statistics')
    stats.value = res
    renderPieChart()
  } catch(e) {
    console.error(e)
  }
}

const renderPieChart = () => {
  if (!pieChartRef.value) return
  if (!pieChart) pieChart = echarts.init(pieChartRef.value)
  
  pieChart.setOption({
    tooltip: { trigger: 'item' },
    legend: { top: '5%', left: 'center' },
    color: ['#67c23a', '#f56c6c'],
    series: [
      {
        name: '流量占比',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 10,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: { show: false, position: 'center' },
        emphasis: {
          label: { show: true, fontSize: 40, fontWeight: 'bold' }
        },
        labelLine: { show: false },
        data: [
          { value: stats.value.normal, name: 'Normal 稳态流量' },
          { value: stats.value.anomaly, name: 'Anomaly 异常攻击流量' }
        ]
      }
    ]
  })
}

onMounted(() => {
  loadStats()
  window.addEventListener('resize', () => {
    pieChart?.resize()
  })
})
</script>

<style scoped>
.stat-card {
  padding: 10px 0;
}
.stat-item {
  text-align: center;
}
.border-left { border-left: 1px solid #ebeef5; }
.border-right { border-right: 1px solid #ebeef5; }
.stat-title {
  font-size: 14px;
  color: #909399;
  margin-bottom: 10px;
}
.stat-value {
  font-size: 36px;
  font-weight: 700;
}
.unit {
  font-size: 14px;
  font-weight: normal;
}
.primary-color { color: #409eff; }
.success-color { color: #67c23a; }
.danger-color { color: #f56c6c; }
.chart-header {
  font-size: 16px;
  font-weight: 600;
  color: #333;
}
</style>
