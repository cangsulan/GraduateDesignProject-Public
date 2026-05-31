import { defineStore } from 'pinia'
import { ref } from 'vue'
import request from '@/utils/request'

export interface AnomalyRecord {
  id: number
  traceId: string
  sourceIp: string
  timestamp: number
  rfProb: number
  gcnProb: number
  finalProb: number
  isAnomaly: number
  featuresJson: string
  callGraphJson: string
}

export const useDashboardStore = defineStore('dashboard', () => {
  // 定义大屏实时监控所需要的核心状态数据
  const alertCount = ref(0)
  const realtimeQps = ref(0)
  
  // 保留最新的 100 条实时检测记录用于表格和图表展示
  const latestRecords = ref<AnomalyRecord[]>([])
  
  // 用于大屏左上角的动态拦截量统计
  const totalAnomalyCount = ref(0)
  const totalDetectCount = ref(0)

  // 当前激活的时间范围，null表示“全部”
  const currentStartTime = ref<number | null>(null)
  const currentEndTime = ref<number | null>(null)

  const fetchBaseStats = async (startTime: number | null = null, endTime: number | null = null) => {
    currentStartTime.value = startTime
    currentEndTime.value = endTime
    try {
      // 1. 获取基础统计数据
      let statsUrl = '/detect/statistics'
      const params = []
      if (startTime) params.push(`startTime=${startTime}`)
      if (endTime) params.push(`endTime=${endTime}`)
      if (params.length > 0) statsUrl += '?' + params.join('&')
      
      const statsRes = await request.get(statsUrl) as any
      if (statsRes) {
        totalDetectCount.value = statsRes.total
        totalAnomalyCount.value = statsRes.anomaly
        alertCount.value = statsRes.anomaly // 对于回看历史，alertCount 等同于 anomalyCount
      }

      // 2. 获取最近 100 条历史记录
      let historyUrl = '/detect/history?current=1&size=100'
      if (startTime) historyUrl += `&startTime=${startTime}`
      if (endTime) historyUrl += `&endTime=${endTime}`
      
      const historyRes = await request.get(historyUrl) as any
      if (historyRes && historyRes.records) {
        latestRecords.value = historyRes.records
      }
    } catch (e) {
      console.error('Failed to fetch dashboard base stats', e)
    }
  }

  const addRecord = (record: AnomalyRecord) => {
    // 安全拦截：如果当前查看的是固定的历史记录快照（endTime 在过去），则不允许新的 WebSocket 数据插入！
    if (currentEndTime.value !== null && currentEndTime.value < Date.now() - 60000) {
      return
    }

    // 智能拦截：新加入的记录时间必须大于等于 startTime
    if (currentStartTime.value !== null && record.timestamp < currentStartTime.value) {
      return
    }

    // 插入到表头
    latestRecords.value.unshift(record)
    totalDetectCount.value++
    
    if (record.isAnomaly === 1) {
       totalAnomalyCount.value++
       alertCount.value++
    }
    
    // 如果超过100条则截断，确保前端性能及 ECharts 图表不崩溃
    if (latestRecords.value.length > 100) {
      latestRecords.value.pop()
    }
  }

  const clearRecords = () => {
    latestRecords.value = []
    totalAnomalyCount.value = 0
    totalDetectCount.value = 0
    alertCount.value = 0
    realtimeQps.value = 0
  }

  return { 
    alertCount, realtimeQps, 
    latestRecords, totalAnomalyCount, totalDetectCount,
    currentStartTime, currentEndTime,
    fetchBaseStats, addRecord, clearRecords
  }
})
