<template>
  <div class="anomaly-records">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <span>历史请求溯源与分析管理</span>
          <el-button type="success" :icon="Refresh" circle @click="loadData" />
        </div>
      </template>

      <!-- 多条件筛选栏 -->
      <el-form :inline="true" :model="searchQuery" class="filter-form" style="margin-bottom: 12px;" label-width="auto">
        <el-form-item label="Trace ID">
          <el-input v-model="searchQuery.traceId" placeholder="按ID搜索" clearable style="width: 160px;" @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item label="来源IP">
          <el-input v-model="searchQuery.sourceIp" placeholder="按IP筛查" clearable style="width: 140px;" @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item label="置信度">
          <div style="display: flex; align-items: center; gap: 4px;">
            <el-input-number v-model="searchQuery.minProb" :min="0" :max="100" :precision="2" :step="5" controls-position="right" placeholder="最低" style="width: 100px;" size="default" />
            <span>%&nbsp;—&nbsp;</span>
            <el-input-number v-model="searchQuery.maxProb" :min="0" :max="100" :precision="2" :step="5" controls-position="right" placeholder="最高" style="width: 100px;" size="default" />
            <span>%</span>
          </div>
        </el-form-item>
        <el-form-item label="检测模式">
          <el-select v-model="searchQuery.detectType" placeholder="全部" clearable style="width: 120px;">
            <el-option label="实时流" :value="0" />
            <el-option label="文件批处理" :value="1" />
          </el-select>
        </el-form-item>
        <el-form-item label="任务ID" v-if="searchQuery.detectType === 1">
          <el-input-number v-model="searchQuery.fileId" :min="1" controls-position="right" placeholder="任务ID" style="width: 110px;" size="default" />
        </el-form-item>
        <el-form-item label="任务名称" v-if="searchQuery.detectType === 1">
          <el-input v-model="searchQuery.taskName" placeholder="按任务名搜索" clearable style="width: 140px;" @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item label="判决结果">
          <el-select v-model="searchQuery.isAnomaly" placeholder="全部" clearable style="width: 110px;">
            <el-option label="异常攻击" :value="1" />
            <el-option label="正常" :value="0" />
          </el-select>
        </el-form-item>
        <el-form-item label="发生时间">
          <el-date-picker
            v-model="searchQuery.dateRange"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
            value-format="x"
            style="width: 340px;"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">查询</el-button>
          <el-button @click="resetSearch">重置</el-button>
        </el-form-item>
      </el-form>

      <el-table :data="tableData" v-loading="loading" style="width: 100%" stripe>
        <el-table-column prop="traceId" label="Trace ID" width="220" />
        <el-table-column prop="sourceIp" label="请求来源 IP" width="150" />
        <el-table-column label="检测置信度" width="120">
          <template #default="{ row }">
            {{ (row.finalProb * 100).toFixed(2) }}%
          </template>
        </el-table-column>
        <el-table-column label="检测模式" width="100">
          <template #default="{ row }">
            <el-tag size="small" :type="row.detectType === 0 ? 'primary' : 'warning'">
              {{ row.detectType === 0 ? '实时流' : '文件批处理' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="综合判决" width="120">
          <template #default="{ row }">
            <el-tag :type="row.isAnomaly ? 'danger' : 'success'">
              {{ row.isAnomaly ? '异常攻击' : '正常' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="发生时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.timestamp) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" fixed="right" min-width="150">
          <template #default="{ row }">
            <el-button type="primary" size="small" @click="openXaiModal(row.traceId)" :icon="Search">
              发起/查看深度溯源 (XAI)
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <div style="margin-top: 20px; display: flex; justify-content: flex-end;">
         <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next, jumper"
          :total="total"
          @size-change="loadData"
          @current-change="loadData"
        />
      </div>
    </el-card>

    <!-- XAI 结果弹窗 (系统参数位置: width="90%" 控制左右跨度，top="5vh" 控制距离顶部多高，调小顶边距即可整体拔高垂直视窗) -->
    <el-dialog v-model="xaiDialogVisible" title="XAI 多模态深度溯源大盘" width="90%" top="5vh" :before-close="handleClose">
      <div class="xai-content" v-loading="xaiLoading" element-loading-text="解析引擎正在狂奔计算 LIME & PyG 结果中...">
        <div v-if="xaiStatus === 0">
          <el-skeleton :rows="5" animated />
          <div style="text-align: center; margin-top: 30px; color: #909399;">
            <p>任务正在排队处理。RabbitMQ -> Python Worker -> LIME/GCNExplainer_100Epoch</p>
            <el-button @click="refreshXai" type="primary" plain size="small" style="margin-top: 10px;">点击刷新进度</el-button>
          </div>
        </div>
        
        <div v-else-if="xaiStatus === 1">
          <el-row :gutter="20">
            <!-- LIME 左侧柱状图 -->
            <el-col :span="12">
              <el-card shadow="hover">
                <template #header><strong>LIME 随机森林树特征权重正负向分析</strong></template>
                <!-- 【系统参数位置】: 外层 div 控制“视窗”大小，修改此处 height=580px 能把整个框拉高 -->
                <!-- overflow: auto; 这个强制允许当内容超出外框时浏览器渲染原生上下、左右滑动条，而不是让里面挤扁 -->
                <div style="width: 95%; height: 600px; border: 1px solid #ebeef5; border-radius: 4px;">
                  <!-- 【系统参数位置】: 里层 div 是“画布实际宽高”。宽(width)越大字越拉拉伸不开；高(height)越大柱子越空疏 -->
                  <div ref="limeChartRef" style="height: 98%; width: 98%; padding: 8px; /* overflow: auto; */"></div>
                </div>
              </el-card>
            </el-col>
            
            <!-- PyG 拓扑异常边 右侧 -->
            <el-col :span="12">
              <el-card shadow="hover">
                <template #header>
                  <strong>GNNExplainer 空间拓扑异常链路分析</strong>
                  <div style="font-size: 12px; color: #909399; margin-top: 5px;">通过掩码矩阵挖掘子图贡献得出最高危边</div>
                </template>
                <!-- 【系统参数位置】: 修改此处 height: 580px 让他和左侧外面的 580px 保持齐平 -->
                <div ref="gcnChartRef" style="height: 600px; width: 100%;"></div>
              </el-card>
            </el-col>
          </el-row>
        </div>
        
        <div v-else>
          <el-empty description="暂无分析数据或尚未发起分析" />
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, nextTick } from 'vue'
import { Search, Refresh } from '@element-plus/icons-vue'
import request from '@/utils/request'
import dayjs from 'dayjs'
import * as echarts from 'echarts'

const tableData = ref([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

const searchQuery = reactive({
  traceId: '',
  sourceIp: '',
  minProb: undefined as number | undefined,
  maxProb: undefined as number | undefined,
  detectType: undefined as number | undefined,
  isAnomaly: undefined as number | undefined,
  dateRange: null as number[] | null,
  fileId: undefined as number | undefined,
  taskName: ''
})

const handleSearch = () => {
  currentPage.value = 1
  loadData()
}

const resetSearch = () => {
  searchQuery.traceId = ''
  searchQuery.sourceIp = ''
  searchQuery.minProb = undefined
  searchQuery.maxProb = undefined
  searchQuery.detectType = undefined
  searchQuery.isAnomaly = undefined
  searchQuery.dateRange = null
  searchQuery.fileId = undefined
  searchQuery.taskName = ''
  currentPage.value = 1
  loadData()
}

// 获取列表数据
const loadData = async () => {
  loading.value = true
  try {
    const params: any = { current: currentPage.value, size: pageSize.value }
    if (searchQuery.traceId) params.traceId = searchQuery.traceId
    if (searchQuery.sourceIp) params.sourceIp = searchQuery.sourceIp
    if (searchQuery.isAnomaly !== undefined && searchQuery.isAnomaly !== null) params.isAnomaly = searchQuery.isAnomaly
    if (searchQuery.detectType !== undefined && searchQuery.detectType !== null) params.detectType = searchQuery.detectType
    if (searchQuery.minProb !== undefined && searchQuery.minProb !== null) params.minProb = searchQuery.minProb / 100
    if (searchQuery.maxProb !== undefined && searchQuery.maxProb !== null) params.maxProb = searchQuery.maxProb / 100
    if (searchQuery.dateRange && searchQuery.dateRange.length === 2) {
      params.startTime = searchQuery.dateRange[0]
      params.endTime = searchQuery.dateRange[1]
    }
    if (searchQuery.fileId !== undefined && searchQuery.fileId !== null) params.fileId = searchQuery.fileId
    if (searchQuery.taskName) params.taskName = searchQuery.taskName

    const res: any = await request.get('/detect/history', { params })
    tableData.value = res.records
    total.value = res.total
  } finally {
    loading.value = false
  }
}

// 格式化时间
const formatTime = (ts: number) => {
  if (!ts) return '-'
  return dayjs(ts).format('YYYY-MM-DD HH:mm:ss.SSS')
}

// ------------------- XAI 逻辑 -------------------
const xaiDialogVisible = ref(false)
const xaiLoading = ref(false)
const currentTraceId = ref('')
const xaiStatus = ref(-1) // -1 未知, 0 处理中, 1 完成
const abnormalEdgesData = ref([])

let pollTimer: any = null // 新增定时器引用

const limeChartRef = ref()
let limeChart: echarts.ECharts | null = null

const gcnChartRef = ref()
let gcnChart: echarts.ECharts | null = null

const openXaiModal = async (traceId: string) => {
  currentTraceId.value = traceId
  xaiDialogVisible.value = true
  xaiStatus.value = -1 // 重置
  
  // 发起/查询分析请求
  try {
    await request.post(`/xai/analyze/${traceId}`)
    await refreshXai()
  } catch(e) {
    console.error(e)
  }
}

const refreshXai = async () => {
  xaiLoading.value = true
  try {
    const res: any = await request.get(`/xai/result/${currentTraceId.value}`)
    // 如果 code 为 202 (我们在后端定的是 202 表示未完成/错误)
    // 根据 Axios 拦截器，错误被 Promise.reject，但如果是成功包则返回 data。
    // 这里我们因为返回的格式是被去皮了，得看 request.ts 配置。
    // 由于如果是 202 error 会被吃掉弹窗，我们用 try-catch 继续：
    if (res) {
      xaiStatus.value = res.status
      if (res.status === 1) {
        // 完成
        const limeWeights = JSON.parse(res.limeWeightsJson || '{}')
        const edges = JSON.parse(res.abnormalEdgesJson || '[]')
        abnormalEdgesData.value = edges
        
        // 渲染图表
        nextTick(() => {
          renderLimeChart(limeWeights)
          renderGcnChart(edges)
        })
      }
    }
  } catch (error: any) {
    if (error.isPending) {
        xaiStatus.value = 0 // 处理中
        if (xaiDialogVisible.value) {
            pollTimer = setTimeout(refreshXai, 2500)
        }
    } else {
        // 如果后端返回 code != 200 会进 axios 下面的错误。如果是未完成其实也可以正常等
         xaiStatus.value = 0
    }
  } finally {
    xaiLoading.value = false
  }
}

const renderLimeChart = (weightsObj: Record<string, number>) => {
  if (!limeChartRef.value) return
  if (!limeChart) limeChart = echarts.init(limeChartRef.value)
  
  const keys = Object.keys(weightsObj)
  // 按重要性绝对值排序
  keys.sort((a, b) => Math.abs(weightsObj[b] || 0) - Math.abs(weightsObj[a] || 0))
  
  const yAxisData = keys.reverse() // 让最大的在上面
  const seriesData = yAxisData.map(k => {
    const val = weightsObj[k] || 0
    return {
      value: val,
      itemStyle: { color: val > 0 ? '#f56c6c' : '#67c23a' } // 红色促使异常，绿色抑制异常
    }
  })

  limeChart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: '5%', right: '15%', bottom: '15%', top: '10%', containLabel: true },
    xAxis: {
      type: 'value',
      name: '权重 (W)'
    },
    yAxis: { type: 'category', data: yAxisData, axisLabel: { interval: 0 } },
    series: [
      {
        name: '特征对本次预测的局部归因影响',
        type: 'bar',
        label: { show: true, position: 'right', formatter: (params: any) => parseFloat(params.value).toFixed(4) },
        data: seriesData
      }
    ]
  })
}

const renderGcnChart = (abnormalEdges: any[]) => {
  if (!gcnChartRef.value) return
  if (!gcnChart) gcnChart = echarts.init(gcnChartRef.value)

  const nodesMap = new Map()
  abnormalEdges.forEach(edge => {
    if (!nodesMap.has(edge.fromId)) nodesMap.set(edge.fromId, { name: edge.fromId, symbolSize: 30, itemStyle: { color: '#f56c6c' } })
    if (!nodesMap.has(edge.toId)) nodesMap.set(edge.toId, { name: edge.toId, symbolSize: 30, itemStyle: { color: '#f56c6c' } })
  })
  const nodes = Array.from(nodesMap.values())

  const links = abnormalEdges.map(edge => ({
    source: edge.fromId,
    target: edge.toId,
    value: parseFloat(edge.importance).toFixed(2),
    lineStyle: {
      color: '#ff4d4f',
      width: Math.max(2, parseFloat(edge.importance) * 10),
      curveness: 0.35
    }
  }))

  gcnChart.setOption({
    tooltip: { trigger: 'item', formatter: '{b}' },
    series: [
      {
        name: '异常微服务拓扑',
        type: 'graph',
        layout: 'force',
        force: {
          repulsion: 1000,
          edgeLength: 150
        },
        roam: true,
        label: {
          show: true,
          position: 'right', // 节点名称位置 (top/bottom/left/right)。换到右边能最大限度避红边自环
          distance: 15,
          color: '#333',
          backgroundColor: 'rgba(255, 255, 255, 0.85)',
          padding: [4, 6],
          borderRadius: 4,
          formatter: (params: any) => params.name.substring(0, 8)
        },
        data: nodes,
        links: links,
        edgeSymbol: ['circle', 'arrow'],
        edgeSymbolSize: [4, 10],
        edgeLabel: {
          show: true,
          position: 'middle',
          formatter: '{c}',
          color: '#ff4d4f',
          backgroundColor: 'rgba(255, 255, 255, 0.85)',
          padding: [2, 4],
          borderRadius: 4
        }
      }
    ]
  })
}

const handleClose = (done: () => void) => {
  if (pollTimer) {
    clearTimeout(pollTimer)
    pollTimer = null
  }
  if (limeChart) limeChart.dispose()
  if (gcnChart) gcnChart.dispose()
  limeChart = null
  gcnChart = null
  done()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
