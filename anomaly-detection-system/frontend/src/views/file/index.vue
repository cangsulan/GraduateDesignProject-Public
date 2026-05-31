<template>
  <div class="file-detect">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <span>创建新检测任务</span>
        </div>
      </template>

      <div class="upload-area" v-loading="uploading">
        <el-form :model="form" class="task-form" label-width="auto">
          <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px;">
            <span style="white-space: nowrap; font-weight: 500;"><span style="color: #f56c6c;">*</span> 任务名称</span>
            <el-input v-model="form.taskName" placeholder="单次检测会诊任务名" style="flex: 1;" />
            <el-button type="primary" @click="submitTask" style="width: 160px;">提交异步检测策略</el-button>
          </div>
          
          <!-- 选项卡切换 -->
          <el-tabs v-model="uploadMode" class="upload-tabs">
            <el-tab-pane label="CSV/JSON模式" name="csv_json">
              <el-row :gutter="16">
                <el-col :span="12">
                  <div class="upload-box-label">特征数据 (CSV)</div>
                  <el-upload
                    class="upload-component"
                    drag
                    action=""
                    :auto-upload="false"
                    :limit="1"
                    :on-change="handleCsvChange"
                    :on-remove="handleCsvRemove"
                    accept=".csv"
                  >
                    <el-icon class="el-icon--upload" :size="28"><Document /></el-icon>
                    <div class="el-upload__text">拖拽 CSV 或 <em>点击选取</em></div>
                  </el-upload>
                </el-col>
                <el-col :span="12">
                  <div class="upload-box-label">调用拓扑 (JSON)</div>
                  <el-upload
                    class="upload-component"
                    drag
                    action=""
                    :auto-upload="false"
                    :limit="1"
                    :on-change="handleJsonChange"
                    :on-remove="handleJsonRemove"
                    accept=".json"
                  >
                    <el-icon class="el-icon--upload" :size="28"><Share /></el-icon>
                    <div class="el-upload__text">拖拽 JSON 或 <em>点击选取</em></div>
                  </el-upload>
                </el-col>
              </el-row>
              <el-alert
                title="提示：双文件齐备将启动最高精度融合策略，仅传一项则系统自适应单模型降级防御。"
                type="info"
                :closable="false"
                show-icon
                style="margin-top: 12px;"
              />
            </el-tab-pane>
            
            <el-tab-pane label="PCAP模式 (原始流量)" name="pcap">
              <div class="pcap-upload-area">
                <div class="upload-box-label">原始流量文件 (PCAP)</div>
                <el-upload
                  class="upload-component pcap-upload"
                  drag
                  multiple
                  action=""
                  :auto-upload="false"
                  :on-change="handlePcapChange"
                  :on-remove="handlePcapRemove"
                  :file-list="pcapFileList"
                  accept=".pcap,.pcapng"
                >
                  <el-icon class="el-icon--upload" :size="36"><Upload /></el-icon>
                  <div class="el-upload__text">拖拽 PCAP 文件 或 <em>点击选取</em></div>
                  <template #tip>
                    <div class="el-upload__tip">
                      支持上传多个pcap文件，系统将自动提取特征并进行异常检测
                    </div>
                  </template>
                </el-upload>
                <el-alert
                  title="PCAP模式：系统将自动从原始网络流量中提取特征（API调用序列、时间间隔等）和调用拓扑图，无需手动准备CSV/JSON文件。"
                  type="success"
                  :closable="false"
                  show-icon
                  style="margin-top: 12px;"
                />
              </div>
            </el-tab-pane>
          </el-tabs>
        </el-form>
      </div>
    </el-card>

    <el-card shadow="never" style="margin-top: 20px;">
      <template #header>
        <div class="card-header">
          <span>历史检测审计看板</span>
          <el-button type="success" :icon="Refresh" circle @click="loadData" />
        </div>
      </template>

      <!-- 多条件筛选栏 -->
      <el-form :inline="true" :model="searchQuery" class="filter-form" style="margin-bottom: 12px;">
        <el-form-item label="任务名称">
          <el-input v-model="searchQuery.taskName" placeholder="按任务名搜索" clearable style="width: 160px;" @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item label="任务状态">
          <el-select v-model="searchQuery.status" placeholder="全部" clearable style="width: 130px;">
            <el-option label="待调度" value="PENDING" />
            <el-option label="分析中" value="DETECTING" />
            <el-option label="已完成" value="COMPLETED" />
            <el-option label="失败" value="FAILED" />
          </el-select>
        </el-form-item>
        <el-form-item label="创建时间">
          <el-date-picker
            v-model="searchQuery.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            value-format="YYYY-MM-DD"
            style="width: 240px;"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">查询</el-button>
          <el-button @click="resetSearch">重置</el-button>
        </el-form-item>
      </el-form>

      <el-table :data="tableData" v-loading="loading" stripe border style="width: 100%">
        <!-- 任务ID -->
        <el-table-column prop="id" label="任务ID" width="80" />
        <!-- 任务名 -->
        <el-table-column prop="taskName" label="任务名称" min-width="120" show-overflow-tooltip/>
        
        <!-- 数据来源类型 -->
        <el-table-column label="数据来源" width="100" align="center">
          <template #default="scope">
            <el-tag v-if="scope.row.csvUrl && scope.row.csvUrl.includes('.pcap')" type="success" size="small">PCAP</el-tag>
            <el-tag v-else type="primary" size="small">CSV/JSON</el-tag>
          </template>
        </el-table-column>
        
        <!-- 文件链接 -->
        <el-table-column label="多模介质链接" min-width="120">
          <template #default="scope">
            <template v-if="scope.row.csvUrl && scope.row.csvUrl.includes('.pcap')">
              <el-link type="success" :href="getDownloadUrl(scope.row.csvUrl)" target="_blank">PCAP文件</el-link>
            </template>
            <template v-else>
              <el-link v-if="scope.row.csvUrl" type="primary" :href="getDownloadUrl(scope.row.csvUrl)" target="_blank" style="margin-right: 10px;">CSV</el-link>
              <el-link v-if="scope.row.jsonUrl" type="success" :href="getDownloadUrl(scope.row.jsonUrl)" target="_blank">JSON</el-link>
              <span v-if="!scope.row.csvUrl && !scope.row.jsonUrl" style="color: #999;">无外网链接</span>
            </template>
          </template>
        </el-table-column>

        <!-- 检测任务创建时间 -->
        <el-table-column label="创建时间" min-width="110" align="center">
          <template #default="scope">
            <div v-if="scope.row.createTime" style="line-height: 1.2;">
              <div>{{ formatDate(scope.row.createTime) }}</div>
              <div style="font-size: 0.9em; color: #888;">{{ formatTimeOnly(scope.row.createTime) }}</div>
            </div>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <!-- 文件上传成功时间 -->
        <el-table-column label="上传成功时间" min-width="110" align="center">
          <template #default="scope">
            <div v-if="scope.row.uploadTime" style="line-height: 1.2;">
              <div>{{ formatDate(scope.row.uploadTime) }}</div>
              <div style="font-size: 0.9em; color: #888;">{{ formatTimeOnly(scope.row.uploadTime) }}</div>
            </div>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <!-- 检测开始时间 -->
        <el-table-column label="检测开始时间" min-width="110" align="center">
          <template #default="scope">
            <div v-if="scope.row.startTime" style="line-height: 1.2;">
              <div>{{ formatDate(scope.row.startTime) }}</div>
              <div style="font-size: 0.9em; color: #888;">{{ formatTimeOnly(scope.row.startTime) }}</div>
            </div>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <!-- 检测结束时间 -->
        <el-table-column label="检测结束时间" min-width="110" align="center">
          <template #default="scope">
            <div v-if="scope.row.endTime" style="line-height: 1.2;">
              <div>{{ formatDate(scope.row.endTime) }}</div>
              <div style="font-size: 0.9em; color: #888;">{{ formatTimeOnly(scope.row.endTime) }}</div>
            </div>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <!-- 检测耗时 -->
        <el-table-column label="检测耗时" min-width="100" align="center">
          <template #default="scope">
            <span v-if="scope.row.duration !== null && scope.row.duration !== undefined">{{ (scope.row.duration / 1000).toFixed(3) }} s</span>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <!-- 任务状态 -->
        <el-table-column label="任务状态" min-width="100" align="center">
          <template #default="scope">
            <el-tag v-if="scope.row.status === 'PENDING'" type="info">待调度</el-tag>
            <el-tag v-else-if="scope.row.status === 'DETECTING'" type="warning" class="is-loading">分析中</el-tag>
            <el-tag v-else-if="scope.row.status === 'COMPLETED'" type="success">已完成</el-tag>
            <el-tag v-else type="danger">系统失败</el-tag>
          </template>
        </el-table-column>

        <!-- 记录数 -->
        <el-table-column label="记录数" min-width="80" align="center">
          <template #default="scope">
            {{ scope.row.status === 'COMPLETED' ? scope.row.recordCount : '-' }}
          </template>
        </el-table-column>

        <!-- 异常数量 -->
        <el-table-column label="异常数量" min-width="80" align="center">
          <template #default="scope">
            <strong v-if="scope.row.status === 'COMPLETED'" :class="{'danger-text': scope.row.anomalyCount > 0}">{{ scope.row.anomalyCount }}</strong>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <!-- 异常率 -->
        <el-table-column label="异常率" min-width="80" align="center">
          <template #default="scope">
            <span v-if="scope.row.status === 'COMPLETED' && scope.row.anomalyRate !== null">
              {{ (scope.row.anomalyRate * 100).toFixed(2) }}%
            </span>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <!-- 操作 -->
        <el-table-column label="操作" min-width="70" align="center" fixed="right">
          <template #default="scope">
             <el-button type="danger" size="small" :icon="Delete" plain @click="deleteTask(scope.row.id)" />
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
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { Document, Share, Search, Refresh, Delete, Upload } from '@element-plus/icons-vue'
import request from '@/utils/request'
import { ElMessage, ElMessageBox } from 'element-plus'
import dayjs from 'dayjs'

const getDownloadUrl = (url: string) => {
  if (!url) return ''
  let parsedUrl = url.replace('http://anomaly-minio:9000', `http://${window.location.hostname}:9000`)
  if (parsedUrl.includes('?')) {
    parsedUrl = parsedUrl.split('?')[0] || parsedUrl
  }
  return parsedUrl
}

const uploading = ref(false)
const uploadMode = ref<'csv_json' | 'pcap'>('csv_json')
const form = reactive({
  taskName: ''
})

// CSV/JSON模式文件
let csvFileRaw: File | null = null
let jsonFileRaw: File | null = null

// PCAP模式文件
const pcapFileList = ref<any[]>([])
let pcapFilesRaw: File[] = []

const handleCsvChange = (file: any) => {
  csvFileRaw = file.raw
}
const handleCsvRemove = () => {
  csvFileRaw = null
}

const handleJsonChange = (file: any) => {
  jsonFileRaw = file.raw
}
const handleJsonRemove = () => {
  jsonFileRaw = null
}

const handlePcapChange = (_file: any, fileList: any[]) => {
  pcapFilesRaw = fileList.map(f => f.raw)
}
const handlePcapRemove = (_file: any, fileList: any[]) => {
  pcapFilesRaw = fileList.map(f => f.raw)
}

const submitTask = async () => {
  if (!form.taskName.trim()) {
    ElMessage.warning('请设定专属的任务名称！')
    return
  }
  
  if (uploadMode.value === 'pcap') {
    // PCAP模式提交
    if (pcapFilesRaw.length === 0) {
      ElMessage.warning('请至少上传一个PCAP文件！')
      return
    }
    
    const formData = new FormData()
    formData.append('taskName', form.taskName.trim())
    pcapFilesRaw.forEach(file => {
      formData.append('pcapFiles', file)
    })
    
    uploading.value = true
    try {
      const res: any = await request.post('/detect/file-task/create-from-pcap', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      ElMessage.success(typeof res === 'string' ? res : 'PCAP检测任务已提交，正在后台处理...')
      form.taskName = ''
      pcapFilesRaw = []
      pcapFileList.value = []
      loadData()
    } catch (e: any) {
      ElMessage.error(e.message || '网络失连或内部错误！')
    } finally {
      uploading.value = false
    }
  } else {
    // CSV/JSON模式提交
    if (!csvFileRaw && !jsonFileRaw) {
      ElMessage.warning('拒绝立案：您至少需要提供一个特征模态文件 (CSV/JSON)。')
      return
    }
    
    const formData = new FormData()
    formData.append('taskName', form.taskName.trim())
    if (csvFileRaw) formData.append('csvFile', csvFileRaw)
    if (jsonFileRaw) formData.append('jsonFile', jsonFileRaw)
    
    uploading.value = true
    try {
      const res: any = await request.post('/detect/file-task/create', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      ElMessage.success(typeof res === 'string' ? res : '已部署推行后台检测，请静候端点回调')
      form.taskName = ''
      csvFileRaw = null
      jsonFileRaw = null
      loadData()
    } catch (e: any) {
      ElMessage.error(e.message || '网络失连或内部错误！')
    } finally {
      uploading.value = false
    }
  }
}

// ---------------- Table Loading ----------------
const tableData = ref([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const searchQuery = reactive({
  taskName: '',
  status: '',
  dateRange: null as string[] | null
})

const handleSearch = () => {
  currentPage.value = 1
  loadData()
}

const resetSearch = () => {
  searchQuery.taskName = ''
  searchQuery.status = ''
  searchQuery.dateRange = null
  currentPage.value = 1
  loadData()
}

const loadData = async () => {
  loading.value = true
  try {
    const params: any = {
      pageNum: currentPage.value,
      pageSize: pageSize.value
    }
    if (searchQuery.taskName) params.taskName = searchQuery.taskName
    if (searchQuery.status) params.status = searchQuery.status
    if (searchQuery.dateRange && searchQuery.dateRange.length === 2) {
      params.startDate = searchQuery.dateRange[0]
      params.endDate = searchQuery.dateRange[1]
    }
    const res: any = await request.get('/detect/file-task/page', { params })
    if (res && typeof res.records !== 'undefined') {
      tableData.value = res.records
      total.value = res.total
    }
  } finally {
    loading.value = false
  }
}

const deleteTask = (id: number) => {
  ElMessageBox.confirm('是否抹除该任务的一切历史痕迹?', '敏感操作核查', {
    confirmButtonText: '确定',
    cancelButtonText: '保守取消',
    type: 'warning'
  }).then(async () => {
    try {
      const res: any = await request.delete(`/detect/file-task/${id}`)
      ElMessage.success(typeof res === 'string' ? res : '删除成功')
      loadData()
    } catch (e: any) {
      ElMessage.error(e.message || '删除请求失败')
    }
  })
}

const formatDate = (time: string) => {
  if (!time) return '-'
  return dayjs(time).format('YYYY-MM-DD')
}

const formatTimeOnly = (time: string) => {
  if (!time) return '-'
  return dayjs(time).format('HH:mm:ss')
}

// Global hook
let pollingTimer: any = null

onMounted(() => {
  loadData()
  pollingTimer = setInterval(() => {
    const hasPending = tableData.value.some((t: any) => t.status === 'PENDING' || t.status === 'DETECTING')
    if (hasPending && !loading.value) {
      loadData()
    }
  }, 5000)
})

onUnmounted(() => {
  if (pollingTimer) {
    clearInterval(pollingTimer)
    pollingTimer = null
  }
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.upload-area {
  padding: 10px 16px;
}
.task-form {
  width: 100%;
}
.upload-box-label {
  font-size: 13px;
  color: #606266;
  margin-bottom: 6px;
  font-weight: 500;
}
.upload-component {
  width: 100%;
}
.upload-component :deep(.el-upload-dragger) {
  padding: 16px 10px;
}
.danger-text {
  color: #f56c6c;
}
.upload-tabs {
  margin-top: 10px;
}
.pcap-upload-area {
  min-height: 200px;
}
.pcap-upload :deep(.el-upload-dragger) {
  padding: 30px 20px;
}
</style>
