<template>
  <div class="simulator-control">
    <div class="status-indicator">
      <span class="status-dot" :class="{ active: isRunning }"></span>
      <span class="status-text">{{ isRunning ? '模拟器运行中' : '模拟器已停止' }}</span>
    </div>
    
    <div class="control-actions">
      <div class="qps-slider">
        <span class="qps-label">并发发包 QPS: <span class="qps-val">{{ currentQps }}</span></span>
        <el-slider 
          v-model="currentQps" 
          :min="1" 
          :max="50" 
          @change="handleQpsChange"
          :disabled="!isRunning"
          style="width: 150px; margin: 0 15px;"
        />
      </div>
      
      <el-button 
        :type="isRunning ? 'danger' : 'success'" 
        :icon="isRunning ? 'VideoPause' : 'VideoPlay'"
        @click="toggleSimulator"
        :loading="loading"
        round
      >
        {{ isRunning ? '停止并发攻击包' : '启动模拟器发送流量' }}
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import request from '@/utils/request'
import { ElMessage } from 'element-plus'

const isRunning = ref(false)
const currentQps = ref(10)
const loading = ref(false)

const fetchStatus = async () => {
  try {
    const data: any = await request.get('/simulator/status')
    if (data) {
      isRunning.value = data.isRunning || false
      currentQps.value = data.qps || 10
    }
  } catch (e) {
    console.error('获取模拟器状态失败:', e)
  }
}

onMounted(() => {
  fetchStatus()
})

const toggleSimulator = async () => {
  loading.value = true
  const action = isRunning.value ? 'stop' : 'start'
  try {
    await request.post('/simulator/control', { action, qps: currentQps.value })
    isRunning.value = !isRunning.value
    ElMessage.success(`模拟器已${isRunning.value ? '启动' : '停止'}`)
  } catch(e) {
    // error handled by interceptor
  } finally {
    loading.value = false
  }
}

const handleQpsChange = async (val: number) => {
  if (!isRunning.value) return
  try {
    await request.post('/simulator/control', { action: 'update', qps: val })
    ElMessage.success(`已更新发包 QPS 为 ${val}`)
  } catch(e) {
    // error handled by interceptor
  }
}
</script>

<style scoped lang="scss">
.simulator-control {
  display: flex;
  align-items: center;
  background: rgba(240, 242, 245, 0.8);
  padding: 6px 20px;
  border-radius: 30px;
  border: 1px solid #e4e7ed;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);

  .status-indicator {
    display: flex;
    align-items: center;
    margin-right: 30px;
    
    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background-color: #909399;
      margin-right: 8px;
      position: relative;
      transition: all 0.3s;
      
      &.active {
        background-color: #67c23a;
        box-shadow: 0 0 8px rgba(103, 194, 58, 0.8);
        
        &::after {
          content: '';
          position: absolute;
          top: -3px; left: -3px; right: -3px; bottom: -3px;
          border-radius: 50%;
          border: 1px solid rgba(103, 194, 58, 0.6);
          animation: pulse 1.5s infinite;
        }
      }
    }
    
    .status-text {
      font-size: 14px;
      font-weight: 500;
      color: #606266;
    }
  }
  
  .control-actions {
    display: flex;
    align-items: center;
    
    .qps-slider {
      display: flex;
      align-items: center;
      
      .qps-label {
        font-size: 13px;
        color: #606266;
        white-space: nowrap;
        
        .qps-val {
          font-weight: bold;
          color: #409eff;
          display: inline-block;
          width: 20px;
          text-align: right;
        }
      }
    }
  }
}

@keyframes pulse {
  0% { transform: scale(1); opacity: 1; }
  100% { transform: scale(1.8); opacity: 0; }
}
</style>
