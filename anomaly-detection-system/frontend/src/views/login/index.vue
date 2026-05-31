<template>
  <div class="login-container">
    <div class="login-wrapper">
      <div class="login-header">
        <el-icon :size="40" color="#409EFF" class="logo-icon"><Odometer /></el-icon>
        <span class="logo-text">微服务异常流量检测系统</span>
      </div>
      
      <el-form 
        class="login-form" 
        :model="loginForm" 
        :rules="rules"
        ref="loginFormRef"
      >
        <h2 class="login-title">管理后台登录</h2>
        <div class="login-subtitle">基于双模态算法的实时威胁溯源引擎</div>
        <el-form-item prop="username">
          <el-input 
            v-model="loginForm.username" 
            placeholder="账号" 
            size="large"
            :prefix-icon="User"
          />
        </el-form-item>
        <el-form-item prop="password">
          <el-input 
            v-model="loginForm.password" 
            type="password" 
            placeholder="密码" 
            show-password
            size="large"
            :prefix-icon="Lock"
            @keyup.enter="handleLogin"
          />
        </el-form-item>
        <el-form-item>
          <el-button 
            type="primary" 
            class="login-btn" 
            :loading="loading" 
            @click="handleLogin"
            size="large"
          >
            登录
          </el-button>
        </el-form-item>
      </el-form>
      
      <div class="login-footer">
        Powered by RDF & GCN Dual-Modal Analysis
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/store/user'
import { User, Lock, Odometer } from '@element-plus/icons-vue'
import request from '@/utils/request'
import { ElMessage } from 'element-plus'

const router = useRouter()
const userStore = useUserStore()
const loginFormRef = ref()

const loading = ref(false)
const loginForm = reactive({ username: '', password: '' })
const rules = {
  username: [{ required: true, message: '请输入账号', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

const handleLogin = () => {
  loginFormRef.value?.validate(async (valid: boolean) => {
    if (valid) {
      loading.value = true
      try {
        const res: any = await request.post('/auth/login', loginForm)
        userStore.setToken(res.token)
        userStore.setUserInfo(res.username, res.role, res.email, res.status)
        ElMessage.success('登录成功')
        router.push('/')
      } finally {
        loading.value = false
      }
    }
  })
}
</script>

<style scoped lang="scss">
.login-container {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: 
    linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%),
    url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80') center/cover no-repeat;
  
  .login-wrapper {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    border-radius: 16px;
    padding: 0;
    width: 480px;
    overflow: hidden;
    position: relative;
    
    &::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 4px;
      background: linear-gradient(90deg, #409EFF, #67C23A, #E6A23C);
    }
  }
}

.login-header {
  padding: 40px 40px 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  
  .logo-text {
    font-size: 24px;
    font-weight: 700;
    color: #fff;
    letter-spacing: 1px;
  }
}

.login-form {
  padding: 0 50px 30px;
  
  .login-title {
    color: #E2E8F0;
    font-size: 20px;
    margin: 0 0 10px;
    text-align: center;
    font-weight: 600;
  }
  
  .login-subtitle {
    color: #94A3B8;
    font-size: 14px;
    text-align: center;
    margin-bottom: 35px;
  }
  
  :deep(.el-input__wrapper) {
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
    
    &.is-focus {
      box-shadow: 0 0 0 1px #409EFF inset;
    }
    
    .el-input__inner {
      color: #fff;
      &::placeholder { color: #94A3B8; }
    }
  }
  
  .login-btn {
    width: 100%;
    margin-top: 10px;
    font-size: 16px;
    font-weight: bold;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #409EFF, #3b82f6);
    border: none;
    height: 45px;
    transition: transform 0.2s, box-shadow 0.2s;
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 20px rgba(64, 158, 255, 0.3);
    }
    
    &:active {
      transform: translateY(0);
    }
  }
}

.login-footer {
  text-align: center;
  padding: 15px;
  background: rgba(0, 0, 0, 0.2);
  color: #64748B;
  font-size: 12px;
  letter-spacing: 1px;
}
</style>
