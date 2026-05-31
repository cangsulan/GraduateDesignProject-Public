<template>
  <div class="profile-container">
    <el-row :gutter="20">
      <!-- 个人信息展示卡片 -->
      <el-col :span="8">
        <el-card class="box-card user-card" shadow="hover">
          <div class="user-header">
            <el-avatar :size="80" icon="UserFilled" />
            <div class="user-name">{{ userStore.username }}</div>
            <el-tag :type="userStore.role === 'admin' ? 'danger' : 'info'" size="large">
              {{ userStore.role === 'admin' ? '管理员' : '普通用户' }}
            </el-tag>
          </div>
          <el-divider />
          <div class="user-details">
            <div class="detail-item">
              <el-icon><Message /></el-icon>
              <span class="label">邮箱：</span>
              <span>{{ userStore.email || '未绑定' }}</span>
            </div>
            <div class="detail-item">
              <el-icon><CircleCheckFilled v-if="userStore.status === 1" /><CircleCloseFilled v-else /></el-icon>
              <span class="label">状态：</span>
              <span :style="{color: userStore.status === 1 ? '#67C23A' : '#F56C6C'}">
                {{ userStore.status === 1 ? '正常启用' : '已被冻结' }}
              </span>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- 信息修改面板 -->
      <el-col :span="16">
        <el-card class="box-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span>账号安全设置</span>
            </div>
          </template>

          <el-tabs v-model="activeTab" class="settings-tabs">
            <!-- 基本资料 -->
            <el-tab-pane label="基本资料" name="profile">
              <el-form 
                ref="profileFormRef" 
                :model="profileForm" 
                label-width="120px" 
                @submit.prevent
              >
                <el-form-item label="登录账号">
                  <el-input :value="userStore.username" disabled />
                  <div class="form-tip">账号名注册后不可修改</div>
                </el-form-item>
                <el-form-item label="绑定邮箱" prop="email">
                  <el-input v-model="profileForm.email" placeholder="请输入你的邮箱" />
                </el-form-item>
                <el-form-item>
                  <el-button type="primary" :loading="profileLoading" @click="handleUpdateProfile">
                    保存修改
                  </el-button>
                </el-form-item>
              </el-form>
            </el-tab-pane>

            <!-- 修改密码 -->
            <el-tab-pane label="修改密码" name="password">
              <el-form 
                ref="passwordFormRef" 
                :model="passwordForm" 
                :rules="passwordRules" 
                label-width="120px"
              >
                <el-form-item label="当前密码" prop="oldPassword">
                  <el-input 
                    v-model="passwordForm.oldPassword" 
                    type="password" 
                    show-password 
                    placeholder="请输入当前密码" 
                  />
                </el-form-item>
                <el-form-item label="新密码" prop="newPassword">
                  <el-input 
                    v-model="passwordForm.newPassword" 
                    type="password" 
                    show-password 
                    placeholder="请输入新密码" 
                  />
                </el-form-item>
                <el-form-item label="确认新密码" prop="confirmPassword">
                  <el-input 
                    v-model="passwordForm.confirmPassword" 
                    type="password" 
                    show-password 
                    placeholder="请再次输入新密码" 
                  />
                </el-form-item>
                <el-form-item>
                  <el-button type="primary" :loading="passwordLoading" @click="handleUpdatePassword">
                    更新密码
                  </el-button>
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useUserStore } from '@/store/user'
import request from '@/utils/request'
import { Message, CircleCheckFilled, CircleCloseFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const userStore = useUserStore()
const activeTab = ref('profile')

// ----- Profile -----
const profileLoading = ref(false)
const profileForm = reactive({
  email: ''
})

// ----- Password -----
const passwordFormRef = ref()
const passwordLoading = ref(false)
const passwordForm = reactive({
  oldPassword: '',
  newPassword: '',
  confirmPassword: ''
})

const validateConfirm = (_rule: any, value: any, callback: any) => {
  if (value === '') {
    callback(new Error('请再次输入密码'))
  } else if (value !== passwordForm.newPassword) {
    callback(new Error('两次输入密码不一致!'))
  } else {
    callback()
  }
}

const passwordRules = {
  oldPassword: [{ required: true, message: '请输入当前密码', trigger: 'blur' }],
  newPassword: [{ required: true, message: '请输入新密码', trigger: 'blur' }],
  confirmPassword: [
    { required: true, message: '请确认新密码', trigger: 'blur' },
    { validator: validateConfirm, trigger: 'blur' }
  ]
}

onMounted(async () => {
  try {
    const res: any = await request.get('/users/me')
    // Update local store with latest info
    userStore.setUserInfo(res.username, res.role, res.email, res.status)
    profileForm.email = res.email || ''
  } catch (e) {
    // 忽略异常，依赖原有的 store 内容
    profileForm.email = userStore.email || ''
  }
})

const handleUpdateProfile = async () => {
  profileLoading.value = true
  try {
    await request.put('/users/me/profile', { email: profileForm.email })
    ElMessage.success('个人资料已更新')
    userStore.setUserInfo(userStore.username, userStore.role, profileForm.email, userStore.status)
  } finally {
    profileLoading.value = false
  }
}

const handleUpdatePassword = async () => {
  if (!passwordFormRef.value) return
  await passwordFormRef.value.validate(async (valid: boolean) => {
    if (valid) {
      passwordLoading.value = true
      try {
        await request.put('/users/me/password', {
          oldPassword: passwordForm.oldPassword,
          newPassword: passwordForm.newPassword
        })
        ElMessage.success('密码修改成功，请重新登录')
        passwordFormRef.value.resetFields()
        setTimeout(() => {
          userStore.logout()
          window.location.reload()
        }, 1500)
      } finally {
        passwordLoading.value = false
      }
    }
  })
}
</script>

<style scoped lang="scss">
.profile-container {
  padding: 10px;
}

.user-card {
  .user-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px 0;
    
    .user-name {
      margin: 15px 0 10px;
      font-size: 20px;
      font-weight: bold;
      color: #303133;
    }
  }
  
  .user-details {
    padding: 10px 0;
    
    .detail-item {
      display: flex;
      align-items: center;
      margin-bottom: 20px;
      font-size: 14px;
      
      .el-icon {
        font-size: 18px;
        color: #909399;
        margin-right: 10px;
      }
      
      .label {
        color: #606266;
        width: 60px;
      }
      
      span:last-child {
        color: #303133;
        font-weight: 500;
      }
    }
  }
}

.settings-tabs {
  padding: 0 10px;
}

.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
  line-height: 1.2;
}
</style>
