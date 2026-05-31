<template>
  <el-container class="layout-container">
    <el-aside width="240px" class="aside">
      <div class="logo">
        <el-icon :size="24" color="#409eff"><Odometer /></el-icon>
        <span>微服务异常检测大屏</span>
      </div>
      
      <el-menu
        :default-active="activePath"
        background-color="#1f2d3d"
        text-color="#bfcbd9"
        active-text-color="#409EFF"
        router
      >
        <template v-for="route in layoutRoutes" :key="route.path">
          <el-menu-item :index="'/' + route.path" v-if="route.meta && route.meta.title">
            <el-icon><component :is="route.meta.icon" /></el-icon>
            <span>{{ route.meta.title }}</span>
          </el-menu-item>
        </template>
        
        <el-menu-item index="" @click="handleLogout" style="margin-top: 50px;">
          <el-icon><SwitchButton /></el-icon>
          <span>退出登录</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    
    <el-container>
      <el-header class="header">
        <div class="header-left">
          <span class="page-title">{{ currentRoute.meta.title }}</span>
        </div>
        <div class="header-right">
          <!-- 这里存放模拟器的控制条：只有在实时监控页面才显示 -->
          <SimulatorControl v-if="activePath === '/dashboard'" />
          
          <div class="user-info">
            <el-avatar :size="32" icon="UserFilled" />
            <span class="username">{{ userStore.username || 'Admin' }}</span>
          </div>
        </div>
      </el-header>
      
      <el-main class="main">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useUserStore } from '@/store/user'
import SimulatorControl from './SimulatorControl.vue'

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

const activePath = computed(() => route.path)
const currentRoute = computed(() => route)

const layoutRoutes = computed(() => {
  const root = router.options.routes.find(r => r.path === '/')
  let routes = root?.children || []
  
  // 过滤掉当前角色无权访问的菜单项
  return routes.filter(route => {
    if (route.meta && route.meta.roles && Array.isArray(route.meta.roles)) {
      return route.meta.roles.includes(userStore.role)
    }
    return true
  })
})

const handleLogout = () => {
  userStore.logout()
  router.push('/login')
}
</script>

<style scoped lang="scss">
.layout-container {
  height: 100vh;
  background-color: #f0f2f5;

  .aside {
    background-color: #1f2d3d;
    box-shadow: 2px 0 6px rgba(0, 21, 41, 0.35);
    z-index: 10;
    
    .logo {
      height: 60px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      color: #fff;
      font-size: 18px;
      font-weight: bold;
      letter-spacing: 1px;
      background: #192635; /* 略深的头部背景 */
    }
    
    .el-menu {
      border-right: none;
    }
  }

  .header {
    background: #fff;
    box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 20px;
    z-index: 9;
    
    .header-left {
      .page-title {
        font-size: 18px;
        font-weight: 600;
        color: #303133;
      }
    }
    
    .header-right {
      display: flex;
      align-items: center;
      gap: 30px;
      
      .user-info {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        padding: 5px 10px;
        border-radius: 4px;
        transition: all 0.3s;
        
        &:hover {
          background-color: #f5f7fa;
        }
        
        .username {
          font-weight: 500;
          color: #606266;
        }
      }
    }
  }

  .main {
    padding: 20px;
    overflow-y: auto;
    overflow-x: hidden;
  }
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
