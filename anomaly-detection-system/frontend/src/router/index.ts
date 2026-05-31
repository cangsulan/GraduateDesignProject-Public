import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import { useUserStore } from '@/store/user'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/login/index.vue'),
    meta: { title: '系统登录' }
  },
  {
    path: '/',
    name: 'Layout',
    component: () => import('@/layout/index.vue'),
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/dashboard/index.vue'),
        meta: { title: '实时监控大屏', icon: 'Monitor' }
      },
      {
        path: 'file-detect',
        name: 'FileDetect',
        component: () => import('@/views/file/index.vue'),
        meta: { title: '文件批量检测', icon: 'Document' }
      },
      {
        path: 'anomalies',
        name: 'AnomalyRecords',
        component: () => import('@/views/anomaly/index.vue'),
        meta: { title: '异常记录与溯源', icon: 'Warning' }
      },
      {
        path: 'history',
        name: 'History',
        component: () => import('@/views/history/index.vue'),
        meta: { title: '综合统计分析', icon: 'DataLine' }
      },
      {
        path: 'users',
        name: 'UserManager',
        component: () => import('@/views/user/index.vue'),
        meta: { title: '用户权限管理', icon: 'User', roles: ['admin'] } // Only admin can access
      },
      {
        path: 'profile',
        name: 'Profile',
        component: () => import('@/views/profile/index.vue'),
        meta: { title: '个人信息主页', icon: 'Avatar' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫拦截
router.beforeEach((to, _from, next) => {
  const userStore = useUserStore()
  if (to.path !== '/login' && !userStore.token) {
    next('/login')
  } else {
    // Role based access control for frontend routes
    if (to.meta.roles && Array.isArray(to.meta.roles)) {
      if (!to.meta.roles.includes(userStore.role)) {
        // Stop navigation if normal user tries to access admin routes
        return next('/dashboard')
      }
    }
    document.title = (to.meta.title as string) || '异常流量检测系统'
    next()
  }
})

export default router
