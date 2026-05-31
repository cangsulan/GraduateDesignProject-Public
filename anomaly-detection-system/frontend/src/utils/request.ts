import axios from 'axios'
import { ElMessage } from 'element-plus'
import { useUserStore } from '@/store/user'
import router from '@/router'

const service = axios.create({
  baseURL: '/api',
  timeout: 10000
})

// 请求拦截器: 自动添加 Token
service.interceptors.request.use(
  config => {
    const userStore = useUserStore()
    if (userStore.token) {
      config.headers['Authorization'] = 'Bearer ' + userStore.token
    }
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器: 统一处理状态码
service.interceptors.response.use(
  response => {
    const res = response.data
    // 兼容可能下载文件或者非标准JSON的情况
    if (response.config.responseType === 'blob' || response.config.responseType === 'arraybuffer') {
        return response
    }
    
    if (res.code === 200) {
      return res.data
    } else if (res.code === 202) {
      // 202 表示任务进行中（如 XAI 在后台排队），此状态下不强弹错误 Toast
      return Promise.reject({ isPending: true, message: res.message })
    } else {
      ElMessage.error(res.message || '请求错误')
      return Promise.reject(new Error(res.message || 'Error'))
    }
  },
  error => {
    if (error.response) {
      if (error.response.status === 401 || error.response.status === 403) {
        ElMessage.error('登录回话已过期，请重新登录')
        const userStore = useUserStore()
        userStore.logout()
        router.push('/login')
      } else {
        ElMessage.error(error.response.data?.message || '服务器连接异常')
      }
    } else {
      ElMessage.error('网络错误或服务器未响应')
    }
    return Promise.reject(error)
  }
)

export default service
