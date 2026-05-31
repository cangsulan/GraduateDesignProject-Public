import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUserStore = defineStore('user', () => {
  const token = ref(localStorage.getItem('token') || '')
  const username = ref(localStorage.getItem('username') || '')
  const role = ref(localStorage.getItem('role') || '')
  const email = ref(localStorage.getItem('email') || '')
  const status = ref(Number(localStorage.getItem('status')) || 0)

  const setToken = (t: string) => {
    token.value = t
    localStorage.setItem('token', t)
  }

  const setUserInfo = (u: string, r: string, e: string, s: number) => {
    username.value = u
    role.value = r
    email.value = e
    status.value = s
    
    localStorage.setItem('username', u)
    localStorage.setItem('role', r)
    // Email could be null
    if (e) localStorage.setItem('email', e)
    else localStorage.removeItem('email')
    
    localStorage.setItem('status', s.toString())
  }

  const logout = () => {
    token.value = ''
    username.value = ''
    role.value = ''
    email.value = ''
    status.value = 0
    
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    localStorage.removeItem('role')
    localStorage.removeItem('email')
    localStorage.removeItem('status')
  }

  return { token, username, role, email, status, setToken, setUserInfo, logout }
})
