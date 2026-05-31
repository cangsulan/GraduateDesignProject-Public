<template>
  <div class="user-manager-container">
    <el-card class="box-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="title">系统用户管理</span>
          <el-button type="primary" :icon="Plus" @click="openAddDialog">
            新增用户
          </el-button>
        </div>
      </template>

      <!-- 搜索表单 -->
      <div style="margin-bottom: 20px;">
        <el-form :inline="true" :model="queryParams" class="demo-form-inline">
          <el-form-item label="用户名">
            <el-input v-model="queryParams.username" placeholder="请输入用户名" clearable />
          </el-form-item>
          <el-form-item label="角色">
            <el-select v-model="queryParams.role" placeholder="请选择角色" clearable style="width: 150px">
              <el-option label="管理员" value="admin" />
              <el-option label="普通用户" value="user" />
            </el-select>
          </el-form-item>
          <el-form-item label="状态">
            <el-select v-model="queryParams.status" placeholder="请选择状态" clearable style="width: 150px">
              <el-option label="启用" :value="1" />
              <el-option label="冻结" :value="0" />
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="handleSearch">查询</el-button>
            <el-button @click="resetSearch">重置</el-button>
          </el-form-item>
        </el-form>
      </div>

      <!-- 用户列表表格 -->
      <el-table :data="tableData" v-loading="loading" border style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" align="center" />
        <el-table-column prop="username" label="用户名" />
        <el-table-column prop="role" label="角色">
          <template #default="{ row }">
            <el-tag :type="row.role === 'admin' ? 'danger' : 'info'">
              {{ row.role === 'admin' ? '管理员' : '普通用户' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="email" label="邮箱">
          <template #default="{ row }">
            {{ row.email || '--' }}
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="创建时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.createTime) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-switch
              v-model="row.status"
              :active-value="1"
              :inactive-value="0"
              :disabled="row.username === 'admin'"
              @change="toggleStatus(row)"
              inline-prompt
              active-text="启用"
              inactive-text="冻结"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120" align="center" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link :icon="Edit" @click="openEditDialog(row)">
              编辑
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 底部分页 -->
      <div style="display: flex; justify-content: flex-end; margin-top: 20px;">
        <el-pagination
          v-model:current-page="queryParams.page"
          v-model:page-size="queryParams.size"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          :total="total"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 用户表单弹窗 -->
    <el-dialog
      v-model="dialogVisible"
      :title="isEdit ? '编辑用户' : '新增用户'"
      width="500px"
      @close="resetForm"
    >
      <el-form 
        ref="formRef"
        :model="formData"
        :rules="formRules"
        label-width="100px"
      >
        <el-form-item label="用户名" prop="username">
          <el-input 
            v-model="formData.username" 
            placeholder="请输入用户名" 
            :disabled="isEdit"
          />
        </el-form-item>
        
        <!-- 新增时必填，编辑时选填 -->
        <el-form-item label="密码" prop="password">
          <el-input 
            v-model="formData.password" 
            type="password" 
            placeholder="请输入密码" 
            show-password 
          />
        </el-form-item>

        <el-form-item label="邮箱" prop="email">
          <el-input v-model="formData.email" placeholder="请输入邮箱地址" />
        </el-form-item>

        <el-form-item label="角色" prop="role">
          <el-select v-model="formData.role" placeholder="请选择角色" style="width: 100%">
            <el-option label="普通用户" value="user" />
            <el-option label="管理员" value="admin" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="状态" prop="status">
          <el-radio-group v-model="formData.status" :disabled="formData.username === 'admin'">
            <el-radio :value="1">启用</el-radio>
            <el-radio :value="0">冻结</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" :loading="submitLoading" @click="handleSubmit">
            确认
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { Plus, Edit } from "@element-plus/icons-vue"
import request from "@/utils/request"
import { ElMessage } from "element-plus"

const loading = ref(false)
const tableData = ref<any[]>([])
const total = ref(0) // 总条数

// 搜索和分页参数
const queryParams = reactive({
  page: 1,
  size: 10,
  username: '',
  role: '',
  status: undefined as number | undefined
})

// 弹窗相关
const dialogVisible = ref(false)
const isEdit = ref(false)
const submitLoading = ref(false)
const formRef = ref()

const formData = reactive({
  id: undefined,
  username: "",
  password: "",
  email: "",
  role: "user",
  status: 1
})

const formRules = {
  username: [{ required: true, message: "请输入用户名", trigger: "blur" }],
  role: [{ required: true, message: "请选择用户角色", trigger: "change" }]
}

const fetchUsers = async () => {
  loading.value = true
  try {
    const res: any = await request.get("/users", { params: queryParams })
    if (res && res.records) {
      tableData.value = res.records
      total.value = res.total
    } else {
      tableData.value = res || []
      total.value = res?.length || 0
    }
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  queryParams.page = 1
  fetchUsers()
}

const resetSearch = () => {
  queryParams.page = 1
  queryParams.username = ''
  queryParams.role = ''
  queryParams.status = undefined
  fetchUsers()
}

const handleSizeChange = (val: number) => {
  queryParams.size = val
  fetchUsers()
}

const handleCurrentChange = (val: number) => {
  queryParams.page = val
  fetchUsers()
}

const formatDate = (dateStr: string) => {
  if (!dateStr) return "--"
  const date = new Date(dateStr)
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2,"0")}-${String(date.getDate()).padStart(2,"0")} ${String(date.getHours()).padStart(2,"0")}:${String(date.getMinutes()).padStart(2,"0")}:${String(date.getSeconds()).padStart(2,"0")}`
}

const toggleStatus = async (row: any) => {
  try {
    await request.put(`/users/${row.id}/status`)
    // import { ElMessage } from 'element-plus'
    // ElMessage.success("状态已切换") // Auto import usually handles this, assuming request handles errors
  } catch (e) {
    // 恢复原来的状态
    row.status = row.status === 1 ? 0 : 1
  }
}

const resetForm = () => {
  if (formRef.value) {
    formRef.value.resetFields()
  }
  formData.id = undefined
  formData.username = ""
  formData.password = ""
  formData.email = ""
  formData.role = "user"
  formData.status = 1
}

const openAddDialog = () => {
  isEdit.value = false
  dialogVisible.value = true
}

const openEditDialog = (row: any) => {
  isEdit.value = true
  formData.id = row.id
  formData.username = row.username
  formData.password = "" // 不回显密码
  formData.email = row.email || ""
  formData.role = row.role
  formData.status = row.status
  dialogVisible.value = true
}

const handleSubmit = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid: boolean) => {
    if (valid) {
      if (!isEdit.value && !formData.password) {
         // Vue3 warning auto importer may not be active, but let's assume it is or just use standard throw
         ElMessage.error("新建用户必须设置密码")
         return
      }
      
      submitLoading.value = true
      try {
        if (isEdit.value) {
          await request.put(`/users/${formData.id}`, formData)
          ElMessage.success("更新成功")
        } else {
          await request.post("/users", formData)
          ElMessage.success("新增成功")
        }
        dialogVisible.value = false
        fetchUsers()
      } finally {
        submitLoading.value = false
      }
    }
  })
}

onMounted(() => {
  fetchUsers()
})
</script>

<style scoped>
.user-manager-container {
  padding: 20px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.title {
  font-weight: 600;
  font-size: 16px;
}
</style>
