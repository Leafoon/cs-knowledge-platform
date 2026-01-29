'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, Key, Lock, CheckCircle, XCircle, User, Server, Database, ArrowRight } from 'lucide-react'

type AuthMethod = 'api-key' | 'jwt' | 'rbac'
type FlowStep = 'request' | 'verify' | 'authorize' | 'response'

interface AuthFlow {
  method: AuthMethod
  steps: {
    step: FlowStep
    status: 'pending' | 'processing' | 'success' | 'error'
    message: string
  }[]
}

export default function AuthenticationFlow() {
  const [selectedMethod, setSelectedMethod] = useState<AuthMethod>('api-key')
  const [isRunning, setIsRunning] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  
  const authMethods = [
    {
      id: 'api-key' as AuthMethod,
      name: 'API Key 认证',
      icon: Key,
      description: '简单快速，适合服务间调用',
      color: 'blue' as const,
      steps: [
        { step: 'request' as FlowStep, message: '客户端发送请求 + X-API-Key Header' },
        { step: 'verify' as FlowStep, message: '服务端验证 API Key 是否在白名单中' },
        { step: 'authorize' as FlowStep, message: '检查 Key 对应的权限和配额' },
        { step: 'response' as FlowStep, message: '返回结果或 403 Forbidden' },
      ]
    },
    {
      id: 'jwt' as AuthMethod,
      name: 'OAuth2 + JWT',
      icon: Lock,
      description: '标准协议，支持用户登录和 Token 刷新',
      color: 'green',
      steps: [
        { step: 'request' as FlowStep, message: '客户端使用用户名密码换取 JWT Token' },
        { step: 'verify' as FlowStep, message: '验证 Token 签名和过期时间' },
        { step: 'authorize' as FlowStep, message: '从 Token payload 提取用户信息和权限' },
        { step: 'response' as FlowStep, message: '返回受保护资源或 401 Unauthorized' },
      ]
    },
    {
      id: 'rbac' as AuthMethod,
      name: '基于角色访问控制 (RBAC)',
      icon: Shield,
      description: '细粒度权限管理，适合多租户场景',
      color: 'purple',
      steps: [
        { step: 'request' as FlowStep, message: '客户端携带身份凭证（Token 或 API Key）' },
        { step: 'verify' as FlowStep, message: '验证用户身份并加载角色列表' },
        { step: 'authorize' as FlowStep, message: '检查用户角色是否满足端点要求' },
        { step: 'response' as FlowStep, message: '允许访问或返回 403 Insufficient Permissions' },
      ]
    },
  ]

  const currentMethod = authMethods.find(m => m.id === selectedMethod)!
  const colorClasses: Record<string, { bg: string; border: string; text: string; icon: string }> = {
    blue: { bg: 'bg-blue-50', border: 'border-blue-500', text: 'text-blue-700', icon: 'text-blue-500' },
    green: { bg: 'bg-green-50', border: 'border-green-500', text: 'text-green-700', icon: 'text-green-500' },
    purple: { bg: 'bg-purple-50', border: 'border-purple-500', text: 'text-purple-700', icon: 'text-purple-500' },
  }

  const runFlow = async () => {
    setIsRunning(true)
    setCurrentStep(0)
    
    for (let i = 0; i < currentMethod.steps.length; i++) {
      setCurrentStep(i)
      await new Promise(resolve => setTimeout(resolve, 1500))
    }
    
    setCurrentStep(currentMethod.steps.length)
    setIsRunning(false)
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-lg">
      {/* 标题 */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-3">
          <Shield className="w-8 h-8 text-indigo-600" />
          <h3 className="text-2xl font-bold text-gray-800">LangServe 认证授权流程</h3>
        </div>
        <p className="text-gray-600">选择认证方式，观察完整的认证授权流程</p>
      </div>

      {/* 认证方式选择 */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {authMethods.map((method) => {
          const Icon = method.icon
          const colors = colorClasses[method.color]
          const isSelected = selectedMethod === method.id
          
          return (
            <motion.button
              key={method.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => !isRunning && setSelectedMethod(method.id)}
              disabled={isRunning}
              className={`p-4 rounded-lg border-2 transition-all ${
                isSelected
                  ? `${colors.bg} ${colors.border} shadow-md`
                  : 'bg-white border-gray-200 hover:border-gray-300'
              } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <Icon className={`w-8 h-8 mx-auto mb-2 ${isSelected ? colors.icon : 'text-gray-400'}`} />
              <h4 className={`font-semibold mb-1 ${isSelected ? colors.text : 'text-gray-700'}`}>
                {method.name}
              </h4>
              <p className="text-xs text-gray-500">{method.description}</p>
            </motion.button>
          )
        })}
      </div>

      {/* 流程可视化 */}
      <div className="bg-white rounded-lg p-6 shadow-inner mb-6">
        <div className="flex items-center justify-between mb-6">
          <h4 className="text-lg font-semibold text-gray-800">认证流程步骤</h4>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={runFlow}
            disabled={isRunning}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              isRunning
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {isRunning ? '执行中...' : '演示流程'}
          </motion.button>
        </div>

        {/* 流程图 */}
        <div className="relative">
          {/* 连接线 */}
          <div className="absolute top-12 left-0 right-0 h-0.5 bg-gray-200 z-0" />
          
          <div className="relative z-10 grid grid-cols-4 gap-4">
            {currentMethod.steps.map((stepData, idx) => {
              const stepIcons = {
                request: User,
                verify: Server,
                authorize: Database,
                response: CheckCircle,
              }
              const StepIcon = stepIcons[stepData.step]
              
              const stepStatus = 
                idx < currentStep ? 'success' :
                idx === currentStep ? 'processing' :
                'pending'
              
              const statusColors = {
                pending: 'bg-gray-100 border-gray-300 text-gray-400',
                processing: 'bg-yellow-50 border-yellow-400 text-yellow-600 animate-pulse',
                success: 'bg-green-50 border-green-500 text-green-600',
              }

              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="relative"
                >
                  {/* 步骤卡片 */}
                  <div className={`p-4 rounded-lg border-2 transition-all ${statusColors[stepStatus]}`}>
                    <div className="flex items-center justify-center mb-3">
                      <div className="relative">
                        <StepIcon className="w-8 h-8" />
                        <AnimatePresence>
                          {stepStatus === 'success' && (
                            <motion.div
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              exit={{ scale: 0 }}
                              className="absolute -top-1 -right-1 bg-green-500 rounded-full p-0.5"
                            >
                              <CheckCircle className="w-4 h-4 text-white" />
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </div>
                    
                    <h5 className="font-semibold text-center mb-2 capitalize">
                      {stepData.step === 'request' ? '发送请求' :
                       stepData.step === 'verify' ? '验证身份' :
                       stepData.step === 'authorize' ? '权限检查' :
                       '返回响应'}
                    </h5>
                    
                    <p className="text-xs text-center leading-relaxed">
                      {stepData.message}
                    </p>
                  </div>

                  {/* 箭头 */}
                  {idx < currentMethod.steps.length - 1 && (
                    <div className="absolute top-12 -right-6 z-20">
                      <ArrowRight className={`w-5 h-5 ${
                        idx < currentStep ? 'text-green-500' : 'text-gray-300'
                      } transition-colors`} />
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="bg-gray-900 text-gray-100 rounded-lg p-6 overflow-hidden">
        <div className="flex items-center gap-2 mb-4">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
          </div>
          <span className="text-sm text-gray-400 ml-2">
            {selectedMethod === 'api-key' ? 'api_key_auth.py' :
             selectedMethod === 'jwt' ? 'jwt_auth.py' :
             'rbac_auth.py'}
          </span>
        </div>

        <pre className="text-sm overflow-x-auto">
          <code>
            {selectedMethod === 'api-key' && `# API Key 认证示例
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

VALID_KEYS = {
    "sk-user-alice": {"tier": "premium"},
    "sk-user-bob": {"tier": "basic"}
}

async def verify_api_key(key: str = Security(api_key_header)):
    if key not in VALID_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return VALID_KEYS[key]

@app.post("/translate")
async def translate(user=Depends(verify_api_key)):
    return {"result": "Translation"}
`}
            {selectedMethod === 'jwt' && `# JWT 认证示例
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key"

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        return {"username": username}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/translate")
async def translate(user=Depends(get_current_user)):
    return {"result": f"Hello {user['username']}"}
`}
            {selectedMethod === 'rbac' && `# RBAC 认证示例
from enum import Enum

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"

def require_roles(allowed_roles: List[Role]):
    async def checker(user=Depends(get_current_user)):
        if not set(user["roles"]).intersection(allowed_roles):
            raise HTTPException(status_code=403, 
                detail="Insufficient permissions")
        return user
    return checker

@app.post("/admin/reset")
async def admin_reset(user=Depends(require_roles([Role.ADMIN]))):
    return {"result": "Reset successful"}
`}
          </code>
        </pre>
      </div>

      {/* 最佳实践提示 */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg"
      >
        <div className="flex items-start gap-3">
          <Shield className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <h5 className="font-semibold text-blue-900 mb-2">安全最佳实践</h5>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• 生产环境必须使用 HTTPS 加密传输</li>
              <li>• Secret Key 和 API Keys 使用环境变量或密钥管理服务存储</li>
              <li>• 设置合理的 Token 过期时间（建议 15-30 分钟）</li>
              <li>• 记录所有认证失败事件，实施异常登录检测</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
