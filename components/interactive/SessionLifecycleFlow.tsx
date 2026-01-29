"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, User, MessageSquare, Database, Trash2 } from 'lucide-react';

type SessionStatus = 'created' | 'active' | 'idle' | 'archived' | 'deleted';

interface SessionState {
  sessionId: string;
  userId: string;
  status: SessionStatus;
  createdAt: number;
  lastActive: number;
  messageCount: number;
  ttl: number; // seconds
  metadata: Record<string, any>;
}

const LIFECYCLE_STEPS = [
  { 
    id: 'created',
    label: '创建会话',
    description: '生成唯一 session_id，初始化元数据',
    icon: Database,
    color: 'blue'
  },
  {
    id: 'active',
    label: '活跃状态',
    description: '用户发送消息，更新 last_active 时间',
    icon: MessageSquare,
    color: 'green'
  },
  {
    id: 'idle',
    label: '空闲状态',
    description: '超过N分钟无活动，但仍在 TTL 内',
    icon: Pause,
    color: 'yellow'
  },
  {
    id: 'archived',
    label: '归档状态',
    description: '超过活跃期限，迁移到冷存储',
    icon: Database,
    color: 'purple'
  },
  {
    id: 'deleted',
    label: '删除状态',
    description: 'TTL 过期或用户主动删除',
    icon: Trash2,
    color: 'red'
  }
];

export default function SessionLifecycleFlow() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [sessionState, setSessionState] = useState<SessionState>({
    sessionId: 'user_alice_20260128_a3b4c5d6',
    userId: 'alice',
    status: 'created',
    createdAt: Date.now(),
    lastActive: Date.now(),
    messageCount: 0,
    ttl: 7 * 24 * 3600, // 7 days
    metadata: {
      channel: 'web',
      language: 'zh-CN'
    }
  });

  const simulateLifecycle = () => {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }

    setIsPlaying(true);
    setCurrentStep(0);

    const steps = [
      () => {
        // Step 0: Created
        setSessionState(prev => ({
          ...prev,
          status: 'created',
          createdAt: Date.now(),
          lastActive: Date.now(),
          messageCount: 0
        }));
      },
      () => {
        // Step 1: Active
        setSessionState(prev => ({
          ...prev,
          status: 'active',
          lastActive: Date.now(),
          messageCount: 5
        }));
      },
      () => {
        // Step 2: Idle
        setSessionState(prev => ({
          ...prev,
          status: 'idle',
          lastActive: Date.now() - 30 * 60 * 1000, // 30 min ago
          messageCount: 5
        }));
      },
      () => {
        // Step 3: Archived
        setSessionState(prev => ({
          ...prev,
          status: 'archived',
          lastActive: Date.now() - 8 * 24 * 3600 * 1000, // 8 days ago
          messageCount: 15
        }));
      },
      () => {
        // Step 4: Deleted
        setSessionState(prev => ({
          ...prev,
          status: 'deleted',
          lastActive: Date.now() - 90 * 24 * 3600 * 1000, // 90 days ago
        }));
      }
    ];

    let step = 0;
    const interval = setInterval(() => {
      if (step >= steps.length) {
        clearInterval(interval);
        setIsPlaying(false);
        return;
      }
      steps[step]();
      setCurrentStep(step);
      step++;
    }, 2000);
  };

  const reset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setSessionState({
      sessionId: 'user_alice_20260128_a3b4c5d6',
      userId: 'alice',
      status: 'created',
      createdAt: Date.now(),
      lastActive: Date.now(),
      messageCount: 0,
      ttl: 7 * 24 * 3600,
      metadata: { channel: 'web', language: 'zh-CN' }
    });
  };

  const getStepColor = (stepId: string) => {
    const colors: Record<string, string> = {
      blue: 'bg-blue-500 text-white border-blue-600',
      green: 'bg-green-500 text-white border-green-600',
      yellow: 'bg-yellow-500 text-white border-yellow-600',
      purple: 'bg-purple-500 text-white border-purple-600',
      red: 'bg-red-500 text-white border-red-600'
    };
    const step = LIFECYCLE_STEPS.find(s => s.id === stepId);
    return colors[step?.color || 'blue'];
  };

  const formatTime = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) return `${days} 天前`;
    if (hours > 0) return `${hours} 小时前`;
    if (minutes > 0) return `${minutes} 分钟前`;
    return '刚刚';
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          Session Lifecycle Flow
        </h3>
        <p className="text-slate-600">
          可视化会话从创建到删除的完整生命周期
        </p>
      </div>

      {/* Controls */}
      <div className="mb-6 flex gap-3">
        <button
          onClick={simulateLifecycle}
          className="px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors flex items-center gap-2"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isPlaying ? '暂停' : '播放生命周期'}
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 transition-colors flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
        <div className="ml-auto text-sm text-slate-600 flex items-center gap-2">
          <span>当前步骤:</span>
          <span className="font-semibold">{currentStep + 1} / {LIFECYCLE_STEPS.length}</span>
        </div>
      </div>

      {/* Lifecycle Steps */}
      <div className="mb-8 relative">
        {/* Progress Line */}
        <div className="absolute top-6 left-0 right-0 h-1 bg-slate-200">
          <motion.div
            className="h-full bg-indigo-500"
            initial={{ width: '0%' }}
            animate={{ width: `${(currentStep / (LIFECYCLE_STEPS.length - 1)) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>

        <div className="grid grid-cols-5 gap-4 relative z-10">
          {LIFECYCLE_STEPS.map((step, idx) => {
            const Icon = step.icon;
            const isActive = idx === currentStep;
            const isPassed = idx < currentStep;

            return (
              <div key={step.id} className="flex flex-col items-center">
                <motion.div
                  className={`w-12 h-12 rounded-full border-2 flex items-center justify-center mb-2 ${
                    isActive
                      ? getStepColor(step.id)
                      : isPassed
                      ? 'bg-slate-300 border-slate-400 text-white'
                      : 'bg-white border-slate-300 text-slate-400'
                  }`}
                  animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                  transition={{ duration: 0.5, repeat: isActive ? Infinity : 0, repeatDelay: 1 }}
                >
                  <Icon className="w-6 h-6" />
                </motion.div>
                <div className="text-center">
                  <div className={`text-sm font-semibold ${isActive ? 'text-indigo-600' : 'text-slate-600'}`}>
                    {step.label}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {step.description}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Session State Display */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Current State */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h4 className="font-semibold text-slate-800 mb-4">当前会话状态</h4>

          <div className="space-y-4">
            <div className="flex justify-between items-center pb-3 border-b border-slate-100">
              <span className="text-sm text-slate-600">Session ID</span>
              <span className="font-mono text-sm text-slate-800">
                {sessionState.sessionId}
              </span>
            </div>

            <div className="flex justify-between items-center pb-3 border-b border-slate-100">
              <span className="text-sm text-slate-600">用户</span>
              <div className="flex items-center gap-2">
                <User className="w-4 h-4 text-slate-400" />
                <span className="font-medium text-slate-800">{sessionState.userId}</span>
              </div>
            </div>

            <div className="flex justify-between items-center pb-3 border-b border-slate-100">
              <span className="text-sm text-slate-600">状态</span>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                sessionState.status === 'created' ? 'bg-blue-100 text-blue-700' :
                sessionState.status === 'active' ? 'bg-green-100 text-green-700' :
                sessionState.status === 'idle' ? 'bg-yellow-100 text-yellow-700' :
                sessionState.status === 'archived' ? 'bg-purple-100 text-purple-700' :
                'bg-red-100 text-red-700'
              }`}>
                {sessionState.status.toUpperCase()}
              </span>
            </div>

            <div className="flex justify-between items-center pb-3 border-b border-slate-100">
              <span className="text-sm text-slate-600">消息数量</span>
              <span className="font-semibold text-slate-800">
                {sessionState.messageCount} 条
              </span>
            </div>

            <div className="flex justify-between items-center pb-3 border-b border-slate-100">
              <span className="text-sm text-slate-600">最后活跃</span>
              <span className="text-sm text-slate-800">
                {formatTime(sessionState.lastActive)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-600">TTL（剩余）</span>
              <span className="text-sm text-slate-800">
                {sessionState.ttl / 3600} 小时
              </span>
            </div>
          </div>
        </div>

        {/* Metadata & Actions */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h4 className="font-semibold text-slate-800 mb-4">元数据 & 操作</h4>

          <div className="mb-6">
            <h5 className="text-sm font-semibold text-slate-700 mb-2">会话元数据</h5>
            <div className="p-3 bg-slate-50 rounded border border-slate-200">
              <pre className="text-xs font-mono text-slate-700">
                {JSON.stringify(sessionState.metadata, null, 2)}
              </pre>
            </div>
          </div>

          <div>
            <h5 className="text-sm font-semibold text-slate-700 mb-3">生命周期操作</h5>
            <div className="space-y-2">
              <div className="p-3 bg-green-50 rounded border border-green-200">
                <div className="font-medium text-green-800 text-sm">创建会话</div>
                <div className="text-xs text-green-600 mt-1">
                  <code>manager.create_session(user_id, metadata)</code>
                </div>
              </div>

              <div className="p-3 bg-blue-50 rounded border border-blue-200">
                <div className="font-medium text-blue-800 text-sm">更新活跃时间</div>
                <div className="text-xs text-blue-600 mt-1">
                  <code>manager.update_activity(user_id, session_id)</code>
                </div>
              </div>

              <div className="p-3 bg-purple-50 rounded border border-purple-200">
                <div className="font-medium text-purple-800 text-sm">归档会话</div>
                <div className="text-xs text-purple-600 mt-1">
                  <code>manager.archive_inactive_sessions(days=30)</code>
                </div>
              </div>

              <div className="p-3 bg-red-50 rounded border border-red-200">
                <div className="font-medium text-red-800 text-sm">删除会话</div>
                <div className="text-xs text-red-600 mt-1">
                  <code>manager.delete_session(user_id, session_id)</code>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">完整代码示例</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`class SessionManager:
    def create_session(self, user_id: str, metadata: dict = None) -> str:
        """创建新会话"""
        session_id = generate_session_id(user_id)
        full_id = f"user:{user_id}:session:{session_id}"
        
        meta = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "message_count": 0,
            "status": "active",
            **(metadata or {})
        }
        
        self.redis.set(
            f"session_meta:{full_id}",
            json.dumps(meta),
            ex=7 * 24 * 3600  # 7天过期
        )
        
        return session_id
    
    def update_activity(self, user_id: str, session_id: str):
        """更新会话活跃时间"""
        full_id = f"user:{user_id}:session:{session_id}"
        key = f"session_meta:{full_id}"
        
        meta_str = self.redis.get(key)
        if meta_str:
            meta = json.loads(meta_str)
            meta["last_active"] = datetime.now().isoformat()
            meta["message_count"] = meta.get("message_count", 0) + 1
            self.redis.set(key, json.dumps(meta), ex=7 * 24 * 3600)
    
    def archive_inactive_sessions(self, inactive_days: int = 30):
        """归档不活跃会话"""
        cutoff = datetime.now() - timedelta(days=inactive_days)
        # 迁移到 PostgreSQL 冷存储
        # ...`}
        </pre>
      </div>
    </div>
  );
}
