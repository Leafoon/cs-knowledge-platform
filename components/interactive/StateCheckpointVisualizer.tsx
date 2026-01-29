"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Save, FolderOpen, RotateCcw, GitBranch, Clock } from 'lucide-react';

interface Checkpoint {
  id: string;
  version: number;
  timestamp: number;
  tag: string;
  state: {
    currentStep: string;
    data: Record<string, any>;
    nextSteps: string[];
  };
}

const SAMPLE_WORKFLOW = [
  {
    id: 'step_1',
    name: '数据收集',
    description: '收集用户基本信息',
    data: { name: 'Alice', email: 'alice@example.com' }
  },
  {
    id: 'step_2',
    name: '数据验证',
    description: '验证邮箱格式和必填字段',
    data: { validated: true, errors: [] }
  },
  {
    id: 'step_3',
    name: '数据处理',
    description: '格式化和标准化数据',
    data: { processed: true, normalized_name: 'ALICE' }
  },
  {
    id: 'step_4',
    name: '数据存储',
    description: '保存到数据库',
    data: { saved: true, db_id: 12345 }
  }
];

export default function StateCheckpointVisualizer() {
  const [currentStep, setCurrentStep] = useState(0);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<Checkpoint | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const executeWorkflow = () => {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }

    setIsPlaying(true);
    setCurrentStep(0);
    setCheckpoints([]);

    let step = 0;
    const interval = setInterval(() => {
      if (step >= SAMPLE_WORKFLOW.length) {
        clearInterval(interval);
        setIsPlaying(false);
        return;
      }

      setCurrentStep(step);
      step++;
    }, 1500);
  };

  const saveCheckpoint = () => {
    if (currentStep >= SAMPLE_WORKFLOW.length) return;

    const workflow = SAMPLE_WORKFLOW[currentStep];
    const newCheckpoint: Checkpoint = {
      id: `checkpoint_${Date.now()}`,
      version: checkpoints.length + 1,
      timestamp: Date.now(),
      tag: workflow.name,
      state: {
        currentStep: workflow.id,
        data: workflow.data,
        nextSteps: SAMPLE_WORKFLOW.slice(currentStep + 1).map(s => s.name)
      }
    };

    setCheckpoints(prev => [...prev, newCheckpoint]);
  };

  const loadCheckpoint = (checkpoint: Checkpoint) => {
    const stepIndex = SAMPLE_WORKFLOW.findIndex(s => s.id === checkpoint.state.currentStep);
    if (stepIndex !== -1) {
      setCurrentStep(stepIndex);
      setSelectedCheckpoint(checkpoint);
    }
  };

  const reset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setCheckpoints([]);
    setSelectedCheckpoint(null);
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('zh-CN', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          State Checkpoint Visualizer
        </h3>
        <p className="text-slate-600">
          演示状态快照的保存、加载与版本管理
        </p>
      </div>

      {/* Controls */}
      <div className="mb-6 flex gap-3 flex-wrap">
        <button
          onClick={executeWorkflow}
          className="px-4 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          {isPlaying ? '暂停工作流' : '执行工作流'}
        </button>
        <button
          onClick={saveCheckpoint}
          disabled={currentStep >= SAMPLE_WORKFLOW.length}
          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Save className="w-4 h-4" />
          保存快照
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
          <span className="font-semibold">
            {currentStep + 1} / {SAMPLE_WORKFLOW.length}
          </span>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Workflow Timeline */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h4 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <GitBranch className="w-5 h-5" />
              工作流时间线
            </h4>

            <div className="space-y-4">
              {SAMPLE_WORKFLOW.map((step, idx) => {
                const isActive = idx === currentStep;
                const isPassed = idx < currentStep;

                return (
                  <motion.div
                    key={step.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className={`relative flex items-start gap-4 p-4 rounded-lg border-2 ${
                      isActive
                        ? 'border-cyan-400 bg-cyan-50 shadow-md'
                        : isPassed
                        ? 'border-green-300 bg-green-50'
                        : 'border-slate-200 bg-slate-50'
                    }`}
                  >
                    {/* Step Number */}
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 font-bold ${
                      isActive
                        ? 'bg-cyan-500 text-white'
                        : isPassed
                        ? 'bg-green-500 text-white'
                        : 'bg-slate-300 text-slate-600'
                    }`}>
                      {idx + 1}
                    </div>

                    {/* Step Content */}
                    <div className="flex-1">
                      <div className="font-semibold text-slate-800 mb-1">
                        {step.name}
                      </div>
                      <div className="text-sm text-slate-600 mb-2">
                        {step.description}
                      </div>

                      {/* Step Data */}
                      {(isActive || isPassed) && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          className="mt-2 p-2 bg-white rounded border border-slate-200"
                        >
                          <div className="text-xs font-mono text-slate-700">
                            {JSON.stringify(step.data, null, 2)}
                          </div>
                        </motion.div>
                      )}
                    </div>

                    {/* Checkpoint Indicator */}
                    {checkpoints.some(cp => cp.state.currentStep === step.id) && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="flex-shrink-0"
                      >
                        <div className="w-6 h-6 rounded-full bg-purple-500 flex items-center justify-center">
                          <Save className="w-3 h-3 text-white" />
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Checkpoints List */}
        <div>
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h4 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5" />
              快照历史 ({checkpoints.length})
            </h4>

            {checkpoints.length === 0 ? (
              <div className="text-center py-8 text-slate-400">
                暂无快照
                <div className="text-xs mt-2">
                  执行工作流并点击"保存快照"
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {checkpoints.map((checkpoint) => (
                  <motion.button
                    key={checkpoint.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    onClick={() => loadCheckpoint(checkpoint)}
                    className={`w-full text-left p-3 rounded-lg border-2 transition-all ${
                      selectedCheckpoint?.id === checkpoint.id
                        ? 'border-purple-400 bg-purple-50 shadow-md'
                        : 'border-slate-200 bg-slate-50 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <div className="text-xs font-mono text-purple-600">
                        v{checkpoint.version}
                      </div>
                      <div className="flex-1 font-semibold text-sm text-slate-800">
                        {checkpoint.tag}
                      </div>
                    </div>
                    <div className="text-xs text-slate-500">
                      {formatTime(checkpoint.timestamp)}
                    </div>
                  </motion.button>
                ))}
              </div>
            )}
          </div>

          {/* Selected Checkpoint Details */}
          {selectedCheckpoint && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 bg-white rounded-lg border border-slate-200 p-6"
            >
              <h4 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                <FolderOpen className="w-5 h-5" />
                快照详情
              </h4>

              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">版本号</span>
                  <span className="font-mono font-semibold text-purple-600">
                    v{selectedCheckpoint.version}
                  </span>
                </div>

                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">标签</span>
                  <span className="font-medium text-slate-800">
                    {selectedCheckpoint.tag}
                  </span>
                </div>

                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">时间戳</span>
                  <span className="text-slate-700">
                    {formatTime(selectedCheckpoint.timestamp)}
                  </span>
                </div>

                <div className="pt-3 border-t border-slate-200">
                  <div className="text-xs font-semibold text-slate-700 mb-2">
                    状态数据
                  </div>
                  <div className="p-2 bg-slate-50 rounded border border-slate-200">
                    <pre className="text-xs font-mono text-slate-700 overflow-x-auto">
                      {JSON.stringify(selectedCheckpoint.state, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">快照管理代码</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`class StateCheckpointManager:
    def save_checkpoint(self, session_id: str, checkpoint_id: str, state: Dict[str, Any]):
        """保存状态快照"""
        checkpoint_data = {
            "session_id": session_id,
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "state": state
        }
        
        # 保存快照（7天过期）
        self.redis.set(
            f"checkpoint:{session_id}:{checkpoint_id}",
            json.dumps(checkpoint_data),
            ex=7 * 24 * 3600
        )
        
        # 添加到快照列表（只保留最近10个）
        self.redis.lpush(f"checkpoints:{session_id}", checkpoint_id)
        self.redis.ltrim(f"checkpoints:{session_id}", 0, 9)
    
    def load_checkpoint(self, session_id: str, checkpoint_id: str) -> Dict[str, Any]:
        """加载状态快照"""
        key = f"checkpoint:{session_id}:{checkpoint_id}"
        data_str = self.redis.get(key)
        return json.loads(data_str)["state"] if data_str else None
    
    def list_checkpoints(self, session_id: str) -> List[str]:
        """列出会话的所有快照"""
        return self.redis.lrange(f"checkpoints:{session_id}", 0, -1)

# 使用示例
checkpoint_mgr = StateCheckpointManager("redis://localhost:6379/0")

# 保存快照
checkpoint_mgr.save_checkpoint(
    session_id="workflow_123",
    checkpoint_id="step_2_complete",
    state={"current_step": "validation", "data": {...}}
)

# 恢复快照
restored_state = checkpoint_mgr.load_checkpoint("workflow_123", "step_2_complete")`}
        </pre>
      </div>

      {/* Features */}
      <div className="mt-6 grid md:grid-cols-3 gap-4">
        <div className="p-4 bg-cyan-50 rounded-lg border border-cyan-200">
          <Save className="w-6 h-6 text-cyan-600 mb-2" />
          <h5 className="font-semibold text-cyan-800 mb-1">快照保存</h5>
          <p className="text-sm text-cyan-700">
            在关键节点保存状态，支持随时恢复
          </p>
        </div>

        <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
          <GitBranch className="w-6 h-6 text-purple-600 mb-2" />
          <h5 className="font-semibold text-purple-800 mb-1">版本管理</h5>
          <p className="text-sm text-purple-700">
            自动版本号，支持版本对比和回滚
          </p>
        </div>

        <div className="p-4 bg-green-50 rounded-lg border border-green-200">
          <Clock className="w-6 h-6 text-green-600 mb-2" />
          <h5 className="font-semibold text-green-800 mb-1">时间旅行</h5>
          <p className="text-sm text-green-700">
            加载任意历史快照，调试更高效
          </p>
        </div>
      </div>
    </div>
  );
}
