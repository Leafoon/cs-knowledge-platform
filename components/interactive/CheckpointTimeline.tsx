'use client';

import React, { useState, useMemo } from 'react';

type Checkpoint = {
  id: string;
  timestamp: string;
  state: {
    messages: number;
    iteration: number;
    currentNode: string;
  };
  event: string;
};

export default function CheckpointTimeline() {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);
  const [showRestore, setShowRestore] = useState(false);

  const checkpoints: Checkpoint[] = useMemo(() => [
    {
      id: 'cp-0',
      timestamp: '10:00:00',
      state: { messages: 1, iteration: 0, currentNode: 'START' },
      event: '接收用户输入'
    },
    {
      id: 'cp-1',
      timestamp: '10:00:01',
      state: { messages: 2, iteration: 1, currentNode: 'agent' },
      event: 'LLM 调用完成'
    },
    {
      id: 'cp-2',
      timestamp: '10:00:02',
      state: { messages: 3, iteration: 1, currentNode: 'tools' },
      event: '工具执行: calculator(25*4)'
    },
    {
      id: 'cp-3',
      timestamp: '10:00:03',
      state: { messages: 4, iteration: 2, currentNode: 'agent' },
      event: 'LLM 第二次调用'
    },
    {
      id: 'cp-4',
      timestamp: '10:00:04',
      state: { messages: 5, iteration: 2, currentNode: 'tools' },
      event: '工具执行: search(LangGraph)'
    },
    {
      id: 'cp-5',
      timestamp: '10:00:05',
      state: { messages: 6, iteration: 3, currentNode: 'agent' },
      event: '生成最终答案'
    },
    {
      id: 'cp-6',
      timestamp: '10:00:06',
      state: { messages: 6, iteration: 3, currentNode: 'END' },
      event: '执行完成'
    }
  ], []);

  const selectedData = useMemo(() => 
    checkpoints.find(cp => cp.id === selectedCheckpoint)
  , [selectedCheckpoint, checkpoints]);

  const handleRestore = (checkpointId: string) => {
    setSelectedCheckpoint(checkpointId);
    setShowRestore(true);
    setTimeout(() => setShowRestore(false), 2000);
  };

  return (
    <div className="my-8 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        Checkpoint 时间线与时间旅行
      </h3>

      <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
        <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-100">
          什么是 Checkpoint？
        </h4>
        <p className="text-sm text-gray-700 dark:text-gray-300">
          Checkpoint 是 LangGraph 在每个节点执行后自动保存的状态快照。
          支持：① 崩溃后恢复执行 ② 时间旅行调试 ③ 分支执行 ④ 人工审批中断
        </p>
      </div>

      <div className="relative">
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-300 dark:bg-gray-600"></div>
        
        <div className="space-y-4">
          {checkpoints.map((checkpoint, idx) => (
            <div
              key={checkpoint.id}
              className={`relative pl-16 transition-all ${
                selectedCheckpoint === checkpoint.id
                  ? 'scale-105'
                  : ''
              }`}
            >
              <div
                className={`absolute left-5 top-3 w-6 h-6 rounded-full border-4 transition-all ${
                  selectedCheckpoint === checkpoint.id
                    ? 'bg-green-500 border-green-300 scale-125'
                    : 'bg-blue-500 border-blue-300'
                }`}
              ></div>
              
              <div
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  selectedCheckpoint === checkpoint.id
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-500'
                    : 'bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 hover:border-blue-500'
                }`}
                onClick={() => setSelectedCheckpoint(checkpoint.id)}
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <div className="font-mono text-xs text-gray-500 dark:text-gray-400">
                      {checkpoint.id}
                    </div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {checkpoint.event}
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {checkpoint.timestamp}
                  </div>
                </div>
                
                <div className="flex gap-4 text-xs text-gray-600 dark:text-gray-400">
                  <span>节点: <strong className="text-blue-600 dark:text-blue-400">{checkpoint.state.currentNode}</strong></span>
                  <span>消息数: <strong>{checkpoint.state.messages}</strong></span>
                  <span>迭代: <strong>{checkpoint.state.iteration}</strong></span>
                </div>

                {selectedCheckpoint === checkpoint.id && (
                  <div className="mt-3 pt-3 border-t border-gray-300 dark:border-gray-600">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRestore(checkpoint.id);
                      }}
                      className="px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors"
                    >
                      🔄 恢复到此状态
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {selectedData && (
        <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
          <h4 className="font-semibold mb-3 text-gray-800 dark:text-gray-200">
            状态详情: {selectedData.id}
          </h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">当前节点</div>
              <div className="font-mono text-lg font-semibold text-purple-600 dark:text-purple-400">
                {selectedData.state.currentNode}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">执行时间</div>
              <div className="font-mono text-lg font-semibold text-purple-600 dark:text-purple-400">
                {selectedData.timestamp}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">消息数量</div>
              <div className="font-mono text-lg font-semibold text-purple-600 dark:text-purple-400">
                {selectedData.state.messages}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">迭代次数</div>
              <div className="font-mono text-lg font-semibold text-purple-600 dark:text-purple-400">
                {selectedData.state.iteration}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded border border-purple-300 dark:border-purple-700">
            <div className="text-sm font-semibold mb-1 text-gray-700 dark:text-gray-300">
              可执行操作：
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              <div>• <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">app.get_state(config)</code> - 获取状态</div>
              <div>• <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">app.update_state(config, new_state)</code> - 更新状态</div>
              <div>• <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">app.invoke(None, config)</code> - 从此点继续执行</div>
            </div>
          </div>
        </div>
      )}

      {showRestore && (
        <div className="mt-4 p-4 bg-green-100 dark:bg-green-900/30 rounded border border-green-500 animate-pulse">
          <div className="flex items-center gap-2 text-green-800 dark:text-green-200">
            <span className="text-xl">✓</span>
            <span className="font-semibold">已恢复到 {selectedCheckpoint}！</span>
          </div>
          <div className="mt-1 text-sm text-green-700 dark:text-green-300">
            执行将从该检查点继续，之前的状态已加载。
          </div>
        </div>
      )}

      <div className="mt-6 grid grid-cols-3 gap-3">
        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded text-center">
          <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
            {checkpoints.length}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            总检查点数
          </div>
        </div>
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded text-center">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {selectedData?.state.iteration || 0}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            当前迭代
          </div>
        </div>
        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded text-center">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {selectedData?.state.messages || 0}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            消息数
          </div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded text-sm text-gray-700 dark:text-gray-300">
        <strong>实际应用：</strong> 使用 MemorySaver（内存）或 SqliteSaver（持久化）保存 checkpoint。
        在生产环境中，可以实现：① 长时间运行任务的断点续传 ② 人工审批流程中断 ③ A/B 测试分支执行。
      </div>
    </div>
  );
}
