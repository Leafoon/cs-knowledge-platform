'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Operation = 'all-gather' | 'reduce-scatter';

export default function AllGatherReduceScatter() {
  const [operation, setOperation] = useState<Operation>('all-gather');
  const [step, setStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const numGPUs = 4;

  // 每个 GPU 的初始数据分片
  const initialShards = [
    ['A0', 'A1', 'A2', 'A3'],
    ['B0', 'B1', 'B2', 'B3'],
    ['C0', 'C1', 'C2', 'C3'],
    ['D0', 'D1', 'D2', 'D3'],
  ];

  // Reduce-Scatter 的梯度数据
  const gradients = [
    ['G0', 'G1', 'G2', 'G3'],
    ['G0\'', 'G1\'', 'G2\'', 'G3\''],
    ['G0"', 'G1"', 'G2"', 'G3"'],
    ['G0"\'', 'G1"\'', 'G2"\'', 'G3"\''],
  ];

  const operationInfo = {
    'all-gather': {
      title: 'All-Gather（FSDP 前向传播）',
      description: '每个 GPU 收集所有分片，重建完整参数',
      color: 'blue',
      steps: [
        '初始状态：GPU 0-3 各持有 1/4 参数',
        'GPU 0 广播 A 分片给所有 GPU',
        'GPU 1 广播 B 分片给所有 GPU',
        'GPU 2 广播 C 分片给所有 GPU',
        'GPU 3 广播 D 分片给所有 GPU',
        '完成：所有 GPU 拥有完整参数 [A,B,C,D]',
      ],
      formula: '\\text{all\\_gather}(x_i) \\rightarrow [x_0, x_1, x_2, x_3]',
    },
    'reduce-scatter': {
      title: 'Reduce-Scatter（FSDP 反向传播）',
      description: '梯度求和后分片到各 GPU',
      color: 'purple',
      steps: [
        '初始状态：每个 GPU 有完整梯度 [G0, G1, G2, G3]',
        '对 G0 求和：G0_sum = G0 + G0\' + G0" + G0"\'',
        '对 G1 求和：G1_sum = G1 + G1\' + G1" + G1"\'',
        '对 G2, G3 同样求和',
        '分发：GPU 0 获得 G0_sum，GPU 1 获得 G1_sum...',
        '完成：每个 GPU 拥有 1/4 聚合梯度',
      ],
      formula: '\\text{reduce\\_scatter}([g_0, g_1, g_2, g_3]) \\rightarrow g_i',
    },
  };

  const currentOp = operationInfo[operation];

  useEffect(() => {
    if (isAnimating) {
      const timer = setInterval(() => {
        setStep((prev) => {
          if (prev >= currentOp.steps.length - 1) {
            setIsAnimating(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1500);
      return () => clearInterval(timer);
    }
  }, [isAnimating, currentOp.steps.length]);

  const handlePlay = () => {
    setStep(0);
    setIsAnimating(true);
  };

  const handleReset = () => {
    setStep(0);
    setIsAnimating(false);
  };

  // 计算当前步骤每个 GPU 的数据
  const getCurrentData = (gpuId: number) => {
    if (operation === 'all-gather') {
      if (step === 0) {
        // 初始状态：仅有自己的分片
        return [initialShards[gpuId]];
      } else if (step <= numGPUs) {
        // 逐步收集分片
        const collected = [];
        for (let i = 0; i < Math.min(step, numGPUs); i++) {
          collected.push(initialShards[i]);
        }
        return collected;
      } else {
        // 完成：所有分片
        return initialShards;
      }
    } else {
      // reduce-scatter
      if (step === 0) {
        // 初始状态：每个 GPU 有完整梯度
        return gradients[gpuId];
      } else if (step < numGPUs + 1) {
        // 求和中
        return gradients[gpuId];
      } else {
        // 完成：仅持有聚合后的一个分片
        return [`Σ G${gpuId}`];
      }
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto bg-gradient-to-br from-violet-50 to-fuchsia-50 dark:from-violet-950 dark:to-fuchsia-950 rounded-2xl shadow-2xl p-8">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-slate-100 mb-3">
          All-Gather 与 Reduce-Scatter 详解
        </h3>
        <p className="text-slate-600 dark:text-slate-400">
          FSDP 核心通信原语可视化
        </p>
      </div>

      {/* 操作选择 */}
      <div className="flex gap-4 mb-6 justify-center">
        {(Object.keys(operationInfo) as Operation[]).map((op) => {
          const info = operationInfo[op];
          return (
            <button
              key={op}
              onClick={() => {
                setOperation(op);
                handleReset();
              }}
              className={`px-8 py-4 rounded-xl font-semibold transition-all ${
                operation === op
                  ? `bg-${info.color}-500 text-white shadow-lg scale-105`
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:shadow-md'
              }`}
            >
              <div className="text-lg">{info.title}</div>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 可视化区域 */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="flex justify-between items-center mb-6">
            <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">
              {currentOp.title}
            </h4>
            <div className="flex gap-2">
              <button
                onClick={handlePlay}
                disabled={isAnimating}
                className="px-4 py-2 bg-green-500 text-white rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-green-600 transition-colors"
              >
                ▶ 播放
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-slate-500 text-white rounded-lg font-semibold hover:bg-slate-600 transition-colors"
              >
                ↻ 重置
              </button>
            </div>
          </div>

          {/* GPU 可视化 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {Array.from({ length: numGPUs }, (_, gpuId) => {
              const data = getCurrentData(gpuId);
              return (
                <motion.div
                  key={gpuId}
                  className={`bg-gradient-to-br from-${currentOp.color}-100 to-${currentOp.color}-200 dark:from-${currentOp.color}-900/50 dark:to-${currentOp.color}-800/50 rounded-xl p-4`}
                  animate={{
                    scale: step > 0 && step <= numGPUs + 1 ? [1, 1.02, 1] : 1,
                  }}
                  transition={{ duration: 0.5 }}
                >
                  <div className={`text-center mb-3 text-sm font-bold text-${currentOp.color}-700 dark:text-${currentOp.color}-300`}>
                    GPU {gpuId}
                  </div>
                  <div className="space-y-2">
                    {Array.isArray(data[0]) ? (
                      (data as string[][]).map((shard, idx) => (
                        <motion.div
                          key={idx}
                          className={`bg-${currentOp.color}-500 text-white rounded-lg p-2 text-center`}
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.1 }}
                        >
                          <div className="grid grid-cols-4 gap-1">
                            {shard.map((item: string, i: number) => (
                              <div key={i} className="text-xs font-mono">
                                {item}
                              </div>
                            ))}
                          </div>
                        </motion.div>
                      ))
                    ) : (
                      (data as string[]).map((item, idx) => (
                        <motion.div
                          key={idx}
                          className={`bg-${currentOp.color}-600 text-white rounded-lg p-3 text-center font-mono text-sm`}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: idx * 0.1 }}
                        >
                          {item}
                        </motion.div>
                      ))
                    )}
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* 步骤指示 */}
          <div className="space-y-2">
            {currentOp.steps.map((stepDesc, idx) => (
              <motion.div
                key={idx}
                className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
                  idx === step
                    ? `bg-${currentOp.color}-100 dark:bg-${currentOp.color}-900/30 border-l-4 border-${currentOp.color}-500`
                    : idx < step
                    ? 'bg-green-50 dark:bg-green-900/20'
                    : 'bg-slate-50 dark:bg-slate-800/50'
                }`}
                animate={{ opacity: idx <= step ? 1 : 0.5 }}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                    idx === step
                      ? `bg-${currentOp.color}-500 text-white`
                      : idx < step
                      ? 'bg-green-500 text-white'
                      : 'bg-slate-300 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
                  }`}
                >
                  {idx < step ? '✓' : idx + 1}
                </div>
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  {stepDesc}
                </span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* 信息面板 */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
            <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-3">
              📝 操作说明
            </h4>
            <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed mb-4">
              {currentOp.description}
            </p>
            <div className="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 font-mono text-xs overflow-x-auto">
              {'$'}{currentOp.formula}{'$'}
            </div>
          </div>

          <div className={`bg-${currentOp.color}-50 dark:bg-${currentOp.color}-900/30 rounded-xl p-4`}>
            <h4 className={`text-sm font-bold text-${currentOp.color}-800 dark:text-${currentOp.color}-200 mb-3`}>
              使用场景
            </h4>
            <ul className={`space-y-2 text-sm text-${currentOp.color}-700 dark:text-${currentOp.color}-300`}>
              {operation === 'all-gather' && (
                <>
                  <li>• FSDP 前向传播：重建完整参数</li>
                  <li>• FSDP 反向传播：重建参数计算梯度</li>
                  <li>• 收集所有 GPU 的预测结果</li>
                </>
              )}
              {operation === 'reduce-scatter' && (
                <>
                  <li>• FSDP 反向传播：梯度聚合后分片</li>
                  <li>• ZeRO 优化器：分布式参数更新</li>
                  <li>• 节省显存（避免每个 GPU 持有完整梯度）</li>
                </>
              )}
            </ul>
          </div>

          <div className="bg-slate-900 rounded-xl p-4">
            <h4 className="text-sm font-bold text-slate-300 mb-3">代码示例</h4>
            <div className="font-mono text-xs text-green-400 whitespace-pre-wrap">
              {operation === 'all-gather'
                ? `# All-Gather 示例\ntensor_list = [torch.empty_like(tensor) for _ in range(world_size)]\ntorch.distributed.all_gather(tensor_list, tensor)\ncomplete_tensor = torch.cat(tensor_list, dim=0)`
                : `# Reduce-Scatter 示例\ntensor_list = [tensor_0, tensor_1, tensor_2, tensor_3]\noutput = torch.empty_like(tensor_list[rank])\ntorch.distributed.reduce_scatter(output, tensor_list, op=ReduceOp.SUM)`}
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 p-4 rounded-r-xl">
            <p className="text-xs text-amber-800 dark:text-amber-300">
              <strong>性能提示：</strong>
              {operation === 'all-gather'
                ? ' All-Gather 在前向/反向传播时频繁调用，是 FSDP 的主要通信开销。使用 NCCL 后端可获得最佳性能。'
                : ' Reduce-Scatter 是 All-Reduce 的优化版本，仅传输 1/N 数据到每个 GPU，FSDP 用它替代 All-Reduce 节省显存。'}
            </p>
          </div>
        </div>
      </div>

      {/* 底部对比 */}
      <div className="mt-8 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
        <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
          通信量对比（N = 4 GPU，数据量 D）
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <div className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-2">
              All-Gather
            </div>
            <div className="text-2xl font-bold text-blue-600 mb-1">3D</div>
            <div className="text-xs text-blue-600 dark:text-blue-400">
              每个 GPU 接收 3/4 数据
            </div>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <div className="text-sm font-semibold text-purple-700 dark:text-purple-300 mb-2">
              Reduce-Scatter
            </div>
            <div className="text-2xl font-bold text-purple-600 mb-1">3D/4</div>
            <div className="text-xs text-purple-600 dark:text-purple-400">
              每个 GPU 仅接收 1/N 聚合结果
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <div className="text-sm font-semibold text-green-700 dark:text-green-300 mb-2">
              All-Reduce
            </div>
            <div className="text-2xl font-bold text-green-600 mb-1">6D</div>
            <div className="text-xs text-green-600 dark:text-green-400">
              All-Gather + Reduce-Scatter
            </div>
          </div>
        </div>
        <div className="mt-4 text-sm text-slate-600 dark:text-slate-400">
          <strong>关键洞察：</strong>
          FSDP 使用 All-Gather + Reduce-Scatter 替代 All-Reduce，
          在 ZeRO-3 模式下每个 GPU 仅保存 1/N 梯度，节省显存但增加通信次数。
        </div>
      </div>
    </div>
  );
}
