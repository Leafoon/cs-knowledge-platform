'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type CommunicationType = 'all-reduce' | 'broadcast' | 'gather' | 'scatter' | 'all-gather' | 'reduce-scatter';

export default function DistributedCommunicationVisualizer() {
  const [commType, setCommType] = useState<CommunicationType>('all-reduce');
  const [step, setStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const numGPUs = 4;
  const gpuValues = [0.5, 0.8, 0.6, 0.7]; // 初始梯度值

  const operations: {
    [key in CommunicationType]: {
      title: string;
      description: string;
      steps: string[];
      color: string;
      usageExample: string;
    };
  } = {
    'all-reduce': {
      title: 'All-Reduce',
      description: '所有 GPU 的梯度求和/平均，结果广播回所有 GPU（DDP 的核心操作）',
      steps: [
        '初始状态：每个 GPU 有不同的梯度',
        'Reduce：汇总所有梯度到临时结果',
        'Broadcast：将平均梯度广播给所有 GPU',
        '完成：所有 GPU 的梯度相同',
      ],
      color: 'blue',
      usageExample: 'accelerator.backward(loss)  # 自动 all-reduce 梯度',
    },
    'broadcast': {
      title: 'Broadcast',
      description: '将主进程（Rank 0）的数据复制到所有其他 GPU',
      steps: [
        '初始状态：仅 GPU 0 有数据',
        '广播开始：GPU 0 发送数据',
        '接收数据：其他 GPU 接收',
        '完成：所有 GPU 数据相同',
      ],
      color: 'green',
      usageExample: 'torch.distributed.broadcast(tensor, src=0)',
    },
    'gather': {
      title: 'Gather',
      description: '将所有 GPU 的数据收集到主进程（Rank 0）',
      steps: [
        '初始状态：每个 GPU 有不同数据',
        '收集开始：GPU 0 请求数据',
        '发送数据：其他 GPU 发送',
        '完成：GPU 0 拥有所有数据',
      ],
      color: 'purple',
      usageExample: 'all_losses = accelerator.gather(loss)',
    },
    'scatter': {
      title: 'Scatter',
      description: '将主进程的数据分发到所有 GPU（每个 GPU 获得不同部分）',
      steps: [
        '初始状态：GPU 0 有完整数据集',
        '分割数据：GPU 0 分割成 4 份',
        '分发数据：发送给各 GPU',
        '完成：每个 GPU 拥有不同片段',
      ],
      color: 'orange',
      usageExample: 'torch.distributed.scatter(tensor, scatter_list, src=0)',
    },
    'all-gather': {
      title: 'All-Gather',
      description: '所有 GPU 的数据收集到每个 GPU（每个 GPU 都拥有完整数据）',
      steps: [
        '初始状态：每个 GPU 有不同数据',
        '交换开始：GPU 相互发送数据',
        '接收数据：每个 GPU 收集',
        '完成：所有 GPU 拥有完整数据',
      ],
      color: 'pink',
      usageExample: 'torch.distributed.all_gather(tensor_list, tensor)',
    },
    'reduce-scatter': {
      title: 'Reduce-Scatter',
      description: '先 reduce 再 scatter（FSDP 的核心操作，节省内存）',
      steps: [
        '初始状态：每个 GPU 有不同梯度',
        'Reduce：求和所有梯度',
        'Scatter：分发不同部分',
        '完成：每个 GPU 拥有部分梯度',
      ],
      color: 'teal',
      usageExample: 'torch.distributed.reduce_scatter(output, input_list)',
    },
  };

  const currentOp = operations[commType];

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

  // 计算显示的值
  const getDisplayValues = () => {
    if (commType === 'all-reduce') {
      if (step >= 2) {
        const avg = gpuValues.reduce((a, b) => a + b, 0) / gpuValues.length;
        return gpuValues.map(() => avg);
      }
    } else if (commType === 'broadcast') {
      if (step >= 2) {
        return gpuValues.map(() => gpuValues[0]);
      }
    } else if (commType === 'gather') {
      return gpuValues;
    } else if (commType === 'all-gather') {
      return gpuValues;
    }
    return gpuValues;
  };

  const displayValues = getDisplayValues();

  return (
    <div className="w-full max-w-6xl mx-auto bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-950 dark:to-purple-950 rounded-2xl shadow-2xl p-8">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-slate-100 mb-3">
          分布式通信原语可视化
        </h3>
        <p className="text-slate-600 dark:text-slate-400">
          理解多 GPU 训练中的数据通信模式
        </p>
      </div>

      {/* 操作选择器 */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        {(Object.keys(operations) as CommunicationType[]).map((type) => (
          <button
            key={type}
            onClick={() => {
              setCommType(type);
              handleReset();
            }}
            className={`px-4 py-3 rounded-xl font-semibold transition-all text-sm ${
              commType === type
                ? `bg-${operations[type].color}-500 text-white shadow-lg scale-105`
                : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:shadow-md'
            }`}
          >
            {operations[type].title}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 可视化区域 */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg">
          <div className="flex justify-between items-center mb-6">
            <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100">
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
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {displayValues.map((value, idx) => (
              <motion.div
                key={idx}
                className={`relative bg-gradient-to-br from-${currentOp.color}-400 to-${currentOp.color}-600 rounded-xl p-6 shadow-lg`}
                animate={{
                  scale: step > 0 ? [1, 1.05, 1] : 1,
                  boxShadow:
                    step > 0
                      ? ['0 4px 6px rgba(0,0,0,0.1)', '0 10px 20px rgba(0,0,0,0.3)', '0 4px 6px rgba(0,0,0,0.1)']
                      : '0 4px 6px rgba(0,0,0,0.1)',
                }}
                transition={{ duration: 0.5, delay: idx * 0.1 }}
              >
                <div className="text-white text-center">
                  <div className="text-sm font-semibold mb-2">GPU {idx}</div>
                  <div className="text-3xl font-bold">{value.toFixed(2)}</div>
                  <div className="text-xs mt-2 opacity-80">
                    {commType === 'all-reduce' && step >= 2
                      ? '平均梯度'
                      : commType === 'broadcast' && step >= 2 && idx > 0
                      ? '已接收'
                      : commType === 'gather' && step >= 3 && idx === 0
                      ? '已收集'
                      : '本地值'}
                  </div>
                </div>

                {/* 数据传输动画 */}
                <AnimatePresence>
                  {commType === 'all-reduce' && step === 1 && (
                    <motion.div
                      className="absolute top-1/2 left-1/2 w-3 h-3 bg-yellow-400 rounded-full"
                      initial={{ scale: 0 }}
                      animate={{ scale: [0, 2, 0], opacity: [1, 0.5, 0] }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                  )}
                  {commType === 'broadcast' && step === 1 && idx === 0 && (
                    <motion.div
                      className="absolute top-0 right-0 w-2 h-2 bg-green-300 rounded-full"
                      animate={{ x: [0, 50], y: [0, 30], opacity: [1, 0] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>

          {/* 步骤指示器 */}
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
                <span
                  className={`text-sm font-medium ${
                    idx === step
                      ? `text-${currentOp.color}-800 dark:text-${currentOp.color}-200`
                      : 'text-slate-700 dark:text-slate-300'
                  }`}
                >
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
            <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              {currentOp.description}
            </p>
          </div>

          <div className="bg-slate-900 rounded-xl p-4">
            <h4 className="text-sm font-bold text-slate-300 mb-3">代码示例</h4>
            <div className="font-mono text-xs text-green-400 whitespace-pre-wrap">
              {currentOp.usageExample}
            </div>
          </div>

          <div className={`bg-${currentOp.color}-50 dark:bg-${currentOp.color}-900/30 rounded-xl p-4`}>
            <h4 className={`text-sm font-bold text-${currentOp.color}-800 dark:text-${currentOp.color}-200 mb-3`}>
              使用场景
            </h4>
            <ul className={`space-y-2 text-sm text-${currentOp.color}-700 dark:text-${currentOp.color}-300`}>
              {commType === 'all-reduce' && (
                <>
                  <li>• DDP 梯度同步</li>
                  <li>• 分布式优化器更新</li>
                  <li>• 全局指标计算</li>
                </>
              )}
              {commType === 'broadcast' && (
                <>
                  <li>• 模型参数初始化</li>
                  <li>• 超参数同步</li>
                  <li>• RNG 种子分发</li>
                </>
              )}
              {commType === 'gather' && (
                <>
                  <li>• 收集评估结果</li>
                  <li>• 合并预测输出</li>
                  <li>• 主进程日志记录</li>
                </>
              )}
              {commType === 'scatter' && (
                <>
                  <li>• 数据分片分发</li>
                  <li>• 负载均衡</li>
                  <li>• Pipeline 并行</li>
                </>
              )}
              {commType === 'all-gather' && (
                <>
                  <li>• FSDP 前向传播</li>
                  <li>• 收集完整预测</li>
                  <li>• 全局状态同步</li>
                </>
              )}
              {commType === 'reduce-scatter' && (
                <>
                  <li>• FSDP 反向传播</li>
                  <li>• ZeRO 优化器</li>
                  <li>• 梯度分片更新</li>
                </>
              )}
            </ul>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 p-4 rounded-r-xl">
            <p className="text-xs text-amber-800 dark:text-amber-300">
              <strong>性能提示：</strong>
              {commType === 'all-reduce'
                ? ' All-Reduce 是 DDP 最频繁的操作，使用 NCCL 后端可获得最佳性能。'
                : commType === 'gather'
                ? ' Gather 会导致主进程内存激增，大数据量时应使用流式处理。'
                : commType === 'reduce-scatter'
                ? ' Reduce-Scatter 是 FSDP 的核心，相比 All-Reduce 节省 50% 显存。'
                : ' 理解通信模式对优化分布式训练至关重要。'}
            </p>
          </div>
        </div>
      </div>

      {/* 底部对比表 */}
      <div className="mt-8 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg overflow-x-auto">
        <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
          通信原语对比
        </h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-slate-200 dark:border-slate-700">
              <th className="text-left p-3 font-semibold">操作</th>
              <th className="text-left p-3 font-semibold">输入</th>
              <th className="text-left p-3 font-semibold">输出</th>
              <th className="text-left p-3 font-semibold">通信量</th>
              <th className="text-left p-3 font-semibold">典型用途</th>
            </tr>
          </thead>
          <tbody className="text-slate-700 dark:text-slate-300">
            <tr className="border-b border-slate-100 dark:border-slate-800">
              <td className="p-3 font-semibold">All-Reduce</td>
              <td className="p-3">每个 GPU 不同值</td>
              <td className="p-3">每个 GPU 相同（平均）</td>
              <td className="p-3">O(N)</td>
              <td className="p-3">DDP 梯度同步</td>
            </tr>
            <tr className="border-b border-slate-100 dark:border-slate-800">
              <td className="p-3 font-semibold">Broadcast</td>
              <td className="p-3">GPU 0 有值</td>
              <td className="p-3">所有 GPU 相同</td>
              <td className="p-3">O(N)</td>
              <td className="p-3">参数初始化</td>
            </tr>
            <tr className="border-b border-slate-100 dark:border-slate-800">
              <td className="p-3 font-semibold">Gather</td>
              <td className="p-3">每个 GPU 不同值</td>
              <td className="p-3">仅 GPU 0 有全部</td>
              <td className="p-3">O(N)</td>
              <td className="p-3">收集评估结果</td>
            </tr>
            <tr className="border-b border-slate-100 dark:border-slate-800">
              <td className="p-3 font-semibold">All-Gather</td>
              <td className="p-3">每个 GPU 不同值</td>
              <td className="p-3">所有 GPU 有全部</td>
              <td className="p-3">O(N²)</td>
              <td className="p-3">FSDP 前向传播</td>
            </tr>
            <tr>
              <td className="p-3 font-semibold">Reduce-Scatter</td>
              <td className="p-3">每个 GPU 不同值</td>
              <td className="p-3">每个 GPU 不同部分</td>
              <td className="p-3">O(N)</td>
              <td className="p-3">FSDP 反向传播</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
