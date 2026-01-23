'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';

type ZeROStage = 'ddp' | 'zero1' | 'zero2' | 'zero3';

export default function ZeROStagesComparison() {
  const [selectedStage, setSelectedStage] = useState<ZeROStage>('zero3');
  const [numGPUs] = useState(4);
  const [modelSize] = useState(7); // 7B 参数

  // 计算显存占用（GB）
  const calculateMemory = (stage: ZeROStage) => {
    const paramsMemory = modelSize * 4; // FP32: 4 bytes/param
    const optimizerMemory = modelSize * 4 * 2; // AdamW: 2个状态
    const gradientMemory = modelSize * 4;

    switch (stage) {
      case 'ddp':
        return {
          params: paramsMemory,
          optimizer: optimizerMemory,
          gradients: gradientMemory,
          total: paramsMemory + optimizerMemory + gradientMemory,
        };
      case 'zero1':
        return {
          params: paramsMemory,
          optimizer: optimizerMemory / numGPUs,
          gradients: gradientMemory,
          total: paramsMemory + optimizerMemory / numGPUs + gradientMemory,
        };
      case 'zero2':
        return {
          params: paramsMemory,
          optimizer: optimizerMemory / numGPUs,
          gradients: gradientMemory / numGPUs,
          total: paramsMemory + (optimizerMemory + gradientMemory) / numGPUs,
        };
      case 'zero3':
        return {
          params: paramsMemory / numGPUs,
          optimizer: optimizerMemory / numGPUs,
          gradients: gradientMemory / numGPUs,
          total: (paramsMemory + optimizerMemory + gradientMemory) / numGPUs,
        };
    }
  };

  const stages: {
    [key in ZeROStage]: {
      name: string;
      description: string;
      sharding: string[];
      communication: string;
      color: string;
      fsdp: string;
    };
  } = {
    ddp: {
      name: 'DDP（无分片）',
      description: '传统 DistributedDataParallel，每个 GPU 保存完整模型状态',
      sharding: [],
      communication: 'All-Reduce（仅梯度）',
      color: 'slate',
      fsdp: 'NO_SHARD',
    },
    zero1: {
      name: 'ZeRO-1',
      description: '仅分片优化器状态，参数和梯度保持完整',
      sharding: ['优化器状态'],
      communication: 'All-Gather（更新时）',
      color: 'blue',
      fsdp: '不支持',
    },
    zero2: {
      name: 'ZeRO-2',
      description: '分片优化器状态和梯度，参数保持完整',
      sharding: ['优化器状态', '梯度'],
      communication: 'Reduce-Scatter（反向传播）',
      color: 'purple',
      fsdp: 'SHARD_GRAD_OP',
    },
    zero3: {
      name: 'ZeRO-3',
      description: '分片所有模型状态（参数、优化器、梯度）',
      sharding: ['参数', '优化器状态', '梯度'],
      communication: 'All-Gather + Reduce-Scatter（前向/反向）',
      color: 'green',
      fsdp: 'FULL_SHARD',
    },
  };

  const currentStage = stages[selectedStage];
  const memory = calculateMemory(selectedStage);

  return (
    <div className="w-full max-w-6xl mx-auto bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-950 dark:to-blue-950 rounded-2xl shadow-2xl p-8">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-slate-100 mb-3">
          ZeRO 优化器阶段对比
        </h3>
        <p className="text-slate-600 dark:text-slate-400">
          {modelSize}B 参数模型 | {numGPUs} × GPU | FP32 训练
        </p>
      </div>

      {/* 阶段选择器 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {(Object.keys(stages) as ZeROStage[]).map((stage) => {
          const stageInfo = stages[stage];
          return (
            <motion.button
              key={stage}
              onClick={() => setSelectedStage(stage)}
              className={`p-4 rounded-xl font-semibold transition-all ${
                selectedStage === stage
                  ? `bg-${stageInfo.color}-500 text-white shadow-lg scale-105`
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:shadow-md'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="text-lg mb-1">{stageInfo.name}</div>
              <div className="text-xs opacity-80">
                {stageInfo.fsdp !== '不支持' ? `FSDP: ${stageInfo.fsdp}` : stageInfo.fsdp}
              </div>
            </motion.button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 显存分解图 */}
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
            显存占用分解（每个 GPU）
          </h4>
          
          <div className="space-y-4 mb-6">
            {/* 参数 */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  模型参数
                </span>
                <span className="text-lg font-bold text-blue-600">
                  {memory.params.toFixed(1)} GB
                </span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4">
                <motion.div
                  className="bg-blue-500 h-4 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(memory.params / memory.total) * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>

            {/* 优化器 */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  优化器状态
                </span>
                <span className="text-lg font-bold text-purple-600">
                  {memory.optimizer.toFixed(1)} GB
                </span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4">
                <motion.div
                  className="bg-purple-500 h-4 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(memory.optimizer / memory.total) * 100}%` }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                />
              </div>
            </div>

            {/* 梯度 */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  梯度
                </span>
                <span className="text-lg font-bold text-orange-600">
                  {memory.gradients.toFixed(1)} GB
                </span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4">
                <motion.div
                  className="bg-orange-500 h-4 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(memory.gradients / memory.total) * 100}%` }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                />
              </div>
            </div>
          </div>

          {/* 总显存 */}
          <div className="bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 rounded-xl p-4 text-center">
            <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">
              总显存占用
            </div>
            <div className="text-4xl font-bold text-slate-800 dark:text-slate-100">
              {memory.total.toFixed(1)} GB
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-500 mt-1">
              每个 GPU（不含激活值）
            </div>
          </div>
        </div>

        {/* 详细信息 */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
            <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-3">
              📝 阶段说明
            </h4>
            <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              {currentStage.description}
            </p>
          </div>

          <div className={`bg-${currentStage.color}-50 dark:bg-${currentStage.color}-900/30 rounded-xl p-6`}>
            <h4 className={`text-lg font-bold text-${currentStage.color}-800 dark:text-${currentStage.color}-200 mb-3`}>
              🔄 分片内容
            </h4>
            {currentStage.sharding.length > 0 ? (
              <ul className={`space-y-2 text-sm text-${currentStage.color}-700 dark:text-${currentStage.color}-300`}>
                {currentStage.sharding.map((item, idx) => (
                  <li key={idx} className="flex items-center gap-2">
                    <span className={`text-${currentStage.color}-600 dark:text-${currentStage.color}-400`}>
                      ✓
                    </span>
                    {item}
                  </li>
                ))}
              </ul>
            ) : (
              <p className={`text-sm text-${currentStage.color}-700 dark:text-${currentStage.color}-300`}>
                无分片（完整复制）
              </p>
            )}
          </div>

          <div className="bg-slate-100 dark:bg-slate-800/50 rounded-xl p-4">
            <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">
              通信模式
            </h4>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              {currentStage.communication}
            </p>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 p-4 rounded-r-xl">
            <p className="text-xs text-amber-800 dark:text-amber-300">
              <strong>配置方式：</strong>
              {currentStage.fsdp !== '不支持' ? (
                <span>
                  {' '}
                  FSDP: <code className="bg-amber-100 dark:bg-amber-900/50 px-2 py-1 rounded">
                    ShardingStrategy.{currentStage.fsdp}
                  </code>
                </span>
              ) : (
                ' PyTorch FSDP 不支持 ZeRO-1，请使用 DeepSpeed。'
              )}
            </p>
          </div>
        </div>
      </div>

      {/* 对比表 */}
      <div className="mt-8 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg overflow-x-auto">
        <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
          完整对比（{modelSize}B 模型，{numGPUs} GPU）
        </h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-slate-200 dark:border-slate-700">
              <th className="text-left p-3 font-semibold">阶段</th>
              <th className="text-right p-3 font-semibold">参数</th>
              <th className="text-right p-3 font-semibold">优化器</th>
              <th className="text-right p-3 font-semibold">梯度</th>
              <th className="text-right p-3 font-semibold">总显存/GPU</th>
              <th className="text-right p-3 font-semibold">节省比例</th>
            </tr>
          </thead>
          <tbody className="text-slate-700 dark:text-slate-300">
            {(Object.keys(stages) as ZeROStage[]).map((stage) => {
              const mem = calculateMemory(stage);
              const ddpMem = calculateMemory('ddp').total;
              const savings = ((1 - mem.total / ddpMem) * 100).toFixed(0);
              return (
                <tr
                  key={stage}
                  className={`border-b border-slate-100 dark:border-slate-800 ${
                    stage === selectedStage ? `bg-${stages[stage].color}-50 dark:bg-${stages[stage].color}-900/20` : ''
                  }`}
                >
                  <td className="p-3 font-semibold">{stages[stage].name}</td>
                  <td className="p-3 text-right">{mem.params.toFixed(1)} GB</td>
                  <td className="p-3 text-right">{mem.optimizer.toFixed(1)} GB</td>
                  <td className="p-3 text-right">{mem.gradients.toFixed(1)} GB</td>
                  <td className="p-3 text-right">
                    <span className="font-bold text-lg">{mem.total.toFixed(1)} GB</span>
                  </td>
                  <td className="p-3 text-right">
                    <span
                      className={`font-bold ${
                        Number(savings) > 0 ? 'text-green-600' : 'text-slate-500'
                      }`}
                    >
                      {Number(savings) > 0 ? '-' : ''}
                      {savings}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 底部说明 */}
      <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4 rounded-r-xl">
        <p className="text-sm text-blue-800 dark:text-blue-300">
          <strong>关键洞察：</strong>
          ZeRO-3 在 4 GPU 下将显存从 112 GB 降至 28 GB（节省 75%），使得单卡 40GB 可训练 7B 模型。
          通信开销随分片级别增加，但显存节省更显著，适合大模型训练。
        </p>
      </div>
    </div>
  );
}
