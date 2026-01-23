'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type CodeType = 'single-gpu' | 'ddp-manual' | 'accelerate';

export default function AccelerateWorkflow() {
  const [selectedCode, setSelectedCode] = useState<CodeType>('single-gpu');

  const codeExamples = {
    'single-gpu': {
      title: '单 GPU 代码（原始）',
      lines: [
        'import torch',
        'from transformers import AutoModel',
        'from torch.utils.data import DataLoader',
        '',
        '# 模型加载',
        'model = AutoModel.from_pretrained("bert-base")',
        'model.to("cuda")',
        'optimizer = torch.optim.AdamW(model.parameters())',
        '',
        'dataloader = DataLoader(dataset, batch_size=32)',
        '',
        '# 训练循环',
        'for batch in dataloader:',
        '    batch = {k: v.to("cuda") for k, v in batch.items()}',
        '    outputs = model(**batch)',
        '    loss = outputs.loss',
        '    ',
        '    loss.backward()',
        '    optimizer.step()',
        '    optimizer.zero_grad()',
      ],
      color: 'blue',
      complexity: 'Low',
      scalability: '单卡',
    },
    'ddp-manual': {
      title: '多 GPU DDP（手动实现）',
      lines: [
        'import torch',
        'import torch.distributed as dist',
        'from transformers import AutoModel',
        'from torch.utils.data import DataLoader, DistributedSampler',
        '',
        '# 初始化进程组',
        'dist.init_process_group(backend="nccl")',
        'local_rank = int(os.environ["LOCAL_RANK"])',
        '',
        '# 模型加载',
        'model = AutoModel.from_pretrained("bert-base")',
        'model = model.to(local_rank)',
        'model = torch.nn.parallel.DistributedDataParallel(',
        '    model, device_ids=[local_rank]',
        ')',
        '',
        '# 数据加载（需要 DistributedSampler）',
        'sampler = DistributedSampler(dataset)',
        'dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)',
        'optimizer = torch.optim.AdamW(model.parameters())',
        '',
        '# 训练循环',
        'for batch in dataloader:',
        '    batch = {k: v.to(local_rank) for k, v in batch.items()}',
        '    outputs = model(**batch)',
        '    loss = outputs.loss',
        '    ',
        '    loss.backward()',
        '    optimizer.step()',
        '    optimizer.zero_grad()',
      ],
      color: 'red',
      complexity: 'High',
      scalability: '多卡（需大量修改）',
    },
    'accelerate': {
      title: 'Accelerate 统一代码',
      lines: [
        'import torch',
        'from transformers import AutoModel',
        'from torch.utils.data import DataLoader',
        'from accelerate import Accelerator  # ✅ 添加',
        '',
        '# ✅ 创建 Accelerator',
        'accelerator = Accelerator()',
        '',
        '# 模型加载（无需手动 .to(device)）',
        'model = AutoModel.from_pretrained("bert-base")',
        'optimizer = torch.optim.AdamW(model.parameters())',
        'dataloader = DataLoader(dataset, batch_size=32)',
        '',
        '# ✅ 使用 prepare() 包装',
        'model, optimizer, dataloader = accelerator.prepare(',
        '    model, optimizer, dataloader',
        ')',
        '',
        '# 训练循环（无需手动移动数据）',
        'for batch in dataloader:',
        '    outputs = model(**batch)',
        '    loss = outputs.loss',
        '    ',
        '    # ✅ 使用 accelerator.backward()',
        '    accelerator.backward(loss)',
        '    optimizer.step()',
        '    optimizer.zero_grad()',
      ],
      color: 'green',
      complexity: 'Low',
      scalability: '单卡/多卡/混合精度',
    },
  };

  const currentCode = codeExamples[selectedCode];

  const diffHighlights: { [key in CodeType]: number[] } = {
    'single-gpu': [],
    'ddp-manual': [5, 6, 7, 15, 16, 17, 22],
    'accelerate': [3, 5, 6, 13, 14, 15, 16, 22, 23],
  };

  return (
    <div className="w-full max-w-6xl mx-auto bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-2xl p-8">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-slate-100 mb-3">
          Accelerate 代码转换对比
        </h3>
        <p className="text-slate-600 dark:text-slate-400">
          从单卡到多卡，Accelerate 仅需 3 行修改
        </p>
      </div>

      {/* 选项卡 */}
      <div className="flex gap-4 mb-6 justify-center flex-wrap">
        {(Object.keys(codeExamples) as CodeType[]).map((type) => {
          const code = codeExamples[type];
          return (
            <button
              key={type}
              onClick={() => setSelectedCode(type)}
              className={`px-6 py-3 rounded-xl font-semibold transition-all ${
                selectedCode === type
                  ? `bg-${code.color}-500 text-white shadow-lg scale-105`
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:shadow-md'
              }`}
            >
              {code.title}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* 代码区域 */}
        <div className="lg:col-span-2 bg-slate-900 rounded-xl p-6 overflow-hidden">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="ml-4 text-slate-400 text-sm font-mono">train.py</span>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={selectedCode}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.3 }}
              className="font-mono text-sm"
            >
              {currentCode.lines.map((line, idx) => {
                const isHighlighted = diffHighlights[selectedCode].includes(idx);
                const isComment = line.trim().startsWith('#');
                const isImport = line.trim().startsWith('import') || line.trim().startsWith('from');

                return (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: idx * 0.02 }}
                    className={`py-1 px-4 rounded ${
                      isHighlighted
                        ? `bg-${currentCode.color}-500/20 border-l-4 border-${currentCode.color}-500`
                        : ''
                    }`}
                  >
                    <span className="text-slate-500 select-none mr-4 inline-block w-6 text-right">
                      {idx + 1}
                    </span>
                    <span
                      className={`${
                        isComment
                          ? 'text-green-400'
                          : isImport
                          ? 'text-purple-400'
                          : 'text-slate-300'
                      }`}
                    >
                      {line || ' '}
                    </span>
                  </motion.div>
                );
              })}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* 信息面板 */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
            <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
              特性对比
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-600 dark:text-slate-400">代码复杂度</span>
                <span
                  className={`font-bold ${
                    currentCode.complexity === 'Low'
                      ? 'text-green-600'
                      : currentCode.complexity === 'Medium'
                      ? 'text-yellow-600'
                      : 'text-red-600'
                  }`}
                >
                  {currentCode.complexity}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600 dark:text-slate-400">扩展性</span>
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  {currentCode.scalability}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600 dark:text-slate-400">修改行数</span>
                <span className="text-2xl font-bold text-blue-600">
                  {diffHighlights[selectedCode].length}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 rounded-xl p-6">
            <h4 className="text-lg font-bold text-green-800 dark:text-green-200 mb-3">
              💡 Accelerate 优势
            </h4>
            <ul className="space-y-2 text-sm text-green-700 dark:text-green-300">
              <li className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400 mt-1">✓</span>
                <span>统一代码：单卡/多卡无需修改</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400 mt-1">✓</span>
                <span>自动设备管理：无需手动 .to(device)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400 mt-1">✓</span>
                <span>混合精度：自动 FP16/BF16 支持</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400 mt-1">✓</span>
                <span>梯度累积：零配置集成</span>
              </li>
            </ul>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4">
            <h4 className="text-sm font-bold text-blue-800 dark:text-blue-200 mb-2">
              启动命令
            </h4>
            <div className="bg-slate-900 rounded-lg p-3 font-mono text-xs text-green-400">
              {selectedCode === 'single-gpu' && '# 单 GPU\npython train.py'}
              {selectedCode === 'ddp-manual' &&
                '# 多 GPU（需要环境变量）\ntorchrun --nproc_per_node=4 train.py'}
              {selectedCode === 'accelerate' &&
                '# 自动适配\naccelerate launch --num_processes=4 train.py'}
            </div>
          </div>
        </div>
      </div>

      {/* 底部说明 */}
      <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 p-4 rounded-r-xl">
        <p className="text-sm text-amber-800 dark:text-amber-300">
          <strong>关键优势：</strong>
          使用 Accelerate 后，同一份代码可在单 GPU、多 GPU DDP、FSDP、DeepSpeed、TPU
          等环境无缝切换，仅需修改启动命令或配置文件。
        </p>
      </div>
    </div>
  );
}
