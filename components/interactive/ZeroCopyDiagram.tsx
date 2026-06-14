'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const steps = [
  {
    id: 'source',
    name: '源 NDArray',
    desc: 'CPU 上的原始 NDArray，持有底层 DLTesnor 数据',
    color: 'from-blue-500 to-indigo-600',
    detail: '包含 shape、dtype、data 指针、device 信息',
  },
  {
    id: 'share',
    name: '共享内存池',
    desc: 'DLTensor 的 data 指针指向同一块物理内存',
    color: 'from-purple-500 to-violet-600',
    detail: '引用计数管理，多个 NDArray 可共享同一块内存',
  },
  {
    id: 'target',
    name: '目标 NDArray',
    desc: '新创建的 NDArray，直接引用源数据，无数据拷贝',
    color: 'from-indigo-500 to-blue-600',
    detail: '不同 shape/view 可能，但底层 data 指针相同',
  },
];

const scenarios = [
  {
    name: 'CPU→CPU 传输',
    from: 'CPU',
    to: 'CPU',
    zeroCopy: true,
    desc: '同设备直接共享指针，真正零拷贝',
    code: `src = tvm.nd.array(np_data, device=tvm.cpu())
dst = src  # 零拷贝，共享同一内存`,
    color: 'from-green-500 to-emerald-600',
  },
  {
    name: 'CPU→GPU 传输',
    from: 'CPU',
    to: 'GPU',
    zeroCopy: false,
    desc: '跨设备必须拷贝数据到 GPU 显存',
    code: `cpu_arr = tvm.nd.array(np_data, device=tvm.cpu())
gpu_arr = cpu_arr.copyto(tvm.gpu())  # 数据拷贝`,
    color: 'from-amber-500 to-orange-600',
  },
  {
    name: 'GPU→CPU 传回',
    from: 'GPU',
    to: 'CPU',
    zeroCopy: false,
    desc: '从显存拷贝回主机内存',
    code: `gpu_arr = tvm.nd.array(np_data, device=tvm.gpu())
cpu_arr = gpu_arr.copyto(tvm.cpu())  # 数据拷贝`,
    color: 'from-red-500 to-rose-600',
  },
  {
    name: 'DLPack 零拷贝',
    from: 'PyTorch',
    to: 'TVM',
    zeroCopy: true,
    desc: '通过 DLPack 协议共享底层内存，无拷贝',
    code: `torch_tensor = torch.randn(3, 4)
dlpack = torch_tensor.to_dlpack()
tvm_arr = tvm.nd.from_dlpack(dlpack)  # 零拷贝`,
    color: 'from-cyan-500 to-blue-600',
  },
  {
    name: '同GPU视图',
    from: 'GPU',
    to: 'GPU (View)',
    zeroCopy: true,
    desc: '同一 GPU 上创建数据视图，共享显存',
    code: `arr = tvm.nd.array(np_data, device=tvm.gpu())
# reshape/view 操作，底层数据不变
view = arr.reshape(new_shape)  # 零拷贝`,
    color: 'from-violet-500 to-purple-600',
  },
];

export default function ZeroCopyDiagram() {
  const [selectedScenario, setSelectedScenario] = useState(0);
  const [showSteps, setShowSteps] = useState(false);
  const scenario = scenarios[selectedScenario];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        零拷贝机制图
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        NDArray 共享内存的零拷贝流程与跨设备传输
      </p>

      <div className="flex flex-wrap gap-2 justify-center mb-8">
        {scenarios.map((s, i) => (
          <button
            key={s.name}
            onClick={() => { setSelectedScenario(i); setShowSteps(false); }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              i === selectedScenario
                ? `bg-gradient-to-r ${s.color} text-white shadow-lg`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {s.name}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selectedScenario}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
        >
          <div className="bg-gray-800/40 rounded-xl border border-gray-700 p-6 mb-6">
            <div className="flex items-center justify-center gap-6 mb-4">
              <motion.div
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                className="bg-gray-900/80 rounded-xl px-6 py-4 border border-blue-500/30"
              >
                <div className="text-blue-400 text-xs font-medium mb-1">源</div>
                <div className="text-white font-medium">{scenario.from}</div>
              </motion.div>

              <div className="flex flex-col items-center">
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: 0.2 }}
                  className={`h-1 w-20 rounded ${
                    scenario.zeroCopy
                      ? 'bg-gradient-to-r from-green-500 to-emerald-500'
                      : 'bg-gradient-to-r from-amber-500 to-red-500'
                  }`}
                />
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className={`text-xs font-medium mt-1 ${
                    scenario.zeroCopy ? 'text-green-400' : 'text-amber-400'
                  }`}
                >
                  {scenario.zeroCopy ? '零拷贝 ✓' : '数据拷贝 ✗'}
                </motion.span>
              </div>

              <motion.div
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                className="bg-gray-900/80 rounded-xl px-6 py-4 border border-purple-500/30"
              >
                <div className="text-purple-400 text-xs font-medium mb-1">目标</div>
                <div className="text-white font-medium">{scenario.to}</div>
              </motion.div>
            </div>

            <p className="text-center text-gray-300 text-sm">{scenario.desc}</p>
          </div>

          <div className="bg-gray-950 rounded-xl border border-gray-700 overflow-hidden mb-6">
            <div className="flex items-center gap-2 px-4 py-2 bg-gray-800/80 border-b border-gray-700">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/70" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
                <div className="w-3 h-3 rounded-full bg-green-500/70" />
              </div>
              <span className="text-xs text-gray-500 ml-2">example.py</span>
            </div>
            <pre className="p-4 text-sm font-mono text-gray-300 overflow-x-auto">
              {scenario.code}
            </pre>
          </div>
        </motion.div>
      </AnimatePresence>

      <button
        onClick={() => setShowSteps(!showSteps)}
        className="w-full text-left px-4 py-3 bg-gray-800/40 rounded-xl border border-gray-700 hover:border-gray-600 transition-all flex items-center justify-between"
      >
        <span className="text-sm text-gray-300">零拷贝内部机制</span>
        <motion.span animate={{ rotate: showSteps ? 180 : 0 }} className="text-gray-500">
          ▾
        </motion.span>
      </button>

      <AnimatePresence>
        {showSteps && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-3 space-y-3">
              {steps.map((step, i) => (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.15 }}
                  className="flex items-start gap-4 bg-gray-800/40 rounded-xl p-4 border border-gray-700"
                >
                  <div
                    className={`flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br ${step.color} flex items-center justify-center text-white font-bold text-sm`}
                  >
                    {i + 1}
                  </div>
                  <div>
                    <h4 className="text-white font-medium text-sm mb-1">{step.name}</h4>
                    <p className="text-gray-400 text-xs">{step.desc}</p>
                    <p className="text-gray-500 text-xs mt-1">{step.detail}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-6 bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden">
        <div className="px-4 py-3 bg-gray-800/80 border-b border-gray-700">
          <span className="text-sm font-medium text-gray-300">传输方式对比</span>
        </div>
        <div className="divide-y divide-gray-700/50">
          {scenarios.map((s) => (
            <div key={s.name} className="flex items-center px-4 py-2.5 gap-4">
              <span className="text-xs text-gray-300 w-32">{s.name}</span>
              <span className={`text-xs font-medium ${s.zeroCopy ? 'text-green-400' : 'text-amber-400'}`}>
                {s.zeroCopy ? '零拷贝 ✓' : '需拷贝 ✗'}
              </span>
              <span className="text-xs text-gray-500 flex-1">{s.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
