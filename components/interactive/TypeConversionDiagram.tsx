'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const conversions = [
  {
    id: 'py2tvm',
    name: 'Python → TVM',
    from: 'Python',
    to: 'TVM',
    color: 'from-blue-500 to-indigo-600',
    steps: [
      { from: 'int / float', to: 'tvm.tir.IntImm / FloatImm', desc: 'Python标量转TVM常量节点' },
      { from: 'str', to: 'tvm.tir.StringImm', desc: '字符串转TVM字符串常量' },
      { from: 'list', to: 'tvm.ir.container.Array', desc: 'Python列表转TVM Array' },
      { from: 'dict', to: 'tvm.ir.container.Map', desc: 'Python字典转TVM Map' },
      { from: 'numpy.ndarray', to: 'tvm.nd.array', desc: 'NumPy数组转TVM NDArray（可能拷贝）' },
      { from: 'callable', to: 'tvm.ir.Op', desc: '函数转TVM算子引用' },
    ],
  },
  {
    id: 'tvm2cpp',
    name: 'TVM → C++',
    from: 'TVM',
    to: 'C++',
    color: 'from-purple-500 to-violet-600',
    steps: [
      { from: 'IntImm(64)', to: 'int64_t', desc: 'TVM整型常量转C++整数' },
      { from: 'FloatImm(64)', to: 'double', desc: 'TVM浮点常量转C++浮点' },
      { from: 'tvm::runtime::NDArray', to: 'DLTensor*', desc: 'NDArray转DLPack张量指针' },
      { from: 'tvm::runtime::String', to: 'std::string', desc: 'TVM字符串转C++字符串' },
      { from: 'tvm::runtime::Array', to: 'std::vector', desc: 'TVM Array转C++向量' },
      { from: 'PrimFunc', to: 'LoweredFunc', desc: 'TIR函数转低级函数表示' },
    ],
  },
  {
    id: 'cpp2tvm',
    name: 'C++ → TVM',
    from: 'C++',
    to: 'TVM',
    color: 'from-indigo-500 to-blue-600',
    steps: [
      { from: 'int64_t', to: 'IntImm(64, val)', desc: 'C++整数转TVM常量' },
      { from: 'DLTensor*', to: 'NDArray', desc: 'DLPack张量包装为NDArray（零拷贝）' },
      { from: 'std::string', to: 'tvm::runtime::String', desc: 'C++字符串转TVM字符串' },
      { from: 'TVMRetValue', to: '对应TVM对象', desc: '返回值自动转为对应类型' },
      { from: 'PackedFunc', to: 'tvm.runtime.PackedFunc', desc: 'C++函数暴露为Python可调用对象' },
    ],
  },
];

export default function TypeConversionDiagram() {
  const [selectedConversion, setSelectedConversion] = useState(0);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  const current = conversions[selectedConversion];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        类型转换图
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        Python ↔ TVM ↔ C++ 之间的类型转换关系
      </p>

      <div className="flex items-center justify-center gap-4 mb-8">
        {conversions.map((conv, i) => (
          <button
            key={conv.id}
            onClick={() => { setSelectedConversion(i); setSelectedStep(null); }}
            className={`px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${
              i === selectedConversion
                ? `bg-gradient-to-r ${conv.color} text-white shadow-lg`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {conv.name}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-center gap-6 mb-6">
        <motion.div
          key={current.from}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-gray-800 rounded-xl px-6 py-3 border border-gray-700"
        >
          <span className="text-white font-medium">{current.from}</span>
        </motion.div>
        <motion.div
          key={`arrow-${selectedConversion}`}
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-2xl"
        >
          <span className="text-gray-500">→</span>
        </motion.div>
        <motion.div
          key={current.to}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-gray-800 rounded-xl px-6 py-3 border border-gray-700"
        >
          <span className="text-white font-medium">{current.to}</span>
        </motion.div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selectedConversion}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
        >
          <div className="bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden">
            <div className="grid grid-cols-[1fr,40px,1fr,1fr] gap-0">
              <div className="px-4 py-3 bg-gray-800/80 border-b border-r border-gray-700 text-xs text-gray-400 font-medium">
                {current.from} 类型
              </div>
              <div className="px-2 py-3 bg-gray-800/80 border-b border-r border-gray-700 text-center text-xs text-gray-500" />
              <div className="px-4 py-3 bg-gray-800/80 border-b border-r border-gray-700 text-xs text-gray-400 font-medium">
                {current.to} 类型
              </div>
              <div className="px-4 py-3 bg-gray-800/80 border-b border-gray-700 text-xs text-gray-400 font-medium">
                说明
              </div>

              {current.steps.map((step, i) => (
                <React.Fragment key={i}>
                  <motion.div
                    onClick={() => setSelectedStep(selectedStep === i ? null : i)}
                    className={`px-4 py-3 border-b border-r border-gray-700/50 cursor-pointer transition-colors ${
                      selectedStep === i ? 'bg-blue-900/20' : 'hover:bg-gray-800/40'
                    }`}
                    whileHover={{ x: 2 }}
                  >
                    <code className="text-blue-400 font-mono text-xs">{step.from}</code>
                  </motion.div>
                  <div className="px-2 py-3 border-b border-r border-gray-700/50 flex items-center justify-center">
                    <span className="text-gray-600 text-xs">→</span>
                  </div>
                  <motion.div
                    onClick={() => setSelectedStep(selectedStep === i ? null : i)}
                    className={`px-4 py-3 border-b border-r border-gray-700/50 cursor-pointer transition-colors ${
                      selectedStep === i ? 'bg-purple-900/20' : 'hover:bg-gray-800/40'
                    }`}
                    whileHover={{ x: 2 }}
                  >
                    <code className="text-purple-400 font-mono text-xs">{step.to}</code>
                  </motion.div>
                  <div className="px-4 py-3 border-b border-gray-700/50 text-xs text-gray-400">
                    {step.desc}
                  </div>
                </React.Fragment>
              ))}
            </div>
          </div>

          <AnimatePresence>
            {selectedStep !== null && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="mt-4 bg-gray-800/60 rounded-xl p-4 border border-indigo-500/20"
              >
                <div className="flex items-center gap-3 mb-2">
                  <code className="text-blue-400 font-mono text-sm">{current.steps[selectedStep].from}</code>
                  <span className="text-gray-500">→</span>
                  <code className="text-purple-400 font-mono text-sm">{current.steps[selectedStep].to}</code>
                </div>
                <p className="text-gray-300 text-sm">{current.steps[selectedStep].desc}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </AnimatePresence>

      <div className="mt-6 bg-gray-800/40 rounded-xl border border-gray-700 p-4">
        <div className="text-xs text-gray-500 mb-2">常见转换速查</div>
        <div className="flex flex-wrap gap-2">
          {['NDArray ↔ DLTensor (零拷贝)', 'PackedFunc → Python callable', 'numpy → NDArray (可能拷贝)', 'TVMRetValue → 自动类型推断'].map((item) => (
            <span key={item} className="text-xs bg-gray-900/60 text-gray-400 px-3 py-1.5 rounded-lg border border-gray-700/50">
              {item}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
