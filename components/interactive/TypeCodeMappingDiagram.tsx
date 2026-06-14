'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const typeCodeCategories = [
  {
    category: '整数类型',
    icon: '🔢',
    color: 'from-blue-500 to-indigo-600',
    types: [
      { code: 'kDLInt', value: 0, bits: '8/16/32/64', desc: '有符号整数', tvmType: 'int8, int16, int32, int64' },
      { code: 'kDLUInt', value: 1, bits: '1/8/16/32/64', desc: '无符号整数', tvmType: 'uint1, uint8, uint16, uint32, uint64' },
    ],
  },
  {
    category: '浮点类型',
    icon: '🔢',
    color: 'from-purple-500 to-violet-600',
    types: [
      { code: 'kDLFloat', value: 2, bits: '16/32/64', desc: '浮点数', tvmType: 'float16, float32, float64' },
      { code: 'kDLBfloat', value: 4, bits: '16', desc: 'Brain浮点数', tvmType: 'bfloat16' },
    ],
  },
  {
    category: '特殊类型',
    icon: '⚡',
    color: 'from-indigo-500 to-blue-600',
    types: [
      { code: 'kDLHandle', value: 3, bits: '64', desc: '句柄/指针', tvmType: 'handle' },
      { code: 'kFloat8_e4m3fn', value: 5, bits: '8', desc: 'FP8 E4M3', tvmType: 'float8_e4m3fn' },
      { code: 'kFloat8_e5m2', value: 6, bits: '8', desc: 'FP8 E5M2', tvmType: 'float8_e5m2' },
    ],
  },
];

export default function TypeCodeMappingDiagram() {
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [selectedBits, setSelectedBits] = useState<number>(32);

  const allTypes = typeCodeCategories.flatMap((c) => c.types);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        类型代码映射图
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        TVM 中 kFloat / kInt / kHandle 等类型代码的定义与映射
      </p>

      <div className="bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden mb-6">
        <div className="px-4 py-3 bg-gray-800/80 border-b border-gray-700">
          <span className="text-sm font-medium text-gray-300">DLDataTypeCode 枚举</span>
        </div>
        <div className="divide-y divide-gray-700/50">
          {allTypes.map((t) => (
            <motion.div
              key={t.code}
              whileHover={{ backgroundColor: 'rgba(139, 92, 246, 0.05)' }}
              onClick={() => setSelectedType(selectedType === t.code ? null : t.code)}
              className="flex items-center px-4 py-3 gap-4 cursor-pointer"
            >
              <code className="text-indigo-400 font-mono text-sm min-w-[140px]">{t.code}</code>
              <span className="text-amber-400 font-mono text-sm w-8 text-center">{t.value}</span>
              <span className="text-gray-500 text-xs w-20">{t.bits} bits</span>
              <span className="text-gray-400 text-sm flex-1">{t.desc}</span>
              {selectedType === t.code && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-emerald-400 text-xs"
                >
                  {t.tvmType}
                </motion.span>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {typeCodeCategories.map((cat) => (
          <div key={cat.category} className="bg-gray-800/40 rounded-xl border border-gray-700 p-4">
            <div className="text-sm font-medium text-white mb-3 flex items-center gap-2">
              <span>{cat.icon}</span>
              {cat.category}
            </div>
            <div className="space-y-2">
              {cat.types.map((t) => (
                <motion.div
                  key={t.code}
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setSelectedType(selectedType === t.code ? null : t.code)}
                  className={`rounded-lg p-3 border cursor-pointer transition-all ${
                    selectedType === t.code
                      ? 'bg-gray-800 border-purple-500/40'
                      : 'bg-gray-900/50 border-gray-700/50 hover:border-gray-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <code className="text-white font-mono text-xs">{t.code}</code>
                    <span className="text-gray-500 text-xs font-mono">={t.value}</span>
                  </div>
                  <div className="text-gray-500 text-xs">{t.desc}</div>
                  {selectedType === t.code && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-2 pt-2 border-t border-gray-700"
                    >
                      <div className="text-xs text-gray-400 mb-1">TVM 类型名称:</div>
                      <div className="flex flex-wrap gap-1">
                        {t.tvmType.split(', ').map((name) => (
                          <span
                            key={name}
                            className="text-xs bg-indigo-900/40 text-indigo-300 px-2 py-0.5 rounded"
                          >
                            {name}
                          </span>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 bg-gray-800/40 rounded-xl border border-gray-700 p-4">
        <div className="text-sm text-gray-300 mb-3">位宽组合示例</div>
        <div className="flex items-center gap-4">
          <div className="flex gap-2">
            {[8, 16, 32, 64].map((bits) => (
              <button
                key={bits}
                onClick={() => setSelectedBits(bits)}
                className={`px-3 py-1.5 rounded text-xs font-mono transition-all ${
                  selectedBits === bits
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {bits}-bit
              </button>
            ))}
          </div>
          <div className="flex-1 bg-gray-900 rounded-lg p-3">
            <code className="text-sm font-mono">
              <span className="text-purple-400">DLDataType</span>
              {'{ '}
              <span className="text-blue-400">code</span>=
              <span className="text-amber-400">kDLFloat</span>,{' '}
              <span className="text-blue-400">bits</span>=
              <span className="text-amber-400">{selectedBits}</span>,{' '}
              <span className="text-blue-400">lanes</span>=
              <span className="text-amber-400">1</span>
              {' }'}
              <span className="text-gray-500"> → </span>
              <span className="text-emerald-400">float{selectedBits}</span>
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}
