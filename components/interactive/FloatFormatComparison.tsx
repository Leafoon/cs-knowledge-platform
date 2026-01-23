'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Binary, Zap } from 'lucide-react'

export default function FloatFormatComparison() {
  const [selectedFormat, setSelectedFormat] = useState<'FP32' | 'FP16' | 'BF16'>('BF16')
  const [inputValue, setInputValue] = useState(3.14159)

  const formats = {
    FP32: {
      name: 'FP32 (Float32)',
      bits: 32,
      sign: 1,
      exponent: 8,
      mantissa: 23,
      bias: 127,
      range: '1.2e-38 ~ 3.4e38',
      precision: '~7 位有效数字',
      color: 'blue',
      usage: '科学计算、调试基准',
    },
    FP16: {
      name: 'FP16 (Half)',
      bits: 16,
      sign: 1,
      exponent: 5,
      mantissa: 10,
      bias: 15,
      range: '6.1e-5 ~ 6.55e4',
      precision: '~3 位有效数字',
      color: 'orange',
      usage: 'Volta/Turing GPU，需 loss scaling',
    },
    BF16: {
      name: 'BF16 (BFloat16)',
      bits: 16,
      sign: 1,
      exponent: 8,
      mantissa: 7,
      bias: 127,
      range: '1.2e-38 ~ 3.4e38',
      precision: '~2 位有效数字',
      color: 'green',
      usage: 'Ampere/Hopper GPU，深度学习首选',
    },
  }

  const current = formats[selectedFormat]

  // 简化的浮点数二进制表示（示例）
  const getBinaryRepresentation = () => {
    const sign = inputValue < 0 ? '1' : '0'
    
    // 简化：根据格式生成示例二进制
    if (selectedFormat === 'FP32') {
      return {
        sign: sign,
        exponent: '10000000',  // 示例指数
        mantissa: '10010010000111111011011',  // 23 位
      }
    } else if (selectedFormat === 'FP16') {
      return {
        sign: sign,
        exponent: '10000',  // 5 位
        mantissa: '1001001000',  // 10 位
      }
    } else {
      return {
        sign: sign,
        exponent: '10000000',  // 8 位
        mantissa: '1001001',  // 7 位
      }
    }
  }

  const binary = getBinaryRepresentation()

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Binary className="w-8 h-8 text-blue-600" />
        <h3 className="text-2xl font-bold text-slate-800">浮点数格式深度对比</h3>
      </div>

      {/* 格式选择 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {Object.entries(formats).map(([key, format]) => (
          <button
            key={key}
            onClick={() => setSelectedFormat(key as any)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedFormat === key
                ? `border-${format.color}-600 bg-${format.color}-50 shadow-lg`
                : 'border-slate-200 bg-white hover:border-slate-300'
            }`}
          >
            <div className={`text-lg font-bold ${
              selectedFormat === key ? `text-${format.color}-900` : 'text-slate-700'
            }`}>
              {format.name}
            </div>
            <div className="text-sm text-slate-600 mt-1">{format.bits} bits</div>
          </button>
        ))}
      </div>

      {/* Bit Layout 可视化 */}
      <motion.div
        key={selectedFormat}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white p-6 rounded-lg shadow-lg mb-6"
      >
        <h4 className="font-bold text-slate-800 mb-4">Bit Layout（位布局）</h4>
        
        <div className="flex items-center gap-2 mb-4">
          {/* Sign bit */}
          <div className="flex flex-col items-center">
            <div className="text-xs text-red-600 font-bold mb-1">符号位</div>
            <div className="px-4 py-3 bg-red-100 border-2 border-red-400 rounded font-mono text-lg">
              {binary.sign}
            </div>
            <div className="text-xs text-slate-600 mt-1">{current.sign} bit</div>
          </div>

          {/* Exponent */}
          <div className="flex flex-col items-center flex-1">
            <div className="text-xs text-blue-600 font-bold mb-1">指数位（范围）</div>
            <div className="w-full px-4 py-3 bg-blue-100 border-2 border-blue-400 rounded font-mono text-lg text-center">
              {binary.exponent}
            </div>
            <div className="text-xs text-slate-600 mt-1">{current.exponent} bits</div>
          </div>

          {/* Mantissa */}
          <div className="flex flex-col items-center flex-1">
            <div className="text-xs text-green-600 font-bold mb-1">尾数位（精度）</div>
            <div className="w-full px-4 py-3 bg-green-100 border-2 border-green-400 rounded font-mono text-lg text-center break-all">
              {binary.mantissa}
            </div>
            <div className="text-xs text-slate-600 mt-1">{current.mantissa} bits</div>
          </div>
        </div>

        {/* 公式 */}
        <div className="p-4 bg-slate-100 rounded border border-slate-300 font-mono text-sm">
          <strong>值计算公式</strong>: (-1)<sup>sign</sup> × 2<sup>(exponent - {current.bias})</sup> × (1 + mantissa)
        </div>
      </motion.div>

      {/* 性能指标对比 */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white p-5 rounded-lg shadow">
          <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-600" />
            动态范围
          </h4>
          <div className="text-3xl font-bold text-blue-600 mb-2">{current.range}</div>
          <div className="text-sm text-slate-600">
            {selectedFormat === 'FP16' ? (
              <span className="text-orange-600 font-semibold">
                ⚠️ 范围小，梯度易下溢（&lt;6e-5→0）
              </span>
            ) : selectedFormat === 'BF16' ? (
              <span className="text-green-600 font-semibold">
                ✓ 与 FP32 相同，训练稳定
              </span>
            ) : (
              '最大范围，但速度慢'
            )}
          </div>
        </div>

        <div className="bg-white p-5 rounded-lg shadow">
          <h4 className="font-bold text-slate-800 mb-3">精度（有效数字）</h4>
          <div className="text-3xl font-bold text-green-600 mb-2">{current.precision}</div>
          <div className="text-sm text-slate-600">
            {selectedFormat === 'BF16' ? (
              <span className="text-yellow-600">
                精度略低，但深度学习可容忍
              </span>
            ) : selectedFormat === 'FP16' ? (
              '精度中等，小心累积误差'
            ) : (
              '最高精度'
            )}
          </div>
        </div>
      </div>

      {/* 三格式对比表 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">关键差异总结</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-slate-300">
              <th className="text-left py-2 px-3">格式</th>
              <th className="text-center py-2 px-3">指数位</th>
              <th className="text-center py-2 px-3">尾数位</th>
              <th className="text-center py-2 px-3">动态范围</th>
              <th className="text-left py-2 px-3">最佳场景</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-200">
              <td className="py-3 px-3 font-bold text-blue-600">FP32</td>
              <td className="py-3 px-3 text-center font-mono">8</td>
              <td className="py-3 px-3 text-center font-mono">23</td>
              <td className="py-3 px-3 text-center text-xs">10<sup>±38</sup></td>
              <td className="py-3 px-3 text-slate-700">调试、科学计算</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-3 px-3 font-bold text-orange-600">FP16</td>
              <td className="py-3 px-3 text-center font-mono">5</td>
              <td className="py-3 px-3 text-center font-mono">10</td>
              <td className="py-3 px-3 text-center text-xs">6e<sup>±4</sup></td>
              <td className="py-3 px-3 text-slate-700">旧 GPU，需 loss scaling</td>
            </tr>
            <tr className={selectedFormat === 'BF16' ? 'bg-green-50' : ''}>
              <td className="py-3 px-3 font-bold text-green-600">BF16</td>
              <td className="py-3 px-3 text-center font-mono">8</td>
              <td className="py-3 px-3 text-center font-mono">7</td>
              <td className="py-3 px-3 text-center text-xs">10<sup>±38</sup></td>
              <td className="py-3 px-3 text-green-700 font-semibold">✓ 深度学习首选</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* 使用建议 */}
      <div className="mt-6 p-5 bg-gradient-to-r from-green-50 to-blue-50 border-2 border-green-300 rounded-lg">
        <h4 className="font-bold text-green-800 mb-2">🎯 推荐使用场景</h4>
        <div className="text-slate-700">{current.usage}</div>
        {selectedFormat === 'BF16' && (
          <div className="mt-3 text-sm text-green-700 font-semibold">
            ✓ 适用于 A100/H100/RTX 4090 等 Ampere/Hopper GPU<br/>
            ✓ 训练稳定，几乎无精度损失（&lt;0.1%）<br/>
            ✓ 速度提升 2-3 倍，显存节省 50%
          </div>
        )}
      </div>
    </div>
  )
}
