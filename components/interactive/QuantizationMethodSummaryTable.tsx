'use client'

import React from 'react'
import { CheckCircle2, XCircle, MinusCircle, Star } from 'lucide-react'

export default function QuantizationMethodSummaryTable() {
  const methods = [
    {
      name: 'GPTQ',
      calibration: true,
      quantTime: '5-10 min',
      inferenceSpeed: 4,
      accuracy: 5,
      finetune: false,
      scenario: '生产推理',
      highlight: '精度最高',
    },
    {
      name: 'AWQ',
      calibration: true,
      quantTime: '3-5 min',
      inferenceSpeed: 5,
      accuracy: 4,
      finetune: false,
      scenario: '低延迟推理',
      highlight: '速度最快',
    },
    {
      name: 'bitsandbytes',
      calibration: false,
      quantTime: '<1 min',
      inferenceSpeed: 3,
      accuracy: 3,
      finetune: true,
      scenario: 'QLoRA微调',
      highlight: '支持微调',
    },
    {
      name: 'HQQ',
      calibration: false,
      quantTime: '<1 min',
      inferenceSpeed: 3,
      accuracy: 2,
      finetune: false,
      scenario: '快速实验',
      highlight: '零校准',
    },
    {
      name: 'GGUF (llama.cpp)',
      calibration: false,
      quantTime: '2 min',
      inferenceSpeed: 4,
      accuracy: 4,
      finetune: false,
      scenario: 'CPU部署',
      highlight: '跨平台',
    },
    {
      name: 'EETQ (INT8)',
      calibration: false,
      quantTime: '<1 min',
      inferenceSpeed: 4,
      accuracy: 5,
      finetune: false,
      scenario: '高精度推理',
      highlight: '接近FP16',
    },
  ]

  const renderStars = (count: number) => {
    return (
      <div className="flex gap-0.5">
        {Array.from({ length: 5 }, (_, i) => (
          <Star
            key={i}
            className={`w-4 h-4 ${
              i < count
                ? 'fill-amber-400 text-amber-400'
                : 'text-slate-300'
            }`}
          />
        ))}
      </div>
    )
  }

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        量化方法对比总结
      </h3>

      {/* 桌面端表格 */}
      <div className="hidden md:block bg-white rounded-xl border border-slate-200 overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-gradient-to-r from-slate-100 to-slate-200 border-b-2 border-slate-300">
              <th className="text-left py-4 px-6 font-bold text-slate-700">方法</th>
              <th className="text-center py-4 px-4 font-bold text-slate-700">校准数据</th>
              <th className="text-center py-4 px-4 font-bold text-slate-700">量化时间</th>
              <th className="text-center py-4 px-4 font-bold text-slate-700">推理速度</th>
              <th className="text-center py-4 px-4 font-bold text-slate-700">精度</th>
              <th className="text-center py-4 px-4 font-bold text-slate-700">微调支持</th>
              <th className="text-left py-4 px-6 font-bold text-slate-700">推荐场景</th>
            </tr>
          </thead>
          <tbody>
            {methods.map((method, idx) => (
              <tr
                key={idx}
                className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
              >
                <td className="py-4 px-6">
                  <div className="font-bold text-slate-800">{method.name}</div>
                  <div className="text-xs text-blue-600 mt-1">{method.highlight}</div>
                </td>
                <td className="py-4 px-4 text-center">
                  {method.calibration ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500 mx-auto" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-500 mx-auto" />
                  )}
                </td>
                <td className="py-4 px-4 text-center">
                  <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                    {method.quantTime}
                  </span>
                </td>
                <td className="py-4 px-4">
                  <div className="flex justify-center">
                    {renderStars(method.inferenceSpeed)}
                  </div>
                </td>
                <td className="py-4 px-4">
                  <div className="flex justify-center">
                    {renderStars(method.accuracy)}
                  </div>
                </td>
                <td className="py-4 px-4 text-center">
                  {method.finetune ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500 mx-auto" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-500 mx-auto" />
                  )}
                </td>
                <td className="py-4 px-6">
                  <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">
                    {method.scenario}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 移动端卡片 */}
      <div className="md:hidden space-y-4">
        {methods.map((method, idx) => (
          <div
            key={idx}
            className="bg-white p-6 rounded-xl border border-slate-200"
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h4 className="font-bold text-lg text-slate-800">{method.name}</h4>
                <div className="text-sm text-blue-600 mt-1">{method.highlight}</div>
              </div>
              <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">
                {method.scenario}
              </span>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">校准数据</span>
                {method.calibration ? (
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-500" />
                )}
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">量化时间</span>
                <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                  {method.quantTime}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">推理速度</span>
                {renderStars(method.inferenceSpeed)}
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">精度</span>
                {renderStars(method.accuracy)}
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">微调支持</span>
                {method.finetune ? (
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-500" />
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 图例 */}
      <div className="mt-6 grid md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
          <h4 className="font-bold text-blue-800 mb-3">评分说明</h4>
          <div className="space-y-2 text-sm text-blue-700">
            <div className="flex items-center gap-2">
              {renderStars(5)}
              <span>最优</span>
            </div>
            <div className="flex items-center gap-2">
              {renderStars(4)}
              <span>优秀</span>
            </div>
            <div className="flex items-center gap-2">
              {renderStars(3)}
              <span>良好</span>
            </div>
            <div className="flex items-center gap-2">
              {renderStars(2)}
              <span>一般</span>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
          <h4 className="font-bold text-green-800 mb-3">快速选择指南</h4>
          <ul className="space-y-2 text-sm text-green-700">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span><strong>需要微调</strong> → bitsandbytes</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span><strong>追求速度</strong> → AWQ</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span><strong>追求精度</strong> → GPTQ / EETQ</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span><strong>CPU 部署</strong> → GGUF</span>
            </li>
          </ul>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        ⭐ 星级越高表示该维度性能越好
      </div>
    </div>
  )
}
