"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface Parameter {
  name: string;
  key: string;
  min: number;
  max: number;
  step: number;
  default: number;
  description: string;
  impact: string;
}

const parameters: Parameter[] = [
  {
    name: "Temperature",
    key: "temperature",
    min: 0.1,
    max: 2.0,
    step: 0.1,
    default: 0.7,
    description: "控制输出的随机性",
    impact: "越低越确定，越高越随机"
  },
  {
    name: "Top-K",
    key: "top_k",
    min: 0,
    max: 100,
    step: 5,
    default: 50,
    description: "只从概率最高的 K 个 token 中采样",
    impact: "限制候选集大小"
  },
  {
    name: "Top-P",
    key: "top_p",
    min: 0,
    max: 1,
    step: 0.05,
    default: 0.9,
    description: "核采样，累计概率达到 P 时停止",
    impact: "动态调整候选集"
  },
  {
    name: "Num Beams",
    key: "num_beams",
    min: 1,
    max: 10,
    step: 1,
    default: 1,
    description: "束搜索宽度",
    impact: "提高质量但变慢"
  }
];

export default function GenerationParametersExplorer() {
  const [values, setValues] = useState<Record<string, number>>(
    parameters.reduce((acc, param) => ({
      ...acc,
      [param.key]: param.default
    }), {})
  );

  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  const presets = {
    creative: {
      name: "创意写作",
      emoji: "✨",
      values: { temperature: 0.9, top_k: 0, top_p: 0.95, num_beams: 1 }
    },
    factual: {
      name: "事实性文本",
      emoji: "📚",
      values: { temperature: 0.3, top_k: 50, top_p: 1.0, num_beams: 1 }
    },
    code: {
      name: "代码生成",
      emoji: "💻",
      values: { temperature: 0.2, top_k: 0, top_p: 1.0, num_beams: 4 }
    },
    chat: {
      name: "聊天对话",
      emoji: "💬",
      values: { temperature: 0.7, top_k: 0, top_p: 0.9, num_beams: 1 }
    }
  };

  const applyPreset = (presetKey: string) => {
    const preset = presets[presetKey as keyof typeof presets];
    setValues(preset.values);
    setSelectedPreset(presetKey);
    setTimeout(() => setSelectedPreset(null), 1000);
  };

  const getTemperatureEffect = (temp: number) => {
    if (temp < 0.5) return { text: "非常确定", color: "text-blue-400" };
    if (temp < 0.8) return { text: "平衡", color: "text-green-400" };
    if (temp < 1.2) return { text: "创意", color: "text-yellow-400" };
    return { text: "极度随机", color: "text-red-400" };
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl border border-slate-700 shadow-2xl">
      <h3 className="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
        🎛️ 生成参数交互式调节器
      </h3>

      {/* Presets */}
      <div className="mb-8">
        <h4 className="text-sm font-semibold text-gray-400 mb-3">快速预设</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(presets).map(([key, preset]) => (
            <motion.button
              key={key}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => applyPreset(key)}
              className={`p-3 rounded-lg border transition-all ${
                selectedPreset === key
                  ? "bg-purple-600 border-purple-400 shadow-lg"
                  : "bg-slate-800 border-slate-600 hover:border-purple-500"
              }`}
            >
              <div className="text-2xl mb-1">{preset.emoji}</div>
              <div className="text-sm font-semibold text-white">{preset.name}</div>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Parameter Sliders */}
      <div className="space-y-6">
        {parameters.map((param) => (
          <div key={param.key} className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-white">
                {param.name}
              </label>
              <span className="px-3 py-1 bg-purple-600 text-white text-sm font-mono rounded">
                {values[param.key].toFixed(param.step < 1 ? 2 : 0)}
              </span>
            </div>

            <input
              type="range"
              min={param.min}
              max={param.max}
              step={param.step}
              value={values[param.key]}
              onChange={(e) => setValues({
                ...values,
                [param.key]: parseFloat(e.target.value)
              })}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, 
                  #9333ea 0%, 
                  #9333ea ${((values[param.key] - param.min) / (param.max - param.min)) * 100}%, 
                  #334155 ${((values[param.key] - param.min) / (param.max - param.min)) * 100}%, 
                  #334155 100%)`
              }}
            />

            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-400">{param.description}</span>
              {param.key === "temperature" && (
                <span className={`font-semibold ${getTemperatureEffect(values[param.key]).color}`}>
                  {getTemperatureEffect(values[param.key]).text}
                </span>
              )}
            </div>
            <p className="text-xs text-gray-500">{param.impact}</p>
          </div>
        ))}
      </div>

      {/* Generated Code */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-8 p-4 bg-slate-950 rounded-lg border border-slate-700"
      >
        <h4 className="text-sm font-semibold text-green-400 mb-2 flex items-center gap-2">
          <span>💻</span> 生成的代码
        </h4>
        <pre className="text-xs text-green-300 font-mono overflow-x-auto">
{`generator(
    prompt,
    temperature=${values.temperature.toFixed(1)},
    top_k=${values.top_k === 0 ? 'None' : Math.round(values.top_k)},
    top_p=${values.top_p.toFixed(2)},
    num_beams=${Math.round(values.num_beams)},
    do_sample=${values.temperature > 0 ? 'True' : 'False'}
)`}
        </pre>
      </motion.div>

      {/* Impact Visualization */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-blue-900/20 border border-blue-700/50 rounded-lg">
          <h5 className="text-sm font-semibold text-blue-400 mb-2">📊 输出特征</h5>
          <ul className="text-xs text-gray-300 space-y-1">
            <li>• 随机性: {values.temperature < 0.5 ? "低" : values.temperature < 1 ? "中" : "高"}</li>
            <li>• 候选集: {values.top_k > 0 ? `Top-${Math.round(values.top_k)}` : values.top_p < 1 ? `Top-P (${values.top_p})` : "全部"}</li>
            <li>• 搜索策略: {values.num_beams > 1 ? `束搜索 (${Math.round(values.num_beams)})` : "采样/贪婪"}</li>
          </ul>
        </div>

        <div className="p-4 bg-yellow-900/20 border border-yellow-700/50 rounded-lg">
          <h5 className="text-sm font-semibold text-yellow-400 mb-2">⚡ 性能影响</h5>
          <ul className="text-xs text-gray-300 space-y-1">
            <li>• 速度: {values.num_beams > 1 ? "慢 (束搜索)" : "快"}</li>
            <li>• 质量: {values.num_beams > 1 ? "高" : values.temperature < 0.5 ? "中" : "变化大"}</li>
            <li>• 多样性: {values.temperature > 1 ? "高" : "中"}</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
