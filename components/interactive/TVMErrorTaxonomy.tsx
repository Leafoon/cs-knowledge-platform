"use client";

import { useState } from "react";

const categories = [
  {
    name: "编译期错误",
    icon: "⚙️",
    color: "from-blue-500 to-indigo-500",
    children: [
      { name: "IR 语法错误", desc: "TVM script 语法不正确", severity: "高" },
      { name: "类型不匹配", desc: "dtype 或 shape 不兼容", severity: "高" },
      { name: "Pass 失败", desc: "优化 Pass 无法应用", severity: "中" },
      { name: "Target 不支持", desc: "目标后端不支持某算子", severity: "中" },
    ],
  },
  {
    name: "运行时错误",
    icon: "💥",
    color: "from-purple-500 to-pink-500",
    children: [
      { name: "内存不足 (OOM)", desc: "分配超出可用内存", severity: "高" },
      { name: "设备错误", desc: "GPU kernel launch 失败", severity: "高" },
      { name: "数值异常", desc: "NaN/Inf/Overflow", severity: "中" },
      { name: "Shape 错误", desc: "运行时 shape 不匹配", severity: "高" },
    ],
  },
  {
    name: "逻辑错误",
    icon: "🧠",
    color: "from-indigo-500 to-purple-500",
    children: [
      { name: "调度错误", desc: "Schedule 产生无效 TIR", severity: "中" },
      { name: "精度损失", desc: "数值精度不满足要求", severity: "低" },
      { name: "性能退化", desc: "优化后反而变慢", severity: "低" },
      { name: "内存泄漏", desc: "Tensor 未正确释放", severity: "中" },
    ],
  },
];

const severityColors: Record<string, string> = {
  "高": "bg-red-500",
  "中": "bg-amber-500",
  "低": "bg-emerald-500",
};

export function TVMErrorTaxonomy() {
  const [expanded, setExpanded] = useState<string | null>("编译期错误");
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">TVM 错误分类</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">按类型和严重程度的错误分类树</p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
        {categories.map((cat) => (
          <div
            key={cat.name}
            className="bg-white dark:bg-slate-800/80 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden"
          >
            <button
              onClick={() => setExpanded(expanded === cat.name ? null : cat.name)}
              className="w-full flex items-center gap-3 p-4 transition-all hover:bg-slate-50 dark:hover:bg-slate-700/50"
            >
              <span className="text-xl">{cat.icon}</span>
              <div className="flex-1 text-left">
                <div className="text-sm font-bold text-slate-700 dark:text-slate-200">{cat.name}</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">{cat.children.length} 种错误</div>
              </div>
              <span className={`text-slate-400 transition-transform ${expanded === cat.name ? "rotate-180" : ""}`}>
                ▼
              </span>
            </button>

            {expanded === cat.name && (
              <div className="border-t border-slate-200 dark:border-slate-700 p-3 space-y-1.5">
                {cat.children.map((child) => (
                  <button
                    key={child.name}
                    onClick={() => setSelected(child.name)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all ${
                      selected === child.name
                        ? "bg-indigo-100 dark:bg-indigo-900/40"
                        : "hover:bg-slate-50 dark:hover:bg-slate-700/50"
                    }`}
                  >
                    <span className={`w-2 h-2 rounded-full ${severityColors[child.severity]}`} />
                    <div className="flex-1">
                      <div className="text-xs font-bold text-slate-700 dark:text-slate-200">{child.name}</div>
                      <div className="text-[10px] text-slate-500 dark:text-slate-400">{child.desc}</div>
                    </div>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded text-white ${severityColors[child.severity]}`}>
                      {child.severity}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="bg-white/60 dark:bg-slate-800/60 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">严重程度说明</h4>
        <div className="flex gap-4">
          {[
            { level: "高", desc: "编译/运行失败，必须修复", color: "bg-red-500" },
            { level: "中", desc: "功能异常，需要调查", color: "bg-amber-500" },
            { level: "低", desc: "性能/精度问题，建议优化", color: "bg-emerald-500" },
          ].map((s) => (
            <div key={s.level} className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${s.color}`} />
              <span className="text-xs text-slate-600 dark:text-slate-300">
                <strong>{s.level}</strong>: {s.desc}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
