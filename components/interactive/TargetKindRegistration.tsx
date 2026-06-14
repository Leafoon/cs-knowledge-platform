'use client';

import { useState } from "react";

const steps = [
  {
    label: "定义",
    icon: "📝",
    color: "from-violet-500 to-purple-600",
    code: `@register_target_kind("cuda")\nclass CUDATargetKind(TargetKind):\n    def __init__(self):\n        super().__init__("cuda")`,
    desc: "创建 TargetKind 子类，定义目标平台的基本属性",
  },
  {
    label: "注册",
    icon: "📦",
    color: "from-indigo-500 to-blue-600",
    code: `REGISTER_TARGET_KIND("cuda")\n    .set_attr<TTargetAttrFn>(...)`,
    desc: "通过装饰器或宏将 Kind 注册到全局注册表",
  },
  {
    label: "匹配",
    icon: "🔍",
    code: `if target.kind.name == "cuda":\n    # 使用 CUDA 特定优化`,
    color: "from-blue-500 to-cyan-600",
    desc: "Pass 中通过 target.kind 进行条件匹配",
  },
  {
    label: "使用",
    icon: "🚀",
    code: `target = tvm.target.cuda()\nmod = relay.build(func, target)`,
    color: "from-cyan-500 to-teal-600",
    desc: "用户通过工厂函数创建 Target 并编译",
  },
];

export default function TargetKindRegistration() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        TargetKind 注册流程
      </h3>

      <div className="flex items-center justify-between mb-8">
        {steps.map((step, i) => (
          <div key={step.label} className="flex items-center">
            <button
              onClick={() => setCurrent(i)}
              className={`flex flex-col items-center gap-1 transition-all ${
                current === i ? "scale-110" : "opacity-60 hover:opacity-80"
              }`}
            >
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center text-xl bg-gradient-to-br ${step.color} text-white shadow-lg ${
                  current === i ? "ring-4 ring-indigo-300 dark ring-offset-2" : ""
                }`}
              >
                {step.icon}
              </div>
              <span className="text-xs font-bold text-slate-700 dark:text-slate-300">
                {step.label}
              </span>
            </button>
            {i < steps.length - 1 && (
              <div
                className={`h-0.5 w-16 mx-2 transition-colors ${
                  i < current
                    ? "bg-indigo-400"
                    : "bg-slate-300 dark:bg-slate-600"
                }`}
              />
            )}
          </div>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-5 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-2xl">{steps[current].icon}</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100">
            步骤 {current + 1}: {steps[current].label}
          </h4>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
          {steps[current].desc}
        </p>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm font-mono overflow-x-auto">
          {steps[current].code}
        </pre>
      </div>

      <div className="flex gap-2 mt-4 justify-center">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrent(i)}
            className={`w-3 h-3 rounded-full transition-all ${
              current === i
                ? "bg-indigo-500 scale-125"
                : "bg-slate-300 dark:bg-slate-600"
            }`}
          />
        ))}
      </div>
    </div>
  );
}
