"use client";

import { useState } from "react";

const steps = [
  {
    id: 1,
    label: "本地发起",
    color: "from-violet-500 to-purple-600",
    icon: "💻",
    desc: "Python 客户端序列化函数和参数",
    code: `remote = rpc.connect("target_host", 9090)
func = remote.get_function("my_func")`,
  },
  {
    id: 2,
    label: "RPC 传输",
    color: "from-indigo-500 to-blue-600",
    icon: "🌐",
    desc: "通过 Socket 序列化/反序列化调用",
    code: `// RPC 协议
// 1. 发送: func_name + type_codes + values
// 2. 接收: result_type_code + result_value
// 使用 TVMValue 打包格式`,
  },
  {
    id: 3,
    label: "设备执行",
    color: "from-blue-500 to-cyan-600",
    icon: "⚡",
    desc: "远端设备上执行计算内核",
    code: `// 远端执行
void* handle = GetFunc(name);
TVMFuncCall(handle, args, ret);
// GPU: cuda kernel launch
// CPU: LLVM compiled function`,
  },
  {
    id: 4,
    label: "结果回传",
    color: "from-cyan-500 to-teal-600",
    icon: "📦",
    desc: "结果通过 RPC 回传到本地",
    code: `// 结果回传
// NDArray: 先 copy 到 CPU, 再传输
// 标量: 直接序列化
result = func(input)  # 本地拿到结果`,
  },
];

export function RemoteExecutionFlow() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        远程执行流程 (RPC)
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
        TVM RPC 允许在本地编译，远程设备上执行
      </p>

      <div className="flex items-center justify-between mb-6 relative">
        <div className="absolute top-6 left-8 right-8 h-0.5 bg-slate-200 dark:bg-slate-700" />
        <div
          className="absolute top-6 left-8 h-0.5 bg-indigo-500 transition-all duration-500"
          style={{ width: `${(current / (steps.length - 1)) * (100 - 10)}%` }}
        />
        {steps.map((s, i) => (
          <div key={s.id} className="relative flex flex-col items-center z-10">
            <button
              onClick={() => setCurrent(i)}
              className={`w-12 h-12 rounded-full flex items-center justify-center text-xl transition-all ${
                i <= current
                  ? `bg-gradient-to-r ${s.color} text-white shadow-lg`
                  : "bg-slate-200 dark:bg-slate-700 text-slate-400"
              } ${current === i ? "ring-4 ring-indigo-300 dark:ring-indigo-700 scale-110" : ""}`}
            >
              {s.icon}
            </button>
            <span className={`text-xs font-bold mt-2 ${
              current === i ? "text-indigo-600 dark:text-indigo-400" : "text-slate-400"
            }`}>
              {s.label}
            </span>
          </div>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-5 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">{steps[current].icon}</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100">
            步骤 {current + 1}: {steps[current].label}
          </h4>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
          {steps[current].desc}
        </p>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm font-mono">
          {steps[current].code}
        </pre>
      </div>

      <div className="flex gap-2 mt-4 justify-center">
        <button
          onClick={() => setCurrent(Math.max(0, current - 1))}
          disabled={current === 0}
          className="px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg text-sm font-bold disabled:opacity-30 text-slate-700 dark:text-slate-300"
        >
          ← 上一步
        </button>
        <button
          onClick={() => setCurrent(Math.min(steps.length - 1, current + 1))}
          disabled={current === steps.length - 1}
          className="px-4 py-2 bg-indigo-500 text-white rounded-lg text-sm font-bold disabled:opacity-30"
        >
          下一步 →
        </button>
      </div>
    </div>
  );
}
