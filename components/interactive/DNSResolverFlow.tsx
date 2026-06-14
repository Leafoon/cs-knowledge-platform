"use client";
import { useState } from "react";

interface QueryStep {
  server: string;
  action: string;
  response: string;
  type: "recursive" | "iterative";
}

export function DNSResolverFlow() {
  const [mode, setMode] = useState<"recursive" | "iterative">("recursive");
  const [step, setStep] = useState(-1);

  const recursiveSteps: QueryStep[] = [
    { server: "本地DNS", action: "客户端发送查询: www.example.com", response: "交给递归解析器处理", type: "recursive" },
    { server: "根DNS", action: "递归解析器 → 根服务器", response: "返回 .com NS记录", type: "recursive" },
    { server: ".com TLD", action: "递归解析器 → .com TLD服务器", response: "返回 example.com NS记录", type: "recursive" },
    { server: "权威DNS", action: "递归解析器 → example.com权威服务器", response: "返回 www.example.com A记录: 93.184.216.34", type: "recursive" },
    { server: "客户端", action: "递归解析器返回最终结果", response: "www.example.com = 93.184.216.34", type: "recursive" },
  ];

  const iterativeSteps: QueryStep[] = [
    { server: "本地DNS", action: "客户端发送查询: www.example.com", response: "返回根服务器地址", type: "iterative" },
    { server: "客户端→根", action: "客户端直接查询根服务器", response: "返回 .com TLD地址", type: "iterative" },
    { server: "客户端→TLD", action: "客户端查询 .com TLD服务器", response: "返回 example.com 权威地址", type: "iterative" },
    { server: "客户端→权威", action: "客户端查询权威服务器", response: "返回 A记录: 93.184.216.34", type: "iterative" },
  ];

  const steps = mode === "recursive" ? recursiveSteps : iterativeSteps;

  const getTotalLatency = () => {
    if (mode === "recursive") return step >= 0 ? `${(step + 1) * 20}ms (总)` : "—";
    return step >= 0 ? `${(step + 1) * 25}ms (总)` : "—";
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNS 解析流程</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => { setMode("recursive"); setStep(-1); }}
          className={`flex-1 py-2 rounded text-sm ${mode === "recursive" ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
          递归查询
        </button>
        <button onClick={() => { setMode("iterative"); setStep(-1); }}
          className={`flex-1 py-2 rounded text-sm ${mode === "iterative" ? "bg-purple-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
          迭代查询
        </button>
      </div>
      <p className="text-xs text-text-secondary mb-4">
        {mode === "recursive" ? "递归模式：本地DNS服务器代替客户端完成全部查询" : "迭代模式：客户端自己逐步查询各级DNS服务器"}
      </p>
      <div className="space-y-2 mb-4">
        {steps.map((s, i) => (
          <div key={i} className={`p-3 rounded border transition-all duration-300 ${i === step ? "border-blue-400 bg-blue-50 dark:bg-blue-900/20" : i < step ? "border-green-300 bg-green-50 dark:bg-green-900/10" : "border-border-subtle opacity-50"}`}>
            <div className="flex items-center gap-2 mb-1">
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${i <= step ? "bg-blue-600 text-white" : "bg-gray-300 dark:bg-gray-700 text-text-secondary"}`}>
                {i < step ? "✓" : i + 1}
              </span>
              <span className="font-mono text-sm text-text-primary">{s.server}</span>
            </div>
            {i <= step && (
              <>
                <div className="text-xs text-text-secondary ml-8">{s.action}</div>
                {i < step && <div className="text-xs text-green-600 ml-8 mt-0.5">→ {s.response}</div>}
              </>
            )}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">当前步骤</div>
          <div className="font-bold text-text-primary">{step + 1}/{steps.length}</div>
        </div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">累计延迟</div>
          <div className="font-bold text-text-primary">{getTotalLatency()}</div>
        </div>
      </div>
      <button onClick={() => setStep(step < steps.length - 1 ? step + 1 : -1)}
        className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">
        {step < 0 ? "开始解析" : step < steps.length - 1 ? "下一步" : "重置"}
      </button>
    </div>
  );
}
export default DNSResolverFlow;
