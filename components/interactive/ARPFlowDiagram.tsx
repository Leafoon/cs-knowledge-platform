"use client";
import { useState } from "react";

const flowSteps = [
  { id: 1, label: "检查本地 ARP 缓存", desc: "主机首先检查是否已有目标 IP 的 MAC 映射", type: "check" },
  { id: 2, label: "构造 ARP 请求帧", desc: "目标 MAC 设为 FF:FF:FF:FF:FF:FF（广播），包含发送方 IP/MAC 和目标 IP", type: "process" },
  { id: 3, label: "广播 ARP 请求", desc: "ARP 请求通过广播发送到本地网络的所有主机", type: "send" },
  { id: 4, label: "接收方处理", desc: "所有主机收到请求，只有 IP 匹配的主机会响应", type: "process" },
  { id: 5, label: "构造 ARP 响应帧", desc: "目标主机将自己的 MAC 地址封装在 ARP 响应中，以单播方式发送", type: "process" },
  { id: 6, label: "更新 ARP 缓存", desc: "发送方收到响应后，将 IP-MAC 映射存入 ARP 缓存表", type: "done" },
];

const typeColors: Record<string, string> = {
  check: "bg-yellow-100 dark:bg-yellow-900/30 border-yellow-400",
  process: "bg-blue-100 dark:bg-blue-900/30 border-blue-400",
  send: "bg-green-100 dark:bg-green-900/30 border-green-400",
  done: "bg-purple-100 dark:bg-purple-900/30 border-purple-400",
};

export function ARPFlowDiagram() {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">ARP 工作流程图</h3>
      <div className="flex justify-between mb-4">
        <div className="text-center p-3 rounded bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700">
          <div className="text-xs text-text-secondary">发送主机</div>
          <div className="font-mono text-sm text-text-primary">192.168.1.100</div>
          <div className="font-mono text-xs text-text-secondary">AA:BB:CC:11:22:33</div>
        </div>
        <div className="flex items-center px-4">
          <div className="w-32 h-0.5 bg-gray-300 dark:bg-gray-600 relative">
            {activeStep >= 3 && <div className="absolute top-0 left-0 h-0.5 bg-blue-500 animate-pulse" style={{ width: `${Math.min(100, (activeStep - 2) * 50)}%` }} />}
            {activeStep >= 2 && activeStep < 5 && <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-xs text-blue-500">⇄</span>}
          </div>
        </div>
        <div className="text-center p-3 rounded bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700">
          <div className="text-xs text-text-secondary">目标主机</div>
          <div className="font-mono text-sm text-text-primary">192.168.1.1</div>
          <div className="font-mono text-xs text-text-secondary">AA:BB:CC:DD:EE:FF</div>
        </div>
      </div>
      <div className="space-y-2 mb-4">
        {flowSteps.map((s, i) => (
          <button key={s.id} onClick={() => setActiveStep(i)} className={`w-full text-left p-3 rounded-lg border transition-all ${activeStep === i ? `${typeColors[s.type]} border-l-4` : "border-border-subtle bg-gray-50 dark:bg-gray-800 opacity-60"}`}>
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-xs font-bold text-text-primary">{s.id}</span>
              <span className="text-sm font-medium text-text-primary">{s.label}</span>
              {activeStep === i && <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-blue-500 text-white">当前步骤</span>}
            </div>
            {activeStep === i && <p className="text-xs text-text-secondary mt-2 ml-8">{s.desc}</p>}
          </button>
        ))}
      </div>
      <div className="flex gap-2">
        <button onClick={() => setActiveStep(Math.max(0, activeStep - 1))} disabled={activeStep === 0} className="px-3 py-1 rounded text-sm bg-gray-200 dark:bg-gray-700 text-text-secondary disabled:opacity-50">上一步</button>
        <button onClick={() => setActiveStep(Math.min(flowSteps.length - 1, activeStep + 1))} disabled={activeStep === flowSteps.length - 1} className="px-3 py-1 rounded text-sm bg-blue-500 text-white disabled:opacity-50">下一步</button>
      </div>
    </div>
  );
}
export default ARPFlowDiagram;
