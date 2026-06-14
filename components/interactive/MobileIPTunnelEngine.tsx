"use client";
import { useState } from "react";

export function MobileIPTunnelEngine() {
  const [mode, setMode] = useState<"home" | "foreign">("foreign");
  const [step, setStep] = useState(0);

  const tunnelSteps = [
    {
      label: "原始数据包",
      detail: "CN发送给MN的包: src=CN, dst=MN-HA(家乡地址)",
      color: "bg-blue-100 dark:bg-blue-900/40",
    },
    {
      label: "家乡代理截获",
      detail: "HA收到发往MN家乡地址的包，查找绑定缓存",
      color: "bg-yellow-100 dark:bg-yellow-900/40",
    },
    {
      label: "隧道封装",
      detail: "HA添加外部IP头: src=HA, dst=CoA(转交地址)",
      color: "bg-orange-100 dark:bg-orange-900/40",
    },
    {
      label: "隧道传输",
      detail: "封装包通过互联网发送到外地网络的FA",
      color: "bg-purple-100 dark:bg-purple-900/40",
    },
    {
      label: "解封装",
      detail: "FA/MN剥去外部IP头，恢复原始包",
      color: "bg-green-100 dark:bg-green-900/40",
    },
  ];

  const homeSteps = [
    { label: "MN发送响应", detail: "src=MN-HA, dst=CN" },
    { label: "反向隧道", detail: "MN→FA→HA: 封装src=CoA, dst=HA" },
    { label: "HA解封装", detail: "恢复原始包，转发给CN" },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Mobile IP Tunnel <span className="text-text-secondary text-sm">— 家乡代理隧道封装</span>
      </h3>
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => { setMode("foreign"); setStep(0); }}
          className={`px-3 py-1 rounded text-sm ${mode === "foreign" ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
        >
          通信者→移动节点
        </button>
        <button
          onClick={() => { setMode("home"); setStep(0); }}
          className={`px-3 py-1 rounded text-sm ${mode === "home" ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
        >
          移动节点→通信者
        </button>
      </div>
      {mode === "foreign" ? (
        <>
          <div className="flex gap-1 mb-3 flex-wrap">
            {tunnelSteps.map((_, i) => (
              <button
                key={i}
                onClick={() => setStep(i)}
                className={`px-2 py-1 rounded text-xs ${step === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <div className={`${tunnelSteps[step].color} p-4 rounded`}>
            <div className="font-semibold text-text-primary mb-1">步骤 {step + 1}: {tunnelSteps[step].label}</div>
            <div className="text-sm text-text-secondary">{tunnelSteps[step].detail}</div>
          </div>
        </>
      ) : (
        <>
          <div className="flex gap-1 mb-3">
            {homeSteps.map((_, i) => (
              <button
                key={i}
                onClick={() => setStep(i)}
                className={`px-2 py-1 rounded text-xs ${step === i ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded">
            <div className="font-semibold text-text-primary mb-1">步骤 {step + 1}: {homeSteps[step].label}</div>
            <div className="text-sm text-text-secondary">{homeSteps[step].detail}</div>
          </div>
        </>
      )}
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-center">
          <div className="text-text-secondary">家乡地址 (HoA)</div>
          <div className="font-mono text-text-primary">10.0.0.100</div>
        </div>
        <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-center">
          <div className="text-text-secondary">转交地址 (CoA)</div>
          <div className="font-mono text-text-primary">192.168.5.200</div>
        </div>
        <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-center">
          <div className="text-text-secondary">家乡代理 (HA)</div>
          <div className="font-mono text-text-primary">10.0.0.1</div>
        </div>
      </div>
    </div>
  );
}

export default MobileIPTunnelEngine;
