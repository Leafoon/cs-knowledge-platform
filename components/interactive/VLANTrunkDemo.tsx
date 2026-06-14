"use client";
import { useState } from "react";

export function VLANTrunkDemo() {
  const [vlan, setVlan] = useState(10);
  const [nativeVlan, setNativeVlan] = useState(1);
  const [step, setStep] = useState(0);
  const maxSteps = 5;

  const isNative = vlan === nativeVlan;

  const steps = [
    { label: "原始帧进入 Access 端口", desc: `主机发送标准以太帧（无 VLAN 标签），接入端口属于 VLAN ${vlan}` },
    { label: "交换机标记 VLAN", desc: `交换机在帧中插入 802.1Q 标签：VLAN ID = ${vlan}` },
    { label: "Trunk 端口发送", desc: isNative ? `VLAN ${vlan} 是 Native VLAN，帧不带标签通过 Trunk 发送` : `带标签帧通过 Trunk 链路发送到对端交换机` },
    { label: "对端交换机接收", desc: isNative ? `对端收到无标签帧，将其归入 Native VLAN (${nativeVlan})` : `对端读取 802.1Q 标签，识别 VLAN ID = ${vlan}` },
    { label: "Access 端口转发", desc: `交换机剥离标签，将帧从 VLAN ${vlan} 的 Access 端口转发给目标主机` },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">VLAN Trunk 演示</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-text-muted text-xs block mb-1">VLAN ID</label>
          <select value={vlan} onChange={(e) => setVlan(Number(e.target.value))}
            className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm">
            <option value={10}>VLAN 10 (销售部)</option>
            <option value={20}>VLAN 20 (技术部)</option>
            <option value={30}>VLAN 30 (财务部)</option>
            <option value={1}>VLAN 1 (Native)</option>
          </select>
        </div>
        <div>
          <label className="text-text-muted text-xs block mb-1">Native VLAN</label>
          <select value={nativeVlan} onChange={(e) => setNativeVlan(Number(e.target.value))}
            className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm">
            <option value={1}>VLAN 1</option>
            <option value={99}>VLAN 99</option>
          </select>
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep((s) => Math.min(s + 1, maxSteps))} disabled={step >= maxSteps}
          className="px-3 py-1 rounded bg-blue-500 text-white text-sm disabled:opacity-50">下一步</button>
        <button onClick={() => setStep((s) => Math.max(s - 1, 0))} disabled={step === 0}
          className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">上一步</button>
        <button onClick={() => setStep(0)} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm">重置</button>
        <span className="text-text-secondary text-sm self-center">步骤 {step}/{maxSteps}</span>
      </div>
      <div className="flex items-center gap-3 mb-4">
        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-400/30 text-center flex-shrink-0">
          <span className="text-blue-400 text-xs block">Switch A</span>
          <span className="text-text-muted text-[10px]">Access: VLAN {vlan}</span>
        </div>
        <div className="flex-1 relative h-12">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full h-1 bg-purple-400/30 rounded" />
          </div>
          {step >= 2 && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className={`px-3 py-1 rounded text-xs font-mono animate-pulse ${
                isNative ? "bg-yellow-500/10 text-yellow-400 border border-yellow-400/30" : "bg-purple-500/10 text-purple-400 border border-purple-400/30"
              }`}>
                {isNative ? "无标签帧 (Native)" : `802.1Q VID=${vlan}`}
              </div>
            </div>
          )}
          <span className="absolute -top-1 left-1/2 -translate-x-1/2 text-purple-400 text-[10px]">Trunk</span>
        </div>
        <div className="p-3 rounded-lg bg-green-500/10 border border-green-400/30 text-center flex-shrink-0">
          <span className="text-green-400 text-xs block">Switch B</span>
          <span className="text-text-muted text-[10px]">Access: VLAN {vlan}</span>
        </div>
      </div>
      {step > 0 && (
        <div className={`p-4 rounded-lg border ${step >= 3 ? "bg-green-500/10 border-green-400/30" : "bg-bg-primary border-border-subtle"}`}>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-5 h-5 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs">{step}</span>
            <span className="text-text-primary text-sm font-medium">{steps[step - 1].label}</span>
          </div>
          <p className="text-text-secondary text-xs ml-7">{steps[step - 1].desc}</p>
        </div>
      )}
      <div className="p-3 rounded bg-bg-primary border border-border-subtle mt-4">
        <h4 className="text-text-secondary text-xs font-medium mb-1">Trunk 端口规则</h4>
        <ul className="text-text-muted text-xs space-y-0.5">
          <li>• Trunk 端口可同时传输多个 VLAN 的帧</li>
          <li>• Native VLAN 的帧不带标签传输（默认 VLAN 1）</li>
          <li>• 非 Native VLAN 的帧带 802.1Q 标签</li>
          <li>• 两端交换机的 Native VLAN 配置必须一致</li>
        </ul>
      </div>
    </div>
  );
}
export default VLANTrunkDemo;
