"use client";
import { useState } from "react";

interface Step {
  label: string;
  en: string;
  detail: string;
  direction?: "→" | "←" | "↔";
}

const steps: Step[] = [
  { label: "客户端Hello", en: "Client Hello", detail: "客户端发送支持的TLS版本、密码套件列表、随机数、QUIC传输参数", direction: "→" },
  { label: "服务端Hello", en: "Server Hello", detail: "服务端选择密码套件，发送随机数和密钥共享（Key Share）", direction: "←" },
  { label: "Encrypted Extensions", en: "Encrypted Extensions", detail: "已加密传输的扩展信息（从此步起所有数据已加密）", direction: "←" },
  { label: "证书", en: "Certificate", detail: "服务端发送X.509证书链", direction: "←" },
  { label: "证书验证", en: "Certificate Verify", detail: "服务端用私钥签名握手消息，客户端验证签名", direction: "←" },
  { label: "完成", en: "Finished", detail: "双方发送Finished消息，握手完成，1-RTT即可传输数据", direction: "↔" },
];

export function QUICEncryptionDemo() {
  const [activeStep, setActiveStep] = useState(0);
  const [showComparison, setShowComparison] = useState(false);
  const step = steps[activeStep];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">QUIC加密演示 (TLS 1.3集成)</h3>
      <div className="flex items-center gap-1 mb-4">
        {steps.map((s, i) => (
          <button key={i} onClick={() => setActiveStep(i)}
            className={`flex-1 h-2 rounded-full transition-all ${i <= activeStep ? "bg-sky-500" : "bg-bg-tertiary"}`} />
        ))}
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setActiveStep(Math.max(0, activeStep - 1))} disabled={activeStep === 0}
          className="px-3 py-1 rounded-lg bg-bg-tertiary border border-border-subtle text-xs disabled:opacity-30">←</button>
        <button onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))} disabled={activeStep === steps.length - 1}
          className="px-3 py-1 rounded-lg bg-bg-tertiary border border-border-subtle text-xs disabled:opacity-30">→</button>
        <span className="text-xs text-text-tertiary ml-auto">步骤 {activeStep + 1}/{steps.length}</span>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="flex items-center gap-3 mb-2">
          {step.direction && <span className="text-lg text-sky-500 font-mono">{step.direction}</span>}
          <span className="text-sm font-semibold text-text-primary">{step.label}</span>
          <span className="text-xs text-text-tertiary">{step.en}</span>
          {activeStep >= 2 && <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 text-[10px]">已加密</span>}
        </div>
        <div className="text-xs text-text-secondary">{step.detail}</div>
      </div>
      <div className="flex items-center gap-3 mb-2">
        <div className="flex-1 h-px bg-border-subtle" />
        <button onClick={() => setShowComparison(!showComparison)} className="text-xs text-sky-600 dark:text-sky-400 hover:underline">
          {showComparison ? "隐藏" : "对比"} TLS 1.2 vs TLS 1.3
        </button>
        <div className="flex-1 h-px bg-border-subtle" />
      </div>
      {showComparison && (
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs">
            <div className="font-medium text-amber-500 mb-1">TLS 1.2 (TCP+TLS)</div>
            <div className="text-text-secondary">TCP握手: 1 RTT</div>
            <div className="text-text-secondary">TLS握手: 2 RTT</div>
            <div className="text-text-secondary">总计: 3 RTT才能发送数据</div>
          </div>
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs">
            <div className="font-medium text-emerald-500 mb-1">QUIC (TLS 1.3内置)</div>
            <div className="text-text-secondary">首次连接: 1 RTT</div>
            <div className="text-text-secondary">0-RTT恢复: 0 RTT</div>
            <div className="text-text-secondary">加密从第1个包就开始</div>
          </div>
        </div>
      )}
    </div>
  );
}
export default QUICEncryptionDemo;
