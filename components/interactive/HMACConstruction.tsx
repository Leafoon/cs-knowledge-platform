"use client";
import { useState } from "react";

export function HMACConstruction() {
  const [step, setStep] = useState(0);
  const totalSteps = 5;

  const steps = [
    { title: "原始密钥 K", desc: "输入密钥（若长度>块大小则先哈希）", detail: "K = 0x4b6579...", visual: "密钥 → [64字节]" },
    { title: "K ⊕ ipad", desc: "密钥与内部填充 0x36 异或", detail: "K ⊕ 0x363636...36", visual: "K ⊕ ipad → Si" },
    { title: "Si || M", desc: "将异或结果与消息拼接", detail: "(K⊕ipad) || Message", visual: "Si + Message → 拼接块" },
    { title: "H(Si||M)", desc: "对拼接结果计算哈希值", detail: "Hash = SHA-256(Si||M)", visual: "拼接块 → H_inner" },
    { title: "K ⊕ opad → H", desc: "密钥与外部填充0x5C异或后，再哈希", detail: "HMAC = H((K⊕opad) || H_inner)", visual: "(K⊕opad) + H_inner → HMAC" },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">HMAC 构造演示</h3>
      <p className="text-sm text-text-secondary mb-4">HMAC(K,M) = H((K⊕opad) || H((K⊕ipad) || M))</p>

      <div className="flex gap-1 mb-6">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`flex-1 h-2 rounded-full transition-all ${
              i <= step ? "bg-blue-500" : "bg-bg-muted"
            }`}
          />
        ))}
      </div>

      <div className="p-4 rounded-lg bg-bg-muted border border-border-subtle mb-4">
        <p className="text-sm font-medium text-blue-400 mb-1">步骤 {step + 1}: {steps[step].title}</p>
        <p className="text-sm text-text-secondary mb-2">{steps[step].desc}</p>
        <div className="font-mono text-xs bg-bg-subtle p-2 rounded text-text-muted">{steps[step].detail}</div>
        <div className="mt-2 text-sm text-text-primary font-mono text-center p-2 bg-bg-subtle rounded">
          {steps[step].visual}
        </div>
      </div>

      <div className="flex gap-2">
        <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}
          className="px-3 py-1.5 rounded bg-gray-500 text-white text-sm disabled:opacity-40">上一步</button>
        <button onClick={() => setStep(Math.min(totalSteps - 1, step + 1))} disabled={step === totalSteps - 1}
          className="px-3 py-1.5 rounded bg-blue-500 text-white text-sm disabled:opacity-40">下一步</button>
      </div>
    </div>
  );
}
export default HMACConstruction;
