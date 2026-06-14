"use client";
import { useState } from "react";

export function TLSKDFpipeline() {
  const [version, setVersion] = useState<"1.2" | "1.3">("1.3");
  const [step, setStep] = useState(0);

  const tls12Stages = [
    { label: "Pre-Master Secret", desc: "客户端生成 48 字节随机密钥", size: "48 bytes" },
    { label: "Master Secret", desc: "PRF(Pre-Master, \"master secret\", Client.random + Server.random)", size: "48 bytes" },
    { label: "Key Block", desc: "PRF(Master, \"key expansion\", Server.random + Client.random)", size: "N bytes" },
    { label: "Client Write MAC Key", desc: "用于客户端消息的 HMAC 密钥", size: "20 bytes" },
    { label: "Server Write MAC Key", desc: "用于服务器消息的 HMAC 密钥", size: "20 bytes" },
    { label: "Client Write Key", desc: "客户端加密密钥", size: "16 bytes" },
    { label: "Server Write Key", desc: "服务器加密密钥", size: "16 bytes" },
    { label: "Client Write IV", desc: "客户端初始化向量", size: "4 bytes" },
    { label: "Server Write IV", desc: "服务器初始化向量", size: "4 bytes" },
  ];

  const tls13Stages = [
    { label: "Early Secret", desc: "HKDF-Extract(0, PSK)", size: "32 bytes" },
    { label: "Client Early Traffic Secret", desc: "HKDF-Expand(Early Secret, \"c e traffic\", Hello)", size: "32 bytes" },
    { label: "Handshake Secret", desc: "HKDF-Extract(Early Secret, DH Shared Secret)", size: "32 bytes" },
    { label: "Client Handshake Traffic Secret", desc: "HKDF-Expand(Handshake Secret, \"c hs traffic\")", size: "32 bytes" },
    { label: "Server Handshake Traffic Secret", desc: "HKDF-Expand(Handshake Secret, \"s hs traffic\")", size: "32 bytes" },
    { label: "Master Secret", desc: "HKDF-Extract(Handshake, 0)", size: "32 bytes" },
    { label: "Client Application Traffic Secret", desc: "HKDF-Expand(Master Secret, \"c ap traffic\")", size: "32 bytes" },
    { label: "Server Application Traffic Secret", desc: "HKDF-Expand(Master Secret, \"s ap traffic\")", size: "32 bytes" },
  ];

  const stages = version === "1.2" ? tls12Stages : tls13Stages;
  const maxStep = stages.length - 1;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TLS 密钥派生管线 (KDF Pipeline)</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => { setVersion("1.2"); setStep(0); }}
          className={`px-4 py-2 rounded text-sm font-medium ${version === "1.2" ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          TLS 1.2 (PRF)
        </button>
        <button onClick={() => { setVersion("1.3"); setStep(0); }}
          className={`px-4 py-2 rounded text-sm font-medium ${version === "1.3" ? "bg-green-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          TLS 1.3 (HKDF)
        </button>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep((s) => Math.min(s + 1, maxStep))} disabled={step >= maxStep}
          className="px-3 py-1 rounded bg-blue-500 text-white text-sm disabled:opacity-50">下一步</button>
        <button onClick={() => setStep((s) => Math.max(s - 1, 0))} disabled={step === 0}
          className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">上一步</button>
        <button onClick={() => setStep(0)} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm">重置</button>
      </div>
      <div className="space-y-1">
        {stages.map((s, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 transition-all ${
              i <= step ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-muted"
            }`}>{i + 1}</div>
            <div className={`flex-1 p-3 rounded-lg border transition-all ${
              i === step ? "border-blue-400 bg-blue-500/10" : i < step ? "border-border-subtle bg-bg-primary" : "border-border-subtle opacity-40"
            }`}>
              <div className="flex items-center justify-between">
                <span className={`text-sm font-mono ${i === step ? "text-blue-400" : "text-text-primary"}`}>{s.label}</span>
                <span className="text-text-muted text-xs">{s.size}</span>
              </div>
              {i === step && <p className="text-text-secondary text-xs mt-1">{s.desc}</p>}
            </div>
            {i < stages.length - 1 && (
              <svg width="20" height="16" viewBox="0 0 20 16" className="flex-shrink-0">
                <path d="M10 0 L10 10 M5 7 L10 13 L15 7" stroke={i < step ? "#60a5fa" : "#9ca3af"} strokeWidth="2" fill="none" />
              </svg>
            )}
          </div>
        ))}
      </div>
      <p className="text-text-muted text-xs mt-4">
        {version === "1.2" ? "TLS 1.2 使用 PRF (Pseudo-Random Function) 基于 HMAC-SHA256 派生密钥" : "TLS 1.3 使用 HKDF (HMAC-based Key Derivation Function) RFC 5869，分为 Extract 和 Expand 两步"}
      </p>
    </div>
  );
}
export default TLSKDFpipeline;
