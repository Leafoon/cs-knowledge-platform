"use client";
import { useState } from "react";

const tls12Steps = [
  { from: "client", label: "ClientHello", desc: "支持的密码套件、随机数、协议版本" },
  { from: "server", label: "ServerHello", desc: "选定密码套件、随机数、证书" },
  { from: "server", label: "Certificate", desc: "服务器 X.509 证书链" },
  { from: "server", label: "ServerHelloDone", desc: "服务器握手消息发送完毕" },
  { from: "client", label: "ClientKeyExchange", desc: "预主密钥 (Pre-Master Secret)" },
  { from: "client", label: "ChangeCipherSpec", desc: "通知切换到加密通信" },
  { from: "client", label: "Finished", desc: "验证握手消息完整性" },
  { from: "server", label: "ChangeCipherSpec", desc: "通知切换到加密通信" },
  { from: "server", label: "Finished", desc: "验证握手消息完整性" },
];

const tls13Steps = [
  { from: "client", label: "ClientHello", desc: "支持的密码套件、KeyShare、PSK" },
  { from: "server", label: "ServerHello", desc: "选定密码套件、KeyShare" },
  { from: "server", label: "EncryptedExtensions", desc: "加密扩展信息" },
  { from: "server", label: "Certificate", desc: "加密的服务器证书" },
  { from: "server", label: "CertificateVerify", desc: "证书签名验证" },
  { from: "server", label: "Finished", desc: "服务器握手完成" },
  { from: "client", label: "Finished", desc: "客户端握手完成" },
];

export function TLSHandshakeDemo() {
  const [version, setVersion] = useState<"1.2" | "1.3">("1.3");
  const [step, setStep] = useState(-1);
  const steps = version === "1.2" ? tls12Steps : tls13Steps;

  const next = () => setStep((s) => Math.min(s + 1, steps.length - 1));
  const prev = () => setStep((s) => Math.max(s - 1, -1));
  const reset = () => setStep(-1);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TLS 握手演示</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => { setVersion("1.2"); reset(); }}
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${version === "1.2" ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          TLS 1.2
        </button>
        <button onClick={() => { setVersion("1.3"); reset(); }}
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${version === "1.3" ? "bg-green-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          TLS 1.3
        </button>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={prev} disabled={step < 0} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">← 上一步</button>
        <span className="text-text-secondary text-sm">步骤 {step + 1}/{steps.length}</span>
        <button onClick={next} disabled={step >= steps.length - 1} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">下一步 →</button>
        <button onClick={reset} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm">重置</button>
      </div>
      <div className="relative min-h-[250px]">
        <div className="absolute left-[20%] top-0 bottom-0 w-px bg-blue-400/30" />
        <div className="absolute left-[80%] top-0 bottom-0 w-px bg-green-400/30" />
        <div className="absolute top-0"><span className="text-blue-400 text-xs font-medium" style={{ left: "15%", position: "absolute" }}>客户端</span></div>
        <div className="absolute top-0"><span className="text-green-400 text-xs font-medium" style={{ left: "78%", position: "absolute" }}>服务器</span></div>
        <div className="pt-6 space-y-1">
          {steps.slice(0, step + 1).map((s, i) => {
            const isClient = s.from === "client";
            return (
              <div key={i} className={`relative h-10 flex items-center ${i === step ? "animate-pulse" : ""}`}>
                <span className="absolute text-text-muted text-[10px]" style={{ left: "2%", width: "12%" }}>#{i + 1}</span>
                <svg className="absolute" style={{ left: "20%", width: "60%", height: "100%" }}>
                  <line x1={isClient ? "0%" : "100%"} y1="50%" x2={isClient ? "100%" : "0%"} y2="50%"
                    stroke={isClient ? "#60a5fa" : "#4ade80"} strokeWidth="2" />
                </svg>
                <span className={`absolute text-xs font-mono ${isClient ? "text-blue-400" : "text-green-400"}`}
                  style={{ left: isClient ? "28%" : "52%", maxWidth: "20%", fontSize: "10px" }}>
                  {s.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      {step >= 0 && (
        <div className="p-3 rounded-lg bg-bg-primary border border-border-subtle mt-3">
          <span className="text-text-primary text-sm font-medium">{steps[step].label}</span>
          <p className="text-text-secondary text-xs mt-1">{steps[step].desc}</p>
        </div>
      )}
      <div className="p-3 rounded bg-bg-primary border border-border-subtle mt-3">
        <p className="text-text-muted text-xs">
          TLS 1.2: 2-RTT 完成握手 | TLS 1.3: 1-RTT 完成握手（支持 0-RTT 恢复），移除了不安全的密码套件
        </p>
      </div>
    </div>
  );
}
export default TLSHandshakeDemo;
