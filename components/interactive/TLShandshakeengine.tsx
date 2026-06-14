"use client";
import { useState } from "react";

const cipherSuites = [
  { name: "TLS_AES_256_GCM_SHA384", ke: "ECDHE", auth: "ECDSA", enc: "AES-256-GCM", mac: "SHA384", strength: "强" },
  { name: "TLS_CHACHA20_POLY1305_SHA256", ke: "ECDHE", auth: "RSA", enc: "ChaCha20", mac: "Poly1305", strength: "强" },
  { name: "TLS_AES_128_GCM_SHA256", ke: "ECDHE", auth: "ECDSA", enc: "AES-128-GCM", mac: "SHA256", strength: "强" },
  { name: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA", ke: "ECDHE", auth: "RSA", enc: "AES-256-CBC", mac: "SHA1", strength: "中" },
  { name: "TLS_RSA_WITH_AES_128_CBC_SHA", ke: "RSA", auth: "RSA", enc: "AES-128-CBC", mac: "SHA1", strength: "弱" },
];

const keAlgos = [
  { name: "ECDHE", desc: "椭圆曲线 Diffie-Hellman 临时密钥交换，提供前向安全性", color: "green" },
  { name: "DHE", desc: "Diffie-Hellman 临时密钥交换，提供前向安全性", color: "blue" },
  { name: "RSA", desc: "RSA 密钥交换，无前向安全性（已弃用）", color: "red" },
];

export function TLShandshakeengine() {
  const [selected, setSelected] = useState(0);
  const [step, setStep] = useState(0);
  const suite = cipherSuites[selected];

  const steps = [
    { label: "客户端发送支持的密码套件列表", desc: "ClientHello 包含客户端支持的所有密码套件，按优先级排序" },
    { label: "服务器选择密码套件", desc: `服务器从列表中选择: ${suite.name}` },
    { label: "密钥交换协商", desc: `${suite.ke}: 双方交换公钥参数，计算共享密钥` },
    { label: "身份验证", desc: `${suite.auth}: 服务器用证书证明身份` },
    { label: "会话密钥生成", desc: "基于共享密钥生成对称加密密钥" },
    { label: "加密通信建立", desc: `${suite.enc} + ${suite.mac}: 全双工加密数据传输` },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TLS 握手引擎 (Cipher Suite Negotiation)</h3>
      <div className="mb-4">
        <h4 className="text-text-secondary text-sm mb-2">密码套件列表 (点击选择)</h4>
        <div className="space-y-1">
          {cipherSuites.map((c, i) => (
            <button key={i} onClick={() => { setSelected(i); setStep(0); }}
              className={`w-full text-left p-3 rounded-lg border-2 transition-all cursor-pointer ${
                selected === i ? "border-blue-400 bg-blue-500/10" : "border-border-subtle hover:border-gray-400"
              }`}>
              <div className="flex items-center justify-between">
                <span className={`text-xs font-mono ${selected === i ? "text-blue-400" : "text-text-primary"}`}>{c.name}</span>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  c.strength === "强" ? "bg-green-500/10 text-green-400" : c.strength === "中" ? "bg-yellow-500/10 text-yellow-400" : "bg-red-500/10 text-red-400"
                }`}>{c.strength}</span>
              </div>
            </button>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        {[
          { label: "密钥交换", value: suite.ke, color: "blue" },
          { label: "认证", value: suite.auth, color: "green" },
          { label: "加密", value: suite.enc, color: "yellow" },
          { label: "MAC", value: suite.mac, color: "purple" },
        ].map((d) => (
          <div key={d.label} className={`p-2 rounded bg-${d.color}-500/10 border border-${d.color}-400/30`}>
            <span className={`text-${d.color}-400 text-xs block`}>{d.label}</span>
            <span className="text-text-primary text-sm font-mono">{d.value}</span>
          </div>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <h4 className="text-text-primary text-sm font-medium mb-3">协商流程</h4>
        <div className="space-y-2">
          {steps.map((s, i) => (
            <button key={i} onClick={() => setStep(i)}
              className={`w-full text-left p-3 rounded-lg border transition-all cursor-pointer ${
                i === step ? "border-blue-400 bg-blue-500/10" : i < step ? "border-green-400/30 bg-green-500/5" : "border-border-subtle opacity-50"
              }`}>
              <div className="flex items-center gap-2">
                <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${i <= step ? "bg-blue-500 text-white" : "bg-gray-300 dark:bg-gray-600 text-text-muted"}`}>{i + 1}</span>
                <span className="text-sm text-text-primary">{s.label}</span>
              </div>
              {i === step && <p className="text-text-secondary text-xs mt-1 ml-7">{s.desc}</p>}
            </button>
          ))}
        </div>
      </div>
      <h4 className="text-text-secondary text-sm mb-2">密钥交换算法对比</h4>
      <div className="space-y-1">
        {keAlgos.map((a) => (
          <div key={a.name} className={`p-2 rounded bg-bg-primary border border-border-subtle`}>
            <span className={`text-${a.color}-400 text-xs font-mono font-medium`}>{a.name}</span>
            <p className="text-text-muted text-xs">{a.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
export default TLShandshakeengine;
