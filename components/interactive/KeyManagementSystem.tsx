"use client";
import { useState } from "react";

const phases = [
  {
    name: "密钥生成",
    en: "Key Generation",
    desc: "使用密码学安全随机数生成器创建密钥对",
    methods: ["RSA-2048/4096", "ECC P-256", "Ed25519"],
    color: "bg-blue-600",
  },
  {
    name: "密钥分发",
    en: "Key Distribution",
    desc: "安全地将公钥分发给通信方，私钥严格保密",
    methods: ["CA证书", "密钥交换(DH)", "带外验证"],
    color: "bg-green-600",
  },
  {
    name: "密钥存储",
    en: "Key Storage",
    desc: "密钥必须安全存储，防止未授权访问",
    methods: ["HSM硬件模块", "TPM可信平台", "密钥管理服务(KMS)"],
    color: "bg-purple-600",
  },
  {
    name: "密钥轮换",
    en: "Key Rotation",
    desc: "定期更换密钥以限制泄露影响范围",
    methods: ["自动轮换策略", "前向保密(PFS)", "密钥版本管理"],
    color: "bg-orange-600",
  },
];

export function KeyManagementSystem() {
  const [selected, setSelected] = useState(0);
  const [keyLen, setKeyLen] = useState(2048);
  const [generated, setGenerated] = useState(false);

  const genKey = () => {
    setGenerated(true);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Key Management System <span className="text-text-secondary text-sm">— 密钥生命周期</span>
      </h3>
      <div className="flex items-center gap-1 mb-4">
        {phases.map((p, i) => (
          <div key={i} className="flex items-center">
            <button
              onClick={() => setSelected(i)}
              className={`px-3 py-1.5 rounded text-sm text-white ${p.color} ${selected === i ? "ring-2 ring-offset-1 ring-blue-400" : "opacity-60"}`}
            >
              {p.name}
            </button>
            {i < phases.length - 1 && <span className="mx-1 text-text-secondary">→</span>}
          </div>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">
          {phases[selected].name} ({phases[selected].en})
        </div>
        <p className="text-sm text-text-secondary mb-3">{phases[selected].desc}</p>
        <div className="flex flex-wrap gap-2">
          {phases[selected].methods.map((m, i) => (
            <span key={i} className="px-2 py-1 rounded bg-gray-200 dark:bg-gray-700 text-sm text-text-primary">
              {m}
            </span>
          ))}
        </div>
      </div>
      {selected === 0 && (
        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
          <div className="text-sm text-text-secondary mb-2">密钥长度:</div>
          <div className="flex gap-2 mb-3">
            {[1024, 2048, 4096].map((len) => (
              <button
                key={len}
                onClick={() => { setKeyLen(len); setGenerated(false); }}
                className={`px-3 py-1 rounded text-sm ${keyLen === len ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
              >
                {len} bits
              </button>
            ))}
          </div>
          <button
            onClick={genKey}
            className="px-4 py-2 rounded bg-green-600 text-white text-sm"
          >
            生成密钥对
          </button>
          {generated && (
            <div className="mt-3 text-xs font-mono bg-gray-200 dark:bg-gray-900 p-3 rounded">
              <div className="text-text-secondary">公钥 ({keyLen} bits):</div>
              <div className="text-text-primary break-all">
                MIIBIjANBgkqh...{"*".repeat(keyLen / 64)}AQAB
              </div>
              <div className="text-text-secondary mt-2">私钥: [已安全存储到HSM]</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default KeyManagementSystem;
