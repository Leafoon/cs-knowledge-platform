"use client";
import { useState } from "react";

const protocols = [
  { name: "AH", desc: "认证头 - 提供完整性与认证" },
  { name: "ESP", desc: "封装安全载荷 - 提供加密+认证" },
];

const espModes = [
  { name: "Transport", label: "传输模式", desc: "仅加密载荷，原始IP头保留" },
  { name: "Tunnel", label: "隧道模式", desc: "加密整个原始包，新增外部IP头" },
];

export function IPsecSAengine() {
  const [mode, setMode] = useState(0);
  const [protocol, setProtocol] = useState(1);
  const [step, setStep] = useState(0);

  const tunnelSteps = [
    { label: "原始数据包", detail: "IP头 + TCP头 + 数据" },
    { label: "ESP头添加", detail: "SPI + 序列号" },
    { label: "加密", detail: "原始IP头+TCP+数据 → 密文" },
    { label: "ESP尾+ICV", detail: "填充 + 填充长度 + 下一头部 + 认证标签" },
    { label: "外部IP头", detail: "新IP头（公网地址）" },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        IPsec SA Engine <span className="text-text-secondary text-sm">— 安全关联与ESP封装</span>
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {protocols.map((p, i) => (
          <button
            key={i}
            onClick={() => setProtocol(i)}
            className={`px-3 py-1 rounded text-sm ${protocol === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {p.name}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-3">{protocols[protocol].desc}</p>
      {protocol === 1 && (
        <>
          <div className="flex gap-2 mb-4">
            {espModes.map((m, i) => (
              <button
                key={i}
                onClick={() => { setMode(i); setStep(0); }}
                className={`px-3 py-1 rounded text-sm ${mode === i ? "bg-purple-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <p className="text-sm text-text-secondary mb-3">{espModes[mode].desc}</p>
          {mode === 1 && (
            <>
              <div className="flex gap-2 mb-3">
                {tunnelSteps.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setStep(i)}
                    className={`px-2 py-1 rounded text-xs ${step === i ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
                  >
                    步骤 {i + 1}
                  </button>
                ))}
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded font-mono text-sm">
                <div className="text-center mb-2 text-text-primary font-semibold">
                  {tunnelSteps[step].label}
                </div>
                <div className="flex justify-center gap-1 flex-wrap">
                  {tunnelSteps.slice(0, step + 1).map((s, i) => (
                    <span
                      key={i}
                      className={`px-2 py-1 rounded text-xs ${i === step ? "bg-green-600 text-white" : "bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200"}`}
                    >
                      {s.detail}
                    </span>
                  ))}
                </div>
              </div>
              <p className="text-xs text-text-secondary mt-2">
                SA参数: SPI=0x1A2B, 加密算法=AES-256-CBC, 认证算法=HMAC-SHA256
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}

export default IPsecSAengine;
