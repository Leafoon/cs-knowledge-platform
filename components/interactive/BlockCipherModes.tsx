"use client";
import { useState } from "react";

type Mode = "ECB" | "CBC" | "CFB" | "OFB" | "CTR";

const modeInfo: Record<Mode, { name: string; desc: string; formula: string; parallel: boolean; error: string }> = {
  ECB: { name: "电子密码本", desc: "每个块独立加密", formula: "Cᵢ = E(K, Pᵢ)", parallel: true, error: "单块错误影响当前块" },
  CBC: { name: "密码块链接", desc: "前一密文块参与当前加密", formula: "Cᵢ = E(K, Pᵢ ⊕ Cᵢ₋₁)", parallel: false, error: "影响当前及下一块" },
  CFB: { name: "密码反馈", desc: "密文反馈到加密输入", formula: "Cᵢ = Pᵢ ⊕ E(K, Cᵢ₋₁)", parallel: false, error: "影响当前及后续若干块" },
  OFB: { name: "输出反馈", desc: "密钥流独立于明文/密文", formula: "Oᵢ = E(K, Oᵢ₋₁); Cᵢ = Pᵢ ⊕ Oᵢ", parallel: false, error: "仅影响对应位" },
  CTR: { name: "计数器模式", desc: "对计数器加密生成密钥流", formula: "Cᵢ = Pᵢ ⊕ E(K, counter+i)", parallel: true, error: "仅影响对应位" },
};

const blocks = ["P1=HELLO", "P2=WORLD", "P3=CRYPT", "P4=OGRAPH"];

export function BlockCipherModes() {
  const [mode, setMode] = useState<Mode>("ECB");
  const [step, setStep] = useState(0);
  const info = modeInfo[mode];

  const encryptBlock = (plain: string, idx: number) => {
    return plain.split("").map((c) => String.fromCharCode(((c.charCodeAt(0) + 3 + idx * 7) % 94) + 33)).join("");
  };

  const getCipher = (idx: number) => {
    if (idx > step) return "?????";
    return encryptBlock(blocks[idx], idx);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">分组密码模式对比</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {(Object.keys(modeInfo) as Mode[]).map((m) => (
          <button key={m} onClick={() => { setMode(m); setStep(0); }}
            className={`px-3 py-1.5 rounded text-sm font-mono ${mode === m ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {m}
          </button>
        ))}
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 mb-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm font-bold text-text-primary">{info.name}</span>
          <span className={`text-xs px-2 py-0.5 rounded ${info.parallel ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300" : "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300"}`}>
            {info.parallel ? "可并行" : "串行"}
          </span>
        </div>
        <p className="text-xs text-text-secondary">{info.desc}</p>
        <p className="text-xs font-mono text-text-primary mt-1">{info.formula}</p>
      </div>
      <div className="space-y-2 mb-4">
        {blocks.map((b, i) => (
          <div key={i} className={`flex items-center gap-3 p-2 rounded border transition-all ${i === step ? "border-blue-400 bg-blue-50 dark:bg-blue-900/20" : i < step ? "border-green-300 bg-green-50 dark:bg-green-900/10" : "border-border-subtle opacity-40"}`}>
            <span className="text-xs text-text-secondary w-6">{i + 1}</span>
            <div className="flex-1 flex items-center gap-2">
              <code className="text-xs bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded font-mono text-text-primary">{b}</code>
              <span className="text-xs text-text-secondary">→</span>
              <code className="text-xs bg-blue-100 dark:bg-blue-900/30 px-2 py-0.5 rounded font-mono text-blue-700 dark:text-blue-300">
                {getCipher(i)}
              </code>
            </div>
            {mode !== "ECB" && i > 0 && i <= step && (
              <span className="text-[10px] text-text-secondary">⊕ prev</span>
            )}
          </div>
        ))}
      </div>
      <div className="flex gap-2 mb-3">
        <button onClick={() => setStep(Math.min(step + 1, 3))}
          className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">
          加密下一块
        </button>
        <button onClick={() => setStep(0)} className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded text-sm">重置</button>
      </div>
      <div className="p-2 rounded bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800">
        <p className="text-xs text-yellow-700 dark:text-yellow-300">传播错误: {info.error}</p>
      </div>
    </div>
  );
}
export default BlockCipherModes;
