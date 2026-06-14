"use client";
import { useState } from "react";

const steps = [
  { name: "SubBytes", label: "字节替换", desc: "使用 S-Box 对每个字节进行非线性替换，提供混淆性" },
  { name: "ShiftRows", label: "行移位", desc: "将状态矩阵的每一行循环左移不同的偏移量" },
  { name: "MixColumns", label: "列混合", desc: "对每一列进行有限域上的矩阵乘法，提供扩散性" },
  { name: "AddRoundKey", label: "轮密钥加", desc: "将轮密钥与状态矩阵进行异或运算" },
];

const sampleState = [
  ["63", "EB", "9F", "A0"],
  ["2C", "53", "41", "C7"],
  ["D4", "6B", "86", "5E"],
  ["F1", "3D", "2A", "1B"],
];

const transforms: Record<string, string[][]> = {
  SubBytes: [
    ["FB", "AC", "E0", "63"],
    ["71", "53", "83", "85"],
    ["B2", "6F", "32", "4A"],
    ["8C", "C7", "37", "F1"],
  ],
  ShiftRows: [
    ["FB", "AC", "E0", "63"],
    ["53", "83", "85", "71"],
    ["32", "4A", "B2", "6F"],
    ["F1", "8C", "C7", "37"],
  ],
  MixColumns: [
    ["4F", "7A", "2D", "B8"],
    ["9C", "15", "E3", "A6"],
    ["D8", "62", "4E", "71"],
    ["35", "B9", "F0", "C4"],
  ],
  AddRoundKey: [
    ["2A", "5E", "13", "D7"],
    ["4B", "88", "3F", "6C"],
    ["A1", "27", "9D", "B5"],
    ["60", "E4", "5A", "19"],
  ],
};

export function AESEncryptEngine() {
  const [activeStep, setActiveStep] = useState(0);
  const step = steps[activeStep];
  const state = activeStep === 0 ? sampleState : transforms[steps[activeStep - 1].name];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">AES 加密引擎可视化</h3>
      <div className="flex gap-1 mb-4">
        {steps.map((s, i) => (
          <button key={s.name} onClick={() => setActiveStep(i)} className={`flex-1 px-2 py-2 rounded text-xs font-mono transition-colors ${activeStep === i ? "bg-indigo-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary hover:bg-gray-300 dark:hover:bg-gray-600"}`}>
            {s.name}
          </button>
        ))}
      </div>
      <div className="mb-4 p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <p className="text-sm font-semibold text-text-primary">{step.label} ({step.name})</p>
        <p className="text-xs text-text-secondary mt-1">{step.desc}</p>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-text-secondary mb-2">输入状态矩阵</p>
          <div className="grid grid-cols-4 gap-1">
            {state.flat().map((v, i) => (
              <div key={i} className="p-2 text-center text-xs font-mono rounded bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200">{v}</div>
            ))}
          </div>
        </div>
        <div>
          <p className="text-xs text-text-secondary mb-2">输出状态矩阵</p>
          <div className="grid grid-cols-4 gap-1">
            {transforms[step.name].flat().map((v, i) => (
              <div key={i} className="p-2 text-center text-xs font-mono rounded bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200">{v}</div>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-4 flex items-center gap-2">
        <button onClick={() => setActiveStep(Math.max(0, activeStep - 1))} disabled={activeStep === 0} className="px-3 py-1 rounded text-sm bg-gray-200 dark:bg-gray-700 text-text-secondary disabled:opacity-50">上一步</button>
        <div className="flex-1 h-1 bg-gray-200 dark:bg-gray-700 rounded-full"><div className="h-full bg-indigo-500 rounded-full transition-all" style={{ width: `${((activeStep + 1) / steps.length) * 100}%` }} /></div>
        <button onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))} disabled={activeStep === steps.length - 1} className="px-3 py-1 rounded text-sm bg-gray-200 dark:bg-gray-700 text-text-secondary disabled:opacity-50">下一步</button>
      </div>
    </div>
  );
}
export default AESEncryptEngine;
