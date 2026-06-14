"use client";
import { useState } from "react";

export function CRCVisualizer() {
  const [data, setData] = useState("1101011011");
  const [generator, setGenerator] = useState("10011");
  const [step, setStep] = useState(0);

  const gLen = generator.length;
  const dividend = data + "0".repeat(gLen - 1);
  const genBits = generator.split("").map(Number);

  const computeSteps = () => {
    const steps: { remainder: string; xorWith: string; position: number }[] = [];
    let rem = dividend.slice(0, gLen);
    for (let i = 0; i <= dividend.length - gLen; i++) {
      if (i > 0) rem = rem.slice(1) + dividend[gLen - 1 + i];
      const shouldXor = rem[0] === "1";
      const xorStr = shouldXor ? generator : "0".repeat(gLen);
      const newRem = rem.split("").map((c, j) => (shouldXor ? (c === generator[j] ? "0" : "1") : c)).join("");
      steps.push({ remainder: rem, xorWith: xorStr, position: i });
      rem = newRem;
    }
    return steps;
  };

  const steps = computeSteps();
  const currentRemainder = step > 0 && step <= steps.length ? steps[step - 1].remainder : "";

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">CRC 模2除法可视化</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs text-text-secondary">数据</label>
          <input value={data} onChange={(e) => { setData(e.target.value.replace(/[^01]/g, "")); setStep(0); }}
            className="w-full mt-1 px-2 py-1.5 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 font-mono text-sm text-text-primary" />
        </div>
        <div>
          <label className="text-xs text-text-secondary">生成多项式</label>
          <input value={generator} onChange={(e) => { setGenerator(e.target.value.replace(/[^01]/g, "")); setStep(0); }}
            className="w-full mt-1 px-2 py-1.5 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 font-mono text-sm text-text-primary" />
        </div>
      </div>
      <div className="mb-4 p-3 rounded bg-gray-50 dark:bg-gray-800 font-mono text-sm overflow-x-auto">
        <div className="text-text-secondary mb-1">被除数: {dividend}</div>
        <div className="text-text-secondary">除数: {generator}</div>
      </div>
      <div className="space-y-1 mb-4 font-mono text-xs">
        {steps.slice(0, step).map((s, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="w-6 text-text-secondary text-right">{i + 1}.</span>
            <span className="text-text-primary">{" ".repeat(i)}{s.remainder}</span>
            <span className="text-text-secondary">⊕</span>
            <span className="text-blue-600 dark:text-blue-400">{" ".repeat(i)}{s.xorWith}</span>
            {s.remainder[0] === "0" && <span className="text-yellow-600 dark:text-yellow-400 text-[10px]">(skip)</span>}
          </div>
        ))}
        {step > 0 && step <= steps.length && (
          <div className="flex items-center gap-2 border-t border-border-subtle pt-1">
            <span className="w-6 text-text-secondary text-right">=</span>
            <span className="text-green-600 dark:text-green-400">{" ".repeat(step - 1)}{steps[step - 1].remainder.replace(/^./, "0")}</span>
          </div>
        )}
      </div>
      {step >= steps.length && (
        <div className="p-3 rounded bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 mb-4">
          <p className="text-xs text-green-600 dark:text-green-400 mb-1">CRC 余数</p>
          <p className="font-mono text-lg font-bold text-green-700 dark:text-green-300">{steps[steps.length - 1].remainder}</p>
          <p className="text-xs text-text-secondary mt-1">附加CRC后的发送帧: {data}{steps[steps.length - 1].remainder}</p>
        </div>
      )}
      <div className="flex gap-2">
        <button onClick={() => setStep(Math.min(step + 1, steps.length))}
          className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">下一步</button>
        <button onClick={() => setStep(steps.length)}
          className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm">全部执行</button>
        <button onClick={() => setStep(0)} className="px-4 py-2 bg-gray-500 text-white rounded text-sm">重置</button>
      </div>
      <p className="text-xs text-text-secondary mt-3">CRC通过模2除法生成校验码，接收端用相同多项式验证余数是否为0来检测传输错误。</p>
    </div>
  );
}
export default CRCVisualizer;
