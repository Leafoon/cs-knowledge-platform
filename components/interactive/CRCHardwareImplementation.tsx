"use client";
import { useState } from "react";

export function CRCHardwareImplementation() {
  const [input, setInput] = useState("1101011011");
  const [generator, setGenerator] = useState("10011");
  const [step, setStep] = useState(0);
  const [registers, setRegisters] = useState<number[]>([]);

  const gLen = generator.length;
  const initRegs = () => Array.from({ length: gLen - 1 }, () => 0);

  const runStep = () => {
    const dataBits = (input + "0".repeat(gLen - 1)).split("").map(Number);
    const genBits = generator.split("").map(Number);
    const regs = initRegs();

    if (step === 0) {
      setRegisters(regs);
      setStep(1);
      return;
    }

    const curRegs = registers.length > 0 ? [...registers] : regs;
    const bitIdx = step - 1;
    if (bitIdx >= dataBits.length) return;

    const newRegs = [...curRegs];
    const feedback = (newRegs[gLen - 2] ?? 0) ^ dataBits[bitIdx];
    for (let j = gLen - 2; j > 0; j--) {
      newRegs[j] = newRegs[j - 1] ^ (genBits[j + 1] ? feedback : 0);
    }
    newRegs[0] = feedback;
    setRegisters(newRegs);
    setStep(step + 1);
  };

  const reset = () => { setStep(0); setRegisters(initRegs()); };

  const totalSteps = input.length + gLen - 1;
  const remainder = step > totalSteps ? registers.map((r) => r).join("") : null;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">CRC 硬件实现 (LFSR 移位寄存器)</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs text-text-secondary">数据输入</label>
          <input value={input} onChange={(e) => { setInput(e.target.value.replace(/[^01]/g, "")); reset(); }}
            className="w-full mt-1 px-2 py-1.5 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 text-text-primary font-mono text-sm" />
        </div>
        <div>
          <label className="text-xs text-text-secondary">生成多项式</label>
          <input value={generator} onChange={(e) => { setGenerator(e.target.value.replace(/[^01]/g, "")); reset(); }}
            className="w-full mt-1 px-2 py-1.5 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 text-text-primary font-mono text-sm" />
        </div>
      </div>
      <div className="mb-4">
        <p className="text-xs text-text-secondary mb-2">LFSR 寄存器状态</p>
        <div className="flex items-center gap-1 justify-center">
          <span className="text-xs text-text-secondary mr-2">输入→</span>
          {Array.from({ length: gLen - 1 }).map((_, i) => (
            <div key={i} className="flex items-center gap-1">
              <div className={`w-12 h-12 rounded border-2 flex items-center justify-center text-lg font-mono font-bold transition-all ${registers[i] !== undefined ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300" : "border-gray-300 dark:border-gray-600 text-text-secondary"}`}>
                {registers[i] !== undefined ? registers[i] : "0"}
              </div>
              {i < gLen - 2 && <span className="text-text-secondary">→</span>}
            </div>
          ))}
          <span className="text-xs text-text-secondary ml-2">→ 反馈</span>
        </div>
      </div>
      <div className="mb-4 p-3 rounded bg-gray-50 dark:bg-gray-800">
        <div className="flex justify-between text-xs text-text-secondary mb-1">
          <span>步骤进度</span>
          <span>{step}/{totalSteps}</span>
        </div>
        <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${(step / totalSteps) * 100}%` }} />
        </div>
        {step > 0 && step <= totalSteps && (
          <p className="text-xs text-text-secondary mt-2">
            处理输入位 [{step - 1}]: {(input + "0".repeat(gLen - 1))[step - 1]}
          </p>
        )}
      </div>
      {remainder && (
        <div className="p-3 rounded bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 mb-4">
          <p className="text-xs text-green-600 dark:text-green-400 mb-1">CRC 余数</p>
          <p className="font-mono text-lg font-bold text-green-700 dark:text-green-300">{remainder}</p>
          <p className="text-xs text-text-secondary mt-1">发送数据: {input} + {remainder} = {input}{remainder}</p>
        </div>
      )}
      <div className="flex gap-2">
        <button onClick={runStep} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">移位一步</button>
        <button onClick={() => { let s = step; for (let i = s; i < totalSteps; i++) runStep(); }}
          className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm">运行全部</button>
        <button onClick={reset} className="px-4 py-2 bg-gray-500 text-white rounded text-sm">重置</button>
      </div>
      <p className="text-xs text-text-secondary mt-3">LFSR用XOR门和触发器实现多项式除法，每次移位处理一个输入位，最终寄存器值即为CRC余数。</p>
    </div>
  );
}
export default CRCHardwareImplementation;
