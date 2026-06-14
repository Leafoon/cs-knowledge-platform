"use client";
import { useState } from "react";

export function UDPChecksumCalculator() {
  const [words, setWords] = useState(["4500", "0030", "0001", "0000", "4011", "c0a8", "0101", "c0a8", "0102"]);
  const [result, setResult] = useState<{ sum: string; complement: string; steps: { a: string; b: string; carry: string; sum: string }[] } | null>(null);
  const [verifyInput, setVerifyInput] = useState("");
  const [verifyResult, setVerifyResult] = useState<string | null>(null);

  const hexToBin = (hex: string) => parseInt(hex, 16).toString(2).padStart(16, "0");
  const binToHex = (bin: string) => parseInt(bin, 2).toString(16).padStart(4, "0");

  const addOnesComplement = (a: string, b: string) => {
    const numA = parseInt(a, 16);
    const numB = parseInt(b, 16);
    let sum = numA + numB;
    if (sum > 0xFFFF) {
      sum = (sum & 0xFFFF) + 1;
    }
    return { result: sum.toString(16).padStart(4, "0"), carry: numA + numB > 0xFFFF ? "1" : "0" };
  };

  const calculate = () => {
    let sum = "0000";
    const steps: { a: string; b: string; carry: string; sum: string }[] = [];
    for (const word of words) {
      const { result: r, carry } = addOnesComplement(sum, word);
      steps.push({ a: sum, b: word, carry, sum: r });
      sum = r;
    }
    const complement = (0xFFFF - parseInt(sum, 16)).toString(16).padStart(4, "0");
    setResult({ sum, complement, steps });
  };

  const verify = () => {
    let sum = "0000";
    const allWords = [...words, verifyInput];
    for (const word of allWords) {
      const { result: r } = addOnesComplement(sum, word);
      sum = r;
    }
    setVerifyResult(sum === "ffff" ? "✅ 校验和正确（反码和 = 0xFFFF）" : `❌ 校验和错误（反码和 = ${sum}，应为 FFFF）`);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">UDP 校验和计算器</h3>
      <div className="mb-4">
        <label className="text-text-secondary text-sm mb-2 block">输入16位十六进制字（伪头部 + UDP头部 + 数据）</label>
        <div className="flex flex-wrap gap-1 mb-2">
          {words.map((w, i) => (
            <input key={i} value={w} onChange={(e) => { const n = [...words]; n[i] = e.target.value; setWords(n); }}
              className="w-16 px-2 py-1 rounded border border-border-subtle bg-bg-primary text-text-primary text-xs font-mono text-center" maxLength={4} />
          ))}
        </div>
        <div className="flex gap-2">
          <button onClick={() => setWords([...words, "0000"])} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-xs">+ 添加字</button>
          <button onClick={() => setWords(words.slice(0, -1))} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-xs">- 删除字</button>
        </div>
      </div>
      <button onClick={calculate} className="px-4 py-2 rounded bg-blue-500 text-white text-sm mb-4 hover:bg-blue-600">计算校验和</button>
      {result && (
        <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
          <h4 className="text-text-primary text-sm font-medium mb-2">计算过程（反码求和）</h4>
          <div className="space-y-1 mb-3 max-h-40 overflow-y-auto">
            {result.steps.map((s, i) => (
              <div key={i} className="flex items-center gap-2 text-xs font-mono">
                <span className="text-text-muted w-4">{i + 1}.</span>
                <span className="text-blue-400">{s.a}</span>
                <span className="text-text-muted">+</span>
                <span className="text-green-400">{s.b}</span>
                <span className="text-text-muted">=</span>
                <span className="text-yellow-400">{s.sum}</span>
                {s.carry === "1" && <span className="text-red-400 text-[10px]">(有进位，回卷+1)</span>}
              </div>
            ))}
          </div>
          <div className="p-2 rounded bg-red-500/10 border border-red-400/30">
            <span className="text-text-secondary text-xs">反码和: </span>
            <span className="text-yellow-400 font-mono text-sm">{result.sum}</span>
            <span className="text-text-secondary text-xs ml-3">取反 = 校验和: </span>
            <span className="text-green-400 font-mono text-sm">{result.complement}</span>
          </div>
        </div>
      )}
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-secondary text-xs font-medium mb-1">验证校验和</h4>
        <div className="flex gap-2">
          <input value={verifyInput} onChange={(e) => setVerifyInput(e.target.value)} placeholder="输入校验和"
            className="px-2 py-1 rounded border border-border-subtle bg-bg-elevated text-text-primary text-xs font-mono w-20" maxLength={4} />
          <button onClick={verify} className="px-3 py-1 rounded bg-green-500 text-white text-xs">验证</button>
        </div>
        {verifyResult && <p className="text-text-primary text-xs mt-2">{verifyResult}</p>}
      </div>
      <p className="text-text-muted text-xs mt-3">反码求和：进位回卷加到最低位，最后取反。所有字（含校验和）相加应为 0xFFFF。</p>
    </div>
  );
}
export default UDPChecksumCalculator;
