"use client";
import { useState } from "react";

export function HammingCodeCircuit() {
  const [dataBits, setDataBits] = useState([1, 0, 1, 1]);

  const p1 = dataBits[0] ^ dataBits[1] ^ dataBits[3];
  const p2 = dataBits[0] ^ dataBits[1] ^ dataBits[2];
  const p3 = dataBits[0] ^ dataBits[2] ^ dataBits[3];

  const encoded = [p1, p2, dataBits[0], p3, dataBits[1], dataBits[2], dataBits[3]];
  const posLabels = ["p1", "p2", "d1", "p3", "d2", "d3", "d4"];

  const toggleData = (i: number) => {
    const next = [...dataBits];
    next[i] = next[i] === 0 ? 1 : 0;
    setDataBits(next);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">海明码电路</h3>
      <p className="text-sm text-text-secondary mb-4">Hamming(7,4) 编码器电路设计 — 点击数据位切换</p>

      <div className="mb-4">
        <p className="text-xs text-text-muted mb-2">输入数据位 (d1 d2 d3 d4):</p>
        <div className="flex gap-2">
          {dataBits.map((b, i) => (
            <button key={i} onClick={() => toggleData(i)}
              className={`w-12 h-10 rounded font-mono text-sm font-bold border transition-all ${
                b ? "bg-blue-500/20 border-blue-500 text-blue-400" : "bg-bg-subtle border-border-subtle text-text-secondary"
              }`}>
              {b}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 rounded-lg bg-bg-muted border border-border-subtle mb-4">
        <p className="text-xs text-text-muted mb-3">校验位计算电路:</p>
        <div className="space-y-2 font-mono text-sm">
          <div className="flex items-center gap-2">
            <span className="w-8 text-yellow-400">p1</span>
            <span className="text-text-muted">=</span>
            <span className="text-blue-400">d1({dataBits[0]})</span>
            <span className="text-text-muted">⊕</span>
            <span className="text-blue-400">d2({dataBits[1]})</span>
            <span className="text-text-muted">⊕</span>
            <span className="text-blue-400">d4({dataBits[3]})</span>
            <span className="text-text-muted">=</span>
            <span className="text-green-400 font-bold">{p1}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-8 text-yellow-400">p2</span>
            <span className="text-text-muted">=</span>
            <span className="text-blue-400">d1({dataBits[0]})</span>
            <span className="text-text-muted">⊕</span>
            <span className="text-blue-400">d2({dataBits[1]})</span>
            <span className="text-text-muted">⊕</span>
            <span className="text-blue-400">d3({dataBits[2]})</span>
            <span className="text-text-muted">=</span>
            <span className="text-green-400 font-bold">{p2}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-8 text-yellow-400">p3</span>
            <span className="text-text-muted">=</span>
            <span className="text-blue-400">d1({dataBits[0]})</span>
            <span className="text-text-muted">⊕</span>
            <span className="text-blue-400">d3({dataBits[2]})</span>
            <span className="text-text-muted">⊕</span>
            <span className="text-blue-400">d4({dataBits[3]})</span>
            <span className="text-text-muted">=</span>
            <span className="text-green-400 font-bold">{p3}</span>
          </div>
        </div>
      </div>

      <div>
        <p className="text-xs text-text-muted mb-2">编码输出 (7位):</p>
        <div className="flex gap-1">
          {encoded.map((b, i) => (
            <div key={i} className="flex flex-col items-center">
              <span className={`w-10 h-10 rounded flex items-center justify-center font-mono text-sm font-bold ${
                posLabels[i].startsWith("p") ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500" : "bg-blue-500/20 text-blue-400 border border-blue-500"
              }`}>{b}</span>
              <span className="text-[10px] text-text-muted mt-0.5">{posLabels[i]}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
export default HammingCodeCircuit;
