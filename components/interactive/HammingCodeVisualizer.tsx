"use client";
import { useState } from "react";

export function HammingCodeVisualizer() {
  const [dataBits, setDataBits] = useState("1011001");
  const [errorPos, setErrorPos] = useState(0);

  const data = dataBits.replace(/[^01]/g, "").split("").map(Number);
  const m = data.length;
  const r = Math.ceil(Math.log2(m + Math.ceil(Math.log2(m)) + 1));
  const n = m + r;

  const positions: (number | null)[] = new Array(n + 1).fill(null);
  const parityPositions: number[] = [];
  let dataIdx = 0;

  for (let i = 1; i <= n; i++) {
    if ((i & (i - 1)) === 0) {
      parityPositions.push(i);
      positions[i] = 0;
    } else if (dataIdx < m) {
      positions[i] = data[dataIdx++];
    }
  }

  for (const p of parityPositions) {
    let count = 0;
    for (let i = 1; i <= n; i++) {
      if (i & p) count += positions[i]!;
    }
    positions[p] = count % 2;
  }

  const encoded = positions.slice(1);

  const encodedWithError = [...encoded];
  if (errorPos > 0 && errorPos <= n) {
    encodedWithError[errorPos - 1] = encodedWithError[errorPos - 1] === 0 ? 1 : 0;
  }

  let syndrome = 0;
  for (const p of parityPositions) {
    let count = 0;
    for (let i = 1; i <= n; i++) {
      if (i & p) count += encodedWithError[i - 1] ?? 0;
    }
    if (count % 2 !== 0) syndrome += p;
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">海明码编码与错误检测</h3>
      <div className="mb-4">
        <label className="text-sm text-text-secondary mb-1 block">数据位 (如7位):</label>
        <input type="text" value={dataBits} onChange={(e) => setDataBits(e.target.value.replace(/[^01]/g, "").slice(0, 11))}
          className="w-full px-3 py-2 rounded border border-border-subtle bg-bg-subtle text-text-primary font-mono" />
      </div>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        <div className="text-sm text-text-secondary mb-2">编码结果 (n={n}, 数据{m}位, 校验{r}位):</div>
        <div className="flex flex-wrap gap-1 font-mono">
          {encoded.map((bit, i) => {
            const pos = i + 1;
            const isParity = parityPositions.includes(pos);
            const hasError = syndrome === pos;
            return (
              <span key={i} className={`px-2 py-1 rounded text-sm ${
                hasError ? "bg-red-500 text-white animate-pulse" :
                isParity ? "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300" :
                "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
              }`}>
                {encodedWithError[i]}
                <span className="text-[10px] ml-0.5 opacity-60">{pos}</span>
              </span>
            );
          })}
        </div>
        <div className="flex gap-3 mt-2 text-xs text-text-secondary">
          <span className="px-2 py-0.5 bg-yellow-100 dark:bg-yellow-900/30 rounded">校验位</span>
          <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 rounded">数据位</span>
          {syndrome > 0 && <span className="px-2 py-0.5 bg-red-100 dark:bg-red-900/30 rounded text-red-500">错误位</span>}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary mb-1 block">注入错误位置 (0=无错误):</label>
          <input type="range" min={0} max={n} value={errorPos} onChange={(e) => setErrorPos(Number(e.target.value))} className="w-full" />
          <span className="text-xs text-text-secondary">位置: {errorPos === 0 ? "无错误" : errorPos}</span>
        </div>
        {syndrome > 0 && (
          <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-3">
            <div className="text-sm text-red-700 dark:text-red-300 font-bold">检测到错误!</div>
            <div className="text-xs text-red-600 dark:text-red-400">综合征={syndrome}, 错误在位置{syndrome}</div>
          </div>
        )}
      </div>
      <div className="text-xs text-text-secondary">
        海明码: 2^r ≥ m+r+1。校验位放在2的幂次位置(1,2,4,8...)。每个校验位覆盖二进制表示中对应位为1的所有位置。可纠正1位错误。
      </div>
    </div>
  );
}

export default HammingCodeVisualizer;
