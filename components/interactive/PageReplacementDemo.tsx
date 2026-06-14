"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Replace, ArrowRight, RotateCcw, Zap } from "lucide-react";

type Algorithm = "FIFO" | "LRU" | "OPT" | "Clock";

const ALGO_INFO: Record<Algorithm, { name: string; desc: string }> = {
  FIFO: { name: "FIFO（先进先出）", desc: "替换最早进入内存的页面" },
  LRU: { name: "LRU（最近最少使用）", desc: "替换最长时间未被访问的页面" },
  OPT: { name: "OPT（最优置换）", desc: "替换将来最长时间不会被使用的页面" },
  Clock: { name: "Clock（时钟算法）", desc: "使用引用位的近似LRU，环形扫描" },
};

const REF_STRING = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5];

function simulate(algo: Algorithm, frames: number): { frame: number[]; fault: boolean; replaced?: number }[] {
  const memory: (number | null)[] = Array(frames).fill(null);
  const result: { frame: number[]; fault: boolean; replaced?: number }[] = [];
  const history: number[][] = [];
  let clockHand = 0;
  const refBits = Array(frames).fill(0);

  for (let i = 0; i < REF_STRING.length; i++) {
    const page = REF_STRING[i];
    const idx = memory.indexOf(page);
    if (idx !== -1) {
      if (algo === "LRU") history.push([...memory as number[]]);
      if (algo === "Clock") refBits[idx] = 1;
      result.push({ frame: [...memory as number[]], fault: false });
      continue;
    }
    let replaceIdx = -1;
    if (memory.includes(null)) {
      replaceIdx = memory.indexOf(null);
    } else {
      switch (algo) {
        case "FIFO":
          replaceIdx = 0;
          break;
        case "LRU": {
          const lastUsed = new Map<number, number>();
          for (let j = 0; j < i; j++) lastUsed.set(REF_STRING[j], j);
          let minTime = Infinity;
          memory.forEach((p, pi) => { if (p !== null && (lastUsed.get(p) ?? -1) < minTime) { minTime = lastUsed.get(p)!; replaceIdx = pi; } });
          break;
        }
        case "OPT": {
          let farthest = -1;
          memory.forEach((p, pi) => {
            const nextUse = REF_STRING.slice(i + 1).indexOf(p!);
            const dist = nextUse === -1 ? Infinity : nextUse;
            if (dist > farthest) { farthest = dist; replaceIdx = pi; }
          });
          break;
        }
        case "Clock": {
          while (true) {
            if (refBits[clockHand] === 0) { replaceIdx = clockHand; clockHand = (clockHand + 1) % frames; break; }
            refBits[clockHand] = 0;
            clockHand = (clockHand + 1) % frames;
          }
          break;
        }
      }
    }
    const replaced = memory[replaceIdx];
    memory[replaceIdx] = page;
    if (algo === "Clock") refBits[replaceIdx] = 1;
    if (algo === "LRU") history.push([...memory as number[]]);
    result.push({ frame: [...memory as number[]], fault: true, replaced: replaced ?? undefined });
  }
  return result;
}

export function PageReplacementDemo() {
  const [algo, setAlgo] = useState<Algorithm>("FIFO");
  const [frames, setFrames] = useState(3);
  const [step, setStep] = useState(0);
  const result = simulate(algo, frames);
  const faults = result.filter(r => r.fault).length;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Replace className="w-5 h-5 text-rose-500" />
        页面置换算法演示
      </h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {(Object.keys(ALGO_INFO) as Algorithm[]).map(a => (
          <button key={a} onClick={() => { setAlgo(a); setStep(0); }}
            className={`px-3 py-1.5 rounded text-sm ${algo === a ? "bg-rose-500 text-white" : "bg-bg-subtle"}`}>
            {ALGO_INFO[a].name}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-3">{ALGO_INFO[algo].desc}</p>
      <div className="flex items-center gap-3 mb-4">
        <label className="text-sm">页框数: {frames}</label>
        <input type="range" min={2} max={5} value={frames} onChange={e => { setFrames(+e.target.value); setStep(0); }} />
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep(s => Math.min(s + 1, result.length - 1))}
          className="px-3 py-1 bg-rose-500 text-white rounded text-sm flex items-center gap-1">
          <Zap className="w-3 h-3" /> 下一步
        </button>
        <button onClick={() => setStep(0)}
          className="px-3 py-1 bg-bg-subtle rounded text-sm flex items-center gap-1">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-text-secondary">
              <th className="p-1 text-left">访问</th>
              {Array.from({ length: frames }).map((_, i) => <th key={i} className="p-1">页框{i}</th>)}
              <th className="p-1">状态</th>
            </tr>
          </thead>
          <tbody>
            {result.map((r, i) => (
              <motion.tr key={i} initial={{ opacity: 0 }}
                animate={{ opacity: i <= step ? 1 : 0.3 }}
                className={i === step ? "bg-rose-500/10" : ""}>
                <td className="p-1 font-mono font-bold">{REF_STRING[i]}</td>
                {r.frame.map((p, j) => (
                  <td key={j} className="p-1 text-center">
                    <motion.span animate={{ scale: i === step && r.fault ? 1.2 : 1 }}
                      className={`inline-block w-8 h-8 leading-8 rounded ${p !== null ? "bg-bg-subtle font-mono" : ""}`}>
                      {p ?? ""}
                    </motion.span>
                  </td>
                ))}
                <td className="p-1 text-center">
                  {r.fault ? (
                    <span className="text-red-500 text-xs">缺页{r.replaced !== undefined ? ` (换出${r.replaced})` : ""}</span>
                  ) : (
                    <span className="text-green-500 text-xs">命中</span>
                  )}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 text-sm text-text-secondary">
        缺页次数: <span className="font-bold text-rose-500">{result.slice(0, step + 1).filter(r => r.fault).length}</span>
        {step === result.length - 1 && <span> / 总缺页: {faults} | 缺页率: {((faults / REF_STRING.length) * 100).toFixed(0)}%</span>}
      </div>
    </div>
  );
}
