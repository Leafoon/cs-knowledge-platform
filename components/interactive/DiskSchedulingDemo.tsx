"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, RotateCcw, Zap, HardDrive } from "lucide-react";

type Algo = "FCFS" | "SSTF" | "SCAN" | "C-SCAN" | "LOOK";

const ALGO_DESC: Record<Algo, string> = {
  FCFS: "先来先服务：按请求顺序依次访问",
  SSTF: "最短寻道优先：选择最近的请求",
  SCAN: "电梯算法：同方向移动到端点后折返",
  "C-SCAN": "循环扫描：到端点后跳回起点",
  LOOK: "类似SCAN但不到端点即折返",
};

function schedule(algo: Algo, requests: number[], start: number): number[] {
  const seq = [start];
  const rem = [...requests];
  switch (algo) {
    case "FCFS": seq.push(...rem); break;
    case "SSTF": {
      let pos = start;
      while (rem.length) {
        let mi = 0, md = Infinity;
        rem.forEach((r, i) => { const d = Math.abs(r - pos); if (d < md) { md = d; mi = i; } });
        pos = rem.splice(mi, 1)[0];
        seq.push(pos);
      }
      break;
    }
    case "SCAN": {
      const sorted = [...rem].sort((a, b) => a - b);
      const left = sorted.filter(r => r < start);
      const right = sorted.filter(r => r >= start);
      seq.push(...right, 199, ...left.reverse());
      break;
    }
    case "C-SCAN": {
      const sorted = [...rem].sort((a, b) => a - b);
      const left = sorted.filter(r => r < start);
      const right = sorted.filter(r => r >= start);
      seq.push(...right, 199, 0, ...left);
      break;
    }
    case "LOOK": {
      const sorted = [...rem].sort((a, b) => a - b);
      const left = sorted.filter(r => r < start);
      const right = sorted.filter(r => r >= start);
      seq.push(...right, ...left.reverse());
      break;
    }
  }
  return seq;
}

export function DiskSchedulingDemo() {
  const [algo, setAlgo] = useState<Algo>("FCFS");
  const [start] = useState(50);
  const [requests] = useState([98, 183, 37, 122, 14, 124, 65, 67]);
  const [step, setStep] = useState(0);
  const seq = schedule(algo, requests, start);
  const totalSeek = seq.slice(1).reduce((s, v, i) => s + Math.abs(v - seq[i]), 0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <HardDrive className="w-5 h-5 text-emerald-500" />
        磁盘调度算法
      </h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {(["FCFS", "SSTF", "SCAN", "C-SCAN", "LOOK"] as Algo[]).map(a => (
          <button key={a} onClick={() => { setAlgo(a); setStep(0); }}
            className={`px-3 py-1.5 rounded text-sm ${algo === a ? "bg-emerald-500 text-white" : "bg-bg-subtle"}`}>
            {a}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-3">{ALGO_DESC[algo]}</p>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep(s => Math.min(s + 1, seq.length - 1))}
          className="px-3 py-1 bg-emerald-500 text-white rounded text-sm flex items-center gap-1">
          <Zap className="w-3 h-3" /> 下一步
        </button>
        <button onClick={() => setStep(0)}
          className="px-3 py-1 bg-bg-subtle rounded text-sm flex items-center gap-1">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>
      <div className="relative h-48 mb-4">
        <div className="absolute bottom-0 left-0 right-0 h-px bg-border-subtle" />
        {seq.map((pos, i) => {
          if (i > step) return null;
          const x = (pos / 200) * 100;
          const isStart = i === 0;
          return (
            <motion.div key={i} initial={{ bottom: 0, left: `${x}%` }}
              animate={{ bottom: isStart ? 0 : `${(i / seq.length) * 160}px`, left: `${x}%` }}
              className="absolute flex flex-col items-center">
              <div className={`w-3 h-3 rounded-full ${isStart ? "bg-emerald-500" : "bg-emerald-500/60"}`} />
              <span className="text-xs font-mono mt-1">{pos}</span>
            </motion.div>
          );
        })}
        {step > 0 && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {Array.from({ length: step }).map((_, i) => {
              const x1 = (seq[i] / 200) * 100;
              const x2 = (seq[i + 1] / 200) * 100;
              const y1 = 192 - (i / seq.length) * 160;
              const y2 = 192 - ((i + 1) / seq.length) * 160;
              return <line key={i} x1={`${x1}%`} y1={y1} x2={`${x2}%`} y2={y2}
                stroke="rgb(16 185 129)" strokeWidth="1.5" strokeDasharray="4" />;
            })}
          </svg>
        )}
      </div>
      <div className="flex items-center gap-4 text-sm">
        <span>当前位置: <span className="font-mono font-bold text-emerald-500">{seq[step]}</span></span>
        {step > 0 && (
          <span>寻道距离: <span className="font-mono">{Math.abs(seq[step] - seq[step - 1])}</span></span>
        )}
        <span>总寻道: <span className="font-mono font-bold">{seq.slice(1, step + 1).reduce((s, v, i) => s + Math.abs(v - seq[i]), 0)}</span></span>
      </div>
    </div>
  );
}
