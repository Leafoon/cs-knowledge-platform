"use client";
import { useState } from "react";

interface SACKBlock {
  left: number;
  right: number;
}

export function SACKVisualizer() {
  const [totalPkts, setTotalPkts] = useState(16);
  const [acked, setAcked] = useState<boolean[]>(new Array(16).fill(false));
  const [sacks, setSacks] = useState<SACKBlock[]>([]);
  const [lost, setLost] = useState<Set<number>>(new Set([3, 7, 10]));

  const toggleAck = (i: number) => {
    const next = [...acked];
    next[i] = !next[i];
    setAcked(next);
    computeSACK(next, lost);
  };

  const toggleLost = (i: number) => {
    const next = new Set(lost);
    next.has(i) ? next.delete(i) : next.add(i);
    setLost(next);
    computeSACK(acked, next);
  };

  const computeSACK = (ackArr: boolean[], lostSet: Set<number>) => {
    const blocks: SACKBlock[] = [];
    let start = -1;
    for (let i = 0; i < ackArr.length; i++) {
      if (ackArr[i] && !lostSet.has(i)) {
        if (start === -1) start = i;
      } else {
        if (start !== -1) {
          blocks.push({ left: start, right: i - 1 });
          start = -1;
        }
      }
    }
    if (start !== -1) blocks.push({ left: start, right: ackArr.length - 1 });
    setSacks(blocks);
  };

  const cumulativeAck = acked.findIndex((a, i) => !a && !lost.has(i));

  const handleReset = () => {
    const newAcked = new Array(totalPkts).fill(false);
    for (let i = 0; i < totalPkts; i++) {
      if (!lost.has(i)) newAcked[i] = Math.random() > 0.3;
    }
    setAcked(newAcked);
    computeSACK(newAcked, lost);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">SACK可视化</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={handleReset} className="px-3 py-1.5 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-xs font-medium hover:bg-sky-500/25 transition-colors">模拟随机ACK</button>
        <span className="text-[10px] text-text-tertiary flex items-center">点击数据包切换ACK/丢包状态</span>
      </div>
      <div className="flex flex-wrap gap-1 mb-4">
        {Array.from({ length: totalPkts }, (_, i) => {
          const isLost = lost.has(i);
          const isAcked = acked[i] && !isLost;
          const inSack = sacks.some((s) => i >= s.left && i <= s.right);
          let cls = "bg-bg-tertiary border-border-subtle text-text-secondary";
          if (isLost) cls = "bg-red-500/20 border-red-400/40 text-red-600 dark:text-red-400";
          else if (inSack) cls = "bg-emerald-500/20 border-emerald-400/40 text-emerald-600 dark:text-emerald-400";
          else if (isAcked) cls = "bg-sky-500/20 border-sky-400/40 text-sky-600 dark:text-sky-400";
          return (
            <div key={i} className="flex flex-col gap-0.5">
              <button onClick={() => toggleAck(i)} className={`w-9 h-9 rounded border text-xs font-mono transition-all ${cls}`}>
                {i}
              </button>
              <button onClick={() => toggleLost(i)}
                className={`text-[8px] px-0.5 rounded ${isLost ? "bg-red-500/20 text-red-500" : "text-text-tertiary hover:text-red-400"}`}>
                {isLost ? "丢" : "丢?"}
              </button>
            </div>
          );
        })}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 mb-3">
        <div className="text-xs font-medium text-text-primary mb-2">SACK确认信息</div>
        <div className="space-y-1">
          <div className="text-xs text-text-secondary">
            累积ACK: <span className="font-mono text-sky-500">{cumulativeAck >= 0 ? cumulativeAck : "无"}</span>
            <span className="text-text-tertiary ml-1">（此序号之前全部确认）</span>
          </div>
          {sacks.length === 0 ? (
            <div className="text-xs text-text-tertiary">无SACK块（无乱序确认）</div>
          ) : sacks.map((s, i) => (
            <div key={i} className="text-xs font-mono text-emerald-600 dark:text-emerald-400">
              SACK块{i + 1}: [{s.left}, {s.right}]（长度 {s.right - s.left + 1}）
            </div>
          ))}
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-medium text-text-primary">SACK工作原理</div>
        <div>• 累积ACK：确认某个序号之前的所有数据</div>
        <div>• SACK块：告知发送方哪些非连续数据已收到</div>
        <div>• 发送方只需重传真正丢失的段，避免Go-Back-N的浪费</div>
        <div>• TCP选项中最多携带3-4个SACK块</div>
      </div>
    </div>
  );
}
export default SACKVisualizer;
