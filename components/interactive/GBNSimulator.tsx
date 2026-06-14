"use client";
import { useState } from "react";

const TOTAL_FRAMES = 8;
const WINDOW_SIZE = 4;

export function GBNSimulator() {
  const [base, setBase] = useState(0);
  const [nextSeq, setNextSeq] = useState(0);
  const [acked, setAcked] = useState<boolean[]>(Array(TOTAL_FRAMES).fill(false));
  const [lost, setLost] = useState<Set<number>>(new Set());
  const [log, setLog] = useState<string[]>([]);

  const send = () => {
    if (nextSeq >= TOTAL_FRAMES || nextSeq >= base + WINDOW_SIZE) return;
    const seq = nextSeq;
    setNextSeq((s) => s + 1);
    setLog((l) => [`帧 ${seq} 已发送`, ...l]);
  };

  const ack = (seq: number) => {
    const newAcked = [...acked];
    newAcked[seq] = true;
    setAcked(newAcked);
    let newBase = base;
    while (newBase < TOTAL_FRAMES && newAcked[newBase]) newBase++;
    setBase(newBase);
    setLog((l) => [`确认帧 ${seq}，窗口滑动至 ${newBase}`, ...l]);
  };

  const timeout = () => {
    setLog((l) => [`超时! 从帧 ${base} 开始重传`, ...l]);
    setNextSeq(base);
  };

  const toggleLost = (seq: number) => {
    const s = new Set(lost);
    s.has(seq) ? s.delete(seq) : s.add(seq);
    setLost(s);
  };

  const reset = () => {
    setBase(0); setNextSeq(0);
    setAcked(Array(TOTAL_FRAMES).fill(false));
    setLost(new Set()); setLog([]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">GBN 协议模拟器</h3>
      <p className="text-sm text-text-secondary mb-4">后退N帧协议: 窗口大小={WINDOW_SIZE}，点击帧标记丢失</p>
      <div className="flex gap-1 mb-4">
        {Array.from({ length: TOTAL_FRAMES }, (_, i) => {
          const inWindow = i >= base && i < base + WINDOW_SIZE;
          const isSent = i < nextSeq;
          const isLost = lost.has(i);
          return (
            <button
              key={i}
              onClick={() => toggleLost(i)}
              className={`flex-1 py-3 rounded text-sm font-mono border transition-all ${
                acked[i]
                  ? "bg-green-500/20 border-green-500 text-green-400"
                  : isLost
                    ? "bg-red-500/20 border-red-500 text-red-400"
                    : isSent
                      ? "bg-blue-500/20 border-blue-500 text-blue-400"
                      : inWindow
                        ? "bg-bg-muted border-border-subtle text-text-primary"
                        : "bg-bg-subtle border-transparent text-text-muted"
              }`}
            >
              {i}
              {isLost && " ✗"}
            </button>
          );
        })}
      </div>
      <p className="text-xs text-text-muted mb-3">base={base}  nextSeq={nextSeq}  窗口=[{base},{Math.min(base + WINDOW_SIZE - 1, TOTAL_FRAMES - 1)}]</p>
      <div className="flex gap-2 mb-4">
        <button onClick={send} disabled={nextSeq >= TOTAL_FRAMES || nextSeq >= base + WINDOW_SIZE}
          className="px-3 py-1.5 rounded bg-blue-500 text-white text-sm disabled:opacity-40">发送帧</button>
        <button onClick={() => ack(base)} disabled={!acked.includes(false) || base >= TOTAL_FRAMES}
          className="px-3 py-1.5 rounded bg-green-500 text-white text-sm disabled:opacity-40">确认 ACK</button>
        <button onClick={timeout}
          className="px-3 py-1.5 rounded bg-yellow-500 text-white text-sm">模拟超时</button>
        <button onClick={reset}
          className="px-3 py-1.5 rounded bg-gray-500 text-white text-sm">重置</button>
      </div>
      <div className="h-32 overflow-y-auto bg-bg-muted rounded p-2 text-xs font-mono">
        {log.length === 0 && <p className="text-text-muted">暂无事件...</p>}
        {log.map((l, i) => <p key={i} className="text-text-secondary">{l}</p>)}
      </div>
    </div>
  );
}
export default GBNSimulator;
