"use client";
import { useState } from "react";

interface Segment {
  id: number;
  size: number;
  time: number;
  merged: boolean;
}

export function NagleAlgorithmDemo() {
  const [enabled, setEnabled] = useState(true);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [pendingSize, setPendingSize] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const [sendCount, setSendCount] = useState(0);

  const sendData = (bytes: number) => {
    const now = Date.now();
    if (enabled) {
      const newPending = pendingSize + bytes;
      if (newPending >= 1460 || segments.length === 0) {
        const seg = { id: sendCount, size: newPending, time: now, merged: newPending > bytes };
        setSegments((s) => [...s, seg]);
        setPendingSize(0);
        setSendCount((c) => c + 1);
        setLog((l) => [`发送段 #${seg.id}: ${seg.size} 字节${seg.merged ? " (合并)" : ""}`, ...l].slice(0, 20));
      } else {
        setPendingSize(newPending);
        setLog((l) => [`缓冲 ${bytes} 字节, 累积 ${newPending} 字节, 等待 ACK 或满 MSS`, ...l].slice(0, 20));
      }
    } else {
      const seg = { id: sendCount, size: bytes, time: now, merged: false };
      setSegments((s) => [...s, seg]);
      setSendCount((c) => c + 1);
      setLog((l) => [`立即发送段 #${seg.id}: ${seg.size} 字节`, ...l].slice(0, 20));
    }
  };

  const reset = () => {
    setSegments([]);
    setPendingSize(0);
    setLog([]);
    setSendCount(0);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">Nagle 算法演示 — 小包合并</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setEnabled(!enabled)} className={`px-3 py-1.5 rounded text-sm ${enabled ? "bg-green-500 text-white" : "bg-red-500 text-white"}`}>Nagle: {enabled ? "ON" : "OFF"}</button>
        <button onClick={() => sendData(10)} className="px-3 py-1.5 rounded bg-blue-500 text-white text-sm">发送 10B</button>
        <button onClick={() => sendData(100)} className="px-3 py-1.5 rounded bg-blue-500 text-white text-sm">发送 100B</button>
        <button onClick={() => sendData(1460)} className="px-3 py-1.5 rounded bg-blue-500 text-white text-sm">发送 1460B (MSS)</button>
        <button onClick={reset} className="px-3 py-1.5 rounded bg-gray-200 dark:bg-gray-700 text-text-secondary text-sm">重置</button>
      </div>
      <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle mb-4">
        <p className="text-sm text-text-primary mb-2">已发送 {segments.length} 个段 {pendingSize > 0 && `| 缓冲中 ${pendingSize} 字节`}</p>
        <div className="flex flex-wrap gap-1">
          {segments.map((s) => (
            <div key={s.id} className={`px-2 py-1 rounded text-xs ${s.merged ? "bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300" : "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300"}`}>#{s.id} {s.size}B</div>
          ))}
          {pendingSize > 0 && <div className="px-2 py-1 rounded text-xs bg-gray-200 dark:bg-gray-700 text-text-secondary animate-pulse">等待: {pendingSize}B</div>}
        </div>
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-xs text-text-secondary max-h-28 overflow-y-auto">
        {log.length === 0 ? "点击发送数据观察 Nagle 算法行为" : log.map((l, i) => <div key={i} className="py-0.5">{l}</div>)}
      </div>
      <p className="mt-3 text-xs text-text-secondary">Nagle 规则: 若有未确认数据在飞行中，累积小包直到收到 ACK 或积满 MSS (1460B)。减少网络中小包数量，但增加延迟。</p>
    </div>
  );
}
export default NagleAlgorithmDemo;
