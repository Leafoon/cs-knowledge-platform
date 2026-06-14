"use client";
import { useState, useEffect, useRef } from "react";

interface Packet {
  id: number;
  size: number;
  arrival: number;
  color: string;
}

type Policy = "tail-drop" | "red" | "priority";

export function BufferManagerDemo() {
  const [policy, setPolicy] = useState<Policy>("tail-drop");
  const [bufferSize] = useState(8);
  const [queue, setQueue] = useState<Packet[]>([]);
  const [dropped, setDropped] = useState(0);
  const [processed, setProcessed] = useState(0);
  const [arriving, setArriving] = useState<Packet | null>(null);
  const [nextId, setNextId] = useState(0);
  const [running, setRunning] = useState(false);

  const colors = ["bg-blue-500", "bg-purple-500", "bg-green-500", "bg-orange-500", "bg-pink-500"];

  const generatePacket = () => {
    const pkt: Packet = { id: nextId, size: 1, arrival: Date.now(), color: colors[nextId % colors.length] };
    setNextId((n) => n + 1);
    return pkt;
  };

  const shouldDrop = (q: Packet[]): boolean => {
    if (q.length >= bufferSize) {
      if (policy === "tail-drop") return true;
      if (policy === "red") return Math.random() < (q.length / bufferSize) * 0.8;
      if (policy === "priority") return q.length >= bufferSize;
    }
    return false;
  };

  const step = () => {
    const pkt = generatePacket();
    setArriving(pkt);
    setTimeout(() => {
      setArriving(null);
      setQueue((prev) => {
        if (shouldDrop(prev)) { setDropped((d) => d + 1); return prev; }
        return [...prev, pkt];
      });
    }, 200);
    if (Math.random() < 0.6 && queue.length > 0) {
      setQueue((prev) => prev.slice(1));
      setProcessed((p) => p + 1);
    }
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 400);
    return () => clearInterval(id);
  }, [running, queue, policy, nextId]);

  const utilization = (queue.length / bufferSize) * 100;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">路由器输出端口队列管理</h3>
      <div className="flex gap-2 mb-4">
        {(["tail-drop", "red", "priority"] as Policy[]).map((p) => (
          <button key={p} onClick={() => setPolicy(p)}
            className={`px-3 py-1.5 rounded text-sm ${policy === p ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {p === "tail-drop" ? "尾部丢弃" : p === "red" ? "RED" : "优先级"}
          </button>
        ))}
      </div>
      <div className="mb-4">
        <div className="flex justify-between text-xs text-text-secondary mb-1">
          <span>缓冲区利用率</span>
          <span>{queue.length}/{bufferSize} ({utilization.toFixed(0)}%)</span>
        </div>
        <div className="w-full h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden flex items-center px-1 gap-0.5">
          {Array.from({ length: bufferSize }).map((_, i) => (
            <div key={i} className={`flex-1 h-4 rounded-sm transition-colors ${i < queue.length ? queue[i]?.color || "bg-blue-500" : "bg-gray-300 dark:bg-gray-600"}`} />
          ))}
        </div>
      </div>
      {arriving && (
        <div className="mb-3 flex items-center gap-2">
          <span className="text-xs text-text-secondary">→ 入队</span>
          <div className={`w-8 h-6 rounded ${arriving.color} animate-pulse`} />
          <span className="text-xs text-text-secondary">Packet #{arriving.id}</span>
        </div>
      )}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-3 rounded bg-green-50 dark:bg-green-900/20 text-center">
          <div className="text-xs text-green-600 dark:text-green-400">已转发</div>
          <div className="text-lg font-bold text-green-700 dark:text-green-300">{processed}</div>
        </div>
        <div className="p-3 rounded bg-red-50 dark:bg-red-900/20 text-center">
          <div className="text-xs text-red-600 dark:text-red-400">已丢弃</div>
          <div className="text-lg font-bold text-red-700 dark:text-red-300">{dropped}</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">队列长度</div>
          <div className="text-lg font-bold text-text-primary">{queue.length}</div>
        </div>
      </div>
      <div className="flex gap-2">
        <button onClick={step} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">单步</button>
        <button onClick={() => setRunning(!running)} className={`flex-1 py-2 rounded text-sm ${running ? "bg-red-600 text-white" : "bg-green-600 text-white"}`}>
          {running ? "暂停" : "自动"}
        </button>
        <button onClick={() => { setQueue([]); setDropped(0); setProcessed(0); setNextId(0); }}
          className="px-4 py-2 bg-gray-500 text-white rounded text-sm">重置</button>
      </div>
      <p className="text-xs text-text-secondary mt-3">尾部丢弃：满了直接丢；RED：按概率提前丢弃；优先级：高优先级流量优先转发。</p>
    </div>
  );
}
export default BufferManagerDemo;
