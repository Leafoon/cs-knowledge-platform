"use client";
import { useState, useRef, useCallback } from "react";

interface Slot {
  status: "empty" | "success" | "collision";
  node?: number;
}

export function ALOHASimulator() {
  const [mode, setMode] = useState<"pure" | "slotted">("pure");
  const [slots, setSlots] = useState<Slot[]>([]);
  const [stats, setStats] = useState({ total: 0, success: 0, collision: 0 });
  const [nodes, setNodes] = useState(4);
  const [running, setRunning] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const runSlot = useCallback(() => {
    const senders: number[] = [];
    for (let i = 0; i < nodes; i++) {
      if (Math.random() < 0.3) senders.push(i);
    }
    let newSlot: Slot;
    if (senders.length === 0) {
      newSlot = { status: "empty" };
    } else if (senders.length === 1) {
      newSlot = { status: "success", node: senders[0] };
    } else {
      newSlot = { status: "collision" };
    }
    setSlots((prev) => [...prev.slice(-29), newSlot]);
    setStats((prev) => ({
      total: prev.total + 1,
      success: prev.success + (newSlot.status === "success" ? 1 : 0),
      collision: prev.collision + (newSlot.status === "collision" ? 1 : 0),
    }));
  }, [nodes]);

  const start = () => {
    setSlots([]);
    setStats({ total: 0, success: 0, collision: 0 });
    setRunning(true);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(runSlot, mode === "pure" ? 600 : 800);
  };

  const stop = () => {
    setRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const colors: Record<string, string> = { empty: "bg-gray-300 dark:bg-gray-600", success: "bg-green-500", collision: "bg-red-500" };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">ALOHA 协议模拟器</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setMode("pure")} className={`px-3 py-1 rounded text-sm ${mode === "pure" ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>纯 ALOHA</button>
        <button onClick={() => setMode("slotted")} className={`px-3 py-1 rounded text-sm ${mode === "slotted" ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>时隙 ALOHA</button>
        <button onClick={running ? stop : start} className={`ml-auto px-3 py-1 rounded text-sm text-white ${running ? "bg-red-500" : "bg-green-500"}`}>{running ? "停止" : "启动"}</button>
      </div>
      <div className="mb-3">
        <label className="text-sm text-text-secondary">节点数量: {nodes}</label>
        <input type="range" min={2} max={10} value={nodes} onChange={(e) => setNodes(+e.target.value)} className="w-full mt-1" />
      </div>
      <div className="flex gap-1 mb-4 h-12 items-end">
        {slots.map((s, i) => (
          <div key={i} className={`flex-1 rounded-t transition-all ${colors[s.status]}`} style={{ height: s.status === "empty" ? "25%" : s.status === "success" ? "100%" : "75%" }} title={s.status === "success" ? `节点 ${s.node}` : s.status} />
        ))}
      </div>
      <div className="grid grid-cols-4 gap-3 text-center">
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800"><div className="text-xs text-text-secondary">总时隙</div><div className="font-bold text-text-primary">{stats.total}</div></div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800"><div className="text-xs text-text-secondary">成功</div><div className="font-bold text-green-500">{stats.success}</div></div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800"><div className="text-xs text-text-secondary">碰撞</div><div className="font-bold text-red-500">{stats.collision}</div></div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800"><div className="text-xs text-text-secondary">吞吐率</div><div className="font-bold text-text-primary">{stats.total > 0 ? (stats.success / stats.total).toFixed(2) : "—"}</div></div>
      </div>
      <div className="mt-3 flex gap-3 text-xs text-text-secondary">
        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-green-500 rounded inline-block" />成功</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-500 rounded inline-block" />碰撞</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-gray-400 rounded inline-block" />空闲</span>
      </div>
    </div>
  );
}
export default ALOHASimulator;
