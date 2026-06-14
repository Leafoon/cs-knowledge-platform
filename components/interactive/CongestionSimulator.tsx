"use client";
import { useState, useEffect, useRef } from "react";

interface User { id: number; rate: number; color: string; }

export function CongestionSimulator() {
  const [users, setUsers] = useState<User[]>([
    { id: 1, rate: 2, color: "bg-blue-500" },
    { id: 2, rate: 2, color: "bg-green-500" },
    { id: 3, rate: 2, color: "bg-purple-500" },
  ]);
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const capacity = 6;

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(() => setTick((t) => t + 1), 500);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running]);

  const totalDemand = users.reduce((s, u) => s + u.rate, 0);
  const overload = totalDemand > capacity;
  const effectiveRate = overload ? capacity / users.length : 0;

  const updateRate = (id: number, delta: number) => {
    setUsers((prev) => prev.map((u) => u.id === id ? { ...u, rate: Math.max(0.5, Math.min(5, u.rate + delta)) } : u));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">网络拥塞模拟器</h3>
      <div className="mb-4 text-sm text-text-secondary">
        链路容量: <span className="font-mono text-text-primary">{capacity} Mbps</span> |
        总需求: <span className={`font-mono ${overload ? "text-red-500" : "text-green-500"}`}>{totalDemand.toFixed(1)} Mbps</span>
      </div>
      <div className="space-y-3 mb-4">
        {users.map((u) => (
          <div key={u.id} className="flex items-center gap-3">
            <span className="text-sm text-text-secondary w-16">用户 {u.id}</span>
            <button onClick={() => updateRate(u.id, -0.5)} className="px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-sm">-</button>
            <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded-full h-4 overflow-hidden">
              <div className={`h-full ${u.color} rounded-full transition-all duration-300`}
                style={{ width: `${(u.rate / 5) * 100}%` }} />
            </div>
            <button onClick={() => updateRate(u.id, 0.5)} className="px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-sm">+</button>
            <span className="font-mono text-sm text-text-primary w-14 text-right">{u.rate.toFixed(1)} Mbps</span>
          </div>
        ))}
      </div>
      <button onClick={() => setRunning(!running)}
        className={`w-full py-2 rounded font-medium transition-colors ${running ? "bg-red-600 hover:bg-red-700 text-white" : "bg-blue-600 hover:bg-blue-700 text-white"}`}>
        {running ? "停止模拟" : "开始模拟"}
      </button>
      {running && overload && (
        <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-sm text-red-700 dark:text-red-300">
          ⚠️ 拥塞发生！总需求 {totalDemand.toFixed(1)} Mbps 超过链路容量 {capacity} Mbps。
          每用户实际吞吐: {effectiveRate.toFixed(2)} Mbps，丢包率: {((1 - capacity / totalDemand) * 100).toFixed(0)}%
        </div>
      )}
      <div className="mt-3 text-xs text-text-secondary text-center">
        Tick: {tick} | 吞吐量: {overload ? capacity.toFixed(1) : totalDemand.toFixed(1)} Mbps
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">拥塞控制原理</div>
        <div>当总输入速率超过链路容量时，路由器队列溢出导致丢包。</div>
        <div className="mt-1">典型解决方案: TCP拥塞控制(慢启动/拥塞避免/快重传/快恢复)</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">拥塞现象</div>
        <div>• 轻度拥塞: 延迟增大，吞吐量增长变缓</div>
        <div>• 中度拥塞: 大量丢包，TCP重传导致有效吞吐下降</div>
        <div>• 重度拥塞: 网络崩溃(Convective Collapse)，几乎无有效传输</div>
      </div>
    </div>
  );
}
export default CongestionSimulator;
