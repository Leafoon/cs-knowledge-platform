"use client";
import { useState } from "react";

interface Conn {
  id: number;
  status: "idle" | "active" | "closing";
  requestCount: number;
}

export function ConnectionPoolDemo() {
  const [pool, setPool] = useState<Conn[]>([]);
  const [maxSize, setMaxSize] = useState(5);
  const [logs, setLogs] = useState<string[]>([]);

  const log = (msg: string) => setLogs((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 15));

  const createConn = () => {
    if (pool.length >= maxSize) { log("池已满，等待回收"); return; }
    const conn: Conn = { id: Date.now(), status: "idle", requestCount: 0 };
    setPool((p) => [...p, conn]);
    log(`创建连接 #${conn.id.toString().slice(-4)} (池大小: ${pool.length + 1})`);
  };

  const useConn = () => {
    const idleIdx = pool.findIndex((c) => c.status === "idle");
    if (idleIdx === -1) { log("无空闲连接，请求排队等待"); return; }
    setPool((p) => p.map((c, i) => i === idleIdx ? { ...c, status: "active", requestCount: c.requestCount + 1 } : c));
    log(`复用连接 #${pool[idleIdx].id.toString().slice(-4)} (累计请求: ${pool[idleIdx].requestCount + 1})`);
    setTimeout(() => {
      setPool((p) => p.map((c) => c.id === pool[idleIdx].id ? { ...c, status: "idle" } : c));
    }, 1500);
  };

  const closeConn = () => {
    const target = pool.find((c) => c.status === "idle");
    if (!target) { log("无可关闭的空闲连接"); return; }
    setPool((p) => p.filter((c) => c.id !== target.id));
    log(`关闭连接 #${target.id.toString().slice(-4)} (回收)`);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">连接池演示</h3>
      <div className="mb-4">
        <label className="text-sm text-text-secondary">池大小上限: {maxSize}</label>
        <input type="range" min={2} max={10} value={maxSize} onChange={(e) => setMaxSize(Number(e.target.value))}
          className="w-full mt-1 accent-blue-500" />
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={createConn} className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm">创建连接</button>
        <button onClick={useConn} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">发起请求(复用)</button>
        <button onClick={closeConn} className="flex-1 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm">关闭连接</button>
      </div>
      <div className="flex gap-2 flex-wrap mb-4 min-h-[40px]">
        {pool.map((c) => (
          <div key={c.id} className={`px-3 py-2 rounded border text-xs font-mono transition-all duration-300 ${c.status === "active" ? "bg-blue-100 dark:bg-blue-900/30 border-blue-400 text-blue-700 dark:text-blue-300" : "bg-gray-50 dark:bg-gray-900 border-border-subtle text-text-secondary"}`}>
            #{c.id.toString().slice(-4)} {c.status === "active" ? "🔄 使用中" : "⏸ 空闲"} ({c.requestCount}次)
          </div>
        ))}
        {pool.length === 0 && <div className="text-text-secondary text-sm">池为空</div>}
      </div>
      <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 max-h-32 overflow-y-auto">
        {logs.map((l, i) => <div key={i} className="text-xs font-mono text-text-secondary py-0.5">{l}</div>)}
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">连接池优势</div>
        <div>• 减少TCP三次握手开销 | 连接复用降低延迟</div>
        <div>• 控制最大并发连接数，防止资源耗尽</div>
        <div>• 自动回收空闲连接，释放系统资源</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">连接池参数</div>
        <div>• minIdle: 最小空闲连接数，保持热连接</div>
        <div>• maxActive: 最大活跃连接数，防止过载</div>
        <div>• maxIdle: 最大空闲连接数，超出则回收</div>
        <div>• timeout: 连接超时时间，避免无限等待</div>
      </div>
    </div>
  );
}
export default ConnectionPoolDemo;
