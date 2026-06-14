"use client";
import { useState } from "react";

interface Connection {
  fd: number;
  state: "accepting" | "reading" | "writing" | "idle";
  ip: string;
  data: string;
}

export function ReactorDemo() {
  const [connections, setConnections] = useState<Connection[]>([
    { fd: 4, state: "reading", ip: "192.168.1.10", data: "GET /" },
    { fd: 5, state: "idle", ip: "192.168.1.11", data: "" },
    { fd: 6, state: "writing", ip: "192.168.1.12", data: "HTTP/1.1 200 OK..." },
  ]);
  const [selectedFD, setSelectedFD] = useState<number | null>(null);
  const [eventLog, setEventLog] = useState<string[]>(["[epoll_wait] 等待事件...", "[事件] fd=4 可读", "[事件] fd=6 可写"]);

  const addConnection = () => {
    const fd = Math.max(...connections.map((c) => c.fd), 3) + 1;
    const ip = `192.168.1.${10 + connections.length}`;
    setConnections((prev) => [...prev, { fd, state: "accepting", ip, data: "" }]);
    setEventLog((l) => [...l, `[accept] 新连接 fd=${fd} 来自 ${ip}`]);
    setTimeout(() => {
      setConnections((prev) => prev.map((c) => c.fd === fd ? { ...c, state: "reading" } : c));
      setEventLog((l) => [...l, `[事件] fd=${fd} 可读，开始接收数据`]);
    }, 800);
  };

  const handleEvent = (fd: number, event: string) => {
    setConnections((prev) => prev.map((c) => {
      if (c.fd !== fd) return c;
      if (event === "read") return { ...c, state: "writing", data: "处理中..." };
      if (event === "write") return { ...c, state: "idle", data: "响应已发送" };
      return c;
    }));
    setEventLog((l) => [...l, `[${event}] fd=${fd}`]);
  };

  const stateColors: Record<string, string> = {
    accepting: "bg-violet-500/15 text-violet-600 dark:text-violet-400",
    reading: "bg-sky-500/15 text-sky-600 dark:text-sky-400",
    writing: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
    idle: "bg-gray-500/15 text-gray-500",
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">Reactor模式演示</h3>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={addConnection}
          className="px-4 py-1.5 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-xs font-medium hover:bg-sky-500/25 transition-colors">
          模拟新连接
        </button>
        <span className="text-xs text-text-tertiary">连接数: {connections.length}</span>
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-sky-500">{connections.length}</div>
          <div className="text-[10px] text-text-tertiary">活跃连接</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-emerald-500">{connections.filter((c) => c.state === "reading").length}</div>
          <div className="text-[10px] text-text-tertiary">等待读取</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-amber-500">{connections.filter((c) => c.state === "writing").length}</div>
          <div className="text-[10px] text-text-tertiary">等待写入</div>
        </div>
      </div>
      <div className="space-y-1.5 mb-4">
        {connections.map((c) => (
          <div key={c.fd} onClick={() => setSelectedFD(c.fd)}
            className={`flex items-center gap-3 px-3 py-2 rounded-lg border cursor-pointer transition-all ${selectedFD === c.fd ? "border-sky-400/40 bg-sky-500/10" : "border-border-subtle bg-bg-tertiary hover:border-sky-400/20"}`}>
            <span className="text-xs font-mono text-text-tertiary">fd={c.fd}</span>
            <span className="text-xs text-text-primary">{c.ip}</span>
            <span className={`ml-auto px-2 py-0.5 rounded text-[10px] font-medium ${stateColors[c.state]}`}>{c.state}</span>
            {c.state === "reading" && <button onClick={(e) => { e.stopPropagation(); handleEvent(c.fd, "read"); }} className="px-2 py-0.5 rounded bg-sky-500/20 text-[10px] text-sky-600 dark:text-sky-400">处理</button>}
            {c.state === "writing" && <button onClick={(e) => { e.stopPropagation(); handleEvent(c.fd, "write"); }} className="px-2 py-0.5 rounded bg-amber-500/20 text-[10px] text-amber-600 dark:text-amber-400">发送</button>}
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 max-h-28 overflow-y-auto">
        <div className="text-[10px] font-mono space-y-0.5">
          {eventLog.slice(-8).map((l, i) => (
            <div key={i} className={l.includes("accept") ? "text-violet-400" : l.includes("读") || l.includes("read") ? "text-sky-400" : l.includes("写") || l.includes("write") ? "text-amber-400" : "text-text-tertiary"}>{l}</div>
          ))}
        </div>
      </div>
      <div className="mt-3 text-[10px] text-text-tertiary">Reactor模式：单线程事件循环，通过epoll/kqueue监听IO事件，非阻塞处理多个连接（Nginx/Node.js核心架构）</div>
    </div>
  );
}
export default ReactorDemo;
