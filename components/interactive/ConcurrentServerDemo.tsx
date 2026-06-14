"use client";
import { useState } from "react";

const models = [
  { id: "fork", name: "fork 多进程", desc: "每连接fork一个子进程", concurrency: 300, latency: "高", resource: "高", complexity: "低" },
  { id: "thread", name: "多线程", desc: "每连接创建一个线程", concurrency: 1000, latency: "中", resource: "中", complexity: "中" },
  { id: "select", name: "select/poll", desc: "IO多路复用(线性扫描)", concurrency: 1000, latency: "中", resource: "低", complexity: "中" },
  { id: "epoll", name: "epoll", desc: "IO多路复用(事件驱动)", concurrency: 100000, latency: "低", resource: "低", complexity: "高" },
];

export function ConcurrentServerDemo() {
  const [active, setActive] = useState("fork");
  const [connections, setConnections] = useState(10);

  const current = models.find((m) => m.id === active)!;
  const loadPercent = Math.min(100, (connections / current.concurrency) * 100);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">并发服务器模型对比</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {models.map((m) => (
          <button key={m.id} onClick={() => setActive(m.id)}
            className={`px-3 py-1.5 rounded text-sm font-mono transition-colors ${active === m.id ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700"}`}>
            {m.name}
          </button>
        ))}
      </div>
      <div className="mb-4">
        <label className="text-sm text-text-secondary">并发连接数: {connections}</label>
        <input type="range" min={1} max={100000} value={connections} onChange={(e) => setConnections(Number(e.target.value))}
          className="w-full mt-1 accent-blue-500" />
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
          <div className="text-text-secondary mb-1">模型描述</div>
          <div className="text-text-primary font-medium">{current.desc}</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
          <div className="text-text-secondary mb-1">理论上限</div>
          <div className="text-text-primary font-mono">{current.concurrency.toLocaleString()} 连接</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
          <div className="text-text-secondary mb-1">延迟/资源</div>
          <div className="text-text-primary">{current.latency} / {current.resource}</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
          <div className="text-text-secondary mb-1">实现复杂度</div>
          <div className="text-text-primary">{current.complexity}</div>
        </div>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 rounded-full h-6 overflow-hidden">
        <div className={`h-full rounded-full flex items-center justify-center text-xs text-white font-mono transition-all duration-500 ${loadPercent > 80 ? "bg-red-500" : loadPercent > 50 ? "bg-yellow-500" : "bg-green-500"}`}
          style={{ width: `${loadPercent}%` }}>
          {loadPercent.toFixed(0)}%
        </div>
      </div>
      <div className="text-xs text-text-secondary mt-2 text-center">
        负载: {connections.toLocaleString()} / {current.concurrency.toLocaleString()}
        {loadPercent > 80 && " ⚠️ 接近上限，性能将显著下降"}
      </div>
      <div className="mt-4 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">关键区别</div>
        <div className="space-y-1">
          <div>• fork: 每个连接独立进程，简单但开销大，C10K瓶颈</div>
          <div>• 线程: 共享内存空间，比fork轻量，但上下文切换仍有开销</div>
          <div>• select/poll: 单线程监控多FD，但O(n)扫描开销线性增长</div>
          <div>• epoll: 事件驱动+红黑树+就绪链表，O(1)通知，支持百万连接</div>
        </div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">C10K问题</div>
        <div>当并发连接数达到10K级别时:</div>
        <div>• 进程/线程模型: 内存消耗巨大(每进程约8MB栈空间)</div>
        <div>• select: FD_SET最大1024，O(n)扫描成为瓶颈</div>
        <div>• poll: 解除FD数量限制，但仍是O(n)线性扫描</div>
        <div>• epoll/kqueue: 事件通知O(1)，真正解决C10K问题</div>
      </div>
    </div>
  );
}
export default ConcurrentServerDemo;
