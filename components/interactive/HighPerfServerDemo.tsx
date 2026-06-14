"use client";
import { useState } from "react";

export function HighPerfServerDemo() {
  const [mode, setMode] = useState<"reactor" | "proactor">("reactor");
  const [connections, setConnections] = useState(0);
  const [events, setEvents] = useState<string[]>([]);

  const addConnection = () => {
    const id = connections + 1;
    setConnections(id);
    if (mode === "reactor") {
      setEvents([...events, `[Reactor] 新连接 #${id} → 注册到epoll → 等待可读事件 → 读取数据 → 处理`]);
    } else {
      setEvents([...events, `[Proactor] 新连接 #${id} → 提交异步读操作 → 内核完成后回调 → 处理完成`]);
    }
  };

  const reset = () => {
    setConnections(0);
    setEvents([]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">高性能服务器并发模式</h3>
      <div className="flex gap-3 mb-4">
        <button onClick={() => setMode("reactor")}
          className={`px-4 py-2 rounded text-sm ${mode === "reactor" ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
          Reactor模式
        </button>
        <button onClick={() => setMode("proactor")}
          className={`px-4 py-2 rounded text-sm ${mode === "proactor" ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
          Proactor模式
        </button>
      </div>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        {mode === "reactor" ? (
          <div className="space-y-2 text-sm text-text-secondary">
            <div className="flex items-center gap-2"><span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs">1</span> Reactor(事件循环)监听所有fd</div>
            <div className="flex items-center gap-2"><span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs">2</span> I/O就绪时通知应用层</div>
            <div className="flex items-center gap-2"><span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs">3</span> 应用层同步执行I/O操作</div>
            <div className="flex items-center gap-2"><span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs">4</span> 处理业务逻辑</div>
            <div className="text-xs mt-2 p-2 bg-bg-subtle rounded">代表: Nginx, Redis, libevent</div>
          </div>
        ) : (
          <div className="space-y-2 text-sm text-text-secondary">
            <div className="flex items-center gap-2"><span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">1</span> 应用提交异步I/O请求</div>
            <div className="flex items-center gap-2"><span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">2</span> 内核完成I/O后通知应用</div>
            <div className="flex items-center gap-2"><span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">3</span> 回调函数处理结果</div>
            <div className="flex items-center gap-2"><span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">4</span> I/O和计算可并行</div>
            <div className="text-xs mt-2 p-2 bg-bg-subtle rounded">代表: Boost.Asio, Windows IOCP, Linux io_uring</div>
          </div>
        )}
      </div>
      <div className="flex gap-3 mb-4">
        <button onClick={addConnection} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm">模拟新连接</button>
        <button onClick={reset} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">重置</button>
        <span className="text-sm text-text-secondary self-center">连接数: {connections}</span>
      </div>
      {events.length > 0 && (
        <div className="bg-bg-muted rounded-lg p-3 max-h-32 overflow-y-auto text-xs font-mono text-text-secondary">
          {events.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}
      <div className="text-xs text-text-secondary mt-3">
        Reactor: 同步I/O多路复用,事件就绪后应用主动读写。Proactor: 异步I/O,内核完成后通知应用。前者实现简单,后者吞吐量更高。
      </div>
    </div>
  );
}

export default HighPerfServerDemo;
