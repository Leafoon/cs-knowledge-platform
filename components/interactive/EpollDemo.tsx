"use client";
import { useState } from "react";

interface FdEvent {
  fd: number;
  type: "accept" | "read" | "write" | "close";
  status: "waiting" | "ready";
}

export function EpollDemo() {
  const [events, setEvents] = useState<FdEvent[]>([]);
  const [tree, setTree] = useState<number[]>([]);
  const [readyList, setReadyList] = useState<number[]>([]);
  const [mode, setMode] = useState<"lt" | "et">("lt");
  const [nextFd, setNextFd] = useState(1);

  const addFd = () => {
    const fd = nextFd;
    setNextFd(fd + 1);
    setTree([...tree, fd]);
    setEvents([...events, { fd, type: "read", status: "waiting" }]);
  };

  const triggerReady = (fd: number) => {
    setReadyList([...readyList, fd]);
    setEvents(events.map((e) => e.fd === fd ? { ...e, status: "ready" } : e));
  };

  const consume = (fd: number) => {
    setReadyList(readyList.filter((f) => f !== fd));
    if (mode === "et") {
      setEvents(events.map((e) => e.fd === fd ? { ...e, status: "waiting" } : e));
    }
  };

  const removeFd = (fd: number) => {
    setTree(tree.filter((f) => f !== fd));
    setReadyList(readyList.filter((f) => f !== fd));
    setEvents(events.filter((e) => e.fd !== fd));
  };

  const reset = () => {
    setEvents([]);
    setTree([]);
    setReadyList([]);
    setNextFd(1);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">epoll: 红黑树 + 就绪链表</h3>
      <div className="flex gap-3 mb-4">
        <button onClick={() => setMode("lt")}
          className={`px-3 py-1.5 rounded text-sm ${mode === "lt" ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
          LT (水平触发)
        </button>
        <button onClick={() => setMode("et")}
          className={`px-3 py-1.5 rounded text-sm ${mode === "et" ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
          ET (边缘触发)
        </button>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-bg-muted rounded-lg p-3">
          <div className="text-sm font-semibold text-text-primary mb-2">红黑树 (所有监控fd)</div>
          <div className="flex flex-wrap gap-1.5">
            {tree.length === 0 && <span className="text-xs text-text-secondary">空</span>}
            {tree.map((fd) => (
              <div key={fd} className="flex items-center gap-1">
                <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded text-xs font-mono">fd{fd}</span>
                <button onClick={() => triggerReady(fd)} className="text-xs text-blue-500 hover:underline">触发</button>
                <button onClick={() => removeFd(fd)} className="text-xs text-red-500 hover:underline">移除</button>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-bg-muted rounded-lg p-3">
          <div className="text-sm font-semibold text-text-primary mb-2">就绪链表</div>
          <div className="flex flex-wrap gap-1.5">
            {readyList.length === 0 && <span className="text-xs text-text-secondary">空</span>}
            {readyList.map((fd) => (
              <button key={fd} onClick={() => consume(fd)}
                className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded text-xs font-mono hover:bg-green-200">
                fd{fd} (点击消费)
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="flex gap-3 mb-4">
        <button onClick={addFd} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm">添加fd</button>
        <button onClick={reset} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">重置</button>
      </div>
      <div className="text-xs text-text-secondary">
        epoll使用红黑树管理所有监控的fd,O(log n)增删。就绪事件通过链表返回给应用。
        LT模式:未消费会持续通知;ET模式:仅通知一次,必须一次性读完。
      </div>
    </div>
  );
}

export default EpollDemo;
