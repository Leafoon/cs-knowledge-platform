"use client";
import { useState } from "react";

interface FragState {
  id: number;
  offset: number;
  length: number;
  mf: boolean;
  received: boolean;
}

export function IPReassemblyDemo() {
  const [fragments, setFragments] = useState<FragState[]>([
    { id: 1, offset: 0, length: 1480, mf: true, received: false },
    { id: 2, offset: 185, length: 1480, mf: true, received: false },
    { id: 3, offset: 370, length: 1040, mf: false, received: false },
  ]);
  const [timer, setTimer] = useState(0);
  const [timeout, setTimeoutVal] = useState(15);
  const [log, setLog] = useState<string[]>([]);

  const receive = (id: number) => {
    setFragments(fragments.map((f) => f.id === id ? { ...f, received: true } : f));
    setLog([...log, `收到分片 #${id}: offset=${fragments.find((f) => f.id === id)?.offset}, MF=${fragments.find((f) => f.id === id)?.mf ? 1 : 0}`]);
  };

  const tick = () => {
    const newTimer = timer + 1;
    setTimer(newTimer);
    if (newTimer >= timeout) {
      setLog([...log, `⚠ 重组超时! 已等待${newTimer}秒,丢弃所有分片`]);
      setFragments(fragments.map((f) => ({ ...f, received: false })));
      setTimer(0);
    }
  };

  const reset = () => {
    setFragments(fragments.map((f) => ({ ...f, received: false })));
    setTimer(0);
    setLog([]);
  };

  const allReceived = fragments.every((f) => f.received);
  const totalSize = allReceived ? fragments.reduce((s, f) => s + f.length, 0) : 0;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">IP分片重组 + 超时机制</h3>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-sm text-text-secondary">重组缓冲区:</span>
          <span className="text-sm font-mono text-text-primary">定时器: {timer}s / {timeout}s</span>
        </div>
        <div className="flex gap-1 h-8">
          {fragments.map((f) => {
            const width = (f.length / 4000) * 100;
            return (
              <button key={f.id} onClick={() => receive(f.id)}
                className={`h-full rounded transition-all ${f.received ? "bg-green-500" : "bg-gray-300 dark:bg-gray-700 hover:bg-gray-400"}`}
                style={{ width: `${width}%` }}
                title={`分片 #${f.id}: offset=${f.offset}, len=${f.length}, MF=${f.mf ? 1 : 0}`}>
                <span className="text-xs text-white">{f.id}</span>
              </button>
            );
          })}
        </div>
        {allReceived && (
          <div className="mt-3 p-2 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded text-sm">
            ✓ 所有分片已接收,重组完成! 总大小: {totalSize}B
          </div>
        )}
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {fragments.map((f) => (
          <div key={f.id} className={`p-3 rounded-lg text-sm ${f.received ? "bg-green-100 dark:bg-green-900/30" : "bg-bg-muted"}`}>
            <div className="font-bold text-text-primary">分片 #{f.id}</div>
            <div className="text-xs text-text-secondary">偏移: {f.offset} ({f.offset * 8}B)</div>
            <div className="text-xs text-text-secondary">长度: {f.length}B</div>
            <div className="text-xs text-text-secondary">MF: {f.mf ? 1 : 0}</div>
            <div className={`text-xs mt-1 ${f.received ? "text-green-500" : "text-yellow-500"}`}>
              {f.received ? "✓ 已收到" : "⏳ 等待中"}
            </div>
          </div>
        ))}
      </div>
      <div className="flex gap-3 mb-4">
        {fragments.filter((f) => !f.received).map((f) => (
          <button key={f.id} onClick={() => receive(f.id)}
            className="px-3 py-1.5 bg-blue-500 text-white rounded text-sm hover:bg-blue-600">
            接收 #{f.id}
          </button>
        ))}
        <button onClick={tick} className="px-3 py-1.5 bg-yellow-500 text-white rounded text-sm hover:bg-yellow-600">计时+1s</button>
        <button onClick={reset} className="px-3 py-1.5 bg-bg-subtle text-text-secondary rounded text-sm hover:bg-bg-muted">重置</button>
      </div>
      {log.length > 0 && (
        <div className="bg-bg-muted rounded-lg p-3 max-h-24 overflow-y-auto text-xs font-mono text-text-secondary">
          {log.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      )}
      <div className="text-xs text-text-secondary mt-3">
        重组:目的主机收到分片后,根据ID标识同一数据报,按Offset排列。所有MF=0的分片到达且无空洞时重组完成。超时(通常15-30秒)后丢弃不完整数据报。
      </div>
    </div>
  );
}

export default IPReassemblyDemo;
