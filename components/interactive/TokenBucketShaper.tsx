"use client";
import { useState } from "react";

export function TokenBucketShaper() {
  const [bucketSize, setBucketSize] = useState(10);
  const [tokenRate, setTokenRate] = useState(3);
  const [tokens, setTokens] = useState(10);
  const [queue, setQueue] = useState(0);
  const [sent, setSent] = useState(0);
  const [dropped, setDropped] = useState(0);
  const [log, setLog] = useState<string[]>([]);

  const addToken = () => {
    setTokens((t) => Math.min(t + tokenRate, bucketSize));
  };

  const sendPacket = () => {
    if (tokens > 0) {
      setTokens((t) => t - 1);
      setSent((s) => s + 1);
      setLog((l) => [...l.slice(-9), `✅ 数据包通过 (剩余令牌: ${tokens - 1})`]);
    } else {
      if (queue < 5) {
        setQueue((q) => q + 1);
        setLog((l) => [...l.slice(-9), `⏳ 数据包排队 (队列: ${queue + 1})`]);
      } else {
        setDropped((d) => d + 1);
        setLog((l) => [...l.slice(-9), `❌ 数据包丢弃 (队列已满)`]);
      }
    }
  };

  const processQueue = () => {
    if (queue > 0 && tokens > 0) {
      setQueue((q) => q - 1);
      setTokens((t) => t - 1);
      setSent((s) => s + 1);
    }
  };

  const reset = () => { setTokens(bucketSize); setQueue(0); setSent(0); setDropped(0); setLog([]); };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">令牌桶整形器 (Token Bucket Shaper)</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-text-muted text-xs block mb-1">桶大小: {bucketSize}</label>
          <input type="range" min="3" max="20" value={bucketSize} onChange={(e) => { setBucketSize(Number(e.target.value)); setTokens(Math.min(tokens, Number(e.target.value))); }}
            className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="text-text-muted text-xs block mb-1">令牌速率: {tokenRate}/tick</label>
          <input type="range" min="1" max="8" value={tokenRate} onChange={(e) => setTokenRate(Number(e.target.value))}
            className="w-full accent-green-500" />
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={addToken} className="px-4 py-2 rounded bg-green-500 text-white text-sm hover:bg-green-600">添加令牌 (tick)</button>
        <button onClick={sendPacket} className="px-4 py-2 rounded bg-blue-500 text-white text-sm hover:bg-blue-600">发送数据包</button>
        <button onClick={processQueue} disabled={queue === 0 || tokens === 0}
          className="px-4 py-2 rounded bg-yellow-500 text-white text-sm disabled:opacity-50 hover:bg-yellow-600">处理队列</button>
        <button onClick={reset} className="px-3 py-2 rounded border border-border-subtle text-text-muted text-sm">重置</button>
      </div>
      <div className="flex items-center gap-4 mb-4">
        <div className="flex-1 p-4 rounded-lg bg-blue-500/10 border border-blue-400/30">
          <span className="text-blue-400 text-xs">令牌桶</span>
          <div className="flex gap-1 mt-2 flex-wrap">
            {Array.from({ length: bucketSize }, (_, i) => (
              <div key={i} className={`w-5 h-5 rounded-full border-2 transition-colors ${
                i < tokens ? "bg-green-500 border-green-400" : "bg-gray-200 dark:bg-gray-700 border-gray-300 dark:border-gray-600"
              }`} />
            ))}
          </div>
          <span className="text-text-muted text-xs mt-1 block">{tokens}/{bucketSize} 令牌</span>
        </div>
        <div className="flex-1 p-4 rounded-lg bg-yellow-500/10 border border-yellow-400/30">
          <span className="text-yellow-400 text-xs">等待队列</span>
          <div className="flex gap-1 mt-2">
            {Array.from({ length: 5 }, (_, i) => (
              <div key={i} className={`w-6 h-4 rounded border transition-colors ${
                i < queue ? "bg-yellow-500 border-yellow-400" : "bg-gray-200 dark:bg-gray-700 border-gray-300 dark:border-gray-600"
              }`} />
            ))}
          </div>
          <span className="text-text-muted text-xs mt-1 block">{queue}/5 排队</span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div className="p-2 rounded bg-green-500/10 border border-green-400/30 text-center">
          <span className="text-green-400 text-lg font-bold">{sent}</span><span className="text-text-muted text-xs block">通过</span>
        </div>
        <div className="p-2 rounded bg-yellow-500/10 border border-yellow-400/30 text-center">
          <span className="text-yellow-400 text-lg font-bold">{queue}</span><span className="text-text-muted text-xs block">排队</span>
        </div>
        <div className="p-2 rounded bg-red-500/10 border border-red-400/30 text-center">
          <span className="text-red-400 text-lg font-bold">{dropped}</span><span className="text-text-muted text-xs block">丢弃</span>
        </div>
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle max-h-32 overflow-y-auto">
        {log.map((l, i) => <p key={i} className="text-text-muted text-xs">{l}</p>)}
        {log.length === 0 && <p className="text-text-muted text-xs">点击按钮开始模拟</p>}
      </div>
    </div>
  );
}
export default TokenBucketShaper;
