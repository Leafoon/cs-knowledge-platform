"use client";
import { useState } from "react";

export function WindowScaleVisualizer() {
  const [scaleFactor, setScaleFactor] = useState(7);
  const [baseWindow, setBaseWindow] = useState(65535);
  const [rtt, setRtt] = useState(50);
  const [mss, setMss] = useState(1460);

  const actualWindow = baseWindow << scaleFactor;
  const maxThroughputBps = (actualWindow * 8) / (rtt / 1000);
  const noScaleThroughputBps = (65535 * 8) / (rtt / 1000);
  const bdp = maxThroughputBps / 1000000;

  const formatRate = (bps: number) => {
    if (bps >= 1e9) return `${(bps / 1e9).toFixed(1)} Gbps`;
    if (bps >= 1e6) return `${(bps / 1e6).toFixed(1)} Mbps`;
    return `${(bps / 1e3).toFixed(1)} Kbps`;
  };

  const formatSize = (bytes: number) => {
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">窗口缩放可视化 (Window Scale)</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-text-muted text-xs block mb-1">Shift Count (0-14): {scaleFactor}</label>
          <input type="range" min="0" max="14" value={scaleFactor} onChange={(e) => setScaleFactor(Number(e.target.value))}
            className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="text-text-muted text-xs block mb-1">RTT (ms): {rtt}</label>
          <input type="range" min="10" max="300" value={rtt} onChange={(e) => setRtt(Number(e.target.value))}
            className="w-full accent-green-500" />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-400/30">
          <span className="text-blue-400 text-xs block mb-1">无窗口缩放</span>
          <span className="text-text-primary text-2xl font-bold">{formatSize(65535)}</span>
          <span className="text-text-muted text-xs block mt-1">最大窗口: 65,535 bytes</span>
          <span className="text-text-muted text-xs block">吞吐量: {formatRate(noScaleThroughputBps)}</span>
        </div>
        <div className="p-4 rounded-lg bg-green-500/10 border border-green-400/30">
          <span className="text-green-400 text-xs block mb-1">窗口缩放 (shift={scaleFactor})</span>
          <span className="text-text-primary text-2xl font-bold">{formatSize(actualWindow)}</span>
          <span className="text-text-muted text-xs block mt-1">最大窗口: {actualWindow.toLocaleString()} bytes</span>
          <span className="text-text-muted text-xs block">吞吐量: {formatRate(maxThroughputBps)}</span>
        </div>
      </div>
      <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <h4 className="text-text-secondary text-xs font-medium mb-2">窗口大小可视化</h4>
        <div className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div className="absolute h-full bg-blue-500/30 rounded-full" style={{ width: `${(65535 / actualWindow) * 100}%` }} />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-text-primary text-xs font-mono">
              {formatSize(65535)} / {formatSize(actualWindow)} = {((65535 / actualWindow) * 100).toFixed(4)}%
            </span>
          </div>
        </div>
        <p className="text-text-muted text-xs mt-1">无缩放时，窗口仅为最大值的 {((65535 / actualWindow) * 100).toFixed(4)}%</p>
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-xs font-mono space-y-1 border border-border-subtle">
        <div className="text-text-secondary">实际窗口 = {baseWindow.toLocaleString()} × 2^{scaleFactor} = {actualWindow.toLocaleString()} bytes</div>
        <div className="text-text-secondary">最大吞吐 = {formatSize(actualWindow)} × 8 / {rtt}ms = {formatRate(maxThroughputBps)}</div>
        <div className="text-text-secondary">BDP = {formatRate(maxThroughputBps)} × {rtt}ms = {formatSize(bdp * 1000000 / 8)}</div>
        <div className="text-text-secondary">启用: 双方 SYN 中协商 window scale 选项 (RFC 1323)</div>
      </div>
      <div className="grid grid-cols-3 gap-2 mt-3">
        <div className="p-2 rounded bg-bg-primary border border-border-subtle text-center">
          <span className="text-text-muted text-[10px] block">滑动窗口数</span>
          <span className="text-text-primary text-sm font-mono">{Math.floor(actualWindow / mss)}</span>
        </div>
        <div className="p-2 rounded bg-bg-primary border border-border-subtle text-center">
          <span className="text-text-muted text-[10px] block">缩放倍数</span>
          <span className="text-green-400 text-sm font-mono">{1 << scaleFactor}x</span>
        </div>
        <div className="p-2 rounded bg-bg-primary border border-border-subtle text-center">
          <span className="text-text-muted text-[10px] block">最大序列号空间</span>
          <span className="text-text-primary text-sm font-mono">2³²</span>
        </div>
      </div>
      <p className="text-text-muted text-xs mt-3">高延迟网络（如卫星链路 RTT≈600ms）必须使用窗口缩放才能充分利用带宽。TCP 吞吐量 ≈ Window × 8 / RTT。</p>
    </div>
  );
}
export default WindowScaleVisualizer;
