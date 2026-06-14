"use client";
import { useState, useMemo } from "react";

export function SlidingWindowSimulator() {
  const [windowSize, setWindowSize] = useState(4);
  const [bandwidth, setBandwidth] = useState(100);
  const [frameSize, setFrameSize] = useState(1500);
  const [rtt, setRtt] = useState(50);

  const metrics = useMemo(() => {
    const ttx = (frameSize * 8) / (bandwidth * 1e6) * 1000;
    const a = rtt / (2 * ttx);
    const utilization = windowSize >= Math.ceil(2 * a + 1) ? 100 : (windowSize / (2 * a + 1)) * 100;
    const throughput = (bandwidth * utilization) / 100;
    const optimalWindow = Math.ceil(2 * a + 1);
    return { ttx, a, utilization, throughput, optimalWindow };
  }, [windowSize, bandwidth, frameSize, rtt]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">滑动窗口模拟器</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          窗口大小: <span className="text-text-primary font-mono">{windowSize}</span>
          <input type="range" min={1} max={32} value={windowSize} onChange={(e) => setWindowSize(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          带宽: <span className="text-text-primary font-mono">{bandwidth} Mbps</span>
          <input type="range" min={10} max={1000} value={bandwidth} onChange={(e) => setBandwidth(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          帧大小: <span className="text-text-primary font-mono">{frameSize} B</span>
          <input type="range" min={64} max={9000} step={64} value={frameSize} onChange={(e) => setFrameSize(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          RTT: <span className="text-text-primary font-mono">{rtt} ms</span>
          <input type="range" min={1} max={300} value={rtt} onChange={(e) => setRtt(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-text-secondary">链路利用率</span>
          <span className="text-xs font-mono text-text-primary">{metrics.utilization.toFixed(1)}%</span>
        </div>
        <div className="h-4 bg-bg-elevated rounded-full overflow-hidden">
          <div className={`h-full rounded-full transition-all ${metrics.utilization > 90 ? "bg-emerald-500" : metrics.utilization > 50 ? "bg-amber-500" : "bg-red-500"}`}
            style={{ width: `${metrics.utilization}%` }} />
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-sky-500">{metrics.throughput.toFixed(1)}</div>
          <div className="text-[10px] text-text-tertiary">吞吐量 Mbps</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-violet-500">{metrics.ttx.toFixed(2)}</div>
          <div className="text-[10px] text-text-tertiary">发送时延 ms</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-amber-500">{metrics.a.toFixed(2)}</div>
          <div className="text-[10px] text-text-tertiary">a = RTT/Ttx</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-emerald-500">{metrics.optimalWindow}</div>
          <div className="text-[10px] text-text-tertiary">最优窗口</div>
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1 mb-3">
        <div className="font-medium text-text-primary">公式: 利用率 = min(1, W/(2a+1))</div>
        <div>• a = RTT / (2×Ttx)，代表往返时间与发送时延的比值</div>
        <div>• W ≥ 2a+1 时利用率可达100%</div>
        <div>• 当前最优窗口 = {metrics.optimalWindow}，当前窗口 = {windowSize}</div>
        <div>{windowSize >= metrics.optimalWindow ? "✓ 窗口足够大，链路已满载" : `⚠ 窗口偏小，需增大到 ${metrics.optimalWindow} 以满载链路`}</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">协议实现</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• Go-Back-N: 接收窗口=1</li>
            <li>• Selective Repeat: 接收窗口=W</li>
            <li>• TCP: 滑动窗口 + 拥塞控制</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">窗口缩放 (RFC 1323)</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 默认窗口最大 65535B</li>
            <li>• 缩放因子可达 14 (2^14倍)</li>
            <li>• 适应高 BDP 网络 (如 WAN)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default SlidingWindowSimulator;
