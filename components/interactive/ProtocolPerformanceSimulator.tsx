"use client";
import { useState, useMemo } from "react";

interface SimResult {
  name: string;
  throughput: number;
  latency: number;
  efficiency: number;
  retransmissions: number;
}

export function ProtocolPerformanceSimulator() {
  const [lossRate, setLossRate] = useState(2);
  const [rtt, setRtt] = useState(50);
  const [bandwidth, setBandwidth] = useState(100);
  const [frameSize, setFrameSize] = useState(1000);

  const results = useMemo<SimResult[]>(() => {
    const ttx = (frameSize * 8) / (bandwidth * 1e6) * 1000;
    const p = lossRate / 100;
    const protocols = [
      { name: "停等 (Stop-and-Wait)", window: 1, recovery: "重传" },
      { name: "Go-Back-N (W=8)", window: 8, recovery: "回退N" },
      { name: "选择重传 SR (W=8)", window: 8, recovery: "选择重传" },
    ];
    return protocols.map((proto) => {
      const effectiveRTT = proto.window === 1 ? ttx + rtt : ttx + rtt / 2;
      let retransFactor: number;
      if (proto.window === 1) {
        retransFactor = 1 / (1 - p);
      } else if (proto.recovery === "回退N") {
        retransFactor = (1 + p * proto.window / 2);
      } else {
        retransFactor = 1 + p;
      }
      const baseThroughput = (frameSize * 8) / effectiveRTT * proto.window / 1000;
      const throughput = Math.min(baseThroughput / retransFactor, bandwidth);
      const efficiency = throughput / bandwidth;
      const retransmissions = Math.round(p * 100 * retransFactor);
      return { name: proto.name, throughput, latency: effectiveRTT * retransFactor, efficiency, retransmissions };
    });
  }, [lossRate, rtt, bandwidth, frameSize]);

  const maxTP = Math.max(...results.map((r) => r.throughput));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">协议性能模拟器</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          丢包率: <span className="text-text-primary font-mono">{lossRate}%</span>
          <input type="range" min={0} max={20} value={lossRate} onChange={(e) => setLossRate(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          RTT: <span className="text-text-primary font-mono">{rtt}ms</span>
          <input type="range" min={10} max={300} value={rtt} onChange={(e) => setRtt(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          带宽: <span className="text-text-primary font-mono">{bandwidth}Mbps</span>
          <input type="range" min={10} max={1000} value={bandwidth} onChange={(e) => setBandwidth(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          帧大小: <span className="text-text-primary font-mono">{frameSize}B</span>
          <input type="range" min={64} max={4096} step={64} value={frameSize} onChange={(e) => setFrameSize(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="space-y-3">
        {results.map((r, i) => (
          <div key={i} className="rounded-lg border border-border-subtle bg-bg-tertiary p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-text-primary">{r.name}</span>
              <span className="text-xs font-mono text-sky-600 dark:text-sky-400">{r.throughput.toFixed(1)} Mbps</span>
            </div>
            <div className="h-3 bg-bg-elevated rounded-full overflow-hidden mb-2">
              <div className={`h-full rounded-full transition-all ${i === 0 ? "bg-red-400" : i === 1 ? "bg-amber-400" : "bg-emerald-400"}`}
                style={{ width: `${(r.throughput / maxTP) * 100}%` }} />
            </div>
            <div className="flex gap-4 text-[10px] text-text-tertiary">
              <span>延迟: {r.latency.toFixed(1)}ms</span>
              <span>效率: {(r.efficiency * 100).toFixed(1)}%</span>
              <span>重传因子: {r.retransmissions}%</span>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-[10px] text-text-tertiary">丢包率越高，选择重传协议优势越明显；低丢包时差异较小</div>
    </div>
  );
}
export default ProtocolPerformanceSimulator;
