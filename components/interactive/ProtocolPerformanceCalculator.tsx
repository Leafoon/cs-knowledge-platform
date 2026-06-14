"use client";
import { useState } from "react";

export function ProtocolPerformanceCalculator() {
  const [dataSize, setDataSize] = useState(1500);
  const [headerSize, setHeaderSize] = useState(40);
  const [rtt, setRtt] = useState(50);
  const [bandwidth, setBandwidth] = useState(100);
  const [protocol, setProtocol] = useState<"stopwait" | "gobackn" | "selective">("stopwait");
  const [windowSize, setWindowSize] = useState(8);

  const efficiency = dataSize / (dataSize + headerSize);
  const transmissionTime = ((dataSize + headerSize) * 8) / (bandwidth * 1e6) * 1000;

  let utilization: number;
  let throughput: number;
  let desc: string;

  if (protocol === "stopwait") {
    utilization = (transmissionTime) / (transmissionTime + rtt);
    throughput = (bandwidth * 1e6 * utilization * dataSize) / (dataSize + headerSize) / 1e6;
    desc = "停等协议：每发一帧等待确认，利用率 = Tt/(Tt+RTT)";
  } else if (protocol === "gobackn") {
    const a = rtt / (2 * transmissionTime);
    const w = Math.min(windowSize, Math.pow(2, 4) - 1);
    utilization = a <= w ? 1 : w / (a + 1);
    throughput = bandwidth * utilization;
    desc = `Go-Back-N：窗口W=${w}，利用率取决于min(W, 1+a)，a=${a.toFixed(2)}`;
  } else {
    const a = rtt / (2 * transmissionTime);
    const w = Math.min(Math.floor(windowSize / 2), Math.pow(2, 3));
    utilization = a <= w ? 1 : w / (a + 1);
    throughput = bandwidth * utilization;
    desc = `选择重传：接收窗口W=${w}，利用率取决于min(W, 1+a)，a=${a.toFixed(2)}`;
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">协议性能计算器</h3>
      <div className="flex gap-2 mb-4">
        {(["stopwait", "gobackn", "selective"] as const).map((p) => (
          <button key={p} onClick={() => setProtocol(p)}
            className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${protocol === p ? "bg-sky-500/20 border-sky-400/60 text-sky-700 dark:text-sky-300" : "bg-bg-tertiary border-border-subtle text-text-secondary"}`}>
            {p === "stopwait" ? "停等协议" : p === "gobackn" ? "Go-Back-N" : "选择重传"}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          数据大小: <span className="text-text-primary font-mono">{dataSize}B</span>
          <input type="range" min={64} max={4096} step={64} value={dataSize} onChange={(e) => setDataSize(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          头部大小: <span className="text-text-primary font-mono">{headerSize}B</span>
          <input type="range" min={8} max={128} step={8} value={headerSize} onChange={(e) => setHeaderSize(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          RTT: <span className="text-text-primary font-mono">{rtt}ms</span>
          <input type="range" min={1} max={500} value={rtt} onChange={(e) => setRtt(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          带宽: <span className="text-text-primary font-mono">{bandwidth}Mbps</span>
          <input type="range" min={1} max={1000} value={bandwidth} onChange={(e) => setBandwidth(+e.target.value)} className="w-full mt-1" />
        </label>
        {protocol !== "stopwait" && (
          <label className="text-xs text-text-secondary col-span-2">
            窗口大小: <span className="text-text-primary font-mono">{windowSize}</span>
            <input type="range" min={1} max={64} value={windowSize} onChange={(e) => setWindowSize(+e.target.value)} className="w-full mt-1" />
          </label>
        )}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 space-y-2">
        <div className="text-xs text-text-secondary">{desc}</div>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div>
            <div className="text-lg font-bold text-sky-600 dark:text-sky-400">{(utilization * 100).toFixed(1)}%</div>
            <div className="text-[10px] text-text-tertiary">链路利用率</div>
          </div>
          <div>
            <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{throughput.toFixed(1)}</div>
            <div className="text-[10px] text-text-tertiary">吞吐量 Mbps</div>
          </div>
          <div>
            <div className="text-lg font-bold text-violet-600 dark:text-violet-400">{transmissionTime.toFixed(2)}</div>
            <div className="text-[10px] text-text-tertiary">发送时延 ms</div>
          </div>
        </div>
      </div>
    </div>
  );
}
export default ProtocolPerformanceCalculator;
