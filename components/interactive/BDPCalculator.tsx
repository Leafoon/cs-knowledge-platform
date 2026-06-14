"use client";
import { useState } from "react";

export function BDPCalculator() {
  const [bandwidth, setBandwidth] = useState(100);
  const [unit, setUnit] = useState<"Mbps" | "Gbps">("Mbps");
  const [rtt, setRTT] = useState(50);

  const bps = unit === "Mbps" ? bandwidth * 1e6 : bandwidth * 1e9;
  const rttSec = rtt / 1000;
  const bdp = bps * rttSec;
  const bdpBytes = bdp / 8;
  const bdpKB = bdpBytes / 1024;
  const bdpMB = bdpKB / 1024;

  const formatBDP = () => {
    if (bdpMB >= 1) return `${bdpMB.toFixed(1)} MB`;
    if (bdpKB >= 1) return `${bdpKB.toFixed(1)} KB`;
    return `${bdpBytes.toFixed(0)} Bytes`;
  };

  const windowNeeded = bdpMB >= 1 ? `${bdpMB.toFixed(1)} MB` : `${bdpKB.toFixed(0)} KB`;

  const examples = [
    { bw: "100 Mbps", rtt: "50 ms", bdp: "625 KB" },
    { bw: "1 Gbps", rtt: "100 ms", bdp: "12.5 MB" },
    { bw: "10 Gbps", rtt: "1 ms", bdp: "1.25 MB" },
    { bw: "100 Mbps", rtt: "300 ms", bdp: "3.75 MB" },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">带宽延迟积（BDP）计算器</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div>
          <label className="text-xs text-text-secondary">带宽</label>
          <div className="flex mt-1">
            <input type="number" value={bandwidth} onChange={(e) => setBandwidth(+e.target.value)} className="flex-1 px-2 py-1.5 rounded-l border border-border-subtle bg-gray-50 dark:bg-gray-800 text-text-primary text-sm" />
            <button onClick={() => setUnit(unit === "Mbps" ? "Gbps" : "Mbps")} className="px-2 py-1.5 rounded-r border-t border-b border-r border-border-subtle bg-gray-200 dark:bg-gray-700 text-xs text-text-secondary">{unit}</button>
          </div>
        </div>
        <div>
          <label className="text-xs text-text-secondary">RTT: {rtt} ms</label>
          <input type="range" min={1} max={500} value={rtt} onChange={(e) => setRTT(+e.target.value)} className="w-full mt-2" />
        </div>
        <div className="flex items-end">
          <div className="p-2 rounded bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 w-full text-center">
            <div className="text-xs text-blue-600 dark:text-blue-400">BDP</div>
            <div className="text-lg font-bold text-blue-700 dark:text-blue-300">{formatBDP()}</div>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">所需窗口大小</div>
          <div className="font-bold text-text-primary">{windowNeeded}</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">链路中最大数据量</div>
          <div className="font-bold text-text-primary">{formatBDP()}</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">传输速率</div>
          <div className="font-bold text-text-primary">{bandwidth} {unit}</div>
        </div>
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <p className="text-xs font-medium text-text-primary mb-2">常见场景参考</p>
        <div className="grid grid-cols-4 gap-2 text-xs text-text-secondary">
          {examples.map((e, i) => (
            <div key={i} className="p-2 rounded bg-white dark:bg-gray-900 text-center">
              <div>{e.bw}</div><div>{e.rtt}</div><div className="font-bold text-text-primary">{e.bdp}</div>
            </div>
          ))}
        </div>
      </div>
      <p className="text-xs text-text-secondary mt-3">BDP = 带宽 × RTT，表示在任何时刻链路上能容纳的最大数据量。TCP 窗口大小应 ≥ BDP 才能充分利用链路。</p>
    </div>
  );
}
export default BDPCalculator;
