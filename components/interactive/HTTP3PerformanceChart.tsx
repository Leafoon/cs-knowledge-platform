"use client";
import { useState } from "react";

interface Metric {
  label: string;
  http1: number;
  http2: number;
  http3: number;
  unit: string;
  lowerBetter: boolean;
}

const metrics: Metric[] = [
  { label: "首次字节时间 (TTFB)", http1: 280, http2: 150, http3: 90, unit: "ms", lowerBetter: true },
  { label: "握手延迟", http1: 300, http2: 300, http3: 100, unit: "ms", lowerBetter: true },
  { label: "队头阻塞影响", http1: 100, http2: 40, http3: 5, unit: "%", lowerBetter: true },
  { label: "并发连接效率", http1: 30, http2: 85, http3: 95, unit: "%", lowerBetter: false },
  { label: "连接迁移支持", http1: 0, http2: 0, http3: 100, unit: "%", lowerBetter: false },
  { label: "0-RTT 恢复", http1: 0, http2: 0, http3: 100, unit: "%", lowerBetter: false },
];

export function HTTP3PerformanceChart() {
  const [selected, setSelected] = useState<number | null>(null);

  const getBarColor = (metric: Metric, value: number) => {
    if (metric.lowerBetter) {
      const min = Math.min(metric.http1, metric.http2, metric.http3);
      return value === min ? "bg-green-500" : value === Math.max(metric.http1, metric.http2, metric.http3) ? "bg-red-400" : "bg-yellow-400";
    }
    const max = Math.max(metric.http1, metric.http2, metric.http3);
    return value === max ? "bg-green-500" : value === 0 ? "bg-red-400" : "bg-yellow-400";
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">HTTP/3 性能对比</h3>
      <div className="flex gap-4 mb-4 text-xs">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-blue-400" />HTTP/1.1</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-purple-400" />HTTP/2</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />HTTP/3</span>
      </div>
      <div className="space-y-3">
        {metrics.map((m, i) => (
          <div key={m.label}
            className={`p-3 rounded-lg cursor-pointer transition-all ${selected === i ? "bg-bg-muted border border-border-subtle" : "hover:bg-bg-subtle"}`}
            onClick={() => setSelected(selected === i ? null : i)}>
            <p className="text-xs text-text-secondary mb-1.5">{m.label} ({m.unit}{m.lowerBetter ? ", 越低越好" : ", 越高越好"})</p>
            {[
              { val: m.http1, color: "bg-blue-400", label: "1.1" },
              { val: m.http2, color: "bg-purple-400", label: "2" },
              { val: m.http3, color: "bg-cyan-400", label: "3" },
            ].map(({ val, color, label }) => {
              const maxVal = Math.max(m.http1, m.http2, m.http3) || 1;
              return (
                <div key={label} className="flex items-center gap-2 mb-0.5">
                  <span className="text-[10px] w-6 text-right text-text-muted">{label}</span>
                  <div className="flex-1 bg-bg-subtle rounded-full h-3 overflow-hidden">
                    <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${(val / maxVal) * 100}%` }} />
                  </div>
                  <span className="text-[10px] w-12 text-right text-text-muted font-mono">{val}{m.unit}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
export default HTTP3PerformanceChart;
