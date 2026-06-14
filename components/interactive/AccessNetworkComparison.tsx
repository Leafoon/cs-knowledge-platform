"use client";
import { useState } from "react";

const networks = [
  { name: "FTTH", label: "光纤到户", bandwidth: "1 Gbps+", latency: "1 ms", coverage: "受限（需铺设光纤）", cost: "高", reliability: "极高", color: "bg-blue-500" },
  { name: "DSL", label: "数字用户线路", bandwidth: "100 Mbps", latency: "10 ms", coverage: "广泛（利用电话线）", cost: "低", reliability: "高", color: "bg-yellow-500" },
  { name: "4G LTE", label: "第四代移动通信", bandwidth: "150 Mbps", latency: "30 ms", coverage: "广泛", cost: "中", reliability: "高", color: "bg-green-500" },
  { name: "5G", label: "第五代移动通信", bandwidth: "10 Gbps", latency: "1 ms", coverage: "中等（基站密集部署）", cost: "高", reliability: "高", color: "bg-red-500" },
  { name: "WiFi 6", label: "无线局域网", bandwidth: "9.6 Gbps", latency: "5 ms", coverage: "室内短距离", cost: "低", reliability: "中", color: "bg-purple-500" },
];

export function AccessNetworkComparison() {
  const [selected, setSelected] = useState<number[]>([0, 4]);
  const [metric, setMetric] = useState<"bandwidth" | "latency" | "coverage">("bandwidth");

  const toggle = (idx: number) => {
    setSelected((s) => s.includes(idx) ? s.filter((i) => i !== idx) : [...s, idx]);
  };

  const metrics = [
    { key: "bandwidth" as const, label: "带宽" },
    { key: "latency" as const, label: "延迟" },
    { key: "coverage" as const, label: "覆盖范围" },
  ];

  const bandwidthValues = [100, 100, 15, 100, 96];
  const latencyValues = [1, 10, 30, 1, 5];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">接入网络技术对比</h3>
      <div className="flex gap-2 mb-4">
        {networks.map((n, i) => (
          <button key={n.name} onClick={() => toggle(i)} className={`px-3 py-1 rounded text-sm transition-colors ${selected.includes(i) ? `${n.color} text-white` : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
            {n.name}
          </button>
        ))}
      </div>
      <div className="flex gap-2 mb-4">
        {metrics.map((m) => (
          <button key={m.key} onClick={() => setMetric(m.key)} className={`px-3 py-1 rounded text-sm ${metric === m.key ? "bg-indigo-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>{m.label}</button>
        ))}
      </div>
      <div className="space-y-2 mb-4">
        {selected.sort((a, b) => a - b).map((i) => {
          const n = networks[i];
          const val = metric === "bandwidth" ? bandwidthValues[i] : metric === "latency" ? latencyValues[i] : [80, 90, 70, 40, 30][i];
          return (
            <div key={n.name} className="flex items-center gap-3">
              <span className="w-16 text-sm text-text-primary font-medium">{n.name}</span>
              <div className="flex-1 h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div className={`h-full ${n.color} rounded-full transition-all`} style={{ width: `${val}%` }} />
              </div>
              <span className="w-20 text-xs text-text-secondary text-right">{n[metric]}</span>
            </div>
          );
        })}
      </div>
      <div className="grid grid-cols-2 gap-3">
        {selected.sort((a, b) => a - b).map((i) => {
          const n = networks[i];
          return (
            <div key={n.name} className="p-3 rounded bg-gray-50 dark:bg-gray-800 border border-border-subtle">
              <p className="text-sm font-medium text-text-primary">{n.name} — {n.label}</p>
              <div className="mt-2 space-y-1 text-xs text-text-secondary">
                <p>带宽: {n.bandwidth} | 延迟: {n.latency}</p>
                <p>覆盖: {n.coverage}</p>
                <p>成本: {n.cost} | 可靠性: {n.reliability}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
export default AccessNetworkComparison;
