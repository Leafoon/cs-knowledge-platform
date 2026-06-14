"use client";
import { useState } from "react";

export function PerformanceMetrics() {
  const [bandwidth, setBandwidth] = useState(75);
  const [rtt, setRtt] = useState(30);
  const [jitter, setJitter] = useState(5);
  const [loss, setLoss] = useState(0.5);
  const [throughput, setThroughput] = useState(85);

  const getQuality = () => {
    let score = 100;
    if (rtt > 100) score -= 30; else if (rtt > 50) score -= 15;
    if (jitter > 30) score -= 20; else if (jitter > 10) score -= 10;
    if (loss > 2) score -= 30; else if (loss > 0.5) score -= 10;
    return Math.max(0, score);
  };

  const quality = getQuality();
  const qualityLabel = quality >= 80 ? "优秀" : quality >= 60 ? "良好" : quality >= 40 ? "可接受" : "差";
  const qualityColor = quality >= 80 ? "text-green-400" : quality >= 60 ? "text-blue-400" : quality >= 40 ? "text-yellow-400" : "text-red-400";

  const gauges = [
    { label: "带宽利用率", en: "Bandwidth", value: bandwidth, unit: "%", color: "bg-blue-500", warn: 80 },
    { label: "往返延迟", en: "RTT", value: rtt, unit: "ms", color: "bg-green-500", warn: 100 },
    { label: "抖动", en: "Jitter", value: jitter, unit: "ms", color: "bg-yellow-500", warn: 30 },
    { label: "丢包率", en: "Packet Loss", value: loss, unit: "%", color: "bg-red-500", warn: 2 },
    { label: "吞吐量", en: "Throughput", value: throughput, unit: "%", color: "bg-purple-500", warn: 70 },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">📊 性能指标仪表盘</h3>
      <p className="text-sm text-text-secondary mb-4">展示带宽/延迟/抖动/丢包率等核心指标</p>

      <div className="flex items-center justify-center mb-4">
        <div className="text-center">
          <div className={`text-5xl font-mono font-bold ${qualityColor}`}>{quality}</div>
          <div className="text-sm text-text-secondary">QoS 评分</div>
          <div className={`text-lg font-medium ${qualityColor}`}>{qualityLabel}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 mb-4">
        {gauges.map(g => (
          <div key={g.en} className="bg-bg-surface rounded-lg p-3 text-center">
            <div className="text-xs text-text-secondary mb-1">{g.label}</div>
            <div className={`text-xl font-mono font-bold ${g.value > g.warn ? "text-red-400" : "text-text-primary"}`}>
              {g.value}{g.unit}
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1.5 mt-2">
              <div className={`${g.color} h-1.5 rounded-full transition-all`}
                style={{ width: `${Math.min((g.unit === "%" ? g.value : g.value / 200 * 100), 100)}%` }} />
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-text-secondary">带宽利用率: {bandwidth}%</label>
          <input type="range" min={0} max={100} value={bandwidth} onChange={e => setBandwidth(Number(e.target.value))} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">RTT: {rtt}ms</label>
          <input type="range" min={1} max={500} value={rtt} onChange={e => setRtt(Number(e.target.value))} className="w-full accent-green-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">抖动: {jitter}ms</label>
          <input type="range" min={0} max={100} value={jitter} onChange={e => setJitter(Number(e.target.value))} className="w-full accent-yellow-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">丢包率: {loss}%</label>
          <input type="range" min={0} max={10} step={0.1} value={loss} onChange={e => setLoss(Number(e.target.value))} className="w-full accent-red-500" />
        </div>
      </div>
    </div>
  );
}
export default PerformanceMetrics;
