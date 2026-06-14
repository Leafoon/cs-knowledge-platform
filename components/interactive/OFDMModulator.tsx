"use client";
import { useState } from "react";

export function OFDMModulator() {
  const [numSubcarriers, setNumSubcarriers] = useState(8);
  const [cpRatio, setCpRatio] = useState(25);
  const [view, setView] = useState<"freq" | "time">("freq");

  const subcarrierData = Array.from({ length: numSubcarriers }, (_, i) => ({
    index: i,
    frequency: `f₀ + ${i}×Δf`,
    phase: (i * 45) % 360,
    amplitude: 0.5 + Math.random() * 0.5,
    data: Math.random() > 0.5 ? "1" : "0",
  }));

  const renderFrequencyDomain = () => {
    const w = 360, h = 120;
    const barW = (w - 20) / numSubcarriers;
    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-32">
        {subcarrierData.map((sc, i) => {
          const x = 10 + i * barW;
          const barH = sc.amplitude * (h - 20);
          return (
            <g key={i}>
              <rect x={x + 2} y={h - 10 - barH} width={barW - 4} height={barH}
                fill={sc.data === "1" ? "#3b82f6" : "#6b7280"} rx={2} />
              <text x={x + barW / 2} y={h - 2} textAnchor="middle" fill="#9ca3af" fontSize={8}>
                {i}
              </text>
            </g>
          );
        })}
        <text x={w / 2} y={12} textAnchor="middle" fill="#9ca3af" fontSize={10}>频率域 — 各子载波幅度</text>
      </svg>
    );
  };

  const renderTimeDomain = () => {
    const w = 360, h = 120;
    let pathData = "";
    for (let x = 0; x < w; x++) {
      let y = 0;
      subcarrierData.forEach(sc => {
        y += sc.amplitude * Math.sin((x / w) * Math.PI * 2 * (sc.index + 1) + (sc.phase * Math.PI / 180));
      });
      y = h / 2 - (y / numSubcarriers) * 40;
      pathData += `${x === 0 ? "M" : "L"} ${x} ${y}`;
    }
    const cpLen = (cpRatio / 100) * 60;
    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-32">
        <text x={w / 2} y={12} textAnchor="middle" fill="#9ca3af" fontSize={10}>时域 — OFDM 符号（含循环前缀）</text>
        <line x1={0} y1={h / 2} x2={w} y2={h / 2} stroke="#374151" strokeWidth={0.5} />
        <rect x={0} y={5} width={cpLen} height={h - 10} fill="#f59e0b10" stroke="#f59e0b40" strokeWidth={0.5} />
        <text x={cpLen / 2} y={h - 2} textAnchor="middle" fill="#f59e0b" fontSize={8}>CP</text>
        <path d={pathData} fill="none" stroke="#3b82f6" strokeWidth={1.5} />
      </svg>
    );
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">📶 OFDM 调制器</h3>
      <p className="text-sm text-text-secondary mb-4">展示正交频分复用的 IFFT 调制过程</p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary">子载波数量: {numSubcarriers}</label>
          <input type="range" min={4} max={32} value={numSubcarriers}
            onChange={e => setNumSubcarriers(Number(e.target.value))} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">循环前缀比例: {cpRatio}%</label>
          <input type="range" min={0} max={50} value={cpRatio}
            onChange={e => setCpRatio(Number(e.target.value))} className="w-full accent-yellow-500" />
        </div>
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setView("freq")}
          className={`px-3 py-1.5 rounded text-sm ${view === "freq" ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary"}`}>频率域</button>
        <button onClick={() => setView("time")}
          className={`px-3 py-1.5 rounded text-sm ${view === "time" ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary"}`}>时域</button>
      </div>

      <div className="bg-bg-surface rounded-lg p-3 mb-4 border border-border-subtle">
        {view === "freq" ? renderFrequencyDomain() : renderTimeDomain()}
      </div>

      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "子载波数", value: `${numSubcarriers}` },
          { label: "CP 开销", value: `${cpRatio}%` },
          { label: "频谱效率", value: `${(numSubcarriers * (1 - cpRatio / 100)).toFixed(1)} symbols` },
        ].map(s => (
          <div key={s.label} className="bg-bg-surface rounded-lg p-2 text-center">
            <div className="text-xs text-text-secondary">{s.label}</div>
            <div className="font-mono text-sm font-bold text-text-primary">{s.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
export default OFDMModulator;
