"use client";
import { useState } from "react";

const multiplexTypes = [
  {
    name: "FDM",
    full: "Frequency Division Multiplexing",
    zh: "频分复用",
    desc: "不同信号占用不同频率带宽，同时传输",
    channels: 4,
    resource: "频率",
    color: "bg-blue-600",
    use: "广播电台、ADSL、有线电视",
  },
  {
    name: "TDM",
    full: "Time Division Multiplexing",
    zh: "时分复用",
    desc: "不同信号轮流使用同一频率的不同时间片",
    channels: 4,
    resource: "时间",
    color: "bg-green-600",
    use: "T1/E1线路、GSM",
  },
  {
    name: "WDM",
    full: "Wavelength Division Multiplexing",
    zh: "波分复用",
    desc: "不同信号使用不同光波长，单根光纤传输",
    channels: 4,
    resource: "波长",
    color: "bg-purple-600",
    use: "光纤通信、DWDM海底光缆",
  },
  {
    name: "CDMA",
    full: "Code Division Multiple Access",
    zh: "码分多址",
    desc: "所有用户同时同频传输，用正交码区分",
    channels: 4,
    resource: "码片",
    color: "bg-orange-600",
    use: "3G移动通信、GPS",
  },
];

export function MultiplexingDemuxVisualizer() {
  const [selected, setSelected] = useState(0);
  const [activeChannels, setActiveChannels] = useState([true, true, true, true]);
  const [timeSlot, setTimeSlot] = useState(0);

  const mux = multiplexTypes[selected];

  const toggleChannel = (i: number) => {
    const next = [...activeChannels];
    next[i] = !next[i];
    setActiveChannels(next);
  };

  const colors = ["#3b82f6", "#22c55e", "#a855f7", "#f97316"];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Multiplexing Visualizer <span className="text-text-secondary text-sm">— FDM/TDM/WDM/CDMA</span>
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {multiplexTypes.map((m, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`px-3 py-1 rounded text-sm font-mono ${selected === i ? `${m.color} text-white` : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {m.name}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">{mux.name} — {mux.full}</div>
        <div className="text-sm text-text-secondary mb-2">{mux.zh}: {mux.desc}</div>
        <div className="text-xs text-text-secondary">资源维度: {mux.resource} | 应用: {mux.use}</div>
      </div>
      <div className="flex gap-2 mb-3">
        {activeChannels.map((on, i) => (
          <button
            key={i}
            onClick={() => toggleChannel(i)}
            className={`px-3 py-1 rounded text-sm ${on ? "text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
            style={on ? { backgroundColor: colors[i] } : {}}
          >
            CH{i + 1}
          </button>
        ))}
      </div>
      {selected === 1 && (
        <div className="mb-3">
          <div className="text-xs text-text-secondary mb-1">时间槽: {timeSlot + 1}/4</div>
          <input
            type="range"
            min={0}
            max={3}
            value={timeSlot}
            onChange={(e) => setTimeSlot(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      )}
      <div className="bg-gray-900 p-4 rounded">
        <svg viewBox="0 0 400 120" className="w-full h-28">
          {selected === 0 && activeChannels.map((on, i) => on && (
            <rect key={i} x={i * 100} y={0} width={95} height={120} fill={colors[i]} opacity={0.3} />
          ))}
          {selected === 1 && activeChannels.map((on, i) => on && (
            <rect key={i} x={timeSlot * 100} y={0} width={95} height={120} fill={colors[i]} opacity={timeSlot === i ? 0.8 : 0.15} />
          ))}
          {selected === 2 && activeChannels.map((on, i) => on && (
            <line key={i} x1={0} y1={20 + i * 25} x2={400} y2={20 + i * 25} stroke={colors[i]} strokeWidth={4} opacity={0.8} />
          ))}
          {selected === 3 && activeChannels.map((on, i) => on && (
            <rect key={i} x={0} y={0} width={400} height={120} fill={colors[i]} opacity={0.12 + i * 0.05} />
          ))}
          <text x={200} y={115} textAnchor="middle" fill="#9ca3af" fontSize={10}>
            {selected === 0 ? "频率 →" : selected === 1 ? "时间 →" : selected === 2 ? "波长 →" : "码片序列叠加"}
          </text>
        </svg>
      </div>
    </div>
  );
}

export default MultiplexingDemuxVisualizer;
