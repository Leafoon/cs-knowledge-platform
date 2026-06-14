"use client";
import { useState } from "react";

const profiles = [
  { name: "温湿度传感器", tx: 0.05, rx: 0.01, idle: 0.001, interval: 300, duty: 0.02, cap: 2400 },
  { name: "智能门锁", tx: 0.1, rx: 0.05, idle: 0.005, interval: 60, duty: 0.05, cap: 1200 },
  { name: "视频监控", tx: 2.0, rx: 0.1, idle: 0.05, interval: 1, duty: 0.8, cap: 5000 },
  { name: "智能灯泡", tx: 0.02, rx: 0.02, idle: 0.0005, interval: 10, duty: 0.01, cap: 800 },
];

export function IoTBatteryCalculator() {
  const [selected, setSelected] = useState(0);
  const [customTx, setCustomTx] = useState(0.05);
  const [customInterval, setCustomInterval] = useState(300);

  const p = profiles[selected];
  const tx = selected === 0 ? customTx : p.tx;
  const interval = selected === 0 ? customInterval : p.interval;
  const avgCurrent = (tx * p.duty + p.idle * (1 - p.duty)) * 1000;
  const lifeDays = p.cap / (avgCurrent / 1000) / 24;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        IoT Battery Calculator <span className="text-text-secondary text-sm">— 设备续航计算</span>
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {profiles.map((p, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`px-3 py-1 rounded text-sm ${selected === i ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {p.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded">
          <div className="text-xs text-text-secondary">发送电流</div>
          <div className="text-lg font-bold text-text-primary">{tx} A</div>
          {selected === 0 && (
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={customTx}
              onChange={(e) => setCustomTx(parseFloat(e.target.value))}
              className="w-full mt-1"
            />
          )}
        </div>
        <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded">
          <div className="text-xs text-text-secondary">空闲电流</div>
          <div className="text-lg font-bold text-text-primary">{p.idle} A</div>
        </div>
        <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded">
          <div className="text-xs text-text-secondary">上报间隔</div>
          <div className="text-lg font-bold text-text-primary">{interval} s</div>
          {selected === 0 && (
            <input
              type="range"
              min="10"
              max="3600"
              step="10"
              value={customInterval}
              onChange={(e) => setCustomInterval(parseInt(e.target.value))}
              className="w-full mt-1"
            />
          )}
        </div>
        <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded">
          <div className="text-xs text-text-secondary">电池容量</div>
          <div className="text-lg font-bold text-text-primary">{p.cap} mAh</div>
        </div>
      </div>
      <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded text-center">
        <div className="text-sm text-text-secondary">平均电流</div>
        <div className="text-2xl font-bold text-green-700 dark:text-green-300">
          {avgCurrent.toFixed(2)} mA
        </div>
        <div className="text-sm text-text-secondary mt-1">预估续航</div>
        <div className="text-2xl font-bold text-green-700 dark:text-green-300">
          {lifeDays > 365 ? `${(lifeDays / 365).toFixed(1)} 年` : `${Math.round(lifeDays)} 天`}
        </div>
      </div>
    </div>
  );
}

export default IoTBatteryCalculator;
