"use client";
import { useState } from "react";

const antennaTypes = [
  { name: "omnidirectional", label: "全向天线", desc: "水平方向均匀辐射，垂直方向有方向性", gain: "2-5 dBi", pattern: "donut" },
  { name: "directional", label: "定向天线", desc: "能量集中在一个方向，增益高", gain: "10-20 dBi", pattern: "beam" },
  { name: "sector", label: "扇区天线", desc: "覆盖特定角度范围，常用于基站", gain: "15-18 dBi", pattern: "sector" },
];

export function AntennaPatternVisualizer() {
  const [active, setActive] = useState(0);
  const [angle, setAngle] = useState(0);
  const antenna = antennaTypes[active];

  const getRadius = (theta: number) => {
    if (antenna.pattern === "donut") return 60 + 20 * Math.abs(Math.sin(theta * Math.PI / 180));
    if (antenna.pattern === "beam") {
      const diff = Math.abs(theta - angle);
      const norm = Math.min(diff, 360 - diff);
      return 30 + 70 * Math.exp(-((norm / 30) ** 2));
    }
    const diff = Math.abs(theta - angle);
    const norm = Math.min(diff, 360 - diff);
    return norm < 60 ? 30 + 70 * (1 - norm / 60) : 30;
  };

  const points = Array.from({ length: 360 }, (_, i) => {
    const r = getRadius(i);
    const rad = (i * Math.PI) / 180;
    return `${100 + r * Math.cos(rad)},${100 + r * Math.sin(rad)}`;
  }).join(" ");

  const getHalfPowerBW = () => {
    if (antenna.pattern === "donut") return "360° (全向)";
    if (antenna.pattern === "beam") return "60°";
    return "120°";
  };

  const frontToBack = antenna.pattern === "donut" ? "0 dB" : antenna.pattern === "beam" ? "20 dB" : "12 dB";

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">天线辐射方向图可视化</h3>
      <div className="flex gap-2 mb-4">
        {antennaTypes.map((a, i) => (
          <button key={a.name} onClick={() => setActive(i)} className={`px-3 py-1 rounded text-sm ${active === i ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
            {a.label}
          </button>
        ))}
      </div>
      <div className="flex gap-6 items-center">
        <svg viewBox="0 0 200 200" className="w-48 h-48">
          <circle cx="100" cy="100" r="90" fill="none" stroke="currentColor" className="text-gray-200 dark:text-gray-700" strokeWidth="0.5" />
          <circle cx="100" cy="100" r="60" fill="none" stroke="currentColor" className="text-gray-200 dark:text-gray-700" strokeWidth="0.5" />
          <circle cx="100" cy="100" r="30" fill="none" stroke="currentColor" className="text-gray-200 dark:text-gray-700" strokeWidth="0.5" />
          <line x1="100" y1="10" x2="100" y2="190" stroke="currentColor" className="text-gray-200 dark:text-gray-700" strokeWidth="0.5" />
          <line x1="10" y1="100" x2="190" y2="100" stroke="currentColor" className="text-gray-200 dark:text-gray-700" strokeWidth="0.5" />
          <polygon points={points} fill="rgba(59,130,246,0.2)" stroke="rgb(59,130,246)" strokeWidth="1.5" />
          <circle cx="100" cy="100" r="4" fill="rgb(59,130,246)" />
        </svg>
        <div className="flex-1 space-y-3">
          <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 border border-border-subtle">
            <p className="text-sm font-medium text-text-primary">{antenna.label}</p>
            <p className="text-xs text-text-secondary mt-1">{antenna.desc}</p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center"><div className="text-xs text-text-secondary">增益</div><div className="font-bold text-text-primary">{antenna.gain}</div></div>
            <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center"><div className="text-xs text-text-secondary">方向角</div><div className="font-bold text-text-primary">{angle}°</div></div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center"><div className="text-xs text-text-secondary">半功率波束宽度</div><div className="font-bold text-text-primary">{getHalfPowerBW()}</div></div>
            <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center"><div className="text-xs text-text-secondary">前后比</div><div className="font-bold text-text-primary">{frontToBack}</div></div>
          </div>
          {antenna.pattern !== "donut" && (
            <div>
              <label className="text-xs text-text-secondary">主瓣方向: {angle}°</label>
              <input type="range" min={0} max={360} value={angle} onChange={(e) => setAngle(+e.target.value)} className="w-full mt-1" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
export default AntennaPatternVisualizer;
