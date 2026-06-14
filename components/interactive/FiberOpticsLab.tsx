"use client";
import { useState } from "react";

export function FiberOpticsLab() {
  const [mode, setMode] = useState<"single" | "multi">("single");
  const [distance, setDistance] = useState(10);
  const [wavelength, setWavelength] = useState(1310);

  const specs = {
    single: {
      name: "单模光纤 (SMF)",
      core: "8-10 μm",
      color: "bg-blue-500",
      distance: "2-100 km",
      bandwidth: "极高 (100+ Gbps)",
      light: "单一模式,直线传播",
      dispersion: "低",
      cost: "较高",
      use: "长距离:城域网、骨干网",
    },
    multi: {
      name: "多模光纤 (MMF)",
      core: "50-62.5 μm",
      color: "bg-green-500",
      distance: "300m-2km",
      bandwidth: "中等 (10 Gbps)",
      light: "多模式,多次反射",
      dispersion: "高 (模间色散)",
      cost: "较低",
      use: "短距离:数据中心、LAN",
    },
  };

  const s = specs[mode];
  const attenuation = mode === "single" ? 0.35 : 3.5;
  const signalLoss = (distance * attenuation).toFixed(1);
  const signalPower = Math.max(0, 100 - parseFloat(signalLoss));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">光纤特性实验室</h3>
      <div className="flex gap-3 mb-4">
        <button onClick={() => setMode("single")}
          className={`px-4 py-2 rounded text-sm ${mode === "single" ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>单模 SMF</button>
        <button onClick={() => setMode("multi")}
          className={`px-4 py-2 rounded text-sm ${mode === "multi" ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>多模 MMF</button>
      </div>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        <div className="flex items-center gap-2 mb-3">
          <span className={`w-3 h-3 rounded-full ${s.color}`} />
          <span className="font-semibold text-text-primary">{s.name}</span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm text-text-secondary">
          <div><strong>纤芯直径:</strong> {s.core}</div>
          <div><strong>传输距离:</strong> {s.distance}</div>
          <div><strong>带宽:</strong> {s.bandwidth}</div>
          <div><strong>色散:</strong> {s.dispersion}</div>
          <div><strong>光源:</strong> {s.light}</div>
          <div><strong>成本:</strong> {s.cost}</div>
          <div className="col-span-2"><strong>典型应用:</strong> {s.use}</div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary">传输距离: {distance} km</label>
          <input type="range" min={1} max={100} value={distance} onChange={(e) => setDistance(Number(e.target.value))}
            className="w-full" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">波长: {wavelength} nm</label>
          <input type="range" min={850} max={1550} step={10} value={wavelength} onChange={(e) => setWavelength(Number(e.target.value))}
            className="w-full" />
        </div>
      </div>
      <div className="bg-bg-muted rounded-lg p-3 mb-4">
        <div className="flex items-center gap-4 text-sm">
          <span className="text-text-secondary">信号衰减: <strong className="text-red-500">{signalLoss} dB</strong></span>
          <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div className={`h-3 rounded-full ${signalPower > 50 ? "bg-green-500" : signalPower > 20 ? "bg-yellow-500" : "bg-red-500"}`}
              style={{ width: `${signalPower}%` }} />
          </div>
          <span className="text-text-secondary">信号强度: {signalPower.toFixed(0)}%</span>
        </div>
      </div>
      <div className="text-xs text-text-secondary">
        光纤衰减系数: SMF约0.35 dB/km (1310nm),MMF约3.5 dB/km。三个低损耗窗口: 850nm/1310nm/1550nm。
      </div>
    </div>
  );
}

export default FiberOpticsLab;
