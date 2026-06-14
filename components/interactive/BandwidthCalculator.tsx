"use client";
import { useState } from "react";

export function BandwidthCalculator() {
  const [mode, setMode] = useState<"nyquist" | "shannon">("nyquist");
  const [bandwidth, setBandwidth] = useState(3000);
  const [levels, setLevels] = useState(2);
  const [snr, setSnr] = useState(20);

  const nyquistRate = 2 * bandwidth * Math.log2(Math.max(levels, 2));
  const snrLinear = Math.pow(10, snr / 10);
  const shannonCapacity = bandwidth * Math.log2(1 + snrLinear);

  const bitsPerSymbol = Math.log2(Math.max(levels, 2));

  const formatRate = (bps: number) => {
    if (bps >= 1e6) return `${(bps / 1e6).toFixed(2)} Mbps`;
    if (bps >= 1e3) return `${(bps / 1e3).toFixed(2)} Kbps`;
    return `${bps.toFixed(2)} bps`;
  };

  const presets = [
    { label: "电话语音", bw: 3400, levels: 2, snr: 30 },
    { label: "ADSL下行", bw: 1104000, levels: 256, snr: 40 },
    { label: "5GHz WiFi", bw: 20000000, levels: 1024, snr: 35 },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">信道容量计算器</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setMode("nyquist")}
          className={`flex-1 py-2 rounded text-sm font-medium ${mode === "nyquist" ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
          Nyquist 奈奎斯特
        </button>
        <button onClick={() => setMode("shannon")}
          className={`flex-1 py-2 rounded text-sm font-medium ${mode === "shannon" ? "bg-purple-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
          Shannon 香农
        </button>
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 mb-4 text-center">
        <p className="text-xs text-text-secondary mb-1">公式</p>
        <p className="font-mono text-sm text-text-primary">
          {mode === "nyquist" ? "C = 2B × log₂(L)" : "C = B × log₂(1 + SNR)"}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs text-text-secondary">带宽 B = {bandwidth.toLocaleString()} Hz</label>
          <input type="range" min={100} max={20000000} step={100} value={bandwidth}
            onChange={(e) => setBandwidth(+e.target.value)} className="w-full mt-1" />
        </div>
        {mode === "nyquist" ? (
          <div>
            <label className="text-xs text-text-secondary">信号电平数 L = {levels}</label>
            <input type="range" min={2} max={1024} step={2} value={levels}
              onChange={(e) => setLevels(+e.target.value)} className="w-full mt-1" />
          </div>
        ) : (
          <div>
            <label className="text-xs text-text-secondary">信噪比 SNR = {snr} dB</label>
            <input type="range" min={0} max={60} step={1} value={snr}
              onChange={(e) => setSnr(+e.target.value)} className="w-full mt-1" />
          </div>
        )}
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-3 rounded bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 text-center">
          <div className="text-xs text-blue-600 dark:text-blue-400">最大比特率</div>
          <div className="text-lg font-bold text-blue-700 dark:text-blue-300">
            {formatRate(mode === "nyquist" ? nyquistRate : shannonCapacity)}
          </div>
        </div>
        {mode === "nyquist" && (
          <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
            <div className="text-xs text-text-secondary">每符号比特数</div>
            <div className="text-lg font-bold text-text-primary">{bitsPerSymbol.toFixed(1)}</div>
          </div>
        )}
        {mode === "shannon" && (
          <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
            <div className="text-xs text-text-secondary">SNR(线性)</div>
            <div className="text-lg font-bold text-text-primary">{snrLinear.toFixed(1)}</div>
          </div>
        )}
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">带宽</div>
          <div className="text-lg font-bold text-text-primary">{formatRate(bandwidth)}</div>
        </div>
      </div>
      <div className="mb-3">
        <p className="text-xs font-medium text-text-primary mb-2">预设场景</p>
        <div className="flex gap-2">
          {presets.map((p) => (
            <button key={p.label} onClick={() => { setBandwidth(p.bw); setLevels(p.levels); setSnr(p.snr); }}
              className="px-2 py-1 rounded text-xs bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700">
              {p.label}
            </button>
          ))}
        </div>
      </div>
      <p className="text-xs text-text-secondary mt-2">
        {mode === "nyquist"
          ? "Nyquist定理：无噪声信道中，最大数据率 = 2B log₂(L)，B为带宽，L为离散信号电平数。"
          : "Shannon定理：有噪声信道容量上限 = B log₂(1+SNR)，SNR为信噪比。实际系统无法超越此极限。"}
      </p>
    </div>
  );
}
export default BandwidthCalculator;
