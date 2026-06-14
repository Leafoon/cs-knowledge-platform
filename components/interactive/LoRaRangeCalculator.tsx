"use client";
import { useState } from "react";

const sfOptions = [7, 8, 9, 10, 11, 12];
const bwOptions = [125, 250, 500];
const envOptions = [
  { label: "城市密集", en: "Urban Dense", loss: 30 },
  { label: "城市", en: "Urban", loss: 25 },
  { label: "郊区", en: "Suburban", loss: 20 },
  { label: "乡村", en: "Rural", loss: 15 },
];

export function LoRaRangeCalculator() {
  const [sf, setSf] = useState(7);
  const [bw, setBw] = useState(125);
  const [txPower, setTxPower] = useState(14);
  const [envIdx, setEnvIdx] = useState(2);

  const sensitivity: Record<number, number> = { 7: -123, 8: -126, 9: -129, 10: -132, 11: -134.5, 12: -137 };
  const linkBudget = txPower - sensitivity[sf];
  const maxPathLoss = linkBudget - 2 - envOptions[envIdx].loss;
  const pathLossExp = 2.7 + envOptions[envIdx].loss / 30;
  const distanceKm = Math.pow(10, (maxPathLoss - 32.44 - 20 * Math.log10(868)) / (10 * pathLossExp));
  const clamped = Math.min(Math.max(distanceKm, 0.1), 30);
  const dr = Math.round(bw * (4 / Math.pow(2, sf)) / 1000 * 100) / 100;
  const airtime = Math.ceil((255 * 8) / (bw * sf / Math.pow(2, sf)) * 1000);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">📡 LoRa 距离计算器</h3>
      <p className="text-sm text-text-secondary mb-4">根据扩频因子/带宽/功率计算通信距离</p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm font-medium text-text-secondary">扩频因子 (Spreading Factor)</label>
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {sfOptions.map(s => (
              <button key={s} onClick={() => setSf(s)}
                className={`w-10 h-10 rounded text-sm font-mono ${sf === s ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
                SF{s}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-sm font-medium text-text-secondary">带宽 (Bandwidth)</label>
          <div className="flex gap-2 mt-2">
            {bwOptions.map(b => (
              <button key={b} onClick={() => setBw(b)}
                className={`px-3 py-2 rounded text-sm font-mono ${bw === b ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
                {b} kHz
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-sm font-medium text-text-secondary">发射功率: {txPower} dBm</label>
          <input type="range" min={2} max={20} value={txPower} onChange={e => setTxPower(Number(e.target.value))}
            className="w-full mt-2 accent-blue-500" />
        </div>
        <div>
          <label className="text-sm font-medium text-text-secondary">环境类型</label>
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {envOptions.map((e, i) => (
              <button key={e.en} onClick={() => setEnvIdx(i)}
                className={`px-2.5 py-1.5 rounded text-xs ${envIdx === i ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
                {e.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "接收灵敏度", value: `${sensitivity[sf]} dBm` },
          { label: "链路预算", value: `${linkBudget} dB` },
          { label: "数据速率", value: `${dr} kbps` },
          { label: "空中时间(255B)", value: `${airtime} ms` },
        ].map(item => (
          <div key={item.label} className="bg-bg-surface rounded-lg p-3 text-center">
            <div className="text-xs text-text-secondary">{item.label}</div>
            <div className="text-lg font-mono font-bold text-text-primary">{item.value}</div>
          </div>
        ))}
      </div>

      <div className="mt-4 bg-bg-surface rounded-lg p-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-text-secondary">估算通信距离</span>
          <span className="text-2xl font-mono font-bold text-blue-400">{clamped.toFixed(1)} km</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-3">
          <div className="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full transition-all"
            style={{ width: `${Math.min(clamped / 30 * 100, 100)}%` }} />
        </div>
      </div>
    </div>
  );
}
export default LoRaRangeCalculator;
