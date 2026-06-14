"use client";
import { useState } from "react";

export function CrosstalkVisualizer() {
  const [separation, setSeparation] = useState(3);
  const [frequency, setFrequency] = useState(100);
  const [selectedPair, setSelectedPair] = useState(0);

  const pairs = ["橙白/橙", "绿白/绿", "蓝白/蓝", "棕白/棕"];
  const crosstalkDB = -20 + separation * 3 - frequency / 50;
  const signalQuality = Math.max(0, Math.min(100, 50 + separation * 8 - frequency / 10));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">串扰可视化 (Crosstalk)</h3>
      <p className="text-sm text-text-secondary mb-4">
        展示双绞线中相邻线对之间的电磁干扰(NEXT: 近端串扰)。
      </p>
      <div className="flex gap-2 mb-4">
        {pairs.map((p, i) => (
          <button key={i} onClick={() => setSelectedPair(i)}
            className={`px-3 py-1.5 rounded text-xs transition-colors ${selectedPair === i ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            线对 {i + 1}
          </button>
        ))}
      </div>
      <div className="mb-4">
        <label className="text-sm text-text-secondary">线对间距: {separation}mm</label>
        <input type="range" min={1} max={10} value={separation} onChange={(e) => setSeparation(Number(e.target.value))}
          className="w-full mt-1 accent-blue-500" />
      </div>
      <div className="mb-4">
        <label className="text-sm text-text-secondary">信号频率: {frequency} MHz</label>
        <input type="range" min={10} max={500} value={frequency} onChange={(e) => setFrequency(Number(e.target.value))}
          className="w-full mt-1 accent-purple-500" />
      </div>
      <div className="relative h-32 bg-gray-50 dark:bg-gray-900 rounded mb-4 overflow-hidden">
        {pairs.map((_, i) => {
          const y = 15 + i * 25;
          const interference = i !== selectedPair ? Math.max(0, 30 - separation * 2 + frequency / 30) : 0;
          return (
            <g key={i}>
              <div className="absolute left-4 text-xs text-text-secondary" style={{ top: y - 4 }}>{pairs[i]}</div>
              <div className={`absolute h-2 rounded-full transition-all duration-500 ${i === selectedPair ? "bg-blue-500" : "bg-gray-400 dark:bg-gray-600"}`}
                style={{ top: y, left: 100, width: 300, opacity: i === selectedPair ? 1 : 0.5 }} />
              {i !== selectedPair && interference > 5 && (
                <div className="absolute h-1 bg-red-400 rounded-full animate-pulse"
                  style={{ top: y + 2, left: 100, width: interference * 3 }} />
              )}
            </g>
          );
        })}
      </div>
      <div className="grid grid-cols-3 gap-3 text-center text-sm">
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
          <div className="text-text-secondary text-xs">NEXT衰减</div>
          <div className="font-mono text-text-primary">{crosstalkDB.toFixed(1)} dB</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
          <div className="text-text-secondary text-xs">信号质量</div>
          <div className={`font-mono ${signalQuality > 70 ? "text-green-500" : signalQuality > 40 ? "text-yellow-500" : "text-red-500"}`}>
            {signalQuality.toFixed(0)}%
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
          <div className="text-text-secondary text-xs">干扰线对</div>
          <div className="font-mono text-text-primary">{pairs.filter((_, i) => i !== selectedPair).length}</div>
        </div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">串扰类型</div>
        <div>• NEXT (近端串扰): 干扰信号在发送端附近泄漏</div>
        <div>• FEXT (远端串扰): 干扰信号在接收端附近泄漏</div>
        <div>• 双绞线通过扭绞抵消电磁干扰，扭绞越密抗干扰越强</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">线缆等级与串扰</div>
        <div>• Cat5e: 100MHz，NEXT≥35dB，适用于千兆以太网</div>
        <div>• Cat6: 250MHz，NEXT≥44dB，适用于万兆以太网(55m)</div>
        <div>• Cat6a: 500MHz，NEXT≥48dB，适用于万兆以太网(100m)</div>
      </div>
    </div>
  );
}
export default CrosstalkVisualizer;
