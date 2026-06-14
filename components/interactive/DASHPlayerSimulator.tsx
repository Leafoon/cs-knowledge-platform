"use client";
import { useState, useEffect, useRef } from "react";

const bitrates = [500, 1000, 2000, 4000, 8000];
const bitrateLabels = ["500kbps", "1Mbps", "2Mbps", "4Mbps", "8Mbps"];

export function DASHPlayerSimulator() {
  const [buffer, setBuffer] = useState(5);
  const [currentBR, setCurrentBR] = useState(2);
  const [bandwidth, setBandwidth] = useState(4000);
  const [running, setRunning] = useState(false);
  const [rebufferCount, setRebufferCount] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(() => {
        setBuffer((b) => {
          const consumption = 1;
          const fetchRate = bandwidth >= bitrates[currentBR] ? 1.5 : 0.5;
          const newBuf = Math.max(0, Math.min(15, b - consumption + fetchRate));
          if (newBuf === 0) setRebufferCount((r) => r + 1);
          return newBuf;
        });
      }, 800);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running, currentBR, bandwidth]);

  useEffect(() => {
    if (buffer < 3 && currentBR > 0) setCurrentBR((br) => br - 1);
    else if (buffer > 10 && currentBR < 4) setCurrentBR((br) => br + 1);
  }, [buffer, currentBR]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DASH 自适应播放器模拟器</h3>
      <div className="mb-4">
        <label className="text-sm text-text-secondary">可用带宽: {(bandwidth / 1000).toFixed(1)} Mbps</label>
        <input type="range" min={500} max={10000} step={500} value={bandwidth}
          onChange={(e) => setBandwidth(Number(e.target.value))} className="w-full mt-1 accent-green-500" />
      </div>
      <div className="relative h-16 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden mb-4">
        <div className={`absolute left-0 top-0 h-full rounded-full transition-all duration-500 ${buffer < 3 ? "bg-red-400" : buffer < 7 ? "bg-yellow-400" : "bg-green-400"}`}
          style={{ width: `${(buffer / 15) * 100}%` }} />
        <div className="absolute inset-0 flex items-center justify-center text-sm font-mono text-text-primary">
          缓冲区: {buffer.toFixed(1)}s / 15s
        </div>
      </div>
      <div className="flex gap-1 mb-4">
        {bitrateLabels.map((label, i) => (
          <div key={i} className={`flex-1 text-center py-2 rounded text-xs font-mono transition-all ${i === currentBR ? "bg-blue-600 text-white scale-105" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {label}
          </div>
        ))}
      </div>
      <button onClick={() => setRunning(!running)}
        className={`w-full py-2 rounded font-medium transition-colors ${running ? "bg-red-600 text-white" : "bg-blue-600 text-white"}`}>
        {running ? "停止播放" : "开始播放"}
      </button>
      <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-2 text-center">
          <div className="text-text-secondary">当前码率</div>
          <div className="font-mono text-text-primary">{bitrateLabels[currentBR]}</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-2 text-center">
          <div className="text-text-secondary">卡顿次数</div>
          <div className="font-mono text-red-500">{rebufferCount}</div>
        </div>
      </div>
      <div className="mt-2 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">ABR算法原理</div>
        <div>• 缓冲区低 → 降低码率，防止卡顿</div>
        <div>• 缓冲区高 → 提升码率，改善画质</div>
        <div>• 目标: 在带宽波动下最大化视频质量，最小化卡顿</div>
      </div>
    </div>
  );
}
export default DASHPlayerSimulator;
