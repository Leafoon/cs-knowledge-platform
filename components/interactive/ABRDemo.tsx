"use client";
import { useState, useEffect, useRef } from "react";

const BITRATES = [500, 1000, 2000, 4000, 8000];
const BITRATE_LABELS = ["500kbps", "1Mbps", "2Mbps", "4Mbps", "8Mbps"];

export function ABRDemo() {
  const [buffer, setBuffer] = useState(5);
  const [bitrateIdx, setBitrateIdx] = useState(2);
  const [bandwidth, setBandwidth] = useState(3000);
  const [strategy, setStrategy] = useState<"buffer" | "throughput">("buffer");
  const [log, setLog] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startSimulation = () => {
    setBuffer(5);
    setBitrateIdx(2);
    setLog([]);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setBuffer((prev) => {
        const consumed = BITRATES[bitrateIdx] / 1000;
        const fillRate = bandwidth / 1000;
        const next = Math.max(0, Math.min(15, prev - consumed + fillRate));
        return parseFloat(next.toFixed(1));
      });
    }, 800);
  };

  useEffect(() => {
    if (strategy === "buffer") {
      if (buffer < 3 && bitrateIdx > 0) {
        setBitrateIdx((i) => i - 1);
        setLog((l) => [`↓ 降至 ${BITRATE_LABELS[bitrateIdx - 1]}`, ...l].slice(0, 8));
      } else if (buffer > 10 && bitrateIdx < BITRATES.length - 1) {
        setBitrateIdx((i) => i + 1);
        setLog((l) => [`↑ 升至 ${BITRATE_LABELS[bitrateIdx + 1]}`, ...l].slice(0, 8));
      }
    } else {
      const best = BITRATES.findIndex((b) => b > bandwidth * 0.8);
      const target = best === -1 ? BITRATES.length - 1 : Math.max(0, best - 1);
      if (target !== bitrateIdx) {
        setBitrateIdx(target);
        setLog((l) => [`→ 切换至 ${BITRATE_LABELS[target]}`, ...l].slice(0, 8));
      }
    }
  }, [buffer, bandwidth, strategy]);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">自适应比特率流媒体演示</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStrategy("buffer")} className={`px-3 py-1 rounded text-sm ${strategy === "buffer" ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>Buffer-based</button>
        <button onClick={() => setStrategy("throughput")} className={`px-3 py-1 rounded text-sm ${strategy === "throughput" ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>Throughput-based</button>
        <button onClick={startSimulation} className="ml-auto px-3 py-1 rounded text-sm bg-green-500 text-white hover:bg-green-600">启动模拟</button>
      </div>
      <div className="mb-3">
        <label className="text-sm text-text-secondary">可用带宽: {bandwidth} kbps</label>
        <input type="range" min={500} max={10000} step={100} value={bandwidth} onChange={(e) => setBandwidth(+e.target.value)} className="w-full mt-1" />
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800">
          <div className="text-xs text-text-secondary mb-1">缓冲区</div>
          <div className="w-full h-4 bg-gray-300 dark:bg-gray-600 rounded-full overflow-hidden">
            <div className="h-full bg-blue-500 transition-all" style={{ width: `${(buffer / 15) * 100}%` }} />
          </div>
          <div className="text-xs text-text-secondary mt-1">{buffer.toFixed(1)}s / 15s</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800">
          <div className="text-xs text-text-secondary mb-1">当前码率</div>
          <div className="text-lg font-bold text-text-primary">{BITRATE_LABELS[bitrateIdx]}</div>
        </div>
      </div>
      <div className="flex gap-1 mb-2">
        {BITRATES.map((b, i) => (
          <div key={b} className={`flex-1 h-6 rounded text-xs flex items-center justify-center text-white ${i === bitrateIdx ? "bg-blue-500" : "bg-gray-300 dark:bg-gray-600"}`}>
            {BITRATE_LABELS[i]}
          </div>
        ))}
      </div>
      <div className="mt-3 p-2 rounded bg-gray-50 dark:bg-gray-800 text-xs text-text-secondary max-h-24 overflow-y-auto">
        {log.length === 0 ? "等待模拟启动..." : log.map((l, i) => <div key={i}>{l}</div>)}
      </div>
    </div>
  );
}
export default ABRDemo;
