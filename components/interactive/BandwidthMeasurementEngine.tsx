"use client";
import { useState, useRef, useCallback, useEffect } from "react";

interface MeasurementResult {
  timestamp: number;
  throughput: number;
  method: string;
}

export function BandwidthMeasurementEngine() {
  const [running, setRunning] = useState(false);
  const [method, setMethod] = useState<"tcp" | "udp">("tcp");
  const [duration, setDuration] = useState(5);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<MeasurementResult[]>([]);
  const [currentSpeed, setCurrentSpeed] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const simulateMeasurement = useCallback(() => {
    setRunning(true);
    setResults([]);
    setProgress(0);
    let elapsed = 0;
    const interval = 100;
    const baseSpeed = method === "tcp" ? 850 : 920;
    const jitter = method === "tcp" ? 80 : 150;

    timerRef.current = setInterval(() => {
      elapsed += interval;
      const pct = (elapsed / (duration * 1000)) * 100;
      setProgress(Math.min(pct, 100));
      const speed = baseSpeed + (Math.random() - 0.5) * jitter + Math.sin(elapsed / 500) * 30;
      setCurrentSpeed(Math.max(0, speed));
      if (elapsed % 1000 === 0) {
        setResults((prev) => [...prev, { timestamp: elapsed / 1000, throughput: speed, method }]);
      }
      if (elapsed >= duration * 1000) {
        if (timerRef.current) clearInterval(timerRef.current);
        setRunning(false);
      }
    }, interval);
  }, [method, duration]);

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  const avgSpeed = results.length > 0 ? results.reduce((s, r) => s + r.throughput, 0) / results.length : 0;
  const maxSpeed = results.length > 0 ? Math.max(...results.map((r) => r.throughput)) : 0;
  const minSpeed = results.length > 0 ? Math.min(...results.map((r) => r.throughput)) : 0;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">带宽测量引擎 (iperf3模拟)</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs text-text-secondary mb-2 block">测量协议</label>
          <div className="flex gap-2">
            <button onClick={() => setMethod("tcp")}
              className={`flex-1 py-1.5 rounded text-sm ${method === "tcp" ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
              TCP
            </button>
            <button onClick={() => setMethod("udp")}
              className={`flex-1 py-1.5 rounded text-sm ${method === "udp" ? "bg-purple-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
              UDP
            </button>
          </div>
        </div>
        <div>
          <label className="text-xs text-text-secondary">测试时长: {duration}s</label>
          <input type="range" min={3} max={30} value={duration} onChange={(e) => setDuration(+e.target.value)}
            className="w-full mt-1" />
        </div>
      </div>
      <div className="mb-4">
        <div className="flex justify-between text-xs text-text-secondary mb-1">
          <span>进度</span>
          <span>{progress.toFixed(0)}%</span>
        </div>
        <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${progress}%` }} />
        </div>
      </div>
      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="p-3 rounded bg-blue-50 dark:bg-blue-900/20 text-center">
          <div className="text-xs text-blue-600 dark:text-blue-400">当前速率</div>
          <div className="text-lg font-bold text-blue-700 dark:text-blue-300">{currentSpeed.toFixed(1)} Mbps</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">平均</div>
          <div className="text-lg font-bold text-text-primary">{avgSpeed.toFixed(1)}</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">峰值</div>
          <div className="text-lg font-bold text-text-primary">{maxSpeed.toFixed(1)}</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">最低</div>
          <div className="text-lg font-bold text-text-primary">{minSpeed.toFixed(1)}</div>
        </div>
      </div>
      {results.length > 0 && (
        <div className="mb-4 h-32 flex items-end gap-1 px-2">
          {results.map((r, i) => (
            <div key={i} className="flex-1 flex flex-col items-center">
              <div className="w-full bg-blue-500 rounded-t" style={{ height: `${(r.throughput / 1100) * 100}%` }} />
              <span className="text-[9px] text-text-secondary mt-1">{r.timestamp}s</span>
            </div>
          ))}
        </div>
      )}
      <button onClick={simulateMeasurement} disabled={running}
        className="w-full py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded text-sm font-medium">
        {running ? "测量中..." : "开始测量"}
      </button>
      <p className="text-xs text-text-secondary mt-3">iperf3通过TCP/UDP流测量端到端带宽，TCP受拥塞控制影响，UDP可测试网络极限。</p>
    </div>
  );
}
export default BandwidthMeasurementEngine;
