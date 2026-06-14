"use client";
import { useState, useEffect, useRef, useCallback } from "react";

export function TCPPerformanceDashboard() {
  const [cwnd, setCwnd] = useState(1);
  const [ssthresh, setSsthresh] = useState(32);
  const [rtt, setRtt] = useState(50);
  const [retransmits, setRetransmits] = useState(0);
  const [phase, setPhase] = useState("慢启动");
  const [history, setHistory] = useState<number[]>([]);
  const [lossRate, setLossRate] = useState(5);
  const timerRef = useRef<ReturnType<typeof setInterval>>();

  const step = useCallback(() => {
    setCwnd((c) => {
      const loss = Math.random() * 100 < lossRate;
      if (loss) {
        setRetransmits((r) => r + 1);
        setSsthresh(Math.max(2, Math.floor(c / 2)));
        setPhase("快速恢复");
        return Math.max(1, Math.floor(c / 2));
      }
      if (c < ssthresh) {
        setPhase("慢启动");
        return Math.min(c * 2, 128);
      }
      setPhase("拥塞避免");
      return Math.min(c + 1, 128);
    });
  }, [lossRate, ssthresh]);

  const [autoRun, setAutoRun] = useState(false);
  useEffect(() => {
    if (autoRun) {
      timerRef.current = setInterval(() => step(), 600);
      return () => clearInterval(timerRef.current);
    }
  }, [autoRun, step]);

  useEffect(() => {
    setHistory((h) => [...h.slice(-49), cwnd]);
  }, [cwnd]);

  const W = 400; const H = 100; const maxC = 130;
  const path = history.map((v, i) => `${i === 0 ? "M" : "L"} ${(i / 49) * W} ${H - (v / maxC) * H}`).join(" ");

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCP 性能仪表盘</h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        {[
          { label: "cwnd", value: cwnd, unit: "MSS", color: "blue" },
          { label: "ssthresh", value: ssthresh, unit: "MSS", color: "yellow" },
          { label: "RTT", value: rtt, unit: "ms", color: "green" },
          { label: "重传", value: retransmits, unit: "次", color: "red" },
        ].map((m) => (
          <div key={m.label} className={`p-3 rounded-lg bg-${m.color}-500/10 border border-${m.color}-400/30`}>
            <span className={`text-${m.color}-400 text-xs`}>{m.label}</span>
            <p className="text-text-primary text-xl font-bold">{m.value}<span className="text-text-muted text-xs ml-1">{m.unit}</span></p>
          </div>
        ))}
      </div>
      <div className="mb-4">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md bg-bg-primary rounded border border-border-subtle">
          {path && <path d={path} fill="none" stroke="#60a5fa" strokeWidth="2" />}
          <line x1="0" y1={H - (ssthresh / maxC) * H} x2={W} y2={H - (ssthresh / maxC) * H} stroke="#facc15" strokeWidth="1" strokeDasharray="4,2" />
        </svg>
        <div className="flex justify-between text-text-muted text-xs mt-1">
          <span>黄色虚线 = ssthresh</span><span>蓝色 = cwnd 曲线</span>
        </div>
      </div>
      <div className="flex items-center gap-3 mb-3">
        <button onClick={step} className="px-4 py-2 rounded bg-blue-500 text-white text-sm hover:bg-blue-600">单步</button>
        <button onClick={() => setAutoRun(!autoRun)}
          className={`px-4 py-2 rounded text-sm ${autoRun ? "bg-red-500 text-white" : "bg-green-500 text-white"}`}>
          {autoRun ? "停止" : "自动"}
        </button>
        <span className="text-text-secondary text-sm">阶段: <span className="text-text-primary font-medium">{phase}</span></span>
      </div>
      <div className="mb-3">
        <label className="text-text-muted text-xs">丢包率: {lossRate}%</label>
        <input type="range" min="0" max="30" value={lossRate} onChange={(e) => setLossRate(Number(e.target.value))}
          className="w-full accent-blue-500" />
      </div>
      <p className="text-text-muted text-xs">吞吐量 ≈ cwnd × MSS / RTT。丢包时 cwnd 减半，ssthresh 设为 cwnd/2。</p>
    </div>
  );
}
export default TCPPerformanceDashboard;
