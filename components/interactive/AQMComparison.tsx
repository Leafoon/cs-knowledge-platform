"use client";
import { useState, useEffect, useRef } from "react";

const algorithms = [
  { name: "RED", label: "随机早期检测", desc: "根据平均队列长度概率性丢包", color: "bg-blue-500" },
  { name: "CoDel", label: "受控延迟", desc: "基于数据包在队列中的停留时间", color: "bg-green-500" },
  { name: "FQ-PIE", label: "公平队列 PIE", desc: "结合公平队列和 PI 控制器", color: "bg-purple-500" },
];

export function AQMComparison() {
  const [selected, setSelected] = useState<string[]>(["RED"]);
  const [queueData, setQueueData] = useState<Record<string, number[]>>({ RED: [], CoDel: [], "FQ-PIE": [] });
  const [arrivalRate, setArrivalRate] = useState(70);
  const runningRef = useRef(false);

  useEffect(() => {
    if (selected.length === 0) return;
    runningRef.current = true;
    const id = setInterval(() => {
      if (!runningRef.current) return;
      setQueueData((prev) => {
        const next: Record<string, number[]> = { ...prev };
        for (const alg of algorithms.map((a) => a.name)) {
          const old = next[alg] || [];
          const last = old.length > 0 ? old[old.length - 1] : 20;
          const growth = (arrivalRate / 100) * 8;
          const dropFactor = alg === "RED" ? 0.7 : alg === "CoDel" ? 0.5 : 0.6;
          const q = Math.max(0, Math.min(100, last + growth - (last > 50 ? last * dropFactor * 0.1 : 2)));
          next[alg] = [...old.slice(-39), Math.round(q)];
        }
        return next;
      });
    }, 300);
    return () => { runningRef.current = false; clearInterval(id); };
  }, [selected, arrivalRate]);

  const toggle = (name: string) => {
    setSelected((s) => s.includes(name) ? s.filter((n) => n !== name) : [...s, name]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">主动队列管理算法对比</h3>
      <div className="flex gap-2 mb-4">
        {algorithms.map((a) => (
          <button key={a.name} onClick={() => toggle(a.name)} className={`px-3 py-1 rounded text-sm transition-colors ${selected.includes(a.name) ? `${a.color} text-white` : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>{a.name}</button>
        ))}
      </div>
      <div className="mb-3">
        <label className="text-sm text-text-secondary">流量到达率: {arrivalRate}%</label>
        <input type="range" min={10} max={100} value={arrivalRate} onChange={(e) => setArrivalRate(+e.target.value)} className="w-full mt-1" />
      </div>
      <div className="space-y-3 mb-4">
        {algorithms.filter((a) => selected.includes(a.name)).map((a) => (
          <div key={a.name}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-text-primary font-medium">{a.label} ({a.name})</span>
              <span className="text-text-secondary">{(queueData[a.name] || []).slice(-1)[0] || 0}/100</span>
            </div>
            <div className="flex gap-px items-end h-10">
              {((queueData[a.name] || []).length === 0 ? new Array(40).fill(0) : queueData[a.name]!).map((v, i) => (
                <div key={i} className={`flex-1 rounded-t-sm ${a.color} opacity-70`} style={{ height: `${v}%` }} />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-xs text-text-secondary space-y-1">
        {algorithms.map((a) => <p key={a.name}><strong>{a.name}:</strong> {a.desc}</p>)}
      </div>
    </div>
  );
}
export default AQMComparison;
