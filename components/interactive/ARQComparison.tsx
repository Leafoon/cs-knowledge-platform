"use client";
import { useState } from "react";

const protocols = [
  { name: "Stop-and-Wait", label: "停止-等待", efficiency: "低", throughput: "1/(1+2a)", windowSize: 1, desc: "每次发送一帧，等待确认后才发下一帧", color: "bg-red-500" },
  { name: "Go-Back-N", label: "回退 N", efficiency: "中", throughput: "N/(1+2a)", windowSize: 4, desc: "发送窗口 >1，接收窗口 =1，出错重传后续所有帧", color: "bg-yellow-500" },
  { name: "Selective Repeat", label: "选择重传", efficiency: "高", throughput: "N/(1+2a)", windowSize: 4, desc: "发送和接收窗口均 >1，只重传出错帧", color: "bg-green-500" },
];

export function ARQComparison() {
  const [active, setActive] = useState(0);
  const [propagationDelay, setPropagationDelay] = useState(3);
  const [frameSize, setFrameSize] = useState(1);
  const [bitRate, setBitRate] = useState(10);

  const transmissionDelay = frameSize / bitRate;
  const a = propagationDelay / transmissionDelay;
  const proto = protocols[active];

  const calcEfficiency = () => {
    if (active === 0) return (1 / (1 + 2 * a));
    return Math.min(1, proto.windowSize / (1 + 2 * a));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">ARQ 协议对比</h3>
      <div className="flex gap-2 mb-4">
        {protocols.map((p, i) => (
          <button key={p.name} onClick={() => setActive(i)} className={`px-3 py-2 rounded text-sm transition-colors ${active === i ? `${p.color} text-white` : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-xs text-text-secondary">传播延迟: {propagationDelay} ms</label>
          <input type="range" min={1} max={20} value={propagationDelay} onChange={(e) => setPropagationDelay(+e.target.value)} className="w-full mt-1" />
        </div>
        <div>
          <label className="text-xs text-text-secondary">数据速率: {bitRate} Mbps</label>
          <input type="range" min={1} max={100} value={bitRate} onChange={(e) => setBitRate(+e.target.value)} className="w-full mt-1" />
        </div>
      </div>
      <div className="mb-4 p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <p className="text-sm font-semibold text-text-primary mb-1">{proto.name} — {proto.label}</p>
        <p className="text-xs text-text-secondary mb-3">{proto.desc}</p>
        <div className="grid grid-cols-4 gap-3 text-center">
          <div className="p-2 rounded bg-white dark:bg-gray-900"><div className="text-xs text-text-secondary">效率</div><div className="font-bold text-text-primary">{(calcEfficiency() * 100).toFixed(1)}%</div></div>
          <div className="p-2 rounded bg-white dark:bg-gray-900"><div className="text-xs text-text-secondary">a 值</div><div className="font-bold text-text-primary">{a.toFixed(1)}</div></div>
          <div className="p-2 rounded bg-white dark:bg-gray-900"><div className="text-xs text-text-secondary">发送窗口</div><div className="font-bold text-text-primary">{proto.windowSize}</div></div>
          <div className="p-2 rounded bg-white dark:bg-gray-900"><div className="text-xs text-text-secondary">吞吐量</div><div className="font-bold text-text-primary">{proto.throughput}</div></div>
        </div>
      </div>
      <div className="h-8 flex items-end gap-1">
        {protocols.map((p, i) => {
          const eff = i === 0 ? 1 / (1 + 2 * a) : Math.min(1, p.windowSize / (1 + 2 * a));
          return <div key={p.name} className={`flex-1 rounded-t ${p.color} transition-all`} style={{ height: `${eff * 100}%` }} />;
        })}
      </div>
      <div className="flex gap-1 mt-1">
        {protocols.map((p) => <div key={p.name} className="flex-1 text-center text-xs text-text-secondary">{p.label}</div>)}
      </div>
    </div>
  );
}
export default ARQComparison;
