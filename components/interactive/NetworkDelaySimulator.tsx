"use client";
import { useState } from "react";

export function NetworkDelaySimulator() {
  const [distance, setDistance] = useState(1000);
  const [bandwidth, setBandwidth] = useState(100);
  const [pktSize, setPktSize] = useState(1500);
  const [processing, setProcessing] = useState(1);
  const [queueLen, setQueueLen] = useState(5);

  const propSpeed = 200000;
  const propDelay = (distance / propSpeed) * 1000;
  const transDelay = ((pktSize * 8) / (bandwidth * 1000000)) * 1000;
  const procDelay = processing;
  const queueDelay = queueLen * transDelay;
  const totalDelay = propDelay + transDelay + procDelay + queueDelay;

  const delays = [
    { name: "传播延迟", en: "Propagation", value: propDelay, formula: "d / s", color: "bg-blue-500", desc: `距离 ${distance}km / 光速 ${propSpeed}km/s` },
    { name: "传输延迟", en: "Transmission", value: transDelay, formula: "L / R", color: "bg-green-500", desc: `${pktSize}B × 8 / ${bandwidth}Mbps` },
    { name: "处理延迟", en: "Processing", value: procDelay, formula: "固定值", color: "bg-yellow-500", desc: "查表、检查错误、转发决策" },
    { name: "排队延迟", en: "Queuing", value: queueDelay, formula: "可变", color: "bg-red-500", desc: `队列深度 ${queueLen} 个包` },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">⏱️ 网络延迟模拟器</h3>
      <p className="text-sm text-text-secondary mb-4">展示传播/传输/处理/排队四种延迟</p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary">传播距离: {distance} km</label>
          <input type="range" min={1} max={20000} value={distance} onChange={e => setDistance(Number(e.target.value))} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">链路带宽: {bandwidth} Mbps</label>
          <input type="range" min={1} max={10000} value={bandwidth} onChange={e => setBandwidth(Number(e.target.value))} className="w-full accent-green-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">包大小: {pktSize} B</label>
          <input type="range" min={64} max={9000} value={pktSize} onChange={e => setPktSize(Number(e.target.value))} className="w-full accent-yellow-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">排队深度: {queueLen} 包</label>
          <input type="range" min={0} max={50} value={queueLen} onChange={e => setQueueLen(Number(e.target.value))} className="w-full accent-red-500" />
        </div>
      </div>

      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-text-secondary">延迟分解</span>
          <span className="text-lg font-mono font-bold text-text-primary">{totalDelay.toFixed(2)} ms</span>
        </div>
        <div className="w-full h-8 rounded-lg overflow-hidden flex">
          {delays.map(d => (
            <div key={d.name} className={`${d.color} transition-all flex items-center justify-center`}
              style={{ width: `${Math.max((d.value / totalDelay) * 100, 2)}%` }}>
              {(d.value / totalDelay) > 0.1 && <span className="text-[10px] text-white font-bold">{d.value.toFixed(1)}ms</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {delays.map(d => (
          <div key={d.name} className="bg-bg-surface rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <div className={`w-3 h-3 rounded ${d.color}`} />
              <span className="text-xs text-text-secondary">{d.name}</span>
            </div>
            <div className="font-mono text-lg font-bold text-text-primary">{d.value.toFixed(2)} ms</div>
            <div className="text-[10px] text-text-secondary">{d.formula} — {d.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
export default NetworkDelaySimulator;
