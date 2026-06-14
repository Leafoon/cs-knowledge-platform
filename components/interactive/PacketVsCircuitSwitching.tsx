"use client";
import { useState } from "react";

const features = [
  { name: "连接建立", en: "Connection Setup", packet: "无需建立连接", circuit: "需要建立专用电路" },
  { name: "资源分配", en: "Resource Alloc", packet: "按需动态分配", circuit: "预分配固定资源" },
  { name: "带宽利用", en: "Bandwidth", packet: "统计复用，效率高", circuit: "独占带宽，空闲浪费" },
  { name: "延迟特性", en: "Latency", packet: "可变延迟（排队）", circuit: "固定低延迟" },
  { name: "拥塞处理", en: "Congestion", packet: "可能丢包/排队", circuit: "无拥塞（已预留）" },
  { name: "可靠性", en: "Reliability", packet: "端到端协议保证", circuit: "链路级保证" },
  { name: "抗故障", en: "Fault Tolerance", packet: "可路由绕行", circuit: "电路中断需重建立" },
  { name: "典型应用", en: "Example", packet: "互联网 (IP)", circuit: "电话网 (PSTN)" },
];

export function PacketVsCircuitSwitching() {
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null);
  const [simBurst, setSimBurst] = useState(false);
  const [packetQueue, setPacketQueue] = useState(0);
  const [circuitBusy, setCircuitBusy] = useState(0);

  const simulateBurst = () => {
    setSimBurst(true);
    setPacketQueue(0);
    setCircuitBusy(0);
    let i = 0;
    const iv = setInterval(() => {
      i++;
      setPacketQueue(prev => Math.min(100, prev + 15 - Math.random() * 8));
      setCircuitBusy(prev => Math.min(100, prev + 12));
      if (i >= 10) { clearInterval(iv); setSimBurst(false); }
    }, 300);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">🔀 分组交换 vs 电路交换</h3>
      <p className="text-sm text-text-secondary mb-4">对比两种交换方式的特点</p>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-3 text-center">
          <div className="text-lg mb-1">📦</div>
          <div className="text-sm font-medium text-blue-300">分组交换</div>
          <div className="text-xs text-text-secondary">Packet Switching</div>
        </div>
        <div className="bg-green-900/20 border border-green-700 rounded-lg p-3 text-center">
          <div className="text-lg mb-1">📞</div>
          <div className="text-sm font-medium text-green-300">电路交换</div>
          <div className="text-xs text-text-secondary">Circuit Switching</div>
        </div>
      </div>

      <div className="space-y-1.5 mb-4">
        {features.map((f, i) => (
          <button key={f.en} onClick={() => setSelectedFeature(selectedFeature === i ? null : i)}
            className={`w-full grid grid-cols-3 gap-2 p-2.5 rounded-lg text-sm text-left transition-all ${selectedFeature === i ? "bg-bg-surface border border-blue-500" : "bg-bg-surface border border-border-subtle hover:border-blue-400"}`}>
            <span className="text-text-primary font-medium">{f.name}</span>
            <span className="text-blue-300 text-xs">{f.packet}</span>
            <span className="text-green-300 text-xs">{f.circuit}</span>
          </button>
        ))}
      </div>

      <div className="mb-4">
        <button onClick={simulateBurst} disabled={simBurst}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
          {simBurst ? "模拟中..." : "模拟流量突发"}
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-xs text-text-secondary mb-1">分组交换 — 队列利用率</div>
          <div className="w-full bg-gray-700 rounded-full h-4">
            <div className="bg-blue-500 h-4 rounded-full transition-all" style={{ width: `${packetQueue}%` }} />
          </div>
          <div className="text-xs text-text-secondary mt-1">{packetQueue.toFixed(0)}% (统计复用)</div>
        </div>
        <div>
          <div className="text-xs text-text-secondary mb-1">电路交换 — 带宽利用率</div>
          <div className="w-full bg-gray-700 rounded-full h-4">
            <div className="bg-green-500 h-4 rounded-full transition-all" style={{ width: `${circuitBusy}%` }} />
          </div>
          <div className="text-xs text-text-secondary mt-1">{circuitBusy.toFixed(0)}% (独占带宽)</div>
        </div>
      </div>
    </div>
  );
}
export default PacketVsCircuitSwitching;
