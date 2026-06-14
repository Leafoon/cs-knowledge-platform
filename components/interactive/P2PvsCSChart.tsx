"use client";
import { useState } from "react";

const metrics = [
  { name: "可扩展性", en: "Scalability", p2p: 90, cs: 40, desc: "P2P 节点越多性能越好；C/S 受服务器瓶颈限制" },
  { name: "可靠性", en: "Reliability", p2p: 60, cs: 85, desc: "C/S 有专业运维和冗余；P2P 依赖节点稳定性" },
  { name: "管理成本", en: "Management", p2p: 90, cs: 30, desc: "P2P 无中心管理成本；C/S 需要专业运维团队" },
  { name: "数据一致性", en: "Consistency", p2p: 50, cs: 90, desc: "C/S 有中心数据库保证一致；P2P 需要共识协议" },
  { name: "抗审查性", en: "Censorship Resist", p2p: 95, cs: 20, desc: "P2P 无中心控制点；C/S 可被单点封锁" },
  { name: "初始部署", en: "Initial Setup", p2p: 80, cs: 50, desc: "P2P 无需服务器基础设施；C/S 需要机房和带宽" },
];

export function P2PvsCSChart() {
  const [peerCount, setPeerCount] = useState(100);
  const [selected, setSelected] = useState<number | null>(null);

  const p2pThroughput = Math.min(100, 20 + peerCount * 0.8);
  const csThroughput = Math.max(20, 100 - peerCount * 0.3);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">⚖️ P2P vs C/S 对比图</h3>
      <p className="text-sm text-text-secondary mb-4">对比两种架构的可扩展性和性能</p>

      <div className="mb-4">
        <label className="text-sm text-text-secondary">节点/客户端数量: {peerCount}</label>
        <input type="range" min={10} max={500} value={peerCount}
          onChange={e => setPeerCount(Number(e.target.value))} className="w-full accent-blue-500" />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-3 text-center">
          <div className="text-sm text-blue-300 mb-1">P2P 吞吐量</div>
          <div className="text-2xl font-mono font-bold text-blue-400">{p2pThroughput.toFixed(0)}%</div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div className="bg-blue-500 h-2 rounded-full transition-all" style={{ width: `${p2pThroughput}%` }} />
          </div>
        </div>
        <div className="bg-green-900/20 border border-green-700 rounded-lg p-3 text-center">
          <div className="text-sm text-green-300 mb-1">C/S 吞吐量</div>
          <div className="text-2xl font-mono font-bold text-green-400">{csThroughput.toFixed(0)}%</div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div className="bg-green-500 h-2 rounded-full transition-all" style={{ width: `${csThroughput}%` }} />
          </div>
        </div>
      </div>

      <div className="space-y-2">
        {metrics.map((m, i) => (
          <button key={m.en} onClick={() => setSelected(selected === i ? null : i)}
            className={`w-full p-3 rounded-lg text-sm text-left transition-all ${selected === i ? "bg-bg-surface border border-blue-500" : "bg-bg-surface border border-border-subtle hover:border-blue-400"}`}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-text-primary font-medium">{m.name} <span className="text-text-secondary text-xs">{m.en}</span></span>
            </div>
            <div className="flex gap-2 items-center">
              <span className="text-xs text-blue-400 w-12 text-right">P2P {m.p2p}</span>
              <div className="flex-1 flex gap-0.5">
                <div className="h-4 bg-blue-500 rounded-l transition-all" style={{ width: `${m.p2p}%` }} />
                <div className="h-4 bg-green-500 rounded-r transition-all" style={{ width: `${m.cs}%` }} />
              </div>
              <span className="text-xs text-green-400 w-12">{m.cs} C/S</span>
            </div>
            {selected === i && <div className="text-xs text-text-secondary mt-2">{m.desc}</div>}
          </button>
        ))}
      </div>
    </div>
  );
}
export default P2PvsCSChart;
