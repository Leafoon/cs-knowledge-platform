"use client";
import { useState } from "react";

const headers = [
  { name: "Ethernet", size: 14, color: "bg-purple-600", desc: "DMAC(6B) + SMAC(6B) + EtherType(2B)" },
  { name: "IP", size: 20, color: "bg-blue-600", desc: "IPv4 头部 20 字节 (无选项)" },
  { name: "TCP", size: 20, color: "bg-green-600", desc: "TCP 头部 20 字节 (无选项)" },
  { name: "HTTP/TLS", size: 0, color: "bg-orange-600", desc: "HTTP 请求头 + TLS 记录头 (可变)" },
];

export function OverheadCalculator() {
  const [payloadSize, setPayloadSize] = useState(1460);
  const [useIPv6, setUseIPv6] = useState(false);
  const [useUDP, setUseUDP] = useState(false);

  const ethSize = 14;
  const ipSize = useIPv6 ? 40 : 20;
  const transportSize = useUDP ? 8 : 20;
  const vlanSize = 4;
  const totalHeader = ethSize + ipSize + transportSize;
  const totalSize = totalHeader + payloadSize;
  const efficiency = (payloadSize / totalSize) * 100;
  const overhead = ((totalSize - payloadSize) / totalSize) * 100;

  const layers = [
    { name: "Ethernet", size: ethSize, color: "bg-purple-600", pct: (ethSize / totalSize) * 100 },
    { name: useIPv6 ? "IPv6" : "IPv4", size: ipSize, color: "bg-blue-600", pct: (ipSize / totalSize) * 100 },
    { name: useUDP ? "UDP" : "TCP", size: transportSize, color: "bg-green-600", pct: (transportSize / totalSize) * 100 },
    { name: "Payload", size: payloadSize, color: "bg-gray-500", pct: (payloadSize / totalSize) * 100 },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">📦 封装开销计算器</h3>
      <p className="text-sm text-text-secondary mb-4">计算各层首部的封装开销比例</p>

      <div className="flex gap-3 mb-4 flex-wrap">
        <button onClick={() => setUseIPv6(!useIPv6)}
          className={`px-3 py-1.5 rounded text-sm ${useIPv6 ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary"}`}>
          {useIPv6 ? "IPv6 (40B)" : "IPv4 (20B)"}
        </button>
        <button onClick={() => setUseUDP(!useUDP)}
          className={`px-3 py-1.5 rounded text-sm ${useUDP ? "bg-green-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary"}`}>
          {useUDP ? "UDP (8B)" : "TCP (20B)"}
        </button>
      </div>

      <div className="mb-4">
        <label className="text-sm text-text-secondary">Payload 大小: {payloadSize} 字节</label>
        <input type="range" min={1} max={9000} value={payloadSize}
          onChange={e => setPayloadSize(Number(e.target.value))} className="w-full accent-gray-500" />
      </div>

      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-text-secondary">帧结构</span>
          <span className="font-mono text-sm text-text-primary">总大小: {totalSize} B</span>
        </div>
        <div className="w-full h-10 rounded-lg overflow-hidden flex">
          {layers.map(l => (
            <div key={l.name} className={`${l.color} flex items-center justify-center transition-all`}
              style={{ width: `${Math.max(l.pct, 3)}%` }}>
              {l.pct > 8 && <span className="text-[10px] text-white font-bold">{l.name} {l.size}B</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {layers.map(l => (
          <div key={l.name} className="bg-bg-surface rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <div className={`w-3 h-3 rounded ${l.color}`} />
              <span className="text-xs text-text-secondary">{l.name}</span>
            </div>
            <div className="font-mono text-lg font-bold text-text-primary">{l.size} B</div>
            <div className="text-[10px] text-text-secondary">{l.pct.toFixed(1)}%</div>
          </div>
        ))}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="bg-green-900/20 border border-green-700 rounded-lg p-3 text-center">
          <div className="text-xs text-green-300">有效载荷效率</div>
          <div className="text-2xl font-mono font-bold text-green-400">{efficiency.toFixed(1)}%</div>
        </div>
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-3 text-center">
          <div className="text-xs text-red-300">协议开销</div>
          <div className="text-2xl font-mono font-bold text-red-400">{overhead.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  );
}
export default OverheadCalculator;
