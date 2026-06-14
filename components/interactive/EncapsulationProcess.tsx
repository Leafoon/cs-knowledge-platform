"use client";
import { useState } from "react";

interface LayerHeader {
  layer: string;
  name: string;
  fields: string[];
  size: string;
  color: string;
}

const HEADERS: LayerHeader[] = [
  { layer: "应用层", name: "HTTP首部", fields: ["Method", "URL", "Headers", "Body"], size: "可变", color: "bg-blue-500" },
  { layer: "传输层", name: "TCP首部", fields: ["源端口(16b)", "目的端口(16b)", "序号(32b)", "确认号(32b)", "标志位(6b)", "窗口(16b)", "校验和(16b)"], size: "20-60B", color: "bg-green-500" },
  { layer: "网络层", name: "IP首部", fields: ["版本(4b)", "首部长度(4b)", "TTL(8b)", "协议(8b)", "源IP(32b)", "目的IP(32b)"], size: "20-60B", color: "bg-yellow-500" },
  { layer: "链路层", name: "以太网帧首部", fields: ["目的MAC(48b)", "源MAC(48b)", "类型(16b)"], size: "14B+4B尾部", color: "bg-red-500" },
];

export function EncapsulationProcess() {
  const [active, setActive] = useState(0);
  const [showAll, setShowAll] = useState(false);

  const current = HEADERS[active];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">各层协议首部详解</h3>
      <div className="flex gap-2 mb-4">
        {HEADERS.map((h, i) => (
          <button key={i} onClick={() => { setActive(i); setShowAll(false); }}
            className={`px-3 py-1.5 rounded text-sm ${active === i ? `${h.color} text-white` : "bg-bg-subtle text-text-secondary"}`}>
            {h.layer}
          </button>
        ))}
        <button onClick={() => setShowAll(!showAll)}
          className={`px-3 py-1.5 rounded text-sm ${showAll ? "bg-purple-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
          {showAll ? "隐藏全部" : "显示全部"}
        </button>
      </div>
      {!showAll && (
        <div className="bg-bg-muted rounded-lg p-4 mb-4">
          <div className="flex items-center gap-2 mb-3">
            <span className={`w-3 h-3 rounded-full ${current.color}`} />
            <span className="font-semibold text-text-primary">{current.name}</span>
            <span className="text-xs text-text-secondary">({current.size})</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {current.fields.map((f, i) => (
              <span key={i} className={`px-2 py-1 rounded text-xs font-mono ${current.color} text-white`}>{f}</span>
            ))}
          </div>
        </div>
      )}
      {showAll && (
        <div className="space-y-2 mb-4">
          {HEADERS.map((h, i) => (
            <div key={i} className="bg-bg-muted rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <span className={`w-3 h-3 rounded-full ${h.color}`} />
                <span className="font-semibold text-sm text-text-primary">{h.name}</span>
                <span className="text-xs text-text-secondary">({h.size})</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {h.fields.map((f, j) => (
                  <span key={j} className={`px-2 py-0.5 rounded text-xs font-mono ${h.color} text-white`}>{f}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="bg-bg-subtle rounded p-3 font-mono text-xs text-text-secondary mb-4">
        <div className="mb-1">完整帧结构:</div>
        <div className="flex items-center gap-0.5 overflow-x-auto">
          <span className="px-1.5 py-1 bg-red-500 text-white rounded flex-shrink-0">帧头14B</span>
          <span className="px-1.5 py-1 bg-yellow-500 text-white rounded flex-shrink-0">IP头20B</span>
          <span className="px-1.5 py-1 bg-green-500 text-white rounded flex-shrink-0">TCP头20B</span>
          <span className="px-1.5 py-1 bg-blue-500 text-white rounded flex-shrink-0">HTTP数据</span>
          <span className="px-1.5 py-1 bg-red-500 text-white rounded flex-shrink-0">FCS 4B</span>
        </div>
      </div>
      <div className="text-xs text-text-secondary">
        每层添加自己的首部:应用层数据 → 加TCP首部(段) → 加IP首部(数据报) → 加帧头帧尾(帧) → 比特流传输
      </div>
    </div>
  );
}

export default EncapsulationProcess;
