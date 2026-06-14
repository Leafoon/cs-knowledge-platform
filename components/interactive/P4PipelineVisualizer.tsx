"use client";
import { useState } from "react";

const parserStates = [
  { name: "start", transitions: [{ hdr: "ethernet", next: "parse_ethernet" }] },
  { name: "parse_ethernet", transitions: [{ hdr: "eth_type == 0x0800", next: "parse_ipv4" }, { hdr: "eth_type == 0x0806", next: "parse_arp" }] },
  { name: "parse_ipv4", transitions: [{ hdr: "ip_proto == 6", next: "parse_tcp" }, { hdr: "ip_proto == 17", next: "parse_udp" }] },
  { name: "parse_tcp", transitions: [{ hdr: "tcp_dst == 80", next: "parse_http" }, { hdr: "default", next: "accept" }] },
  { name: "parse_udp", transitions: [{ hdr: "default", next: "accept" }] },
  { name: "parse_arp", transitions: [{ hdr: "default", next: "accept" }] },
  { name: "parse_http", transitions: [{ hdr: "default", next: "accept" }] },
  { name: "accept", transitions: [] },
];

const matchTables = [
  { name: "ipv4_lpm", key: "dst_addr", action: "set_nhop", size: 1024 },
  { name: "acl", key: "src/dst/port", action: "permit/deny", size: 512 },
  { name: "forwarding", key: "egress_port", action: "output", size: 256 },
];

export function P4PipelineVisualizer() {
  const [activeState, setActiveState] = useState(0);
  const [activeTable, setActiveTable] = useState<number | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">⚡ P4 管线可视化</h3>
      <p className="text-sm text-text-secondary mb-4">展示可编程数据面的解析和匹配-动作</p>

      <div className="mb-4">
        <div className="text-sm font-medium text-text-secondary mb-2">解析器状态机 (Parser)</div>
        <div className="flex flex-wrap gap-1.5">
          {parserStates.map((s, i) => (
            <button key={s.name} onClick={() => setActiveState(i)}
              className={`px-2.5 py-1.5 rounded text-xs font-mono ${activeState === i ? "bg-blue-600 text-white ring-2 ring-blue-400" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
              {s.name}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-bg-surface rounded-lg p-3 mb-4 border border-border-subtle">
        <div className="font-mono text-sm text-text-primary mb-1">state {parserStates[activeState].name}</div>
        {parserStates[activeState].transitions.length > 0 ? (
          <div className="space-y-1">
            {parserStates[activeState].transitions.map((t, i) => (
              <div key={i} className="text-xs">
                <span className="text-yellow-400">extract({t.hdr})</span>
                <span className="text-text-secondary"> → </span>
                <span className="text-green-400">{t.next}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-xs text-green-400">accept — 解析完成，进入匹配-动作</div>
        )}
      </div>

      <div className="mb-4">
        <div className="text-sm font-medium text-text-secondary mb-2">匹配-动作表 (Match-Action)</div>
        <div className="space-y-1.5">
          {matchTables.map((t, i) => (
            <button key={t.name} onClick={() => setActiveTable(activeTable === i ? null : i)}
              className={`w-full flex items-center justify-between p-3 rounded-lg text-sm transition-all ${activeTable === i ? "bg-green-900/30 border border-green-600" : "bg-bg-surface border border-border-subtle hover:border-green-400"}`}>
              <div className="flex items-center gap-3">
                <span className="font-mono text-green-400">{t.name}</span>
                <span className="text-xs text-text-secondary">key: {t.key}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-text-secondary">{t.size} entries</span>
                <span className="font-mono text-xs text-blue-300">{t.action}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {activeTable !== null && (
        <div className="bg-bg-surface rounded-lg p-3 border border-border-subtle">
          <div className="font-mono text-sm text-text-primary mb-1">table {matchTables[activeTable].name}</div>
          <div className="text-xs text-text-secondary">
            Match key: {matchTables[activeTable].key} | Action: {matchTables[activeTable].action} | Max entries: {matchTables[activeTable].size}
          </div>
        </div>
      )}
    </div>
  );
}
export default P4PipelineVisualizer;
