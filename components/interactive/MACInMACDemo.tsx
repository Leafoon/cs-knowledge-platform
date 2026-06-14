"use client";
import { useState } from "react";

const outerFields = [
  { name: "Outer DMAC", value: "00:1A:2B:3C:4D:5E", desc: "运营商目的MAC" },
  { name: "Outer SMAC", value: "00:AA:BB:CC:DD:EE", desc: "运营商源MAC" },
  { name: "EtherType", value: "0x88E7", desc: "Provider Backbone Bridge" },
  { name: "B-Tag (VLAN)", value: "100", desc: "骨干网VLAN标签" },
  { name: "I-SID", value: "0x000100", desc: "服务实例标识" },
];

const innerFields = [
  { name: "Inner DMAC", value: "11:22:33:44:55:66", desc: "客户目的MAC" },
  { name: "Inner SMAC", value: "77:88:99:AA:BB:CC", desc: "客户源MAC" },
  { name: "EtherType", value: "0x0800", desc: "IPv4" },
  { name: "C-Tag (VLAN)", value: "200", desc: "客户VLAN标签" },
];

export function MACInMACDemo() {
  const [showOuter, setShowOuter] = useState(true);
  const [showInner, setShowInner] = useState(true);
  const [selectedField, setSelectedField] = useState<string | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">🔀 MAC-in-MAC 演示</h3>
      <p className="text-sm text-text-secondary mb-4">展示运营商以太网的双层 MAC 封装 (IEEE 802.1ah)</p>

      <div className="flex gap-3 mb-4">
        <button onClick={() => setShowOuter(!showOuter)}
          className={`px-3 py-1.5 rounded text-sm ${showOuter ? "bg-orange-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary"}`}>
          外层 MAC {showOuter ? "✓" : ""}
        </button>
        <button onClick={() => setShowInner(!showInner)}
          className={`px-3 py-1.5 rounded text-sm ${showInner ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary"}`}>
          内层 MAC {showInner ? "✓" : ""}
        </button>
      </div>

      <div className="flex flex-col gap-2 mb-4">
        {showOuter && (
          <div className="border-2 border-orange-500 rounded-lg p-3">
            <div className="text-sm font-semibold text-orange-400 mb-2">运营商以太网帧头 (Provider Backbone Bridge)</div>
            <div className="flex flex-wrap gap-1.5">
              {outerFields.map(f => (
                <button key={f.name} onClick={() => setSelectedField(selectedField === f.name ? null : f.name)}
                  className={`px-2.5 py-1.5 rounded text-xs font-mono transition-all ${selectedField === f.name ? "bg-orange-600 text-white scale-105" : "bg-orange-900/30 text-orange-300 border border-orange-700 hover:bg-orange-800/40"}`}>
                  {f.name}
                </button>
              ))}
            </div>
          </div>
        )}
        {showInner && (
          <div className="border-2 border-blue-500 rounded-lg p-3">
            <div className="text-sm font-semibold text-blue-400 mb-2">客户以太网帧头 (Customer Frame)</div>
            <div className="flex flex-wrap gap-1.5">
              {innerFields.map(f => (
                <button key={f.name} onClick={() => setSelectedField(selectedField === f.name ? null : f.name)}
                  className={`px-2.5 py-1.5 rounded text-xs font-mono transition-all ${selectedField === f.name ? "bg-blue-600 text-white scale-105" : "bg-blue-900/30 text-blue-300 border border-blue-700 hover:bg-blue-800/40"}`}>
                  {f.name}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="border border-border-subtle rounded-lg p-3 bg-gray-800/50">
          <span className="text-sm text-text-secondary">Payload (数据载荷)</span>
        </div>
      </div>

      {selectedField && (() => {
        const field = [...outerFields, ...innerFields].find(f => f.name === selectedField);
        if (!field) return null;
        return (
          <div className="bg-bg-surface rounded-lg p-4 border border-border-subtle">
            <div className="font-mono text-lg text-text-primary mb-1">{field.name}</div>
            <div className="font-mono text-sm text-blue-300 mb-1">值: {field.value}</div>
            <div className="text-sm text-text-secondary">{field.desc}</div>
          </div>
        );
      })()}

      <div className="mt-4 bg-bg-surface rounded-lg p-3 text-sm text-text-secondary">
        <strong className="text-text-primary">原理：</strong>运营商在客户帧外再封装一层 MAC 头，实现骨干网与客户网络的隔离。B-Tag 标识骨干网 VLAN，I-SID 标识服务实例，支持 16M 个服务实例。
      </div>
    </div>
  );
}
export default MACInMACDemo;
