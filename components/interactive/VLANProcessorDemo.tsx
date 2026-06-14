"use client";
import { useState } from "react";

export function VLANProcessorDemo() {
  const [vlanId, setVlanId] = useState(100);
  const [priority, setPriority] = useState(0);
  const [mode, setMode] = useState<"tag" | "untag">("tag");
  const [animate, setAnimate] = useState(false);

  const frame = { dstMAC: "FF:FF:FF:FF:FF:FF", srcMAC: "00:1A:2B:3C:4D:5E", type: "0x0800", payload: "...数据...", fcs: "CRC-32" };
  const vlanTag = { tpid: "0x8100", priority, cfi: 0, vlanId };

  const doAction = () => {
    setAnimate(true);
    setTimeout(() => setAnimate(false), 1500);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">VLAN 处理器演示 (802.1Q)</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-text-muted text-xs block mb-1">VLAN ID (0-4095)</label>
          <input type="number" min="0" max="4095" value={vlanId} onChange={(e) => setVlanId(Number(e.target.value))}
            className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm" />
        </div>
        <div>
          <label className="text-text-muted text-xs block mb-1">优先级 (0-7)</label>
          <input type="number" min="0" max="7" value={priority} onChange={(e) => setPriority(Number(e.target.value))}
            className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm" />
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => { setMode("tag"); doAction(); }}
          className={`px-4 py-2 rounded text-sm font-medium ${mode === "tag" ? "bg-green-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          插入标签 (Tag)
        </button>
        <button onClick={() => { setMode("untag"); doAction(); }}
          className={`px-4 py-2 rounded text-sm font-medium ${mode === "untag" ? "bg-red-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          剥离标签 (Untag)
        </button>
      </div>
      <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <h4 className="text-text-secondary text-xs font-medium mb-2">{mode === "tag" ? "插入 802.1Q 标签" : "剥离 802.1Q 标签"}</h4>
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          {mode === "tag" ? (
            <>
              <div className={`p-2 rounded border border-blue-400/30 bg-blue-500/10 text-center transition-all ${animate ? "scale-95 opacity-50" : ""}`}>
                <span className="text-blue-400 text-[10px] block">原始帧</span>
                <span className="text-text-primary text-xs font-mono">{frame.srcMAC}</span>
              </div>
              <svg width="30" height="20"><path d="M0 10 L20 10 M15 5 L20 10 L15 15" stroke="#4ade80" strokeWidth="2" fill="none" /></svg>
              <div className={`p-2 rounded border border-green-400/30 bg-green-500/10 text-center transition-all ${animate ? "scale-105" : ""}`}>
                <span className="text-green-400 text-[10px] block">802.1Q 标签</span>
                <span className="text-text-primary text-[10px] font-mono block">TPID: {vlanTag.tpid}</span>
                <span className="text-text-primary text-[10px] font-mono block">PRI:{priority} CFI:{vlanTag.cfi} VID:{vlanId}</span>
              </div>
              <svg width="30" height="20"><path d="M0 10 L20 10 M15 5 L20 10 L15 15" stroke="#facc15" strokeWidth="2" fill="none" /></svg>
              <div className={`p-2 rounded border border-yellow-400/30 bg-yellow-500/10 text-center`}>
                <span className="text-yellow-400 text-[10px] block">带标签帧</span>
                <span className="text-text-primary text-[10px] font-mono">完整802.1Q帧</span>
              </div>
            </>
          ) : (
            <>
              <div className={`p-2 rounded border border-yellow-400/30 bg-yellow-500/10 text-center transition-all ${animate ? "scale-95 opacity-50" : ""}`}>
                <span className="text-yellow-400 text-[10px] block">带标签帧</span>
                <span className="text-text-primary text-[10px] font-mono">VID:{vlanId}</span>
              </div>
              <svg width="30" height="20"><path d="M0 10 L20 10 M15 5 L20 10 L15 15" stroke="#f87171" strokeWidth="2" fill="none" /></svg>
              <div className={`p-2 rounded border border-red-400/30 bg-red-500/10 text-center transition-all ${animate ? "scale-105" : ""}`}>
                <span className="text-red-400 text-[10px] block">剥离标签</span>
                <span className="text-text-primary text-[10px] font-mono">移除4字节</span>
              </div>
              <svg width="30" height="20"><path d="M0 10 L20 10 M15 5 L20 10 L15 15" stroke="#60a5fa" strokeWidth="2" fill="none" /></svg>
              <div className={`p-2 rounded border border-blue-400/30 bg-blue-500/10 text-center`}>
                <span className="text-blue-400 text-[10px] block">原始帧</span>
                <span className="text-text-primary text-[10px] font-mono">标准以太帧</span>
              </div>
            </>
          )}
        </div>
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-secondary text-xs font-medium mb-1">802.1Q 标签结构 (4 字节)</h4>
        <div className="flex gap-px">
          {["TPID (0x8100)", "Priority (3bit)", "CFI (1bit)", "VLAN ID (12bit)"].map((f) => (
            <div key={f} className="flex-1 p-1.5 rounded bg-purple-500/10 border border-purple-400/30 text-center">
              <span className="text-purple-400 text-[10px]">{f}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
export default VLANProcessorDemo;
