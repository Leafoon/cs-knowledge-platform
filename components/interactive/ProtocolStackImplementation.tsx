"use client";
import { useState } from "react";

interface Layer {
  name: string;
  en: string;
  color: string;
  modules: string[];
  dataUnit: string;
}

const layers: Layer[] = [
  { name: "应用层", en: "Application", color: "bg-blue-500/15 border-blue-400/40 text-blue-700 dark:text-blue-300", modules: ["HTTP", "DNS", "SMTP", "FTP", "SSH"], dataUnit: "报文 Message" },
  { name: "传输层", en: "Transport", color: "bg-emerald-500/15 border-emerald-400/40 text-emerald-700 dark:text-emerald-300", modules: ["TCP", "UDP", "SCTP", "QUIC"], dataUnit: "段/数据报 Segment/Datagram" },
  { name: "网络层", en: "Network", color: "bg-amber-500/15 border-amber-400/40 text-amber-700 dark:text-amber-300", modules: ["IP", "ICMP", "ARP", "OSPF", "BGP"], dataUnit: "分组/包 Packet" },
  { name: "数据链路层", en: "Data Link", color: "bg-violet-500/15 border-violet-400/40 text-violet-700 dark:text-violet-300", modules: ["Ethernet", "PPP", "HDLC", "Wi-Fi"], dataUnit: "帧 Frame" },
  { name: "物理层", en: "Physical", color: "bg-rose-500/15 border-rose-400/40 text-rose-700 dark:text-rose-300", modules: ["RS-232", "DSL", "SONET", "802.11 PHY"], dataUnit: "比特 Bit" },
];

export function ProtocolStackImplementation() {
  const [activeLayer, setActiveLayer] = useState(0);
  const [showFlow, setShowFlow] = useState(false);
  const layer = layers[activeLayer];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">协议栈软件架构</h3>
      <div className="flex flex-col gap-1.5 mb-4">
        {layers.map((l, i) => (
          <button key={i} onClick={() => setActiveLayer(i)}
            className={`flex items-center gap-3 px-4 py-2.5 rounded-lg border text-left transition-all ${activeLayer === i ? l.color : "border-border-subtle bg-bg-tertiary text-text-secondary hover:text-text-primary"}`}>
            <span className="text-[10px] font-mono w-4 text-text-tertiary">{i + 1}</span>
            <span className="text-sm font-medium flex-1">{l.name}</span>
            <span className="text-[10px] opacity-60">{l.en}</span>
          </button>
        ))}
      </div>
      <div className={`rounded-lg border p-4 mb-4 ${layer.color}`}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-semibold">{layer.name}（{layer.en}）</span>
          <span className="text-[10px] opacity-70">数据单元: {layer.dataUnit}</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {layer.modules.map((m) => (
            <span key={m} className="px-2.5 py-1 rounded-lg bg-bg-elevated/60 border border-border-subtle text-xs font-mono">{m}</span>
          ))}
        </div>
      </div>
      <button onClick={() => setShowFlow(!showFlow)}
        className="w-full px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:text-text-primary transition-colors mb-3">
        {showFlow ? "隐藏" : "显示"}数据封装流程
      </button>
      {showFlow && (
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 space-y-1.5">
          {["应用层数据", "+ TCP/UDP 首部", "+ IP 首部", "+ 以太网帧头/帧尾", "→ 物理层比特流"].map((step, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-text-tertiary w-4">{i + 1}</span>
              <div className={`flex-1 px-3 py-1.5 rounded text-xs font-mono ${layers[Math.min(i, 4)].color}`}>
                {step}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="mt-3 text-[10px] text-text-tertiary">操作系统内核实现协议栈 · 每层通过系统调用或接口向上层提供服务</div>
    </div>
  );
}
export default ProtocolStackImplementation;
