"use client";
import { useState } from "react";

const LAYERS = [
  { name: "应用层", protocol: "HTTP/FTP/SMTP", pdu: "数据(Data)", color: "bg-blue-500", header: "应用首部" },
  { name: "传输层", protocol: "TCP/UDP", pdu: "段(Segment)", color: "bg-green-500", header: "TCP/UDP首部" },
  { name: "网络层", protocol: "IP", pdu: "数据报(Datagram)", color: "bg-yellow-500", header: "IP首部" },
  { name: "链路层", protocol: "Ethernet/WiFi", pdu: "帧(Frame)", color: "bg-red-500", header: "帧首部+尾部" },
  { name: "物理层", protocol: "光纤/铜线", pdu: "比特流(Bits)", color: "bg-purple-500", header: "" },
];

export function EncapsulationInteractive() {
  const [activeLayer, setActiveLayer] = useState(0);
  const [direction, setDirection] = useState<"down" | "up">("down");

  const current = LAYERS[activeLayer];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">数据封装过程: 应用→传输→网络→链路</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setDirection("down")}
          className={`px-3 py-1.5 rounded text-sm ${direction === "down" ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>发送(封装)</button>
        <button onClick={() => setDirection("up")}
          className={`px-3 py-1.5 rounded text-sm ${direction === "up" ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>接收(解封)</button>
      </div>
      <div className="space-y-2 mb-4">
        {(direction === "down" ? LAYERS : [...LAYERS].reverse()).map((l, i) => (
          <div
            key={l.name}
            onClick={() => setActiveLayer(direction === "down" ? i : LAYERS.length - 1 - i)}
            className={`cursor-pointer flex items-center gap-3 p-3 rounded-lg transition-all ${
              (direction === "down" ? i : LAYERS.length - 1 - i) === activeLayer
                ? `${l.color} text-white shadow-lg`
                : "bg-bg-muted text-text-secondary hover:bg-bg-subtle"
            }`}
          >
            <span className="w-16 text-xs font-bold">{l.name}</span>
            <span className="flex-1 text-xs font-mono">{l.pdu}</span>
            <span className="text-xs">{l.protocol}</span>
            {l.header && (
              <span className="text-xs px-2 py-0.5 bg-white/20 rounded">
                {direction === "down" ? "+" : "−"}{l.header}
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        <h4 className="font-semibold text-text-primary mb-2">{current.name} - {current.pdu}</h4>
        <div className="flex items-center gap-2 text-sm text-text-secondary mb-2">
          <span>协议: {current.protocol}</span>
          {current.header && <span>| 添加: {current.header}</span>}
        </div>
        <div className="font-mono text-xs bg-bg-subtle p-3 rounded overflow-x-auto">
          {direction === "down" ? (
            <span>
              {LAYERS.slice(0, activeLayer + 1).map((l, i) => (
                <span key={i}>
                  <span className={i === activeLayer ? "text-blue-500 font-bold" : "text-text-secondary"}>[{l.header || l.name}]</span>
                </span>
              ))}
              <span className="text-green-500">[数据]</span>
            </span>
          ) : (
            <span>
              {LAYERS.slice(activeLayer).map((l, i) => (
                <span key={i}>
                  <span className={i === 0 ? "text-blue-500 font-bold" : "text-text-secondary"}>[{l.header || l.name}]</span>
                </span>
              ))}
            </span>
          )}
        </div>
      </div>
      <div className="text-xs text-text-secondary">
        {direction === "down" ? "发送时自顶向下封装:每层添加自己的首部(有时加尾部)" : "接收时自底向上解封:每层移除对应首部,交给上层处理"}
      </div>
    </div>
  );
}

export default EncapsulationInteractive;
