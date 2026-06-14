"use client";
import { useState } from "react";

export function HubVsSwitch() {
  const [device, setDevice] = useState<"hub" | "switch">("hub");
  const [sender, setSender] = useState(0);
  const [receiver, setReceiver] = useState(1);
  const [showFrame, setShowFrame] = useState(false);

  const hosts = ["主机A", "主机B", "主机C", "主机D"];
  const hostColors = ["bg-blue-500", "bg-green-500", "bg-yellow-500", "bg-red-500"];

  const send = () => {
    setShowFrame(true);
    setTimeout(() => setShowFrame(false), 2000);
  };

  const getReceivers = () => {
    if (device === "hub") return [0, 1, 2, 3].filter((i) => i !== sender);
    return [receiver];
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">集线器(Hub) vs 交换机(Switch)</h3>
      <div className="flex gap-3 mb-4">
        <button onClick={() => setDevice("hub")}
          className={`px-4 py-2 rounded text-sm ${device === "hub" ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>Hub (广播)</button>
        <button onClick={() => setDevice("switch")}
          className={`px-4 py-2 rounded text-sm ${device === "switch" ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>Switch (单播)</button>
      </div>
      <svg width={400} height={200} className="mb-4 bg-bg-muted rounded-lg">
        <rect x={160} y={80} width={80} height={40} rx={5} fill={device === "hub" ? "#3b82f6" : "#22c55e"} opacity={0.3} stroke={device === "hub" ? "#3b82f6" : "#22c55e"} strokeWidth={2} />
        <text x={200} y={105} textAnchor="middle" fill="#1f2937" fontSize={12} fontWeight="bold">{device === "hub" ? "Hub" : "Switch"}</text>
        {hosts.map((h, i) => {
          const x = 50 + i * 100;
          const y = i % 2 === 0 ? 30 : 170;
          const isSender = i === sender;
          const isReceiver = showFrame && getReceivers().includes(i);
          return (
            <g key={i}>
              <line x1={x} y1={y} x2={200} y2={100} stroke={isReceiver ? "#ef4444" : isSender ? "#3b82f6" : "#9ca3af"} strokeWidth={isSender || isReceiver ? 2 : 1} />
              <circle cx={x} cy={y} r={20} fill={hostColors[i]} opacity={isSender ? 1 : isReceiver ? 0.8 : 0.5} stroke={isSender ? "#1d4ed8" : "none"} strokeWidth={2} />
              <text x={x} y={y + 5} textAnchor="middle" fill="white" fontSize={10}>{h}</text>
              {isSender && <text x={x} y={y - 25} textAnchor="middle" fill="#3b82f6" fontSize={10}>发送</text>}
              {isReceiver && <text x={x} y={y - 25} textAnchor="middle" fill="#ef4444" fontSize={10}>接收</text>}
            </g>
          );
        })}
        {showFrame && (
          <g>
            <rect x={170} y={55} width={60} height={18} rx={3} fill="#fbbf24" />
            <text x={200} y={68} textAnchor="middle" fill="#1f2937" fontSize={9}>数据帧</text>
          </g>
        )}
      </svg>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary mb-1 block">发送方:</label>
          <div className="flex gap-1">
            {hosts.map((h, i) => (
              <button key={i} onClick={() => setSender(i)}
                className={`px-2 py-1 rounded text-xs ${sender === i ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>{h}</button>
            ))}
          </div>
        </div>
        {device === "switch" && (
          <div>
            <label className="text-sm text-text-secondary mb-1 block">目的方:</label>
            <div className="flex gap-1">
              {hosts.map((h, i) => i !== sender && (
                <button key={i} onClick={() => setReceiver(i)}
                  className={`px-2 py-1 rounded text-xs ${receiver === i ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>{h}</button>
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="flex gap-3 mb-4">
        <button onClick={send} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm">发送帧</button>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs text-text-secondary">
        <div className="p-2 bg-bg-muted rounded"><strong>Hub:</strong> 物理层设备,收到帧后向所有端口广播(除输入端口),冲突域大</div>
        <div className="p-2 bg-bg-muted rounded"><strong>Switch:</strong> 数据链路层设备,学习MAC地址表,只向目的端口转发,隔离冲突域</div>
      </div>
    </div>
  );
}

export default HubVsSwitch;
