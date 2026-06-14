"use client";
import { useState } from "react";

interface LayerFunction {
  name: string;
  ename: string;
  desc: string;
  protocols: string[];
  pdu: string;
  services: string[];
}

const functions: LayerFunction[] = [
  {
    name: "成帧", ename: "Framing",
    desc: "将网络层数据报封装为帧，添加帧头和帧尾，界定帧边界。",
    protocols: ["Ethernet", "PPP", "HDLC"],
    pdu: "帧 (Frame)",
    services: ["帧定界", "帧同步", "帧头/帧尾添加"],
  },
  {
    name: "差错检测", ename: "Error Detection",
    desc: "通过CRC校验检测传输过程中发生的比特错误。",
    protocols: ["CRC-32 (Ethernet)", "CRC-16 (HDLC)", "校验和"],
    pdu: "FCS字段",
    services: ["CRC计算", "奇偶校验", "校验和"],
  },
  {
    name: "可靠传输", ename: "Reliable Delivery",
    desc: "通过ACK/NAK和重传机制保证帧的可靠交付(可选)。",
    protocols: ["停等协议", "Go-Back-N", "选择重传"],
    pdu: "ACK/NAK",
    services: ["确认机制", "超时重传", "序列号"],
  },
  {
    name: "媒体访问控制", ename: "MAC",
    desc: "控制多个节点共享同一传输媒体时的访问权限。",
    protocols: ["CSMA/CD", "CSMA/CA", "TDMA", "ALOHA"],
    pdu: "MAC帧",
    services: ["信道争用", "冲突检测/避免", "退避算法"],
  },
  {
    name: "链路寻址", ename: "Link Addressing",
    desc: "使用MAC地址在同一链路上标识发送方和接收方。",
    protocols: ["MAC (48-bit)", "EUI-64"],
    pdu: "MAC地址",
    services: ["源/目的MAC", "广播/组播"],
  },
  {
    name: "流量控制", ename: "Flow Control",
    desc: "防止发送方发送速率超过接收方处理能力。",
    protocols: ["停等", "滑动窗口"],
    pdu: "窗口字段",
    services: ["滑动窗口", "信用量控制"],
  },
];

export function DataLinkLayerOverview() {
  const [selected, setSelected] = useState<number | null>(null);
  const [showOSI, setShowOSI] = useState(false);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">数据链路层功能概览</h3>
      <button onClick={() => setShowOSI(!showOSI)}
        className="mb-4 px-3 py-1.5 rounded text-sm bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700">
        {showOSI ? "隐藏" : "显示"}OSI模型位置
      </button>
      {showOSI && (
        <div className="mb-4 flex flex-col items-center gap-1">
          {["应用层 L7", "表示层 L6", "会话层 L5", "传输层 L4", "网络层 L3", "数据链路层 L2 ←", "物理层 L1"].map((l, i) => (
            <div key={i} className={`w-full max-w-xs text-center py-1.5 px-3 rounded text-xs font-medium ${i === 5 ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
              {l}
            </div>
          ))}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-4">
        {functions.map((f, i) => (
          <button key={i} onClick={() => setSelected(selected === i ? null : i)}
            className={`p-3 rounded border text-left transition-all ${selected === i ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20" : "border-border-subtle bg-gray-50 dark:bg-gray-800 hover:border-blue-300"}`}>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-sm font-bold text-text-primary">{f.name}</span>
              <span className="text-[10px] text-text-secondary">{f.ename}</span>
            </div>
            <p className="text-[11px] text-text-secondary line-clamp-2">{f.desc}</p>
          </button>
        ))}
      </div>
      {selected !== null && (
        <div className="p-4 rounded bg-gray-50 dark:bg-gray-800 border border-border-subtle mb-4">
          <h4 className="text-sm font-bold text-text-primary mb-2">{functions[selected].name} ({functions[selected].ename})</h4>
          <p className="text-xs text-text-secondary mb-3">{functions[selected].desc}</p>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-xs font-medium text-text-primary mb-1">相关协议</p>
              <div className="flex flex-wrap gap-1">
                {functions[selected].protocols.map((p, i) => (
                  <span key={i} className="text-[10px] px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">{p}</span>
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs font-medium text-text-primary mb-1">提供的服务</p>
              <div className="flex flex-wrap gap-1">
                {functions[selected].services.map((s, i) => (
                  <span key={i} className="text-[10px] px-2 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">{s}</span>
                ))}
              </div>
            </div>
          </div>
          <div className="mt-2 text-xs text-text-secondary">PDU: <span className="font-mono text-text-primary">{functions[selected].pdu}</span></div>
        </div>
      )}
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800">
        <p className="text-xs font-medium text-text-primary mb-2">数据链路层子层</p>
        <div className="grid grid-cols-2 gap-2">
          <div className="p-2 rounded bg-white dark:bg-gray-900 border border-border-subtle">
            <div className="text-xs font-bold text-text-primary">LLC (逻辑链路控制)</div>
            <div className="text-[10px] text-text-secondary">IEEE 802.2 - 多路复用、差错控制</div>
          </div>
          <div className="p-2 rounded bg-white dark:bg-gray-900 border border-border-subtle">
            <div className="text-xs font-bold text-text-primary">MAC (媒体访问控制)</div>
            <div className="text-[10px] text-text-secondary">IEEE 802.3/802.11 - 信道访问、寻址</div>
          </div>
        </div>
      </div>
      <p className="text-xs text-text-secondary mt-3">数据链路层负责在相邻节点间可靠传输帧，提供成帧、差错检测、媒体访问控制等功能。</p>
    </div>
  );
}
export default DataLinkLayerOverview;
