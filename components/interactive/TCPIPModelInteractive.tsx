"use client";
import { useState } from "react";

const layers = [
  {
    id: "app", name: "应用层 (Application)", color: "purple",
    protocols: ["HTTP/HTTPS", "FTP", "SMTP", "DNS", "SSH", "DHCP"],
    desc: "为应用进程提供网络服务接口。HTTP 用于 Web，DNS 解析域名，SMTP 收发邮件。",
    dataUnit: "数据 (Data)",
  },
  {
    id: "transport", name: "传输层 (Transport)", color: "blue",
    protocols: ["TCP", "UDP", "SCTP", "QUIC"],
    desc: "提供端到端通信。TCP 可靠有序，UDP 尽力交付。端口号标识进程。",
    dataUnit: "段 (Segment) / 数据报 (Datagram)",
  },
  {
    id: "network", name: "网络层 (Internet / Network)", color: "green",
    protocols: ["IPv4", "IPv6", "ICMP", "ARP", "OSPF", "BGP"],
    desc: "负责路由和转发。IP 地址标识主机，路由器根据路由表转发分组。",
    dataUnit: "分组 / 包 (Packet)",
  },
  {
    id: "link", name: "链路层 (Link / Network Access)", color: "red",
    protocols: ["Ethernet", "WiFi (802.11)", "PPP", "ARP"],
    desc: "在相邻节点间传输帧。MAC 地址标识网卡，差错检测 (CRC)。",
    dataUnit: "帧 (Frame)",
  },
];

export function TCPIPModelInteractive() {
  const [active, setActive] = useState<string | null>(null);

  const colorMap: Record<string, { bg: string; border: string; text: string }> = {
    purple: { bg: "bg-purple-500/10", border: "border-purple-400", text: "text-purple-400" },
    blue: { bg: "bg-blue-500/10", border: "border-blue-400", text: "text-blue-400" },
    green: { bg: "bg-green-500/10", border: "border-green-400", text: "text-green-400" },
    red: { bg: "bg-red-500/10", border: "border-red-400", text: "text-red-400" },
  };

  const osiMapping = [
    { tcpip: "应用层", osi: "应用层 + 表示层 + 会话层", note: "TCP/IP 合并了 OSI 上三层" },
    { tcpip: "传输层", osi: "传输层", note: "功能基本一致" },
    { tcpip: "网络层", osi: "网络层", note: "IP 对应 OSI 网络层" },
    { tcpip: "链路层", osi: "数据链路层 + 物理层", note: "TCP/IP 不严格区分" },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCP/IP 四层模型 (TCP/IP Model)</h3>
      <div className="flex flex-col items-center gap-1 mb-4">
        {layers.map((layer, i) => {
          const c = colorMap[layer.color];
          const isActive = active === layer.id;
          return (
            <div key={layer.id} className="w-full max-w-lg">
              <button onClick={() => setActive(isActive ? null : layer.id)}
                className={`w-full p-4 rounded-lg border-2 transition-all cursor-pointer ${c.bg} ${isActive ? c.border + " shadow-lg scale-[1.01]" : "border-border-subtle hover:border-gray-400"}`}>
                <div className="flex items-center justify-between">
                  <span className={`font-semibold text-base ${c.text}`}>{layer.name}</span>
                  <span className="text-text-muted text-xs">{layer.dataUnit}</span>
                </div>
              </button>
              {i < layers.length - 1 && (
                <div className="flex justify-center py-0.5">
                  <svg width="20" height="16" viewBox="0 0 20 16"><path d="M10 0 L10 10 M5 7 L10 12 L15 7" stroke="currentColor" strokeWidth="1.5" fill="none" className="text-text-muted" /></svg>
                </div>
              )}
              {isActive && (
                <div className={`mt-1 p-4 rounded-lg ${c.bg} border ${c.border}`}>
                  <p className="text-text-primary text-sm mb-2">{layer.desc}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {layer.protocols.map((p) => (
                      <span key={p} className={`text-xs px-2 py-0.5 rounded ${c.bg} ${c.text} border ${c.border}`}>{p}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle mb-4">
        <h4 className="text-text-primary text-xs font-medium mb-2">TCP/IP vs OSI 映射</h4>
        <div className="grid grid-cols-3 gap-1 text-xs">
          <span className="text-text-muted font-medium">TCP/IP</span>
          <span className="text-text-muted font-medium">OSI</span>
          <span className="text-text-muted font-medium">说明</span>
          {osiMapping.map((m) => (
            <><span className="text-text-primary">{m.tcpip}</span><span className="text-text-secondary">{m.osi}</span><span className="text-text-muted">{m.note}</span></>
          ))}
        </div>
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <p className="text-text-muted text-xs">TCP/IP 模型是互联网实际使用的协议栈。封装过程：数据 → 加传输层头 → 加 IP 头 → 加帧头帧尾 → 物理信号。每层只处理自己的头部。</p>
      </div>
    </div>
  );
}
export default TCPIPModelInteractive;
