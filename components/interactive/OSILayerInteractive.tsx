"use client";
import { useState } from "react";

const layers = [
  { num: 7, name: "应用层", en: "Application", color: "bg-red-600", protocols: "HTTP, FTP, SMTP, DNS, SSH", pdu: "数据 (Data)", devices: "应用网关", func: "为用户应用提供网络服务接口，处理应用协议" },
  { num: 6, name: "表示层", en: "Presentation", color: "bg-orange-600", protocols: "SSL/TLS, JPEG, ASCII, MPEG, XDR", pdu: "数据 (Data)", devices: "—", func: "数据格式转换、加密/解密、压缩/解压" },
  { num: 5, name: "会话层", en: "Session", color: "bg-yellow-600", protocols: "RPC, NetBIOS, PPTP, SQL", pdu: "数据 (Data)", devices: "—", func: "建立、管理和终止会话，同步通信" },
  { num: 4, name: "传输层", en: "Transport", color: "bg-green-600", protocols: "TCP, UDP, SCTP, QUIC", pdu: "段/数据报", devices: "四层交换机", func: "端到端可靠传输、流量控制、拥塞控制" },
  { num: 3, name: "网络层", en: "Network", color: "bg-blue-600", protocols: "IP, ICMP, ARP, OSPF, BGP", pdu: "分组 (Packet)", devices: "路由器", func: "逻辑寻址、路由选择、分组转发" },
  { num: 2, name: "数据链路层", en: "Data Link", color: "bg-indigo-600", protocols: "Ethernet, Wi-Fi, PPP, HDLC", pdu: "帧 (Frame)", devices: "交换机", func: "物理寻址、差错检测、帧同步" },
  { num: 1, name: "物理层", en: "Physical", color: "bg-purple-600", protocols: "RS-232, RJ45, 802.11 PHY", pdu: "比特 (Bits)", devices: "集线器", func: "比特传输、电气/光信号规范" },
];

export function OSILayerInteractive() {
  const [selected, setSelected] = useState<number | null>(null);
  const [showEncap, setShowEncap] = useState(false);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">🏗️ OSI 七层交互模型</h3>
      <p className="text-sm text-text-secondary mb-4">点击各层查看功能和协议</p>

      <button onClick={() => setShowEncap(!showEncap)}
        className="mb-4 px-3 py-1.5 bg-bg-surface border border-border-subtle rounded text-sm text-text-secondary hover:border-blue-400">
        {showEncap ? "隐藏封装视图" : "显示封装视图"}
      </button>

      <div className="space-y-1 mb-4">
        {layers.map((l) => (
          <div key={l.num} onClick={() => setSelected(selected === l.num ? null : l.num)}
            className={`${l.color} rounded-lg p-3 cursor-pointer transition-all flex items-center justify-between ${selected === l.num ? "ring-2 ring-white scale-[1.02]" : "opacity-80 hover:opacity-100"}`}>
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center text-white font-bold text-sm">{l.num}</span>
              <div>
                <span className="text-white font-medium">{l.name}</span>
                <span className="text-white/60 text-xs ml-2">{l.en}</span>
              </div>
            </div>
            {showEncap && (
              <span className="text-white/80 text-xs font-mono bg-white/10 px-2 py-0.5 rounded">{l.pdu}</span>
            )}
          </div>
        ))}
      </div>

      {selected !== null && (() => {
        const l = layers.find(x => x.num === selected)!;
        return (
          <div className="bg-bg-surface rounded-lg p-4 border border-border-subtle">
            <div className="flex items-center gap-2 mb-3">
              <div className={`w-5 h-5 rounded ${l.color}`} />
              <span className="font-semibold text-text-primary">第 {l.num} 层: {l.name} ({l.en})</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
              <div><span className="text-text-secondary">功能：</span><span className="text-text-primary">{l.func}</span></div>
              <div><span className="text-text-secondary">PDU：</span><span className="font-mono text-text-primary">{l.pdu}</span></div>
              <div><span className="text-text-secondary">协议：</span><span className="font-mono text-blue-300">{l.protocols}</span></div>
              <div><span className="text-text-secondary">设备：</span><span className="text-text-primary">{l.devices}</span></div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
export default OSILayerInteractive;
