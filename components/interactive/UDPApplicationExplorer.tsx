"use client";
import { useState } from "react";

const apps = [
  {
    id: "dns", name: "DNS", port: 53, icon: "🌐",
    reason: "请求/响应模式简单，低延迟要求高于可靠性",
    format: [{ field: "Transaction ID", size: "16 bit", desc: "查询标识符" }, { field: "Flags", size: "16 bit", desc: "QR/Opcode/AA/TC/RD/RA" }, { field: "Questions", size: "16 bit", desc: "查询数量" }, { field: "Answer RRs", size: "16 bit", desc: "回答数量" }, { field: "Query", size: "变长", desc: "域名 + 类型 + 类" }],
    example: "客户端 → DNS服务器: 查询 www.example.com A记录\nDNS服务器 → 客户端: 93.184.216.34",
  },
  {
    id: "rtp", name: "RTP (实时传输)", port: "16384-32767", icon: "🎬",
    reason: "实时性要求高，丢包可容忍（播放卡顿比延迟好）",
    format: [{ field: "V/P/X/CC", size: "8 bit", desc: "版本/填充/扩展/CSRC计数" }, { field: "M/PT", size: "8 bit", desc: "标记/载荷类型" }, { field: "Sequence", size: "16 bit", desc: "序列号" }, { field: "Timestamp", size: "32 bit", desc: "采样时间戳" }, { field: "SSRC", size: "32 bit", desc: "同步源标识" }],
    example: "视频流: RTP包 → [seq=100, ts=16000, payload=视频帧数据]\n接收端: 按时间戳排序播放，丢包不重传",
  },
  {
    id: "dhcp", name: "DHCP", port: "67/68", icon: "📡",
    reason: "客户端启动时无IP，无法建立TCP连接",
    format: [{ field: "Op", size: "8 bit", desc: "1=请求, 2=回复" }, { field: "Htype/Hlen", size: "16 bit", desc: "硬件类型/地址长度" }, { field: "XID", size: "32 bit", desc: "事务ID" }, { field: "YIADDR", size: "32 bit", desc: "分配的IP地址" }, { field: "Options", size: "变长", desc: "子网掩码/网关/DNS等" }],
    example: "Discover (广播) → Offer → Request → ACK\n四步获取IP地址、子网掩码、网关、DNS",
  },
  {
    id: "game", name: "在线游戏", port: "动态", icon: "🎮",
    reason: "低延迟关键，丢包用最新状态覆盖（不重传旧数据）",
    format: [{ field: "MsgType", size: "8 bit", desc: "消息类型" }, { field: "PlayerID", size: "16 bit", desc: "玩家ID" }, { field: "SeqNum", size: "16 bit", desc: "序列号" }, { field: "Position", size: "48 bit", desc: "X/Y/Z坐标" }, { field: "Action", size: "8 bit", desc: "动作编码" }],
    example: "客户端 → 服务器: [移动到(100,200)] 每秒30-60次\n服务器 → 所有客户端: 广播游戏状态",
  },
];

export function UDPApplicationExplorer() {
  const [active, setActive] = useState(0);
  const app = apps[active];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">UDP 应用探索器</h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {apps.map((a, i) => (
          <button key={a.id} onClick={() => setActive(i)}
            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
              active === i ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted hover:text-text-primary"
            }`}>
            {a.icon} {a.name}
          </button>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-text-primary font-medium">{app.icon} {app.name}</h4>
          <span className="text-text-muted text-xs">端口: {app.port}</span>
        </div>
        <div className="p-2 rounded bg-green-500/10 border border-green-400/30 mb-3">
          <span className="text-green-400 text-xs">为什么用UDP: {app.reason}</span>
        </div>
        <h5 className="text-text-secondary text-xs font-medium mb-2">报文格式</h5>
        <div className="space-y-1 mb-3">
          {app.format.map((f) => (
            <div key={f.field} className="flex items-center gap-3 p-2 rounded bg-bg-elevated border border-border-subtle">
              <span className="text-blue-400 text-xs font-mono w-28 flex-shrink-0">{f.field}</span>
              <span className="text-text-muted text-xs w-16 flex-shrink-0">{f.size}</span>
              <span className="text-text-secondary text-xs">{f.desc}</span>
            </div>
          ))}
        </div>
        <h5 className="text-text-secondary text-xs font-medium mb-1">通信示例</h5>
        <pre className="p-2 rounded bg-bg-elevated border border-border-subtle text-text-primary text-xs whitespace-pre-wrap font-mono">{app.example}</pre>
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-secondary text-xs font-medium mb-1">UDP 数据报结构 (8字节头部)</h4>
        <div className="flex gap-px">
          {["源端口 (16bit)", "目的端口 (16bit)", "长度 (16bit)", "校验和 (16bit)", "数据 Payload..."].map((f, i) => (
            <div key={f} className="flex-1 p-1.5 rounded bg-blue-500/10 border border-blue-400/30 text-center"
              style={{ flex: i === 4 ? 3 : 1 }}>
              <span className="text-blue-400 text-[10px]">{f}</span>
            </div>
          ))}
        </div>
        <p className="text-text-muted text-xs mt-2">UDP 头部仅 8 字节（源端口 + 目的端口 + 长度 + 校验和），比 TCP 的 20 字节精简得多。无连接、不可靠、保留消息边界。</p>
      </div>
    </div>
  );
}
export default UDPApplicationExplorer;
