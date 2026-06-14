"use client";
import { useState } from "react";

type Feature = "dma" | "interrupt" | "checksum";

const featureInfo: Record<Feature, { title: string; desc: string; detail: string; color: string }> = {
  dma: { title: "DMA (Direct Memory Access)", desc: "网卡直接读写内存，无需 CPU 逐字节搬运", detail: "DMA 引擎通过总线主控（Bus Master）将数据从网卡缓冲区直接写入主机内存的环形缓冲区（Ring Buffer），CPU 仅需设置描述符基地址和长度。零拷贝（Zero-Copy）技术进一步减少内存复制。", color: "bg-blue-500" },
  interrupt: { title: "中断 (Interrupt)", desc: "数据到达后网卡向 CPU 发出中断通知", detail: "传统每包一中断开销大，现代 NIC 使用中断合并（Interrupt Coalescing）：累积多个包后触发一次 MSI-X 中断。NAPI 机制在高负载时切换为轮询模式（polling），避免中断风暴。", color: "bg-green-500" },
  checksum: { title: "校验和卸载 (Checksum Offload)", desc: "硬件计算 IP/TCP/UDP 校验和，释放 CPU", detail: "TX 方向：协议栈填 0 校验和字段，NIC 硬件计算后插入。RX 方向：NIC 验证校验和并标记结果到接收描述符，协议栈直接读取。支持 TCP Segmentation Offload (TSO) 和 Large Receive Offload (LRO)。", color: "bg-purple-500" },
};

export function NICArchitecture() {
  const [active, setActive] = useState<Feature>("dma");
  const [packetFlow, setPacketFlow] = useState(false);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">网卡架构 — DMA / 中断 / 校验和卸载</h3>
      <div className="flex gap-2 mb-4">
        {(Object.keys(featureInfo) as Feature[]).map((f) => (
          <button key={f} onClick={() => setActive(f)} className={`px-3 py-1.5 rounded text-sm ${active === f ? featureInfo[f].color + " text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>{featureInfo[f].title.split(" ")[0]}</button>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle mb-4">
        <h4 className="font-semibold text-text-primary mb-2">{featureInfo[active].title}</h4>
        <p className="text-sm text-text-secondary mb-2">{featureInfo[active].desc}</p>
        <p className="text-xs text-text-secondary">{featureInfo[active].detail}</p>
      </div>
      <div className="mb-4">
        <button onClick={() => setPacketFlow(!packetFlow)} className="px-3 py-1.5 rounded bg-blue-500 text-white text-sm">{packetFlow ? "停止" : "模拟"} 数据包收发</button>
      </div>
      {packetFlow && (
        <div className="flex items-center gap-2 text-xs mb-2">
          {["应用层", "协议栈", "Ring Buffer", "NIC DMA", "网线"].map((node, i) => (
            <span key={i} className="flex items-center gap-1">
              <span className="px-2 py-1 rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">{node}</span>
              {i < 4 && <span>→</span>}
            </span>
          ))}
        </div>
      )}
      <div className="grid grid-cols-3 gap-2 text-xs mt-4">
        {[{ label: "RX Ring", value: "256/1024 描述符" }, { label: "TX Ring", value: "256/1024 描述符" }, { label: "中断合并", value: "每 8 包或 50μs" }].map((s, i) => (
          <div key={i} className="p-2 rounded bg-gray-100 dark:bg-gray-800 text-center"><span className="block font-medium text-text-primary">{s.label}</span><span className="text-text-secondary">{s.value}</span></div>
        ))}
      </div>
    </div>
  );
}
export default NICArchitecture;
