"use client";
import { useState } from "react";

interface PipelineStage {
  name: string;
  label: string;
  desc: string;
}

const stages: PipelineStage[] = [
  { name: "PHY", label: "PHY 收发器", desc: "物理层芯片，负责信号编解码（PAM-4/NRZ）、时钟恢复、自动协商速率（100M/1G/10G/25G/100G）。SerDes 将并行数据序列化为高速串行信号。" },
  { name: "MAC", label: "MAC 控制器", desc: "数据链路层核心，执行帧定界（SFD检测）、CRC-32 校验、MAC 地址过滤（精确匹配+哈希过滤）、VLAN Tag 解析、流控帧（PAUSE）处理。" },
  { name: "Buffer", label: "片上缓冲区", desc: "SRAM 缓冲区暂存突发流量，典型容量 128KB-16MB。支持 Rx/Tx 独立队列，配合 DCB（Data Center Bridging）实现无损网络。" },
  { name: "DMA", label: "DMA 引擎", desc: "总线主控引擎，管理描述符环（Descriptor Ring），Scatter-Gather DMA 支持跨不连续内存页传输。RSS（Receive Side Scaling）将流哈希到多队列。" },
  { name: "Host", label: "Host Interface", desc: "PCIe Gen4/Gen5 接口连接主机，支持 SR-IOV 虚拟化，VF 数量可达 256。BAR 空间映射寄存器和 MSI-X 中断向量表。" },
];

export function NICControllerArchitecture() {
  const [selected, setSelected] = useState(0);
  const [direction, setDirection] = useState<"rx" | "tx">("rx");

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">MAC 控制器内部数据通路</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setDirection("rx")} className={`px-3 py-1.5 rounded text-sm ${direction === "rx" ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>接收 (RX)</button>
        <button onClick={() => setDirection("tx")} className={`px-3 py-1.5 rounded text-sm ${direction === "tx" ? "bg-green-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>发送 (TX)</button>
      </div>
      <div className="flex items-center gap-1 mb-4 overflow-x-auto">
        {(direction === "rx" ? stages : [...stages].reverse()).map((s, i, arr) => (
          <span key={s.name} className="flex items-center">
            <button onClick={() => setSelected(direction === "rx" ? i : stages.length - 1 - i)} className={`px-3 py-2 rounded text-xs whitespace-nowrap ${selected === (direction === "rx" ? i : stages.length - 1 - i) ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>{s.label}</button>
            {i < arr.length - 1 && <span className="mx-1 text-text-secondary">→</span>}
          </span>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <h4 className="font-semibold text-text-primary mb-2">{stages[selected].label}</h4>
        <p className="text-sm text-text-secondary">{stages[selected].desc}</p>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
        {[{ l: "线速转发", v: "100 Gbps" }, { l: "队列深度", v: "4096 描述符" }, { l: "中断向量", v: "128 MSI-X" }].map((m, i) => (
          <div key={i} className="p-2 rounded bg-gray-100 dark:bg-gray-800 text-center"><span className="block font-medium text-text-primary">{m.l}</span><span className="text-text-secondary">{m.v}</span></div>
        ))}
      </div>
    </div>
  );
}
export default NICControllerArchitecture;
