"use client";
import { useState } from "react";

const architectures = [
  {
    name: "传统架构",
    en: "Traditional",
    desc: "每台设备独立运行专有软硬件，控制平面与数据平面紧密耦合",
    pros: ["成熟稳定", "厂商支持完善", "单设备性能高"],
    cons: ["配置复杂", "创新周期长", "厂商锁定"],
    control: "分布式（每设备独立）",
    programmability: "低（CLI/SNMP）",
  },
  {
    name: "SDN架构",
    en: "Software-Defined Networking",
    desc: "控制平面集中化，通过OpenFlow等南向接口编程数据平面",
    pros: ["集中管控", "灵活编程", "快速创新"],
    cons: ["控制器单点风险", "标准化不完善", "规模受限"],
    control: "集中式（控制器）",
    programmability: "高（REST API）",
  },
  {
    name: "NFV架构",
    en: "Network Function Virtualization",
    desc: "网络功能（防火墙、负载均衡等）从专用硬件解耦为虚拟化软件",
    pros: ["降低成本", "弹性伸缩", "快速部署"],
    cons: ["性能开销", "可靠性挑战", "管理复杂"],
    control: "混合（编排器+VNF）",
    programmability: "中高（MANO接口）",
  },
];

export function InteractiveNetworkArchComparison() {
  const [selected, setSelected] = useState([0, 1, 2]);
  const toggle = (idx: number) => {
    setSelected((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
    );
  };

  const visible = architectures.filter((_, i) => selected.includes(i));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Network Architecture Comparison <span className="text-text-secondary text-sm">— 架构对比</span>
      </h3>
      <div className="flex gap-2 mb-4">
        {architectures.map((a, i) => (
          <button
            key={i}
            onClick={() => toggle(i)}
            className={`px-3 py-1 rounded text-sm ${selected.includes(i) ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {a.name}
          </button>
        ))}
      </div>
      <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${visible.length}, 1fr)` }}>
        {visible.map((a, i) => (
          <div key={i} className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
            <div className="font-semibold text-text-primary mb-1">{a.name}</div>
            <div className="text-xs text-text-secondary mb-2">{a.en}</div>
            <p className="text-sm text-text-secondary mb-3">{a.desc}</p>
            <div className="text-xs text-text-secondary mb-1">控制方式: {a.control}</div>
            <div className="text-xs text-text-secondary mb-3">可编程性: {a.programmability}</div>
            <div className="text-sm text-green-600 dark:text-green-400 mb-1">优势:</div>
            <ul className="text-xs text-text-secondary list-disc list-inside mb-2">
              {a.pros.map((p, j) => <li key={j}>{p}</li>)}
            </ul>
            <div className="text-sm text-red-600 dark:text-red-400 mb-1">劣势:</div>
            <ul className="text-xs text-text-secondary list-disc list-inside">
              {a.cons.map((c, j) => <li key={j}>{c}</li>)}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

export default InteractiveNetworkArchComparison;
