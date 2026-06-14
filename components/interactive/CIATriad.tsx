"use client";
import { useState } from "react";

interface CiaElement {
  id: "C" | "I" | "A";
  name: string;
  ename: string;
  desc: string;
  threats: string[];
  controls: string[];
  color: string;
  bgColor: string;
}

const elements: CiaElement[] = [
  {
    id: "C", name: "机密性", ename: "Confidentiality",
    desc: "确保信息不被未授权访问，仅授权用户可读取数据。",
    threats: ["窃听攻击", "SQL注入数据泄露", "社会工程学", "中间人攻击"],
    controls: ["加密(AES/RSA)", "访问控制列表(ACL)", "数据分类标记", "网络分段"],
    color: "text-blue-600 dark:text-blue-400",
    bgColor: "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800",
  },
  {
    id: "I", name: "完整性", ename: "Integrity",
    desc: "确保数据未被篡改，信息在传输和存储过程中保持准确和完整。",
    threats: ["数据篡改", "中间人修改", "恶意软件感染", "内部人员违规修改"],
    controls: ["数字签名", "哈希校验(SHA-256)", "版本控制", "输入验证"],
    color: "text-green-600 dark:text-green-400",
    bgColor: "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800",
  },
  {
    id: "A", name: "可用性", ename: "Availability",
    desc: "确保授权用户在需要时能够访问信息和资源。",
    threats: ["DDoS攻击", "硬件故障", "自然灾害", "软件Bug导致服务中断"],
    controls: ["负载均衡", "冗余备份", "灾备计划(DRP)", "DDoS防护"],
    color: "text-purple-600 dark:text-purple-400",
    bgColor: "bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800",
  },
];

export function CIATriad() {
  const [selected, setSelected] = useState<"C" | "I" | "A" | null>(null);
  const [showThreats, setShowThreats] = useState(true);

  const active = elements.find((e) => e.id === selected);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">CIA 三元组</h3>
      <div className="flex justify-center mb-4">
        <div className="relative w-64 h-56">
          <button onClick={() => setSelected(selected === "C" ? null : "C")}
            className={`absolute top-0 left-1/2 -translate-x-1/2 w-24 h-24 rounded-full flex flex-col items-center justify-center border-2 transition-all ${selected === "C" ? "scale-110 shadow-lg" : ""} ${elements[0].bgColor}`}>
            <span className="text-2xl font-bold text-blue-600">C</span>
            <span className="text-[10px] text-text-secondary">机密性</span>
          </button>
          <button onClick={() => setSelected(selected === "I" ? null : "I")}
            className={`absolute bottom-0 left-4 w-24 h-24 rounded-full flex flex-col items-center justify-center border-2 transition-all ${selected === "I" ? "scale-110 shadow-lg" : ""} ${elements[1].bgColor}`}>
            <span className="text-2xl font-bold text-green-600">I</span>
            <span className="text-[10px] text-text-secondary">完整性</span>
          </button>
          <button onClick={() => setSelected(selected === "A" ? null : "A")}
            className={`absolute bottom-0 right-4 w-24 h-24 rounded-full flex flex-col items-center justify-center border-2 transition-all ${selected === "A" ? "scale-110 shadow-lg" : ""} ${elements[2].bgColor}`}>
            <span className="text-2xl font-bold text-purple-600">A</span>
            <span className="text-[10px] text-text-secondary">可用性</span>
          </button>
        </div>
      </div>
      {active && (
        <div className="p-4 rounded-lg border border-border-subtle bg-gray-50 dark:bg-gray-800 mb-4">
          <div className="flex items-center gap-2 mb-2">
            <span className={`text-lg font-bold ${active.color}`}>{active.id}</span>
            <span className="text-sm font-bold text-text-primary">{active.name} ({active.ename})</span>
          </div>
          <p className="text-xs text-text-secondary mb-3">{active.desc}</p>
          <div className="flex gap-2 mb-2">
            <button onClick={() => setShowThreats(true)}
              className={`px-3 py-1 rounded text-xs ${showThreats ? "bg-red-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
              威胁
            </button>
            <button onClick={() => setShowThreats(false)}
              className={`px-3 py-1 rounded text-xs ${!showThreats ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
              防护措施
            </button>
          </div>
          <ul className="space-y-1">
            {(showThreats ? active.threats : active.controls).map((item, i) => (
              <li key={i} className={`text-xs flex items-center gap-2 ${showThreats ? "text-red-600 dark:text-red-400" : "text-green-600 dark:text-green-400"}`}>
                <span>{showThreats ? "⚠" : "✓"}</span> {item}
              </li>
            ))}
          </ul>
        </div>
      )}
      {!active && (
        <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800 text-center">
          <p className="text-sm text-text-secondary">点击上方圆形查看安全要素详情</p>
        </div>
      )}
      <p className="text-xs text-text-secondary mt-3">CIA三元组是信息安全的核心模型，所有安全机制都围绕保护这三个属性展开。</p>
    </div>
  );
}
export default CIATriad;
