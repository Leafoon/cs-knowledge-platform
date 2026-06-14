"use client";
import { useState } from "react";

const tiers = [
  {
    name: "Tier 1 ISP",
    label: "第一层ISP",
    color: "bg-red-600",
    examples: ["AT&T", "NTT", "Cogent"],
    desc: "全球骨干网，免结算对等互联（settlement-free peering）",
    features: ["无需付费转接", "全球覆盖", "与其他Tier-1对等互联"],
  },
  {
    name: "Tier 2 ISP",
    label: "第二层ISP",
    color: "bg-orange-500",
    examples: ["Regional ISPs", "Content Providers"],
    desc: "区域性提供商，向Tier-1付费转接，也与同级对等互联",
    features: ["部分转接付费", "区域覆盖", "与Tier-1和同级互联"],
  },
  {
    name: "Tier 3 ISP",
    label: "第三层ISP",
    color: "bg-yellow-500",
    examples: ["Local ISPs", "Campus Networks"],
    desc: "本地接入网，仅通过付费转接连接互联网",
    features: ["完全依赖转接", "本地覆盖", "为终端用户提供接入"],
  },
];

export function ISPHierarchy() {
  const [selected, setSelected] = useState(0);
  const [showPeering, setShowPeering] = useState(false);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        ISP Hierarchy <span className="text-text-secondary text-sm">— Tier1/2/3 互联层次</span>
      </h3>
      <div className="flex flex-col items-center gap-3 mb-4">
        {tiers.map((t, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`w-full max-w-md text-center py-3 rounded-lg text-white font-semibold transition-all ${t.color} ${selected === i ? "ring-2 ring-offset-2 ring-blue-500 dark:ring-offset-gray-900 scale-105" : "opacity-70"}`}
          >
            {t.name} — {t.label}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-2">{tiers[selected].name}</div>
        <p className="text-sm text-text-secondary mb-2">{tiers[selected].desc}</p>
        <div className="text-sm text-text-secondary mb-1">代表: {tiers[selected].examples.join(", ")}</div>
        <ul className="list-disc list-inside text-sm text-text-secondary">
          {tiers[selected].features.map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      </div>
      <button
        onClick={() => setShowPeering(!showPeering)}
        className="px-3 py-1 rounded bg-purple-600 text-white text-sm mb-3"
      >
        {showPeering ? "隐藏" : "显示"}互联方式
      </button>
      {showPeering && (
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-blue-50 dark:bg-blue-900/30 p-3 rounded">
            <div className="font-semibold text-blue-700 dark:text-blue-300">对等互联 (Peering)</div>
            <div className="text-text-secondary">双方免费交换流量，通常通过IXP</div>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/30 p-3 rounded">
            <div className="font-semibold text-orange-700 dark:text-orange-300">转接 (Transit)</div>
            <div className="text-text-secondary">客户向提供商付费访问整个互联网</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ISPHierarchy;
