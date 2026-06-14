"use client";
import { useState } from "react";

interface ComparisonItem {
  feature: string;
  stp: string;
  rstp: string;
}

const comparisons: ComparisonItem[] = [
  { feature: "收敛时间", stp: "30-50秒", rstp: "几秒内（<1s理想）" },
  { feature: "端口角色", stp: "根端口/指定端口/非指定端口", rstp: "根端口/指定端口/替代端口/备份端口/禁用端口" },
  { feature: "端口状态", stp: "5种: Blocking→Listening→Learning→Forwarding→Disabled", rstp: "3种: Discarding→Learning→Forwarding" },
  { feature: "BPDU发送", stp: "仅根桥周期发送（2s）", rstp: "所有桥每Hello时间发送，充当keepalive" },
  { feature: "Proposal/Agreement", stp: "不支持", rstp: "支持，快速收敛机制" },
  { feature: "边缘端口", stp: "不支持（需等待转发延迟）", rstp: "支持PortFast，立即进入转发" },
  { feature: "链路类型", stp: "无区分", rstp: "点对点/共享，点对点可快速收敛" },
  { feature: "兼容性", stp: "IEEE 802.1D", rstp: "IEEE 802.1w，向下兼容STP" },
];

export function RSTPComparison() {
  const [activeItem, setActiveItem] = useState(0);
  const [showTimeline, setShowTimeline] = useState(false);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">RSTP vs STP 对比</h3>
      <div className="space-y-1.5 mb-4">
        {comparisons.map((c, i) => (
          <button key={i} onClick={() => setActiveItem(i)}
            className={`w-full text-left px-3 py-2 rounded-lg border text-xs transition-all ${activeItem === i ? "bg-sky-500/15 border-sky-400/40" : "bg-bg-tertiary border-border-subtle hover:border-sky-400/20"}`}>
            <span className="font-medium text-text-primary">{c.feature}</span>
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
          <div className="text-xs font-medium text-amber-600 dark:text-amber-400 mb-1">STP</div>
          <div className="text-xs text-text-secondary">{comparisons[activeItem].stp}</div>
        </div>
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
          <div className="text-xs font-medium text-emerald-600 dark:text-emerald-400 mb-1">RSTP</div>
          <div className="text-xs text-text-secondary">{comparisons[activeItem].rstp}</div>
        </div>
      </div>
      <button onClick={() => setShowTimeline(!showTimeline)} className="w-full px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:text-text-primary transition-colors mb-3">
        {showTimeline ? "隐藏" : "显示"}收敛时间线对比
      </button>
      {showTimeline && (
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4">
          <div className="flex items-end gap-1 h-20 mb-2">
            {["Blocking", "Listening", "Learning", "Forwarding"].map((s, i) => (
              <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
                <div className={`w-full rounded-t ${i < 2 ? "bg-red-400/40" : i === 2 ? "bg-amber-400/40" : "bg-emerald-400/40"}`}
                  style={{ height: `${(i + 1) * 18}px` }} />
                <span className="text-[8px] text-text-tertiary">{s}</span>
              </div>
            ))}
          </div>
          <div className="text-[10px] text-text-tertiary text-center">STP: 需要经历Blocking(20s)→Listening(15s)→Learning(15s)→Forwarding</div>
          <div className="mt-2 flex items-center justify-center gap-2">
            <div className="h-2 flex-1 bg-emerald-400/40 rounded" />
            <span className="text-[10px] text-emerald-500">RSTP: Discarding → Forwarding（几秒内完成）</span>
          </div>
        </div>
      )}
    </div>
  );
}
export default RSTPComparison;
