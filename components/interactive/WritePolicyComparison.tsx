"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Pen, RefreshCw, Clock, Zap } from "lucide-react";

type Policy = "write-through" | "write-back";

const POLICY_INFO: Record<Policy, { name: string; desc: string; pros: string[]; cons: string[] }> = {
  "write-through": {
    name: "写直达 (Write-Through)",
    desc: "每次写操作同时更新Cache和主存",
    pros: ["实现简单", "主存数据始终最新", "易于一致性维护"],
    cons: ["写速度慢（每次写主存）", "总线带宽消耗大"],
  },
  "write-back": {
    name: "写回 (Write-Back)",
    desc: "只更新Cache，被替换时才写回主存",
    pros: ["写速度快", "减少总线流量", "多次写同一块只需一次写回"],
    cons: ["实现复杂（脏位）", "数据可能不一致"],
  },
};

interface Step { label: string; time: number }

const WRITE_THROUGH_STEPS: Step[] = [
  { label: "CPU发出写请求", time: 1 },
  { label: "查找Cache（命中）", time: 2 },
  { label: "更新Cache数据", time: 1 },
  { label: "写入主存", time: 10 },
  { label: "写操作完成", time: 0 },
];

const WRITE_BACK_STEPS: Step[] = [
  { label: "CPU发出写请求", time: 1 },
  { label: "查找Cache（命中）", time: 2 },
  { label: "更新Cache + 设置脏位", time: 1 },
  { label: "写操作完成", time: 0 },
];

export function WritePolicyComparison() {
  const [policy, setPolicy] = useState<Policy>("write-through");
  const [step, setStep] = useState(0);
  const steps = policy === "write-through" ? WRITE_THROUGH_STEPS : WRITE_BACK_STEPS;
  const info = POLICY_INFO[policy];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Pen className="w-5 h-5 text-purple-500" />
        写策略对比
      </h3>
      <div className="flex gap-2 mb-4">
        {(["write-through", "write-back"] as Policy[]).map(p => (
          <button key={p} onClick={() => { setPolicy(p); setStep(0); }}
            className={`px-3 py-1.5 rounded text-sm ${policy === p ? "bg-purple-500 text-white" : "bg-bg-subtle"}`}>
            {POLICY_INFO[p].name}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-4">{info.desc}</p>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-green-500/10 rounded p-3">
          <div className="text-sm font-medium text-green-500 mb-1">优点</div>
          <ul className="text-xs space-y-1">{info.pros.map((p, i) => <li key={i}>• {p}</li>)}</ul>
        </div>
        <div className="bg-red-500/10 rounded p-3">
          <div className="text-sm font-medium text-red-500 mb-1">缺点</div>
          <ul className="text-xs space-y-1">{info.cons.map((c, i) => <li key={i}>• {c}</li>)}</ul>
        </div>
      </div>
      <div className="flex items-center gap-2 mb-3">
        <button onClick={() => setStep(s => Math.min(s + 1, steps.length - 1))}
          className="px-3 py-1 bg-purple-500 text-white rounded text-sm flex items-center gap-1">
          <Zap className="w-3 h-3" /> 步进
        </button>
        <button onClick={() => setStep(0)} className="px-3 py-1 bg-bg-subtle rounded text-sm flex items-center gap-1">
          <RefreshCw className="w-3 h-3" /> 重置
        </button>
      </div>
      <div className="space-y-2">
        {steps.map((s, i) => (
          <motion.div key={i} initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: i <= step ? 1 : 0.3, x: i <= step ? 0 : -20 }}
            transition={{ delay: 0.1 }}
            className={`flex items-center gap-3 p-2 rounded ${i === step ? "bg-purple-500/20 border border-purple-500" : i < step ? "bg-bg-subtle" : ""}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${i <= step ? "bg-purple-500 text-white" : "bg-bg-subtle"}`}>
              {i + 1}
            </div>
            <span className="text-sm flex-1">{s.label}</span>
            {s.time > 0 && (
              <span className="text-xs text-text-secondary flex items-center gap-1">
                <Clock className="w-3 h-3" /> {s.time}周期
              </span>
            )}
          </motion.div>
        ))}
      </div>
      <div className="mt-4 text-xs text-text-secondary">
        总时间: {steps.slice(0, step + 1).reduce((sum, s) => sum + s.time, 0)} 周期
      </div>
    </div>
  );
}
