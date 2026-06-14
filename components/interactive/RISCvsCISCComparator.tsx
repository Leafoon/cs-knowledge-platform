"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Scale, ChevronDown, ChevronUp } from "lucide-react";

const categories = [
  { aspect: "指令数量", risc: "少（50~200条）", cisc: "多（200~500+条）", winner: "risc" },
  { aspect: "指令长度", risc: "固定长度", cisc: "可变长度", winner: "risc" },
  { aspect: "指令格式", risc: "简单、规整", cisc: "复杂、多样", winner: "risc" },
  { aspect: "寻址方式", risc: "少（3~5种）", cisc: "多（10+种）", winner: "cisc" },
  { aspect: "CPI", risc: "接近1", cisc: "大于1（微码执行）", winner: "risc" },
  { aspect: "代码密度", risc: "较低", cisc: "较高", winner: "cisc" },
  { aspect: "流水线", risc: "易于流水线化", cisc: "难以流水线化", winner: "risc" },
  { aspect: "硬件复杂度", risc: "简单", cisc: "复杂（微码ROM）", winner: "risc" },
  { aspect: "编译器复杂度", risc: "高（需优化）", cisc: "低（硬件完成）", winner: "cisc" },
  { aspect: "功耗", risc: "低", cisc: "高", winner: "risc" },
];

export function RISCvsCISCComparator() {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [filter, setFilter] = useState<"all" | "risc" | "cisc">("all");

  const filtered = filter === "all" ? categories : categories.filter(c => c.winner === filter);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Scale className="w-5 h-5 text-rose-500" />
        RISC vs CISC 全面对比
      </h3>

      <div className="flex gap-2 mb-4">
        {(["all", "risc", "cisc"] as const).map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded text-xs ${filter === f ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
            {f === "all" ? "全部" : f.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left p-2">对比维度</th>
              <th className="text-left p-2">RISC</th>
              <th className="text-left p-2">CISC</th>
              <th className="text-left p-2 w-8"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((c, i) => (
              <motion.tr key={c.aspect} className="border-b border-border-subtle cursor-pointer hover:bg-bg-hover"
                onClick={() => setExpanded(expanded === i ? null : i)}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.03 }}>
                <td className="p-2 font-medium">{c.aspect}</td>
                <td className={`p-2 ${c.winner === "risc" ? "text-green-400" : ""}`}>{c.risc}</td>
                <td className={`p-2 ${c.winner === "cisc" ? "text-green-400" : ""}`}>{c.cisc}</td>
                <td className="p-2">{expanded === i ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex gap-4 text-sm">
        <div className="p-2 rounded bg-green-500/10 border border-green-500/30">
          <span className="text-green-400 font-medium">RISC 优势: </span>
          <span className="text-text-muted">{categories.filter(c => c.winner === "risc").length} 项</span>
        </div>
        <div className="p-2 rounded bg-blue-500/10 border border-blue-500/30">
          <span className="text-blue-400 font-medium">CISC 优势: </span>
          <span className="text-text-muted">{categories.filter(c => c.winner === "cisc").length} 项</span>
        </div>
      </div>
    </div>
  );
}
