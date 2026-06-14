"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Clock, ChevronLeft, ChevronRight } from "lucide-react";

const timeline = [
  { year: 1978, name: "x86 (8086)", type: "CISC", desc: "Intel推出8086处理器，开创x86 CISC架构。变长指令、复杂寻址模式。", features: ["变长指令", "微码执行", "复杂寻址"] },
  { year: 1985, name: "ARM1", type: "RISC", desc: "Acorn推出ARM1处理器，采用精简指令集设计。低功耗、高效率。", features: ["定长指令", "Load/Store架构", "条件执行"] },
  { year: 1985, name: "MIPS", type: "RISC", desc: "MIPS架构由Stanford推出，影响深远。规整的32位指令格式。", features: ["三种指令格式", "延迟分支", "流水线友好"] },
  { year: 1989, name: "i486", type: "CISC", desc: "Intel 486引入片上缓存和流水线，RISC内核执行CISC指令。", features: ["RISC内核", "片上缓存", "流水线"] },
  { year: 1992, name: "ARM7TDMI", type: "RISC", desc: "ARM7TDMI成为嵌入式领域最成功的处理器内核。", features: ["Thumb指令集", "低功耗", "嵌入式"] },
  { year: 2001, name: "IA-64/Itanium", type: "CISC", desc: "Intel尝试EPIC架构，但市场反响不佳。", features: ["EPIC/VLIW", "显式并行", "编译器复杂"] },
  { year: 2011, name: "ARMv8 (AArch64)", type: "RISC", desc: "ARM推出64位架构，进入服务器和高性能计算领域。", features: ["64位", "NEON SIMD", "虚拟化"] },
  { year: 2015, name: "RISC-V", type: "RISC", desc: "Berkeley推出开源RISC-V指令集架构，模块化设计。", features: ["开源", "模块化", "可扩展"] },
];

export function ISAEvolutionTimeline() {
  const [active, setActive] = useState(0);

  const prev = () => setActive(Math.max(0, active - 1));
  const next = () => setActive(Math.min(timeline.length - 1, active + 1));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Clock className="w-5 h-5 text-yellow-500" />
        ISA演进时间线
      </h3>

      <div className="relative mb-6">
        <div className="flex items-center justify-between">
          {timeline.map((t, i) => (
            <button key={i} onClick={() => setActive(i)}
              className={`flex flex-col items-center transition-colors ${active === i ? "scale-110" : "opacity-60 hover:opacity-80"}`}>
              <div className={`w-3 h-3 rounded-full mb-1 ${t.type === "RISC" ? "bg-green-500" : "bg-orange-500"} ${active === i ? "ring-2 ring-blue-400" : ""}`} />
              <span className="text-xs font-mono hidden sm:block">{t.year}</span>
            </button>
          ))}
        </div>
        <div className="absolute top-1.5 left-0 right-0 h-0.5 bg-border-subtle -z-10" />
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={active} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
          <div className="p-4 rounded bg-bg-surface border border-border-subtle">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-2xl font-bold font-mono">{timeline[active].year}</span>
              <span className="text-lg font-semibold">{timeline[active].name}</span>
              <span className={`px-2 py-0.5 rounded text-xs ${
                timeline[active].type === "RISC" ? "bg-green-500/20 text-green-400" : "bg-orange-500/20 text-orange-400"
              }`}>
                {timeline[active].type}
              </span>
            </div>
            <p className="text-sm text-text-secondary mb-3">{timeline[active].desc}</p>
            <div className="flex flex-wrap gap-1">
              {timeline[active].features.map(f => (
                <span key={f} className="px-2 py-0.5 rounded text-xs bg-bg-elevated border border-border-subtle">{f}</span>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="flex justify-between mt-4">
        <button onClick={prev} disabled={active === 0}
          className="px-3 py-1 rounded bg-bg-surface border border-border-subtle text-sm flex items-center gap-1 disabled:opacity-30 hover:border-blue-400">
          <ChevronLeft className="w-4 h-4" /> 上一个
        </button>
        <span className="text-xs text-text-muted self-center">{active + 1} / {timeline.length}</span>
        <button onClick={next} disabled={active === timeline.length - 1}
          className="px-3 py-1 rounded bg-bg-surface border border-border-subtle text-sm flex items-center gap-1 disabled:opacity-30 hover:border-blue-400">
          下一个 <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
