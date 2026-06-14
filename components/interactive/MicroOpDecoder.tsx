"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Scissors, ArrowRight } from "lucide-react";

const examples = [
  {
    cisc: "REP MOVSB",
    desc: "重复移动字节串",
    microOps: [
      "LOOP: LOAD R1, [SI]",
      "STORE [DI], R1",
      "SI = SI + 1",
      "DI = DI + 1",
      "CX = CX - 1",
      "IF CX ≠ 0, GOTO LOOP",
    ],
  },
  {
    cisc: "ADD [mem], R1",
    desc: "内存操作数加法",
    microOps: [
      "MAR ← mem_addr",
      "MDR ← M[MAR]",
      "ALU ← MDR + R1",
      "MDR ← ALU",
      "M[MAR] ← MDR",
    ],
  },
  {
    cisc: "PUSH R1",
    desc: "压栈操作",
    microOps: [
      "SP ← SP - 4",
      "MAR ← SP",
      "MDR ← R1",
      "M[MAR] ← MDR",
    ],
  },
  {
    cisc: "CALL func",
    desc: "过程调用",
    microOps: [
      "SP ← SP - 4",
      "M[SP] ← PC+1 (保存返回地址)",
      "PC ← func_addr",
    ],
  },
];

export function MicroOpDecoder() {
  const [selected, setSelected] = useState(0);
  const [step, setStep] = useState(-1);

  const ex = examples[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Scissors className="w-5 h-5 text-red-500" />
        微操作译码器
      </h3>

      <div className="flex flex-wrap gap-2 mb-4">
        {examples.map((e, i) => (
          <button key={i} onClick={() => { setSelected(i); setStep(-1); }}
            className={`px-3 py-1 rounded text-xs font-mono ${selected === i ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
            {e.cisc}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="p-4 rounded bg-orange-500/5 border border-orange-500/30">
          <div className="text-xs text-orange-400 mb-2 font-medium">CISC 复杂指令</div>
          <div className="font-mono text-lg mb-1">{ex.cisc}</div>
          <div className="text-sm text-text-muted">{ex.desc}</div>
          <div className="mt-3 text-xs text-text-muted">
            1条指令 → {ex.microOps.length} 个微操作
          </div>
        </div>

        <div className="p-4 rounded bg-green-500/5 border border-green-500/30">
          <div className="text-xs text-green-400 mb-2 font-medium">分解为 RISC 微操作</div>
          <div className="space-y-1">
            <AnimatePresence>
              {ex.microOps.map((op, i) => (
                <motion.div
                  key={`${selected}-${i}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className={`flex items-center gap-2 p-1.5 rounded text-xs font-mono ${
                    step >= i ? "bg-green-500/10 text-green-400" : "text-text-secondary"
                  }`}
                >
                  <span className="w-4 text-center text-text-muted">{i + 1}</span>
                  <ArrowRight className="w-3 h-3 text-text-muted" />
                  {op}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <div className="flex gap-2 mt-4">
        <button onClick={() => setStep(Math.min(step + 1, ex.microOps.length - 1))}
          className="px-4 py-1.5 rounded bg-green-500 text-white text-sm hover:bg-green-600">
          逐步执行
        </button>
        <button onClick={() => setStep(ex.microOps.length - 1)}
          className="px-4 py-1.5 rounded bg-bg-surface border border-border-subtle text-sm hover:border-blue-400">
          全部展开
        </button>
        <button onClick={() => setStep(-1)}
          className="px-4 py-1.5 rounded bg-bg-surface border border-border-subtle text-sm hover:border-blue-400">
          重置
        </button>
      </div>
    </div>
  );
}
