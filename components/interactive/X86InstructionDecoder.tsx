"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileCode } from "lucide-react";

interface X86Field {
  name: string;
  size: string;
  desc: string;
  color: string;
  value?: string;
}

const examples: { name: string; fields: X86Field[] }[] = [
  {
    name: "ADD EAX, EBX",
    fields: [
      { name: "操作码", size: "1B", desc: "03 C3: ADD r32, r/m32", color: "#3b82f6", value: "03" },
      { name: "ModR/M", size: "1B", desc: "C3=11 000 011: mod=11, reg=000(EAX), r/m=011(EBX)", color: "#10b981", value: "C3" },
    ],
  },
  {
    name: "MOV [EBX+8], EAX",
    fields: [
      { name: "操作码", size: "1B", desc: "89: MOV r/m32, r32", color: "#3b82f6", value: "89" },
      { name: "ModR/M", size: "1B", desc: "43: mod=01, reg=000(EAX), r/m=011(EBX)", color: "#10b981", value: "43" },
      { name: "位移", size: "1B", desc: "08: disp8 = 8", color: "#f59e0b", value: "08" },
    ],
  },
  {
    name: "ADD EAX, [EBX+ECX*4+0x100]",
    fields: [
      { name: "操作码", size: "1B", desc: "03: ADD r32, r/m32", color: "#3b82f6", value: "03" },
      { name: "ModR/M", size: "1B", desc: "84: mod=10, reg=000(EAX), r/m=100(SIB)", color: "#10b981", value: "84" },
      { name: "SIB", size: "1B", desc: "8B: scale=10(×4), index=011(ECX), base=011(EBX)", color: "#8b5cf6", value: "8B" },
      { name: "位移", size: "4B", desc: "00000100: disp32 = 0x100", color: "#f59e0b", value: "00000100" },
    ],
  },
  {
    name: "PUSH 0x42",
    fields: [
      { name: "操作码", size: "1B", desc: "6A: PUSH imm8", color: "#3b82f6", value: "6A" },
      { name: "立即数", size: "1B", desc: "42: imm8 = 0x42", color: "#ef4444", value: "42" },
    ],
  },
];

export function X86InstructionDecoder() {
  const [selected, setSelected] = useState(0);
  const [expandedField, setExpandedField] = useState<number | null>(null);
  const ex = examples[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <FileCode className="w-5 h-5 text-orange-500" />
        x86指令译码器
      </h3>

      <div className="flex flex-wrap gap-2 mb-4">
        {examples.map((e, i) => (
          <button key={i} onClick={() => { setSelected(i); setExpandedField(null); }}
            className={`px-3 py-1 rounded text-xs font-mono ${selected === i ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
            {e.name}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={selected} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="flex h-10 rounded overflow-hidden mb-4 border border-border-subtle">
            {ex.fields.map((f, i) => (
              <motion.div
                key={i}
                onClick={() => setExpandedField(expandedField === i ? null : i)}
                className="flex items-center justify-center text-xs font-mono text-white cursor-pointer border-r border-white/20 last:border-0"
                style={{ backgroundColor: f.color, flex: f.size === "4B" ? 4 : 1 }}
                initial={{ width: 0 }}
                animate={{ width: "auto", flex: f.size === "4B" ? 4 : 1 }}
                transition={{ delay: i * 0.15 }}
              >
                {f.name} ({f.size})
              </motion.div>
            ))}
          </div>

          <div className="font-mono text-sm mb-4 tracking-widest">
            {ex.fields.map((f, i) => (
              <span key={i} className="text-blue-400">{f.value} </span>
            ))}
          </div>

          <div className="space-y-2">
            {ex.fields.map((f, i) => (
              <motion.div
                key={i}
                className={`p-3 rounded border cursor-pointer transition-colors ${
                  expandedField === i ? "border-blue-500 bg-blue-500/5" : "border-border-subtle bg-bg-surface hover:border-blue-400"
                }`}
                onClick={() => setExpandedField(expandedField === i ? null : i)}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: f.color }} />
                  <span className="font-medium text-sm">{f.name}</span>
                  <span className="text-xs text-text-muted">({f.size})</span>
                  {f.value && <span className="ml-auto font-mono text-xs text-blue-400">{f.value}</span>}
                </div>
                <AnimatePresence>
                  {expandedField === i && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden">
                      <div className="mt-2 text-xs text-text-secondary font-mono">{f.desc}</div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
