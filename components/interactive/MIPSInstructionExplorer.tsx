"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, ChevronRight } from "lucide-react";

type Format = "R" | "I" | "J";

interface InstrDef {
  name: string;
  format: Format;
  opcode: number;
  funct?: number;
  fields: Record<string, number>;
  desc: string;
}

const instructions: InstrDef[] = [
  { name: "ADD", format: "R", opcode: 0, funct: 32, fields: { rs: 2, rt: 3, rd: 1, shamt: 0 }, desc: "rd = rs + rt" },
  { name: "SUB", format: "R", opcode: 0, funct: 34, fields: { rs: 2, rt: 3, rd: 1, shamt: 0 }, desc: "rd = rs - rt" },
  { name: "AND", format: "R", opcode: 0, funct: 36, fields: { rs: 2, rt: 3, rd: 1, shamt: 0 }, desc: "rd = rs & rt" },
  { name: "LW", format: "I", opcode: 35, fields: { rs: 2, rt: 1, imm: 100 }, desc: "rt = M[rs + imm]" },
  { name: "SW", format: "I", opcode: 43, fields: { rs: 2, rt: 1, imm: 100 }, desc: "M[rs + imm] = rt" },
  { name: "BEQ", format: "I", opcode: 4, fields: { rs: 1, rt: 2, imm: -4 }, desc: "if(rs==rt) PC += imm" },
  { name: "ADDI", format: "I", opcode: 8, fields: { rs: 2, rt: 1, imm: 5 }, desc: "rt = rs + imm" },
  { name: "J", format: "J", opcode: 2, fields: { addr: 0x1000 }, desc: "PC = addr" },
  { name: "JAL", format: "J", opcode: 3, fields: { addr: 0x2000 }, desc: "ra=PC; PC = addr" },
];

const formatBits: Record<Format, Record<string, number>> = {
  R: { opcode: 6, rs: 5, rt: 5, rd: 5, shamt: 5, funct: 6 },
  I: { opcode: 6, rs: 5, rt: 5, imm: 16 },
  J: { opcode: 6, addr: 26 },
};

function toBin(n: number, bits: number): string {
  if (n < 0) return ((1 << bits) + n).toString(2).padStart(bits, "0");
  return n.toString(2).padStart(bits, "0");
}

export function MIPSInstructionExplorer() {
  const [selected, setSelected] = useState(0);
  const instr = instructions[selected];
  const bits = formatBits[instr.format];

  const encode = (): { field: string; value: number; binary: string }[] => {
    const result: { field: string; value: number; binary: string }[] = [];
    result.push({ field: "opcode", value: instr.opcode, binary: toBin(instr.opcode, 6) });
    if (instr.format === "R") {
      result.push({ field: "rs", value: instr.fields.rs, binary: toBin(instr.fields.rs, 5) });
      result.push({ field: "rt", value: instr.fields.rt, binary: toBin(instr.fields.rt, 5) });
      result.push({ field: "rd", value: instr.fields.rd, binary: toBin(instr.fields.rd, 5) });
      result.push({ field: "shamt", value: instr.fields.shamt, binary: toBin(instr.fields.shamt, 5) });
      result.push({ field: "funct", value: instr.funct!, binary: toBin(instr.funct!, 6) });
    } else if (instr.format === "I") {
      result.push({ field: "rs", value: instr.fields.rs, binary: toBin(instr.fields.rs, 5) });
      result.push({ field: "rt", value: instr.fields.rt, binary: toBin(instr.fields.rt, 5) });
      result.push({ field: "imm", value: instr.fields.imm, binary: toBin(instr.fields.imm, 16) });
    } else {
      result.push({ field: "addr", value: instr.fields.addr, binary: toBin(instr.fields.addr, 26) });
    }
    return result;
  };

  const encoded = encode();

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Search className="w-5 h-5 text-blue-500" />
        MIPS指令探索器
      </h3>

      <div className="flex flex-wrap gap-1 mb-4">
        {instructions.map((inst, i) => (
          <button key={i} onClick={() => setSelected(i)}
            className={`px-2 py-1 rounded text-xs font-mono ${selected === i ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
            {inst.name}
            <span className={`ml-1 px-1 rounded text-xs ${inst.format === "R" ? "bg-green-500/20" : inst.format === "I" ? "bg-yellow-500/20" : "bg-purple-500/20"}`}>
              {inst.format}
            </span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={selected} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="flex h-10 rounded overflow-hidden mb-3 border border-border-subtle">
            {encoded.map((f, i) => (
              <div key={i} className="flex items-center justify-center text-xs font-mono text-white border-r border-white/20 last:border-0"
                style={{
                  width: `${(f.binary.length / 32) * 100}%`,
                  backgroundColor: f.field === "opcode" ? "#3b82f6" : f.field === "funct" ? "#8b5cf6" : f.field === "addr" ? "#ec4899" : `hsl(${i * 60}, 60%, 45%)`
                }}>
                {f.field}
              </div>
            ))}
          </div>

          <div className="font-mono text-sm mb-3 break-all tracking-wide">
            {encoded.map((f, i) => (
              <span key={i} className="text-blue-400">{f.binary}</span>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-2">
            {encoded.map((f, i) => (
              <div key={i} className="flex items-center gap-2 p-2 rounded bg-bg-surface border border-border-subtle text-sm">
                <span className="font-medium">{f.field}</span>
                <span className="text-xs text-text-muted">= {f.value} (0x{f.value.toString(16)})</span>
                <span className="ml-auto font-mono text-xs text-blue-400">{f.binary}</span>
              </div>
            ))}
          </div>

          <div className="mt-3 text-sm text-text-muted">
            语义: <span className="font-mono text-text-secondary">{instr.desc}</span>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
