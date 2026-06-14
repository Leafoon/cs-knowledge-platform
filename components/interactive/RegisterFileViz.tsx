"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Database, ArrowRight } from "lucide-react";

const registers = [
  { name: "R0", alias: "zero", value: 0x00000000, desc: "常量0" },
  { name: "R1", alias: "at", value: 0x00000004, desc: "汇编器保留" },
  { name: "R2", alias: "v0", value: 0x0000002A, desc: "返回值" },
  { name: "R3", alias: "v1", value: 0x00000000, desc: "返回值" },
  { name: "R4", alias: "a0", value: 0x00000100, desc: "参数0" },
  { name: "R5", alias: "a1", value: 0x00000008, desc: "参数1" },
  { name: "R6", alias: "t0", value: 0x0000FFFF, desc: "临时寄存器" },
  { name: "R7", alias: "t1", value: 0x12345678, desc: "临时寄存器" },
];

export function RegisterFileViz() {
  const [selected, setSelected] = useState<number | null>(null);
  const [readPort1, setReadPort1] = useState(4);
  const [readPort2, setReadPort2] = useState(5);
  const [writePort, setWritePort] = useState(2);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Database className="w-5 h-5 text-violet-500" />
        寄存器文件可视化
      </h3>

      <div className="grid grid-cols-4 gap-2 mb-4">
        {registers.map((r, i) => (
          <motion.div
            key={r.name}
            onClick={() => setSelected(selected === i ? null : i)}
            className={`p-2 rounded border cursor-pointer text-center transition-colors ${
              readPort1 === i ? "border-blue-500 bg-blue-500/10" :
              readPort2 === i ? "border-green-500 bg-green-500/10" :
              writePort === i ? "border-orange-500 bg-orange-500/10" :
              selected === i ? "border-purple-500 bg-purple-500/10" :
              "border-border-subtle bg-bg-surface hover:border-blue-400"
            }`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.05 }}
          >
            <div className="text-xs font-bold">{r.name}</div>
            <div className="text-xs text-text-muted">${r.alias}</div>
            <div className="font-mono text-xs mt-1">0x{r.value.toString(16).padStart(8, "0")}</div>
          </motion.div>
        ))}
      </div>

      {selected !== null && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="p-3 rounded bg-bg-surface border border-border-subtle text-sm mb-4">
          <span className="font-medium">{registers[selected].name} (${registers[selected].alias})</span>
          <span className="text-text-muted ml-2">— {registers[selected].desc}</span>
        </motion.div>
      )}

      <div className="p-4 rounded bg-bg-surface border border-border-subtle">
        <div className="text-xs text-text-muted mb-3">数据通路端口</div>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-xs text-blue-400 mb-1">读端口 1 (rs)</div>
            <select value={readPort1} onChange={e => setReadPort1(Number(e.target.value))}
              className="w-full px-1 py-1 rounded border border-border-subtle bg-bg-surface text-sm">
              {registers.map((r, i) => <option key={i} value={i}>{r.name}</option>)}
            </select>
            <ArrowRight className="w-4 h-4 mx-auto mt-1 text-blue-400" />
            <div className="font-mono text-xs mt-1">0x{registers[readPort1].value.toString(16).padStart(8, "0")}</div>
          </div>
          <div>
            <div className="text-xs text-green-400 mb-1">读端口 2 (rt)</div>
            <select value={readPort2} onChange={e => setReadPort2(Number(e.target.value))}
              className="w-full px-1 py-1 rounded border border-border-subtle bg-bg-surface text-sm">
              {registers.map((r, i) => <option key={i} value={i}>{r.name}</option>)}
            </select>
            <ArrowRight className="w-4 h-4 mx-auto mt-1 text-green-400" />
            <div className="font-mono text-xs mt-1">0x{registers[readPort2].value.toString(16).padStart(8, "0")}</div>
          </div>
          <div>
            <div className="text-xs text-orange-400 mb-1">写端口 (rd)</div>
            <select value={writePort} onChange={e => setWritePort(Number(e.target.value))}
              className="w-full px-1 py-1 rounded border border-border-subtle bg-bg-surface text-sm">
              {registers.map((r, i) => <option key={i} value={i}>{r.name}</option>)}
            </select>
            <div className="font-mono text-xs mt-1 text-orange-400">← 写入目标</div>
          </div>
        </div>
      </div>
    </div>
  );
}
