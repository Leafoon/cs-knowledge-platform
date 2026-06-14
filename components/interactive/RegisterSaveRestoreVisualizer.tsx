"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, Save, Info } from "lucide-react";

type SaveType = "callee" | "caller" | "special";

interface RegInfo {
  name: string;
  alias: string;
  type: SaveType;
  savedBy: string;
  role: string;
  contextField?: string;
}

const REGISTERS: RegInfo[] = [
  { name: "x0", alias: "zero", type: "special", savedBy: "---", role: "硬编码为 0，不可写" },
  { name: "x1", alias: "ra", type: "callee", savedBy: "swtch()", role: "返回地址 (Return Address)", contextField: "ra" },
  { name: "x2", alias: "sp", type: "callee", savedBy: "swtch()", role: "栈指针 (Stack Pointer)", contextField: "sp" },
  { name: "x3", alias: "gp", type: "special", savedBy: "---", role: "全局指针 (Global Pointer)" },
  { name: "x4", alias: "tp", type: "special", savedBy: "---", role: "线程指针 (Thread Pointer)" },
  { name: "x5", alias: "t0", type: "caller", savedBy: "编译器 (栈)", role: "临时寄存器 0" },
  { name: "x6", alias: "t1", type: "caller", savedBy: "编译器 (栈)", role: "临时寄存器 1" },
  { name: "x7", alias: "t2", type: "caller", savedBy: "编译器 (栈)", role: "临时寄存器 2" },
  { name: "x8", alias: "s0/fp", type: "callee", savedBy: "swtch()", role: "保存寄存器 0 / 帧指针", contextField: "s0" },
  { name: "x9", alias: "s1", type: "callee", savedBy: "swtch()", role: "保存寄存器 1", contextField: "s1" },
  { name: "x10", alias: "a0", type: "caller", savedBy: "编译器 (栈)", role: "参数 0 / 返回值 0" },
  { name: "x11", alias: "a1", type: "caller", savedBy: "编译器 (栈)", role: "参数 1 / 返回值 1" },
  { name: "x12", alias: "a2", type: "caller", savedBy: "编译器 (栈)", role: "参数 2" },
  { name: "x13", alias: "a3", type: "caller", savedBy: "编译器 (栈)", role: "参数 3" },
  { name: "x14", alias: "a4", type: "caller", savedBy: "编译器 (栈)", role: "参数 4" },
  { name: "x15", alias: "a5", type: "caller", savedBy: "编译器 (栈)", role: "参数 5" },
  { name: "x16", alias: "a6", type: "caller", savedBy: "编译器 (栈)", role: "参数 6" },
  { name: "x17", alias: "a7", type: "caller", savedBy: "编译器 (栈)", role: "参数 7" },
  { name: "x18", alias: "s2", type: "callee", savedBy: "swtch()", role: "保存寄存器 2", contextField: "s2" },
  { name: "x19", alias: "s3", type: "callee", savedBy: "swtch()", role: "保存寄存器 3", contextField: "s3" },
  { name: "x20", alias: "s4", type: "callee", savedBy: "swtch()", role: "保存寄存器 4", contextField: "s4" },
  { name: "x21", alias: "s5", type: "callee", savedBy: "swtch()", role: "保存寄存器 5", contextField: "s5" },
  { name: "x22", alias: "s6", type: "callee", savedBy: "swtch()", role: "保存寄存器 6", contextField: "s6" },
  { name: "x23", alias: "s7", type: "callee", savedBy: "swtch()", role: "保存寄存器 7", contextField: "s7" },
  { name: "x24", alias: "s8", type: "callee", savedBy: "swtch()", role: "保存寄存器 8", contextField: "s8" },
  { name: "x25", alias: "s9", type: "callee", savedBy: "swtch()", role: "保存寄存器 9", contextField: "s9" },
  { name: "x26", alias: "s10", type: "callee", savedBy: "swtch()", role: "保存寄存器 10", contextField: "s10" },
  { name: "x27", alias: "s11", type: "callee", savedBy: "swtch()", role: "保存寄存器 11", contextField: "s11" },
  { name: "x28", alias: "t3", type: "caller", savedBy: "编译器 (栈)", role: "临时寄存器 3" },
  { name: "x29", alias: "t4", type: "caller", savedBy: "编译器 (栈)", role: "临时寄存器 4" },
  { name: "x30", alias: "t5", type: "caller", savedBy: "编译器 (栈)", role: "临时寄存器 5" },
  { name: "x31", alias: "t6", type: "caller", savedBy: "编译器 (栈)", role: "临时寄存器 6" },
];

const typeColor: Record<SaveType, { bg: string; border: string; text: string; label: string }> = {
  callee: {
    bg: "bg-green-100 dark:bg-green-900/40",
    border: "border-green-500 dark:border-green-400",
    text: "text-green-800 dark:text-green-200",
    label: "Callee-saved (swtch)",
  },
  caller: {
    bg: "bg-blue-100 dark:bg-blue-900/40",
    border: "border-blue-500 dark:border-blue-400",
    text: "text-blue-800 dark:text-blue-200",
    label: "Caller-saved (compiler)",
  },
  special: {
    bg: "bg-gray-100 dark:bg-gray-700",
    border: "border-gray-400 dark:border-gray-500",
    text: "text-gray-600 dark:text-gray-300",
    label: "Special (not saved)",
  },
};

export default function RegisterSaveRestoreVisualizer() {
  const [selected, setSelected] = useState<number | null>(null);
  const [filter, setFilter] = useState<SaveType | "all">("all");

  const filtered = filter === "all" ? REGISTERS : REGISTERS.filter((r) => r.type === filter);

  const sel = selected !== null ? REGISTERS[selected] : null;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center flex items-center justify-center gap-2">
        <Cpu className="w-7 h-7 text-emerald-600 dark:text-emerald-400" />
        RISC-V Register Save / Restore Visualization
      </h2>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-4 mb-4">
        {(Object.entries(typeColor) as [SaveType, (typeof typeColor)[SaveType]][]).map(
          ([key, c]) => (
            <button
              key={key}
              onClick={() => setFilter(filter === key ? "all" : key)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold transition-all border-2 ${
                filter === key
                  ? `${c.bg} ${c.border} ${c.text}`
                  : "bg-white dark:bg-gray-800 border-slate-200 dark:border-gray-600 text-slate-600 dark:text-gray-300"
              }`}
            >
              <span
                className={`w-3 h-3 rounded-sm ${
                  key === "callee"
                    ? "bg-green-500"
                    : key === "caller"
                    ? "bg-blue-500"
                    : "bg-gray-400"
                }`}
              />
              {c.label}
            </button>
          )
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Register grid */}
        <div className="lg:col-span-2">
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
            {filtered.map((reg, idx) => {
              const globalIdx = REGISTERS.indexOf(reg);
              const c = typeColor[reg.type];
              const isSelected = selected === globalIdx;
              return (
                <motion.button
                  key={reg.name}
                  onClick={() => setSelected(isSelected ? null : globalIdx)}
                  whileHover={{ scale: 1.08 }}
                  whileTap={{ scale: 0.95 }}
                  animate={{
                    scale: isSelected ? 1.1 : 1,
                    boxShadow: isSelected
                      ? "0 4px 12px rgba(0,0,0,0.2)"
                      : "0 1px 3px rgba(0,0,0,0.1)",
                  }}
                  className={`relative p-2 rounded-lg border-2 text-center transition-colors ${c.bg} ${
                    isSelected ? `${c.border} ring-2 ring-offset-1 ring-${reg.type === "callee" ? "green" : reg.type === "caller" ? "blue" : "gray"}-400` : "border-transparent"
                  }`}
                >
                  <div className={`text-xs font-bold ${c.text}`}>
                    {reg.alias}
                  </div>
                  <div className="text-[10px] text-slate-500 dark:text-gray-400">
                    {reg.name}
                  </div>
                  {reg.contextField && (
                    <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-amber-400 border border-white dark:border-gray-800" />
                  )}
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Detail panel */}
        <div className="lg:col-span-1">
          <AnimatePresence mode="wait">
            {sel ? (
              <motion.div
                key={selected}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className={`p-4 rounded-lg border-2 ${typeColor[sel.type].bg} ${typeColor[sel.type].border}`}
              >
                <h3 className="font-bold text-lg text-slate-800 dark:text-gray-100">
                  {sel.alias}{" "}
                  <span className="text-sm font-normal text-slate-500 dark:text-gray-400">
                    ({sel.name})
                  </span>
                </h3>
                <div className="mt-3 space-y-2 text-sm">
                  <div>
                    <span className="font-semibold text-slate-700 dark:text-gray-300">
                      Role:
                    </span>{" "}
                    <span className="text-slate-600 dark:text-gray-400">
                      {sel.role}
                    </span>
                  </div>
                  <div>
                    <span className="font-semibold text-slate-700 dark:text-gray-300">
                      Category:
                    </span>{" "}
                    <span className={typeColor[sel.type].text}>
                      {typeColor[sel.type].label}
                    </span>
                  </div>
                  <div>
                    <span className="font-semibold text-slate-700 dark:text-gray-300">
                      Saved by:
                    </span>{" "}
                    <span className="text-slate-600 dark:text-gray-400">
                      {sel.savedBy}
                    </span>
                  </div>
                  {sel.contextField && (
                    <div className="flex items-center gap-1.5 mt-2 p-2 bg-amber-50 dark:bg-amber-900/30 rounded">
                      <Save className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                      <span className="text-xs font-mono text-amber-800 dark:text-amber-300">
                        context-&gt;{sel.contextField}
                      </span>
                    </div>
                  )}
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="p-4 rounded-lg border-2 border-dashed border-slate-300 dark:border-gray-600 text-center"
              >
                <Info className="w-8 h-8 text-slate-400 dark:text-gray-500 mx-auto mb-2" />
                <p className="text-sm text-slate-500 dark:text-gray-400">
                  Click a register to see details
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Context struct mapping */}
          <div className="mt-4 p-4 rounded-lg bg-white dark:bg-gray-800 border border-slate-200 dark:border-gray-700">
            <h4 className="text-sm font-bold text-slate-700 dark:text-gray-300 mb-2">
              struct context (xv6)
            </h4>
            <div className="space-y-1 font-mono text-xs">
              {REGISTERS.filter((r) => r.contextField).map((r) => (
                <div
                  key={r.name}
                  className={`flex justify-between px-2 py-0.5 rounded cursor-pointer transition-colors ${
                    selected === REGISTERS.indexOf(r)
                      ? "bg-amber-100 dark:bg-amber-900/30"
                      : "hover:bg-slate-50 dark:hover:bg-gray-700"
                  }`}
                  onClick={() => setSelected(REGISTERS.indexOf(r))}
                >
                  <span className="text-green-700 dark:text-green-300">
                    uint64 {r.contextField};
                  </span>
                  <span className="text-slate-500 dark:text-gray-400">
                    {r.alias}
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[10px] text-slate-400 dark:text-gray-500">
              14 fields: ra, sp, s0-s11 (callee-saved registers)
            </p>
          </div>
        </div>
      </div>

      {/* Summary bar */}
      <div className="mt-4 grid grid-cols-3 gap-3 text-center text-xs">
        <div className="p-2 rounded-lg bg-green-50 dark:bg-green-900/20">
          <div className="font-bold text-green-700 dark:text-green-300">14</div>
          <div className="text-green-600 dark:text-green-400">Callee-saved (swtch)</div>
        </div>
        <div className="p-2 rounded-lg bg-blue-50 dark:bg-blue-900/20">
          <div className="font-bold text-blue-700 dark:text-blue-300">15</div>
          <div className="text-blue-600 dark:text-blue-400">Caller-saved (compiler)</div>
        </div>
        <div className="p-2 rounded-lg bg-gray-50 dark:bg-gray-700/50">
          <div className="font-bold text-gray-700 dark:text-gray-300">3</div>
          <div className="text-gray-600 dark:text-gray-400">Special (not saved)</div>
        </div>
      </div>
    </div>
  );
}
