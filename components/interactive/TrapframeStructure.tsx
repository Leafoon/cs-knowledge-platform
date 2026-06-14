"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, Info, HardDrive, CircuitBoard } from "lucide-react";

type FieldCategory = "kernel" | "saved_pc" | "saved_regs" | "temp_regs" | "arg_regs";

interface TrapField {
  offset: number;
  field: string;
  riscvReg: string;
  category: FieldCategory;
  size: number;
  hardwareSaved: boolean;
  description: string;
  whenSaved: string;
  chineseDetail: string;
}

const trapFields: TrapField[] = [
  { offset: 0, field: "kernel_satp", riscvReg: "CSR satp", category: "kernel", size: 8, hardwareSaved: false, description: "Kernel page table", whenSaved: "usertrap() 软件保存", chineseDetail: "内核页表的 satp 寄存器值，用于在 trap 处理期间进行内核地址翻译。usertrap() 入口时软件写入。" },
  { offset: 8, field: "kernel_sp", riscvReg: "stack pointer", category: "kernel", size: 8, hardwareSaved: false, description: "Top of process's kernel stack", whenSaved: "usertrap() 软件保存", chineseDetail: "进程内核栈顶地址。每个进程有独立的内核栈，trap 发生时切换到此栈。" },
  { offset: 16, field: "kernel_trap", riscvReg: "function ptr", category: "kernel", size: 8, hardwareSaved: false, description: "usertrap() address", whenSaved: "usertrap() 软件保存", chineseDetail: "usertrap() 函数的地址。trap 入口代码通过此字段跳转到 C 语言的 trap 处理函数。" },
  { offset: 24, field: "kernel_hartid", riscvReg: "a0 (tp)", category: "kernel", size: 8, hardwareSaved: false, description: "hartid for cpuid()", whenSaved: "usertrap() 软件保存", chineseDetail: "当前硬件线程 (hart) 的 ID。cpuid() 通过读取此字段获取 CPU 编号。" },
  { offset: 32, field: "epc", riscvReg: "CSR sepc", category: "saved_pc", size: 8, hardwareSaved: true, description: "Saved user program counter", whenSaved: "硬件自动保存到 sepc", chineseDetail: "用户态程序计数器。发生 trap 时硬件自动将 PC 写入 sepc，返回用户态时 sret 从 sepc 恢复。" },
  { offset: 40, field: "ra", riscvReg: "x1 (ra)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Return address", whenSaved: "uservec 软件保存", chineseDetail: "返回地址寄存器。用于函数调用返回，uservec 中通过 sd 指令保存到 trapframe。" },
  { offset: 48, field: "sp", riscvReg: "x2 (sp)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Stack pointer", whenSaved: "uservec 软件保存", chineseDetail: "用户态栈指针。trap 发生后在 uservec 中保存，返回用户态时恢复。" },
  { offset: 56, field: "gp", riscvReg: "x3 (gp)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Global pointer", whenSaved: "uservec 软件保存", chineseDetail: "全局指针，指向全局数据区。通常在程序启动时设置，trap 时保存以防被破坏。" },
  { offset: 64, field: "tp", riscvReg: "x4 (tp)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Thread pointer", whenSaved: "uservec 软件保存", chineseDetail: "线程指针。在 xv6 RISC-V 中存储 hartid，用于多核识别。" },
  { offset: 72, field: "t0", riscvReg: "x5 (t0)", category: "temp_regs", size: 8, hardwareSaved: false, description: "Temporary register", whenSaved: "uservec 软件保存", chineseDetail: "临时寄存器 t0，caller-saved。函数调用者负责保存，trap 时需要保存到 trapframe。" },
  { offset: 80, field: "t1", riscvReg: "x6 (t1)", category: "temp_regs", size: 8, hardwareSaved: false, description: "Temporary register", whenSaved: "uservec 软件保存", chineseDetail: "临时寄存器 t1，caller-saved。用于临时计算，trap 处理期间可能被内核使用。" },
  { offset: 88, field: "t2", riscvReg: "x7 (t2)", category: "temp_regs", size: 8, hardwareSaved: false, description: "Temporary register", whenSaved: "uservec 软件保存", chineseDetail: "临时寄存器 t2，caller-saved。" },
  { offset: 96, field: "s0/fp", riscvReg: "x8 (s0)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved / frame pointer", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s0 / 帧指针 fp。callee-saved，同时用作栈帧指针，用于调试回溯。" },
  { offset: 104, field: "s1", riscvReg: "x9 (s1)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s1，callee-saved。被调用者保证值不变。" },
  { offset: 112, field: "a0", riscvReg: "x10 (a0)", category: "arg_regs", size: 8, hardwareSaved: false, description: "Arg0 / return value", whenSaved: "uservec 软件保存", chineseDetail: "参数寄存器 0 / 返回值。系统调用号或返回结果。用户态传递第一个参数，内核返回时写入返回值。" },
  { offset: 120, field: "a1", riscvReg: "x11 (a1)", category: "arg_regs", size: 8, hardwareSaved: false, description: "Argument 1", whenSaved: "uservec 软件保存", chineseDetail: "参数寄存器 1。系统调用的第一个参数。" },
  { offset: 128, field: "a2", riscvReg: "x12 (a2)", category: "arg_regs", size: 8, hardwareSaved: false, description: "Argument 2", whenSaved: "uservec 软件保存", chineseDetail: "参数寄存器 2。系统调用的第二个参数。" },
  { offset: 136, field: "a3", riscvReg: "x13 (a3)", category: "arg_regs", size: 8, hardwareSaved: false, description: "Argument 3", whenSaved: "uservec 软件保存", chineseDetail: "参数寄存器 3。系统调用的第三个参数。" },
  { offset: 144, field: "a4", riscvReg: "x14 (a4)", category: "arg_regs", size: 8, hardwareSaved: false, description: "Argument 4", whenSaved: "uservec 软件保存", chineseDetail: "参数寄存器 4。" },
  { offset: 152, field: "a5", riscvReg: "x15 (a5)", category: "arg_regs", size: 8, hardwareSaved: false, description: "Argument 5", whenSaved: "uservec 软件保存", chineseDetail: "参数寄存器 5。" },
  { offset: 160, field: "a6", riscvReg: "x16 (a6)", category: "arg_regs", size: 8, hardwareSaved: false, description: "Argument 6", whenSaved: "uservec 软件保存", chineseDetail: "参数寄存器 6。" },
  { offset: 168, field: "a7", riscvReg: "x17 (a7)", category: "arg_regs", size: 8, hardwareSaved: false, description: "Arg7 / syscall number", whenSaved: "uservec 软件保存", chineseDetail: "参数寄存器 7 / 系统调用号。ecall 时 a7 包含系统调用编号（如 SYS_write = 16）。" },
  { offset: 176, field: "s2", riscvReg: "x18 (s2)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s2，callee-saved。" },
  { offset: 184, field: "s3", riscvReg: "x19 (s3)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s3，callee-saved。" },
  { offset: 192, field: "s4", riscvReg: "x20 (s4)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s4，callee-saved。" },
  { offset: 200, field: "s5", riscvReg: "x21 (s5)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s5，callee-saved。" },
  { offset: 208, field: "s6", riscvReg: "x22 (s6)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s6，callee-saved。" },
  { offset: 216, field: "s7", riscvReg: "x23 (s7)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s7，callee-saved。" },
  { offset: 224, field: "s8", riscvReg: "x24 (s8)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s8，callee-saved。" },
  { offset: 232, field: "s9", riscvReg: "x25 (s9)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s9，callee-saved。" },
  { offset: 240, field: "s10", riscvReg: "x26 (s10)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s10，callee-saved。" },
  { offset: 248, field: "s11", riscvReg: "x27 (s11)", category: "saved_regs", size: 8, hardwareSaved: false, description: "Saved register", whenSaved: "uservec 软件保存", chineseDetail: "保存寄存器 s11，callee-saved。" },
  { offset: 256, field: "t3", riscvReg: "x28 (t3)", category: "temp_regs", size: 8, hardwareSaved: false, description: "Temporary register", whenSaved: "uservec 软件保存", chineseDetail: "临时寄存器 t3，caller-saved。" },
  { offset: 264, field: "t4", riscvReg: "x29 (t4)", category: "temp_regs", size: 8, hardwareSaved: false, description: "Temporary register", whenSaved: "uservec 软件保存", chineseDetail: "临时寄存器 t4，caller-saved。" },
  { offset: 272, field: "t5", riscvReg: "x30 (t5)", category: "temp_regs", size: 8, hardwareSaved: false, description: "Temporary register", whenSaved: "uservec 软件保存", chineseDetail: "临时寄存器 t5，caller-saved。" },
  { offset: 280, field: "t6", riscvReg: "x31 (t6)", category: "temp_regs", size: 8, hardwareSaved: false, description: "Temporary register", whenSaved: "uservec 软件保存", chineseDetail: "临时寄存器 t6，caller-saved。" },
];

const categoryMeta: Record<FieldCategory, { label: string; bar: string; dot: string; bg: string; border: string; text: string; darkBg: string; darkBorder: string; darkText: string }> = {
  kernel: {
    label: "内核元数据",
    bar: "bg-red-400 dark:bg-red-600",
    dot: "bg-red-400",
    bg: "bg-red-50", border: "border-red-400", text: "text-red-700",
    darkBg: "dark:bg-red-950/40", darkBorder: "dark:border-red-700", darkText: "dark:text-red-300",
  },
  saved_pc: {
    label: "保存的 PC",
    bar: "bg-amber-400 dark:bg-amber-600",
    dot: "bg-amber-400",
    bg: "bg-amber-50", border: "border-amber-400", text: "text-amber-700",
    darkBg: "dark:bg-amber-950/40", darkBorder: "dark:border-amber-700", darkText: "dark:text-amber-300",
  },
  saved_regs: {
    label: "保存寄存器 (callee-saved)",
    bar: "bg-blue-400 dark:bg-blue-600",
    dot: "bg-blue-400",
    bg: "bg-blue-50", border: "border-blue-400", text: "text-blue-700",
    darkBg: "dark:bg-blue-950/40", darkBorder: "dark:border-blue-700", darkText: "dark:text-blue-300",
  },
  temp_regs: {
    label: "临时寄存器 (caller-saved)",
    bar: "bg-sky-400 dark:bg-sky-600",
    dot: "bg-sky-400",
    bg: "bg-sky-50", border: "border-sky-400", text: "text-sky-700",
    darkBg: "dark:bg-sky-950/40", darkBorder: "dark:border-sky-700", darkText: "dark:text-sky-300",
  },
  arg_regs: {
    label: "参数寄存器",
    bar: "bg-violet-400 dark:bg-violet-600",
    dot: "bg-violet-400",
    bg: "bg-violet-50", border: "border-violet-400", text: "text-violet-700",
    darkBg: "dark:bg-violet-950/40", darkBorder: "dark:border-violet-700", darkText: "dark:text-violet-300",
  },
};

const categories: FieldCategory[] = ["kernel", "saved_pc", "saved_regs", "arg_regs", "temp_regs"];

export default function TrapframeStructure() {
  const [selectedField, setSelectedField] = useState<string>("kernel_satp");
  const [activeFilter, setActiveFilter] = useState<FieldCategory | "all">("all");
  const [showHex, setShowHex] = useState(false);

  const filteredFields = activeFilter === "all" ? trapFields : trapFields.filter(f => f.category === activeFilter);
  const selected = trapFields.find(f => f.field === selectedField);

  const getRowClass = (f: TrapField, isSelected: boolean) => {
    const meta = categoryMeta[f.category];
    if (isSelected) {
      return `${meta.bg} ${meta.border} ${meta.darkBg} ${meta.darkBorder} shadow-md`;
    }
    return `bg-white border-slate-200 hover:border-slate-300 dark:bg-slate-800/60 dark:border-slate-700 dark:hover:border-slate-500`;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-indigo-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2 text-center flex items-center justify-center gap-2">
        <CircuitBoard className="w-7 h-7 text-indigo-600 dark:text-indigo-400" />
        xv6 RISC-V trapframe 结构
      </h3>
      <p className="text-center text-sm text-slate-500 dark:text-slate-400 mb-5">
        struct trapframe — 共 {trapFields.length * 8} 字节，{trapFields.length} 个字段
      </p>

      {/* Category Filter */}
      <div className="flex flex-wrap gap-2 mb-5 justify-center">
        <button
          onClick={() => setActiveFilter("all")}
          className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all border ${
            activeFilter === "all"
              ? "bg-slate-800 text-white border-slate-800 dark:bg-slate-200 dark:text-slate-900 dark:border-slate-200"
              : "bg-white text-slate-600 border-slate-300 hover:bg-slate-100 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-600 dark:hover:bg-slate-700"
          }`}
        >
          全部
        </button>
        {categories.map(cat => {
          const meta = categoryMeta[cat];
          const isActive = activeFilter === cat;
          return (
            <button
              key={cat}
              onClick={() => setActiveFilter(cat)}
              className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all border ${
                isActive
                  ? `${meta.bg} ${meta.text} ${meta.border} ${meta.darkBg} ${meta.darkText} ${meta.darkBorder}`
                  : "bg-white text-slate-600 border-slate-300 hover:bg-slate-100 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-600 dark:hover:bg-slate-700"
              }`}
            >
              {meta.label}
            </button>
          );
        })}
      </div>

      {/* Toggle: hex / decimal offset */}
      <div className="flex justify-end mb-3">
        <button
          onClick={() => setShowHex(!showHex)}
          className="text-xs px-2.5 py-1 rounded border border-slate-300 text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-400 dark:hover:bg-slate-700 transition-colors"
        >
          {showHex ? "十进制偏移" : "十六进制偏移"}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Memory Layout */}
        <div className="lg:col-span-2 space-y-1">
          <AnimatePresence mode="popLayout">
            {filteredFields.map((f, idx) => {
              const meta = categoryMeta[f.category];
              const isSelected = selectedField === f.field;
              return (
                <motion.div
                  key={f.field}
                  layout
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  transition={{ duration: 0.2, delay: idx * 0.015 }}
                  whileHover={{ scale: 1.01 }}
                  onClick={() => setSelectedField(f.field)}
                  className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all border-2 ${getRowClass(f, isSelected)}`}
                >
                  <span className="text-xs font-mono w-12 text-right text-slate-400 dark:text-slate-500 shrink-0">
                    {showHex ? `0x${f.offset.toString(16).padStart(2, "0")}` : f.offset}
                  </span>
                  <span className={`w-1 h-6 rounded-full shrink-0 ${meta.bar}`} />
                  <span className="font-mono font-semibold text-slate-800 dark:text-slate-100 w-24 shrink-0">{f.field}</span>
                  <span className="text-xs text-slate-500 dark:text-slate-400 w-20 shrink-0">{f.size} 字节</span>
                  <span className="text-sm text-slate-600 dark:text-slate-300 flex-1 truncate">{f.description}</span>
                  {f.hardwareSaved ? (
                    <span title="硬件保存"><HardDrive className="w-4 h-4 text-emerald-500 dark:text-emerald-400 shrink-0" /></span>
                  ) : (
                    <span title="软件保存"><Cpu className="w-4 h-4 text-slate-400 dark:text-slate-500 shrink-0" /></span>
                  )}
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>

        {/* Detail Panel */}
        <div className="lg:col-span-1">
          <AnimatePresence mode="wait">
            {selected && (
              <motion.div
                key={selected.field}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`p-5 rounded-lg border-2 sticky top-4 ${categoryMeta[selected.category].bg} ${categoryMeta[selected.category].border} ${categoryMeta[selected.category].darkBg} ${categoryMeta[selected.category].darkBorder}`}
              >
                <div className="flex items-center gap-2 mb-4">
                  <Info className="w-5 h-5 text-slate-700 dark:text-slate-200" />
                  <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">字段详情</h4>
                </div>
                <div className="space-y-3 text-sm">
                  <DetailRow label="字段名" value={selected.field} mono />
                  <DetailRow label="RISC-V 寄存器" value={selected.riscvReg} mono />
                  <DetailRow label="偏移量" value={`${selected.offset} (0x${selected.offset.toString(16)})`} mono />
                  <DetailRow label="大小" value={`${selected.size} 字节`} />
                  <DetailRow label="分类" value={categoryMeta[selected.category].label} />
                  <DetailRow
                    label="保存方式"
                    value={selected.hardwareSaved ? "硬件自动保存" : "软件保存 (uservec)"}
                    badge={selected.hardwareSaved}
                  />
                  <DetailRow label="何时保存" value={selected.whenSaved} />
                  <div>
                    <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 block mb-1">详细说明</span>
                    <p className="text-slate-700 dark:text-slate-200 leading-relaxed">{selected.chineseDetail}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* C struct preview */}
      <div className="mt-6 bg-slate-900 dark:bg-slate-950 rounded-lg p-4 overflow-x-auto">
        <pre className="text-sm text-green-400 font-mono leading-relaxed">
{`// kernel/trapframe.h  (xv6-riscv)
struct trapframe {
${trapFields.map(f =>
  `  uint64 ${f.field.padEnd(16)}; // ${showHex ? "0x" + f.offset.toString(16).padStart(2, "0") : String(f.offset).padStart(3, " ")}  ${selectedField === f.field ? "<<< " : ""}${f.description}`
).join("\n")}
};`}
        </pre>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap items-center gap-4 text-xs text-slate-500 dark:text-slate-400 justify-center">
        <span className="flex items-center gap-1"><HardDrive className="w-3.5 h-3.5 text-emerald-500" /> 硬件自动保存</span>
        <span className="flex items-center gap-1"><Cpu className="w-3.5 h-3.5" /> 软件保存</span>
        {categories.map(cat => (
          <span key={cat} className="flex items-center gap-1.5">
            <span className={`w-2.5 h-2.5 rounded-full ${categoryMeta[cat].dot}`} />
            {categoryMeta[cat].label}
          </span>
        ))}
      </div>
    </div>
  );
}

function DetailRow({ label, value, mono, badge }: { label: string; value: string; mono?: boolean; badge?: boolean }) {
  return (
    <div>
      <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 block mb-0.5">{label}</span>
      {badge ? (
        <span className="inline-block px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300">
          {value}
        </span>
      ) : (
        <p className={`text-slate-800 dark:text-slate-100 ${mono ? "font-mono" : ""}`}>{value}</p>
      )}
    </div>
  );
}
