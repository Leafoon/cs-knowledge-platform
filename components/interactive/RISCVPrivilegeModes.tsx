"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield,
  ChevronDown,
  ChevronUp,
  ArrowDown,
  ArrowUp,
  Lock,
  Unlock,
  Settings,
  Cpu,
} from "lucide-react";

interface CSR {
  name: string;
  fullName: string;
  description: string;
  example: string;
}

interface Mode {
  id: string;
  name: string;
  shortName: string;
  level: number;
  color: string;
  darkColor: string;
  bg: string;
  darkBg: string;
  borderColor: string;
  darkBorderColor: string;
  icon: React.ReactNode;
  description: string;
  capabilities: string[];
  xv6Usage: string;
  csrs: CSR[];
  switchDown?: string;
  switchUp?: string;
}

const MODES: Mode[] = [
  {
    id: "M",
    name: "Machine Mode (M-mode)",
    shortName: "M-mode",
    level: 3,
    color: "text-red-700 dark:text-red-300",
    darkColor: "",
    bg: "bg-red-50",
    darkBg: "dark:bg-red-900/30",
    borderColor: "border-red-300",
    darkBorderColor: "dark:border-red-700",
    icon: <Shield className="w-6 h-6" />,
    description: "最高特权模式，拥有对硬件的完全访问权限。M-mode 是 RISC-V 中唯一必须实现的模式，固件和引导加载程序运行在此模式。",
    capabilities: [
      "可以访问所有物理内存",
      "可以访问所有 CSR 寄存器",
      "可以执行所有特权指令",
      "处理最底层的中断和异常",
      "不可被更低特权级中断（除非配置 mideleg）",
    ],
    xv6Usage: "xv6 中仅在启动阶段使用：start.c 配置硬件，然后 mret 切换到 S-mode。时钟中断处理程序（timervec）也在 M-mode 运行。",
    csrs: [
      { name: "mstatus", fullName: "Machine Status", description: "M-mode 状态寄存器，包含全局中断使能 MIE 等", example: "w_mstatus(r_mstatus() & ~MSTATUS_MIE)" },
      { name: "mepc", fullName: "Machine Exception PC", description: "保存异常发生时的 PC，mret 从此恢复", example: "w_mepc((uint64)main)" },
      { name: "mcause", fullName: "Machine Cause", description: "记录异常/中断的原因编号", example: "中断时最高位=1" },
      { name: "mtvec", fullName: "Machine Trap Vector", description: "M-mode 陷阱处理程序入口地址", example: "w_mtvec((uint64)timervec)" },
      { name: "medeleg", fullName: "Machine Exception Delegation", description: "异常委托寄存器，控制哪些异常交给 S-mode", example: "w_medeleg(0xffff)" },
      { name: "mideleg", fullName: "Machine Interrupt Delegation", description: "中断委托寄存器，控制哪些中断交给 S-mode", example: "w_mideleg(0xffff)" },
      { name: "mhartid", fullName: "Machine Hart ID", description: "当前硬件线程（hart）的唯一 ID", example: "csrr a1, mhartid" },
    ],
    switchDown: "mret：将 mepc 加载到 PC，切换到 S-mode 或 U-mode",
  },
  {
    id: "S",
    name: "Supervisor Mode (S-mode)",
    shortName: "S-mode",
    level: 2,
    color: "text-blue-700 dark:text-blue-300",
    darkColor: "",
    bg: "bg-blue-50",
    darkBg: "dark:bg-blue-900/30",
    borderColor: "border-blue-300",
    darkBorderColor: "dark:border-blue-700",
    icon: <Settings className="w-6 h-6" />,
    description: "操作系统内核运行的特权模式。S-mode 可以管理页表、处理中断和异常、控制系统调用，但不能直接访问 M-mode 的 CSR。",
    capabilities: [
      "管理虚拟内存页表（satp）",
      "处理系统调用（ecall from U-mode）",
      "处理设备中断（如果委托给 S-mode）",
      "执行 sret 返回用户态",
      "不能直接配置硬件（需通过 M-mode 委托）",
    ],
    xv6Usage: "xv6 内核的所有代码都运行在 S-mode：进程管理、内存管理、文件系统、设备驱动、陷阱处理等。",
    csrs: [
      { name: "sstatus", fullName: "Supervisor Status", description: "S-mode 状态寄存器，包含 SIE（中断使能）等位", example: "w_sstatus(r_sstatus() | SSTATUS_SIE)" },
      { name: "sepc", fullName: "Supervisor Exception PC", description: "保存异常发生时的用户程序 PC", example: "w_sepc(p->trapframe->epc)" },
      { name: "scause", fullName: "Supervisor Cause", description: "记录 trap 的原因（中断/异常+编号）", example: "8 = 系统调用, 13 = load page fault" },
      { name: "stvec", fullName: "Supervisor Trap Vector", description: "S-mode 陷阱处理程序入口地址", example: "w_stvec((uint64)kernelvec)" },
      { name: "sscratch", fullName: "Supervisor Scratch", description: "临时寄存器，xv6 存放 trapframe 地址", example: "w_sscratch((uint64)p->trapframe)" },
      { name: "stval", fullName: "Supervisor Trap Value", description: "trap 附加信息（如 page fault 的虚拟地址）", example: "r_stval() 获取 fault 地址" },
      { name: "satp", fullName: "Supervisor Address Translation", description: "页表控制：MODE=Sv39, PPN=根页表物理页号", example: "w_satp(MAKE_SATP(kernel_pagetable))" },
      { name: "sie", fullName: "Supervisor Interrupt Enable", description: "S-mode 中断使能位", example: "w_sie(r_sie() | SIE_SEIE | SIE_STIE)" },
    ],
    switchDown: "sret：将 sepc 加载到 PC，切换到 U-mode",
    switchUp: "ecall / 中断 / 异常：硬件自动保存 PC 到 sepc，切换到 S-mode",
  },
  {
    id: "U",
    name: "User Mode (U-mode)",
    shortName: "U-mode",
    level: 1,
    color: "text-green-700 dark:text-green-300",
    darkColor: "",
    bg: "bg-green-50",
    darkBg: "dark:bg-green-900/30",
    borderColor: "border-green-300",
    darkBorderColor: "dark:border-green-700",
    icon: <Cpu className="w-6 h-6" />,
    description: "应用程序运行的低特权模式。U-mode 只能访问自己的虚拟地址空间，不能执行特权指令，不能直接访问硬件。需要内核服务时通过 ecall 系统调用。",
    capabilities: [
      "只能访问自己的虚拟地址空间",
      "可以执行普通计算指令",
      "通过 ecall 发起系统调用",
      "不能修改页表、不能禁用中断",
      "不能执行特权指令（会触发 illegal instruction 异常）",
    ],
    xv6Usage: "所有用户程序（init.c, sh.c, cat.c 等）都运行在 U-mode。通过系统调用（fork, exec, read, write 等）请求内核服务。",
    csrs: [
      { name: "（无）", fullName: "U-mode 没有 CSR", description: "U-mode 不能直接访问任何 CSR 寄存器，这是特权隔离的核心", example: "尝试读 CSR 会触发 illegal instruction 异常" },
    ],
    switchUp: "ecall：用户程序主动发起系统调用，切换到 S-mode",
  },
];

export default function RISCVPrivilegeModes() {
  const [selectedMode, setSelectedMode] = useState<string | null>(null);
  const [showTransitions, setShowTransitions] = useState(true);

  const selected = MODES.find((m) => m.id === selectedMode) || null;

  return (
    <div className="max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-1">
        RISC-V 特权模式与 CSR 寄存器
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-5">
        点击特权模式查看详情和 CSR 寄存器
      </p>

      <div className="flex flex-col gap-0 relative">
        {[...MODES].reverse().map((mode, idx) => {
          const isActive = selectedMode === mode.id;
          return (
            <div key={mode.id}>
              <motion.button
                onClick={() => setSelectedMode(isActive ? null : mode.id)}
                className={`w-full text-left p-4 rounded-xl border-2 transition-all ${mode.bg} ${mode.darkBg} ${isActive ? `${mode.borderColor} ${mode.darkBorderColor} shadow-lg` : "border-transparent hover:border-slate-200 dark:hover:border-gray-700"}`}
                whileHover={{ scale: 1.005 }}
                whileTap={{ scale: 0.995 }}
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${mode.bg} ${mode.darkBg} ${mode.color}`}>
                    {mode.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className={`font-bold text-base ${mode.color}`}>
                        {mode.name}
                      </span>
                      <span className="text-xs text-slate-400 dark:text-slate-500 font-mono">
                        Level {mode.level}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                      {mode.description.slice(0, 80)}...
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-1 rounded-full ${mode.bg} ${mode.darkBg} ${mode.color} ${mode.borderColor} ${mode.darkBorderColor} border font-mono`}>
                      {mode.csrs.length} CSR
                    </span>
                    {isActive ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                  </div>
                </div>
              </motion.button>

              {showTransitions && idx < MODES.length - 1 && (
                <div className="flex items-center justify-center gap-2 py-2 relative">
                  <div className="absolute left-8 right-8 h-px bg-slate-200 dark:bg-gray-700" style={{ top: "50%" }} />
                  <div className="relative z-10 flex gap-2">
                    <div className="flex items-center gap-1 px-2 py-0.5 bg-white dark:bg-gray-800 rounded-full border border-slate-200 dark:border-gray-700 text-xs text-slate-500 dark:text-slate-400">
                      <ArrowDown className="w-3 h-3 text-green-500" />
                      <span>mret / sret</span>
                    </div>
                    <div className="flex items-center gap-1 px-2 py-0.5 bg-white dark:bg-gray-800 rounded-full border border-slate-200 dark:border-gray-700 text-xs text-slate-500 dark:text-slate-400">
                      <ArrowUp className="w-3 h-3 text-red-500" />
                      <span>ecall / trap</span>
                    </div>
                  </div>
                </div>
              )}

              <AnimatePresence>
                {isActive && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="overflow-hidden"
                  >
                    <div className={`p-5 rounded-xl border ${mode.borderColor} ${mode.darkBorderColor} ${mode.bg} ${mode.darkBg} mb-2`}>
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <div>
                          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2 flex items-center gap-1">
                            <Unlock className="w-4 h-4" /> 权限能力
                          </h4>
                          <ul className="space-y-1 mb-3">
                            {mode.capabilities.map((c, i) => (
                              <li key={i} className="text-xs text-slate-600 dark:text-slate-400 flex items-start gap-2">
                                <span className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${mode.id === "M" ? "bg-red-400" : mode.id === "S" ? "bg-blue-400" : "bg-green-400"}`} />
                                {c}
                              </li>
                            ))}
                          </ul>
                          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2 flex items-center gap-1">
                            <Cpu className="w-4 h-4" /> xv6 中的使用
                          </h4>
                          <p className="text-xs text-slate-600 dark:text-slate-400">{mode.xv6Usage}</p>
                        </div>

                        <div>
                          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2 flex items-center gap-1">
                            <Settings className="w-4 h-4" /> CSR 寄存器
                          </h4>
                          <div className="space-y-2">
                            {mode.csrs.map((csr, i) => (
                              <div key={i} className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-2 border border-slate-200 dark:border-gray-600">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-100 bg-slate-100 dark:bg-gray-700 px-1.5 py-0.5 rounded">
                                    {csr.name}
                                  </span>
                                  <span className="text-xs text-slate-500 dark:text-slate-400">{csr.fullName}</span>
                                </div>
                                <p className="text-xs text-slate-600 dark:text-slate-400">{csr.description}</p>
                                <code className="text-xs text-emerald-600 dark:text-emerald-400 mt-1 block">{csr.example}</code>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      {mode.switchDown && (
                        <div className="mt-3 flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
                          <ArrowDown className="w-3 h-3 text-green-500" />
                          <span>{mode.switchDown}</span>
                        </div>
                      )}
                      {mode.switchUp && (
                        <div className="mt-1 flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
                          <ArrowUp className="w-3 h-3 text-red-500" />
                          <span>{mode.switchUp}</span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
        <p className="text-xs text-amber-800 dark:text-amber-300">
          <strong>xv6 使用约定：</strong>M-mode 仅在启动和时钟中断时使用；内核运行在 S-mode；用户程序运行在 U-mode。
          特权级通过 ecall（向上）和 mret/sret（向下）切换。
        </p>
      </div>
    </div>
  );
}
