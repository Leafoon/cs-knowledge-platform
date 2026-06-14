"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Database, HardDrive, Zap, Settings, Shield, FileText } from "lucide-react";

interface IDTEntry {
  vector: number;
  type: string;
  description: string;
  dpl: number;
  handler: string;
}

const idtEntries: IDTEntry[] = [
  { vector: 0, type: "Trap", description: "除零错误 (#DE)", dpl: 0, handler: "divide_error" },
  { vector: 1, type: "Trap", description: "调试异常 (#DB)", dpl: 0, handler: "debug" },
  { vector: 2, type: "Interrupt", description: "NMI", dpl: 0, handler: "nmi" },
  { vector: 3, type: "Trap", description: "断点 (#BP)", dpl: 3, handler: "int3" },
  { vector: 4, type: "Trap", description: "溢出 (#OF)", dpl: 3, handler: "overflow" },
  { vector: 6, type: "Trap", description: "非法指令 (#UD)", dpl: 0, handler: "invalid_op" },
  { vector: 8, type: "Abort", description: "双重故障 (#DF)", dpl: 0, handler: "double_fault" },
  { vector: 13, type: "Trap", description: "通用保护 (#GP)", dpl: 0, handler: "general_protection" },
  { vector: 14, type: "Trap", description: "页故障 (#PF)", dpl: 0, handler: "page_fault" },
  { vector: 32, type: "Interrupt", description: "定时器中断 (IRQ0)", dpl: 0, handler: "timer_interrupt" },
  { vector: 33, type: "Interrupt", description: "键盘中断 (IRQ1)", dpl: 0, handler: "keyboard_interrupt" },
  { vector: 128, type: "Trap", description: "系统调用 (INT 0x80)", dpl: 3, handler: "system_call" }
];

export default function IDTStructureExplorer() {
  const [selectedVector, setSelectedVector] = useState<number>(14);
  const selectedEntry = idtEntries.find(e => e.vector === selectedVector)!;

  // 模拟 IDT 描述符结构 (x86-64)
  const descriptor = {
    offsetLow: "0x1234",
    selector: "0x0010",
    ist: "0",
    type: selectedEntry.type === "Interrupt" ? "0xE" : "0xF",
    dpl: selectedEntry.dpl,
    p: 1,
    offsetMid: "0x5678",
    offsetHigh: "0x9ABCDEF0"
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Database className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          中断描述符表 (IDT) 结构浏览器
        </h3>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* IDT 向量列表 */}
        <div className="lg:col-span-1 space-y-2 max-h-[600px] overflow-y-auto">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 sticky top-0 bg-indigo-100 dark:bg-indigo-950 p-2 rounded">
            IDT 向量表
          </h4>
          {idtEntries.map((entry) => (
            <motion.button
              key={entry.vector}
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelectedVector(entry.vector)}
              className={`
                w-full text-left p-3 rounded-lg transition-all
                ${selectedVector === entry.vector
                  ? "bg-indigo-600 text-white shadow-lg"
                  : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                }
              `}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold">Vector {entry.vector}</span>
                <span className={`text-xs px-2 py-1 rounded ${
                  selectedVector === entry.vector 
                    ? "bg-white/20" 
                    : entry.type === "Interrupt"
                    ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                    : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                }`}>
                  {entry.type}
                </span>
              </div>
              <p className={`text-xs ${
                selectedVector === entry.vector 
                  ? "text-indigo-100" 
                  : "text-slate-600 dark:text-slate-400"
              }`}>
                {entry.description}
              </p>
            </motion.button>
          ))}
        </div>

        {/* 描述符详情 */}
        <div className="lg:col-span-2 space-y-6">
          {/* 描述符结构可视化 */}
          <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5 text-indigo-600" />
              IDT 描述符结构 (16 字节)
            </h4>
            <div className="space-y-2 font-mono text-xs">
              <div className="grid grid-cols-2 gap-2">
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
                  <div className="text-blue-600 dark:text-blue-400 mb-1">Offset Low (0-15)</div>
                  <div className="font-bold text-blue-700 dark:text-blue-300">{descriptor.offsetLow}</div>
                </div>
                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-800">
                  <div className="text-green-600 dark:text-green-400 mb-1">Segment Selector (16-31)</div>
                  <div className="font-bold text-green-700 dark:text-green-300">{descriptor.selector}</div>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-2">
                <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded border border-purple-200 dark:border-purple-800">
                  <div className="text-purple-600 dark:text-purple-400 mb-1">IST</div>
                  <div className="font-bold text-purple-700 dark:text-purple-300">{descriptor.ist}</div>
                </div>
                <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded border border-orange-200 dark:border-orange-800">
                  <div className="text-orange-600 dark:text-orange-400 mb-1">Type</div>
                  <div className="font-bold text-orange-700 dark:text-orange-300">{descriptor.type}</div>
                </div>
                <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded border border-pink-200 dark:border-pink-800">
                  <div className="text-pink-600 dark:text-pink-400 mb-1">DPL</div>
                  <div className="font-bold text-pink-700 dark:text-pink-300">{descriptor.dpl}</div>
                </div>
                <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded border border-cyan-200 dark:border-cyan-800">
                  <div className="text-cyan-600 dark:text-cyan-400 mb-1">P</div>
                  <div className="font-bold text-cyan-700 dark:text-cyan-300">{descriptor.p}</div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded border border-indigo-200 dark:border-indigo-800">
                  <div className="text-indigo-600 dark:text-indigo-400 mb-1">Offset Mid (48-63)</div>
                  <div className="font-bold text-indigo-700 dark:text-indigo-300">{descriptor.offsetMid}</div>
                </div>
                <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded border border-teal-200 dark:border-teal-800">
                  <div className="text-teal-600 dark:text-teal-400 mb-1">Offset High (64-127)</div>
                  <div className="font-bold text-teal-700 dark:text-teal-300">{descriptor.offsetHigh}</div>
                </div>
              </div>
            </div>
          </div>

          {/* 字段说明 */}
          <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-indigo-600" />
              字段说明
            </h4>
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded">
                <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1">Offset (64-bit)</div>
                <div className="text-slate-600 dark:text-slate-400">
                  中断处理程序地址: <span className="font-mono text-indigo-600">{descriptor.offsetHigh}{descriptor.offsetMid}{descriptor.offsetLow}</span>
                </div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded">
                <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1">Segment Selector</div>
                <div className="text-slate-600 dark:text-slate-400">
                  代码段选择子 (通常为内核代码段 0x10)
                </div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded">
                <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1">IST (Interrupt Stack Table)</div>
                <div className="text-slate-600 dark:text-slate-400">
                  0 = 使用当前栈，1-7 = 切换到 IST 栈 (用于关键异常)
                </div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded">
                <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1">Type</div>
                <div className="text-slate-600 dark:text-slate-400">
                  0xE = 中断门 (禁用中断)，0xF = 陷阱门 (不禁用中断)
                </div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded">
                <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1">DPL (Descriptor Privilege Level)</div>
                <div className="text-slate-600 dark:text-slate-400">
                  {descriptor.dpl === 0 ? "仅内核可调用 (Ring 0)" : "用户态可调用 (Ring 3)"}
                </div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded">
                <div className="font-semibold text-slate-700 dark:text-slate-300 mb-1">P (Present)</div>
                <div className="text-slate-600 dark:text-slate-400">
                  描述符是否有效 (1 = 有效)
                </div>
              </div>
            </div>
          </div>

          {/* 处理程序信息 */}
          <div className="p-6 bg-gradient-to-r from-indigo-500 to-indigo-600 rounded-lg shadow-lg text-white">
            <h4 className="font-bold mb-4 flex items-center gap-2">
              <Zap className="w-6 h-6" />
              当前向量: {selectedEntry.vector} - {selectedEntry.description}
            </h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="text-sm opacity-90 mb-1">处理程序</div>
                <div className="font-mono font-bold">{selectedEntry.handler}()</div>
              </div>
              <div>
                <div className="text-sm opacity-90 mb-1">类型</div>
                <div className="font-bold">{selectedEntry.type}</div>
              </div>
              <div>
                <div className="text-sm opacity-90 mb-1">特权级 (DPL)</div>
                <div className="font-bold">Ring {selectedEntry.dpl}</div>
              </div>
              <div>
                <div className="text-sm opacity-90 mb-1">访问权限</div>
                <div className="font-bold">
                  {selectedEntry.dpl === 0 ? "内核专用" : "用户可调用"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">IDT 初始化代码</h4>
        <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded font-mono text-xs text-slate-800 dark:text-slate-200 overflow-x-auto">
          <pre>{`// arch/x86/kernel/idt.c
void idt_setup_traps(void)
{
    idt_setup_from_table(idt_table, def_idts, ARRAY_SIZE(def_idts), true);
}

static const struct idt_data def_idts[] = {
    INTG(X86_TRAP_DE,       divide_error),
    INTG(X86_TRAP_DB,       debug),
    INTG(X86_TRAP_NMI,      nmi),
    SYSG(X86_TRAP_BP,       int3),      // DPL=3
    INTG(X86_TRAP_PF,       page_fault),
};`}</pre>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
        <p className="text-sm text-indigo-900 dark:text-indigo-100">
          <strong>IDT 结构：</strong> x86-64 中断描述符表包含 256 个 16 字节描述符，每个描述符指向一个中断/异常处理程序。
          前 32 个向量保留给 CPU 异常，32-255 用于硬件中断和软件中断。
          DPL 字段控制访问权限：0 表示仅内核可触发，3 表示用户态可通过 INT 指令调用 (如 INT 0x80 系统调用)。
        </p>
      </div>
    </div>
  );
}
