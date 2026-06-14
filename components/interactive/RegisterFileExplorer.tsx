"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Layers } from "lucide-react"

interface Register {
  name: string
  alias?: string
  width: number
  purpose: string
  usage: string
  bits?: string
}

type RegisterSet = "RISC" | "CISC"

const riscRegisters: Register[] = [
  { name: "R0", width: 32, purpose: "通用寄存器", usage: "存储操作数、运算中间结果", bits: "00000000000000000000000000000000" },
  { name: "R1", width: 32, purpose: "通用寄存器", usage: "存储操作数、运算中间结果", bits: "00000000000000000000000000000000" },
  { name: "R2", width: 32, purpose: "通用寄存器", usage: "存储操作数、运算中间结果", bits: "00000000000000000000000000000000" },
  { name: "R3", width: 32, purpose: "通用寄存器", usage: "存储操作数、运算中间结果", bits: "00000000000000000000000000000000" },
  { name: "R4", width: 32, purpose: "通用寄存器", usage: "存储操作数、运算中间结果", bits: "00000000000000000000000000000000" },
  { name: "R5", width: 32, purpose: "通用寄存器", usage: "存储操作数、运算中间结果", bits: "00000000000000000000000000000000" },
  { name: "R6", width: 32, purpose: "通用寄存器 (SP)", usage: "栈指针，指向栈顶", bits: "00000000000000000000000000000000" },
  { name: "R7", width: 32, purpose: "通用寄存器 (LR)", usage: "链接寄存器，保存返回地址", bits: "00000000000000000000000000000000" },
  { name: "PC", width: 32, purpose: "程序计数器", usage: "指向下一条要执行的指令地址", bits: "00000000000000000000000000000000" },
  { name: "CPSR", width: 32, purpose: "当前程序状态寄存器", usage: "存储条件标志(N/Z/C/V)、中断使能、处理器模式", bits: "NZCV0000000000000000000000000000" },
]

const ciscRegisters: Register[] = [
  { name: "EAX", alias: "AX", width: 32, purpose: "累加器", usage: "乘除运算隐含操作数、函数返回值", bits: "00000000000000000000000000000000" },
  { name: "EBX", alias: "BX", width: 32, purpose: "基址寄存器", usage: "内存寻址基址、通用数据存储", bits: "00000000000000000000000000000000" },
  { name: "ECX", alias: "CX", width: 32, purpose: "计数寄存器", usage: "循环计数、移位操作计数", bits: "00000000000000000000000000000000" },
  { name: "EDX", alias: "DX", width: 32, purpose: "数据寄存器", usage: "I/O端口地址、乘除高32位", bits: "00000000000000000000000000000000" },
  { name: "ESI", alias: "SI", width: 32, purpose: "源变址寄存器", usage: "字符串操作源地址", bits: "00000000000000000000000000000000" },
  { name: "EDI", alias: "DI", width: 32, purpose: "目的变址寄存器", usage: "字符串操作目的地址", bits: "00000000000000000000000000000000" },
  { name: "ESP", alias: "SP", width: 32, purpose: "栈指针", usage: "指向栈顶元素", bits: "00000000000000000000000000000000" },
  { name: "EBP", alias: "BP", width: 32, purpose: "基址指针", usage: "指向栈帧基址，用于访问局部变量", bits: "00000000000000000000000000000000" },
  { name: "EIP", alias: "IP", width: 32, purpose: "指令指针", usage: "指向下一条要执行的指令", bits: "00000000000000000000000000000000" },
  { name: "EFLAGS", alias: "FLAGS", width: 32, purpose: "标志寄存器", usage: "存储运算结果标志(CF/ZF/SF/OF/IF等)", bits: "00000000000000000000000000000000" },
]

export function RegisterFileExplorer() {
  const [mode, setMode] = useState<RegisterSet>("RISC")
  const [selected, setSelected] = useState<string>("R0")

  const registers = mode === "RISC" ? riscRegisters : ciscRegisters
  const active = registers.find((r) => r.name === selected) ?? registers[0]

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Layers className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">寄存器组探索器</h3>
      </div>

      <div className="flex gap-2 mb-4">
        {(["RISC", "CISC"] as const).map((t) => (
          <button
            key={t}
            onClick={() => { setMode(t); setSelected(t === "RISC" ? "R0" : "EAX") }}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
              mode === t
                ? "bg-accent text-white"
                : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-accent"
            }`}
          >
            {t} 寄存器集
          </button>
        ))}
      </div>

      <div className="flex gap-6">
        <div className="flex flex-wrap gap-2 flex-1">
          {registers.map((reg, i) => (
            <motion.button
              key={reg.name}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.03 }}
              onClick={() => setSelected(reg.name)}
              className={`px-3 py-2 rounded text-sm font-mono border transition-all min-w-[70px] text-center ${
                selected === reg.name
                  ? "bg-accent text-white border-accent shadow-lg shadow-accent/20"
                  : "bg-bg-surface border-border-subtle text-text-primary hover:border-accent"
              }`}
            >
              {reg.name}
            </motion.button>
          ))}
        </div>

        <motion.div
          key={active.name + mode}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-64 flex flex-col gap-3"
        >
          <div className="p-4 rounded-lg border border-border-subtle bg-bg-surface">
            <div className="text-xl font-mono font-bold text-accent mb-1">
              {active.name}
              {active.alias && <span className="text-sm text-text-secondary ml-2">({active.alias})</span>}
            </div>
            <div className="text-sm text-text-secondary mb-3">{active.purpose}</div>
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-text-secondary">位宽: </span>
                <span className="font-mono">{active.width} bit</span>
              </div>
              <div>
                <span className="text-text-secondary">用途: </span>
                <span>{active.usage}</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-lg border border-border-subtle bg-bg-surface">
            <div className="text-xs text-text-secondary mb-2">位表示</div>
            <div className="flex flex-wrap gap-[2px] font-mono text-[10px]">
              {active.bits?.split("").map((bit, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.01 }}
                  className={`w-4 h-4 flex items-center justify-center rounded ${
                    bit === "0" ? "bg-bg-elevated text-text-secondary" : "bg-accent/20 text-accent"
                  }`}
                >
                  {bit}
                </motion.span>
              ))}
            </div>
            <div className="flex justify-between text-[9px] text-text-secondary mt-1 font-mono">
              <span>{active.width - 1}</span>
              <span>0</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
