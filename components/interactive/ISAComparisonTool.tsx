"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type ISAType = "x86" | "arm" | "riscv";

const isaData = {
  x86: {
    name: "x86 (Intel/AMD)",
    bits: "32/64",
    style: "CISC",
    regs: "16 (64-bit) / 8 (32-bit)",
    instLen: "变长 (1~15字节)",
    addrModes: "立即数、寄存器、直接、间接、基址+变址+偏移",
    features: ["变长指令编码复杂", "大量寻址方式", "向后兼容8086", "内存操作数可直接参与运算"],
    examples: ["MOV EAX, [EBX+8]", "ADD [mem], 1", "PUSH EBP"],
  },
  arm: {
    name: "ARM",
    bits: "32/64",
    style: "RISC",
    regs: "31个通用寄存器 (AArch64)",
    instLen: "固定32位 (ARM模式) / 16位 (Thumb)",
    addrModes: "寄存器、立即数、基址+偏移、PC相对",
    features: ["Load/Store架构", "条件执行", "Thumb指令集节省空间", "低功耗设计"],
    examples: ["LDR X0, [X1, #8]", "ADD X0, X1, X2", "B.LT label"],
  },
  riscv: {
    name: "RISC-V",
    bits: "32/64/128",
    style: "RISC",
    regs: "32个通用寄存器 (x0硬连线0)",
    instLen: "固定32位 (基础) / 可扩展",
    addrModes: "寄存器、立即数、PC相对",
    features: ["完全开源ISA", "模块化扩展(M/F/D等)", "简洁的编码格式", "适合教学和研究"],
    examples: ["lw x1, 8(x2)", "add x1, x2, x3", "beq x1, x2, label"],
  },
};

const categories = ["风格", "通用寄存器", "指令长度", "寻址方式"] as const;
const catKeys = ["style", "regs", "instLen", "addrModes"] as const;

export function ISAComparisonTool() {
  const [selected, setSelected] = useState<ISAType[]>(["x86", "riscv"]);

  function toggle(isa: ISAType) {
    setSelected((prev) =>
      prev.includes(isa) ? prev.filter((s) => s !== isa) : [...prev, isa]
    );
  }

  const active: ISAType[] = selected.length > 0 ? selected : ["x86"];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        ISA 对比工具
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        选择指令集架构进行对比（可多选）
      </p>

      {/* toggle buttons */}
      <div className="flex flex-wrap gap-2 mb-6">
        {(["x86", "arm", "riscv"] as ISAType[]).map((isa) => {
          const on = selected.includes(isa);
          return (
            <button
              key={isa}
              onClick={() => toggle(isa)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                on
                  ? "bg-accent-primary text-white border-accent-primary"
                  : "bg-bg-secondary text-text-secondary border-border-subtle hover:border-accent-primary"
              }`}
            >
              {isaData[isa].name}
            </button>
          );
        })}
      </div>

      {/* comparison table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left py-2 px-3 text-text-secondary font-medium w-28">特性</th>
              {active.map((isa) => (
                <th key={isa} className="text-left py-2 px-3 text-text-primary font-semibold">
                  {isaData[isa].name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {categories.map((cat, i) => (
              <motion.tr
                key={cat}
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="border-b border-border-subtle/50"
              >
                <td className="py-2 px-3 text-text-secondary font-medium">{cat}</td>
                {active.map((isa) => (
                  <td key={isa} className="py-2 px-3 text-text-primary font-mono text-xs">
                    {isaData[isa][catKeys[i]]}
                  </td>
                ))}
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* feature cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
        {active.map((isa) => {
          const d = isaData[isa];
          return (
            <motion.div
              key={isa}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-lg border border-border-subtle bg-bg-secondary p-4"
            >
              <h4 className="font-semibold text-sm text-text-primary mb-2">{d.name} 特点</h4>
              <ul className="space-y-1">
                {d.features.map((f, i) => (
                  <li key={i} className="text-xs text-text-secondary flex items-start gap-1.5">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-accent-primary shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
              <div className="mt-3 border-t border-border-subtle pt-2">
                <p className="text-xs text-text-secondary mb-1">指令示例：</p>
                {d.examples.map((ex, i) => (
                  <code key={i} className="block text-xs font-mono text-accent-primary">{ex}</code>
                ))}
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
