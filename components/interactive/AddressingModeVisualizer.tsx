"use client";

import { useState } from "react";

/* ------------------------------------------------------------------ */
/*  Data model for each addressing mode                               */
/* ------------------------------------------------------------------ */

interface AddressingMode {
  id: string;
  name: string;         // Chinese name
  subtitle: string;     // English name
  description: string;
  eaFormula: string;    // Effective address formula (plain text)
  steps: string[];      // Step-by-step EA calculation
  resultValue: string;  // Final operand value
}

const MEMORY_START = 0x100;

// Simulated memory: address 0x100 .. 0x10F
const memoryValues: number[] = [
  0x3e, 0x7a, 0x05, 0x20, 0x8b, 0x42, 0x19, 0xd0,
  0xff, 0x03, 0xc7, 0x64, 0x5e, 0x91, 0x2b, 0xa8,
];

// Registers
const registers: Record<string, number> = {
  R0: 0x000,
  R1: 0x100,
  R2: 0x108,
  R3: 0x005,
};

// Instruction fields
const instruction = {
  opcode: "LOAD",
  addressField: 0x102,   // address / displacement / immediate
};

const modes: AddressingMode[] = [
  {
    id: "immediate",
    name: "立即寻址",
    subtitle: "Immediate",
    description: "操作数直接包含在指令中，无需访问内存。",
    eaFormula: "操作数 = 地址字段 (A)",
    steps: [
      "指令中的地址字段 A 就是操作数本身",
      "A = 0x102",
      "无需计算有效地址，直接使用 A 作为操作数",
    ],
    resultValue: "0x102",
  },
  {
    id: "direct",
    name: "直接寻址",
    subtitle: "Direct",
    description: "指令的地址字段就是操作数在内存中的有效地址。",
    eaFormula: "EA = A",
    steps: [
      "指令中的地址字段 A = 0x102",
      "EA = A = 0x102",
      "访问 Memory[0x102]，读取操作数",
    ],
    resultValue: "0x05",
  },
  {
    id: "indirect",
    name: "间接寻址",
    subtitle: "Indirect",
    description: "指令的地址字段指向的内存单元中存放的是操作数的有效地址。",
    eaFormula: "EA = (A)",
    steps: [
      "指令中的地址字段 A = 0x102",
      "访问 Memory[0x102]，读取地址 = 0x05",
      "EA = 0x05 (从内存中间接获得)",
      "访问 Memory[0x05 - 0x100 = 超出范围，示例简化为: Memory[0x105])",
    ],
    resultValue: "0x42",
  },
  {
    id: "register",
    name: "寄存器寻址",
    subtitle: "Register",
    description: "操作数存放在寄存器中，指令指定寄存器编号。",
    eaFormula: "操作数 = R[i]",
    steps: [
      "指令中的地址字段指定寄存器 R1",
      "R1 = 0x100",
      "操作数直接从 R1 中获取",
    ],
    resultValue: "0x100",
  },
  {
    id: "register-indirect",
    name: "寄存器间接寻址",
    subtitle: "Register Indirect",
    description: "寄存器中存放的是操作数在内存中的有效地址。",
    eaFormula: "EA = (R[i])",
    steps: [
      "指令指定寄存器 R1",
      "R1 = 0x100",
      "EA = R1 = 0x100",
      "访问 Memory[0x100]，读取操作数",
    ],
    resultValue: "0x3e",
  },
  {
    id: "base",
    name: "基址寻址",
    subtitle: "Base",
    description: "有效地址 = 基址寄存器的内容 + 指令中的地址字段（偏移量）。",
    eaFormula: "EA = (BR) + A",
    steps: [
      "基址寄存器 BR = R1 = 0x100",
      "地址字段（偏移量）A = 0x102",
      "EA = 0x100 + 0x102 = 0x202 (实际取低 8 位偏移)",
      "简化示例: EA = R1 + 2 = 0x100 + 0x02 = 0x102",
      "访问 Memory[0x102]，读取操作数",
    ],
    resultValue: "0x05",
  },
  {
    id: "indexed",
    name: "变址寻址",
    subtitle: "Indexed",
    description: "有效地址 = 变址寄存器的内容 + 指令中的地址字段。",
    eaFormula: "EA = (IX) + A",
    steps: [
      "变址寄存器 IX = R3 = 0x005",
      "地址字段 A = 0x102 (基地址)",
      "EA = 0x102 + 0x005 = 0x107",
      "访问 Memory[0x107]，读取操作数",
    ],
    resultValue: "0xd0",
  },
  {
    id: "relative",
    name: "相对寻址",
    subtitle: "Relative",
    description: "有效地址 = 程序计数器 PC 的内容 + 指令中的地址字段（偏移量）。",
    eaFormula: "EA = (PC) + A",
    steps: [
      "当前 PC = 0x200 (下一条指令地址)",
      "地址字段（偏移量）A = 0x006",
      "EA = 0x200 + 0x006 = 0x206",
      "简化为相对偏移示例: EA = 0x100 + 0x06 = 0x106",
      "访问 Memory[0x106]，读取操作数",
    ],
    resultValue: "0x19",
  },
];

/* ------------------------------------------------------------------ */
/*  Visual diagram for each mode (SVG-based)                          */
/* ------------------------------------------------------------------ */

function ModeDiagram({ mode }: { mode: AddressingMode }) {
  const boxH = 40;
  const boxW = 120;

  // Common style classes
  const boxActive = "fill-teal-100 dark:fill-teal-900 stroke-teal-500 stroke-2";
  const boxResult = "fill-emerald-100 dark:fill-emerald-900 stroke-emerald-500 stroke-2";
  const txt = "text-[11px] font-mono fill-slate-700 dark:fill-slate-200";
  const txtLabel = "text-[10px] fill-slate-500 dark:fill-slate-400";
  const arrow = "stroke-teal-500 stroke-[1.5] fill-none marker-end";

  // SVG viewBox width
  const vw = 500;
  const vh = 200;

  // Instruction box (always at top-left)
  const instrX = 30;
  const instrY = 10;

  switch (mode.id) {
    case "immediate":
      return (
        <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-auto">
          {/* Instruction */}
          <rect x={instrX} y={instrY} width={boxW + 40} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW / 2 + 20} y={instrY + 16} textAnchor="middle" className={txt}>
            {instruction.opcode} | A
          </text>
          <text x={instrX + boxW / 2 + 20} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            A = 0x102 (立即数)
          </text>

          {/* Arrow straight to result */}
          <line x1={instrX + boxW / 2 + 20} y1={instrY + boxH} x2={instrX + boxW / 2 + 20} y2={instrY + boxH + 30} className={arrow} markerEnd="url(#arrowTeal)" />

          {/* Result box */}
          <rect x={instrX} y={instrY + boxH + 35} width={boxW + 40} height={boxH} rx={6} className={boxResult} />
          <text x={instrX + boxW / 2 + 20} y={instrY + boxH + 51} textAnchor="middle" className={txt}>
            操作数 = 0x102
          </text>
          <text x={instrX + boxW / 2 + 20} y={instrY + boxH + 67} textAnchor="middle" className={txtLabel}>
            (即 A 本身)
          </text>

          <defs>
            <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" className="fill-teal-500" />
            </marker>
          </defs>
        </svg>
      );

    case "direct":
      return (
        <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-auto">
          <defs>
            <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" className="fill-teal-500" />
            </marker>
          </defs>
          {/* Instruction */}
          <rect x={instrX} y={instrY} width={boxW + 40} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW / 2 + 20} y={instrY + 16} textAnchor="middle" className={txt}>
            {instruction.opcode} | A
          </text>
          <text x={instrX + boxW / 2 + 20} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            A = 0x102
          </text>

          {/* Arrow to memory */}
          <line x1={instrX + boxW + 40} y1={instrY + boxH / 2} x2={instrX + boxW + 120} y2={instrY + boxH / 2} className={arrow} markerEnd="url(#arrowTeal)" />
          <text x={instrX + boxW + 80} y={instrY + boxH / 2 - 8} textAnchor="middle" className={txtLabel}>EA = A</text>

          {/* Memory cell */}
          <rect x={instrX + boxW + 125} y={instrY} width={boxW + 20} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW + 185} y={instrY + 16} textAnchor="middle" className={txt}>
            M[0x102]
          </text>
          <text x={instrX + boxW + 185} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x05
          </text>

          {/* Arrow down to result */}
          <line x1={instrX + boxW + 185} y1={instrY + boxH} x2={instrX + boxW + 185} y2={instrY + boxH + 30} className={arrow} markerEnd="url(#arrowTeal)" />

          {/* Result */}
          <rect x={instrX + boxW + 125} y={instrY + boxH + 35} width={boxW + 20} height={boxH} rx={6} className={boxResult} />
          <text x={instrX + boxW + 185} y={instrY + boxH + 55} textAnchor="middle" className={txt}>
            操作数 = 0x05
          </text>
        </svg>
      );

    case "indirect":
      return (
        <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-auto">
          <defs>
            <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" className="fill-teal-500" />
            </marker>
          </defs>
          {/* Instruction */}
          <rect x={instrX} y={instrY} width={boxW + 20} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW / 2 + 10} y={instrY + 16} textAnchor="middle" className={txt}>
            {instruction.opcode} | A
          </text>
          <text x={instrX + boxW / 2 + 10} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            A = 0x102
          </text>

          {/* Arrow to first memory access */}
          <line x1={instrX + boxW + 20} y1={instrY + boxH / 2} x2={instrX + boxW + 80} y2={instrY + boxH / 2} className={arrow} markerEnd="url(#arrowTeal)" />
          <text x={instrX + boxW + 50} y={instrY + boxH / 2 - 8} textAnchor="middle" className={txtLabel}>①</text>

          {/* Memory cell 1: holds address */}
          <rect x={instrX + boxW + 85} y={instrY} width={boxW + 20} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW + 145} y={instrY + 16} textAnchor="middle" className={txt}>
            M[0x102]
          </text>
          <text x={instrX + boxW + 145} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x05 (地址)
          </text>

          {/* Arrow to second memory access */}
          <line x1={instrX + boxW + 145} y1={instrY + boxH} x2={instrX + boxW + 145} y2={instrY + boxH + 30} className={arrow} markerEnd="url(#arrowTeal)" />
          <text x={instrX + boxW + 155} y={instrY + boxH + 18} textAnchor="start" className={txtLabel}>② EA = 0x05</text>

          {/* Memory cell 2: holds operand */}
          <rect x={instrX + boxW + 85} y={instrY + boxH + 35} width={boxW + 20} height={boxH} rx={6} className={boxResult} />
          <text x={instrX + boxW + 145} y={instrY + boxH + 55} textAnchor="middle" className={txt}>
            M[0x05+0x100] = 0x42
          </text>
        </svg>
      );

    case "register":
      return (
        <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-auto">
          <defs>
            <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" className="fill-teal-500" />
            </marker>
          </defs>
          {/* Instruction */}
          <rect x={instrX} y={instrY} width={boxW + 40} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW / 2 + 20} y={instrY + 16} textAnchor="middle" className={txt}>
            {instruction.opcode} | R1
          </text>
          <text x={instrX + boxW / 2 + 20} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            寄存器编号 = R1
          </text>

          {/* Arrow to register file */}
          <line x1={instrX + boxW + 40} y1={instrY + boxH / 2} x2={instrX + boxW + 120} y2={instrY + boxH / 2} className={arrow} markerEnd="url(#arrowTeal)" />

          {/* Register */}
          <rect x={instrX + boxW + 125} y={instrY} width={boxW} height={boxH} rx={6} className={boxResult} />
          <text x={instrX + boxW + 185} y={instrY + 16} textAnchor="middle" className={txt}>
            R1
          </text>
          <text x={instrX + boxW + 185} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x100
          </text>
        </svg>
      );

    case "register-indirect":
      return (
        <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-auto">
          <defs>
            <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" className="fill-teal-500" />
            </marker>
          </defs>
          {/* Instruction */}
          <rect x={instrX} y={instrY} width={boxW + 20} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW / 2 + 10} y={instrY + 16} textAnchor="middle" className={txt}>
            {instruction.opcode} | (R1)
          </text>
          <text x={instrX + boxW / 2 + 10} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            寄存器编号 = R1
          </text>

          {/* Arrow to register */}
          <line x1={instrX + boxW + 20} y1={instrY + boxH / 2} x2={instrX + boxW + 80} y2={instrY + boxH / 2} className={arrow} markerEnd="url(#arrowTeal)" />
          <text x={instrX + boxW + 50} y={instrY + boxH / 2 - 8} textAnchor="middle" className={txtLabel}>①</text>

          {/* Register */}
          <rect x={instrX + boxW + 85} y={instrY} width={boxW - 20} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW + 135} y={instrY + 16} textAnchor="middle" className={txt}>
            R1 = 0x100
          </text>
          <text x={instrX + boxW + 135} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            (EA)
          </text>

          {/* Arrow to memory */}
          <line x1={instrX + boxW + 135} y1={instrY + boxH} x2={instrX + boxW + 135} y2={instrY + boxH + 30} className={arrow} markerEnd="url(#arrowTeal)" />
          <text x={instrX + boxW + 145} y={instrY + boxH + 18} textAnchor="start" className={txtLabel}>② EA = 0x100</text>

          {/* Memory */}
          <rect x={instrX + boxW + 85} y={instrY + boxH + 35} width={boxW - 20} height={boxH} rx={6} className={boxResult} />
          <text x={instrX + boxW + 135} y={instrY + boxH + 55} textAnchor="middle" className={txt}>
            M[0x100] = 0x3e
          </text>
        </svg>
      );

    case "base":
      return (
        <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-auto">
          <defs>
            <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" className="fill-teal-500" />
            </marker>
          </defs>
          {/* Base register */}
          <rect x={instrX} y={instrY} width={boxW} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW / 2} y={instrY + 16} textAnchor="middle" className={txt}>
            BR (R1)
          </text>
          <text x={instrX + boxW / 2} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x100
          </text>

          {/* Plus sign */}
          <text x={instrX + boxW + 15} y={instrY + 25} textAnchor="middle" className="text-lg font-bold fill-teal-500">+</text>

          {/* Address field A */}
          <rect x={instrX + boxW + 30} y={instrY} width={boxW - 20} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW + 80} y={instrY + 16} textAnchor="middle" className={txt}>
            A (偏移)
          </text>
          <text x={instrX + boxW + 80} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x02
          </text>

          {/* Arrow down to EA */}
          <line x1={instrX + boxW / 2 + 50} y1={instrY + boxH} x2={instrX + boxW / 2 + 50} y2={instrY + boxH + 30} className={arrow} markerEnd="url(#arrowTeal)" />
          <text x={instrX + boxW / 2 + 60} y={instrY + boxH + 18} textAnchor="start" className={txtLabel}>EA = 0x102</text>

          {/* Memory */}
          <rect x={instrX + boxW / 2 - 10} y={instrY + boxH + 35} width={boxW + 20} height={boxH} rx={6} className={boxResult} />
          <text x={instrX + boxW / 2 + 50} y={instrY + boxH + 55} textAnchor="middle" className={txt}>
            M[0x102] = 0x05
          </text>
        </svg>
      );

    case "indexed":
      return (
        <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-auto">
          <defs>
            <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" className="fill-teal-500" />
            </marker>
          </defs>
          {/* Address field A (base) */}
          <rect x={instrX} y={instrY} width={boxW} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW / 2} y={instrY + 16} textAnchor="middle" className={txt}>
            A (基地址)
          </text>
          <text x={instrX + boxW / 2} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x102
          </text>

          {/* Plus sign */}
          <text x={instrX + boxW + 15} y={instrY + 25} textAnchor="middle" className="text-lg font-bold fill-teal-500">+</text>

          {/* Index register */}
          <rect x={instrX + boxW + 30} y={instrY} width={boxW - 20} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW + 80} y={instrY + 16} textAnchor="middle" className={txt}>
            IX (R3)
          </text>
          <text x={instrX + boxW + 80} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x005
          </text>

          {/* Arrow down to EA */}
          <line x1={instrX + boxW / 2 + 50} y1={instrY + boxH} x2={instrX + boxW / 2 + 50} y2={instrY + boxH + 30} className={arrow} markerEnd="url(#arrowTeal)" />
          <text x={instrX + boxW / 2 + 60} y={instrY + boxH + 18} textAnchor="start" className={txtLabel}>EA = 0x107</text>

          {/* Memory */}
          <rect x={instrX + boxW / 2 - 10} y={instrY + boxH + 35} width={boxW + 20} height={boxH} rx={6} className={boxResult} />
          <text x={instrX + boxW / 2 + 50} y={instrY + boxH + 55} textAnchor="middle" className={txt}>
            M[0x107] = 0xd0
          </text>
        </svg>
      );

    case "relative":
      return (
        <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-auto">
          <defs>
            <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" className="fill-teal-500" />
            </marker>
          </defs>
          {/* PC */}
          <rect x={instrX} y={instrY} width={boxW} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW / 2} y={instrY + 16} textAnchor="middle" className={txt}>
            PC
          </text>
          <text x={instrX + boxW / 2} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x100 (简化)
          </text>

          {/* Plus sign */}
          <text x={instrX + boxW + 15} y={instrY + 25} textAnchor="middle" className="text-lg font-bold fill-teal-500">+</text>

          {/* Offset */}
          <rect x={instrX + boxW + 30} y={instrY} width={boxW - 20} height={boxH} rx={6} className={boxActive} />
          <text x={instrX + boxW + 80} y={instrY + 16} textAnchor="middle" className={txt}>
            A (偏移)
          </text>
          <text x={instrX + boxW + 80} y={instrY + 32} textAnchor="middle" className={txtLabel}>
            = 0x006
          </text>

          {/* Arrow down to EA */}
          <line x1={instrX + boxW / 2 + 50} y1={instrY + boxH} x2={instrX + boxW / 2 + 50} y2={instrY + boxH + 30} className={arrow} markerEnd="url(#arrowTeal)" />
          <text x={instrX + boxW / 2 + 60} y={instrY + boxH + 18} textAnchor="start" className={txtLabel}>EA = 0x106</text>

          {/* Memory */}
          <rect x={instrX + boxW / 2 - 10} y={instrY + boxH + 35} width={boxW + 20} height={boxH} rx={6} className={boxResult} />
          <text x={instrX + boxW / 2 + 50} y={instrY + boxH + 55} textAnchor="middle" className={txt}>
            M[0x106] = 0x19
          </text>
        </svg>
      );

    default:
      return null;
  }
}

/* ------------------------------------------------------------------ */
/*  Memory grid component                                             */
/* ------------------------------------------------------------------ */

function MemoryGrid({ highlight }: { highlight?: number }) {
  return (
    <div className="grid grid-cols-8 gap-1">
      {memoryValues.map((val, i) => {
        const addr = MEMORY_START + i;
        const isHighlight = highlight !== undefined && addr === highlight;
        return (
          <div
            key={addr}
            className={`
              flex flex-col items-center p-1.5 rounded text-[10px] font-mono border transition-all duration-300
              ${isHighlight
                ? "bg-teal-200 dark:bg-teal-800 border-teal-500 scale-110 shadow-md"
                : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-600"
              }
            `}
          >
            <span className="text-[9px] text-slate-400 dark:text-slate-500">
              0x{addr.toString(16)}
            </span>
            <span className={`font-semibold ${isHighlight ? "text-teal-800 dark:text-teal-200" : "text-slate-700 dark:text-slate-200"}`}>
              0x{val.toString(16).padStart(2, "0")}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Register state component                                          */
/* ------------------------------------------------------------------ */

function RegisterState({ highlight }: { highlight?: string }) {
  return (
    <div className="flex gap-3 flex-wrap">
      {Object.entries(registers).map(([name, val]) => {
        const isHighlight = highlight === name;
        return (
          <div
            key={name}
            className={`
              flex items-center gap-2 px-3 py-1.5 rounded border font-mono text-sm transition-all duration-300
              ${isHighlight
                ? "bg-teal-100 dark:bg-teal-900 border-teal-400 shadow-sm"
                : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-600"
              }
            `}
          >
            <span className={`font-bold ${isHighlight ? "text-teal-700 dark:text-teal-300" : "text-slate-600 dark:text-slate-400"}`}>
              {name}
            </span>
            <span className={`${isHighlight ? "text-teal-900 dark:text-teal-100" : "text-slate-800 dark:text-slate-200"}`}>
              0x{val.toString(16).padStart(3, "0")}
            </span>
          </div>
        );
      })}
      {/* PC */}
      <div
        className={`
          flex items-center gap-2 px-3 py-1.5 rounded border font-mono text-sm transition-all duration-300
          ${highlight === "PC"
            ? "bg-teal-100 dark:bg-teal-900 border-teal-400 shadow-sm"
            : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-600"
          }
        `}
      >
        <span className={`font-bold ${highlight === "PC" ? "text-teal-700 dark:text-teal-300" : "text-slate-600 dark:text-slate-400"}`}>
          PC
        </span>
        <span className={`${highlight === "PC" ? "text-teal-900 dark:text-teal-100" : "text-slate-800 dark:text-slate-200"}`}>
          0x200
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                    */
/* ------------------------------------------------------------------ */

export function AddressingModeVisualizer() {
  const [selectedId, setSelectedId] = useState<string>(modes[0].id);
  const [stepIdx, setStepIdx] = useState<number>(0);
  const [animating, setAnimating] = useState<boolean>(false);

  const selected = modes.find((m) => m.id === selectedId) ?? modes[0];

  // Determine which memory address to highlight based on mode
  const getMemoryHighlight = (): number | undefined => {
    switch (selected.id) {
      case "direct":
      case "base":
        return 0x102;
      case "indirect":
        return 0x105;
      case "register-indirect":
        return 0x100;
      case "indexed":
        return 0x107;
      case "relative":
        return 0x106;
      default:
        return undefined;
    }
  };

  // Determine which register to highlight
  const getRegisterHighlight = (): string | undefined => {
    switch (selected.id) {
      case "register":
      case "register-indirect":
      case "base":
        return "R1";
      case "indexed":
        return "R3";
      case "relative":
        return "PC";
      default:
        return undefined;
    }
  };

  const handleSelectMode = (id: string) => {
    setSelectedId(id);
    setStepIdx(0);
    setAnimating(true);
    // Auto-play steps
    const mode = modes.find((m) => m.id === id);
    if (!mode) return;

    let idx = 0;
    const timer = setInterval(() => {
      idx++;
      if (idx >= mode.steps.length) {
        clearInterval(timer);
        setAnimating(false);
      } else {
        setStepIdx(idx);
      }
    }, 700);
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-teal-50 to-emerald-50 dark:from-slate-900 dark:to-teal-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Title */}
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">
        寻址方式可视化
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">
        Addressing Modes Visualizer -- 点击标签切换寻址方式，观察数据流和有效地址计算过程
      </p>

      {/* Mode selector tabs */}
      <div className="flex flex-wrap gap-1.5 mb-6">
        {modes.map((m) => (
          <button
            key={m.id}
            onClick={() => handleSelectMode(m.id)}
            className={`
              px-3 py-1.5 rounded-lg text-sm font-medium border transition-all duration-200
              ${selectedId === m.id
                ? "bg-teal-600 text-white border-teal-600 shadow-md shadow-teal-200 dark:shadow-teal-900"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-600 hover:border-teal-400 hover:text-teal-700 dark:hover:text-teal-300"
              }
            `}
          >
            {m.name}
          </button>
        ))}
      </div>

      {/* Selected mode info */}
      <div className="mb-4">
        <div className="flex items-baseline gap-3 mb-1">
          <span className="text-lg font-bold text-teal-700 dark:text-teal-300">
            {selected.name}
          </span>
          <span className="text-sm font-mono text-slate-500 dark:text-slate-400">
            ({selected.subtitle})
          </span>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          {selected.description}
        </p>
      </div>

      {/* Two-column layout: diagram + EA info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Left: Visual diagram */}
        <div className="bg-white dark:bg-slate-800/60 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
          <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
            数据流图
          </h4>
          <ModeDiagram mode={selected} />
        </div>

        {/* Right: EA formula and steps */}
        <div className="bg-white dark:bg-slate-800/60 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
          <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
            有效地址计算
          </h4>

          {/* Formula */}
          <div className="bg-slate-50 dark:bg-slate-900/60 rounded-lg p-3 mb-4 border border-slate-100 dark:border-slate-700">
            <span className="font-mono text-sm text-teal-700 dark:text-teal-300 font-semibold">
              {selected.eaFormula}
            </span>
          </div>

          {/* Steps */}
          <div className="space-y-2">
            {selected.steps.map((step, i) => (
              <div
                key={i}
                className={`
                  flex items-start gap-2 text-sm transition-all duration-300
                  ${i <= stepIdx
                    ? "opacity-100 translate-x-0"
                    : "opacity-0 translate-x-4"
                  }
                `}
              >
                <span
                  className={`
                    flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold mt-0.5
                    ${i < stepIdx
                      ? "bg-teal-500 text-white"
                      : i === stepIdx
                        ? "bg-teal-500 text-white animate-pulse"
                        : "bg-slate-200 dark:bg-slate-700 text-slate-400"
                    }
                  `}
                >
                  {i + 1}
                </span>
                <span className="font-mono text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
                  {step}
                </span>
              </div>
            ))}
          </div>

          {/* Result */}
          <div
            className={`
              mt-4 p-3 rounded-lg border transition-all duration-500
              ${stepIdx >= selected.steps.length - 1
                ? "bg-emerald-50 dark:bg-emerald-900/30 border-emerald-300 dark:border-emerald-700 opacity-100"
                : "bg-slate-50 dark:bg-slate-900/30 border-slate-200 dark:border-slate-700 opacity-40"
              }
            `}
          >
            <span className="text-xs text-slate-500 dark:text-slate-400">操作数</span>
            <span className="ml-2 font-mono text-base font-bold text-emerald-700 dark:text-emerald-300">
              {selected.resultValue}
            </span>
          </div>
        </div>
      </div>

      {/* Bottom: Memory and Register state */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Memory */}
        <div className="bg-white dark:bg-slate-800/60 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
          <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
            内存状态 (0x100 -- 0x10F)
          </h4>
          <MemoryGrid highlight={getMemoryHighlight()} />
        </div>

        {/* Registers */}
        <div className="bg-white dark:bg-slate-800/60 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
          <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
            寄存器状态
          </h4>
          <RegisterState highlight={getRegisterHighlight()} />
        </div>
      </div>

      {/* Instruction display */}
      <div className="mt-4 flex items-center justify-center gap-4 p-3 bg-white/70 dark:bg-slate-800/40 rounded-lg border border-slate-200 dark:border-slate-700">
        <span className="text-xs text-slate-500 dark:text-slate-400">当前指令:</span>
        <span className="font-mono text-sm font-semibold text-slate-800 dark:text-slate-200">
          {instruction.opcode} | {selected.subtitle === "Immediate" ? `#${instruction.addressField.toString(16)}` :
            selected.subtitle === "Register" ? "R1" :
            selected.subtitle === "Register Indirect" ? "(R1)" :
            `0x${instruction.addressField.toString(16)}`}
        </span>
        <span className="text-xs text-slate-400 dark:text-slate-500">
          -- {selected.name}
        </span>
      </div>
    </div>
  );
}
