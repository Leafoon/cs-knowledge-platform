"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Calculator, Cpu } from "lucide-react";

const modes = [
  { id: "immediate", name: "立即寻址", ea: "操作数 = A", steps: ["地址字段A即为操作数本身", "无需访问内存"] },
  { id: "direct", name: "直接寻址", ea: "EA = A", steps: ["取地址字段A", "EA = A", "访问 Memory[A] 得操作数"] },
  { id: "indirect", name: "间接寻址", ea: "EA = (A)", steps: ["取地址字段A", "访问 Memory[A] 得到地址B", "EA = B", "访问 Memory[B] 得操作数"] },
  { id: "register", name: "寄存器寻址", ea: "EA = R", steps: ["取寄存器号R", "读取 R 的值即为操作数"] },
  { id: "regIndirect", name: "寄存器间接", ea: "EA = (R)", steps: ["取寄存器号R", "读取 R 的值 → 地址", "EA = R的值", "访问 Memory[EA] 得操作数"] },
  { id: "displacement", name: "偏移寻址", ea: "EA = (R) + A", steps: ["读取寄存器R的值", "取地址字段A（偏移量）", "EA = R + A", "访问 Memory[EA] 得操作数"] },
];

const regValues: Record<string, number> = { R0: 0, R1: 0x200, R2: 0x150, R3: 0x300 };
const memValues: Record<number, number> = { 0x100: 0x42, 0x150: 0x08, 0x200: 0x300, 0x300: 0xAB, 0x42: 0xFF, 0x08: 0x55 };

export function AddressCalculationUnit() {
  const [mode, setMode] = useState(0);
  const [addrField, setAddrField] = useState(0x100);
  const [reg, setReg] = useState("R1");

  const current = modes[mode];
  const rv = regValues[reg] ?? 0;

  const calcEA = (): number => {
    switch (current.id) {
      case "immediate": return addrField;
      case "direct": return addrField;
      case "indirect": return memValues[addrField] ?? 0;
      case "register": return rv;
      case "regIndirect": return rv;
      case "displacement": return rv + addrField;
      default: return 0;
    }
  };

  const ea = calcEA();
  const operand = current.id === "immediate" ? addrField : (memValues[ea] ?? 0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Calculator className="w-5 h-5 text-teal-500" />
        地址计算单元
      </h3>

      <div className="flex flex-wrap gap-1 mb-4">
        {modes.map((m, i) => (
          <button key={m.id} onClick={() => setMode(i)}
            className={`px-3 py-1 rounded text-xs ${mode === i ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle hover:border-blue-400"}`}>
            {m.name}
          </button>
        ))}
      </div>

      <div className="flex gap-4 mb-4">
        <div>
          <label className="block text-xs text-text-muted mb-1">地址字段 A</label>
          <input type="text" value={`0x${addrField.toString(16)}`}
            onChange={e => { const v = parseInt(e.target.value.replace("0x", ""), 16); if (!isNaN(v)) setAddrField(v); }}
            className="w-24 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">寄存器</label>
          <select value={reg} onChange={e => setReg(e.target.value)}
            className="px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm">
            {Object.entries(regValues).map(([name, val]) => (
              <option key={name} value={name}>{name} = 0x{val.toString(16)}</option>
            ))}
          </select>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={mode} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
          <div className="mb-3 p-3 rounded bg-bg-surface border border-border-subtle">
            <div className="text-xs text-text-muted mb-1">EA 公式</div>
            <div className="font-mono text-sm text-blue-400">{current.ea}</div>
          </div>

          <div className="space-y-1 mb-4">
            {current.steps.map((s, i) => (
              <motion.div key={i} className="flex items-center gap-2 text-sm"
                initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.15 }}>
                <span className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-xs font-mono">{i + 1}</span>
                <span className="text-text-secondary">{s}</span>
              </motion.div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded bg-bg-surface border border-border-subtle text-center">
              <div className="text-xs text-text-muted">有效地址 EA</div>
              <div className="text-lg font-mono font-bold text-green-400">0x{ea.toString(16).toUpperCase()}</div>
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle text-center">
              <div className="text-xs text-text-muted">操作数</div>
              <div className="text-lg font-mono font-bold text-purple-400">0x{operand.toString(16).toUpperCase().padStart(2, "0")}</div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
