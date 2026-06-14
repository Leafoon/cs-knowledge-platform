"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ToggleRight } from "lucide-react";

const conditions = [
  { code: "0000", name: "EQ", desc: "相等", flag: "Z=1" },
  { code: "0001", name: "NE", desc: "不等", flag: "Z=0" },
  { code: "0010", name: "CS/HS", desc: "无符号>= ", flag: "C=1" },
  { code: "0011", name: "CC/LO", desc: "无符号<", flag: "C=0" },
  { code: "0100", name: "MI", desc: "负数", flag: "N=1" },
  { code: "0101", name: "PL", desc: "非负", flag: "N=0" },
  { code: "0110", name: "VS", desc: "溢出", flag: "V=1" },
  { code: "0111", name: "VC", desc: "无溢出", flag: "V=0" },
  { code: "1000", name: "HI", desc: "无符号>", flag: "C=1, Z=0" },
  { code: "1001", name: "LS", desc: "无符号<= ", flag: "C=0 or Z=1" },
  { code: "1010", name: "GE", desc: "有符号>= ", flag: "N=V" },
  { code: "1011", name: "LT", desc: "有符号<", flag: "N≠V" },
  { code: "1100", name: "GT", desc: "有符号>", flag: "Z=0, N=V" },
  { code: "1101", name: "LE", desc: "有符号<= ", flag: "Z=1 or N≠V" },
  { code: "1110", name: "AL", desc: "无条件", flag: "总是" },
  { code: "1111", name: "NV", desc: "从不执行", flag: "保留" },
];

export function ARMConditionField() {
  const [selected, setSelected] = useState(14);
  const [nFlag, setNFlag] = useState(0);
  const [zFlag, setZFlag] = useState(0);
  const [cFlag, setCFlag] = useState(0);
  const [vFlag, setVFlag] = useState(0);

  const cond = conditions[selected];

  const checkCondition = (code: string): boolean => {
    switch (code) {
      case "0000": return zFlag === 1;
      case "0001": return zFlag === 0;
      case "0010": return cFlag === 1;
      case "0011": return cFlag === 0;
      case "0100": return nFlag === 1;
      case "0101": return nFlag === 0;
      case "0110": return vFlag === 1;
      case "0111": return vFlag === 0;
      case "1000": return cFlag === 1 && zFlag === 0;
      case "1001": return cFlag === 0 || zFlag === 1;
      case "1010": return nFlag === vFlag;
      case "1011": return nFlag !== vFlag;
      case "1100": return zFlag === 0 && nFlag === vFlag;
      case "1101": return zFlag === 1 || nFlag !== vFlag;
      case "1110": return true;
      default: return false;
    }
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <ToggleRight className="w-5 h-5 text-teal-500" />
        ARM条件域
      </h3>

      <div className="flex gap-3 mb-4">
        {[
          { label: "N", value: nFlag, set: setNFlag, desc: "负数标志" },
          { label: "Z", value: zFlag, set: setZFlag, desc: "零标志" },
          { label: "C", value: cFlag, set: setCFlag, desc: "进位标志" },
          { label: "V", value: vFlag, set: setVFlag, desc: "溢出标志" },
        ].map(f => (
          <button key={f.label} onClick={() => f.set(f.value === 1 ? 0 : 1)}
            className={`px-3 py-2 rounded border text-sm font-mono ${f.value === 1 ? "bg-blue-500 text-white border-blue-500" : "bg-bg-surface border-border-subtle"}`}>
            {f.label}={f.value}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-4 gap-1 mb-4">
        {conditions.map((c, i) => {
          const active = checkCondition(c.code);
          return (
            <motion.button key={i} onClick={() => setSelected(i)}
              className={`p-2 rounded border text-xs text-center transition-colors ${
                selected === i ? "border-blue-500 bg-blue-500/10" :
                active ? "border-green-500/30 bg-green-500/5" :
                "border-border-subtle bg-bg-surface opacity-50"
              }`}
              initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: active ? 1 : 0.5, scale: 1 }} transition={{ delay: i * 0.02 }}>
              <div className="font-mono font-bold">{c.code}</div>
              <div className="font-medium">{c.name}</div>
            </motion.button>
          );
        })}
      </div>

      <motion.div key={selected} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="p-4 rounded bg-bg-surface border border-border-subtle">
        <div className="flex items-center gap-3 mb-2">
          <span className="font-mono text-lg font-bold">{cond.code}</span>
          <span className="text-lg font-semibold">{cond.name}</span>
          <span className={`px-2 py-0.5 rounded text-xs ${checkCondition(cond.code) ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}>
            {checkCondition(cond.code) ? "满足条件" : "不满足"}
          </span>
        </div>
        <div className="text-sm text-text-secondary">条件: {cond.desc}</div>
        <div className="text-xs text-text-muted mt-1">标志位要求: {cond.flag}</div>
        <div className="text-xs text-text-muted mt-1">示例: BEQ{cond.name === "AL" ? "" : cond.name} label → {checkCondition(cond.code) ? "跳转到label" : "不跳转"}</div>
      </motion.div>
    </div>
  );
}
