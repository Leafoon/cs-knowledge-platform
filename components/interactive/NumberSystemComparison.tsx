"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Table, ChevronDown, ChevronUp } from "lucide-react";

const representations = [
  {
    name: "原码",
    en: "Sign-Magnitude",
    range8: "-127 ~ +127",
    zero: "+0: 00000000\n-0: 10000000",
    addition: "不能直接加减，需判断符号",
    hardware: "简单",
    detail: "最高位为符号位，其余位为数值的绝对值。加减法复杂，需要比较符号位。",
  },
  {
    name: "反码",
    en: "Ones' Complement",
    range8: "-127 ~ +127",
    zero: "+0: 00000000\n-0: 11111111",
    addition: "有循环进位（end-around carry）",
    hardware: "中等",
    detail: "正数同原码；负数：符号位不变，数值位取反。存在+0和-0两种零。",
  },
  {
    name: "补码",
    en: "Two's Complement",
    range8: "-128 ~ +127",
    zero: "00000000（唯一）",
    addition: "直接加减，统一加法器",
    hardware: "最常用",
    detail: "正数同原码；负数：反码+1。零的表示唯一，加减法统一，硬件实现最简单。",
  },
  {
    name: "移码",
    en: "Excess/Bias",
    range8: "-128 ~ +127",
    zero: "10000000",
    addition: "浮点数阶码比较方便",
    hardware: "浮点数专用",
    detail: "补码的符号位取反。便于比较大小，常用于浮点数的阶码表示。",
  },
];

export function NumberSystemComparison() {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Table className="w-5 h-5 text-blue-500" />
        数制综合对比表
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left p-2">编码方式</th>
              <th className="text-left p-2">8位表示范围</th>
              <th className="text-left p-2">零的表示</th>
              <th className="text-left p-2">加减运算</th>
              <th className="text-left p-2">硬件复杂度</th>
              <th className="text-left p-2 w-8"></th>
            </tr>
          </thead>
          <tbody>
            {representations.map((rep, i) => (
              <motion.tr
                key={rep.name}
                className="border-b border-border-subtle cursor-pointer hover:bg-bg-hover"
                onClick={() => setExpanded(expanded === i ? null : i)}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                <td className="p-2 font-medium">
                  <div>{rep.name}</div>
                  <div className="text-xs text-text-muted">{rep.en}</div>
                </td>
                <td className="p-2 font-mono text-xs">{rep.range8}</td>
                <td className="p-2 font-mono text-xs whitespace-pre">{rep.zero}</td>
                <td className="p-2 text-xs">{rep.addition}</td>
                <td className="p-2">
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    rep.hardware === "最常用" ? "bg-green-500/20 text-green-400" :
                    rep.hardware === "简单" ? "bg-blue-500/20 text-blue-400" :
                    rep.hardware === "浮点数专用" ? "bg-purple-500/20 text-purple-400" :
                    "bg-yellow-500/20 text-yellow-400"
                  }`}>
                    {rep.hardware}
                  </span>
                </td>
                <td className="p-2">
                  {expanded === i ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      <AnimatePresence>
        {expanded !== null && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-3 p-4 rounded bg-bg-surface border border-border-subtle">
              <h4 className="font-medium mb-2">{representations[expanded].name} ({representations[expanded].en})</h4>
              <p className="text-sm text-text-secondary">{representations[expanded].detail}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
