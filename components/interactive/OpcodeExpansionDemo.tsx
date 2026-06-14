"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, ChevronRight } from "lucide-react";

const levels = [
  { name: "三地址", format: "OP A B C", opcodeBits: 4, addrBits: 12, instrCount: 16, usage: "高级语言中间代码" },
  { name: "二地址", format: "OP A B", opcodeBits: 8, addrBits: 8, instrCount: 256, usage: "通用计算机指令" },
  { name: "一地址", format: "OP A", opcodeBits: 12, addrBits: 4, instrCount: 4096, usage: "累加器型指令" },
  { name: "零地址", format: "OP", opcodeBits: 16, addrBits: 0, instrCount: 65536, usage: "堆栈型指令" },
];

export function OpcodeExpansionDemo() {
  const [active, setActive] = useState(0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Layers className="w-5 h-5 text-emerald-500" />
        扩展操作码演示
      </h3>
      <p className="text-xs text-text-muted mb-4">16位指令字，从三地址到零地址的操作码扩展过程</p>

      <div className="flex items-center gap-1 mb-6 overflow-x-auto">
        {levels.map((lv, i) => (
          <div key={lv.name} className="flex items-center">
            <button onClick={() => setActive(i)}
              className={`px-4 py-2 rounded text-sm whitespace-nowrap transition-colors ${
                active === i ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle hover:border-blue-400"
              }`}>
              {lv.name}
            </button>
            {i < levels.length - 1 && <ChevronRight className="w-4 h-4 text-text-muted mx-1" />}
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={active} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
          <div className="flex h-10 rounded overflow-hidden mb-4 border border-border-subtle">
            <div className="flex items-center justify-center text-xs font-mono text-white bg-blue-500"
              style={{ width: `${(levels[active].opcodeBits / 16) * 100}%` }}>
              操作码 {levels[active].opcodeBits}b
            </div>
            {levels[active].addrBits > 0 && (
              <div className="flex items-center justify-center text-xs font-mono text-white bg-emerald-500"
                style={{ width: `${(levels[active].addrBits / 16) * 100}%` }}>
                地址码 {levels[active].addrBits}b
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded bg-bg-surface border border-border-subtle">
              <div className="text-xs text-text-muted mb-1">指令格式</div>
              <div className="font-mono text-sm">{levels[active].format}</div>
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle">
              <div className="text-xs text-text-muted mb-1">可编码指令数</div>
              <div className="font-mono text-sm text-blue-400">2^{levels[active].opcodeBits} = {levels[active].instrCount.toLocaleString()}</div>
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle">
              <div className="text-xs text-text-muted mb-1">地址码位数</div>
              <div className="font-mono text-sm">{levels[active].addrBits} 位/地址</div>
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle">
              <div className="text-xs text-text-muted mb-1">典型用途</div>
              <div className="text-sm">{levels[active].usage}</div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
