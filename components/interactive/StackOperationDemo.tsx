"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, ArrowUp, ArrowDown } from "lucide-react";

export function StackOperationDemo() {
  const [stack, setStack] = useState<number[]>([0x11, 0x22, 0x33]);
  const [sp, setSp] = useState(0x1FC);
  const [log, setLog] = useState<string[]>([]);
  const [inputVal, setInputVal] = useState(0x44);

  const push = () => {
    const newSp = sp - 4;
    setStack([inputVal, ...stack]);
    setSp(newSp);
    setLog(prev => [`PUSH 0x${inputVal.toString(16).toUpperCase()}: SP=0x${sp.toString(16)}→0x${newSp.toString(16)}, M[SP]=0x${inputVal.toString(16)}`, ...prev]);
  };

  const pop = () => {
    if (stack.length === 0) return;
    const val = stack[0];
    const newSp = sp + 4;
    setStack(stack.slice(1));
    setSp(newSp);
    setLog(prev => [`POP → 0x${val.toString(16).toUpperCase()}: M[SP]=0x${val.toString(16)}, SP=0x${sp.toString(16)}→0x${newSp.toString(16)}`, ...prev]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Layers className="w-5 h-5 text-cyan-500" />
        堆栈操作演示
      </h3>

      <div className="flex gap-2 mb-4">
        <input type="text" value={`0x${inputVal.toString(16)}`}
          onChange={e => { const v = parseInt(e.target.value.replace("0x", ""), 16); if (!isNaN(v)) setInputVal(v); }}
          className="w-24 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        <button onClick={push} className="px-4 py-1 rounded bg-blue-500 text-white text-sm flex items-center gap-1 hover:bg-blue-600">
          <ArrowDown className="w-4 h-4" /> PUSH
        </button>
        <button onClick={pop} disabled={stack.length === 0}
          className="px-4 py-1 rounded bg-orange-500 text-white text-sm flex items-center gap-1 disabled:opacity-40 hover:bg-orange-600">
          <ArrowUp className="w-4 h-4" /> POP
        </button>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div>
          <div className="text-xs text-text-muted mb-2">堆栈内容（高地址 → 低地址）</div>
          <div className="flex flex-col items-center">
            <div className="text-xs text-text-muted mb-1">栈底 (高地址)</div>
            <div className="w-32 border-x-2 border-t-2 border-border-subtle rounded-t px-2 py-1 text-center text-xs text-text-muted">
              ...
            </div>
            <AnimatePresence>
              {stack.map((val, i) => (
                <motion.div
                  key={`${sp}-${i}`}
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: 50 }}
                  className={`w-32 border-x-2 border-border-subtle px-2 py-2 text-center font-mono text-sm ${
                    i === 0 ? "bg-blue-500/20 border-blue-500" : "bg-bg-surface"
                  }`}
                >
                  0x{val.toString(16).toUpperCase().padStart(2, "0")}
                  {i === 0 && <span className="text-xs text-blue-400 ml-1">← 栈顶</span>}
                </motion.div>
              ))}
            </AnimatePresence>
            <div className="w-32 border-2 border-border-subtle rounded-b px-2 py-1 text-center text-xs text-text-muted">
              栈顶 (低地址)
            </div>
          </div>
        </div>

        <div>
          <div className="p-3 rounded bg-bg-surface border border-border-subtle mb-3">
            <div className="text-xs text-text-muted mb-1">栈指针 SP</div>
            <div className="font-mono text-lg text-blue-400">0x{sp.toString(16).toUpperCase()}</div>
          </div>
          <div className="text-xs text-text-muted mb-1">操作日志</div>
          <div className="max-h-40 overflow-y-auto space-y-1">
            {log.map((l, i) => (
              <div key={i} className="text-xs font-mono text-text-secondary p-1 rounded bg-bg-surface">{l}</div>
            ))}
            {log.length === 0 && <div className="text-xs text-text-muted">暂无操作</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
