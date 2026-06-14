"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, SkipForward, RotateCcw, ChevronRight } from "lucide-react";

// ─── TM Definition ───
// Recognizes binary strings with an EVEN number of 0s
// States: q_even (start/accept), q_odd
// Transitions: (state, symbol) → (nextState, write, direction R)
type State = "q_even" | "q_odd" | "q_acc" | "q_rej";
type Symbol = "0" | "1" | "_";
type Direction = "R" | "L" | "S";

interface TMConfig {
  tape: Symbol[];
  head: number;
  state: State;
  step: number;
}

const TRANSITIONS: Record<string, [State, Symbol, Direction]> = {
  "q_even,0": ["q_odd", "0", "R"],
  "q_even,1": ["q_even", "1", "R"],
  "q_even,_": ["q_acc", "_", "S"],
  "q_odd,0": ["q_even", "0", "R"],
  "q_odd,1": ["q_odd", "1", "R"],
  "q_odd,_": ["q_rej", "_", "S"],
};

const EXAMPLES = [
  { label: "0011  (2 个 0，偶数)", tape: ["0", "0", "1", "1"] },
  { label: "001   (2 个 0，偶数)", tape: ["0", "0", "1"] },
  { label: "010   (2 个 0，偶数)", tape: ["0", "1", "0"] },
  { label: "0111  (1 个 0，奇数)", tape: ["0", "1", "1", "1"] },
  { label: "10010 (2 个 0，偶数)", tape: ["1", "0", "0", "1", "0"] },
];

function makeTape(input: Symbol[]): Symbol[] {
  return [...input, "_", "_", "_"];
}

const STATE_STYLE: Record<State, string> = {
  q_even: "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300 border-violet-400",
  q_odd:  "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300 border-orange-400",
  q_acc:  "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300 border-emerald-400",
  q_rej:  "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300 border-red-400",
};

const STATE_LABEL: Record<State, string> = {
  q_even: "q_even（偶）",
  q_odd:  "q_odd（奇）",
  q_acc:  "q_acc ✓ 接受",
  q_rej:  "q_rej ✗ 拒绝",
};

export function TuringMachineSimulator() {
  const [exampleIdx, setExampleIdx] = useState(0);
  const [config, setConfig] = useState<TMConfig>({
    tape: makeTape(EXAMPLES[0].tape as Symbol[]),
    head: 0,
    state: "q_even",
    step: 0,
  });

  const reset = useCallback((idx: number) => {
    setConfig({
      tape: makeTape(EXAMPLES[idx].tape as Symbol[]),
      head: 0,
      state: "q_even",
      step: 0,
    });
    setExampleIdx(idx);
  }, []);

  const step = useCallback(() => {
    setConfig(prev => {
      if (prev.state === "q_acc" || prev.state === "q_rej") return prev;
      const sym = prev.tape[prev.head] ?? "_";
      const key = `${prev.state},${sym}`;
      const trans = TRANSITIONS[key];
      if (!trans) return prev;
      const [nextState, writeSymbol, dir] = trans;
      const newTape = [...prev.tape];
      newTape[prev.head] = writeSymbol;
      // extend tape if needed
      if (prev.head + 1 >= newTape.length) newTape.push("_");
      const newHead = dir === "R" ? prev.head + 1 : dir === "L" ? Math.max(0, prev.head - 1) : prev.head;
      return { tape: newTape, head: newHead, state: nextState, step: prev.step + 1 };
    });
  }, []);

  const runToEnd = useCallback(() => {
    setConfig(prev => {
      let cfg = { ...prev, tape: [...prev.tape] };
      let maxSteps = 200;
      while (cfg.state !== "q_acc" && cfg.state !== "q_rej" && maxSteps-- > 0) {
        const sym = cfg.tape[cfg.head] ?? "_";
        const key = `${cfg.state},${sym}`;
        const trans = TRANSITIONS[key];
        if (!trans) break;
        const [nextState, writeSymbol, dir] = trans;
        cfg.tape[cfg.head] = writeSymbol;
        if (cfg.head + 1 >= cfg.tape.length) cfg.tape.push("_");
        cfg.head = dir === "R" ? cfg.head + 1 : dir === "L" ? Math.max(0, cfg.head - 1) : cfg.head;
        cfg.state = nextState;
        cfg.step++;
      }
      return cfg;
    });
  }, []);

  const isHalted = config.state === "q_acc" || config.state === "q_rej";

  // show 9-cell window centered on head
  const WIN = 4;
  const start = Math.max(0, config.head - WIN);
  const end = Math.min(config.tape.length - 1, config.head + WIN + 1);
  const visibleCells = config.tape.slice(start, end + 1);

  return (
    <div className="w-full max-w-3xl mx-auto my-6 rounded-2xl overflow-hidden border border-violet-200 dark:border-violet-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 to-purple-600 px-6 py-4 text-white">
        <h3 className="text-lg font-bold">图灵机模拟器</h3>
        <p className="text-sm text-violet-100 mt-0.5">识别"0 的个数为偶数"的语言 · 确定型图灵机（DTM）</p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-6 space-y-5">
        {/* Example picker */}
        <div>
          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">选择输入字符串</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => reset(i)}
                className={`px-3 py-1.5 text-xs rounded-lg border font-mono transition-all ${
                  i === exampleIdx
                    ? "bg-violet-600 text-white border-violet-600 shadow"
                    : "border-slate-200 dark:border-slate-700 hover:border-violet-400 dark:hover:border-violet-600 text-slate-700 dark:text-slate-300"
                }`}
              >
                {ex.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tape */}
        <div>
          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">纸带（当前窗口）</p>
          <div className="flex items-end justify-center gap-1.5">
            {start > 0 && <span className="text-slate-400 text-sm pb-1">…</span>}
            {visibleCells.map((sym, i) => {
              const absoluteIdx = start + i;
              const isCurrent = absoluteIdx === config.head;
              return (
                <div key={absoluteIdx} className="flex flex-col items-center gap-1">
                  {isCurrent && (
                    <motion.div
                      layoutId="head"
                      className="text-violet-600 dark:text-violet-400"
                    >
                      <div className="text-xs font-bold text-center">▼</div>
                    </motion.div>
                  )}
                  <motion.div
                    layout
                    animate={isCurrent ? { scale: 1.15 } : { scale: 1 }}
                    className={`w-10 h-10 flex items-center justify-center border-2 rounded-lg font-mono text-sm font-bold transition-colors ${
                      isCurrent
                        ? "border-violet-500 bg-violet-50 dark:bg-violet-900/50 text-violet-700 dark:text-violet-200 shadow-md"
                        : "border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                    }`}
                  >
                    {sym}
                  </motion.div>
                  <span className="text-xs text-slate-400">{absoluteIdx}</span>
                </div>
              );
            })}
            {end < config.tape.length - 1 && <span className="text-slate-400 text-sm pb-1">…</span>}
          </div>
        </div>

        {/* State display */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">当前状态</p>
            <AnimatePresence mode="wait">
              <motion.div
                key={config.state}
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 4 }}
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border-2 font-mono font-semibold text-sm ${STATE_STYLE[config.state]}`}
              >
                {STATE_LABEL[config.state]}
              </motion.div>
            </AnimatePresence>
          </div>
          <div className="grid grid-cols-2 gap-3 text-center">
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
              <p className="text-2xl font-bold text-violet-600 dark:text-violet-400">{config.step}</p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">步数</p>
            </div>
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
              <p className="text-2xl font-bold text-violet-600 dark:text-violet-400">{config.head}</p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">磁头位置</p>
            </div>
          </div>
        </div>

        {/* Result banner */}
        <AnimatePresence>
          {isHalted && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className={`rounded-xl px-5 py-3 text-sm font-semibold ${
                config.state === "q_acc"
                  ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-700"
                  : "bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-700"
              }`}
            >
              {config.state === "q_acc"
                ? "✓ 接受！输入中 0 的个数为偶数。"
                : "✗ 拒绝！输入中 0 的个数为奇数。"}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Controls */}
        <div className="flex gap-3 pt-1">
          <button
            onClick={step}
            disabled={isHalted}
            className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-700 disabled:opacity-40 text-white text-sm rounded-xl font-medium transition-all"
          >
            <Play className="w-4 h-4" /> 单步执行
          </button>
          <button
            onClick={runToEnd}
            disabled={isHalted}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-40 text-white text-sm rounded-xl font-medium transition-all"
          >
            <SkipForward className="w-4 h-4" /> 运行到底
          </button>
          <button
            onClick={() => reset(exampleIdx)}
            className="flex items-center gap-2 px-4 py-2 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 text-sm rounded-xl font-medium transition-all"
          >
            <RotateCcw className="w-4 h-4" /> 重置
          </button>
        </div>

        {/* Transition table */}
        <details className="group">
          <summary className="flex items-center gap-2 cursor-pointer text-xs font-semibold text-slate-500 hover:text-violet-600 dark:text-slate-400 dark:hover:text-violet-400 select-none">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-open:rotate-90" />
            查看转移函数 δ
          </summary>
          <div className="mt-3 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
            <table className="text-xs w-full font-mono">
              <thead className="bg-slate-50 dark:bg-slate-800">
                <tr>
                  <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-300">状态</th>
                  <th className="px-4 py-2 text-center text-slate-600 dark:text-slate-300">读 "0"</th>
                  <th className="px-4 py-2 text-center text-slate-600 dark:text-slate-300">读 "1"</th>
                  <th className="px-4 py-2 text-center text-slate-600 dark:text-slate-300">读 "_"</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {(["q_even", "q_odd"] as State[]).map(s => (
                  <tr key={s} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                    <td className={`px-4 py-2 font-semibold ${s === config.state && !isHalted ? "text-violet-600 dark:text-violet-400" : "text-slate-700 dark:text-slate-300"}`}>{s}</td>
                    {(["0", "1", "_"] as Symbol[]).map(sym => {
                      const t = TRANSITIONS[`${s},${sym}`];
                      const isActive = config.state === s && (config.tape[config.head] ?? "_") === sym && !isHalted;
                      return (
                        <td key={sym} className={`px-4 py-2 text-center ${isActive ? "bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300 font-bold" : "text-slate-600 dark:text-slate-400"}`}>
                          {t ? `→ ${t[0]}, write ${t[1]}, ${t[2]}` : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </div>
  );
}
