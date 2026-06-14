"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, SkipForward, ArrowDown } from "lucide-react";

const regs = ["ra", "sp", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"] as const;

interface Step {
  label: string;
  desc: string;
  saving: number[];
  loading: number[];
  savedRegs: Record<string, string>;
  loadedRegs: Record<string, string>;
}

const procARegs: Record<string, string> = {
  ra: "0x80001234", sp: "0x3fffff800", s0: "42", s1: "100",
  s2: "0x80004000", s3: "7", s4: "0", s5: "256",
  s6: "0x80008000", s7: "3", s8: "99", s9: "0x1000",
  s10: "0x80010000", s11: "15",
};

const schedulerRegs: Record<string, string> = {
  ra: "0x80002a10", sp: "0x80012000", s0: "0", s1: "0",
  s2: "0", s3: "0", s4: "0", s5: "0",
  s6: "0", s7: "0", s8: "0", s9: "0",
  s10: "0", s11: "0",
};

const procBRegs: Record<string, string> = {
  ra: "0x80001a88", sp: "0x3ffffe000", s0: "17", s1: "55",
  s2: "0x80005000", s3: "12", s4: "1", s5: "512",
  s6: "0x80009000", s7: "8", s8: "33", s9: "0x2000",
  s10: "0x80020000", s11: "21",
};

const buildSteps = (): Step[] => {
  const steps: Step[] = [];
  const saved: Record<string, string> = {};
  const loaded: Record<string, string> = {};

  steps.push({
    label: "swtch(&p->context, &c->context)",
    desc: "调用 swtch()：a0 = &A->context（保存），a1 = &c->context（加载）",
    saving: [],
    loading: [],
    savedRegs: {},
    loadedRegs: {},
  });

  regs.forEach((r, i) => {
    saved[r] = procARegs[r];
    steps.push({
      label: `sd ${r}, ${i * 8}(a0)`,
      desc: `保存 ${r} = ${procARegs[r]} → A->context.${r}`,
      saving: [i],
      loading: [],
      savedRegs: { ...saved },
      loadedRegs: {},
    });
  });

  steps.push({
    label: "--- 保存完成，开始加载 ---",
    desc: "A 的 14 个寄存器已保存到 A->context，现在从 c->context 加载调度器的寄存器",
    saving: [],
    loading: [],
    savedRegs: { ...saved },
    loadedRegs: {},
  });

  regs.forEach((r, i) => {
    loaded[r] = schedulerRegs[r];
    steps.push({
      label: `ld ${r}, ${i * 8}(a1)`,
      desc: `加载 ${r} = ${schedulerRegs[r]} ← c->context.${r}`,
      saving: [],
      loading: [i],
      savedRegs: { ...saved },
      loadedRegs: { ...loaded },
    });
  });

  steps.push({
    label: "ret",
    desc: "跳转到 c->context.ra = 0x80002a10（scheduler() 中 swtch 返回后的位置）",
    saving: [],
    loading: [],
    savedRegs: { ...saved },
    loadedRegs: { ...loaded },
  });

  steps.push({
    label: "scheduler() 恢复",
    desc: "调度器继续执行：找到进程 B，swtch(&c->context, &B->context)",
    saving: [],
    loading: [],
    savedRegs: { ...saved },
    loadedRegs: { ...loaded },
  });

  const saved2: Record<string, string> = {};
  regs.forEach((r, i) => {
    saved2[r] = schedulerRegs[r];
    steps.push({
      label: `sd ${r}, ${i * 8}(a0)`,
      desc: `保存 c->context.${r} = ${schedulerRegs[r]}`,
      saving: [i],
      loading: [],
      savedRegs: { ...saved2 },
      loadedRegs: {},
    });
  });

  steps.push({
    label: "--- 开始加载 B 的上下文 ---",
    desc: "调度器寄存器已保存，现在从 B->context 加载进程 B 的寄存器",
    saving: [],
    loading: [],
    savedRegs: { ...saved2 },
    loadedRegs: {},
  });

  const loaded2: Record<string, string> = {};
  regs.forEach((r, i) => {
    loaded2[r] = procBRegs[r];
    steps.push({
      label: `ld ${r}, ${i * 8}(a1)`,
      desc: `加载 ${r} = ${procBRegs[r]} ← B->context.${r}`,
      saving: [],
      loading: [i],
      savedRegs: { ...saved2 },
      loadedRegs: { ...loaded2 },
    });
  });

  steps.push({
    label: "ret → 进程 B 的 sched()",
    desc: "跳转到 B->context.ra = 0x80001a88，进程 B 从 sched() 中的 swtch() 返回",
    saving: [],
    loading: [],
    savedRegs: { ...saved2 },
    loadedRegs: { ...loaded2 },
  });

  return steps;
};

export default function Xv6ContextSwitch() {
  const [stepIdx, setStepIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const steps = buildSteps();

  const advance = useCallback(() => {
    setStepIdx((prev) => (prev < steps.length - 1 ? prev + 1 : prev));
  }, [steps.length]);

  useEffect(() => {
    if (isPlaying && stepIdx < steps.length - 1) {
      const t = setTimeout(advance, 400);
      return () => clearTimeout(t);
    }
    if (stepIdx >= steps.length - 1) setIsPlaying(false);
  }, [isPlaying, stepIdx, advance, steps.length]);

  const reset = () => {
    setIsPlaying(false);
    setStepIdx(0);
  };

  const step = steps[stepIdx];
  const phase = stepIdx <= 14 ? "save-A" : stepIdx <= 15 ? "transition" : stepIdx <= 30 ? "load-sched" : stepIdx <= 31 ? "transition" : stepIdx <= 45 ? "save-sched" : stepIdx <= 46 ? "transition" : stepIdx <= 60 ? "load-B" : "done";

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
          <ArrowDown className="w-5 h-5 text-blue-500" />
          swtch() 上下文切换
        </h3>
        <div className="flex gap-2">
          <button onClick={() => { setIsPlaying(false); advance(); }} className="flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
            <SkipForward className="w-3 h-3" /> 单步
          </button>
          <button onClick={() => setIsPlaying(!isPlaying)} className={`flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg transition-colors ${isPlaying ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300" : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"}`}>
            {isPlaying ? <><Pause className="w-3 h-3" /> 暂停</> : <><Play className="w-3 h-3" /> 播放</>}
          </button>
          <button onClick={reset} className="flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
            <RotateCcw className="w-3 h-3" /> 重置
          </button>
        </div>
      </div>

      <div className="mb-3 p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
        <p className="text-xs font-mono text-blue-600 dark:text-blue-400 font-bold">{step.label}</p>
        <p className="text-xs text-slate-600 dark:text-slate-300 mt-1">{step.desc}</p>
      </div>

      <div className="flex justify-between mb-2">
        <span className="text-xs font-mono text-slate-500 dark:text-slate-400">步骤 {stepIdx + 1} / {steps.length}</span>
        <div className="flex gap-2">
          {(["save-A", "load-sched", "save-sched", "load-B"] as const).map((p) => (
            <span key={p} className={`text-xs px-2 py-0.5 rounded ${phase === p ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-bold" : "text-slate-400 dark:text-slate-500"}`}>
              {p}
            </span>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <th className="py-1 px-2 text-left text-slate-500 dark:text-slate-400">寄存器</th>
              <th className="py-1 px-2 text-center text-slate-500 dark:text-slate-400">旧 context (保存)</th>
              <th className="py-1 px-2 text-center text-slate-500 dark:text-slate-400">新 context (加载)</th>
            </tr>
          </thead>
          <tbody>
            {regs.map((r, i) => {
              const isSaving = step.saving.includes(i);
              const isLoading = step.loading.includes(i);
              return (
                <tr key={r} className={`border-b border-slate-100 dark:border-slate-800 ${isSaving ? "bg-red-50 dark:bg-red-900/20" : isLoading ? "bg-emerald-50 dark:bg-emerald-900/20" : ""}`}>
                  <td className="py-1 px-2 font-bold text-slate-700 dark:text-slate-200">{r}</td>
                  <td className="py-1 px-2 text-center">
                    <AnimatePresence mode="wait">
                      {step.savedRegs[r] ? (
                        <motion.span
                          key={step.savedRegs[r]}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className={isSaving ? "text-red-600 dark:text-red-400 font-bold" : "text-slate-500 dark:text-slate-400"}
                        >
                          {step.savedRegs[r]}
                        </motion.span>
                      ) : (
                        <span className="text-slate-300 dark:text-slate-600">—</span>
                      )}
                    </AnimatePresence>
                  </td>
                  <td className="py-1 px-2 text-center">
                    <AnimatePresence mode="wait">
                      {step.loadedRegs[r] ? (
                        <motion.span
                          key={step.loadedRegs[r]}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className={isLoading ? "text-emerald-600 dark:text-emerald-400 font-bold" : "text-slate-500 dark:text-slate-400"}
                        >
                          {step.loadedRegs[r]}
                        </motion.span>
                      ) : (
                        <span className="text-slate-300 dark:text-slate-600">—</span>
                      )}
                    </AnimatePresence>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-3 w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
        <motion.div
          className="bg-blue-500 h-1.5 rounded-full"
          animate={{ width: `${((stepIdx + 1) / steps.length) * 100}%` }}
          transition={{ duration: 0.2 }}
        />
      </div>
    </div>
  );
}
