"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, XCircle, ArrowRight } from "lucide-react";

const STEPS = [
  { label: "task.cancel()", desc: "请求取消任务", detail: "设置任务的取消标志", color: "bg-blue-500" },
  { label: "CancelledError", desc: "在 await 点抛出异常", detail: "任务在下一个 await 处收到 CancelledError", color: "bg-amber-500" },
  { label: "清理资源", desc: "执行 finally 块", detail: "任务可以在 finally 中清理资源", color: "bg-purple-500" },
  { label: "任务完成", desc: "状态变为 cancelled", detail: "task.cancelled() 返回 True", color: "bg-emerald-500" },
];

export function TaskCancellationFlow() {
  const [step, setStep] = useState(-1);
  const [running, setRunning] = useState(false);
  const [codeLines, setCodeLines] = useState<string[]>([]);

  const code = [
    'task = asyncio.create_task(long_running())',
    'await asyncio.sleep(0.5)',
    'task.cancel()               # ① 请求取消',
    '',
    '# 在 long_running() 中:',
    'async def long_running():',
    '    try:',
    '        while True:',
    '            await asyncio.sleep(1)  # ← CancelledError ②',
    '    except asyncio.CancelledError:',
    '        print("收到取消请求")',
    '    finally:',
    '        cleanup()           # ③ 清理资源',
    '    # ④ 任务状态: cancelled',
  ];

  const start = () => {
    setRunning(true);
    setStep(-1);
    setCodeLines([]);

    STEPS.forEach((_, i) => {
      setTimeout(() => {
        setStep(i);
        if (i === 0) setCodeLines(code.slice(0, 3));
        if (i === 1) setCodeLines(code.slice(0, 9));
        if (i === 2) setCodeLines(code.slice(0, 12));
        if (i === 3) setCodeLines(code);
      }, (i + 1) * 1500);
    });

    setTimeout(() => setRunning(false), STEPS.length * 1500 + 200);
  };

  const reset = () => {
    setRunning(false);
    setStep(-1);
    setCodeLines([]);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-orange-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <XCircle className="w-7 h-7 text-orange-600 dark:text-orange-400" />
        任务取消流程
      </h3>

      <div className="flex justify-center gap-3 mb-5">
        <button onClick={start} disabled={running} className="px-5 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 dark:bg-orange-500">
          <span className="flex items-center gap-1"><Play className="w-4 h-4" /> 演示</span>
        </button>
        <button onClick={reset} className="px-5 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600">
          <span className="flex items-center gap-1"><RotateCcw className="w-4 h-4" /> 重置</span>
        </button>
      </div>

      <div className="flex items-center justify-center gap-2 mb-6">
        {STEPS.map((s, i) => (
          <React.Fragment key={i}>
            <motion.div animate={{ scale: step >= i ? 1.1 : 1, opacity: step >= i ? 1 : 0.3 }}
              className={`px-4 py-2 rounded-lg text-white text-sm font-bold ${step >= i ? s.color : "bg-slate-300 dark:bg-slate-700"}`}>
              <div>{s.label}</div>
              <div className="text-xs font-normal opacity-80">{s.desc}</div>
            </motion.div>
            {i < STEPS.length - 1 && <ArrowRight className={`w-5 h-5 ${step > i ? "text-slate-600 dark:text-slate-300" : "text-slate-300 dark:text-slate-700"}`} />}
          </React.Fragment>
        ))}
      </div>

      <AnimatePresence>
        {step >= 0 && step < STEPS.length && (
          <motion.div key={step} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow mb-4">
            <h4 className={`font-bold text-lg ${step === 1 ? "text-amber-700 dark:text-amber-400" : "text-slate-800 dark:text-slate-100"}`}>
              步骤 {step + 1}: {STEPS[step].label}
            </h4>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">{STEPS[step].detail}</p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">代码执行</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded text-xs overflow-x-auto min-h-[180px]">
          {codeLines.map((line, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.03 }}>
              {line}
            </motion.div>
          ))}
          {codeLines.length === 0 && <span className="text-slate-500">点击演示开始...</span>}
        </pre>
      </div>
    </div>
  );
}
