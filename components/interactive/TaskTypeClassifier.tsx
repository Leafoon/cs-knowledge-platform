"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { HelpCircle, CheckCircle, ArrowRight } from "lucide-react";

interface Question {
  id: number;
  text: string;
  options: { label: string; next: number | null; result: string | null }[];
}

const questions: Question[] = [
  { id: 0, text: "任务主要在做什么？", options: [
    { label: "等待外部资源（网络、磁盘、用户输入）", next: 1, result: null },
    { label: "持续进行数学计算", next: 2, result: null },
  ]},
  { id: 1, text: "使用的库支持异步接口吗？", options: [
    { label: "支持（有 async 版本）", next: null, result: "asyncio" },
    { label: "不支持（只有同步接口）", next: null, result: "threading" },
  ]},
  { id: 2, text: "计算量有多大？", options: [
    { label: "非常大，需要多核并行", next: null, result: "multiprocessing" },
    { label: "一般，单核可以处理", next: null, result: "asyncio (with to_thread)" },
  ]},
];

const results: Record<string, { title: string; color: string; description: string; code: string }> = {
  asyncio: { title: "asyncio", color: "#10b981", description: "异步协程最适合大量 I/O 等待任务", code: "await asyncio.gather(*tasks)" },
  threading: { title: "多线程", color: "#3b82f6", description: "同步阻塞库需要放在线程中执行", code: "await asyncio.to_thread(blocking_func)" },
  multiprocessing: { title: "多进程", color: "#ef4444", description: "CPU 密集型需要多进程利用多核", code: "with ProcessPoolExecutor() as pool: await loop.run_in_executor(pool, cpu_task)" },
};

export function TaskTypeClassifier() {
  const [currentQ, setCurrentQ] = useState(0);
  const [result, setResult] = useState<string | null>(null);
  const [history, setHistory] = useState<number[]>([]);

  const handleChoice = (next: number | null, res: string | null) => {
    setHistory((h) => [...h, currentQ]);
    if (res) {
      setResult(res);
    } else if (next !== null) {
      setCurrentQ(next);
    }
  };

  const reset = () => { setCurrentQ(0); setResult(null); setHistory([]); };

  const q = questions[currentQ];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <HelpCircle className="w-5 h-5" />
        任务类型选择器
      </h3>
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 min-h-[200px]">
        {!result ? (
          <motion.div key={currentQ} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100 mb-4">{q.text}</p>
            <div className="space-y-3">
              {q.options.map((opt, i) => (
                <button key={i} onClick={() => handleChoice(opt.next, opt.result)}
                  className="w-full text-left px-4 py-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:border-indigo-300 dark:hover:border-indigo-600 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-all flex items-center gap-3">
                  <ArrowRight className="w-4 h-4 text-slate-400" />
                  <span className="text-sm text-slate-700 dark:text-slate-300">{opt.label}</span>
                </button>
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="text-center">
            <CheckCircle className="w-12 h-12 mx-auto mb-4" style={{ color: results[result].color }} />
            <h4 className="text-2xl font-bold mb-2" style={{ color: results[result].color }}>{results[result].title}</h4>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">{results[result].description}</p>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 text-left inline-block">
              <code className="text-xs text-indigo-600 dark:text-indigo-400 whitespace-pre">{results[result].code}</code>
            </div>
            <div className="mt-4">
              <button onClick={reset} className="px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg text-sm">重新选择</button>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
