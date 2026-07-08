"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, Plus, ArrowRight } from "lucide-react";

export function CreateTaskDemo() {
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [mainProgress, setMainProgress] = useState(0);
  const [taskProgress, setTaskProgress] = useState(0);
  const [taskScheduled, setTaskScheduled] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = () => {
    setRunning(true);
    setElapsed(0);
    setMainProgress(0);
    setTaskProgress(0);
    setTaskScheduled(false);
    setLog(["main() 开始执行"]);

    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      const t = (Date.now() - startTime) / 1000;
      setElapsed(t);

      // Task scheduled at t=1
      if (t >= 1 && !taskScheduled) {
        setTaskScheduled(true);
        setLog((p) => [...p, `t=${t.toFixed(1)}s: create_task(worker()) 创建任务`]);
      }

      // Main runs from 0-4s with a break at 1-2 (await)
      if (t < 1) {
        setMainProgress((t / 4) * 100);
      } else if (t < 2) {
        setMainProgress((1 / 4) * 100);
      } else {
        setMainProgress(Math.min(((1 + (t - 2)) / 4) * 100, 100));
      }

      // Task runs from 1-3s (concurrent with main after main resumes)
      if (t >= 1 && t < 3) {
        setTaskProgress(((t - 1) / 2) * 100);
      } else if (t >= 3) {
        setTaskProgress(100);
      }

      // Log events
      if (Math.abs(t - 1.5) < 0.06 && log.length < 3) {
        setLog((p) => [...p, `t=1.5s: worker() 开始运行（main 遇到 await 暂停）`]);
      }
      if (Math.abs(t - 2) < 0.06 && log.length < 4) {
        setLog((p) => [...p, `t=2.0s: main() 恢复执行`]);
      }
      if (Math.abs(t - 3) < 0.06 && log.length < 5) {
        setLog((p) => [...p, `t=3.0s: worker() 完成`]);
      }
      if (Math.abs(t - 4) < 0.06 && log.length < 6) {
        setLog((p) => [...p, `t=4.0s: main() 完成`]);
      }

      if (t >= 4.5) {
        setRunning(false);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setElapsed(0);
    setMainProgress(0);
    setTaskProgress(0);
    setTaskScheduled(false);
    setLog([]);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Plus className="w-5 h-5 text-blue-500" />
        create_task() 调度演示
      </h3>

      <div className="flex gap-3 mb-6">
        <button onClick={start} disabled={running}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> {running ? "运行中..." : "开始演示"}
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg">
          <RotateCcw className="w-4 h-4" />
        </button>
        <span className="ml-auto text-sm text-slate-500 self-center">时间: {elapsed.toFixed(1)}s</span>
      </div>

      {/* Code */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto">
{`async def worker():
    await asyncio.sleep(2)     # 模拟耗时操作
    return "done"

async def main():
    task = asyncio.create_task(worker())  # 创建任务
    print("任务已创建，main 继续执行")      # main 继续
    await asyncio.sleep(1)                 # main 暂停
    result = await task                    # 等待任务完成`}
        </pre>
      </div>

      {/* Timeline */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-4">执行时间线</h4>
        <div className="space-y-4">
          {/* Main thread */}
          <div className="flex items-center gap-3">
            <span className="w-16 text-xs font-medium text-slate-600 dark:text-slate-400">main()</span>
            <div className="flex-1 h-8 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden relative">
              <motion.div className="h-full rounded-full bg-blue-500 flex items-center justify-end pr-2"
                style={{ width: `${mainProgress}%` }}>
                {mainProgress > 20 && <span className="text-[10px] text-white">{Math.round(mainProgress)}%</span>}
              </motion.div>
              {elapsed >= 1 && elapsed < 2 && (
                <motion.div className="absolute left-[25%] top-0 h-full w-[25%] bg-amber-400/40 flex items-center justify-center"
                  animate={{ opacity: [0.3, 0.7, 0.3] }} transition={{ repeat: Infinity, duration: 1 }}>
                  <span className="text-[9px] text-amber-800 dark:text-amber-200">await</span>
                </motion.div>
              )}
            </div>
          </div>
          {/* Task */}
          <div className="flex items-center gap-3">
            <span className="w-16 text-xs font-medium text-slate-600 dark:text-slate-400">worker()</span>
            <div className="flex-1 h-8 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
              <motion.div className="h-full rounded-full bg-green-500 flex items-center justify-end pr-2"
                style={{ width: `${taskProgress}%` }}>
                {taskProgress > 20 && <span className="text-[10px] text-white">{Math.round(taskProgress)}%</span>}
              </motion.div>
            </div>
          </div>
        </div>

        {/* Time markers */}
        <div className="flex justify-between text-[10px] text-slate-500 mt-2 px-[68px]">
          <span>0s</span>
          <span>1s</span>
          <span>2s</span>
          <span>3s</span>
          <span>4s</span>
        </div>
      </div>

      {/* Event Log */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
        <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-3">事件日志</h4>
        <div className="space-y-1">
          {log.map((entry, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-2 text-sm">
              <ArrowRight className="w-3 h-3 text-indigo-500" />
              <span className="text-slate-700 dark:text-slate-300">{entry}</span>
            </motion.div>
          ))}
          {log.length === 0 && <p className="text-sm text-slate-500">点击"开始演示"查看执行过程</p>}
        </div>
      </div>
    </div>
  );
}
