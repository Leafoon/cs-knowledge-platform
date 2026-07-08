"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, ArrowRight, Trophy } from "lucide-react";

interface TaskInfo {
  name: string;
  delay: number;
  result: string;
  color: string;
  progress: number;
  done: boolean;
  finishTime?: number;
}

export function CompletionVsResultOrder() {
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [tasks, setTasks] = useState<TaskInfo[]>([
    { name: "task_A", delay: 3, result: '"A完成"', color: "#ef4444", progress: 0, done: false },
    { name: "task_B", delay: 1, result: '"B完成"', color: "#3b82f6", progress: 0, done: false },
    { name: "task_C", delay: 2, result: '"C完成"', color: "#10b981", progress: 0, done: false },
  ]);
  const [gatherResult, setGatherResult] = useState<string | null>(null);
  const [completionOrder, setCompletionOrder] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = () => {
    setRunning(true);
    setElapsed(0);
    setGatherResult(null);
    setCompletionOrder([]);
    setTasks((prev) => prev.map((t) => ({ ...t, progress: 0, done: false, finishTime: undefined })));

    const startTime = Date.now();
    const finishTimes: Record<string, number> = {};

    timerRef.current = setInterval(() => {
      const t = (Date.now() - startTime) / 1000;
      setElapsed(t);

      setTasks((prev) =>
        prev.map((task) => {
          const progress = Math.min((t / task.delay) * 100, 100);
          const done = progress >= 100;
          if (done && !task.done && !finishTimes[task.name]) {
            finishTimes[task.name] = t;
          }
          return { ...task, progress, done, finishTime: finishTimes[task.name] };
        })
      );

      // Track completion order
      const newlyDone = tasks.filter((task) => {
        const prog = Math.min((t / task.delay) * 100, 100);
        return prog >= 100 && !finishTimes[task.name];
      });
      newlyDone.forEach((task) => {
        if (!finishTimes[task.name]) {
          finishTimes[task.name] = t;
          setCompletionOrder((prev) => [...prev, task.name]);
        }
      });

      if (t >= 3.5 && !gatherResult) {
        setGatherResult('["A完成", "B完成", "C完成"]');
      }

      if (t >= 4) {
        setRunning(false);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setElapsed(0);
    setGatherResult(null);
    setCompletionOrder([]);
    setTasks((prev) => prev.map((t) => ({ ...t, progress: 0, done: false, finishTime: undefined })));
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const totalDelay = Math.max(...tasks.map((t) => t.delay));

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Trophy className="w-5 h-5 text-amber-500" />
        完成顺序 vs 结果顺序
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        gather() 按输入顺序返回结果，而非完成顺序
      </p>

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
        <pre className="bg-slate-900 text-green-400 p-3 rounded-lg text-xs overflow-x-auto">
{`results = await asyncio.gather(task_A, task_B, task_C)
# results[0] = task_A 的结果（总是第一个）
# results[1] = task_B 的结果（总是第二个）
# results[2] = task_C 的结果（总是第三个）
# 无论哪个任务先完成！`}
        </pre>
      </div>

      {/* Progress Bars */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-4">任务进度</h4>
        <div className="space-y-3">
          {tasks.map((task, i) => (
            <div key={task.name} className="flex items-center gap-3">
              <div className="flex items-center gap-2 w-24">
                <span className="text-xs font-medium text-slate-600 dark:text-slate-400">{task.name}</span>
                {task.done && (
                  <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }}
                    className="text-xs bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-1 rounded">
                    #{completionOrder.indexOf(task.name) + 1}
                  </motion.span>
                )}
              </div>
              <div className="flex-1 h-8 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
                <motion.div className="h-full rounded-full flex items-center justify-end pr-2"
                  style={{ width: `${task.progress}%`, backgroundColor: task.color }}>
                  {task.progress > 30 && <span className="text-[10px] text-white">{Math.round(task.progress)}%</span>}
                </motion.div>
              </div>
              <span className="w-8 text-xs text-slate-400">{task.delay}s</span>
            </div>
          ))}
        </div>
      </div>

      {/* Two Order Panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800 p-4">
          <h4 className="font-bold text-amber-700 dark:text-amber-300 mb-2 flex items-center gap-2">
            <Trophy className="w-4 h-4" /> 完成顺序（谁先完成）
          </h4>
          <div className="flex items-center gap-2 flex-wrap">
            {completionOrder.length > 0 ? completionOrder.map((name, i) => (
              <React.Fragment key={name}>
                <motion.span initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                  className="px-3 py-1 rounded text-sm font-mono text-white"
                  style={{ backgroundColor: tasks.find((t) => t.name === name)?.color }}>
                  {name}
                </motion.span>
                {i < completionOrder.length - 1 && <ArrowRight className="w-4 h-4 text-amber-400" />}
              </React.Fragment>
            )) : <span className="text-sm text-slate-500">等待运行...</span>}
          </div>
          <p className="text-xs text-slate-500 mt-2">取决于各任务的延迟时间</p>
        </div>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl border border-indigo-200 dark:border-indigo-800 p-4">
          <h4 className="font-bold text-indigo-700 dark:text-indigo-300 mb-2 flex items-center gap-2">
            <ArrowRight className="w-4 h-4" /> gather() 结果顺序
          </h4>
          <div className="flex items-center gap-2">
            {tasks.map((task, i) => (
              <React.Fragment key={task.name}>
                <span className="px-3 py-1 rounded text-sm font-mono text-white" style={{ backgroundColor: task.color }}>
                  {task.name}
                </span>
                {i < tasks.length - 1 && <ArrowRight className="w-4 h-4 text-indigo-400" />}
              </React.Fragment>
            ))}
          </div>
          <p className="text-xs text-slate-500 mt-2">始终与 create_task 的顺序一致</p>
        </div>
      </div>

      {/* Result Display */}
      {gatherResult && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className="bg-green-50 dark:bg-green-900/20 rounded-xl border border-green-200 dark:border-green-800 p-5">
          <h4 className="font-bold text-green-700 dark:text-green-300 mb-2">gather() 返回值</h4>
          <pre className="bg-slate-900 text-green-400 p-3 rounded-lg text-xs">
            {`# 输入顺序: task_A, task_B, task_C\n# 结果顺序: ${gatherResult}\n# 即使 B 先完成，A 的结果仍在第一个！`}
          </pre>
        </motion.div>
      )}
    </div>
  );
}
