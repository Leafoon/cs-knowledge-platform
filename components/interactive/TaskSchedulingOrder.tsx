"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, ArrowRight, Shuffle } from "lucide-react";

interface Task {
  name: string;
  createOrder: number;
  duration: number;
  color: string;
  startTime?: number;
  endTime?: number;
}

export function TaskSchedulingOrder() {
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [executionOrder, setExecutionOrder] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const taskDefs: Task[] = [
    { name: "task_A", createOrder: 0, duration: 3, color: "#ef4444" },
    { name: "task_B", createOrder: 1, duration: 1, color: "#3b82f6" },
    { name: "task_C", createOrder: 2, duration: 2, color: "#10b981" },
  ];

  const start = () => {
    setRunning(true);
    setElapsed(0);
    setLog(["创建所有任务..."]);
    setExecutionOrder([]);

    const activeTasks = taskDefs.map((t) => ({ ...t, startTime: undefined as number | undefined, endTime: undefined as number | undefined }));
    const order: string[] = [];
    let time = 0;

    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      const t = (Date.now() - startTime) / 1000;
      setElapsed(t);

      // Simulate: tasks created at t=0, but execute in different order
      // Task B is shortest (1s), Task C medium (2s), Task A longest (3s)
      // Event loop picks ready tasks - order depends on await points
      const updatedTasks = activeTasks.map((task) => {
        const copy = { ...task };
        if (task.name === "task_B" && t >= 0 && t < 1) {
          copy.startTime = 0;
          if (t >= 1) copy.endTime = 1;
        } else if (task.name === "task_B" && t >= 1) {
          copy.startTime = 0;
          copy.endTime = 1;
        }

        if (task.name === "task_C" && t >= 1 && t < 3) {
          copy.startTime = 1;
          if (t >= 3) copy.endTime = 3;
        } else if (task.name === "task_C" && t >= 3) {
          copy.startTime = 1;
          copy.endTime = 3;
        }

        if (task.name === "task_A" && t >= 0 && t < 3) {
          copy.startTime = 0;
          if (t >= 3) copy.endTime = 3;
        } else if (task.name === "task_A" && t >= 3) {
          copy.startTime = 0;
          copy.endTime = 3;
        }

        return copy;
      });

      setTasks(updatedTasks);

      if (Math.abs(t - 0.1) < 0.06 && log.length < 2) {
        setLog((p) => [...p, "所有任务已创建（create_order: A, B, C）"]);
      }
      if (Math.abs(t - 0.5) < 0.06 && log.length < 3) {
        setLog((p) => [...p, "事件循环开始调度：task_B 最先获得执行机会"]);
      }
      if (Math.abs(t - 1.2) < 0.06 && log.length < 4) {
        setLog((p) => [...p, "task_B 完成！执行顺序: B → ..."]);
      }
      if (Math.abs(t - 1.5) < 0.06 && log.length < 5) {
        setLog((p) => [...p, "task_C 获得执行机会"]);
      }
      if (Math.abs(t - 3.2) < 0.06 && log.length < 6) {
        setLog((p) => [...p, "task_A 和 task_C 完成！"]);
      }

      if (t >= 4) {
        setRunning(false);
        setExecutionOrder(["B", "C", "A"]);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setElapsed(0);
    setTasks([]);
    setLog([]);
    setExecutionOrder([]);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const getTaskProgress = (taskName: string) => {
    const task = tasks.find((t) => t.name === taskName);
    if (!task || task.startTime === undefined) return 0;
    const duration = taskDefs.find((t) => t.name === taskName)?.duration || 1;
    if (task.endTime !== undefined && elapsed >= task.endTime) return 100;
    return Math.min(((elapsed - task.startTime) / duration) * 100, 100);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Shuffle className="w-5 h-5 text-amber-500" />
        任务创建顺序 vs 执行顺序
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        任务按 A→B→C 顺序创建，但执行顺序可能不同
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
{`# 创建顺序: A → B → C
task_a = asyncio.create_task(task_A())  # duration: 3s
task_b = asyncio.create_task(task_B())  # duration: 1s
task_c = asyncio.create_task(task_C())  # duration: 2s

# 执行顺序可能: B → C → A（取决于事件循环调度）`}
        </pre>
      </div>

      {/* Execution Timeline */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-4">执行时间线</h4>
        <div className="space-y-3">
          {taskDefs.map((task) => (
            <div key={task.name} className="flex items-center gap-3">
              <span className="w-16 text-xs font-medium text-slate-600 dark:text-slate-400">{task.name}</span>
              <div className="flex-1 h-8 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
                <motion.div className="h-full rounded-full flex items-center justify-end pr-2"
                  style={{ width: `${getTaskProgress(task.name)}%`, backgroundColor: task.color }}>
                  {getTaskProgress(task.name) > 30 && (
                    <span className="text-[10px] text-white">{Math.round(getTaskProgress(task.name))}%</span>
                  )}
                </motion.div>
              </div>
              <span className="w-8 text-xs text-slate-400">{task.duration}s</span>
            </div>
          ))}
        </div>
      </div>

      {/* Order Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-4">
          <h4 className="font-bold text-blue-700 dark:text-blue-300 mb-2">创建顺序</h4>
          <div className="flex items-center gap-2">
            {["A", "B", "C"].map((name, i) => (
              <React.Fragment key={name}>
                <span className="px-3 py-1 bg-blue-100 dark:bg-blue-800 rounded text-sm font-mono text-blue-700 dark:text-blue-300">
                  {name}
                </span>
                {i < 2 && <ArrowRight className="w-4 h-4 text-blue-400" />}
              </React.Fragment>
            ))}
          </div>
        </div>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-xl border border-green-200 dark:border-green-800 p-4">
          <h4 className="font-bold text-green-700 dark:text-green-300 mb-2">执行顺序</h4>
          <div className="flex items-center gap-2">
            {executionOrder.length > 0 ? executionOrder.map((name, i) => (
              <React.Fragment key={name}>
                <motion.span initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                  className="px-3 py-1 bg-green-100 dark:bg-green-800 rounded text-sm font-mono text-green-700 dark:text-green-300">
                  {name}
                </motion.span>
                {i < executionOrder.length - 1 && <ArrowRight className="w-4 h-4 text-green-400" />}
              </React.Fragment>
            )) : <span className="text-sm text-slate-500">等待运行...</span>}
          </div>
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
        </div>
      </div>
    </div>
  );
}
