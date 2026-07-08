"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, CheckSquare, Square, AlertTriangle, CheckCircle } from "lucide-react";

export function ConcurrencyRequirements() {
  const [multipleTasks, setMultipleTasks] = useState(true);
  const [yieldControl, setYieldControl] = useState(true);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [taskStates, setTaskStates] = useState([0, 0, 0]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const COLORS = ["#ef4444", "#3b82f6", "#10b981"];
  const TASK_NAMES = ["任务 A", "任务 B", "任务 C"];
  const DELAY = 2;

  const hasConcurrency = multipleTasks && yieldControl;

  const start = () => {
    setRunning(true);
    setElapsed(0);
    setTaskStates([0, 0, 0]);

    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      const t = (Date.now() - startTime) / 1000;
      setElapsed(t);

      if (hasConcurrency) {
        // Concurrent: all tasks run simultaneously
        setTaskStates([0, 1, 2].map(() => {
          if (t <= 0) return 0;
          if (t >= DELAY) return 100;
          return (t / DELAY) * 100;
        }));
        if (t >= DELAY) {
          setRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      } else if (multipleTasks && !yieldControl) {
        // Multiple tasks but blocking sleep - sequential
        setTaskStates(() => {
          const states = [0, 0, 0];
          for (let i = 0; i < 3; i++) {
            const start = i * DELAY;
            const end = start + DELAY;
            if (t <= start) states[i] = 0;
            else if (t >= end) states[i] = 100;
            else states[i] = ((t - start) / DELAY) * 100;
          }
          return states;
        });
        if (t >= DELAY * 3) {
          setRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      } else if (!multipleTasks && yieldControl) {
        // Single task with yield - just one task runs
        setTaskStates(() => {
          if (t <= 0) return [0, 0, 0];
          if (t >= DELAY) return [100, 0, 0];
          return [(t / DELAY) * 100, 0, 0];
        });
        if (t >= DELAY) {
          setRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      } else {
        // Neither - single blocking task
        setTaskStates(() => {
          if (t <= 0) return [0, 0, 0];
          if (t >= DELAY) return [100, 0, 0];
          return [(t / DELAY) * 100, 0, 0];
        });
        if (t >= DELAY) {
          setRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setElapsed(0);
    setTaskStates([0, 0, 0]);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const totalTime = hasConcurrency ? DELAY : multipleTasks && !yieldControl ? DELAY * 3 : DELAY;

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <AlertTriangle className="w-5 h-5 text-amber-500" />
        并发的两个必要条件
      </h3>

      {/* Checkboxes */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-3">选择条件</h4>
        <div className="space-y-3">
          <button onClick={() => { setMultipleTasks(!multipleTasks); reset(); }}
            className="flex items-center gap-3 w-full text-left">
            {multipleTasks ? (
              <CheckSquare className="w-5 h-5 text-indigo-600" />
            ) : (
              <Square className="w-5 h-5 text-slate-400" />
            )}
            <div>
              <span className="text-sm font-medium text-slate-800 dark:text-slate-200">
                多个任务已调度 (multiple tasks scheduled)
              </span>
              <p className="text-xs text-slate-500">使用 create_task() 创建多个任务</p>
            </div>
          </button>
          <button onClick={() => { setYieldControl(!yieldControl); reset(); }}
            className="flex items-center gap-3 w-full text-left">
            {yieldControl ? (
              <CheckSquare className="w-5 h-5 text-indigo-600" />
            ) : (
              <Square className="w-5 h-5 text-slate-400" />
            )}
            <div>
              <span className="text-sm font-medium text-slate-800 dark:text-slate-200">
                任务让出控制权 (tasks yield control)
              </span>
              <p className="text-xs text-slate-500">使用 await asyncio.sleep() 而非 time.sleep()</p>
            </div>
          </button>
        </div>
      </div>

      {/* Status Banner */}
      <div className={`mb-6 p-4 rounded-xl flex items-center gap-3 ${
        hasConcurrency
          ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
          : "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
      }`}>
        {hasConcurrency ? (
          <CheckCircle className="w-5 h-5 text-green-600" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-red-600" />
        )}
        <div>
          <span className={`font-bold ${hasConcurrency ? "text-green-700 dark:text-green-300" : "text-red-700 dark:text-red-300"}`}>
            {hasConcurrency ? "并发生效" : "无法实现并发"}
          </span>
          <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
            {hasConcurrency && "多个任务 + 让出控制权 = 事件循环可以交替执行"}
            {!multipleTasks && !yieldControl && "单个阻塞任务，无法并发"}
            {!multipleTasks && yieldControl && "只有一个任务，没有其他任务可以交替执行"}
            {multipleTasks && !yieldControl && "time.sleep() 阻塞事件循环，其他任务无法运行"}
          </p>
        </div>
      </div>

      <div className="flex gap-3 mb-6">
        <button onClick={start} disabled={running}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> {running ? "运行中..." : "开始演示"}
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg">
          <RotateCcw className="w-4 h-4" />
        </button>
        <span className="ml-auto text-sm text-slate-500 self-center">时间: {elapsed.toFixed(1)}s / {totalTime}s</span>
      </div>

      {/* Task Progress */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <div className="space-y-3">
          {TASK_NAMES.map((name, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="w-12 text-xs font-medium text-slate-600 dark:text-slate-400">{name}</span>
              <div className="flex-1 h-8 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
                <motion.div className="h-full rounded-full flex items-center justify-end pr-2"
                  style={{ width: `${taskStates[i]}%`, backgroundColor: COLORS[i] }}>
                  {taskStates[i] > 20 && <span className="text-[10px] text-white">{Math.round(taskStates[i])}%</span>}
                </motion.div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3 text-sm text-slate-600 dark:text-slate-400">
          预计总耗时: <strong>{totalTime}s</strong>
        </div>
      </div>

      {/* Summary Table */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <h4 className="font-bold text-slate-800 dark:text-slate-200 p-4 bg-slate-50 dark:bg-slate-900">
          条件组合总结
        </h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-400">多任务</th>
              <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-400">让出控制</th>
              <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-400">结果</th>
            </tr>
          </thead>
          <tbody>
            {[
              { mt: true, yc: true, result: "并发运行", good: true },
              { mt: true, yc: false, result: "顺序执行 (time.sleep 阻塞)", good: false },
              { mt: false, yc: true, result: "单任务运行", good: false },
              { mt: false, yc: false, result: "单任务阻塞", good: false },
            ].map((row, i) => (
              <tr key={i} className={`border-b border-slate-100 dark:border-slate-800 ${
                row.mt === multipleTasks && row.yc === yieldControl ? "bg-indigo-50 dark:bg-indigo-900/20" : ""
              }`}>
                <td className="px-4 py-2">{row.mt ? "✓" : "✗"}</td>
                <td className="px-4 py-2">{row.yc ? "✓" : "✗"}</td>
                <td className={`px-4 py-2 font-medium ${row.good ? "text-green-600" : "text-red-600"}`}>
                  {row.result}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
