"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, Layers, List } from "lucide-react";

const TASK_NAMES = ["任务 A", "任务 B", "任务 C"];
const COLORS = ["#ef4444", "#3b82f6", "#10b981"];

export function GatherVsSequential() {
  const [delays, setDelays] = useState([2, 3, 1]);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [seqStates, setSeqStates] = useState([0, 0, 0]);
  const [gatherStates, setGatherStates] = useState([0, 0, 0]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const seqTotal = delays.reduce((a, b) => a + b, 0);
  const gatherTotal = Math.max(...delays);

  const start = () => {
    setRunning(true);
    setElapsed(0);
    setSeqStates([0, 0, 0]);
    setGatherStates([0, 0, 0]);

    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      const t = (Date.now() - startTime) / 1000;
      setElapsed(t);

      // Sequential: each task runs after previous
      setSeqStates(() => {
        const states = [0, 0, 0];
        let offset = 0;
        for (let i = 0; i < 3; i++) {
          if (t <= offset) states[i] = 0;
          else if (t >= offset + delays[i]) states[i] = 100;
          else states[i] = ((t - offset) / delays[i]) * 100;
          offset += delays[i];
        }
        return states;
      });

      // Gather: all tasks run concurrently
      setGatherStates(() => {
        return delays.map((d) => {
          if (t <= 0) return 0;
          if (t >= d) return 100;
          return (t / d) * 100;
        });
      });

      if (t >= seqTotal) {
        setRunning(false);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setElapsed(0);
    setSeqStates([0, 0, 0]);
    setGatherStates([0, 0, 0]);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Layers className="w-5 h-5 text-indigo-500" />
        gather() vs 顺序执行
      </h3>

      {/* Delay Sliders */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-3">设置各任务延迟（秒）</h4>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {TASK_NAMES.map((name, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-sm font-medium w-14" style={{ color: COLORS[i] }}>{name}</span>
              <input type="range" min={1} max={5} value={delays[i]}
                onChange={(e) => {
                  const newDelays = [...delays];
                  newDelays[i] = Number(e.target.value);
                  setDelays(newDelays);
                }}
                disabled={running}
                className="flex-1 accent-indigo-600" />
              <span className="text-sm font-mono w-8 text-slate-600 dark:text-slate-400">{delays[i]}s</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-3 mb-6">
        <button onClick={start} disabled={running}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> {running ? "运行中..." : "开始对比"}
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg">
          <RotateCcw className="w-4 h-4" />
        </button>
        <span className="ml-auto text-sm text-slate-500 self-center">时间: {elapsed.toFixed(1)}s</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Sequential */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-red-200 dark:border-red-800 p-5">
          <h4 className="font-bold text-red-600 dark:text-red-400 mb-2 flex items-center gap-2">
            <List className="w-4 h-4" /> 顺序执行 — await each
          </h4>
          <p className="text-xs text-slate-500 mb-3">总耗时 = {delays.join(" + ")} = {seqTotal}s</p>
          <div className="space-y-2">
            {TASK_NAMES.map((name, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-12 text-xs text-slate-600 dark:text-slate-400">{name}</span>
                <div className="flex-1 h-6 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
                  <motion.div className="h-full rounded-full flex items-center justify-end pr-1"
                    style={{ width: `${seqStates[i]}%`, backgroundColor: COLORS[i] }}>
                    {seqStates[i] > 50 && <span className="text-[9px] text-white">{Math.round(seqStates[i])}%</span>}
                  </motion.div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Gather */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-green-200 dark:border-green-800 p-5">
          <h4 className="font-bold text-green-600 dark:text-green-400 mb-2 flex items-center gap-2">
            <Layers className="w-4 h-4" /> asyncio.gather() — 并发执行
          </h4>
          <p className="text-xs text-slate-500 mb-3">总耗时 = max({delays.join(", ")}) = {gatherTotal}s</p>
          <div className="space-y-2">
            {TASK_NAMES.map((name, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-12 text-xs text-slate-600 dark:text-slate-400">{name}</span>
                <div className="flex-1 h-6 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
                  <motion.div className="h-full rounded-full flex items-center justify-end pr-1"
                    style={{ width: `${gatherStates[i]}%`, backgroundColor: COLORS[i] }}>
                    {gatherStates[i] > 50 && <span className="text-[9px] text-white">{Math.round(gatherStates[i])}%</span>}
                  </motion.div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-6 bg-indigo-50 dark:bg-indigo-900/20 border-l-4 border-indigo-400 p-4 rounded">
        <h4 className="font-bold text-indigo-800 dark:text-indigo-300 mb-2">性能对比</h4>
        <div className="text-sm text-slate-700 dark:text-slate-300">
          顺序执行需要 <strong>{seqTotal}s</strong>，gather() 只需 <strong>{gatherTotal}s</strong>，
          加速比 <strong>{(seqTotal / gatherTotal).toFixed(1)}x</strong>
        </div>
      </div>
    </div>
  );
}
