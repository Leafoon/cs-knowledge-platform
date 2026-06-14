"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Cpu, SkipForward } from "lucide-react";

interface Proc {
  pid: number;
  name: string;
  state: "UNUSED" | "SLEEPING" | "RUNNABLE" | "RUNNING";
}

const stateColors: Record<string, string> = {
  UNUSED: "bg-slate-400 dark:bg-slate-600",
  SLEEPING: "bg-blue-400 dark:bg-blue-600",
  RUNNABLE: "bg-emerald-400 dark:bg-emerald-600",
  RUNNING: "bg-amber-400 dark:bg-amber-600",
};

const stateTextColors: Record<string, string> = {
  UNUSED: "text-slate-700 dark:text-slate-300",
  SLEEPING: "text-blue-700 dark:text-blue-300",
  RUNNABLE: "text-emerald-700 dark:text-emerald-300",
  RUNNING: "text-amber-700 dark:text-amber-300",
};

const initialProcs: Proc[] = [
  { pid: 0, name: "init", state: "RUNNABLE" },
  { pid: 1, name: "sh", state: "SLEEPING" },
  { pid: 2, name: "cat", state: "RUNNABLE" },
  { pid: 3, name: "echo", state: "UNUSED" },
  { pid: 4, name: "grep", state: "SLEEPING" },
  { pid: 5, name: "ls", state: "RUNNABLE" },
  { pid: 6, name: "vi", state: "UNUSED" },
  { pid: 7, name: "sleep", state: "SLEEPING" },
];

type Phase = "scan" | "found" | "switch" | "resume" | "idle";

export default function Xv6SchedulerLoop() {
  const [procs, setProcs] = useState<Proc[]>(initialProcs);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [runningPid, setRunningPid] = useState<number | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [isPlaying, setIsPlaying] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [...prev.slice(-6), msg]);
  }, []);

  const step = useCallback(() => {
    setProcs((prev) => {
      const next = [...prev];
      const idx = currentIndex;

      if (phase === "idle" || phase === "scan") {
        const p = next[idx];
        if (p.state === "RUNNABLE") {
          setPhase("found");
          addLog(`proc[${idx}] (pid=${p.pid}, ${p.name}) = RUNNABLE ✓`);
          return next;
        } else {
          setPhase("scan");
          setCurrentIndex((idx + 1) % next.length);
          addLog(`proc[${idx}] (pid=${p.pid}, ${p.name}) = ${p.state} — 跳过`);
          return next;
        }
      }

      if (phase === "found") {
        next[currentIndex].state = "RUNNING";
        setRunningPid(next[currentIndex].pid);
        setPhase("switch");
        addLog(`scheduler(): p->state = RUNNING, c->proc = p`);
        addLog(`swtch(&c->context, &p->context)`);
        return next;
      }

      if (phase === "switch") {
        setPhase("resume");
        addLog(`swtch() 返回 — 进程让出 CPU`);
        return next;
      }

      if (phase === "resume") {
        if (runningPid !== null) {
          const rIdx = next.findIndex((p) => p.pid === runningPid);
          if (rIdx >= 0 && next[rIdx].state === "RUNNING") {
            next[rIdx].state = "RUNNABLE";
          }
        }
        setRunningPid(null);
        setPhase("scan");
        setCurrentIndex((idx + 1) % next.length);
        addLog(`scheduler(): c->proc = 0, 继续扫描`);
        return next;
      }

      return next;
    });
  }, [currentIndex, phase, runningPid, addLog]);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setTimeout(step, 800);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, step]);

  const reset = () => {
    setIsPlaying(false);
    setProcs(initialProcs);
    setCurrentIndex(0);
    setRunningPid(null);
    setPhase("idle");
    setLog([]);
  };

  const stepOnce = () => {
    setIsPlaying(false);
    step();
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-emerald-500" />
          xv6 scheduler() 调度循环
        </h3>
        <div className="flex gap-2">
          <button onClick={stepOnce} className="flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
            <SkipForward className="w-3 h-3" /> 单步
          </button>
          <button onClick={() => setIsPlaying(!isPlaying)} className={`flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg transition-colors ${isPlaying ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300" : "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"}`}>
            {isPlaying ? <><Pause className="w-3 h-3" /> 暂停</> : <><Play className="w-3 h-3" /> 播放</>}
          </button>
          <button onClick={reset} className="flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
            <RotateCcw className="w-3 h-3" /> 重置
          </button>
        </div>
      </div>

      <div className="mb-4 p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
        <p className="text-xs font-mono text-slate-500 dark:text-slate-400">
          阶段：<span className={stateTextColors[phase === "idle" || phase === "scan" ? "SLEEPING" : phase === "found" ? "RUNNABLE" : "RUNNING"]}>{phase}</span>
          {" | "}扫描索引：proc[{currentIndex}]
          {runningPid !== null && ` | 运行中：pid=${runningPid}`}
        </p>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-4">
        {procs.map((p, i) => (
          <motion.div
            key={p.pid}
            layout
            className={`relative p-3 rounded-lg border-2 transition-colors ${
              i === currentIndex && (phase === "scan" || phase === "found")
                ? "border-amber-400 dark:border-amber-500"
                : "border-transparent"
            } ${p.state === "UNUSED" ? "opacity-40" : ""}`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-mono text-slate-500 dark:text-slate-400">[{i}]</span>
              <span className="text-xs font-mono text-slate-400 dark:text-slate-500">pid={p.pid}</span>
            </div>
            <p className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-1">{p.name}</p>
            <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono font-bold ${stateColors[p.state]} text-white`}>
              {p.state}
            </span>
            {i === currentIndex && phase === "scan" && (
              <motion.div
                className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-amber-400"
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ repeat: Infinity, duration: 0.8 }}
              />
            )}
          </motion.div>
        ))}
      </div>

      <div className="p-3 rounded-lg bg-slate-900 dark:bg-slate-950 border border-slate-700 h-32 overflow-y-auto">
        <p className="text-xs font-mono text-slate-400 mb-1">{"// scheduler() 日志"}</p>
        <AnimatePresence>
          {log.map((line, i) => (
            <motion.p
              key={`${i}-${line}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xs font-mono text-emerald-400"
            >
              {">"} {line}
            </motion.p>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
