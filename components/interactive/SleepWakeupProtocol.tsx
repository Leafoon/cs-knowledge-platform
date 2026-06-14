"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, SkipForward, AlertTriangle, CheckCircle2, Lock, Unlock } from "lucide-react";

interface LogEntry {
  actor: "consumer" | "producer" | "system";
  msg: string;
  highlight?: boolean;
}

type Mode = "broken" | "correct";
type Phase =
  | "idle"
  | "check-cond"
  | "acquire-plock"
  | "release-clock"
  | "set-sleeping"
  | "sched"
  | "wakeup"
  | "reacquire"
  | "done";

const phases: { id: Phase; label: string; desc: string }[] = [
  { id: "check-cond", label: "检查条件", desc: "消费者检查 buffer_ready == 0" },
  { id: "acquire-plock", label: "acquire(&p->lock)", desc: "获取进程锁" },
  { id: "release-clock", label: "release(&buffer_lock)", desc: "释放调用者的锁" },
  { id: "set-sleeping", label: "设 SLEEPING + chan", desc: "设置睡眠状态和通道" },
  { id: "sched", label: "sched()", desc: "让出 CPU，进入调度器" },
  { id: "wakeup", label: "wakeup()", desc: "生产者写入数据并唤醒" },
  { id: "reacquire", label: "重新获取锁", desc: "被唤醒后重新获取 buffer_lock" },
  { id: "done", label: "完成", desc: "消费者继续执行" },
];

export default function SleepWakeupProtocol() {
  const [mode, setMode] = useState<Mode>("correct");
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [bufferReady, setBufferReady] = useState(0);
  const [consumerState, setConsumerState] = useState("RUNNING");
  const [pLockHeld, setPLockHeld] = useState(false);
  const [bufLockHeld, setBufLockHeld] = useState<"none" | "consumer" | "producer">("none");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const addLog = useCallback((entry: LogEntry) => {
    setLog((prev) => [...prev.slice(-8), entry]);
  }, []);

  const resetSim = useCallback(() => {
    setPhaseIdx(0);
    setLog([]);
    setBufferReady(0);
    setConsumerState("RUNNING");
    setPLockHeld(false);
    setBufLockHeld("none");
  }, []);

  const brokenPhases: (() => void)[] = [
    () => {
      addLog({ actor: "consumer", msg: "检查 buffer_ready == 0 → true" });
      setPhaseIdx(0);
    },
    () => {
      addLog({ actor: "consumer", msg: "❌ 没有获取 p->lock" });
      addLog({ actor: "system", msg: "--- 此时生产者获得执行 ---", highlight: true });
      setPhaseIdx(1);
    },
    () => {
      addLog({ actor: "producer", msg: "buffer_ready = 1" });
      setBufferReady(1);
    },
    () => {
      addLog({ actor: "producer", msg: "wakeup(&buffer) → 没有进程在 chan 上！" });
      addLog({ actor: "system", msg: "⚠️ Lost Wakeup! wakeup 无效", highlight: true });
      setPhaseIdx(4);
    },
    () => {
      addLog({ actor: "consumer", msg: "p->state = SLEEPING, p->chan = &buffer" });
      setConsumerState("SLEEPING");
      setPhaseIdx(5);
    },
    () => {
      addLog({ actor: "consumer", msg: "sched() → 永远不会被唤醒！", highlight: true });
      setConsumerState("SLEEPING");
      setPhaseIdx(7);
    },
  ];

  const correctPhases: (() => void)[] = [
    () => {
      addLog({ actor: "consumer", msg: "acquire(&buffer_lock)" });
      setBufLockHeld("consumer");
      setPhaseIdx(0);
    },
    () => {
      addLog({ actor: "consumer", msg: "检查 buffer_ready == 0 → true，进入 sleep" });
    },
    () => {
      addLog({ actor: "consumer", msg: "sleep() 内部: acquire(&p->lock)" });
      setPLockHeld(true);
      setPhaseIdx(1);
    },
    () => {
      addLog({ actor: "consumer", msg: "sleep() 内部: release(&buffer_lock)" });
      setBufLockHeld("none");
      addLog({ actor: "system", msg: "--- 生产者现在可以获取 buffer_lock ---", highlight: true });
      setPhaseIdx(2);
    },
    () => {
      addLog({ actor: "consumer", msg: "p->state = SLEEPING, p->chan = &buffer" });
      setConsumerState("SLEEPING");
      setPhaseIdx(3);
    },
    () => {
      addLog({ actor: "consumer", msg: "sched() → 让出 CPU" });
      setPLockHeld(false);
      setPhaseIdx(4);
    },
    () => {
      addLog({ actor: "producer", msg: "acquire(&buffer_lock)" });
      setBufLockHeld("producer");
    },
    () => {
      addLog({ actor: "producer", msg: "buffer_ready = 1" });
      setBufferReady(1);
      setPhaseIdx(5);
    },
    () => {
      addLog({ actor: "producer", msg: "wakeup(&buffer) → 看到消费者在 chan 上睡眠 ✓" });
      setConsumerState("RUNNABLE");
      addLog({ actor: "system", msg: "✅ 正确唤醒！消费者变为 RUNNABLE", highlight: true });
    },
    () => {
      addLog({ actor: "producer", msg: "release(&buffer_lock)" });
      setBufLockHeld("none");
    },
    () => {
      addLog({ actor: "consumer", msg: "被唤醒: acquire(&p->lock) → release(&p->lock)" });
      setPLockHeld(true);
      setPhaseIdx(6);
    },
    () => {
      addLog({ actor: "consumer", msg: "重新获取 buffer_lock" });
      setBufLockHeld("consumer");
      setPLockHeld(false);
      setPhaseIdx(7);
    },
    () => {
      addLog({ actor: "consumer", msg: "buffer_ready == 1 → 退出 while 循环 ✓" });
      setConsumerState("RUNNING");
      setPhaseIdx(8);
      addLog({ actor: "system", msg: "✅ 消费者成功获取数据！", highlight: true });
    },
  ];

  const step = useCallback(() => {
    const phases = mode === "broken" ? brokenPhases : correctPhases;
    if (phaseIdx < phases.length) {
      phases[phaseIdx]();
      if (mode === "broken" && phaseIdx >= brokenPhases.length - 1) return;
      if (mode === "correct" && phaseIdx >= correctPhases.length - 1) return;
      setPhaseIdx((p) => p + 1);
    }
  }, [phaseIdx, mode, addLog]);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setTimeout(() => {
        step();
      }, 1200);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, step]);

  const handleModeChange = (m: Mode) => {
    setMode(m);
    setIsPlaying(false);
    resetSim();
  };

  const isBrokenLostWakeup = mode === "broken" && phaseIdx >= 4;

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
          {mode === "broken" ? <AlertTriangle className="w-5 h-5 text-red-500" /> : <CheckCircle2 className="w-5 h-5 text-emerald-500" />}
          sleep / wakeup 协议
        </h3>
        <div className="flex gap-2">
          <button onClick={() => handleModeChange("broken")} className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${mode === "broken" ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 font-bold" : "bg-slate-100 dark:bg-slate-800 text-slate-500"}`}>
            ❌ 错误实现
          </button>
          <button onClick={() => handleModeChange("correct")} className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${mode === "correct" ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 font-bold" : "bg-slate-100 dark:bg-slate-800 text-slate-500"}`}>
            ✅ 正确实现
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
          <p className="text-xs font-bold text-slate-500 dark:text-slate-400 mb-1">buffer_ready</p>
          <p className={`text-2xl font-mono font-bold ${bufferReady ? "text-emerald-600 dark:text-emerald-400" : "text-slate-400 dark:text-slate-500"}`}>{bufferReady}</p>
        </div>
        <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
          <p className="text-xs font-bold text-slate-500 dark:text-slate-400 mb-1">消费者状态</p>
          <p className={`text-sm font-mono font-bold ${consumerState === "SLEEPING" ? "text-blue-600 dark:text-blue-400" : consumerState === "RUNNABLE" ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}`}>{consumerState}</p>
        </div>
        <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
          <p className="text-xs font-bold text-slate-500 dark:text-slate-400 mb-1">锁状态</p>
          <div className="flex items-center gap-1">
            {pLockHeld ? <Lock className="w-3 h-3 text-red-500" /> : <Unlock className="w-3 h-3 text-slate-400" />}
            <span className="text-xs font-mono text-slate-600 dark:text-slate-300">{"p->lock"}</span>
          </div>
          <div className="flex items-center gap-1 mt-0.5">
            {bufLockHeld !== "none" ? <Lock className="w-3 h-3 text-red-500" /> : <Unlock className="w-3 h-3 text-slate-400" />}
            <span className="text-xs font-mono text-slate-600 dark:text-slate-300">buffer_lock ({bufLockHeld})</span>
          </div>
        </div>
      </div>

      {mode === "correct" && (
        <div className="mb-4">
          <div className="flex gap-1 flex-wrap">
            {phases.map((p, i) => (
              <div
                key={p.id}
                className={`px-2 py-1 rounded text-xs font-mono ${
                  i === phaseIdx
                    ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-bold border border-blue-300 dark:border-blue-700"
                    : i < phaseIdx
                    ? "bg-emerald-50 dark:bg-emerald-900/10 text-emerald-600 dark:text-emerald-400"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-400"
                }`}
                title={p.desc}
              >
                {p.label}
              </div>
            ))}
          </div>
        </div>
      )}

      <AnimatePresence>
        {isBrokenLostWakeup && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-4 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border-2 border-red-300 dark:border-red-700"
          >
            <p className="text-sm font-bold text-red-700 dark:text-red-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Lost Wakeup 发生！
            </p>
            <p className="text-xs text-red-600 dark:text-red-400 mt-1">
              wakeup() 在消费者进入睡眠之前被调用，信号丢失。消费者将永远睡眠。
              原因：检查条件和进入睡眠不是原子操作。
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex gap-2 mb-3">
        <button onClick={() => { setIsPlaying(false); step(); }} className="flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          <SkipForward className="w-3 h-3" /> 单步
        </button>
        <button onClick={() => setIsPlaying(!isPlaying)} className={`flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg transition-colors ${isPlaying ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300" : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"}`}>
          {isPlaying ? <><Pause className="w-3 h-3" /> 暂停</> : <><Play className="w-3 h-3" /> 播放</>}
        </button>
        <button onClick={() => { setIsPlaying(false); resetSim(); }} className="flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>

      <div className="p-3 rounded-lg bg-slate-900 dark:bg-slate-950 border border-slate-700 h-40 overflow-y-auto">
        <p className="text-xs font-mono text-slate-400 mb-1">{"/ " + (mode === "broken" ? "错误实现 — Lost Wakeup" : "正确实现 — 锁协议")}</p>
        <AnimatePresence>
          {log.map((entry, i) => (
            <motion.p
              key={`${i}-${entry.msg}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className={`text-xs font-mono ${
                entry.actor === "consumer"
                  ? "text-blue-400"
                  : entry.actor === "producer"
                  ? "text-emerald-400"
                  : entry.highlight
                  ? "text-amber-400 font-bold"
                  : "text-slate-400"
              }`}
            >
              [{entry.actor}] {entry.msg}
            </motion.p>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
