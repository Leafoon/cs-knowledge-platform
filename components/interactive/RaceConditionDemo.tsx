"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Play, Pause, RotateCcw, Lock, Unlock, AlertTriangle, ChevronRight } from "lucide-react";

type Instruction = { op: string; arg?: string };
type ThreadState = {
  id: number;
  instructions: Instruction[];
  pc: number;
  register: number | null;
  color: string;
  bg: string;
  border: string;
  light: string;
};

const COLORS = [
  { color: "text-blue-600", bg: "bg-blue-500", border: "border-blue-600", light: "bg-blue-50 dark:bg-blue-900/30" },
  { color: "text-emerald-600", bg: "bg-emerald-500", border: "border-emerald-600", light: "bg-emerald-50 dark:bg-emerald-900/30" },
  { color: "text-purple-600", bg: "bg-purple-500", border: "border-purple-600", light: "bg-purple-50 dark:bg-purple-900/30" },
  { color: "text-orange-600", bg: "bg-orange-500", border: "border-orange-600", light: "bg-orange-50 dark:bg-orange-900/30" },
];

function makeInstructions(useLock: boolean): Instruction[] {
  const instrs: Instruction[] = [];
  if (useLock) instrs.push({ op: "LOCK", arg: "mutex" });
  instrs.push({ op: "LOAD", arg: "counter" });
  instrs.push({ op: "ADD", arg: "1" });
  instrs.push({ op: "STORE", arg: "counter" });
  if (useLock) instrs.push({ op: "UNLOCK", arg: "mutex" });
  return instrs;
}

function opLabel(op: string): string {
  switch (op) {
    case "LOCK": return "LOCK mutex";
    case "UNLOCK": return "UNLOCK mutex";
    case "LOAD": return "R1 = LOAD counter";
    case "ADD": return "R1 = R1 + 1";
    case "STORE": return "STORE R1 -> counter";
    default: return op;
  }
}

export default function RaceConditionDemo() {
  const [numThreads, setNumThreads] = useState(2);
  const [useLock, setUseLock] = useState(false);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(800);
  const [counter, setCounter] = useState(0);
  const [threads, setThreads] = useState<ThreadState[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [finished, setFinished] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const turnRef = useRef(0);

  const initThreads = useCallback(() => {
    const t: ThreadState[] = [];
    for (let i = 0; i < numThreads; i++) {
      t.push({
        id: i,
        instructions: makeInstructions(useLock),
        pc: 0,
        register: null,
        ...COLORS[i],
      });
    }
    setThreads(t);
    setCounter(0);
    setLog([]);
    setFinished(false);
    turnRef.current = 0;
  }, [numThreads, useLock]);

  useEffect(() => {
    initThreads();
  }, [initThreads]);

  const step = useCallback(() => {
    setThreads((prev) => {
      const all = prev.map((t) => ({ ...t }));
      const activeThreads = all.filter((t) => t.pc < t.instructions.length);
      if (activeThreads.length === 0) {
        setRunning(false);
        setFinished(true);
        if (intervalRef.current) clearInterval(intervalRef.current);
        return all;
      }

      // Round-robin pick among active threads
      let pick = -1;
      for (let tries = 0; tries < all.length; tries++) {
        const idx = (turnRef.current + tries) % all.length;
        if (all[idx].pc < all[idx].instructions.length) {
          pick = idx;
          break;
        }
      }
      if (pick === -1) {
        setRunning(false);
        setFinished(true);
        return all;
      }
      turnRef.current = (pick + 1) % all.length;

      const t = all[pick];
      const instr = t.instructions[t.pc];
      const threadLabel = `T${t.id}`;

      if (instr.op === "LOCK") {
        setLog((l) => [...l, `${threadLabel}: LOCK mutex (acquired)`]);
        t.pc++;
      } else if (instr.op === "UNLOCK") {
        setLog((l) => [...l, `${threadLabel}: UNLOCK mutex (released)`]);
        t.pc++;
      } else if (instr.op === "LOAD") {
        setCounter((c) => {
          t.register = c;
          setLog((l) => [...l, `${threadLabel}: LOAD counter -> R1 = ${c}`]);
          return c;
        });
        t.pc++;
      } else if (instr.op === "ADD") {
        const newVal = (t.register ?? 0) + 1;
        t.register = newVal;
        setLog((l) => [...l, `${threadLabel}: ADD 1 -> R1 = ${newVal}`]);
        t.pc++;
      } else if (instr.op === "STORE") {
        const val = t.register ?? 0;
        setCounter(val);
        setLog((l) => [...l, `${threadLabel}: STORE R1 = ${val} -> counter`]);
        t.pc++;
      }

      return all;
    });
  }, []);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(step, speed);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, speed, step]);

  const reset = () => {
    setRunning(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    initThreads();
  };

  const expectedTotal = numThreads;
  const hasRace = finished && !useLock && counter < expectedTotal;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center flex items-center justify-center gap-2">
        <Zap className="w-7 h-7 text-rose-600" />
        Race Condition Demonstration
      </h2>
      <p className="text-center text-slate-500 dark:text-gray-400 text-sm mb-6">
        Watch how threads interleave on a shared counter — with and without locks
      </p>

      {/* Controls */}
      <div className="flex flex-wrap justify-center gap-3 mb-6">
        <button
          onClick={() => setRunning(!running)}
          disabled={finished}
          className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 ${
            running ? "bg-amber-500 text-white" : "bg-emerald-600 text-white hover:bg-emerald-700"
          } disabled:opacity-50`}
        >
          {running ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {running ? "Pause" : "Start"}
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 bg-white text-slate-700 border border-slate-300 rounded-lg hover:bg-red-50 dark:bg-gray-700 dark:text-gray-200 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>

        <div className="flex items-center gap-2 bg-white dark:bg-gray-700 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-gray-600">
          <label className="text-xs text-slate-500 dark:text-gray-400">Threads:</label>
          <select
            value={numThreads}
            onChange={(e) => setNumThreads(Number(e.target.value))}
            disabled={running}
            className="bg-transparent text-sm font-medium text-slate-700 dark:text-gray-200 outline-none"
          >
            {[2, 3, 4].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2 bg-white dark:bg-gray-700 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-gray-600">
          <label className="text-xs text-slate-500 dark:text-gray-400">Speed:</label>
          <input
            type="range"
            min={200}
            max={2000}
            step={100}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-24"
          />
          <span className="text-xs text-slate-600 dark:text-gray-300 w-12">{speed}ms</span>
        </div>

        <button
          onClick={() => { if (!running) setUseLock(!useLock); }}
          className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 ${
            useLock ? "bg-emerald-600 text-white" : "bg-red-500 text-white"
          } ${running ? "opacity-50 cursor-not-allowed" : "hover:brightness-110"}`}
          disabled={running}
        >
          {useLock ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
          {useLock ? "Lock ON" : "Lock OFF"}
        </button>
      </div>

      {/* Main Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-6 items-start mb-6">
        {/* Thread Lanes */}
        <div className="space-y-4">
          {threads.map((t, i) => (
            <div key={t.id} className={`${t.light} rounded-xl p-4 border ${t.border} border-opacity-30`}>
              <div className="flex items-center justify-between mb-3">
                <span className={`font-bold text-sm ${t.color}`}>Thread {t.id}</span>
                <span className="text-xs text-slate-500 dark:text-gray-400">
                  PC: {t.pc}/{t.instructions.length}
                  {t.register !== null && ` | R1: ${t.register}`}
                </span>
              </div>
              <div className="space-y-1.5">
                {t.instructions.map((instr, j) => {
                  const isCurrent = t.pc === j && running;
                  const isDone = t.pc > j;
                  return (
                    <motion.div
                      key={j}
                      animate={{
                        scale: isCurrent ? 1.02 : 1,
                      }}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-mono transition-all ${
                        isDone
                          ? "text-slate-400 line-through"
                          : isCurrent
                          ? "text-slate-800 dark:text-gray-100 font-semibold shadow-sm bg-white/90 dark:bg-gray-700/90"
                          : "text-slate-600 dark:text-gray-300 bg-white/40"
                      }`}
                    >
                      {isCurrent && <ChevronRight className="w-3 h-3 text-rose-500" />}
                      {isDone && <span className="text-emerald-500 text-xs">&#10003;</span>}
                      <span>{opLabel(instr.op)}</span>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Shared Memory */}
        <div className="flex flex-col items-center justify-center">
          <div className="bg-white dark:bg-gray-800 border-2 border-slate-300 dark:border-gray-600 rounded-xl p-6 shadow-lg text-center min-w-[140px]">
            <div className="text-xs text-slate-500 dark:text-gray-400 mb-2 font-medium">Shared Memory</div>
            <div className="text-xs text-slate-400 dark:text-gray-500 mb-1 font-mono">counter</div>
            <motion.div
              key={counter}
              initial={{ scale: 1.3 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 300 }}
              className={`text-4xl font-bold ${
                hasRace ? "text-red-500" : useLock && finished ? "text-emerald-500" : "text-slate-800 dark:text-gray-100"
              }`}
            >
              {counter}
            </motion.div>
            <div className="text-xs text-slate-400 dark:text-gray-500 mt-2">
              Expected: {expectedTotal}
            </div>
            {hasRace && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-2 text-xs text-red-500 font-semibold flex items-center gap-1 justify-center"
              >
                <AlertTriangle className="w-3 h-3" />
                RACE DETECTED!
              </motion.div>
            )}
            {useLock && finished && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-2 text-xs text-emerald-500 font-semibold"
              >
                CORRECT!
              </motion.div>
            )}
          </div>
          {!useLock && (
            <div className="mt-2 text-xs text-red-400 dark:text-red-500 text-center max-w-[140px]">
              No lock — threads can interleave!
            </div>
          )}
          {useLock && (
            <div className="mt-2 text-xs text-emerald-500 dark:text-emerald-400 text-center max-w-[140px] flex items-center gap-1 justify-center">
              <Lock className="w-3 h-3" /> Protected by mutex
            </div>
          )}
        </div>

        {/* Event Log */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-slate-200 dark:border-gray-700 shadow-md overflow-hidden">
          <div className="px-4 py-2 bg-slate-100 dark:bg-gray-700 border-b border-slate-200 dark:border-gray-600 flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${running ? "bg-emerald-500 animate-pulse" : "bg-slate-400"}`} />
            <span className="text-xs font-semibold text-slate-600 dark:text-gray-300">Event Log</span>
          </div>
          <div className="h-[300px] overflow-y-auto p-3 space-y-1">
            <AnimatePresence>
              {log.length === 0 ? (
                <div className="text-center text-slate-400 dark:text-gray-500 text-sm py-8">
                  Press Start to begin
                </div>
              ) : (
                log.map((entry, i) => {
                  const isLock = entry.includes("LOCK mutex (acquired)");
                  const isUnlock = entry.includes("UNLOCK mutex (released)");
                  const tidMatch = entry.match(/T(\d)/);
                  const tidNum = tidMatch ? parseInt(tidMatch[1]) : 0;
                  const c = COLORS[tidNum]?.color || "text-slate-600";
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`text-xs font-mono px-2 py-1 rounded ${
                        isLock ? "bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400" :
                        isUnlock ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400" :
                        "text-slate-600 dark:text-gray-300"
                      }`}
                    >
                      <span className={`font-semibold ${c}`}>T{tidNum}</span>
                      {entry.replace(/^T\d: /, ": ")}
                    </motion.div>
                  );
                })
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Result Summary */}
      <AnimatePresence>
        {finished && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className={`p-4 rounded-xl border ${
              hasRace
                ? "bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-700"
                : "bg-emerald-50 dark:bg-emerald-900/30 border-emerald-200 dark:border-emerald-700"
            }`}
          >
            <div className="flex items-start gap-3">
              {hasRace ? (
                <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
              ) : (
                <Lock className="w-5 h-5 text-emerald-500 mt-0.5 shrink-0" />
              )}
              <div className="text-sm">
                {hasRace ? (
                  <>
                    <strong className="text-red-700 dark:text-red-400">Race Condition Detected!</strong>
                    <p className="text-red-600 dark:text-red-300 mt-1">
                      Final counter = {counter}, expected = {expectedTotal}. Without synchronization,
                      multiple threads read the same value before any writes are committed,
                      causing lost updates. For example: Thread A loads counter (0), Thread B loads counter (0),
                      both add 1, both store 1 — one increment is lost.
                    </p>
                  </>
                ) : (
                  <>
                    <strong className="text-emerald-700 dark:text-emerald-400">Correct Result with Lock!</strong>
                    <p className="text-emerald-600 dark:text-emerald-300 mt-1">
                      Final counter = {counter}. The mutex ensures mutual exclusion — only one thread
                      can execute the critical section (LOAD, ADD, STORE) at a time,
                      preventing the lost update problem.
                    </p>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
