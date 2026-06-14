"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, BookOpen, PenTool, Clock, Users, AlertTriangle } from "lucide-react";

type PriorityMode = "reader" | "writer";
type ActorRole = "reader" | "writer";
type ActorStatus = "waiting" | "reading" | "writing" | "done";

interface Actor {
  id: number;
  role: ActorRole;
  status: ActorStatus;
  arrivalTime: number;
  startTime?: number;
}

export default function ReaderWriterVisualizer() {
  const [mode, setMode] = useState<PriorityMode>("reader");
  const [actors, setActors] = useState<Actor[]>([]);
  const [readerCount, setReaderCount] = useState(0);
  const [writerActive, setWriterActive] = useState(false);
  const [waitingReaders, setWaitingReaders] = useState<Actor[]>([]);
  const [waitingWriters, setWaitingWriters] = useState<Actor[]>([]);
  const [activeReaders, setActiveReaders] = useState<Actor[]>([]);
  const [activeWriter, setActiveWriter] = useState<Actor | null>(null);
  const [completedActors, setCompletedActors] = useState<Actor[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const [starvationCount, setStarvationCount] = useState(0);
  const nextId = useRef(1);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [msg, ...prev].slice(0, 15));
  }, []);

  const addActor = useCallback((role: ActorRole) => {
    const actor: Actor = {
      id: nextId.current++,
      role,
      status: "waiting",
      arrivalTime: tick,
    };
    setActors((prev) => [...prev, actor]);
    if (role === "reader") {
      setWaitingReaders((prev) => [...prev, actor]);
    } else {
      setWaitingWriters((prev) => [...prev, actor]);
    }
    addLog(`${role === "reader" ? "Reader" : "Writer"} ${actor.id} arrives`);
  }, [tick, addLog]);

  const reset = useCallback(() => {
    setIsRunning(false);
    setActors([]);
    setReaderCount(0);
    setWriterActive(false);
    setWaitingReaders([]);
    setWaitingWriters([]);
    setActiveReaders([]);
    setActiveWriter(null);
    setCompletedActors([]);
    setLog([]);
    setTick(0);
    setStarvationCount(0);
    nextId.current = 1;
  }, []);

  const simulateStep = useCallback(() => {
    setTick((t) => t + 1);

    // Try to admit actors based on priority mode
    if (mode === "reader") {
      // Reader priority: readers always go if no writer active
      if (!writerActive) {
        // Admit all waiting readers
        if (waitingReaders.length > 0) {
          const newReaders = waitingReaders.map((r) => ({ ...r, status: "reading" as ActorStatus, startTime: tick }));
          setActiveReaders((prev) => [...prev, ...newReaders]);
          setReaderCount((c) => c + newReaders.length);
          setWaitingReaders([]);
          newReaders.forEach((r) => addLog(`Reader ${r.id} starts reading (${readerCount + newReaders.length} readers)`));
        }
        // Also try to admit a writer if no readers waiting
        if (waitingWriters.length > 0 && waitingReaders.length === 0 && activeReaders.length === 0) {
          const w = { ...waitingWriters[0], status: "writing" as ActorStatus, startTime: tick };
          setActiveWriter(w);
          setWriterActive(true);
          setWaitingWriters((prev) => prev.slice(1));
          addLog(`Writer ${w.id} starts writing`);
        }
      } else {
        // Writer active, everyone waits
        // Check starvation for readers
        if (waitingReaders.length > 3) {
          setStarvationCount((s) => s + 1);
        }
      }

      // Finish active actors randomly
      setActiveReaders((prev) => {
        const remaining: Actor[] = [];
        prev.forEach((r) => {
          if (Math.random() < 0.3) {
            setCompletedActors((c) => [...c, { ...r, status: "done" }]);
            setReaderCount((c) => c - 1);
            addLog(`Reader ${r.id} finishes reading`);
          } else {
            remaining.push(r);
          }
        });
        return remaining;
      });

      if (activeWriter && Math.random() < 0.25) {
        setCompletedActors((c) => [...c, { ...activeWriter, status: "done" }]);
        setWriterActive(false);
        setActiveWriter(null);
        addLog(`Writer ${activeWriter.id} finishes writing`);
      }
    } else {
      // Writer priority: writer goes first, blocks new readers
      if (!writerActive && activeReaders.length === 0) {
        // Admit a writer first
        if (waitingWriters.length > 0) {
          const w = { ...waitingWriters[0], status: "writing" as ActorStatus, startTime: tick };
          setActiveWriter(w);
          setWriterActive(true);
          setWaitingWriters((prev) => prev.slice(1));
          addLog(`Writer ${w.id} starts writing (writer priority)`);
        } else if (waitingReaders.length > 0) {
          // No writers waiting, admit readers
          const newReaders = waitingReaders.map((r) => ({ ...r, status: "reading" as ActorStatus, startTime: tick }));
          setActiveReaders((prev) => [...prev, ...newReaders]);
          setReaderCount((c) => c + newReaders.length);
          setWaitingReaders([]);
          newReaders.forEach((r) => addLog(`Reader ${r.id} starts reading`));
        }
      } else if (writerActive) {
        // Writer active, check starvation of waiting writers
        if (waitingWriters.length > 2) {
          setStarvationCount((s) => s + 1);
        }
      }

      // Finish active actors
      setActiveReaders((prev) => {
        const remaining: Actor[] = [];
        prev.forEach((r) => {
          if (Math.random() < 0.3) {
            setCompletedActors((c) => [...c, { ...r, status: "done" }]);
            setReaderCount((c) => c - 1);
            addLog(`Reader ${r.id} finishes reading`);
          } else {
            remaining.push(r);
          }
        });
        return remaining;
      });

      if (activeWriter && Math.random() < 0.25) {
        setCompletedActors((c) => [...c, { ...activeWriter, status: "done" }]);
        setWriterActive(false);
        setActiveWriter(null);
        addLog(`Writer ${activeWriter.id} finishes writing`);
      }
    }

    // Auto-generate actors
    if (Math.random() < 0.5) {
      addActor(Math.random() < 0.7 ? "reader" : "writer");
    }
  }, [mode, writerActive, waitingReaders, waitingWriters, activeReaders, activeWriter, tick, readerCount, addLog, addActor]);

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(simulateStep, 1200);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, simulateStep]);

  const renderActorBar = (actor: Actor, index: number) => {
    const isReader = actor.role === "reader";
    return (
      <motion.div
        key={actor.id}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 20 }}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${
          isReader
            ? "bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800"
            : "bg-rose-50 border-rose-200 dark:bg-rose-900/20 dark:border-rose-800"
        }`}
      >
        {isReader ? (
          <BookOpen className="w-4 h-4 text-blue-500" />
        ) : (
          <PenTool className="w-4 h-4 text-rose-500" />
        )}
        <span className="text-xs font-mono font-bold text-slate-700 dark:text-gray-200">
          {isReader ? "R" : "W"}{actor.id}
        </span>
        {actor.status === "reading" && (
          <motion.div
            animate={{ scaleX: [0, 1] }}
            transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
            className="h-1.5 bg-blue-400 rounded-full flex-1 min-w-[40px]"
          />
        )}
        {actor.status === "writing" && (
          <motion.div
            animate={{ scaleX: [0, 1] }}
            transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
            className="h-1.5 bg-rose-400 rounded-full flex-1 min-w-[40px]"
          />
        )}
        {actor.status === "waiting" && (
          <Clock className="w-3 h-3 text-amber-400 animate-pulse" />
        )}
      </motion.div>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Reader-Writer Lock Visualization
      </h2>

      {/* Priority Mode Toggle */}
      <div className="flex justify-center gap-4 mb-6">
        {(["reader", "writer"] as PriorityMode[]).map((m) => (
          <button
            key={m}
            onClick={() => { setMode(m); reset(); }}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              mode === m
                ? m === "reader"
                  ? "bg-blue-500 text-white shadow-lg"
                  : "bg-rose-500 text-white shadow-lg"
                : "bg-white dark:bg-gray-800 text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700 hover:bg-slate-50"
            }`}
          >
            <div className="flex items-center gap-2">
              {m === "reader" ? <BookOpen className="w-4 h-4" /> : <PenTool className="w-4 h-4" />}
              {m === "reader" ? "Reader Priority" : "Writer Priority"}
            </div>
            <div className="text-xs opacity-75 mt-1">
              {m === "reader" ? "Readers may starve writers" : "Writers may starve readers"}
            </div>
          </button>
        ))}
      </div>

      {/* Status Bar */}
      <div className="flex justify-center gap-4 mb-6">
        <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <BookOpen className="w-4 h-4 text-blue-500" />
          <span className="text-sm font-bold text-blue-700 dark:text-blue-300">Active Readers: {activeReaders.length}</span>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-rose-50 dark:bg-rose-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
          <PenTool className="w-4 h-4 text-rose-500" />
          <span className="text-sm font-bold text-rose-700 dark:text-rose-300">Active Writer: {activeWriter ? `W${activeWriter.id}` : "None"}</span>
        </div>
        {starvationCount > 0 && (
          <div className="flex items-center gap-2 px-4 py-2 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            <span className="text-sm font-bold text-amber-700 dark:text-amber-300">Starvation events: {starvationCount}</span>
          </div>
        )}
      </div>

      {/* Main Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Waiting Queue */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <Clock className="w-4 h-4 text-amber-500" />
            Waiting Queue
          </h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            <AnimatePresence>
              {waitingReaders.map((r, i) => renderActorBar(r, i))}
              {waitingWriters.map((w, i) => renderActorBar(w, i + waitingReaders.length))}
            </AnimatePresence>
            {waitingReaders.length === 0 && waitingWriters.length === 0 && (
              <div className="text-xs text-slate-400 dark:text-gray-500 text-center py-4">Empty</div>
            )}
          </div>
        </div>

        {/* Critical Section */}
        <div className="bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-800 dark:to-gray-700 rounded-lg p-4 border-2 border-slate-300 dark:border-gray-600">
          <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 text-center">Critical Section (Shared Resource)</h4>

          {/* Active Readers as parallel bars */}
          <div className="mb-3">
            <div className="text-xs text-slate-500 dark:text-gray-400 mb-2">Active Readers:</div>
            <div className="space-y-1">
              <AnimatePresence>
                {activeReaders.map((r, i) => (
                  <motion.div
                    key={r.id}
                    initial={{ width: 0 }}
                    animate={{ width: "100%" }}
                    exit={{ width: 0 }}
                    className="h-6 bg-blue-200 dark:bg-blue-800 rounded flex items-center px-2"
                  >
                    <BookOpen className="w-3 h-3 text-blue-600 dark:text-blue-300 mr-1" />
                    <span className="text-xs font-mono text-blue-700 dark:text-blue-200">R{r.id} reading...</span>
                    <motion.div
                      animate={{ scaleX: [0, 1] }}
                      transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                      className="h-1 bg-blue-400 rounded-full flex-1 ml-2"
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
              {activeReaders.length === 0 && (
                <div className="h-6 bg-slate-100 dark:bg-gray-700 rounded flex items-center justify-center">
                  <span className="text-xs text-slate-400 dark:text-gray-500">No active readers</span>
                </div>
              )}
            </div>
          </div>

          {/* Active Writer */}
          <div>
            <div className="text-xs text-slate-500 dark:text-gray-400 mb-2">Active Writer:</div>
            <AnimatePresence>
              {activeWriter ? (
                <motion.div
                  key={activeWriter.id}
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  exit={{ width: 0 }}
                  className="h-8 bg-rose-200 dark:bg-rose-800 rounded flex items-center px-2"
                >
                  <PenTool className="w-4 h-4 text-rose-600 dark:text-rose-300 mr-1" />
                  <span className="text-xs font-mono text-rose-700 dark:text-rose-200">W{activeWriter.id} writing...</span>
                  <motion.div
                    animate={{ scaleX: [0, 1] }}
                    transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                    className="h-1.5 bg-rose-400 rounded-full flex-1 ml-2"
                  />
                </motion.div>
              ) : (
                <div className="h-8 bg-slate-100 dark:bg-gray-700 rounded flex items-center justify-center">
                  <span className="text-xs text-slate-400 dark:text-gray-500">No active writer</span>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Completed */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <Users className="w-4 h-4 text-green-500" />
            Completed ({completedActors.length})
          </h4>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {completedActors.slice(-8).map((a) => (
              <div key={a.id} className="flex items-center gap-2 text-xs text-slate-500 dark:text-gray-400">
                {a.role === "reader" ? <BookOpen className="w-3 h-3" /> : <PenTool className="w-3 h-3" />}
                <span className="font-mono">{a.role === "reader" ? "R" : "W"}{a.id}</span>
                <span className="text-green-500">done</span>
              </div>
            ))}
            {completedActors.length === 0 && (
              <div className="text-xs text-slate-400 dark:text-gray-500 text-center py-4">None yet</div>
            )}
          </div>
        </div>
      </div>

      {/* Starvation Warning */}
      <AnimatePresence>
        {starvationCount > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mb-4 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-300 dark:border-amber-700 rounded-lg flex items-center gap-2"
          >
            <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0" />
            <span className="text-sm text-amber-700 dark:text-amber-300">
              {mode === "reader"
                ? "Starvation detected! Continuous readers are blocking writers from ever getting access."
                : "Starvation detected! Continuous writers are blocking readers from ever getting access."}
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Event Log */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-slate-200 dark:border-gray-700 mb-4 max-h-28 overflow-y-auto">
        {log.length === 0 ? (
          <p className="text-xs text-slate-400 dark:text-gray-500 text-center">Click Run to start simulation...</p>
        ) : (
          log.map((entry, i) => (
            <div key={i} className="text-xs font-mono text-slate-600 dark:text-gray-300 py-0.5 border-b border-slate-100 dark:border-gray-700 last:border-0">
              {entry}
            </div>
          ))
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300">
          <RotateCcw className="w-5 h-5" />
        </button>
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`px-5 py-2 rounded-lg text-white font-medium flex items-center gap-2 ${
            mode === "reader" ? "bg-blue-500 hover:bg-blue-600" : "bg-rose-500 hover:bg-rose-600"
          }`}
        >
          {isRunning ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Run</>}
        </button>
        <button
          onClick={simulateStep}
          disabled={isRunning}
          className="px-4 py-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 text-sm disabled:opacity-40"
        >
          Step
        </button>
        <div className="ml-4 flex gap-2">
          <button
            onClick={() => addActor("reader")}
            className="px-3 py-2 rounded-lg bg-blue-100 hover:bg-blue-200 dark:bg-blue-900/30 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-300 text-sm font-medium"
          >
            + Reader
          </button>
          <button
            onClick={() => addActor("writer")}
            className="px-3 py-2 rounded-lg bg-rose-100 hover:bg-rose-200 dark:bg-rose-900/30 dark:hover:bg-rose-900/50 text-rose-700 dark:text-rose-300 text-sm font-medium"
          >
            + Writer
          </button>
        </div>
      </div>

      {/* Mode Description */}
      <div className="mt-4 text-center text-xs text-slate-500 dark:text-gray-400">
        {mode === "reader"
          ? "Reader Priority: Multiple readers can access simultaneously. Writers must wait until ALL readers finish. New readers can jump ahead of waiting writers."
          : "Writer Priority: Writers get priority access. New readers are blocked while a writer is waiting. Prevents writer starvation."}
      </div>
    </div>
  );
}
