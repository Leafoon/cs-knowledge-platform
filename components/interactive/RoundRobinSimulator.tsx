"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Plus, Trash2, Timer, ListOrdered } from "lucide-react";

interface Process {
  id: number;
  name: string;
  arrival: number;
  burst: number;
}

interface GanttSlice {
  pid: number;
  name: string;
  start: number;
  end: number;
}

interface QueueSnapshot {
  time: number;
  queue: string[];
  running: string | null;
}

const COLORS = [
  "bg-blue-400", "bg-emerald-400", "bg-amber-400", "bg-rose-400",
  "bg-violet-400", "bg-cyan-400", "bg-pink-400", "bg-lime-400",
];

function simulateRR(procs: Process[], quantum: number) {
  const n = procs.length;
  const remaining = procs.map(p => ({ ...p, remaining: p.burst }));
  const slices: GanttSlice[] = [];
  const queueSnaps: QueueSnapshot[] = [];
  const sorted = [...remaining].sort((a, b) => a.arrival - b.arrival);
  const queue: number[] = [];
  const inQueue = new Set<number>();
  let time = 0;
  let arrIdx = 0;

  const addArrivals = () => {
    while (arrIdx < sorted.length && sorted[arrIdx].arrival <= time) {
      const pi = remaining.findIndex(r => r.id === sorted[arrIdx].id);
      if (pi >= 0 && !inQueue.has(pi) && remaining[pi].remaining > 0) {
        queue.push(pi);
        inQueue.add(pi);
      }
      arrIdx++;
    }
  };

  addArrivals();
  if (queue.length === 0 && arrIdx < sorted.length) {
    time = sorted[arrIdx].arrival;
    addArrivals();
  }

  while (queue.length > 0 || arrIdx < sorted.length) {
    if (queue.length === 0) {
      time = sorted[arrIdx].arrival;
      addArrivals();
      continue;
    }

    const ci = queue.shift()!;
    inQueue.delete(ci);
    const p = remaining[ci];
    const exec = Math.min(quantum, p.remaining);

    queueSnaps.push({
      time,
      queue: queue.map(qi => remaining[qi].name),
      running: p.name,
    });

    slices.push({ pid: p.id, name: p.name, start: time, end: time + exec });
    p.remaining -= exec;
    time += exec;
    addArrivals();

    if (p.remaining > 0) {
      queue.push(ci);
      inQueue.add(ci);
    }
  }

  return { slices, queueSnaps };
}

export default function RoundRobinSimulator() {
  const [processes, setProcesses] = useState<Process[]>([
    { id: 1, name: "P1", arrival: 0, burst: 10 },
    { id: 2, name: "P2", arrival: 0, burst: 4 },
    { id: 3, name: "P3", arrival: 0, burst: 6 },
    { id: 4, name: "P4", arrival: 0, burst: 2 },
  ]);
  const [quantum, setQuantum] = useState(3);
  const [nextId, setNextId] = useState(5);
  const [slices, setSlices] = useState<GanttSlice[]>([]);
  const [queueSnaps, setQueueSnaps] = useState<QueueSnapshot[]>([]);
  const [visibleSlices, setVisibleSlices] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [snapIdx, setSnapIdx] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const totalTime = slices.length > 0 ? Math.max(...slices.map(s => s.end)) : 0;

  const addProcess = () => {
    setProcesses(prev => [...prev, { id: nextId, name: `P${nextId}`, arrival: 0, burst: 5 }]);
    setNextId(n => n + 1);
  };

  const removeProcess = (id: number) => setProcesses(prev => prev.filter(p => p.id !== id));

  const updateProcess = (id: number, field: keyof Omit<Process, "id">, value: string) => {
    setProcesses(prev => prev.map(p => p.id === id ? { ...p, [field]: field === "name" ? value : Math.max(0, parseInt(value) || 0) } : p));
  };

  const startSimulation = useCallback(() => {
    const result = simulateRR(processes, quantum);
    setSlices(result.slices);
    setQueueSnaps(result.queueSnaps);
    setVisibleSlices(0);
    setSnapIdx(0);
    setPlaying(true);
  }, [processes, quantum]);

  const reset = () => {
    setPlaying(false);
    if (timerRef.current) clearInterval(timerRef.current);
    setSlices([]);
    setQueueSnaps([]);
    setVisibleSlices(0);
    setSnapIdx(0);
  };

  useEffect(() => {
    if (playing && visibleSlices < slices.length) {
      timerRef.current = setTimeout(() => {
        setVisibleSlices(v => v + 1);
        setSnapIdx(s => Math.min(s + 1, queueSnaps.length - 1));
      }, 600);
      return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    }
    if (visibleSlices >= slices.length) setPlaying(false);
  }, [playing, visibleSlices, slices.length, queueSnaps.length]);

  const togglePause = () => setPlaying(p => !p);

  // Compute metrics
  const metrics = processes.map(p => {
    const pSlices = slices.filter(s => s.pid === p.id);
    if (pSlices.length === 0) return { name: p.name, completion: 0, turnaround: 0, waiting: 0, response: 0 };
    const completion = Math.max(...pSlices.map(s => s.end));
    const firstStart = Math.min(...pSlices.map(s => s.start));
    return {
      name: p.name,
      completion,
      turnaround: completion - p.arrival,
      waiting: completion - p.arrival - p.burst,
      response: firstStart - p.arrival,
    };
  });

  const currentSnap = queueSnaps[snapIdx] || null;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Round Robin Scheduler Simulator
      </h2>

      {/* Quantum slider */}
      <div className="flex items-center gap-4 mb-6 justify-center">
        <Timer size={18} className="text-teal-500" />
        <label className="text-sm font-medium text-slate-600 dark:text-gray-300">Time Quantum: {quantum}ms</label>
        <input
          type="range"
          min={1}
          max={20}
          value={quantum}
          onChange={e => setQuantum(parseInt(e.target.value))}
          className="w-48 accent-teal-500"
        />
      </div>

      {/* Process table */}
      <div className="mb-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-100 dark:bg-gray-700">
              <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Process</th>
              <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Arrival</th>
              <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Burst</th>
              <th className="px-3 py-2 text-center text-slate-600 dark:text-gray-300">Action</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence>
              {processes.map(p => (
                <motion.tr key={p.id} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, x: -20 }} className="border-b border-slate-200 dark:border-gray-700">
                  <td className="px-3 py-2">
                    <input value={p.name} onChange={e => updateProcess(p.id, "name", e.target.value)} className="w-16 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100" />
                  </td>
                  <td className="px-3 py-2">
                    <input type="number" min={0} value={p.arrival} onChange={e => updateProcess(p.id, "arrival", e.target.value)} className="w-16 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100 text-center" />
                  </td>
                  <td className="px-3 py-2">
                    <input type="number" min={1} value={p.burst} onChange={e => updateProcess(p.id, "burst", e.target.value)} className="w-16 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100 text-center" />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <button onClick={() => removeProcess(p.id)} className="p-1 text-red-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors">
                      <Trash2 size={16} />
                    </button>
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 mb-6 justify-center">
        <button onClick={addProcess} className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors shadow">
          <Plus size={16} /> Add Process
        </button>
        <button onClick={startSimulation} disabled={playing || processes.length === 0} className="flex items-center gap-2 px-4 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 transition-colors shadow disabled:opacity-50">
          <Play size={16} /> Start
        </button>
        <button onClick={togglePause} disabled={slices.length === 0} className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors shadow disabled:opacity-50">
          {playing ? <Pause size={16} /> : <Play size={16} />} {playing ? "Pause" : "Resume"}
        </button>
        <button onClick={reset} className="flex items-center gap-2 px-4 py-2 bg-slate-400 text-white rounded-lg hover:bg-slate-500 transition-colors shadow">
          <RotateCcw size={16} /> Reset
        </button>
      </div>

      {/* Gantt Chart */}
      {slices.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-3">Gantt Chart</h3>
          <div className="relative bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 overflow-x-auto">
            <div className="flex items-stretch min-w-fit h-12">
              {slices.slice(0, visibleSlices).map((s, i) => {
                const w = ((s.end - s.start) / Math.max(totalTime, 1)) * 100;
                const color = COLORS[(s.pid - 1) % COLORS.length];
                return (
                  <motion.div
                    key={i}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.25 }}
                    className={`${color} dark:opacity-80 flex items-center justify-center text-white text-xs font-bold border-r border-white/30`}
                    style={{ width: `${Math.max(w, 2)}%`, minWidth: 24 }}
                  >
                    {s.name}
                  </motion.div>
                );
              })}
            </div>
            <div className="flex min-w-fit mt-1">
              {slices.slice(0, visibleSlices).map((s, i) => {
                const w = ((s.end - s.start) / Math.max(totalTime, 1)) * 100;
                return (
                  <div key={i} className="flex justify-between text-xs text-slate-500 dark:text-gray-400" style={{ width: `${Math.max(w, 2)}%`, minWidth: 24 }}>
                    <span>{s.start}</span>
                    {i === visibleSlices - 1 && <span>{s.end}</span>}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Ready Queue State */}
      {currentSnap && (
        <motion.div
          key={snapIdx}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700"
        >
          <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <ListOrdered size={18} /> Ready Queue at t={currentSnap.time}
          </h3>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="px-3 py-1.5 bg-teal-100 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300 rounded font-semibold text-sm">
              CPU: {currentSnap.running ?? "idle"}
            </div>
            <span className="text-slate-400">|</span>
            <span className="text-sm text-slate-600 dark:text-gray-300">Queue:</span>
            {currentSnap.queue.length === 0 ? (
              <span className="text-sm text-slate-400 italic">empty</span>
            ) : (
              currentSnap.queue.map((name, i) => (
                <span key={i} className="px-2 py-1 bg-slate-100 dark:bg-gray-700 text-slate-700 dark:text-gray-200 rounded text-sm font-mono">
                  {name}
                </span>
              ))
            )}
          </div>
        </motion.div>
      )}

      {/* Metrics */}
      {metrics.some(m => m.completion > 0) && (
        <div>
          <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-3">Process Metrics</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-teal-50 dark:bg-teal-900/30">
                  {["Process", "Completion", "Turnaround", "Waiting", "Response"].map(h => (
                    <th key={h} className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <motion.tr key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.1 }} className="hover:bg-slate-50 dark:hover:bg-gray-700/50">
                    <td className="px-3 py-2 font-medium text-slate-700 dark:text-gray-200 border border-slate-200 dark:border-gray-700 text-center">{m.name}</td>
                    <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{m.completion}</td>
                    <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{m.turnaround}</td>
                    <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{m.waiting}</td>
                    <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{m.response}</td>
                  </motion.tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="bg-teal-50 dark:bg-teal-900/30 font-semibold">
                  <td className="px-3 py-2 text-center text-slate-700 dark:text-gray-200 border border-slate-200 dark:border-gray-700">Average</td>
                  <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-slate-400">-</td>
                  <td className="px-3 py-2 text-center text-teal-600 dark:text-teal-400 border border-slate-200 dark:border-gray-700">
                    {(metrics.reduce((s, m) => s + m.turnaround, 0) / metrics.length).toFixed(2)}
                  </td>
                  <td className="px-3 py-2 text-center text-teal-600 dark:text-teal-400 border border-slate-200 dark:border-gray-700">
                    {(metrics.reduce((s, m) => s + m.waiting, 0) / metrics.length).toFixed(2)}
                  </td>
                  <td className="px-3 py-2 text-center text-teal-600 dark:text-teal-400 border border-slate-200 dark:border-gray-700">
                    {(metrics.reduce((s, m) => s + m.response, 0) / metrics.length).toFixed(2)}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
