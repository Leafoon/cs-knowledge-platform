"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Trash2, Play, RotateCcw, Calculator, BarChart3 } from "lucide-react";

interface Process {
  id: number;
  name: string;
  arrival: number;
  burst: number;
}

interface GanttEntry {
  pid: number;
  name: string;
  start: number;
  end: number;
}

interface ProcessMetrics {
  name: string;
  completion: number;
  turnaround: number;
  waiting: number;
  response: number;
}

const PROCESS_COLORS = [
  "bg-blue-400", "bg-emerald-400", "bg-amber-400", "bg-rose-400",
  "bg-violet-400", "bg-cyan-400", "bg-pink-400", "bg-lime-400",
];

function scheduleFCFS(procs: Process[]): GanttEntry[] {
  const sorted = [...procs].sort((a, b) => a.arrival - b.arrival);
  const entries: GanttEntry[] = [];
  let time = 0;
  for (const p of sorted) {
    if (time < p.arrival) time = p.arrival;
    entries.push({ pid: p.id, name: p.name, start: time, end: time + p.burst });
    time += p.burst;
  }
  return entries;
}

function scheduleSJF(procs: Process[]): GanttEntry[] {
  const remaining = procs.map(p => ({ ...p, remaining: p.burst }));
  const entries: GanttEntry[] = [];
  let time = 0;
  const done: boolean[] = new Array(procs.length).fill(false);
  let completed = 0;

  while (completed < procs.length) {
    const available = remaining.filter((p, i) => !done[i] && p.arrival <= time);
    if (available.length === 0) {
      time = Math.min(...remaining.filter((_, i) => !done[i]).map(p => p.arrival));
      continue;
    }
    available.sort((a, b) => a.remaining - b.remaining);
    const chosen = available[0];
    const idx = remaining.findIndex(p => p.id === chosen.id);
    entries.push({ pid: chosen.id, name: chosen.name, start: time, end: time + chosen.remaining });
    time += chosen.remaining;
    done[idx] = true;
    completed++;
  }
  return entries;
}

function scheduleRR(procs: Process[], quantum: number): GanttEntry[] {
  const remaining = procs.map(p => ({ ...p, remaining: p.burst, started: false, firstRun: -1 }));
  const entries: GanttEntry[] = [];
  const queue: number[] = [];
  let time = 0;
  let idx = 0;
  const sorted = [...remaining].sort((a, b) => a.arrival - b.arrival);
  const inQueue = new Set<number>();

  const addArrivals = () => {
    while (idx < sorted.length && sorted[idx].arrival <= time) {
      const pi = remaining.findIndex(r => r.id === sorted[idx].id);
      if (pi >= 0 && !inQueue.has(pi) && remaining[pi].remaining > 0) {
        queue.push(pi);
        inQueue.add(pi);
      }
      idx++;
    }
  };

  addArrivals();
  if (queue.length === 0 && idx < sorted.length) {
    time = sorted[idx].arrival;
    addArrivals();
  }

  while (queue.length > 0 || idx < sorted.length) {
    if (queue.length === 0) {
      time = sorted[idx].arrival;
      addArrivals();
      continue;
    }
    const ci = queue.shift()!;
    inQueue.delete(ci);
    const p = remaining[ci];
    const exec = Math.min(quantum, p.remaining);
    entries.push({ pid: p.id, name: p.name, start: time, end: time + exec });
    p.remaining -= exec;
    time += exec;
    addArrivals();
    if (p.remaining > 0) {
      queue.push(ci);
      inQueue.add(ci);
    }
  }
  return entries;
}

function calcMetrics(procs: Process[], gantt: GanttEntry[]): ProcessMetrics[] {
  return procs.map(p => {
    const lastEnd = Math.max(...gantt.filter(g => g.pid === p.id).map(g => g.end));
    const firstStart = Math.min(...gantt.filter(g => g.pid === p.id).map(g => g.start));
    return {
      name: p.name,
      completion: lastEnd,
      turnaround: lastEnd - p.arrival,
      waiting: lastEnd - p.arrival - p.burst,
      response: firstStart - p.arrival,
    };
  });
}

export default function SchedulingMetricsCalculator() {
  const [processes, setProcesses] = useState<Process[]>([
    { id: 1, name: "P1", arrival: 0, burst: 5 },
    { id: 2, name: "P2", arrival: 1, burst: 3 },
    { id: 3, name: "P3", arrival: 2, burst: 8 },
  ]);
  const [algorithm, setAlgorithm] = useState<"FCFS" | "SJF" | "RR">("FCFS");
  const [quantum, setQuantum] = useState(2);
  const [nextId, setNextId] = useState(4);
  const [gantt, setGantt] = useState<GanttEntry[]>([]);
  const [metrics, setMetrics] = useState<ProcessMetrics[]>([]);
  const [animatedSteps, setAnimatedSteps] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const addProcess = () => {
    setProcesses(prev => [...prev, { id: nextId, name: `P${nextId}`, arrival: 0, burst: 5 }]);
    setNextId(n => n + 1);
  };

  const removeProcess = (id: number) => {
    setProcesses(prev => prev.filter(p => p.id !== id));
  };

  const updateProcess = (id: number, field: keyof Omit<Process, "id">, value: string) => {
    setProcesses(prev => prev.map(p => p.id === id ? { ...p, [field]: field === "name" ? value : Math.max(0, parseInt(value) || 0) } : p));
  };

  const runSchedule = useCallback(() => {
    let result: GanttEntry[];
    if (algorithm === "FCFS") result = scheduleFCFS(processes);
    else if (algorithm === "SJF") result = scheduleSJF(processes);
    else result = scheduleRR(processes, quantum);

    setGantt([]);
    setMetrics([]);
    setIsAnimating(true);

    let step = 0;
    const interval = setInterval(() => {
      step++;
      setAnimatedSteps(step);
      if (step >= result.length) {
        clearInterval(interval);
        setIsAnimating(false);
        setMetrics(calcMetrics(processes, result));
      }
    }, 400);

    setGantt(result);
    setAnimatedSteps(0);
  }, [processes, algorithm, quantum]);

  const totalTime = gantt.length > 0 ? Math.max(...gantt.map(g => g.end)) : 0;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Scheduling Metrics Calculator
      </h2>

      {/* Algorithm selector */}
      <div className="flex flex-wrap items-center gap-4 mb-6 justify-center">
        {(["FCFS", "SJF", "RR"] as const).map(algo => (
          <button
            key={algo}
            onClick={() => setAlgorithm(algo)}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              algorithm === algo
                ? "bg-indigo-500 text-white shadow-md"
                : "bg-white dark:bg-gray-700 text-slate-600 dark:text-gray-300 hover:bg-indigo-50 dark:hover:bg-gray-600"
            }`}
          >
            {algo}
          </button>
        ))}
        {algorithm === "RR" && (
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-slate-600 dark:text-gray-300">Quantum:</label>
            <input
              type="number"
              min={1}
              value={quantum}
              onChange={e => setQuantum(Math.max(1, parseInt(e.target.value) || 1))}
              className="w-16 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100 text-center"
            />
          </div>
        )}
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
                <motion.tr
                  key={p.id}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="border-b border-slate-200 dark:border-gray-700"
                >
                  <td className="px-3 py-2">
                    <input
                      value={p.name}
                      onChange={e => updateProcess(p.id, "name", e.target.value)}
                      className="w-16 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number"
                      min={0}
                      value={p.arrival}
                      onChange={e => updateProcess(p.id, "arrival", e.target.value)}
                      className="w-16 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100 text-center"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number"
                      min={1}
                      value={p.burst}
                      onChange={e => updateProcess(p.id, "burst", e.target.value)}
                      className="w-16 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100 text-center"
                    />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <button
                      onClick={() => removeProcess(p.id)}
                      className="p-1 text-red-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors"
                    >
                      <Trash2 size={16} />
                    </button>
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {/* Action buttons */}
      <div className="flex flex-wrap gap-3 mb-6 justify-center">
        <button onClick={addProcess} className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors shadow">
          <Plus size={16} /> Add Process
        </button>
        <button onClick={runSchedule} disabled={isAnimating || processes.length === 0} className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors shadow disabled:opacity-50">
          <Play size={16} /> Run {algorithm}
        </button>
        <button onClick={() => { setGantt([]); setMetrics([]); setAnimatedSteps(0); }} className="flex items-center gap-2 px-4 py-2 bg-slate-400 text-white rounded-lg hover:bg-slate-500 transition-colors shadow">
          <RotateCcw size={16} /> Reset
        </button>
      </div>

      {/* Gantt Chart */}
      {gantt.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-6">
          <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <BarChart3 size={18} /> Gantt Chart ({algorithm})
          </h3>
          <div className="relative bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 overflow-x-auto">
            <div className="flex items-stretch min-w-fit h-12">
              {gantt.slice(0, animatedSteps || (isAnimating ? 0 : gantt.length)).map((entry, i) => {
                const width = ((entry.end - entry.start) / totalTime) * 100;
                const color = PROCESS_COLORS[(entry.pid - 1) % PROCESS_COLORS.length];
                return (
                  <motion.div
                    key={i}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.3 }}
                    className={`${color} dark:opacity-80 flex items-center justify-center text-white text-xs font-bold border-r border-white/30`}
                    style={{ width: `${Math.max(width, 3)}%`, minWidth: 30 }}
                  >
                    {entry.name}
                  </motion.div>
                );
              })}
            </div>
            {/* Time axis */}
            <div className="flex min-w-fit mt-1">
              {gantt.slice(0, animatedSteps || (isAnimating ? 0 : gantt.length)).map((entry, i) => {
                const width = ((entry.end - entry.start) / totalTime) * 100;
                return (
                  <div key={i} className="flex justify-between text-xs text-slate-500 dark:text-gray-400" style={{ width: `${Math.max(width, 3)}%`, minWidth: 30 }}>
                    <span>{entry.start}</span>
                    {i === (animatedSteps || gantt.length) - 1 && <span>{entry.end}</span>}
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>
      )}

      {/* Metrics Table */}
      {metrics.length > 0 && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <Calculator size={18} /> Results
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-indigo-50 dark:bg-indigo-900/30">
                  <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">Process</th>
                  <th className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">Completion</th>
                  <th className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">Turnaround</th>
                  <th className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">Waiting</th>
                  <th className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">Response</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <motion.tr
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="hover:bg-slate-50 dark:hover:bg-gray-700/50"
                  >
                    <td className="px-3 py-2 font-medium text-slate-700 dark:text-gray-200 border border-slate-200 dark:border-gray-700">{m.name}</td>
                    <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{m.completion}</td>
                    <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{m.turnaround}</td>
                    <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{m.waiting}</td>
                    <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{m.response}</td>
                  </motion.tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="bg-indigo-50 dark:bg-indigo-900/30 font-semibold">
                  <td className="px-3 py-2 text-slate-700 dark:text-gray-200 border border-slate-200 dark:border-gray-700">Average</td>
                  <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">-</td>
                  <td className="px-3 py-2 text-center text-indigo-600 dark:text-indigo-400 border border-slate-200 dark:border-gray-700">
                    {(metrics.reduce((s, m) => s + m.turnaround, 0) / metrics.length).toFixed(2)}
                  </td>
                  <td className="px-3 py-2 text-center text-indigo-600 dark:text-indigo-400 border border-slate-200 dark:border-gray-700">
                    {(metrics.reduce((s, m) => s + m.waiting, 0) / metrics.length).toFixed(2)}
                  </td>
                  <td className="px-3 py-2 text-center text-indigo-600 dark:text-indigo-400 border border-slate-200 dark:border-gray-700">
                    {(metrics.reduce((s, m) => s + m.response, 0) / metrics.length).toFixed(2)}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  );
}
