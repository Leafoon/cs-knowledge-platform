"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Plus, Trash2, BarChart3, Trophy } from "lucide-react";

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

const COLORS = [
  "bg-blue-400", "bg-emerald-400", "bg-amber-400", "bg-rose-400",
  "bg-violet-400", "bg-cyan-400", "bg-pink-400", "bg-lime-400",
];

function scheduleFCFS(procs: Process[]): GanttEntry[] {
  const sorted = [...procs].sort((a, b) => a.arrival - b.arrival || a.id - b.id);
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
  const done = new Set<number>();

  while (done.size < procs.length) {
    const available = remaining.filter(p => !done.has(p.id) && p.arrival <= time);
    if (available.length === 0) {
      time = Math.min(...remaining.filter(p => !done.has(p.id)).map(p => p.arrival));
      continue;
    }
    available.sort((a, b) => a.remaining - b.remaining || a.id - b.id);
    const chosen = available[0];
    entries.push({ pid: chosen.id, name: chosen.name, start: time, end: time + chosen.remaining });
    time += chosen.remaining;
    done.add(chosen.id);
  }
  return entries;
}

function scheduleSRTF(procs: Process[]): GanttEntry[] {
  const remaining = procs.map(p => ({ ...p, remaining: p.burst }));
  const entries: GanttEntry[] = [];
  let time = 0;
  const maxTime = Math.max(...procs.map(p => p.arrival)) + procs.reduce((s, p) => s + p.burst, 0) + 1;
  let current: number | null = null;
  let segStart = 0;

  const finish = () => {
    if (current !== null) {
      const p = remaining[current];
      entries.push({ pid: p.id, name: p.name, start: segStart, end: time });
      current = null;
    }
  };

  for (time = 0; time <= maxTime; time++) {
    const available = remaining.filter(p => p.remaining > 0 && p.arrival <= time);
    if (available.length === 0 && remaining.every(p => p.remaining <= 0)) break;

    if (available.length === 0) {
      finish();
      continue;
    }

    available.sort((a, b) => a.remaining - b.remaining || a.arrival - b.arrival);
    const best = remaining.findIndex(r => r.id === available[0].id);

    if (current !== best) {
      finish();
      current = best;
      segStart = time;
    }

    remaining[best].remaining--;
    if (remaining[best].remaining === 0) {
      finish();
    }
  }
  finish();
  return entries;
}

function scheduleRR(procs: Process[], quantum: number): GanttEntry[] {
  const remaining = procs.map(p => ({ ...p, remaining: p.burst }));
  const entries: GanttEntry[] = [];
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

function calcAvgMetrics(procs: Process[], gantt: GanttEntry[]) {
  const metrics = procs.map(p => {
    const pGantt = gantt.filter(g => g.pid === p.id);
    if (pGantt.length === 0) return { turnaround: 0, waiting: 0, response: 0 };
    const completion = Math.max(...pGantt.map(g => g.end));
    const firstStart = Math.min(...pGantt.map(g => g.start));
    return {
      turnaround: completion - p.arrival,
      waiting: completion - p.arrival - p.burst,
      response: firstStart - p.arrival,
    };
  });
  const n = metrics.length;
  return {
    turnaround: metrics.reduce((s, m) => s + m.turnaround, 0) / n,
    waiting: metrics.reduce((s, m) => s + m.waiting, 0) / n,
    response: metrics.reduce((s, m) => s + m.response, 0) / n,
  };
}

type AlgorithmName = "FCFS" | "SJF" | "SRTF" | "RR";

const ALGO_COLORS: Record<AlgorithmName, string> = {
  FCFS: "bg-blue-500",
  SJF: "bg-emerald-500",
  SRTF: "bg-amber-500",
  RR: "bg-rose-500",
};

export default function SchedulingAlgorithmComparison() {
  const [processes, setProcesses] = useState<Process[]>([
    { id: 1, name: "P1", arrival: 0, burst: 8 },
    { id: 2, name: "P2", arrival: 1, burst: 4 },
    { id: 3, name: "P3", arrival: 2, burst: 9 },
    { id: 4, name: "P4", arrival: 3, burst: 5 },
  ]);
  const [nextId, setNextId] = useState(5);
  const [quantum, setQuantum] = useState(3);

  const addProcess = () => {
    setProcesses(prev => [...prev, { id: nextId, name: `P${nextId}`, arrival: 0, burst: 5 }]);
    setNextId(n => n + 1);
  };

  const removeProcess = (id: number) => setProcesses(prev => prev.filter(p => p.id !== id));

  const updateProcess = (id: number, field: keyof Omit<Process, "id">, value: string) => {
    setProcesses(prev => prev.map(p => p.id === id ? { ...p, [field]: field === "name" ? value : Math.max(0, parseInt(value) || 0) } : p));
  };

  const results = useMemo(() => {
    const algos: { name: AlgorithmName; schedule: () => GanttEntry[] }[] = [
      { name: "FCFS", schedule: () => scheduleFCFS(processes) },
      { name: "SJF", schedule: () => scheduleSJF(processes) },
      { name: "SRTF", schedule: () => scheduleSRTF(processes) },
      { name: "RR", schedule: () => scheduleRR(processes, quantum) },
    ];

    return algos.map(a => {
      const gantt = a.schedule();
      const totalTime = gantt.length > 0 ? Math.max(...gantt.map(g => g.end)) : 0;
      const avg = calcAvgMetrics(processes, gantt);
      return { name: a.name, gantt, totalTime, avg };
    });
  }, [processes, quantum]);

  const maxTotalTime = Math.max(...results.map(r => r.totalTime), 1);
  const maxTurnaround = Math.max(...results.map(r => r.avg.turnaround), 1);
  const maxWaiting = Math.max(...results.map(r => r.avg.waiting), 1);
  const maxResponse = Math.max(...results.map(r => r.avg.response), 1);

  const findBest = (metric: "turnaround" | "waiting" | "response") => {
    let best = results[0];
    for (const r of results) {
      if (r.avg[metric] < best.avg[metric]) best = r;
    }
    return best.name;
  };

  const bestTA = findBest("turnaround");
  const bestWT = findBest("waiting");
  const bestRT = findBest("response");

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Scheduling Algorithm Comparison
      </h2>

      {/* Quantum for RR */}
      <div className="flex items-center gap-4 mb-6 justify-center">
        <label className="text-sm font-medium text-slate-600 dark:text-gray-300">RR Quantum: {quantum}ms</label>
        <input type="range" min={1} max={20} value={quantum} onChange={e => setQuantum(parseInt(e.target.value))} className="w-48 accent-orange-500" />
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
            {processes.map(p => (
              <tr key={p.id} className="border-b border-slate-200 dark:border-gray-700">
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
              </tr>
            ))}
          </tbody>
        </table>
        <button onClick={addProcess} className="mt-3 flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors shadow text-sm">
          <Plus size={16} /> Add Process
        </button>
      </div>

      {/* Gantt charts side by side */}
      <div className="mb-8 space-y-4">
        <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 flex items-center gap-2">
          <BarChart3 size={18} /> Gantt Charts
        </h3>
        {results.map((r, ri) => (
          <motion.div
            key={r.name}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: ri * 0.1 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-slate-200 dark:border-gray-700"
          >
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-3 h-3 rounded ${ALGO_COLORS[r.name]}`} />
              <span className="font-semibold text-sm text-slate-700 dark:text-gray-200">{r.name}{r.name === "RR" ? ` (q=${quantum})` : ""}</span>
            </div>
            <div className="flex items-stretch h-8">
              {r.gantt.map((g, i) => {
                const w = ((g.end - g.start) / Math.max(maxTotalTime, 1)) * 100;
                return (
                  <div
                    key={i}
                    className={`${COLORS[(g.pid - 1) % COLORS.length]} dark:opacity-80 flex items-center justify-center text-white text-xs font-bold border-r border-white/30`}
                    style={{ width: `${Math.max(w, 2)}%`, minWidth: 20 }}
                  >
                    {g.name}
                  </div>
                );
              })}
            </div>
            <div className="flex mt-0.5">
              {r.gantt.map((g, i) => {
                const w = ((g.end - g.start) / Math.max(maxTotalTime, 1)) * 100;
                return (
                  <div key={i} className="flex justify-between text-[10px] text-slate-400 dark:text-gray-500" style={{ width: `${Math.max(w, 2)}%`, minWidth: 20 }}>
                    <span>{g.start}</span>
                    {i === r.gantt.length - 1 && <span>{g.end}</span>}
                  </div>
                );
              })}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Metric bar charts */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Trophy size={18} className="text-amber-500" /> Average Metrics Comparison
        </h3>
        <div className="space-y-6">
          {([
            { key: "turnaround" as const, label: "Avg Turnaround Time", max: maxTurnaround, best: bestTA },
            { key: "waiting" as const, label: "Avg Waiting Time", max: maxWaiting, best: bestWT },
            { key: "response" as const, label: "Avg Response Time", max: maxResponse, best: bestRT },
          ]).map(metric => (
            <div key={metric.key}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-slate-600 dark:text-gray-300">{metric.label}</span>
                <span className="text-xs px-2 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-full font-medium">
                  Best: {metric.best}
                </span>
              </div>
              <div className="space-y-1.5">
                {results.map(r => (
                  <div key={r.name} className="flex items-center gap-3">
                    <span className="w-12 text-xs font-medium text-slate-600 dark:text-gray-300 text-right">{r.name}</span>
                    <div className="flex-1 bg-slate-100 dark:bg-gray-700 rounded-full h-6 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(r.avg[metric.key] / Math.max(metric.max, 1)) * 100}%` }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                        className={`h-full ${ALGO_COLORS[r.name]} dark:opacity-80 rounded-full flex items-center justify-end pr-2`}
                      >
                        <span className="text-xs text-white font-bold">{r.avg[metric.key].toFixed(1)}</span>
                      </motion.div>
                    </div>
                    {r.name === metric.best && (
                      <Trophy size={14} className="text-amber-500" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
