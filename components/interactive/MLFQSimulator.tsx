"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Plus, Trash2, ArrowDown, Info, Zap } from "lucide-react";

interface Process {
  id: number;
  name: string;
  arrival: number;
  burst: number;
}

interface SimEvent {
  time: number;
  running: string | null;
  queues: [string[], string[], string[]];
  action: string;
}

interface GanttSlice {
  pid: number;
  name: string;
  start: number;
  end: number;
  level: number;
}

const LEVEL_COLORS = ["bg-blue-400", "bg-amber-400", "bg-rose-400"];
const LEVEL_QUANTA = [4, 8, 16];
const BOOST_INTERVAL = 100;

const MLFQ_RULES = [
  "Rule 1: Higher priority queue runs first",
  "Rule 2: Same-priority processes run RR",
  "Rule 3: New process enters highest priority queue",
  "Rule 4: Process uses full quantum -> demoted one level",
  "Rule 5: Process yields CPU before quantum expires -> stays at same level",
  "Rule 6: After BOOST_INTERVAL, all processes return to top queue (anti-starvation)",
];

function simulateMLFQ(procs: Process[]) {
  const n = procs.length;
  const remaining = procs.map(p => ({ ...p, remaining: p.burst, level: 0, quantumUsed: 0 }));
  const sorted = [...remaining].sort((a, b) => a.arrival - b.arrival);
  const queues: number[][] = [[], [], []];
  const inQueue = new Set<number>();
  const slices: GanttSlice[] = [];
  const events: SimEvent[] = [];
  let time = 0;
  let arrIdx = 0;
  let lastBoost = 0;

  const addArrivals = () => {
    while (arrIdx < sorted.length && sorted[arrIdx].arrival <= time) {
      const pi = remaining.findIndex(r => r.id === sorted[arrIdx].id);
      if (pi >= 0 && !inQueue.has(pi) && remaining[pi].remaining > 0) {
        queues[0].push(pi);
        inQueue.add(pi);
        remaining[pi].level = 0;
      }
      arrIdx++;
    }
  };

  const snapshot = (action: string, running: string | null) => {
    events.push({
      time,
      running,
      queues: [
        queues[0].map(i => remaining[i].name),
        queues[1].map(i => remaining[i].name),
        queues[2].map(i => remaining[i].name),
      ],
      action,
    });
  };

  addArrivals();

  while (inQueue.size > 0 || arrIdx < sorted.length) {
    // Priority boost
    if (time - lastBoost >= BOOST_INTERVAL) {
      for (let lvl = 1; lvl < 3; lvl++) {
        while (queues[lvl].length > 0) {
          const pi = queues[lvl].shift()!;
          remaining[pi].level = 0;
          remaining[pi].quantumUsed = 0;
          queues[0].push(pi);
        }
      }
      lastBoost = time;
      snapshot("Priority boost: all processes promoted to Q0", null);
    }

    // Find highest non-empty queue
    let targetLevel = -1;
    for (let lvl = 0; lvl < 3; lvl++) {
      if (queues[lvl].length > 0) { targetLevel = lvl; break; }
    }

    if (targetLevel === -1) {
      if (arrIdx < sorted.length) {
        time = sorted[arrIdx].arrival;
        addArrivals();
        continue;
      }
      break;
    }

    const ci = queues[targetLevel].shift()!;
    inQueue.delete(ci);
    const p = remaining[ci];
    const quantum = LEVEL_QUANTA[targetLevel];
    const exec = Math.min(quantum, p.remaining);

    snapshot(`${p.name} runs on Q${targetLevel} for ${exec}ms`, p.name);

    slices.push({ pid: p.id, name: p.name, start: time, end: time + exec, level: targetLevel });
    p.remaining -= exec;
    p.quantumUsed += exec;
    time += exec;

    // Check for boost mid-run
    if (time - lastBoost >= BOOST_INTERVAL) {
      // Demote or keep before boost
      if (p.remaining > 0) {
        if (exec >= quantum) {
          p.level = Math.min(2, targetLevel + 1);
        }
        queues[p.level].push(p.id);
        inQueue.add(p.id);
      }
      // Boost
      for (let lvl = 0; lvl < 3; lvl++) {
        const toBoost: number[] = [];
        for (const qi of queues[lvl]) {
          if (qi !== ci || p.remaining <= 0) {
            // already in queue
          }
        }
      }
      // Simpler: just boost everything
      for (let lvl = 1; lvl < 3; lvl++) {
        while (queues[lvl].length > 0) {
          const pi = queues[lvl].shift()!;
          remaining[pi].level = 0;
          remaining[pi].quantumUsed = 0;
          queues[0].push(pi);
        }
      }
      // Also boost the current process if still remaining
      if (p.remaining > 0) {
        p.level = 0;
        // Remove from current queue and re-add to Q0
        const inQ = queues[targetLevel].indexOf(p.id);
        if (inQ >= 0) queues[targetLevel].splice(inQ, 1);
        else {
          const inQ0 = queues[0].indexOf(p.id);
          if (inQ0 >= 0) queues[0].splice(inQ0, 1);
        }
        if (!queues[0].includes(p.id)) {
          queues[0].push(p.id);
          inQueue.add(p.id);
        }
      }
      lastBoost = time;
      snapshot("Priority boost!", null);
      addArrivals();
      continue;
    }

    addArrivals();

    if (p.remaining > 0) {
      if (exec >= quantum) {
        // Used full quantum, demote
        p.level = Math.min(2, targetLevel + 1);
      }
      // else stays at same level
      queues[p.level].push(p.id);
      inQueue.add(p.id);
    }
  }

  return { slices, events };
}

export default function MLFQSimulator() {
  const [processes, setProcesses] = useState<Process[]>([
    { id: 1, name: "P1", arrival: 0, burst: 20 },
    { id: 2, name: "P2", arrival: 0, burst: 12 },
    { id: 3, name: "P3", arrival: 5, burst: 8 },
  ]);
  const [nextId, setNextId] = useState(4);
  const [slices, setSlices] = useState<GanttSlice[]>([]);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const [visibleSlices, setVisibleSlices] = useState(0);
  const [eventIdx, setEventIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const totalTime = slices.length > 0 ? Math.max(...slices.map(s => s.end)) : 0;

  const addProcess = () => {
    setProcesses(prev => [...prev, { id: nextId, name: `P${nextId}`, arrival: 0, burst: 10 }]);
    setNextId(n => n + 1);
  };

  const removeProcess = (id: number) => setProcesses(prev => prev.filter(p => p.id !== id));

  const updateProcess = (id: number, field: keyof Omit<Process, "id">, value: string) => {
    setProcesses(prev => prev.map(p => p.id === id ? { ...p, [field]: field === "name" ? value : Math.max(0, parseInt(value) || 0) } : p));
  };

  const startSimulation = useCallback(() => {
    const result = simulateMLFQ(processes);
    setSlices(result.slices);
    setEvents(result.events);
    setVisibleSlices(0);
    setEventIdx(0);
    setPlaying(true);
  }, [processes]);

  const reset = () => {
    setPlaying(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    setSlices([]);
    setEvents([]);
    setVisibleSlices(0);
    setEventIdx(0);
  };

  useEffect(() => {
    if (playing && visibleSlices < slices.length) {
      timerRef.current = setTimeout(() => {
        setVisibleSlices(v => v + 1);
        setEventIdx(e => Math.min(e + 1, events.length - 1));
      }, 500);
      return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    }
    if (visibleSlices >= slices.length) setPlaying(false);
  }, [playing, visibleSlices, slices.length, events.length]);

  const currentEvent = events[eventIdx] || null;

  // Metrics
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

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        Multi-Level Feedback Queue Simulator
      </h2>

      {/* Rules toggle */}
      <div className="flex justify-center mb-4">
        <button onClick={() => setShowRules(!showRules)} className="flex items-center gap-2 text-sm text-purple-600 dark:text-purple-400 hover:underline">
          <Info size={14} /> {showRules ? "Hide" : "Show"} MLFQ Rules
        </button>
      </div>
      <AnimatePresence>
        {showRules && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mb-4 overflow-hidden"
          >
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
              <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-1">
                {MLFQ_RULES.map((rule, i) => <li key={i}>{rule}</li>)}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Queue level legend */}
      <div className="flex justify-center gap-6 mb-6">
        {LEVEL_COLORS.map((c, i) => (
          <div key={i} className="flex items-center gap-2 text-sm text-slate-600 dark:text-gray-300">
            <div className={`w-4 h-4 rounded ${c}`} />
            <span>Q{i} (quantum={LEVEL_QUANTA[i]}ms)</span>
          </div>
        ))}
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
        <button onClick={startSimulation} disabled={playing || processes.length === 0} className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors shadow disabled:opacity-50">
          <Play size={16} /> Simulate
        </button>
        <button onClick={() => setPlaying(p => !p)} disabled={slices.length === 0} className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors shadow disabled:opacity-50">
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
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 overflow-x-auto">
            <div className="flex items-stretch min-w-fit h-12">
              {slices.slice(0, visibleSlices).map((s, i) => {
                const w = ((s.end - s.start) / Math.max(totalTime, 1)) * 100;
                return (
                  <motion.div
                    key={i}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.2 }}
                    className={`${LEVEL_COLORS[s.level]} dark:opacity-80 flex items-center justify-center text-white text-xs font-bold border-r border-white/30`}
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

      {/* Queue State */}
      {currentEvent && (
        <motion.div
          key={eventIdx}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700"
        >
          <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <Zap size={18} className="text-purple-500" /> Queue State at t={currentEvent.time}
          </h3>
          <div className="mb-2 text-sm text-purple-600 dark:text-purple-400 font-medium">{currentEvent.action}</div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {currentEvent.queues.map((q, lvl) => (
              <div key={lvl} className={`rounded-lg p-3 ${lvl === 0 ? "bg-blue-50 dark:bg-blue-900/20" : lvl === 1 ? "bg-amber-50 dark:bg-amber-900/20" : "bg-rose-50 dark:bg-rose-900/20"}`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-3 h-3 rounded ${LEVEL_COLORS[lvl]}`} />
                  <span className="text-sm font-semibold text-slate-700 dark:text-gray-200">Queue {lvl} (q={LEVEL_QUANTA[lvl]})</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {q.length === 0 ? (
                    <span className="text-xs text-slate-400 italic">empty</span>
                  ) : (
                    q.map((name, i) => (
                      <span key={i} className="px-2 py-0.5 bg-white dark:bg-gray-700 text-slate-700 dark:text-gray-200 rounded text-xs font-mono border border-slate-200 dark:border-gray-600">
                        {name}
                      </span>
                    ))
                  )}
                </div>
              </div>
            ))}
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
                <tr className="bg-purple-50 dark:bg-purple-900/30">
                  {["Process", "Completion", "Turnaround", "Waiting", "Response"].map(h => (
                    <th key={h} className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <motion.tr key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.1 }} className="hover:bg-slate-50 dark:hover:bg-gray-700/50">
                    <td className="px-3 py-2 font-medium text-slate-700 dark:text-gray-200 border border-slate-200 dark:border-gray-700 text-center">{m.name}</td>
                    <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-slate-600 dark:text-gray-300">{m.completion}</td>
                    <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-slate-600 dark:text-gray-300">{m.turnaround}</td>
                    <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-slate-600 dark:text-gray-300">{m.waiting}</td>
                    <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-slate-600 dark:text-gray-300">{m.response}</td>
                  </motion.tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="bg-purple-50 dark:bg-purple-900/30 font-semibold">
                  <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-slate-700 dark:text-gray-200">Average</td>
                  <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-slate-400">-</td>
                  <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-purple-600 dark:text-purple-400">{(metrics.reduce((s, m) => s + m.turnaround, 0) / metrics.length).toFixed(2)}</td>
                  <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-purple-600 dark:text-purple-400">{(metrics.reduce((s, m) => s + m.waiting, 0) / metrics.length).toFixed(2)}</td>
                  <td className="px-3 py-2 text-center border border-slate-200 dark:border-gray-700 text-purple-600 dark:text-purple-400">{(metrics.reduce((s, m) => s + m.response, 0) / metrics.length).toFixed(2)}</td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
