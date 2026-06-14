"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Plus, Trash2, CheckCircle2, XCircle, Activity, ShieldCheck } from "lucide-react";

interface Task {
  id: number;
  name: string;
  period: number;
  execution: number;
  deadline: number;
}

function rmsBound(n: number): number {
  if (n <= 0) return 0;
  return n * (Math.pow(2, 1 / n) - 1);
}

function edfBound(): number {
  return 1.0;
}

export default function RealTimeSchedulabilityAnalyzer() {
  const [tasks, setTasks] = useState<Task[]>([
    { id: 1, name: "T1", period: 10, execution: 2, deadline: 10 },
    { id: 2, name: "T2", period: 15, execution: 4, deadline: 15 },
    { id: 3, name: "T3", period: 30, execution: 5, deadline: 30 },
  ]);
  const [nextId, setNextId] = useState(4);

  const addTask = () => {
    setTasks(prev => [...prev, { id: nextId, name: `T${nextId}`, period: 20, execution: 3, deadline: 20 }]);
    setNextId(n => n + 1);
  };

  const removeTask = (id: number) => setTasks(prev => prev.filter(t => t.id !== id));

  const updateTask = (id: number, field: keyof Omit<Task, "id">, value: string) => {
    setTasks(prev => prev.map(t => t.id === id ? { ...t, [field]: field === "name" ? value : Math.max(1, parseInt(value) || 1) } : t));
  };

  const analysis = useMemo(() => {
    const n = tasks.length;
    const utilization = tasks.reduce((sum, t) => sum + t.execution / t.period, 0);
    const rms = rmsBound(n);
    const edf = edfBound();
    const rmsSchedulable = utilization <= rms;
    const edfSchedulable = utilization <= edf;

    // Liu-Layland curve data points for visualization
    const curvePoints: { n: number; bound: number }[] = [];
    for (let i = 1; i <= 15; i++) {
      curvePoints.push({ n: i, bound: rmsBound(i) });
    }

    return { n, utilization, rms, edf, rmsSchedulable, edfSchedulable, curvePoints };
  }, [tasks]);

  const barMax = 1.2;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Real-Time Schedulability Analyzer
      </h2>

      {/* Task table */}
      <div className="mb-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-100 dark:bg-gray-700">
              <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Task</th>
              <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Period (T)</th>
              <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Exec (C)</th>
              <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Deadline (D)</th>
              <th className="px-3 py-2 text-center text-slate-600 dark:text-gray-300">Ci/Ti</th>
              <th className="px-3 py-2 text-center text-slate-600 dark:text-gray-300">Action</th>
            </tr>
          </thead>
          <tbody>
            {tasks.map(t => (
              <motion.tr key={t.id} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="border-b border-slate-200 dark:border-gray-700">
                <td className="px-3 py-2">
                  <input value={t.name} onChange={e => updateTask(t.id, "name", e.target.value)} className="w-16 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100" />
                </td>
                <td className="px-3 py-2">
                  <input type="number" min={1} value={t.period} onChange={e => updateTask(t.id, "period", e.target.value)} className="w-20 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100 text-center" />
                </td>
                <td className="px-3 py-2">
                  <input type="number" min={1} value={t.execution} onChange={e => updateTask(t.id, "execution", e.target.value)} className="w-20 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100 text-center" />
                </td>
                <td className="px-3 py-2">
                  <input type="number" min={1} value={t.deadline} onChange={e => updateTask(t.id, "deadline", e.target.value)} className="w-20 px-2 py-1 rounded border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-800 dark:text-gray-100 text-center" />
                </td>
                <td className="px-3 py-2 text-center text-slate-600 dark:text-gray-300 font-mono">
                  {(t.execution / t.period).toFixed(3)}
                </td>
                <td className="px-3 py-2 text-center">
                  <button onClick={() => removeTask(t.id)} className="p-1 text-red-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors">
                    <Trash2 size={16} />
                  </button>
                </td>
              </motion.tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="bg-red-50 dark:bg-red-900/20 font-semibold">
              <td className="px-3 py-2 text-slate-700 dark:text-gray-200" colSpan={4}>Total CPU Utilization U</td>
              <td className="px-3 py-2 text-center text-red-600 dark:text-red-400 font-mono">{analysis.utilization.toFixed(4)}</td>
              <td />
            </tr>
          </tfoot>
        </table>
        <button onClick={addTask} className="mt-3 flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors shadow text-sm">
          <Plus size={16} /> Add Task
        </button>
      </div>

      {/* Schedulability results */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
        {/* RMS */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className={`rounded-lg p-5 border-2 ${analysis.rmsSchedulable ? "border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/20" : "border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/20"}`}
        >
          <div className="flex items-center gap-3 mb-3">
            {analysis.rmsSchedulable ? (
              <CheckCircle2 size={24} className="text-emerald-500" />
            ) : (
              <XCircle size={24} className="text-red-500" />
            )}
            <div>
              <h3 className="font-bold text-lg text-slate-800 dark:text-gray-100">Rate Monotonic (RMS)</h3>
              <p className="text-sm text-slate-500 dark:text-gray-400">Fixed-priority, preemptive</p>
            </div>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-gray-300">Bound: n(2^(1/n) - 1)</span>
              <span className="font-mono font-bold text-slate-800 dark:text-gray-100">{analysis.rms.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-gray-300">Utilization U</span>
              <span className={`font-mono font-bold ${analysis.rmsSchedulable ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}>
                {analysis.utilization.toFixed(4)}
              </span>
            </div>
            <div className="text-center mt-2">
              <span className={`text-lg font-bold ${analysis.rmsSchedulable ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}>
                {analysis.rmsSchedulable ? "SCHEDULABLE" : "NOT GUARANTEED"}
              </span>
            </div>
            <p className="text-xs text-slate-400 dark:text-gray-500 text-center mt-1">
              {analysis.rmsSchedulable
                ? `U = ${analysis.utilization.toFixed(4)} <= ${analysis.rms.toFixed(4)} (sufficient condition met)`
                : `U = ${analysis.utilization.toFixed(4)} > ${analysis.rms.toFixed(4)} (sufficient condition NOT met, but may still be schedulable)`}
            </p>
          </div>
        </motion.div>

        {/* EDF */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className={`rounded-lg p-5 border-2 ${analysis.edfSchedulable ? "border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/20" : "border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/20"}`}
        >
          <div className="flex items-center gap-3 mb-3">
            {analysis.edfSchedulable ? (
              <CheckCircle2 size={24} className="text-emerald-500" />
            ) : (
              <XCircle size={24} className="text-red-500" />
            )}
            <div>
              <h3 className="font-bold text-lg text-slate-800 dark:text-gray-100">Earliest Deadline First (EDF)</h3>
              <p className="text-sm text-slate-500 dark:text-gray-400">Dynamic-priority, optimal</p>
            </div>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-gray-300">Bound: 1.0</span>
              <span className="font-mono font-bold text-slate-800 dark:text-gray-100">1.0000</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-gray-300">Utilization U</span>
              <span className={`font-mono font-bold ${analysis.edfSchedulable ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}>
                {analysis.utilization.toFixed(4)}
              </span>
            </div>
            <div className="text-center mt-2">
              <span className={`text-lg font-bold ${analysis.edfSchedulable ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}>
                {analysis.edfSchedulable ? "SCHEDULABLE" : "NOT SCHEDULABLE"}
              </span>
            </div>
            <p className="text-xs text-slate-400 dark:text-gray-500 text-center mt-1">
              {analysis.edfSchedulable
                ? `U = ${analysis.utilization.toFixed(4)} <= 1.0 (necessary and sufficient condition met)`
                : `U = ${analysis.utilization.toFixed(4)} > 1.0 (overloaded, no algorithm can schedule)`}
            </p>
          </div>
        </motion.div>
      </div>

      {/* Utilization bar chart */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Activity size={18} /> Utilization vs. Bounds
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <div className="relative h-48">
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1.0].map(v => (
              <div key={v} className="absolute left-0 right-0 border-t border-dashed border-slate-200 dark:border-gray-600" style={{ bottom: `${(v / barMax) * 100}%` }}>
                <span className="absolute -left-8 -top-2 text-[10px] text-slate-400">{v.toFixed(2)}</span>
              </div>
            ))}

            {/* Bars */}
            <div className="absolute bottom-0 left-8 right-0 flex items-end justify-around h-full gap-4 px-4">
              {/* U bar */}
              <div className="flex flex-col items-center flex-1">
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: `${(analysis.utilization / barMax) * 100}%` }}
                  transition={{ duration: 0.6 }}
                  className={`w-full max-w-20 rounded-t ${analysis.utilization > 1 ? "bg-red-500" : analysis.utilization > analysis.rms ? "bg-amber-500" : "bg-emerald-500"} dark:opacity-80`}
                />
                <span className="text-xs text-slate-600 dark:text-gray-300 mt-1 text-center">U = {analysis.utilization.toFixed(3)}</span>
              </div>

              {/* RMS bound bar */}
              <div className="flex flex-col items-center flex-1">
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: `${(analysis.rms / barMax) * 100}%` }}
                  transition={{ duration: 0.6, delay: 0.1 }}
                  className="w-full max-w-20 rounded-t bg-blue-400 dark:opacity-80"
                />
                <span className="text-xs text-slate-600 dark:text-gray-300 mt-1 text-center">RMS = {analysis.rms.toFixed(3)}</span>
              </div>

              {/* EDF bound bar */}
              <div className="flex flex-col items-center flex-1">
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: `${(1.0 / barMax) * 100}%` }}
                  transition={{ duration: 0.6, delay: 0.2 }}
                  className="w-full max-w-20 rounded-t bg-violet-400 dark:opacity-80"
                />
                <span className="text-xs text-slate-600 dark:text-gray-300 mt-1 text-center">EDF = 1.000</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Liu-Layland bound curve */}
      <div>
        <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200 mb-4 flex items-center gap-2">
          <ShieldCheck size={18} /> Liu-Layland RMS Bound Curve: n(2^(1/n) - 1)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <div className="relative h-48">
            {/* Y axis labels */}
            {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map(v => (
              <div key={v} className="absolute left-0 right-0 border-t border-dashed border-slate-200 dark:border-gray-600" style={{ bottom: `${v * 100}%` }}>
                <span className="absolute -left-10 -top-2 text-[10px] text-slate-400">{v.toFixed(1)}</span>
              </div>
            ))}

            {/* Asymptotic line at ln(2) ~ 0.693 */}
            <div className="absolute left-8 right-0 border-t-2 border-dashed border-amber-400/50" style={{ bottom: `${0.693 * 100}%` }}>
              <span className="absolute right-0 -top-4 text-[10px] text-amber-500">ln(2) = 0.693</span>
            </div>

            {/* Current task count marker */}
            {analysis.n <= 15 && (
              <div className="absolute bottom-0 left-8" style={{ left: `${8 + ((analysis.n - 1) / 14) * 90}%` }}>
                <div className="absolute bottom-full mb-1 -translate-x-1/2">
                  <div className="px-1.5 py-0.5 bg-red-100 dark:bg-red-900/40 text-red-600 dark:text-red-400 rounded text-[10px] font-bold whitespace-nowrap">
                    n={analysis.n}
                  </div>
                </div>
                <div className="w-2 h-2 bg-red-500 rounded-full -translate-x-1/2" style={{ bottom: `${(analysis.rms / 0.7) * 100}%` }} />
              </div>
            )}

            {/* Curve points */}
            <div className="absolute bottom-0 left-8 right-0 h-full flex items-end">
              {analysis.curvePoints.map((pt, i) => {
                const x = (i / (analysis.curvePoints.length - 1)) * 100;
                const y = (pt.bound / 0.7) * 100;
                const nextPt = analysis.curvePoints[i + 1];
                const nextX = nextPt ? ((i + 1) / (analysis.curvePoints.length - 1)) * 100 : x;
                const nextY = nextPt ? (nextPt.bound / 0.7) * 100 : y;

                return (
                  <div
                    key={i}
                    className="absolute"
                    style={{ left: `${x}%`, bottom: `${y}%` }}
                  >
                    <div className="w-2 h-2 bg-blue-500 rounded-full -translate-x-1 -translate-y-1" />
                    {i < 5 && (
                      <span className="absolute -bottom-5 -translate-x-1/2 text-[9px] text-slate-400">
                        {pt.bound.toFixed(2)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* X axis labels */}
            <div className="absolute bottom-0 left-8 right-0 flex justify-between">
              {analysis.curvePoints.filter((_, i) => i % 2 === 0 || i < 5).map(pt => (
                <span key={pt.n} className="text-[10px] text-slate-400">{pt.n}</span>
              ))}
            </div>
          </div>
          <p className="text-xs text-slate-400 dark:text-gray-500 text-center mt-2">
            The RMS bound approaches ln(2) ~ 69.3% as n increases. EDF achieves 100% utilization.
          </p>
        </div>
      </div>
    </div>
  );
}
