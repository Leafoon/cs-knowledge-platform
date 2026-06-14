"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { CheckCircle, XCircle, AlertTriangle, Plus, Trash2, BarChart3 } from "lucide-react";

interface Task {
  id: number;
  name: string;
  period: number;
  execution: number;
  deadline: number;
}

interface AnalysisResult {
  utilization: number;
  rmsBound: number;
  rmsSchedulable: boolean | null;
  edfSchedulable: boolean;
  rtaResults: { id: number; name: string; responseTime: number; deadline: number; ok: boolean }[];
}

const defaultTasks: Task[] = [
  { id: 1, name: "τ₁", period: 5, execution: 1, deadline: 5 },
  { id: 2, name: "τ₂", period: 8, execution: 2, deadline: 8 },
  { id: 3, name: "τ₃", period: 10, execution: 2, deadline: 10 },
];

function liuLaybound(n: number): number {
  if (n <= 0) return 0;
  return n * (Math.pow(2, 1 / n) - 1);
}

function computeRTA(tasks: Task[]): AnalysisResult {
  const sorted = [...tasks].sort((a, b) => a.period - b.period);
  const utilization = sorted.reduce((s, t) => s + t.execution / t.period, 0);
  const rmsBound = liuLaybound(sorted.length);
  const edfSchedulable = utilization <= 1.0;

  const rtaResults: AnalysisResult["rtaResults"] = [];

  for (let i = 0; i < sorted.length; i++) {
    const t = sorted[i];
    let r = t.execution;
    let prev = 0;
    for (let iter = 0; iter < 100; iter++) {
      prev = r;
      let interference = 0;
      for (let j = 0; j < i; j++) {
        interference += Math.ceil(r / sorted[j].period) * sorted[j].execution;
      }
      r = t.execution + interference;
      if (r === prev) break;
      if (r > t.deadline) break;
    }
    rtaResults.push({
      id: t.id,
      name: t.name,
      responseTime: r,
      deadline: t.deadline,
      ok: r <= t.deadline,
    });
  }

  const allRtaOk = rtaResults.every((r) => r.ok);

  return {
    utilization,
    rmsBound,
    rmsSchedulable: utilization <= rmsBound ? true : allRtaOk ? true : false,
    edfSchedulable,
    rtaResults,
  };
}

export default function SchedulabilityAnalyzer() {
  const [tasks, setTasks] = useState<Task[]>(defaultTasks);
  const [newPeriod, setNewPeriod] = useState(6);
  const [newExec, setNewExec] = useState(1);
  const [newDeadline, setNewDeadline] = useState(6);

  const result = useMemo(() => computeRTA(tasks), [tasks]);

  const addTask = () => {
    const id = Math.max(0, ...tasks.map((t) => t.id)) + 1;
    setTasks([
      ...tasks,
      { id, name: `τ${id}`, period: newPeriod, execution: newExec, deadline: newDeadline },
    ]);
  };

  const removeTask = (id: number) => {
    setTasks(tasks.filter((t) => t.id !== id));
  };

  const utilizationPct = Math.min(result.utilization * 100, 100);
  const boundPct = result.rmsBound * 100;

  return (
    <div className="w-full space-y-6 p-4 bg-white dark:bg-gray-900 rounded-xl">
      <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-purple-500" />
        可调度性分析器
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400">任务集配置</h4>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 dark:text-gray-400 border-b dark:border-gray-700">
                  <th className="pb-2 pr-3">任务</th>
                  <th className="pb-2 pr-3">执行时间 C</th>
                  <th className="pb-2 pr-3">周期 T</th>
                  <th className="pb-2 pr-3">截止时间 D</th>
                  <th className="pb-2 pr-3">利用率</th>
                  <th className="pb-2"></th>
                </tr>
              </thead>
              <tbody>
                {tasks.map((t) => (
                  <tr key={t.id} className="border-b dark:border-gray-800">
                    <td className="py-2 pr-3 font-mono font-bold text-gray-800 dark:text-gray-200">
                      {t.name}
                    </td>
                    <td className="py-2 pr-3 font-mono text-gray-700 dark:text-gray-300">
                      {t.execution}
                    </td>
                    <td className="py-2 pr-3 font-mono text-gray-700 dark:text-gray-300">
                      {t.period}
                    </td>
                    <td className="py-2 pr-3 font-mono text-gray-700 dark:text-gray-300">
                      {t.deadline}
                    </td>
                    <td className="py-2 pr-3 font-mono text-gray-500">
                      {(t.execution / t.period).toFixed(3)}
                    </td>
                    <td className="py-2">
                      <button
                        onClick={() => removeTask(t.id)}
                        className="text-red-400 hover:text-red-600"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap gap-2 items-end">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">C</label>
              <input
                type="number"
                min={1}
                value={newExec}
                onChange={(e) => setNewExec(Number(e.target.value))}
                className="w-14 px-2 py-1 rounded border dark:bg-gray-800 dark:border-gray-600 text-sm"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">T</label>
              <input
                type="number"
                min={1}
                value={newPeriod}
                onChange={(e) => setNewPeriod(Number(e.target.value))}
                className="w-14 px-2 py-1 rounded border dark:bg-gray-800 dark:border-gray-600 text-sm"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">D</label>
              <input
                type="number"
                min={1}
                value={newDeadline}
                onChange={(e) => setNewDeadline(Number(e.target.value))}
                className="w-14 px-2 py-1 rounded border dark:bg-gray-800 dark:border-gray-600 text-sm"
              />
            </div>
            <button
              onClick={addTask}
              className="px-3 py-1.5 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors flex items-center gap-1 text-sm"
            >
              <Plus className="w-4 h-4" /> 添加任务
            </button>
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400">分析结果</h4>

          <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">总 CPU 利用率 U</span>
              <span className="text-lg font-mono font-bold text-gray-800 dark:text-gray-200">
                {(result.utilization * 100).toFixed(1)}%
              </span>
            </div>

            <div className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className="absolute h-full bg-blue-500 rounded-full"
                animate={{ width: `${utilizationPct}%` }}
                transition={{ duration: 0.5 }}
              />
              <motion.div
                className="absolute h-full border-r-2 border-dashed border-amber-500"
                animate={{ left: `${boundPct}%` }}
                transition={{ duration: 0.5 }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">
                {(result.utilization * 100).toFixed(1)}%
              </div>
            </div>

            <div className="flex justify-between text-xs text-gray-500">
              <span>0%</span>
              <span className="text-amber-500">RMS 上界 {boundPct.toFixed(1)}%</span>
              <span>100%</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 flex items-center gap-2">
              {result.edfSchedulable ? (
                <CheckCircle className="w-5 h-5 text-emerald-500 shrink-0" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500 shrink-0" />
              )}
              <div>
                <div className="text-xs text-gray-500">EDF</div>
                <div className="text-sm font-bold text-gray-800 dark:text-gray-200">
                  {result.edfSchedulable ? "可调度" : "不可调度"}
                </div>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 flex items-center gap-2">
              {result.rmsSchedulable === true ? (
                <CheckCircle className="w-5 h-5 text-emerald-500 shrink-0" />
              ) : result.rmsSchedulable === false ? (
                <XCircle className="w-5 h-5 text-red-500 shrink-0" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0" />
              )}
              <div>
                <div className="text-xs text-gray-500">RMS</div>
                <div className="text-sm font-bold text-gray-800 dark:text-gray-200">
                  {result.rmsSchedulable === true
                    ? "可调度"
                    : result.rmsSchedulable === false
                    ? "不可调度"
                    : "需进一步分析"}
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <h5 className="text-xs font-semibold text-gray-600 dark:text-gray-400">
              响应时间分析 (RTA)
            </h5>
            {result.rtaResults.map((r) => (
              <div
                key={r.id}
                className="flex items-center gap-3 p-2 rounded-lg bg-gray-50 dark:bg-gray-800"
              >
                <span className="font-mono font-bold text-sm text-gray-700 dark:text-gray-300 w-8">
                  {r.name}
                </span>
                <div className="flex-1 h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${r.ok ? "bg-emerald-500" : "bg-red-500"}`}
                    animate={{ width: `${Math.min((r.responseTime / r.deadline) * 100, 100)}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-xs font-mono text-gray-500 w-20 text-right">
                  R={r.responseTime} ≤ D={r.deadline}
                </span>
                {r.ok ? (
                  <CheckCircle className="w-4 h-4 text-emerald-500 shrink-0" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-500 shrink-0" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
