"use client";

import { useState, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Plus, Trash2, Timer, Zap } from "lucide-react";

interface Task {
  id: number;
  name: string;
  period: number;
  execution: number;
  deadline: number;
  color: string;
}

interface ScheduleEntry {
  time: number;
  taskId: number | null;
  taskName: string;
  color: string;
}

const TASK_COLORS = [
  "#3b82f6", "#ef4444", "#10b981", "#f59e0b",
  "#8b5cf6", "#ec4899", "#06b6d4", "#f97316",
];

type Algorithm = "rms" | "edf";

const defaultTasks: Task[] = [
  { id: 1, name: "τ₁", period: 4, execution: 1, deadline: 4, color: TASK_COLORS[0] },
  { id: 2, name: "τ₂", period: 6, execution: 2, deadline: 6, color: TASK_COLORS[1] },
  { id: 3, name: "τ₃", period: 12, execution: 3, deadline: 12, color: TASK_COLORS[2] },
];

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

function lcm(a: number, b: number): number {
  return (a * b) / gcd(a, b);
}

function computeRMS(tasks: Task[]): ScheduleEntry[] {
  const sorted = [...tasks].sort((a, b) => a.period - b.period);
  const hyperPeriod = sorted.reduce((acc, t) => lcm(acc, t.period), 1);
  const schedule: ScheduleEntry[] = [];
  const remaining: Map<number, number> = new Map();
  const nextRelease: Map<number, number> = new Map();
  const deadlines: Map<number, number> = new Map();

  sorted.forEach((t) => {
    remaining.set(t.id, 0);
    nextRelease.set(t.id, 0);
    deadlines.set(t.id, t.deadline);
  });

  for (let time = 0; time < hyperPeriod && time < 60; time++) {
    sorted.forEach((t) => {
      if (time === nextRelease.get(t.id)) {
        remaining.set(t.id, (remaining.get(t.id) || 0) + t.execution);
        nextRelease.set(t.id, time + t.period);
        deadlines.set(t.id, time + t.deadline);
      }
    });

    let selected: Task | null = null;
    for (const t of sorted) {
      if ((remaining.get(t.id) || 0) > 0) {
        selected = t;
        break;
      }
    }

    if (selected) {
      remaining.set(selected.id, (remaining.get(selected.id) || 0) - 1);
      schedule.push({
        time,
        taskId: selected.id,
        taskName: selected.name,
        color: selected.color,
      });
    } else {
      schedule.push({ time, taskId: null, taskName: "idle", color: "#374151" });
    }
  }
  return schedule;
}

function computeEDF(tasks: Task[]): ScheduleEntry[] {
  const hyperPeriod = tasks.reduce((acc, t) => lcm(acc, t.period), 1);
  const schedule: ScheduleEntry[] = [];
  const remaining: Map<number, number> = new Map();
  const nextRelease: Map<number, number> = new Map();
  const currentDeadline: Map<number, number> = new Map();

  tasks.forEach((t) => {
    remaining.set(t.id, 0);
    nextRelease.set(t.id, 0);
    currentDeadline.set(t.id, Infinity);
  });

  for (let time = 0; time < hyperPeriod && time < 60; time++) {
    tasks.forEach((t) => {
      if (time === nextRelease.get(t.id)) {
        remaining.set(t.id, (remaining.get(t.id) || 0) + t.execution);
        nextRelease.set(t.id, time + t.period);
        currentDeadline.set(t.id, time + t.deadline);
      }
    });

    let selected: Task | null = null;
    let earliestDeadline = Infinity;
    for (const t of tasks) {
      if ((remaining.get(t.id) || 0) > 0) {
        const dl = currentDeadline.get(t.id) || Infinity;
        if (dl < earliestDeadline) {
          earliestDeadline = dl;
          selected = t;
        }
      }
    }

    if (selected) {
      remaining.set(selected.id, (remaining.get(selected.id) || 0) - 1);
      schedule.push({
        time,
        taskId: selected.id,
        taskName: selected.name,
        color: selected.color,
      });
    } else {
      schedule.push({ time, taskId: null, taskName: "idle", color: "#374151" });
    }
  }
  return schedule;
}

function computeUtilization(tasks: Task[]): number {
  return tasks.reduce((sum, t) => sum + t.execution / t.period, 0);
}

function liuLaybound(n: number): number {
  if (n <= 0) return 0;
  return n * (Math.pow(2, 1 / n) - 1);
}

export default function RealTimeSchedulingSimulator() {
  const [tasks, setTasks] = useState<Task[]>(defaultTasks);
  const [algorithm, setAlgorithm] = useState<Algorithm>("rms");
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [newPeriod, setNewPeriod] = useState(8);
  const [newExec, setNewExec] = useState(2);

  const schedule = useMemo(
    () => (algorithm === "rms" ? computeRMS(tasks) : computeEDF(tasks)),
    [tasks, algorithm]
  );

  const utilization = useMemo(() => computeUtilization(tasks), [tasks]);
  const bound = useMemo(() => liuLaybound(tasks.length), [tasks.length]);
  const isSchedulable = utilization <= bound;

  const hyperPeriod = useMemo(
    () => tasks.reduce((acc, t) => lcm(acc, t.period), 1),
    [tasks]
  );

  const handlePlay = useCallback(() => {
    if (playing) {
      setPlaying(false);
      return;
    }
    setPlaying(true);
    setCurrentTime(0);
    let t = 0;
    const interval = setInterval(() => {
      t++;
      if (t >= schedule.length) {
        clearInterval(interval);
        setPlaying(false);
        return;
      }
      setCurrentTime(t);
    }, 300);
  }, [playing, schedule.length]);

  const handleReset = () => {
    setPlaying(false);
    setCurrentTime(0);
  };

  const addTask = () => {
    const id = Math.max(0, ...tasks.map((t) => t.id)) + 1;
    setTasks([
      ...tasks,
      {
        id,
        name: `τ${id}`,
        period: newPeriod,
        execution: newExec,
        deadline: newPeriod,
        color: TASK_COLORS[(id - 1) % TASK_COLORS.length],
      },
    ]);
  };

  const removeTask = (id: number) => {
    setTasks(tasks.filter((t) => t.id !== id));
  };

  return (
    <div className="w-full space-y-6 p-4 bg-white dark:bg-gray-900 rounded-xl">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
          <Timer className="w-5 h-5 text-blue-500" />
          实时调度模拟器 — {algorithm.toUpperCase()}
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => setAlgorithm("rms")}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              algorithm === "rms"
                ? "bg-blue-500 text-white shadow-lg"
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
            }`}
          >
            RMS
          </button>
          <button
            onClick={() => setAlgorithm("edf")}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              algorithm === "edf"
                ? "bg-emerald-500 text-white shadow-lg"
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
            }`}
          >
            EDF
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-1 space-y-3">
          <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400">
            任务集
          </h4>
          <div className="space-y-2">
            {tasks.map((t) => (
              <div
                key={t.id}
                className="flex items-center gap-2 p-2 rounded-lg bg-gray-50 dark:bg-gray-800"
              >
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: t.color }}
                />
                <span className="text-sm font-mono text-gray-700 dark:text-gray-300 flex-1">
                  {t.name}: C={t.execution}, T={t.period}
                </span>
                <button
                  onClick={() => removeTask(t.id)}
                  className="text-red-400 hover:text-red-600 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>

          <div className="flex gap-2 items-end">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">执行时间 C</label>
              <input
                type="number"
                min={1}
                value={newExec}
                onChange={(e) => setNewExec(Number(e.target.value))}
                className="w-16 px-2 py-1 rounded border dark:bg-gray-800 dark:border-gray-600 text-sm"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">周期 T</label>
              <input
                type="number"
                min={1}
                value={newPeriod}
                onChange={(e) => setNewPeriod(Number(e.target.value))}
                className="w-16 px-2 py-1 rounded border dark:bg-gray-800 dark:border-gray-600 text-sm"
              />
            </div>
            <button
              onClick={addTask}
              className="px-3 py-1.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-1 text-sm"
            >
              <Plus className="w-4 h-4" /> 添加
            </button>
          </div>

          <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">CPU 利用率</span>
              <span className="font-mono font-bold text-gray-800 dark:text-gray-200">
                {(utilization * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${
                  isSchedulable ? "bg-emerald-500" : "bg-red-500"
                }`}
                animate={{ width: `${Math.min(utilization * 100, 100)}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            {algorithm === "rms" && (
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Liu-Layland 上界: {(bound * 100).toFixed(1)}% →{" "}
                <span
                  className={isSchedulable ? "text-emerald-500 font-bold" : "text-red-500 font-bold"}
                >
                  {isSchedulable ? "可调度 ✅" : "需进一步分析 ⚠️"}
                </span>
              </div>
            )}
            <div className="text-xs text-gray-500 dark:text-gray-400">
              超周期 (LCM): {hyperPeriod > 60 ? `>${60}` : hyperPeriod}
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handlePlay}
              className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                playing
                  ? "bg-amber-500 text-white"
                  : "bg-emerald-500 text-white hover:bg-emerald-600"
              }`}
            >
              {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {playing ? "暂停" : "播放"}
            </button>
            <button
              onClick={handleReset}
              className="px-3 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="md:col-span-2 space-y-3">
          <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400">
            调度时间线
          </h4>
          <div className="overflow-x-auto">
            <div className="flex gap-0.5 min-w-max">
              {schedule.map((entry, i) => (
                <motion.div
                  key={i}
                  className="flex flex-col items-center"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{
                    opacity: i <= currentTime || !playing ? 1 : 0.3,
                    y: 0,
                    scale: i === currentTime && playing ? 1.15 : 1,
                  }}
                  transition={{ duration: 0.15 }}
                >
                  <div
                    className="w-7 h-7 rounded-sm flex items-center justify-center text-xs font-bold text-white"
                    style={{ backgroundColor: entry.color }}
                  >
                    {entry.taskId ? entry.taskName.replace("τ", "") : ""}
                  </div>
                  <span className="text-[9px] text-gray-400 mt-0.5">{entry.time}</span>
                </motion.div>
              ))}
            </div>
          </div>

          <div className="flex gap-4 flex-wrap mt-2">
            {tasks.map((t) => (
              <div key={t.id} className="flex items-center gap-1.5">
                <div
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: t.color }}
                />
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  {t.name} (C={t.execution}, T={t.period})
                </span>
              </div>
            ))}
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm bg-gray-700" />
              <span className="text-xs text-gray-600 dark:text-gray-400">idle</span>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
            <p className="text-xs text-blue-700 dark:text-blue-300">
              <Zap className="w-3 h-3 inline mr-1" />
              {algorithm === "rms"
                ? "RMS: 固定优先级，周期越短优先级越高。CPU 利用率上界为 n(2^(1/n)-1)。"
                : "EDF: 动态优先级，截止时间越近优先级越高。理论上 CPU 利用率可达 100%。"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
