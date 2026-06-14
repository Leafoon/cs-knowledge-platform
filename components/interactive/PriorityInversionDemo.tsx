"use client";

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, AlertTriangle, Shield, Zap } from "lucide-react";

type TaskId = "H" | "M" | "L";
type Mode = "inversion" | "inheritance";
type TaskState = "idle" | "running" | "waiting" | "blocked" | "holding_lock";

interface TaskInfo {
  id: TaskId;
  name: string;
  priority: number;
  color: string;
  state: TaskState;
  effectivePriority: number;
}

interface TimelineEvent {
  time: number;
  taskId: TaskId;
  action: string;
  detail: string;
}

const inversionEvents: TimelineEvent[] = [
  { time: 0, taskId: "L", action: "获取锁 R", detail: "低优先级任务获取共享资源 R" },
  { time: 1, taskId: "H", action: "释放并等待", detail: "高优先级任务需要资源 R，被 L 阻塞" },
  { time: 2, taskId: "M", action: "抢占 L", detail: "中优先级任务抢占低优先级任务 L" },
  { time: 3, taskId: "M", action: "继续执行", detail: "M 继续运行，H 仍在等待 R" },
  { time: 4, taskId: "M", action: "继续执行", detail: "M 仍在运行，H 被间接阻塞！" },
  { time: 5, taskId: "L", action: "恢复执行", detail: "M 完成，L 恢复运行" },
  { time: 6, taskId: "L", action: "释放锁 R", detail: "L 终于释放资源 R" },
  { time: 7, taskId: "H", action: "获取锁并执行", detail: "H 获取 R 并开始执行" },
  { time: 8, taskId: "H", action: "完成", detail: "H 完成（严重延迟）" },
];

const inheritanceEvents: TimelineEvent[] = [
  { time: 0, taskId: "L", action: "获取锁 R", detail: "低优先级任务获取共享资源 R" },
  { time: 1, taskId: "H", action: "释放并继承", detail: "H 需要 R → L 继承 H 的优先级" },
  { time: 2, taskId: "L", action: "高优先级运行", detail: "L 以 H 的优先级运行，M 无法抢占" },
  { time: 3, taskId: "L", action: "释放锁 R", detail: "L 完成并释放 R，恢复原优先级" },
  { time: 4, taskId: "H", action: "获取锁并执行", detail: "H 获取 R 并立即执行" },
  { time: 5, taskId: "H", action: "完成", detail: "H 快速完成 ✅" },
  { time: 6, taskId: "M", action: "开始执行", detail: "M 现在可以执行" },
  { time: 7, taskId: "M", action: "完成", detail: "M 完成" },
];

export default function PriorityInversionDemo() {
  const [mode, setMode] = useState<Mode>("inversion");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const events = mode === "inversion" ? inversionEvents : inheritanceEvents;

  const getTaskStates = useCallback(
    (currentStep: number): TaskInfo[] => {
      const base: TaskInfo[] = [
        { id: "H", name: "高优先级 (H)", priority: 3, color: "#ef4444", state: "idle", effectivePriority: 3 },
        { id: "M", name: "中优先级 (M)", priority: 2, color: "#f59e0b", state: "idle", effectivePriority: 2 },
        { id: "L", name: "低优先级 (L)", priority: 1, color: "#3b82f6", state: "idle", effectivePriority: 1 },
      ];

      if (currentStep <= 0) return base;

      const currentEvent = events[currentStep];
      if (!currentEvent) return base;

      if (mode === "inversion") {
        if (currentStep >= 0) base.find((t) => t.id === "L")!.state = "holding_lock";
        if (currentStep >= 1) base.find((t) => t.id === "H")!.state = "waiting";
        if (currentStep >= 2 && currentStep <= 4) base.find((t) => t.id === "M")!.state = "running";
        if (currentStep >= 5 && currentStep <= 5) {
          base.find((t) => t.id === "L")!.state = "running";
          base.find((t) => t.id === "M")!.state = "idle";
        }
        if (currentStep >= 6) {
          base.find((t) => t.id === "L")!.state = "idle";
          base.find((t) => t.id === "H")!.state = "running";
          base.find((t) => t.id === "M")!.state = "idle";
        }
        if (currentStep >= 7) {
          base.find((t) => t.id === "H")!.state = "running";
        }
        if (currentStep >= 8) {
          base.find((t) => t.id === "H")!.state = "idle";
        }
      } else {
        if (currentStep >= 0) base.find((t) => t.id === "L")!.state = "holding_lock";
        if (currentStep >= 1) {
          base.find((t) => t.id === "H")!.state = "waiting";
          const lTask = base.find((t) => t.id === "L")!;
          lTask.effectivePriority = 3;
        }
        if (currentStep >= 2) {
          base.find((t) => t.id === "L")!.state = "running";
        }
        if (currentStep >= 3) {
          base.find((t) => t.id === "L")!.state = "idle";
          base.find((t) => t.id === "L")!.effectivePriority = 1;
          base.find((t) => t.id === "H")!.state = "running";
        }
        if (currentStep >= 4) {
          base.find((t) => t.id === "H")!.state = "running";
        }
        if (currentStep >= 5) {
          base.find((t) => t.id === "H")!.state = "idle";
        }
        if (currentStep >= 6) {
          base.find((t) => t.id === "M")!.state = "running";
        }
        if (currentStep >= 7) {
          base.find((t) => t.id === "M")!.state = "idle";
        }
      }

      return base;
    },
    [mode, events]
  );

  const taskStates = useMemo(() => getTaskStates(step), [getTaskStates, step]);

  const handlePlay = useCallback(() => {
    if (playing) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      setPlaying(false);
      return;
    }
    setPlaying(true);
    if (step >= events.length - 1) setStep(0);
    let s = step >= events.length - 1 ? 0 : step;
    intervalRef.current = setInterval(() => {
      s++;
      if (s >= events.length) {
        if (intervalRef.current) clearInterval(intervalRef.current);
        setPlaying(false);
        return;
      }
      setStep(s);
    }, 1200);
  }, [playing, step, events.length]);

  const handleReset = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setPlaying(false);
    setStep(0);
  };

  const handleModeChange = (m: Mode) => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setPlaying(false);
    setMode(m);
    setStep(0);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const currentEvent = events[step];

  return (
    <div className="w-full space-y-6 p-4 bg-white dark:bg-gray-900 rounded-xl">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-amber-500" />
          优先级反转演示
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => handleModeChange("inversion")}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              mode === "inversion"
                ? "bg-red-500 text-white shadow-lg"
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
            }`}
          >
            优先级反转
          </button>
          <button
            onClick={() => handleModeChange("inheritance")}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              mode === "inheritance"
                ? "bg-emerald-500 text-white shadow-lg"
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
            }`}
          >
            优先级继承
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {taskStates.map((task) => {
          const isRunning = task.state === "running";
          const isWaiting = task.state === "waiting";
          const isHolding = task.state === "holding_lock";
          const inherited = mode === "inheritance" && task.id === "L" && task.effectivePriority > task.priority;

          return (
            <motion.div
              key={task.id}
              layout
              className={`p-4 rounded-xl border-2 transition-all ${
                isRunning
                  ? "border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20"
                  : isWaiting
                  ? "border-red-400 bg-red-50 dark:bg-red-900/20"
                  : "border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
              }`}
            >
              <div className="flex items-center gap-3 mb-3">
                <motion.div
                  className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-lg"
                  style={{ backgroundColor: task.color }}
                  animate={isRunning ? { scale: [1, 1.15, 1] } : { scale: 1 }}
                  transition={{ repeat: isRunning ? Infinity : 0, duration: 0.8 }}
                >
                  {task.id}
                </motion.div>
                <div>
                  <div className="font-bold text-sm text-gray-800 dark:text-gray-200">
                    {task.name}
                  </div>
                  <div className="text-xs text-gray-500">
                    优先级: {task.priority}
                    {inherited && (
                      <span className="ml-1 text-emerald-500 font-bold">
                        → {task.effectivePriority} (继承)
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <AnimatePresence mode="wait">
                <motion.div
                  key={task.state}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium text-center ${
                    isRunning
                      ? "bg-emerald-500 text-white"
                      : isWaiting
                      ? "bg-red-500 text-white"
                      : isHolding
                      ? "bg-amber-500 text-white"
                      : "bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
                  }`}
                >
                  {isRunning
                    ? "运行中 ▶"
                    : isWaiting
                    ? "等待资源 ⏳"
                    : isHolding
                    ? "持有锁 🔒"
                    : "空闲"}
                </motion.div>
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={handlePlay}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
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

        <div className="flex-1 flex items-center gap-2 ml-4">
          {events.map((_, i) => (
            <button
              key={i}
              onClick={() => { setStep(i); setPlaying(false); }}
              className={`w-6 h-6 rounded-full text-xs font-bold transition-all ${
                i === step
                  ? "bg-blue-500 text-white scale-125"
                  : i < step
                  ? "bg-blue-200 dark:bg-blue-800 text-blue-700 dark:text-blue-300"
                  : "bg-gray-200 dark:bg-gray-700 text-gray-500"
              }`}
            >
              {i + 1}
            </button>
          ))}
        </div>
      </div>

      {currentEvent && (
        <motion.div
          key={step}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
        >
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-bold text-blue-700 dark:text-blue-300">
              步骤 {step + 1}: {currentEvent.action}
            </span>
          </div>
          <p className="text-xs text-blue-600 dark:text-blue-400">{currentEvent.detail}</p>
        </motion.div>
      )}

      {mode === "inversion" && step >= 4 && step <= 5 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700 flex items-center gap-2"
        >
          <AlertTriangle className="w-5 h-5 text-red-500 shrink-0" />
          <span className="text-sm text-red-700 dark:text-red-300 font-medium">
            无界优先级反转！高优先级任务 H 被中优先级任务 M 间接阻塞
          </span>
        </motion.div>
      )}

      {mode === "inheritance" && step >= 1 && step <= 2 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-300 dark:border-emerald-700 flex items-center gap-2"
        >
          <Shield className="w-5 h-5 text-emerald-500 shrink-0" />
          <span className="text-sm text-emerald-700 dark:text-emerald-300 font-medium">
            优先级继承生效！L 临时获得 H 的优先级，M 无法抢占
          </span>
        </motion.div>
      )}
    </div>
  );
}
