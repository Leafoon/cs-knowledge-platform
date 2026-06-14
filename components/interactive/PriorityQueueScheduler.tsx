"use client";
import React, { useState, useCallback } from "react";

// ──────────── Task Priority Scheduler Demo ────────────
interface Task {
  id: number;
  name: string;
  priority: number;  // lower = higher priority (min-heap)
  duration: number;  // ticks
  color: string;
}

interface SchedulerState {
  queue: Task[];
  running: Task | null;
  completed: { task: Task; completedAt: number }[];
  tick: number;
  log: string[];
}

const TASK_COLORS = ["#6366f1", "#3b82f6", "#0ea5e9", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#f97316"];

const PRESET_TASKS: Omit<Task, "id" | "color">[] = [
  { name: "系统中断", priority: 1, duration: 2 },
  { name: "用户键盘输入", priority: 2, duration: 1 },
  { name: "网络数据接收", priority: 3, duration: 3 },
  { name: "后台备份", priority: 8, duration: 4 },
  { name: "磁盘刷新", priority: 6, duration: 2 },
  { name: "日志写入", priority: 7, duration: 2 },
  { name: "屏幕刷新", priority: 4, duration: 1 },
];

function initTasks(): Task[] {
  return PRESET_TASKS.map((t, i) => ({ ...t, id: i + 1, color: TASK_COLORS[i % TASK_COLORS.length] }));
}

// Min-heap helpers
function heapPush(heap: Task[], task: Task) {
  const h = [...heap, task];
  h.sort((a, b) => a.priority - b.priority);
  return h;
}

export default function PriorityQueueScheduler() {
  const [tasks] = useState<Task[]>(initTasks());
  const [state, setState] = useState<SchedulerState>({
    queue: [...initTasks()].sort((a, b) => a.priority - b.priority),
    running: null,
    completed: [],
    tick: 0,
    log: ["调度器就绪，所有任务已加入优先队列"],
  });
  const [ticksLeft, setTicksLeft] = useState(0);
  const [customName, setCustomName] = useState("");
  const [customPri, setCustomPri] = useState(5);
  const [customDur, setCustomDur] = useState(2);
  const [nextId, setNextId] = useState(PRESET_TASKS.length + 1);

  const tick = useCallback(() => {
    setState(prev => {
      const { queue, running, completed, tick, log } = prev;
      const newLog = [...log];
      const newTick = tick + 1;
      let newQueue = [...queue];
      let newRunning = running;
      let newCompleted = [...completed];
      let newTicksLeft = ticksLeft;

      if (newRunning) {
        // Continue running current task
        newTicksLeft = ticksLeft - 1;
        newLog.push(`[Tick ${newTick}] 执行任务「${newRunning.name}」（优先级=${newRunning.priority}），剩余 ${newTicksLeft} tick`);
        if (newTicksLeft <= 0) {
          newLog.push(`✅ 任务「${newRunning.name}」执行完毕`);
          newCompleted = [...newCompleted, { task: newRunning, completedAt: newTick }];
          newRunning = null;
          newTicksLeft = 0;
        }
      }

      if (!newRunning && newQueue.length > 0) {
        newRunning = newQueue[0];
        newQueue = newQueue.slice(1);
        newTicksLeft = newRunning.duration;
        newLog.push(`[Tick ${newTick}] EXTRACT-MIN：提取优先级最高的「${newRunning.name}」（p=${newRunning.priority}），开始执行 ${newRunning.duration} tick`);
      } else if (!newRunning && newQueue.length === 0) {
        newLog.push(`[Tick ${newTick}] 队列为空，调度器空闲`);
      }

      setTicksLeft(newTicksLeft);
      return { queue: newQueue, running: newRunning, completed: newCompleted, tick: newTick, log: newLog.slice(-12) };
    });
  }, [ticksLeft]);

  const addTask = useCallback(() => {
    if (!customName.trim()) return;
    const task: Task = { id: nextId, name: customName.trim(), priority: customPri, duration: customDur, color: TASK_COLORS[nextId % TASK_COLORS.length] };
    setNextId(id => id + 1);
    setCustomName("");
    setState(prev => ({
      ...prev,
      queue: heapPush(prev.queue, task),
      log: [...prev.log, `📥 新任务「${task.name}」（p=${task.priority}）入队，队列重新排序`].slice(-12),
    }));
  }, [customName, customPri, customDur, nextId]);

  const reset = () => {
    setState({ queue: [...initTasks()].sort((a, b) => a.priority - b.priority), running: null, completed: [], tick: 0, log: ["调度器重置"] });
    setTicksLeft(0);
    setNextId(PRESET_TASKS.length + 1);
  };

  return (
    <div className="rounded-xl border p-4 space-y-4 select-none" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-card)" }}>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="font-bold text-base" style={{ color: "var(--color-text-primary)" }}>
          ⚙️ 优先级任务调度模拟器
        </h3>
        <button onClick={reset} className="px-3 py-1 rounded text-xs border" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)" }}>↺ 重置</button>
      </div>

      <div className="text-xs rounded-lg px-3 py-2" style={{ background: "rgba(99,102,241,0.08)", color: "var(--color-text-muted)" }}>
        💡 优先队列（最小堆）：每次 <strong style={{ color: "#6366f1" }}>EXTRACT-MIN</strong> 取出优先级数字最小（最紧急）的任务执行。
      </div>

      <div className="grid grid-cols-1 gap-3" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {/* Queue */}
        <div>
          <p className="text-xs font-semibold mb-2" style={{ color: "var(--color-text-muted)" }}>📋 优先队列（MIN-HEAP）</p>
          <div className="rounded-lg p-2 space-y-1 min-h-24" style={{ background: "var(--color-bg-secondary)" }}>
            {state.queue.length === 0 ? (
              <p className="text-xs text-center py-3" style={{ color: "var(--color-text-muted)" }}>队列为空</p>
            ) : state.queue.map((t, i) => (
              <div key={t.id} className="flex items-center gap-2 px-2 py-1.5 rounded text-xs" style={{ background: i === 0 ? `${t.color}22` : "var(--color-bg-card)", borderLeft: i === 0 ? `3px solid ${t.color}` : "3px solid transparent" }}>
                <div className="w-5 h-5 rounded-full flex items-center justify-center text-white text-xs font-bold flex-shrink-0" style={{ background: t.color }}>{t.priority}</div>
                <span className="font-medium flex-1" style={{ color: "var(--color-text-primary)" }}>{t.name}</span>
                <span style={{ color: "var(--color-text-muted)" }}>{t.duration}tick</span>
                {i === 0 && <span className="text-xs px-1 rounded" style={{ background: t.color, color: "#fff" }}>堆顶</span>}
              </div>
            ))}
          </div>
        </div>

        {/* Running + Completed */}
        <div className="space-y-2">
          <div>
            <p className="text-xs font-semibold mb-1" style={{ color: "var(--color-text-muted)" }}>🏃 当前执行</p>
            {state.running ? (
              <div className="rounded-lg p-2 flex items-center gap-2" style={{ background: `${state.running.color}22`, border: `2px solid ${state.running.color}` }}>
                <div className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold" style={{ background: state.running.color }}>{state.running.priority}</div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-bold truncate" style={{ color: "var(--color-text-primary)" }}>{state.running.name}</p>
                  <div className="w-full rounded-full h-1.5 mt-1" style={{ background: "var(--color-border)" }}>
                    <div className="h-1.5 rounded-full transition-all" style={{ width: `${(1 - ticksLeft / state.running.duration) * 100}%`, background: state.running.color }} />
                  </div>
                  <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>剩余 {ticksLeft}/{state.running.duration} tick</p>
                </div>
              </div>
            ) : (
              <div className="rounded-lg p-2 text-center text-xs" style={{ background: "var(--color-bg-secondary)", color: "var(--color-text-muted)" }}>CPU 空闲</div>
            )}
          </div>

          <div>
            <p className="text-xs font-semibold mb-1" style={{ color: "var(--color-text-muted)" }}>✅ 已完成（{state.completed.length}）</p>
            <div className="max-h-32 overflow-y-auto space-y-1">
              {state.completed.slice().reverse().map(({ task, completedAt }) => (
                <div key={`${task.id}-${completedAt}`} className="flex items-center gap-2 px-2 py-1 rounded text-xs" style={{ background: "var(--color-bg-secondary)" }}>
                  <div className="w-4 h-4 rounded-full flex items-center justify-center text-white text-xs font-bold flex-shrink-0" style={{ background: task.color }}>{task.priority}</div>
                  <span className="flex-1 truncate" style={{ color: "var(--color-text-muted)" }}>{task.name}</span>
                  <span style={{ color: "var(--color-text-muted)" }}>@{completedAt}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Add custom task */}
      <div className="rounded-lg p-3 space-y-2" style={{ background: "var(--color-bg-secondary)" }}>
        <p className="text-xs font-semibold" style={{ color: "var(--color-text-muted)" }}>📥 动态添加任务（INSERT + HEAPIFY-UP）</p>
        <div className="flex gap-2 flex-wrap">
          <input value={customName} onChange={e => setCustomName(e.target.value)} placeholder="任务名称"
            className="flex-1 min-w-24 rounded px-2 py-1 text-xs border" style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)", color: "var(--color-text-primary)" }} />
          <div className="flex items-center gap-1 text-xs" style={{ color: "var(--color-text-muted)" }}>
            优先级:
            <input type="number" min={1} max={10} value={customPri} onChange={e => setCustomPri(Number(e.target.value))}
              className="w-10 rounded px-1 py-1 border text-center" style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)", color: "var(--color-text-primary)" }} />
          </div>
          <div className="flex items-center gap-1 text-xs" style={{ color: "var(--color-text-muted)" }}>
            时长:
            <input type="number" min={1} max={8} value={customDur} onChange={e => setCustomDur(Number(e.target.value))}
              className="w-10 rounded px-1 py-1 border text-center" style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)", color: "var(--color-text-primary)" }} />
          </div>
          <button onClick={addTask} className="px-3 py-1 rounded text-xs font-bold" style={{ background: "#22c55e", color: "#fff" }}>+ 添加</button>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-2 justify-center">
        <button onClick={tick}
          className="px-6 py-2 rounded-lg font-bold text-sm transition-all"
          style={{ background: "#6366f1", color: "#fff" }}>
          ⏱ 执行 1 Tick（t={state.tick}）
        </button>
      </div>

      {/* Log */}
      <div className="rounded-lg p-2 max-h-28 overflow-y-auto space-y-0.5 text-xs" style={{ background: "var(--color-bg-secondary)" }}>
        {state.log.slice().reverse().map((entry, i) => (
          <p key={i} style={{ color: i === 0 ? "var(--color-text-primary)" : "var(--color-text-muted)" }}>{entry}</p>
        ))}
      </div>
    </div>
  );
}
