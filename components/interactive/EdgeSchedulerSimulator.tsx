"use client";
import { useState } from "react";

interface Task {
  id: number;
  name: string;
  compute: number;
  deadline: number;
  assigned: string | null;
  status: "pending" | "running" | "done";
}

const INIT_TASKS: Task[] = [
  { id: 1, name: "视频转码", compute: 8, deadline: 10, assigned: null, status: "pending" },
  { id: 2, name: "图像识别", compute: 5, deadline: 8, assigned: null, status: "pending" },
  { id: 3, name: "数据压缩", compute: 3, deadline: 15, assigned: null, status: "pending" },
  { id: 4, name: "AR渲染", compute: 10, deadline: 6, assigned: null, status: "pending" },
  { id: 5, name: "语音合成", compute: 4, deadline: 12, assigned: null, status: "pending" },
];

const EDGES = ["边缘节点A (GPU)", "边缘节点B (CPU)", "边缘节点C (NPU)"];

export function EdgeSchedulerSimulator() {
  const [tasks, setTasks] = useState<Task[]>(INIT_TASKS.map((t) => ({ ...t })));
  const [strategy, setStrategy] = useState<"deadline" | "load" | "compute">("deadline");
  const [time, setTime] = useState(0);

  const schedule = () => {
    const pending = tasks.filter((t) => t.status === "pending").sort((a, b) => {
      if (strategy === "deadline") return a.deadline - b.deadline;
      if (strategy === "compute") return b.compute - a.compute;
      return 0;
    });

    const load = [0, 0, 0];
    const updated = tasks.map((t) => ({ ...t }));
    for (const task of pending) {
      let target = 0;
      if (strategy === "load") {
        target = load.indexOf(Math.min(...load));
      } else {
        target = task.id % 3;
      }
      const t = updated.find((x) => x.id === task.id)!;
      t.assigned = EDGES[target];
      t.status = "running";
      load[target] += t.compute;
    }
    setTasks(updated);
  };

  const tick = () => {
    const updated = tasks.map((t) => {
      if (t.status === "running") return { ...t, status: "done" as const };
      return t;
    });
    setTasks(updated);
    setTime(time + 1);
  };

  const reset = () => {
    setTasks(INIT_TASKS.map((t) => ({ ...t })));
    setTime(0);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">MEC边缘任务调度</h3>
      <div className="flex gap-2 mb-4">
        {(["deadline", "load", "compute"] as const).map((s) => (
          <button key={s} onClick={() => setStrategy(s)}
            className={`px-3 py-1.5 rounded text-sm ${strategy === s ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
            {s === "deadline" ? "最早截止" : s === "load" ? "负载均衡" : "计算优先"}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        {EDGES.map((edge, i) => (
          <div key={i} className="bg-bg-muted rounded-lg p-3">
            <div className="font-semibold text-sm text-text-primary mb-2">{edge}</div>
            {tasks.filter((t) => t.assigned === edge).map((t) => (
              <div key={t.id} className={`text-xs p-1.5 rounded mb-1 ${t.status === "done" ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300" : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"}`}>
                {t.name} ({t.compute} CU)
              </div>
            ))}
            {tasks.filter((t) => t.assigned === edge).length === 0 && <div className="text-xs text-text-secondary">空闲</div>}
          </div>
        ))}
      </div>
      <div className="flex gap-3 mb-4">
        <button onClick={schedule} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm">调度分配</button>
        <button onClick={tick} className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 text-sm">执行完成</button>
        <button onClick={reset} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">重置</button>
      </div>
      <div className="text-xs text-text-secondary">
        MEC(多接入边缘计算)将计算任务卸载到网络边缘,减少延迟。调度策略包括:截止时间优先、负载均衡、计算能力匹配。
      </div>
    </div>
  );
}

export default EdgeSchedulerSimulator;
