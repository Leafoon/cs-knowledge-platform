"use client";

import React, { useState } from "react";

/** 两个栈模拟队列（LeetCode #232）：动画展示 inbox/outbox 的转移过程 */

interface QueueState {
  inbox: number[];     // 入队栈（push 的目标）
  outbox: number[];    // 出队栈（pop 的来源）
  totalEnqueued: number;
}

interface LogEntry {
  op: string;
  detail: string;
  type: "enqueue" | "dequeue" | "transfer" | "info";
}

const initState = (): QueueState => ({ inbox: [], outbox: [], totalEnqueued: 0 });

function enqueue(state: QueueState, val: number): { state: QueueState; logs: LogEntry[] } {
  const logs: LogEntry[] = [
    { op: `enqueue(${val})`, detail: `将 ${val} 压入 inbox 栈顶，O(1)`, type: "enqueue" },
  ];
  return {
    state: { ...state, inbox: [...state.inbox, val], totalEnqueued: state.totalEnqueued + 1 },
    logs,
  };
}

function dequeue(state: QueueState): { state: QueueState; val: number | null; logs: LogEntry[] } {
  const logs: LogEntry[] = [];

  let inbox = [...state.inbox];
  let outbox = [...state.outbox];

  // 如果 outbox 空，先把 inbox 全部倒入 outbox
  if (outbox.length === 0) {
    if (inbox.length === 0) {
      return { state, val: null, logs: [{ op: "dequeue()", detail: "队列为空！", type: "info" }] };
    }
    logs.push({
      op: "transfer",
      detail: `outbox 为空，将 inbox 所有元素（共 ${inbox.length} 个）倒入 outbox`,
      type: "transfer",
    });
    while (inbox.length > 0) {
      outbox.push(inbox.pop()!);
    }
  }

  const val = outbox.pop()!;
  logs.push({
    op: `dequeue() → ${val}`,
    detail: `从 outbox 栈顶弹出 ${val}（这是最早入队的元素）`,
    type: "dequeue",
  });

  return {
    state: { ...state, inbox, outbox },
    val,
    logs,
  };
}

// 栈可视化：从底到顶展示，顶部在上
function StackView({ title, items, highlight, color }: {
  title: string;
  items: number[];
  highlight?: boolean;
  color: "green" | "orange";
}) {
  const borderColor = color === "green" ? "border-green-500" : "border-orange-500";
  const titleColor = color === "green" ? "text-green-400" : "text-orange-400";
  const topBg = color === "green" ? "bg-green-500/20 border-green-500 text-green-200" : "bg-orange-500/20 border-orange-500 text-orange-200";

  return (
    <div className={`flex flex-col rounded-xl border-2 ${highlight ? borderColor : "border-border-subtle"} bg-bg-tertiary transition-colors duration-300`}
      style={{ minHeight: 220 }}>
      <div className={`text-center text-xs font-bold py-2 border-b border-border-subtle ${highlight ? titleColor : "text-text-secondary"}`}>
        {title}
        <span className="ml-2 text-text-tertiary font-normal">({items.length} 个)</span>
      </div>
      <div className="flex flex-col-reverse gap-1 p-2 flex-1 overflow-y-auto">
        {items.length === 0 ? (
          <div className="text-center text-text-tertiary text-xs italic py-4">（空栈）</div>
        ) : (
          items.map((v, i) => {
            const isTop = i === items.length - 1;
            return (
              <div key={i}
                className={`flex items-center justify-between px-3 py-1.5 rounded border text-sm font-mono transition-all duration-300 ${
                  isTop ? topBg : "bg-bg-secondary border-border-subtle text-text-secondary"
                }`}>
                <span className="font-bold">{v}</span>
                {isTop && <span className="text-[10px] opacity-70">← 栈顶</span>}
              </div>
            );
          })
        )}
      </div>
      <div className={`text-center text-[10px] border-t border-border-subtle py-1 ${highlight ? titleColor : "text-text-tertiary"} opacity-60`}>
        ▽ 栈底
      </div>
    </div>
  );
}

export default function TwoStackQueue() {
  const [state, setState] = useState(initState());
  const [logs, setLogs] = useState<LogEntry[]>([
    { op: "初始化", detail: "两个栈均为空。enqueue(x) 压入 inbox；dequeue() 从 outbox 弹出（必要时转移）", type: "info" },
  ]);
  const [inputVal, setInputVal] = useState("10");
  const [activeStack, setActiveStack] = useState<"inbox" | "outbox" | null>(null);
  const [dequeued, setDequeued] = useState<number | null>(null);

  const handleEnqueue = () => {
    const v = parseInt(inputVal);
    if (isNaN(v)) return;
    const { state: ns, logs: newLogs } = enqueue(state, v);
    setState(ns);
    setLogs((prev) => [...newLogs, ...prev].slice(0, 12));
    setActiveStack("inbox");
    setDequeued(null);
    setInputVal(String(Math.floor(Math.random() * 90) + 10));
    setTimeout(() => setActiveStack(null), 800);
  };

  const handleDequeue = () => {
    const { state: ns, val, logs: newLogs } = dequeue(state);
    setState(ns);
    setLogs((prev) => [...newLogs, ...prev].slice(0, 12));
    setDequeued(val);
    setActiveStack("outbox");
    setTimeout(() => { setActiveStack(null); setDequeued(null); }, 1200);
  };

  const handleReset = () => {
    setState(initState());
    setLogs([{ op: "重置", detail: "两个栈均已清空", type: "info" }]);
    setActiveStack(null);
    setDequeued(null);
  };

  const totalSize = state.inbox.length + state.outbox.length;

  const logColor = (type: LogEntry["type"]) => {
    if (type === "enqueue") return "text-green-400";
    if (type === "dequeue") return "text-orange-400";
    if (type === "transfer") return "text-purple-400";
    return "text-text-tertiary";
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      {/* 标题 */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-base font-bold text-text-primary">⚖️ 双栈模拟队列</h3>
          <p className="text-xs text-text-tertiary mt-0.5">
            inbox（入队栈）+ outbox（出队栈）→ 均摊 O(1) dequeue
          </p>
        </div>
        <div className="text-xs bg-bg-tertiary border border-border-subtle rounded px-3 py-1">
          队列大小：<span className="text-blue-400 font-bold">{totalSize}</span>
        </div>
      </div>

      {/* 队列语义标注 */}
      <div className="flex justify-between text-xs text-text-tertiary border border-border-subtle rounded px-4 py-2">
        <span>⬅ 出队方向（dequeue）</span>
        <span>enqueue 方向 ➡</span>
      </div>

      {/* 双栈可视化 */}
      <div className="grid grid-cols-2 gap-4">
        <StackView
          title="outbox（出队栈）"
          items={state.outbox}
          highlight={activeStack === "outbox"}
          color="orange"
        />
        <StackView
          title="inbox（入队栈）"
          items={state.inbox}
          highlight={activeStack === "inbox"}
          color="green"
        />
      </div>

      {/* 转移动画提示 */}
      {dequeued !== null && (
        <div className="flex items-center justify-center gap-2 bg-purple-500/10 border border-purple-500/40 rounded-lg px-4 py-2 text-purple-300 text-sm font-bold">
          <span>出队值：</span>
          <span className="text-2xl text-purple-200">{dequeued}</span>
        </div>
      )}

      {/* 操作区 */}
      <div className="flex flex-wrap gap-2 items-center justify-center">
        <input
          type="number"
          value={inputVal}
          onChange={(e) => setInputVal(e.target.value)}
          className="w-20 px-2 py-2 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-sm focus:outline-none focus:border-blue-400"
          placeholder="值"
        />
        <button onClick={handleEnqueue}
          className="px-4 py-2 rounded-lg bg-green-600 text-white text-sm hover:bg-green-700 transition-colors">
          ＋ 入队 (enqueue)
        </button>
        <button onClick={handleDequeue} disabled={totalSize === 0}
          className="px-4 py-2 rounded-lg bg-orange-600 text-white text-sm hover:bg-orange-700 disabled:opacity-40 transition-colors">
          − 出队 (dequeue)
        </button>
        <button onClick={handleReset}
          className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-sm transition-colors">
          ↩ 重置
        </button>
      </div>

      {/* 算法要点 */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-green-500/10 border border-green-500/30 rounded p-2">
          <div className="text-green-400 font-semibold mb-1">enqueue(x)</div>
          <div className="text-text-secondary">直接 push 到 inbox</div>
          <div className="text-text-tertiary">时间：严格 O(1)</div>
        </div>
        <div className="bg-orange-500/10 border border-orange-500/30 rounded p-2">
          <div className="text-orange-400 font-semibold mb-1">dequeue()</div>
          <div className="text-text-secondary">outbox 空时才将 inbox 全部倒入</div>
          <div className="text-text-tertiary">均摊 O(1) ← 每元素最多移动一次</div>
        </div>
      </div>

      {/* 操作日志 */}
      <div className="bg-bg-tertiary border border-border-subtle rounded-lg p-3">
        <div className="text-xs text-text-tertiary mb-2">操作日志（最新在前）</div>
        <div className="space-y-1 max-h-36 overflow-y-auto">
          {logs.map((log, i) => (
            <div key={i} className="flex gap-2 text-xs">
              <span className={`font-semibold min-w-[120px] ${logColor(log.type)}`}>{log.op}</span>
              <span className="text-text-tertiary">{log.detail}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
