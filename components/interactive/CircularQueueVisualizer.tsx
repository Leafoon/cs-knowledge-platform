"use client";

import React, { useState } from "react";

/** 循环队列可视化：数组槽环形排列，head/tail 指针动态移动 */

const CAPACITY = 7; // 实际可存元素数（数组大小 = CAPACITY + 1）
const ARR_SIZE = CAPACITY + 1; // 内部数组预留一个空槽

interface QueueState {
  data: (number | null)[];
  head: number;
  tail: number;
}

function isFull(q: QueueState): boolean {
  return (q.tail + 1) % ARR_SIZE === q.head;
}

function isEmpty(q: QueueState): boolean {
  return q.head === q.tail;
}

function queueSize(q: QueueState): number {
  return (q.tail - q.head + ARR_SIZE) % ARR_SIZE;
}

function enqueue(q: QueueState, val: number): QueueState | null {
  if (isFull(q)) return null;
  const newData = [...q.data];
  newData[q.tail] = val;
  return { ...q, data: newData, tail: (q.tail + 1) % ARR_SIZE };
}

function dequeue(q: QueueState): { state: QueueState; val: number } | null {
  if (isEmpty(q)) return null;
  const val = q.data[q.head] as number;
  const newData = [...q.data];
  newData[q.head] = null;
  return { state: { ...q, data: newData, head: (q.head + 1) % ARR_SIZE }, val };
}

function initQueue(): QueueState {
  return { data: Array(ARR_SIZE).fill(null), head: 0, tail: 0 };
}

// 计算圆上第 i 个槽的 (x, y)，中心 (cx, cy)，半径 r
function slotPos(i: number, total: number, cx: number, cy: number, r: number) {
  const angle = (2 * Math.PI * i) / total - Math.PI / 2; // 从 12 点位开始
  return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
}

export default function CircularQueueVisualizer() {
  const [queue, setQueue] = useState<QueueState>(initQueue());
  const [inputVal, setInputVal] = useState("42");
  const [message, setMessage] = useState("点击「入队」或「出队」开始演示");
  const [lastAction, setLastAction] = useState<"enqueue" | "dequeue" | null>(null);
  const [highlightSlot, setHighlightSlot] = useState<number | null>(null);

  const handleEnqueue = () => {
    const v = parseInt(inputVal);
    if (isNaN(v)) { setMessage("⚠️ 请输入有效数字"); return; }
    if (isFull(queue)) {
      setMessage(`❌ 队列已满！满条件：(tail+1)%${ARR_SIZE} == head → (${queue.tail}+1)%${ARR_SIZE}=${((queue.tail + 1) % ARR_SIZE)} == ${queue.head}`);
      return;
    }
    const next = enqueue(queue, v)!;
    setHighlightSlot(queue.tail);
    setLastAction("enqueue");
    setQueue(next);
    setMessage(
      `✅ 入队 ${v} → 写入槽 [${queue.tail}]，tail: ${queue.tail} → ${next.tail} (=(${queue.tail}+1)%${ARR_SIZE})`
    );
    setInputVal(String(Math.floor(Math.random() * 90) + 10));
    setTimeout(() => setHighlightSlot(null), 800);
  };

  const handleDequeue = () => {
    if (isEmpty(queue)) {
      setMessage(`❌ 队列为空！空条件：head == tail → ${queue.head} == ${queue.tail}`);
      return;
    }
    const res = dequeue(queue)!;
    setHighlightSlot(queue.head);
    setLastAction("dequeue");
    setQueue(res.state);
    setMessage(
      `✅ 出队 → 读取槽 [${queue.head}] = ${res.val}，head: ${queue.head} → ${res.state.head} (=(${queue.head}+1)%${ARR_SIZE})`
    );
    setTimeout(() => setHighlightSlot(null), 800);
  };

  const handleReset = () => {
    setQueue(initQueue());
    setMessage("已重置队列");
    setLastAction(null);
    setHighlightSlot(null);
  };

  // SVG 圆形绘制参数
  const cx = 160, cy = 160, r = 110, svgSize = 320;
  const slotR = 24;

  const full = isFull(queue);
  const empty = isEmpty(queue);
  const size = queueSize(queue);

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h3 className="text-base font-bold text-text-primary">
          🔁 循环队列可视化
        </h3>
        <div className="flex gap-2 text-xs">
          <span className={`px-2 py-1 rounded border ${full ? "border-red-500 text-red-400" : "border-border-subtle text-text-tertiary"}`}>
            {full ? "🔴 已满" : `容量 ${size}/${CAPACITY}`}
          </span>
          <span className={`px-2 py-1 rounded border ${empty ? "border-amber-500 text-amber-400" : "border-border-subtle text-text-tertiary"}`}>
            {empty ? "🟡 空队列" : "有元素"}
          </span>
        </div>
      </div>

      {/* 公式说明 */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-bg-tertiary rounded p-2 border border-border-subtle">
          <span className="text-text-tertiary">空条件：</span>
          <span className={`font-semibold ${empty ? "text-amber-400" : "text-text-secondary"}`}>
            head({queue.head}) == tail({queue.tail}) → {empty ? "True ✓" : "False"}
          </span>
        </div>
        <div className="bg-bg-tertiary rounded p-2 border border-border-subtle">
          <span className="text-text-tertiary">满条件：</span>
          <span className={`font-semibold ${full ? "text-red-400" : "text-text-secondary"}`}>
            (tail+1)%{ARR_SIZE}({(queue.tail + 1) % ARR_SIZE}) == head({queue.head}) → {full ? "True ✓" : "False"}
          </span>
        </div>
      </div>

      {/* SVG 圆形队列 */}
      <div className="flex justify-center">
        <svg width={svgSize} height={svgSize} className="select-none">
          {/* 圆环背景 */}
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-border-subtle)" strokeWidth={2} strokeDasharray="4 4" />

          {/* 空槽提示 */}
          {Array.from({ length: ARR_SIZE }).map((_, i) => {
            // 计算槽是否在有效队列范围内（head 到 tail 之间）
            const inQueue = !empty && (
              queue.head <= queue.tail
                ? i >= queue.head && i < queue.tail
                : i >= queue.head || i < queue.tail
            );
            const pos = slotPos(i, ARR_SIZE, cx, cy, r);
            const isHead = i === queue.head;
            const isTail = i === queue.tail;
            const isHighlight = i === highlightSlot;
            const val = queue.data[i];

            let fill = "var(--color-bg-secondary)";
            let stroke = "var(--color-border-subtle)";
            let strokeWidth = 1.5;
            if (isHighlight) { fill = lastAction === "enqueue" ? "#22c55e33" : "#ef444433"; stroke = lastAction === "enqueue" ? "#22c55e" : "#ef4444"; strokeWidth = 2.5; }
            else if (inQueue) { fill = "#3b82f620"; stroke = "#3b82f6"; strokeWidth = 2; }
            else if (isTail && !empty) { fill = "var(--color-bg-tertiary)"; }

            return (
              <g key={i}>
                <circle
                  cx={pos.x} cy={pos.y} r={slotR}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                  className="transition-all duration-300"
                />
                {/* 槽内的值 */}
                <text x={pos.x} y={pos.y + 1} textAnchor="middle" dominantBaseline="middle"
                  fontSize={val !== null ? 13 : 10}
                  fontFamily="monospace"
                  fill={val !== null ? (inQueue ? "#93c5fd" : "var(--color-text-tertiary)") : "var(--color-text-tertiary)"}
                  fontWeight={val !== null ? "bold" : "normal"}
                >
                  {val !== null ? val : "·"}
                </text>
                {/* 下标 */}
                <text x={pos.x} y={pos.y + slotR + 12} textAnchor="middle"
                  fontSize={9} fill="var(--color-text-tertiary)" fontFamily="monospace">
                  [{i}]
                </text>
                {/* head/tail 标签 */}
                {(isHead || isTail) && (
                  <text
                    x={pos.x + (isTail && isHead ? 28 : 0) + (!isTail && isHead ? -36 : 0) + (isTail && !isHead ? 36 : 0)}
                    y={pos.y}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={10}
                    fontFamily="monospace"
                    fontWeight="bold"
                    fill={isHead && isTail ? "#a78bfa" : isHead ? "#f59e0b" : "#22c55e"}
                  >
                    {isHead && isTail ? "H=T" : isHead ? "H" : "T"}
                  </text>
                )}
              </g>
            );
          })}

          {/* 中央状态显示 */}
          <text x={cx} y={cy - 12} textAnchor="middle" fontSize={13} fontFamily="monospace"
            fill={empty ? "#f59e0b" : full ? "#ef4444" : "#93c5fd"} fontWeight="bold">
            {empty ? "EMPTY" : full ? "FULL" : `size=${size}`}
          </text>
          <text x={cx} y={cy + 10} textAnchor="middle" fontSize={10} fontFamily="monospace"
            fill="var(--color-text-tertiary)">
            cap={CAPACITY}
          </text>
          <text x={cx} y={cy + 26} textAnchor="middle" fontSize={10} fontFamily="monospace"
            fill="var(--color-text-tertiary)">
            arr_size={ARR_SIZE}
          </text>
        </svg>
      </div>

      {/* 图例 */}
      <div className="flex gap-4 text-xs text-text-secondary justify-center flex-wrap">
        <span><span className="inline-block w-3 h-3 rounded-full bg-amber-400 mr-1" />H = head（出队端）</span>
        <span><span className="inline-block w-3 h-3 rounded-full bg-green-500 mr-1" />T = tail（入队端）</span>
        <span><span className="inline-block w-3 h-3 rounded-full border-2 border-blue-400 mr-1" />队列中的元素</span>
      </div>

      {/* 消息 */}
      <div className="bg-bg-tertiary border border-border-subtle rounded-lg px-4 py-2 text-xs text-text-primary min-h-[42px] flex items-center">
        {message}
      </div>

      {/* 操作区 */}
      <div className="flex flex-wrap gap-2 items-center justify-center">
        <input
          type="number"
          value={inputVal}
          onChange={(e) => setInputVal(e.target.value)}
          className="w-20 px-2 py-2 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-sm focus:outline-none focus:border-blue-400"
          placeholder="值"
        />
        <button
          onClick={handleEnqueue}
          disabled={full}
          className="px-4 py-2 rounded-lg bg-green-600 text-white text-sm hover:bg-green-700 disabled:opacity-40 transition-colors"
        >
          ＋ 入队
        </button>
        <button
          onClick={handleDequeue}
          disabled={empty}
          className="px-4 py-2 rounded-lg bg-red-600 text-white text-sm hover:bg-red-700 disabled:opacity-40 transition-colors"
        >
          − 出队
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-sm transition-colors"
        >
          ↩ 重置
        </button>
      </div>

      {/* 线性数组视图 */}
      <div>
        <div className="text-xs text-text-tertiary mb-2">底层数组（线性视图）：</div>
        <div className="flex gap-1">
          {queue.data.map((v, i) => {
            const isHead = i === queue.head;
            const isTail = i === queue.tail;
            return (
              <div key={i} className="flex flex-col items-center flex-1">
                <div className={`w-full rounded text-center py-1 text-xs border transition-all ${
                  v !== null ? "bg-blue-600/20 border-blue-500 text-blue-200 font-bold"
                  : "bg-bg-tertiary border-border-subtle text-text-tertiary"
                }`}>
                  {v !== null ? v : "∅"}
                </div>
                <div className="text-[9px] text-text-tertiary">[{i}]</div>
                <div className="text-[9px] font-bold">
                  {isHead && isTail ? <span className="text-purple-400">H=T</span>
                  : isHead ? <span className="text-amber-400">H</span>
                  : isTail ? <span className="text-green-400">T</span>
                  : null}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
