"use client";
import { useState } from "react";

interface Timer { id: number; remaining: number; label: string }

export function TimerWheelVisualizer() {
  const slots = 12;
  const [wheel, setWheel] = useState<(Timer | null)[]>(Array(slots).fill(null));
  const [nextId, setNextId] = useState(1);
  const [timeout, setTimeout_] = useState(5);
  const [label, setLabel] = useState("重传定时器");
  const [expired, setExpired] = useState<Timer[]>([]);
  const [tick, setTick] = useState(0);

  const addTimer = () => {
    const pos = (tick + timeout) % slots;
    const newWheel = [...wheel];
    newWheel[pos] = { id: nextId, remaining: timeout, label };
    setWheel(newWheel);
    setNextId((n) => n + 1);
  };

  const step = () => {
    const newTick = (tick + 1) % slots;
    setTick(newTick);
    const newWheel = [...wheel];
    const current = newWheel[newTick];
    if (current) {
      setExpired((e) => [...e, current]);
      newWheel[newTick] = null;
    }
    setWheel(newWheel);
  };

  const reset = () => { setWheel(Array(slots).fill(null)); setTick(0); setExpired([]); setNextId(1); };

  const R = 100; const cx = 120; const cy = 120;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">定时器轮可视化 (Timer Wheel)</h3>
      <div className="flex gap-3 mb-4">
        <div>
          <label className="text-text-muted text-xs block mb-1">超时 (格数)</label>
          <input type="number" min="1" max={slots} value={timeout} onChange={(e) => setTimeout_(Number(e.target.value))}
            className="w-16 px-2 py-1 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm" />
        </div>
        <div>
          <label className="text-text-muted text-xs block mb-1">标签</label>
          <select value={label} onChange={(e) => setLabel(e.target.value)}
            className="px-2 py-1 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm">
            <option>重传定时器</option><option>保活定时器</option><option>TIME_WAIT</option><option>坚持定时器</option>
          </select>
        </div>
        <button onClick={addTimer} className="self-end px-4 py-1 rounded bg-blue-500 text-white text-sm hover:bg-blue-600">添加</button>
        <button onClick={step} className="self-end px-4 py-1 rounded bg-green-500 text-white text-sm hover:bg-green-600">tick →</button>
        <button onClick={reset} className="self-end px-3 py-1 rounded border border-border-subtle text-text-muted text-sm">重置</button>
      </div>
      <div className="flex flex-col sm:flex-row gap-4 items-start">
        <svg viewBox={`0 0 ${cx * 2} ${cy * 2}`} className="w-60 h-60 flex-shrink-0">
          <circle cx={cx} cy={cy} r={R} fill="none" stroke="currentColor" strokeWidth="1" className="text-text-muted" />
          {Array.from({ length: slots }, (_, i) => {
            const angle = (i / slots) * Math.PI * 2 - Math.PI / 2;
            const x = cx + Math.cos(angle) * R;
            const y = cy + Math.sin(angle) * R;
            const isCurrent = i === tick;
            const hasTimer = wheel[i] !== null;
            return (
              <g key={i}>
                <circle cx={x} cy={y} r={16} fill={isCurrent ? "rgba(59,130,246,0.3)" : hasTimer ? "rgba(234,179,8,0.2)" : "rgba(156,163,175,0.1)"}
                  stroke={isCurrent ? "#60a5fa" : hasTimer ? "#eab308" : "#6b7280"} strokeWidth={isCurrent ? 2 : 1} />
                <text x={x} y={y - 4} textAnchor="middle" className="fill-text-primary text-[9px] font-mono">{i}</text>
                {hasTimer && <text x={x} y={y + 6} textAnchor="middle" className="fill-yellow-400 text-[7px]">T{wheel[i]!.id}</text>}
              </g>
            );
          })}
          <circle cx={cx} cy={cy} r={3} fill="#60a5fa" />
          {(() => {
            const angle = (tick / slots) * Math.PI * 2 - Math.PI / 2;
            return <line x1={cx} y1={cy} x2={cx + Math.cos(angle) * (R - 20)} y2={cy + Math.sin(angle) * (R - 20)} stroke="#60a5fa" strokeWidth="2" />;
          })()}
        </svg>
        <div className="flex-1">
          <div className="p-3 rounded bg-bg-primary border border-border-subtle mb-2">
            <span className="text-text-secondary text-xs">当前位置: {tick}/{slots - 1}</span>
          </div>
          {expired.length > 0 && (
            <div className="p-3 rounded bg-red-500/10 border border-red-400/30">
              <h4 className="text-red-400 text-xs font-medium mb-1">已过期定时器</h4>
              {expired.map((e) => <p key={e.id} className="text-text-muted text-xs">T{e.id} ({e.label})</p>)}
            </div>
          )}
        </div>
      </div>
      <p className="text-text-muted text-xs mt-3">定时器轮通过环形数组实现 O(1) 定时器插入和过期检测，Linux 内核广泛使用。</p>
    </div>
  );
}
export default TimerWheelVisualizer;
