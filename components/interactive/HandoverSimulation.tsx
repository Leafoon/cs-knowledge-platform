"use client";
import { useState } from "react";

interface Cell {
  id: number;
  x: number;
  y: number;
  radius: number;
  color: string;
  label: string;
}

const CELLS: Cell[] = [
  { id: 1, x: 100, y: 100, radius: 70, color: "#3b82f6", label: "基站A" },
  { id: 2, x: 250, y: 100, radius: 70, color: "#22c55e", label: "基站B" },
  { id: 3, x: 400, y: 100, radius: 70, color: "#a855f7", label: "基站C" },
  { id: 4, x: 175, y: 230, radius: 70, color: "#f59e0b", label: "基站D" },
  { id: 5, x: 325, y: 230, radius: 70, color: "#ef4444", label: "基站E" },
];

export function HandoverSimulation() {
  const [userX, setUserX] = useState(100);
  const [userY, setUserY] = useState(100);
  const [connectedTo, setConnectedTo] = useState(1);
  const [handoverCount, setHandoverCount] = useState(0);
  const [history, setHistory] = useState<string[]>([]);

  const getDist = (c: Cell) => Math.sqrt((userX - c.x) ** 2 + (userY - c.y) ** 2);
  const getSignal = (c: Cell) => Math.max(0, 100 - (getDist(c) / c.radius) * 100);

  const findBest = () => {
    let best = CELLS[0];
    let bestSignal = 0;
    for (const cell of CELLS) {
      const sig = getSignal(cell);
      if (sig > bestSignal) { bestSignal = sig; best = cell; }
    }
    return best;
  };

  const handleMove = (x: number, y: number) => {
    setUserX(x);
    setUserY(y);
    const best = findBest();
    if (best.id !== connectedTo && getSignal(best) > 30) {
      setConnectedTo(best.id);
      setHandoverCount(handoverCount + 1);
      setHistory([...history, `切换: ${CELLS.find((c) => c.id === connectedTo)?.label} → ${best.label}`]);
    }
  };

  const connectedCell = CELLS.find((c) => c.id === connectedTo)!;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">基站间移动切换仿真</h3>
      <svg width={500} height={320} className="mb-4 bg-bg-muted rounded-lg" style={{ cursor: "crosshair" }}
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          handleMove(e.clientX - rect.left, e.clientY - rect.top);
        }}>
        {CELLS.map((c) => (
          <g key={c.id}>
            <circle cx={c.x} cy={c.y} r={c.radius} fill={c.color} opacity={0.15} stroke={c.color} strokeWidth={c.id === connectedTo ? 3 : 1} />
            <text x={c.x} y={c.y - c.radius - 5} textAnchor="middle" fill={c.color} fontSize={12} fontWeight="bold">{c.label}</text>
          </g>
        ))}
        <circle cx={userX} cy={userY} r={8} fill="white" stroke="#1f2937" strokeWidth={2} />
        <text x={userX} y={userY - 15} textAnchor="middle" fill="#1f2937" fontSize={11}>用户</text>
        <line x1={userX} y1={userY} x2={connectedCell.x} y2={connectedCell.y} stroke={connectedCell.color} strokeWidth={2} strokeDasharray="4" />
      </svg>
      <div className="grid grid-cols-5 gap-2 mb-4">
        {CELLS.map((c) => {
          const sig = getSignal(c);
          return (
            <div key={c.id} className={`p-2 rounded text-xs text-center ${c.id === connectedTo ? "ring-2 ring-blue-500 bg-bg-muted" : "bg-bg-muted"}`}>
              <div className="font-bold" style={{ color: c.color }}>{c.label}</div>
              <div className="text-text-secondary">{sig.toFixed(0)}%</div>
            </div>
          );
        })}
      </div>
      <div className="flex gap-4 text-sm text-text-secondary">
        <span>当前连接: <strong className="text-text-primary">{connectedCell.label}</strong></span>
        <span>切换次数: <strong className="text-text-primary">{handoverCount}</strong></span>
      </div>
      {history.length > 0 && (
        <div className="mt-2 text-xs text-text-secondary">
          {history.slice(-3).map((h, i) => <div key={i}>{h}</div>)}
        </div>
      )}
      <div className="text-xs text-text-secondary mt-3">
        移动切换(Handover):当用户移动到当前基站覆盖边缘时,自动切换到信号更强的相邻基站,保持通信连续性。
      </div>
    </div>
  );
}

export default HandoverSimulation;
