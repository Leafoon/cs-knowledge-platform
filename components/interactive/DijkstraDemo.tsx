"use client";
import { useState } from "react";

interface Node {
  id: string;
  x: number;
  y: number;
  dist: number;
  visited: boolean;
  prev: string | null;
}

const INITIAL_NODES: Node[] = [
  { id: "A", x: 60, y: 40, dist: 0, visited: false, prev: null },
  { id: "B", x: 200, y: 40, dist: Infinity, visited: false, prev: null },
  { id: "C", x: 340, y: 40, dist: Infinity, visited: false, prev: null },
  { id: "D", x: 60, y: 160, dist: Infinity, visited: false, prev: null },
  { id: "E", x: 200, y: 160, dist: Infinity, visited: false, prev: null },
  { id: "F", x: 340, y: 160, dist: Infinity, visited: false, prev: null },
];

const EDGES: [string, string, number][] = [
  ["A", "B", 4], ["A", "D", 2], ["B", "C", 3], ["B", "E", 1],
  ["C", "F", 5], ["D", "E", 6], ["E", "F", 8],
];

export function DijkstraDemo() {
  const [nodes, setNodes] = useState<Node[]>(INITIAL_NODES.map((n) => ({ ...n })));
  const [step, setStep] = useState(0);
  const [done, setDone] = useState(false);

  const getNeighbors = (id: string) =>
    EDGES.filter((e) => e[0] === id || e[1] === id).map((e) => ({
      neighbor: e[0] === id ? e[1] : e[0],
      weight: e[2],
    }));

  const reset = () => {
    setNodes(INITIAL_NODES.map((n) => ({ ...n })));
    setStep(0);
    setDone(false);
  };

  const doStep = () => {
    const current = [...nodes].sort((a, b) => a.dist - b.dist).find((n) => !n.visited);
    if (!current || current.dist === Infinity) { setDone(true); return; }

    const updated = nodes.map((n) => ({ ...n }));
    const cur = updated.find((n) => n.id === current.id)!;
    cur.visited = true;

    const neighbors = getNeighbors(cur.id);
    for (const { neighbor, weight } of neighbors) {
      const nb = updated.find((n) => n.id === neighbor)!;
      if (!nb.visited && cur.dist + weight < nb.dist) {
        nb.dist = cur.dist + weight;
        nb.prev = cur.id;
      }
    }
    setNodes(updated);
    setStep(step + 1);
    if (updated.every((n) => n.visited || n.dist === Infinity)) setDone(true);
  };

  const nodeMap = Object.fromEntries(nodes.map((n) => [n.id, n]));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">Dijkstra最短路径交互演示</h3>
      <svg width={420} height={220} className="mb-4 bg-bg-muted rounded-lg">
        {EDGES.map(([a, b, w]) => {
          const na = nodeMap[a], nb = nodeMap[b];
          return (
            <g key={`${a}-${b}`}>
              <line x1={na.x} y1={na.y} x2={nb.x} y2={nb.y} stroke="#6b7280" strokeWidth={1.5} />
              <text x={(na.x + nb.x) / 2} y={(na.y + nb.y) / 2 - 8} textAnchor="middle" fill="#9ca3af" fontSize={12}>{w}</text>
            </g>
          );
        })}
        {nodes.map((n) => (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r={18} fill={n.visited ? "#22c55e" : n.dist === Infinity ? "#374151" : "#3b82f6"} stroke="#1f2937" strokeWidth={2} />
            <text x={n.x} y={n.y - 2} textAnchor="middle" fill="white" fontWeight="bold" fontSize={14}>{n.id}</text>
            <text x={n.x} y={n.y + 12} textAnchor="middle" fill="#d1d5db" fontSize={9}>
              {n.dist === Infinity ? "∞" : n.dist}
            </text>
          </g>
        ))}
      </svg>
      <div className="flex gap-3 mb-4">
        <button onClick={doStep} disabled={done} className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 hover:bg-blue-600 text-sm">下一步</button>
        <button onClick={reset} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">重置</button>
        <span className="text-sm text-text-secondary self-center">步骤: {step} {done && "✅ 已完成"}</span>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs text-text-secondary">
        {nodes.map((n) => (
          <div key={n.id} className="p-2 bg-bg-muted rounded">
            <span className="font-mono text-text-primary">{n.id}</span>: 距离={n.dist === Infinity ? "∞" : n.dist}, 前驱={n.prev ?? "无"}
            {n.visited && " ✓"}
          </div>
        ))}
      </div>
    </div>
  );
}

export default DijkstraDemo;
