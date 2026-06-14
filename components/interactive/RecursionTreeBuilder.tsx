'use client';

import { useState, useCallback } from 'react';

interface TreeNode {
  depth: number;
  index: number;   // position in the level
  n: number;       // problem size at this node
  cost: number;    // f(n) at this node
}

const PRESETS = [
  { label: '归并排序', a: 2, b: 2, fExp: 1, fLabel: 'n' },
  { label: '二分搜索', a: 1, b: 2, fExp: 0, fLabel: '1' },
  { label: 'Strassen', a: 7, b: 2, fExp: 2, fLabel: 'n²' },
  { label: '情形1示例', a: 4, b: 2, fExp: 1, fLabel: 'n' },
  { label: '情形3示例', a: 2, b: 2, fExp: 2, fLabel: 'n²' },
];

function nodeCost(n: number, fExp: number): number {
  if (fExp === 0) return 1;
  return Math.pow(n, fExp);
}

function buildTree(a: number, b: number, fExp: number, maxDepth: number): TreeNode[][] {
  // Returns array of levels (each level is array of nodes)
  const levels: TreeNode[][] = [];
  let currentLevel: TreeNode[] = [{ depth: 0, index: 0, n: 64, cost: nodeCost(64, fExp) }];
  levels.push(currentLevel);

  for (let d = 1; d <= maxDepth; d++) {
    const nextLevel: TreeNode[] = [];
    currentLevel.forEach((node, idx) => {
      if (node.n <= 1) return;
      for (let c = 0; c < a; c++) {
        const childN = Math.max(1, Math.floor(node.n / b));
        nextLevel.push({
          depth: d,
          index: idx * a + c,
          n: childN,
          cost: nodeCost(childN, fExp),
        });
      }
    });
    if (nextLevel.length === 0) break;
    levels.push(nextLevel);
    currentLevel = nextLevel;
  }
  return levels;
}

const NODE_R = 18;
const LEVEL_H = 68;
const SVG_W = 580;

export default function RecursionTreeBuilder() {
  const [preset, setPreset] = useState(0);
  const [revealedLevels, setRevealedLevels] = useState(1);

  const { a, b, fExp, fLabel } = PRESETS[preset];

  // Compute tree
  const maxDepth = Math.min(6, Math.ceil(Math.log(64) / Math.log(b)));
  const tree = buildTree(a, b, fExp, maxDepth);
  const shownLevels = tree.slice(0, revealedLevels);

  const levelCosts = tree.map(level => level.reduce((s, n) => s + n.cost, 0));

  // Layout
  const maxNodesInLevel = Math.max(...shownLevels.map(l => l.length));
  const svgH = shownLevels.length * LEVEL_H + 60;

  const getX = useCallback((index: number, totalInLevel: number) => {
    const usable = SVG_W - 60;
    if (totalInLevel === 1) return SVG_W / 2;
    return 30 + (index / (totalInLevel - 1)) * usable;
  }, []);

  const getY = (depth: number) => 40 + depth * LEVEL_H;

  const applyPreset = (i: number) => {
    setPreset(i);
    setRevealedLevels(1);
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-4 space-y-4">
      <h3 className="text-base font-semibold text-text-primary">🌳 递归树展开可视化</h3>
      <p className="text-xs text-text-secondary">
        展示 <span className="font-mono">T(n) = {a}T(n/{b}) + f(n)</span>（其中 f(n) = {fLabel}）的递归树结构，逐层展开。
      </p>

      {/* Presets */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p, i) => (
          <button key={p.label} onClick={() => applyPreset(i)}
            className={`px-3 py-1 rounded-full text-xs border transition-colors ${i === preset
              ? 'border-blue-400 bg-blue-400/10 text-blue-400'
              : 'border-border-subtle text-text-secondary hover:border-blue-400 hover:text-blue-400'
            }`}>
            {p.label}
          </button>
        ))}
      </div>

      {/* SVG tree */}
      <div className="rounded-lg bg-bg-primary border border-border-subtle overflow-x-auto">
        <svg viewBox={`0 0 ${SVG_W} ${svgH}`} className="w-full min-w-[400px]">
          {/* Edges */}
          {shownLevels.slice(1).map((level, di) => {
            const parentLevel = shownLevels[di];
            return level.map((node) => {
              const parentIdx = Math.floor(node.index / a);
              const parent = parentLevel[parentIdx];
              if (!parent) return null;
              const px = getX(parent.index, parentLevel.length);
              const py = getY(di);
              const cx = getX(node.index, level.length);
              const cy = getY(di + 1);
              return (
                <line key={`e-${di}-${node.index}`}
                  x1={px} y1={py + NODE_R} x2={cx} y2={cy - NODE_R}
                  stroke="currentColor" strokeOpacity={0.2} strokeWidth={1} />
              );
            });
          })}

          {/* Nodes */}
          {shownLevels.map((level, di) =>
            level.map((node) => {
              const x = getX(node.index, level.length);
              const y = getY(di);
              const tooMany = level.length > 12;
              if (tooMany && node.index > 3 && node.index < level.length - 2) {
                if (node.index === 4) return (
                  <text key={`dots-${di}`} x={x} y={y + 5} textAnchor="middle" fontSize={14} fill="currentColor" opacity={0.5}>…</text>
                );
                return null;
              }
              return (
                <g key={`n-${di}-${node.index}`}>
                  <circle cx={x} cy={y} r={NODE_R}
                    fill="#3b82f6" fillOpacity={0.15}
                    stroke="#3b82f6" strokeWidth={1.5} />
                  <text x={x} y={y - 3} textAnchor="middle" dominantBaseline="middle" fontSize={8} fill="#60a5fa">
                    n={node.n}
                  </text>
                  <text x={x} y={y + 9} textAnchor="middle" dominantBaseline="middle" fontSize={8} fill="#60a5fa">
                    {node.cost >= 1000 ? `${(node.cost / 1000).toFixed(0)}K` : node.cost}
                  </text>
                </g>
              );
            })
          )}

          {/* Level cost summary (right side) */}
          {shownLevels.map((_, di) => {
            const cost = levelCosts[di];
            return (
              <g key={`cost-${di}`}>
                <text x={SVG_W - 4} y={getY(di) + 5} textAnchor="end" fontSize={9} fill="#f59e0b" opacity={0.8}>
                  Σ={cost >= 1e6 ? `${(cost / 1e6).toFixed(1)}M` : cost >= 1000 ? `${(cost / 1000).toFixed(0)}K` : cost}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Controls */}
      <div className="flex gap-3 items-center">
        <button
          onClick={() => setRevealedLevels(l => Math.min(l + 1, tree.length))}
          disabled={revealedLevels >= tree.length}
          className="px-4 py-1.5 rounded-lg text-sm bg-blue-500 text-white font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
        >
          展开下一层 →
        </button>
        <button
          onClick={() => setRevealedLevels(1)}
          className="px-4 py-1.5 rounded-lg text-sm border border-border-subtle text-text-secondary hover:border-blue-400 hover:text-blue-400 transition-colors"
        >
          重置
        </button>
        <span className="text-xs text-text-tertiary">第 {revealedLevels - 1} 层 / 共 {tree.length - 1} 层</span>
      </div>

      {/* Level summary table */}
      <div className="rounded-lg border border-border-subtle overflow-hidden text-xs">
        <table className="w-full">
          <thead>
            <tr className="bg-bg-tertiary">
              <th className="px-3 py-2 text-left text-text-tertiary font-medium">层数</th>
              <th className="px-3 py-2 text-left text-text-tertiary font-medium">节点数</th>
              <th className="px-3 py-2 text-left text-text-tertiary font-medium">每节点规模</th>
              <th className="px-3 py-2 text-left text-text-tertiary font-medium">本层总代价</th>
            </tr>
          </thead>
          <tbody>
            {shownLevels.map((level, di) => (
              <tr key={di} className={`border-t border-border-subtle ${di === revealedLevels - 1 ? 'bg-blue-500/10' : ''}`}>
                <td className="px-3 py-1.5 text-text-primary font-mono">{di}</td>
                <td className="px-3 py-1.5 text-text-secondary font-mono">{level.length}（={a}^{di}）</td>
                <td className="px-3 py-1.5 text-text-secondary font-mono">{level[0]?.n ?? 0}</td>
                <td className="px-3 py-1.5 font-mono font-semibold text-amber-400">
                  {levelCosts[di] >= 1e6 ? `${(levelCosts[di] / 1e6).toFixed(2)}M` : levelCosts[di]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
