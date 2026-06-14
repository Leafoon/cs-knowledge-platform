"use client";
import React, { useState } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
// Relation code (relative to parent): 0 = same kind, 1 = I eat parent, 2 = parent eats me
type Rel = 0 | 1 | 2;

interface DSUNode {
  parent: number;
  rel: Rel; // relation to parent; if parent==self, rel=0 (root)
}

// ─── DSU operations ────────────────────────────────────────────────────────────
function makeSet(n: number): DSUNode[] {
  return Array.from({ length: n }, (_, i) => ({ parent: i, rel: 0 }));
}

// Returns { root, relToRoot }
// relToRoot: relation of node to root (0=same, 1=node eats root, 2=root eats node)
function find(dsu: DSUNode[], x: number): { root: number; rel: Rel } {
  if (dsu[x].parent === x) return { root: x, rel: 0 };
  const { root, rel: parentRel } = find(dsu, dsu[x].parent);
  // Compose: node→parent→root
  // rel(node, root) = compose(rel(node, parent), rel(parent, root))
  const composed = composeRel(dsu[x].rel, parentRel);
  return { root, rel: composed };
}

function composeRel(a: Rel, b: Rel): Rel {
  // a = relation of node to parent; b = relation of parent to root
  // We want: relation of node to root
  // 0+x = x, 1+1=2, 1+2=0, 2+1=0, 2+2=1
  return ((a + b) % 3) as Rel;
}

// Union: assert that x and y have relationship xToY (0=same, 1=x eats y, 2=y eats x)
function unite(dsu: DSUNode[], x: number, y: number, xToY: Rel): { dsu: DSUNode[]; ok: boolean; contradiction?: string } {
  const d = dsu.map(n => ({ ...n }));
  const { root: rx, rel: rxRel } = find(d, x);
  const { root: ry, rel: ryRel } = find(d, y);

  if (rx === ry) {
    // Check consistency: relToRoot(x) and relToRoot(y) must match xToY
    // rel(x, y) = compose(rel(x, root), inverse(rel(y, root)))
    const actualXtoY: Rel = ((rxRel - ryRel + 3) % 3) as Rel;
    if (actualXtoY === xToY) return { dsu: d, ok: true };
    return { dsu, ok: false, contradiction: `矛盾！已知 ${ANIMAL_NAMES[x]} 与 ${ANIMAL_NAMES[y]} 的关系为"${REL_NAMES[actualXtoY]}"，但断言"${REL_NAMES[xToY]}"与之冲突。` };
  }

  // Merge: attach ry under rx
  // We need: rel(ry, rx) such that compose(rel(x, rx), rel(ry, rx)_inv) = xToY
  // rel(x, y) = rxRel - ryRel (mod 3) after merge should equal xToY
  // So rel(ry, rx) = rxRel - xToY + ... actually: 
  // We want rel(y,x) = inverse(xToY)
  // rel(y, rx) = compose(ryRel, ???) ... let's compute:
  // rel(x, root_x) = rxRel
  // rel(y, root_x) should ultimately be: inverse(xToY) XOR rxRel ... 
  // Simpler: rel(y, rx) = (rxRel - xToY + ryRel... )
  // The formula: new_rel_ry = (rxRel - xToY - ryRel + 6) % 3
  const newRel: Rel = (((rxRel - xToY) % 3 + 3) % 3) as Rel;
  // Wait, let me reason more carefully:
  // After merge, ry.parent = rx, and ry.rel = newRel
  // find(y) = find(ry) since ry is root of y's tree.
  // find(y).rel in the new tree = compose(ryRel, newRel) [ryRel = rel of y to ry in original]
  // We want find(y).rel = rel of y to rx
  // rel(x, rx) = rxRel
  // We want rel(x, y) = xToY, i.e., rel(y, x) = inverse(xToY) = (3-xToY)%3
  // rel(y, rx) = rel(y, x) + rel(x, rx) = inverse(xToY) + rxRel (mod 3)
  // find(y).rel = compose(ryRel, newRel)
  // So: compose(ryRel, newRel) = (3 - xToY + rxRel) % 3
  // compose(a,b) = (a+b)%3, so ryRel + newRel = (3 - xToY + rxRel) %3
  // newRel = (3 - xToY + rxRel - ryRel + 3) % 3
  const correctNewRel: Rel = (((3 - xToY + rxRel - ryRel) % 3 + 3) % 3) as Rel;
  d[ry].parent = rx;
  d[ry].rel = correctNewRel;

  return { dsu: d, ok: true };
}

// ─── Assertions ───────────────────────────────────────────────────────────────
// N = 5 animals; 0=Rabbit, 1=Fox, 2=Snake, 3=Grass, 4=Mouse
const N_ANIMALS = 5;
const ANIMAL_NAMES = ["🐰 兔", "🦊 狐", "🐍 蛇", "🌿 鼠", "🦅 鹰"]; // 0-4
const REL_NAMES: Record<Rel, string> = { 0: "同类", 1: "前者吃后者", 2: "后者吃前者" };
const REL_COLORS: Record<Rel, string> = {
  0: "#6366f1",     // indigo = same
  1: "#ef4444",     // red = x eats y
  2: "#f59e0b",     // amber = y eats x
};
const REL_ICON: Record<Rel, string> = { 0: "≡", 1: "≻", 2: "≺" };

interface Assertion { x: number; y: number; xToY: Rel; label: string }

const ASSERTION_STEPS: (Assertion & { autoResult?: string })[] = [
  { x: 0, y: 3, xToY: 2, label: "兔(0)和鼠(3)：后者吃前者（🌿鼠 吃 🐰兔？不对，仅演示格式）\n即鼠 吃 兔", autoResult: undefined },
  { x: 1, y: 0, xToY: 1, label: "狐(1) 吃 兔(0)", autoResult: undefined },
  { x: 2, y: 1, xToY: 1, label: "蛇(2) 吃 狐(1)", autoResult: undefined },
  { x: 4, y: 2, xToY: 1, label: "鹰(4) 吃 蛇(2)", autoResult: undefined },
  { x: 0, y: 2, xToY: 1, label: "兔(0) 吃 蛇(2)？（谎言检测！）", autoResult: "矛盾" },
  { x: 3, y: 4, xToY: 0, label: "鼠(3) 与 鹰(4) 同类？（谎言检测！）", autoResult: "矛盾" },
];

// ─── Layout ───────────────────────────────────────────────────────────────────
function computeForestLayout(dsu: DSUNode[], n: number) {
  // Build parent map without path compression (iterate parent chain)
  const parentArr = dsu.map(d => d.parent);
  const children: number[][] = Array.from({ length: n }, () => []);
  const roots: number[] = [];
  for (let i = 0; i < n; i++) {
    if (parentArr[i] === i) roots.push(i);
    else children[parentArr[i]].push(i);
  }

  const W = 480, H = 180;
  const positions: { x: number; y: number }[] = Array(n);
  const totalRoots = roots.length;
  const rootSlotW = W / Math.max(totalRoots, 1);

  function assignPos(node: number, depth: number, slotX: number, slotW: number) {
    positions[node] = { x: slotX + slotW / 2, y: 30 + depth * 70 };
    const ch = children[node];
    if (ch.length === 0) return;
    const childSlotW = slotW / ch.length;
    ch.forEach((c, i) => assignPos(c, depth + 1, slotX + i * childSlotW, childSlotW));
  }
  roots.forEach((r, i) => assignPos(r, 0, i * rootSlotW, rootSlotW));

  return { positions, children, parentArr };
}

// ─── Main Component ────────────────────────────────────────────────────────────
export default function WeightedUnionFind() {
  const [stepIdx, setStepIdx] = useState(-1);
  const [dsuHistory, setDsuHistory] = useState<DSUNode[][]>([makeSet(N_ANIMALS)]);
  const [results, setResults] = useState<Array<{ ok: boolean; msg: string }>>([]);

  const currentDSU = dsuHistory[Math.max(0, stepIdx + 1)] ?? dsuHistory[0];

  const handleNext = () => {
    if (stepIdx >= ASSERTION_STEPS.length - 1) return;
    const nextStep = ASSERTION_STEPS[stepIdx + 1];
    const currentD = dsuHistory[stepIdx + 1] ?? dsuHistory[0];
    const result = unite(currentD, nextStep.x, nextStep.y, nextStep.xToY);

    if (result.ok) {
      setDsuHistory(h => [...h, result.dsu]);
      setResults(r => [...r, { ok: true, msg: `✅ 接受：${nextStep.label.split("\n")[0]}` }]);
    } else {
      setDsuHistory(h => [...h, currentD]); // unchanged
      setResults(r => [...r, { ok: false, msg: `❌ ${result.contradiction}` }]);
    }
    setStepIdx(s => s + 1);
  };

  const handlePrev = () => {
    if (stepIdx < 0) return;
    setDsuHistory(h => h.slice(0, -1));
    setResults(r => r.slice(0, -1));
    setStepIdx(s => s - 1);
  };

  const handleReset = () => {
    setStepIdx(-1);
    setDsuHistory([makeSet(N_ANIMALS)]);
    setResults([]);
  };

  const { positions, children, parentArr } = computeForestLayout(currentDSU, N_ANIMALS);
  const SVG_W = 480, SVG_H = 185;
  const progress = stepIdx < 0 ? 0 : ((stepIdx + 1) / ASSERTION_STEPS.length) * 100;

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-2 mb-1">
        <h3 className="text-base font-bold text-slate-800 dark:text-slate-100">
          🍃 带权并查集 — 食物链关系推断
        </h3>
        <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 font-mono">
          {stepIdx + 1} / {ASSERTION_STEPS.length} 条断言
        </span>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        带权并查集通过<span className="font-semibold text-indigo-500">边的权值（relation）</span>记录节点与父节点的关系。
        FIND 时沿路径累积权值，即可在 O(α) 时间内判断任意两节点关系，并检测矛盾。
      </p>

      {/* Animal legend */}
      <div className="flex flex-wrap gap-2 mb-4">
        {ANIMAL_NAMES.map((name, i) => (
          <div key={i} className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 text-xs">
            <span className="w-5 h-5 rounded-full bg-indigo-500 text-white flex items-center justify-center text-xs font-bold">{i}</span>
            <span className="text-slate-700 dark:text-slate-300">{name}</span>
          </div>
        ))}
      </div>

      {/* Forest SVG */}
      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl overflow-hidden mb-4">
        <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full" style={{ height: SVG_H }}>
          {/* Edges with relation labels */}
          {parentArr.map((p, i) => {
            if (p === i) return null;
            const from = positions[p], to = positions[i];
            const rel = currentDSU[i].rel;
            const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
            return (
              <g key={`e${i}`}>
                <line x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                  stroke={REL_COLORS[rel]} strokeWidth={2} opacity={0.7} />
                <rect x={mx - 12} y={my - 9} width={24} height={17} rx={4}
                  fill={REL_COLORS[rel]} opacity={0.15} />
                <text x={mx} y={my + 1} textAnchor="middle" dominantBaseline="middle"
                  fill={REL_COLORS[rel]} fontSize={10} fontWeight="bold">
                  {REL_ICON[rel]}
                </text>
              </g>
            );
          })}
          {/* Nodes */}
          {Array.from({ length: N_ANIMALS }, (_, i) => {
            const { x, y } = positions[i];
            const isRoot = currentDSU[i].parent === i;
            return (
              <g key={`n${i}`}>
                <circle cx={x} cy={y} r={20}
                  fill={isRoot ? "#10b981" : "#64748b"}
                  stroke={isRoot ? "#6ee7b7" : "none"}
                  strokeWidth={2} opacity={0.9}
                />
                <text x={x} y={y - 3} textAnchor="middle" dominantBaseline="middle"
                  fill="white" fontSize={13}>{ANIMAL_NAMES[i].split(" ")[0]}</text>
                <text x={x} y={y + 9} textAnchor="middle" dominantBaseline="middle"
                  fill="white" fontSize={9} fontFamily="monospace">{i}</text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Relation legend */}
      <div className="flex flex-wrap gap-4 mb-4 text-xs">
        {([0, 1, 2] as Rel[]).map(r => (
          <span key={r} className="flex items-center gap-1.5">
            <span style={{ color: REL_COLORS[r] }} className="font-bold text-sm">{REL_ICON[r]}</span>
            <span className="text-slate-600 dark:text-slate-300">{REL_NAMES[r]}</span>
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" />
          <span className="text-slate-500 dark:text-slate-400">集合根节点</span>
        </span>
      </div>

      {/* Assertion steps panel */}
      <div className="mb-4 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-3">
        <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">断言历史</div>
        <div className="space-y-1.5 max-h-40 overflow-y-auto">
          {results.length === 0 && (
            <div className="text-xs text-slate-400 dark:text-slate-500">尚未添加任何断言，点击「下一步」逐条执行…</div>
          )}
          {results.map((r, i) => (
            <div key={i} className={`flex items-start gap-2 rounded px-2 py-1.5 text-xs ${
              r.ok ? "bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300"
                   : "bg-rose-50 dark:bg-rose-900/20 text-rose-700 dark:text-rose-300"
            }`}>
              <span className="font-mono text-slate-400 dark:text-slate-500 shrink-0"># {i + 1}</span>
              <span>{r.msg}</span>
            </div>
          ))}
        </div>
        {/* Next assertion preview */}
        {stepIdx < ASSERTION_STEPS.length - 1 && (
          <div className="mt-2 rounded px-2 py-1.5 text-xs bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800">
            待执行：断言 #{stepIdx + 2} — {ASSERTION_STEPS[stepIdx + 1].label.split("\n")[0]}
          </div>
        )}
      </div>

      {/* DSU internals */}
      <div className="mb-4 bg-slate-50 dark:bg-slate-800/40 rounded-lg border border-slate-200 dark:border-slate-700 p-3">
        <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">内部 DSU 状态（parent[], rel[]）</div>
        <div className="flex gap-2 flex-wrap">
          {currentDSU.map((node, i) => (
            <div key={i} className="flex flex-col items-center rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 px-2 py-1.5 text-xs min-w-[52px]">
              <span className="text-slate-400 dark:text-slate-500 font-mono">[{i}]</span>
              <span className="font-bold font-mono text-slate-800 dark:text-slate-100">{ANIMAL_NAMES[i].split(" ")[0]}</span>
              <span className="text-slate-400 font-mono mt-0.5">p={node.parent}</span>
              <span style={{ color: REL_COLORS[node.rel] }} className="font-mono font-bold">{REL_ICON[node.rel]}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2">
        <button onClick={handleReset}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400
            text-xs hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          ↺ 重置
        </button>
        <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-indigo-500 rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
        </div>
        <button onClick={handlePrev} disabled={stepIdx < 0}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
            text-xs disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          ◀ 上一步
        </button>
        <button onClick={handleNext} disabled={stepIdx >= ASSERTION_STEPS.length - 1}
          className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-xs
            disabled:opacity-30 hover:bg-indigo-700 transition-colors">
          下一步 ▶
        </button>
      </div>
    </div>
  );
}
