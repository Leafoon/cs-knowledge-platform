"use client";
import React, { useState } from "react";

/* ─── Complexity data ────────────────────────────────────────────────────── */

type Speed = "fast" | "medium" | "slow" | "varies";

interface ComplexityCell {
  label: string;      // e.g. "O(1)"
  speed: Speed;
  note?: string;      // brief explanation
}

interface Operation {
  name: string;
  desc: string;
  detail: string;     // expanded explanation
  matrix: ComplexityCell;
  listVec: ComplexityCell;
  listHash: ComplexityCell;
  winnerFor: "matrix" | "list" | "tie";  // which is faster for this op
}

const OPERATIONS: Operation[] = [
  {
    name: "初始化（空图）",
    desc: "创建一个 V 个顶点的空图",
    detail: "邻接矩阵需要分配并初始化整个 V×V 数组（全部填 0 或 ∞），即使没有任何边，也要花费 O(V²) 时间和空间。邻接表只需初始化 V 个空列表，O(V) 即可完成。",
    matrix:   { label: "O(V²)", speed: "slow",   note: "全部填 0" },
    listVec:  { label: "O(V)",  speed: "fast",   note: "V 个空列表" },
    listHash: { label: "O(V)",  speed: "fast",   note: "V 个空集合" },
    winnerFor: "list",
  },
  {
    name: "添加一条边",
    desc: "加入边 (u, v, w)",
    detail: "邻接矩阵只需设置 mat[u][v]=w（有向图）或同时设置 mat[v][u]=w（无向图），均为 O(1)。邻接表需要向 adj[u] 追加元素，动态数组均摊 O(1)，哈希集 O(1) 期望。",
    matrix:   { label: "O(1)",  speed: "fast",   note: "直接写格子" },
    listVec:  { label: "O(1)₊", speed: "fast",   note: "追加（均摊）" },
    listHash: { label: "O(1)",  speed: "fast",   note: "哈希插入" },
    winnerFor: "tie",
  },
  {
    name: "删除一条边",
    desc: "移除边 (u, v)",
    detail: "邻接矩阵只需将 mat[u][v] 置 0，O(1)。但邻接表（向量版）需要线性扫描 adj[u] 找到 v 再删除，O(deg(u))。哈希集版本可以 O(1) 期望删除。",
    matrix:   { label: "O(1)",      speed: "fast",   note: "置 0 即可" },
    listVec:  { label: "O(deg(u))", speed: "medium", note: "需线性查找" },
    listHash: { label: "O(1)",      speed: "fast",   note: "哈希删除" },
    winnerFor: "matrix",
  },
  {
    name: "判断边是否存在",
    desc: "查询 (u, v) 是否有边",
    detail: "这是邻接矩阵最大的优势：直接读取 mat[u][v]，O(1) 常数时间。邻接表（向量版）需要扫描 adj[u] 整个列表，最坏 O(deg(u))；哈希集版本也是 O(1) 期望，但常数因子比矩阵大。",
    matrix:   { label: "O(1)",      speed: "fast",   note: "★ 最大优势" },
    listVec:  { label: "O(deg(u))", speed: "medium", note: "扫描邻居列表" },
    listHash: { label: "O(1)",      speed: "fast",   note: "哈希查询" },
    winnerFor: "matrix",
  },
  {
    name: "遍历顶点 u 的邻居",
    desc: "枚举所有与 u 相邻的顶点",
    detail: "这是邻接表最大的优势：只需遍历 adj[u].size() = deg(u) 个元素，时间 O(deg(u))，与总顶点数 V 无关。而邻接矩阵必须扫描整行，需要 O(V) 时间，即使这行中大部分都是 0（即稀疏图中的大量无边格子）。",
    matrix:   { label: "O(V)",      speed: "slow",   note: "扫整行（含大量 0）" },
    listVec:  { label: "O(deg(u))", speed: "fast",   note: "★ 最大优势" },
    listHash: { label: "O(deg(u))", speed: "fast",   note: "迭代集合" },
    winnerFor: "list",
  },
  {
    name: "遍历所有边",
    desc: "枚举图中全部 E 条边",
    detail: "BFS/DFS 等图遍历算法的时间复杂度 O(V+E) 依赖于此操作的效率。邻接表只需扫描所有邻居列表，总代价 O(V+E)。邻接矩阵必须扫描全部 V² 格子，代价 O(V²)——对稀疏图而言，V² 远大于 V+E，这是矩阵的致命弱点。",
    matrix:   { label: "O(V²)",   speed: "slow",   note: "扫全矩阵" },
    listVec:  { label: "O(V+E)",  speed: "fast",   note: "★ BFS/DFS 快" },
    listHash: { label: "O(V+E)",  speed: "fast",   note: "遍历所有集合" },
    winnerFor: "list",
  },
  {
    name: "空间占用",
    desc: "存储图结构所需内存",
    detail: "邻接矩阵无论边数多少，始终占用 O(V²) 空间。当图稀疏（E ≪ V²）时，矩阵中绝大部分格子是 0，造成严重浪费。例如 V=10⁵ 时矩阵需要约 40GB，远超内存限制！邻接表只存实际存在的边，占用 O(V+E)。",
    matrix:   { label: "O(V²)",  speed: "slow",   note: "与边数无关" },
    listVec:  { label: "O(V+E)", speed: "fast",   note: "只存实际边" },
    listHash: { label: "O(V+E)", speed: "medium", note: "哈希额外开销" },
    winnerFor: "list",
  },
];

/* ─── Scenario selector ──────────────────────────────────────────────────── */

interface Scenario {
  id: string;
  name: string;
  emoji: string;
  V: number;
  E: number;
  example: string;
  winner: "matrix" | "list";
  reason: string;
}

const SCENARIOS: Scenario[] = [
  {
    id: "sparse",
    name: "稀疏图",
    emoji: "🌐",
    V: 10000,
    E: 30000,
    example: "路网、社交图（平均度 3）",
    winner: "list",
    reason: "E/V² ≈ 0.3%，矩阵 99.7% 空间浪费；遍历邻居用矩阵需 O(V)=10000 次，用邻接表只需 O(3) 次",
  },
  {
    id: "medium",
    name: "中等密度",
    emoji: "🔗",
    V: 1000,
    E: 50000,
    example: "蛋白质相互作用网络（度~100）",
    winner: "list",
    reason: "E/V² = 10%，矩阵可用但邻接表仍更节省；BFS/DFS 时遍历邻居邻接表快 10 倍",
  },
  {
    id: "dense",
    name: "稠密图",
    emoji: "🔷",
    V: 500,
    E: 100000,
    example: "完全图/密集交互图（度~400）",
    winner: "matrix",
    reason: "E/V² = 80%，矩阵空间浪费可接受；频繁判断边是否存在时 O(1) vs O(400) 优势显著",
  },
];

/* ─── Speed color mapping ────────────────────────────────────────────────── */

function speedClass(speed: Speed, isWinner: boolean): string {
  const base = {
    fast:   "bg-emerald-50  dark:bg-emerald-900/25  text-emerald-700  dark:text-emerald-300  border-emerald-200  dark:border-emerald-700",
    medium: "bg-amber-50    dark:bg-amber-900/25    text-amber-700    dark:text-amber-300    border-amber-200    dark:border-amber-700",
    slow:   "bg-rose-50     dark:bg-rose-900/25     text-rose-700     dark:text-rose-300     border-rose-200     dark:border-rose-700",
    varies: "bg-slate-50    dark:bg-slate-800/60    text-slate-600    dark:text-slate-300    border-slate-200    dark:border-slate-700",
  }[speed];
  return base + (isWinner ? " ring-2 ring-offset-1 ring-indigo-400 dark:ring-indigo-500" : "");
}

function speedIcon(speed: Speed): string {
  return { fast: "✓", medium: "~", slow: "✗", varies: "?" }[speed];
}

/* ─── Space Bar ──────────────────────────────────────────────────────────── */

function SpaceBar({ V, E }: { V: number; E: number }) {
  const matrixBytes = V * V * 4;       // int = 4 bytes
  const listBytes = (V + E * 2) * 8;  // pointer + value ~ 8 bytes each
  const maxBytes = matrixBytes;
  const matPct = 100;
  const listPct = Math.round((listBytes / maxBytes) * 100);

  const fmt = (n: number) => {
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
    return `${n} B`;
  };

  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 p-4 space-y-3">
      <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">
        空间占用对比（V={V.toLocaleString()}, E={E.toLocaleString()}）
      </div>

      {[
        { label: "邻接矩阵", bytes: matrixBytes, pct: matPct,  color: "bg-rose-400 dark:bg-rose-500" },
        { label: "邻接表",   bytes: listBytes,   pct: listPct, color: "bg-emerald-400 dark:bg-emerald-500" },
      ].map(({ label, bytes, pct, color }) => (
        <div key={label}>
          <div className="flex justify-between text-[11px] mb-1">
            <span className="font-medium text-slate-600 dark:text-slate-300">{label}</span>
            <span className="font-mono font-bold text-slate-500 dark:text-slate-400">{fmt(bytes)}</span>
          </div>
          <div className="h-5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
            <div
              className={`h-full rounded-full flex items-center pl-2 transition-all duration-500 ${color}`}
              style={{ width: `${Math.max(pct, 3)}%` }}>
              {pct > 15 && (
                <span className="text-[10px] font-bold text-white">{pct}%</span>
              )}
            </div>
          </div>
        </div>
      ))}

      <div className="text-[10px] text-slate-400 dark:text-slate-500">
        邻接表仅占矩阵空间的 <strong className="text-emerald-600 dark:text-emerald-400">{listPct}%</strong>（节省 {100 - listPct}%）
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function AdjMatrixVsListPerf() {
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const [activeOp, setActiveOp] = useState<number | null>(null);

  const scenario = SCENARIOS[scenarioIdx];

  const matWins = OPERATIONS.filter(o => o.winnerFor === "matrix").length;
  const listWins = OPERATIONS.filter(o => o.winnerFor === "list").length;

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-sky-500 via-blue-500 to-indigo-500 p-5">
        <h3 className="text-white font-bold text-lg tracking-tight">邻接矩阵 vs 邻接表——性能全面对比</h3>
        <p className="text-sky-100 text-sm mt-0.5">
          点击操作行查看详细解释 · 切换场景观察空间占用变化
        </p>
      </div>

      {/* ── Scenario selection + Space bar ── */}
      <div className="p-5 border-b border-slate-100 dark:border-slate-800 space-y-4">
        <div className="flex gap-2 flex-wrap">
          {SCENARIOS.map((s, i) => (
            <button key={s.id} onClick={() => setScenarioIdx(i)}
              className={`flex-1 min-w-[100px] rounded-xl px-3 py-2.5 text-left border transition-all ${
                i === scenarioIdx
                  ? "bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 shadow-sm"
                  : "border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800"
              }`}>
              <div className="text-base">{s.emoji}</div>
              <div className={`text-xs font-bold mt-0.5 ${i === scenarioIdx ? "text-blue-700 dark:text-blue-300" : "text-slate-700 dark:text-slate-300"}`}>
                {s.name}
              </div>
              <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">{s.example}</div>
            </button>
          ))}
        </div>

        <SpaceBar V={scenario.V} E={scenario.E} />

        <div className={`rounded-xl border-2 px-4 py-3 text-sm ${
          scenario.winner === "list"
            ? "border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/20"
            : "border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20"
        }`}>
          <span className="font-bold text-slate-700 dark:text-slate-200">
            {scenario.emoji} {scenario.name}推荐：
          </span>
          <span className={`font-bold ml-1 ${
            scenario.winner === "list" ? "text-emerald-700 dark:text-emerald-300" : "text-blue-700 dark:text-blue-300"
          }`}>
            {scenario.winner === "list" ? "邻接表" : "邻接矩阵"}
          </span>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{scenario.reason}</div>
        </div>
      </div>

      {/* ── Operations table ── */}
      <div className="p-5 space-y-3">
        {/* Score header */}
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
            操作复杂度对比（点击行展开说明）
          </div>
          <div className="flex gap-2 text-[10px]">
            <span className="px-2 py-1 rounded-lg bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 font-bold border border-emerald-200 dark:border-emerald-700">
              邻接表优势 {listWins} 项
            </span>
            <span className="px-2 py-1 rounded-lg bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 font-bold border border-blue-200 dark:border-blue-700">
              矩阵优势 {matWins} 项
            </span>
          </div>
        </div>

        {/* Column headers */}
        <div className="grid grid-cols-[minmax(0,1.8fr)_1fr_1fr_1fr] gap-1.5 text-[11px] font-bold text-center px-1">
          <div className="text-left text-slate-400 dark:text-slate-500 uppercase pl-1">操作</div>
          <div className="text-slate-600 dark:text-slate-300">邻接矩阵</div>
          <div className="text-slate-600 dark:text-slate-300">邻接表<br /><span className="font-normal text-[9px]">(vector)</span></div>
          <div className="text-slate-600 dark:text-slate-300">邻接表<br /><span className="font-normal text-[9px]">(hash set)</span></div>
        </div>

        {OPERATIONS.map((op, i) => {
          const isOpen = activeOp === i;
          const cells = [
            { cell: op.matrix,   isWinner: op.winnerFor === "matrix" },
            { cell: op.listVec,  isWinner: op.winnerFor === "list" },
            { cell: op.listHash, isWinner: op.winnerFor === "list" },
          ];
          return (
            <div key={i}>
              <div
                onClick={() => setActiveOp(prev => prev === i ? null : i)}
                className={`grid grid-cols-[minmax(0,1.8fr)_1fr_1fr_1fr] gap-1.5 cursor-pointer rounded-xl transition-all ${
                  isOpen ? "ring-2 ring-indigo-400 dark:ring-indigo-500" : "hover:bg-slate-50 dark:hover:bg-slate-800/40"
                }`}>
                {/* Operation name */}
                <div className={`py-2.5 px-3 rounded-xl flex flex-col justify-center border transition-colors ${
                  isOpen
                    ? "bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800"
                    : "bg-slate-50 dark:bg-slate-800/60 border-slate-200 dark:border-slate-700"
                }`}>
                  <div className={`text-xs font-bold ${isOpen ? "text-indigo-700 dark:text-indigo-300" : "text-slate-700 dark:text-slate-200"}`}>
                    {op.name}
                  </div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">{op.desc}</div>
                </div>

                {/* Complexity cells */}
                {cells.map(({ cell, isWinner }, j) => (
                  <div key={j} className={`py-2.5 px-2 rounded-xl text-center border flex flex-col items-center justify-center gap-0.5 transition-all ${speedClass(cell.speed, isWinner && isOpen)}`}>
                    <div className="flex items-center gap-1 text-xs font-bold font-mono">
                      <span>{speedIcon(cell.speed)}</span>
                      <span>{cell.label}</span>
                    </div>
                    {cell.note && (
                      <div className="text-[9px] opacity-75 leading-tight text-center">{cell.note}</div>
                    )}
                  </div>
                ))}
              </div>

              {/* Detail panel */}
              {isOpen && (
                <div className="mt-1.5 mx-1 rounded-xl bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 p-4">
                  <div className="text-xs font-semibold text-indigo-700 dark:text-indigo-300 mb-1.5">
                    📖 {op.name}——为什么？
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                    {op.detail}
                  </div>
                  <div className={`mt-2 text-[11px] font-semibold ${
                    op.winnerFor === "list"
                      ? "text-emerald-600 dark:text-emerald-400"
                      : op.winnerFor === "matrix"
                      ? "text-blue-600 dark:text-blue-400"
                      : "text-slate-500 dark:text-slate-400"
                  }`}>
                    {op.winnerFor === "list" ? "👉 此操作邻接表更高效"
                     : op.winnerFor === "matrix" ? "👉 此操作邻接矩阵更高效"
                     : "👉 两者同等高效，无明显优劣"}
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {/* Legend */}
        <div className="flex flex-wrap gap-3 pt-2 text-[10px]">
          {[
            { speed: "fast" as Speed, label: "高效（O(1) 或接近常数）" },
            { speed: "medium" as Speed, label: "中等（O(deg) 或 O(V)）" },
            { speed: "slow" as Speed, label: "较慢（O(V²) 或类似）" },
          ].map(({ speed, label }) => (
            <div key={speed} className="flex items-center gap-1.5">
              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold border ${
                speedClass(speed, false)
              }`}>{speedIcon(speed)}</span>
              <span className="text-slate-500 dark:text-slate-400">{label}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5 ml-2">
            <span className="inline-block w-4 h-4 rounded border-2 border-indigo-400 dark:border-indigo-500" />
            <span className="text-slate-500 dark:text-slate-400">展开状态优势高亮</span>
          </div>
        </div>
      </div>
    </div>
  );
}
