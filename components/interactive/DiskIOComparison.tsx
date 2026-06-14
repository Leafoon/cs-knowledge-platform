"use client";
import React, { useState, useMemo } from "react";

// ─── Model ──────────────────────────────────────────────────────────────────
// BST height ≈ O(log₂ n) worst case balanced
// B-tree height ≈ log_t(ceil((n+1)/2))
function bstHeight(n: number): number {
  return Math.ceil(Math.log2(n + 1));
}
function btreeHeight(n: number, t: number): number {
  if (n <= 1) return 1;
  return Math.ceil(Math.log(Math.ceil((n + 1) / 2)) / Math.log(t)) + 1;
}

// Disk access per operation: BST = height (each node = 1 disk read if not cached)
// B-tree = height (each node fits in one disk block, but bigger → fewer levels)
// Comparison per operation: BST = height comparisons, B-tree = height × t comparisons
function comparisonsPerSearch(n: number, t: number): { bst: number; btree: number } {
  const bh = bstHeight(n);
  const bt = btreeHeight(n, t);
  return { bst: bh, btree: bt * t }; // roughly
}

const N_VALUES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];
const T_OPTIONS = [2, 4, 8, 16, 64, 128, 256, 512];

function fmtNum(n: number, decimals = 0) {
  return n.toLocaleString("zh-CN", { maximumFractionDigits: decimals });
}

// ─── Bar Chart ──────────────────────────────────────────────────────────────
function BarChart({
  data,
  maxVal,
  colors,
  labels,
}: {
  data: { label: string; values: number[] }[];
  maxVal: number;
  colors: string[];
  labels: string[];
}) {
  const BAR_H = 28;
  const GAP = 10;
  const LABEL_W = 140;
  const CHART_W = 360;
  const groupH = data[0].values.length * (BAR_H + 4) + GAP;
  const totalH = data.length * groupH + 40;

  return (
    <svg width={LABEL_W + CHART_W + 50} height={totalH} className="block">
      {data.map((group, gi) => {
        const gy = 20 + gi * groupH;
        return (
          <g key={gi}>
            <text x={LABEL_W - 6} y={gy + groupH / 2} textAnchor="end" fontSize={11} fill="#94a3b8">
              n={group.label}
            </text>
            {group.values.map((v, vi) => {
              const bw = Math.max(2, (v / maxVal) * CHART_W);
              return (
                <g key={vi}>
                  <rect x={LABEL_W} y={gy + vi * (BAR_H + 4)} width={bw} height={BAR_H} rx={4}
                    fill={colors[vi]} opacity={0.85} />
                  <text x={LABEL_W + bw + 5} y={gy + vi * (BAR_H + 4) + BAR_H / 2 + 4}
                    fontSize={11} fill="#e2e8f0">
                    {fmtNum(v)}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
      {/* Legend */}
      {labels.map((l, i) => (
        <g key={i} transform={`translate(${LABEL_W + i * 130}, ${totalH - 20})`}>
          <rect width={14} height={14} rx={3} fill={colors[i]} />
          <text x={18} y={12} fontSize={11} fill="#94a3b8">{l}</text>
        </g>
      ))}
    </svg>
  );
}

// ─── Main ────────────────────────────────────────────────────────────────────
export default function DiskIOComparison() {
  const [t, setT] = useState(16);
  const [metric, setMetric] = useState<"io" | "cmp">("io");

  const tableData = useMemo(() => {
    return N_VALUES.map(n => {
      const bh = bstHeight(n);
      const bt = btreeHeight(n, t);
      const cmpBst = bh;
      const cmpBt = Math.ceil(Math.log2(t)) * bt; // binary search per node ≈ log₂(t)
      return { n, bh, bt, cmpBst, cmpBt };
    });
  }, [t]);

  const maxIO = Math.max(...tableData.map(d => d.bh));
  const maxCmp = Math.max(...tableData.map(d => d.cmpBst));

  const barData = tableData.map(d => ({
    label: d.n >= 1_000_000 ? `${d.n / 1_000_000}M` : d.n >= 1000 ? `${d.n / 1000}K` : String(d.n),
    values: metric === "io" ? [d.bh, d.bt] : [d.cmpBst, d.cmpBt],
  }));

  const colors = ["#ef4444", "#3b82f6"];
  const legends = metric === "io"
    ? ["平衡 BST（磁盘 I/O）", `B 树 t=${t}（磁盘 I/O）`]
    : ["平衡 BST（磁盘读取次数）", `B 树 t=${t}（节点内比较）`];

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-5 my-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-5">
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-sm">最小度数 t =</span>
          <select value={t} onChange={e => setT(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 text-slate-200 text-sm rounded px-2 py-1">
            {T_OPTIONS.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span className="text-slate-500 text-xs">（每节点最多 {2 * t - 1} 个键）</span>
        </div>
        <div className="flex gap-2">
          <button onClick={() => setMetric("io")}
            className={`px-3 py-1 rounded text-sm ${metric === "io" ? "bg-rose-700 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"}`}>
            磁盘 I/O 次数
          </button>
          <button onClick={() => setMetric("cmp")}
            className={`px-3 py-1 rounded text-sm ${metric === "cmp" ? "bg-blue-700 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"}`}>
            键比较次数
          </button>
        </div>
      </div>

      {/* Insight banner */}
      <div className="rounded-lg bg-blue-950 border border-blue-800 px-4 py-3 mb-5 text-sm text-slate-200">
        <span className="text-blue-300 font-bold">核心洞察：</span>
        当 t={t}，B 树节点最多存 {2 * t - 1} 个键，可放入{" "}
        <span className="text-yellow-300 font-bold">{(2 * t - 1) * 8 + (2 * t) * 8} 字节</span>（假设键 8B + 指针 8B），
        而磁盘页通常 4KB–16KB，t={Math.floor(16384 / 16)} 级别可将百万条记录压缩到{" "}
        <span className="text-green-300 font-bold">{btreeHeight(1_000_000, Math.floor(16384 / 16))} 层</span>！
      </div>

      {/* Bar chart */}
      <div className="rounded-lg bg-slate-950 border border-slate-800 p-4 overflow-x-auto mb-5">
        <div className="text-slate-400 text-xs mb-3">
          {metric === "io" ? "每次查找需要的磁盘 I/O 次数（= 需要访问的树层数）" : "每次查找需要的键比较总次数"}
        </div>
        <BarChart
          data={barData}
          maxVal={metric === "io" ? maxIO : maxCmp}
          colors={colors}
          labels={legends}
        />
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="text-slate-400 text-xs border-b border-slate-700">
              <th className="py-2 px-3 text-left">数据量 n</th>
              <th className="py-2 px-3 text-right">BST 高度</th>
              <th className="py-2 px-3 text-right">B树高度(t={t})</th>
              <th className="py-2 px-3 text-right">BST I/O</th>
              <th className="py-2 px-3 text-right">B树 I/O</th>
              <th className="py-2 px-3 text-right">I/O 节省</th>
            </tr>
          </thead>
          <tbody>
            {tableData.map(({ n, bh, bt }) => (
              <tr key={n} className="border-b border-slate-800 hover:bg-slate-800/50">
                <td className="py-2 px-3 text-slate-300 font-mono">
                  {n >= 1_000_000 ? `${n / 1_000_000}M` : n >= 1000 ? `${n / 1000}K` : n}
                </td>
                <td className="py-2 px-3 text-right text-red-400 font-mono">{bh}</td>
                <td className="py-2 px-3 text-right text-blue-400 font-mono">{bt}</td>
                <td className="py-2 px-3 text-right text-red-300 font-mono">{bh}</td>
                <td className="py-2 px-3 text-right text-blue-300 font-mono">{bt}</td>
                <td className="py-2 px-3 text-right font-bold text-green-400 font-mono">
                  {bh > bt ? `${((1 - bt / bh) * 100).toFixed(0)}%↓` : "–"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Key formulas */}
      <div className="mt-5 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
        <div className="rounded-lg bg-red-950/50 border border-red-900 px-3 py-2">
          <div className="text-red-300 font-bold mb-1">平衡 BST</div>
          <div className="text-slate-300 font-mono">h = O(log₂ n)</div>
          <div className="text-slate-400 text-xs mt-1">每个内部节点单独占一次磁盘 I/O</div>
        </div>
        <div className="rounded-lg bg-blue-950/50 border border-blue-900 px-3 py-2">
          <div className="text-blue-300 font-bold mb-1">B 树（度数 t）</div>
          <div className="text-slate-300 font-mono">h ≤ log_t⌈(n+1)/2⌉</div>
          <div className="text-slate-400 text-xs mt-1">每层一次磁盘 I/O，但节点内有多个键</div>
        </div>
      </div>
    </div>
  );
}
