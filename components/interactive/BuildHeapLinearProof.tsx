"use client";
import React, { useState } from "react";

// ─────────── Build-Heap Level Contribution Visualizer ───────────
// Shows: for each level h (from leaf up), how many nodes × O(h) = total cost
// The series Σ h * n/2^(h+1) converges to O(n)

const MAX_N = 15;

function computeLevels(n: number) {
  // levels[h] = { nodesAtHeight, cost, cumulativeCost }
  const levels: { height: number; nodes: number; costEach: number; totalCost: number }[] = [];
  let level = 0;
  let remainder = n;
  let h = 0;
  // Compute the height of the heap
  const heapH = Math.floor(Math.log2(n));
  for (let h = 0; h <= heapH; h++) {
    // nodes at height h = ceil(n / 2^(h+1))
    const nodes = Math.ceil(n / Math.pow(2, h + 1));
    levels.push({ height: h, nodes, costEach: h, totalCost: nodes * h });
  }
  return levels;
}

const COLORS = ["#6366f1", "#3b82f6", "#0ea5e9", "#14b8a6", "#22c55e", "#eab308", "#f59e0b", "#ef4444"];

export default function BuildHeapLinearProof() {
  const [n, setN] = useState(10);
  const [highlightLevel, setHighlightLevel] = useState<number | null>(null);

  const levels = computeLevels(n);
  const totalCost = levels.reduce((s, l) => s + l.totalCost, 0);
  const nodeCount = n;
  const heapH = Math.floor(Math.log2(n));

  // Upper bound: n * Σ h/2^(h+1) ≈ 2n
  const seriesApprox = (2 * n).toFixed(1);

  return (
    <div className="rounded-xl border p-4 space-y-4" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-card)" }}>
      <h3 className="font-bold text-base" style={{ color: "var(--color-text-primary)" }}>
        📐 线性建堆 O(n) 级数求和可视化
      </h3>

      {/* n slider */}
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium whitespace-nowrap" style={{ color: "var(--color-text-muted)" }}>
          n = <strong style={{ color: "var(--color-text-primary)" }}>{n}</strong>
        </label>
        <input type="range" min={3} max={MAX_N} value={n} onChange={e => setN(Number(e.target.value))}
          className="flex-1 accent-blue-500" />
        <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>堆高 h={heapH}</span>
      </div>

      {/* Core insight */}
      <div className="rounded-lg p-3 text-sm" style={{ background: "rgba(99,102,241,0.08)", borderLeft: "4px solid #6366f1" }}>
        <p className="font-semibold mb-1" style={{ color: "var(--color-text-primary)" }}>核心公式</p>
        <p style={{ color: "var(--color-text-muted)" }}>
          总代价 = Σ<sub>h=0</sub><sup>⌊log n⌋</sup> ⌈n/2<sup>h+1</sup>⌉ · O(h)
          &nbsp;≤ n · Σ<sub>h=0</sub><sup>∞</sup> h/2<sup>h</sup>
          &nbsp;= <strong style={{ color: "#6366f1" }}>2n = O(n)</strong>
        </p>
        <p className="mt-1 text-xs" style={{ color: "var(--color-text-muted)" }}>
          （等比数列求和：Σ h·x<sup>h</sup> = x/(1-x)²，令 x=1/2 得 Σ h/2<sup>h</sup> = 2）
        </p>
      </div>

      {/* Level table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr style={{ borderBottom: "2px solid var(--color-border)" }}>
              {["高度 h", "该高度节点数 ⌈n/2^(h+1)⌉", "每节点代价 O(h)", "该层总代价", "占比"].map(h => (
                <th key={h} className="px-2 py-2 text-left font-semibold text-xs" style={{ color: "var(--color-text-muted)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {levels.map((l, idx) => {
              const pct = totalCost > 0 ? (l.totalCost / totalCost * 100) : 0;
              const color = COLORS[idx % COLORS.length];
              const isHL = highlightLevel === l.height;
              return (
                <tr key={l.height}
                  onMouseEnter={() => setHighlightLevel(l.height)}
                  onMouseLeave={() => setHighlightLevel(null)}
                  className="cursor-pointer transition-colors"
                  style={{ background: isHL ? "rgba(99,102,241,0.08)" : "transparent" }}>
                  <td className="px-2 py-1.5 font-mono font-bold" style={{ color }}>{l.height}</td>
                  <td className="px-2 py-1.5 font-mono" style={{ color: "var(--color-text-primary)" }}>{l.nodes}</td>
                  <td className="px-2 py-1.5 font-mono" style={{ color: "var(--color-text-muted)" }}>{l.costEach === 0 ? "0（叶子无需 HEAPIFY）" : l.costEach}</td>
                  <td className="px-2 py-1.5 font-mono font-bold" style={{ color: "var(--color-text-primary)" }}>{l.totalCost}</td>
                  <td className="px-2 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 rounded-full h-2" style={{ background: "var(--color-border)" }}>
                        <div className="h-2 rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: color }} />
                      </div>
                      <span className="text-xs font-mono w-10" style={{ color }}>{pct.toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr style={{ borderTop: "2px solid var(--color-border)" }}>
              <td colSpan={3} className="px-2 py-2 font-bold text-sm" style={{ color: "var(--color-text-primary)" }}>合计</td>
              <td className="px-2 py-2 font-bold text-sm" style={{ color: "#6366f1" }}>{totalCost}</td>
              <td className="px-2 py-2 text-xs" style={{ color: "var(--color-text-muted)" }}>
                ≤ 2n = {2 * n}（理论上界）
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      {/* Stacked bar chart */}
      <div>
        <p className="text-xs font-semibold mb-2" style={{ color: "var(--color-text-muted)" }}>各层代价分布（总代价 = {totalCost}，n = {n}）</p>
        <div className="flex rounded-lg overflow-hidden h-8">
          {levels.filter(l => l.totalCost > 0).map((l, idx) => {
            const color = COLORS[idx % COLORS.length];
            const pct = l.totalCost / totalCost * 100;
            const isHL = highlightLevel === l.height;
            return (
              <div key={l.height}
                className="flex items-center justify-center text-xs font-bold text-white transition-all duration-300 cursor-pointer"
                style={{ width: `${pct}%`, background: color, opacity: isHL ? 1 : 0.8, filter: isHL ? "brightness(1.15)" : "none" }}
                onMouseEnter={() => setHighlightLevel(l.height)}
                onMouseLeave={() => setHighlightLevel(null)}
                title={`h=${l.height}: ${l.totalCost} 次操作`}>
                {pct > 10 ? `h=${l.height}` : ""}
              </div>
            );
          })}
        </div>
        <p className="text-xs mt-1" style={{ color: "var(--color-text-muted)" }}>
          💡 叶节点（h=0）占节点总数一半但代价为 0；高层节点少但代价高——两者抵消使总和 O(n)。
        </p>
      </div>

      {/* BUILD process description */}
      <div className="rounded-lg p-3 text-sm space-y-1" style={{ background: "var(--color-bg-secondary)" }}>
        <p className="font-semibold" style={{ color: "var(--color-text-primary)" }}>BUILD-MAX-HEAP 算法流程</p>
        <p style={{ color: "var(--color-text-muted)" }}>
          从 <code className="rounded px-1" style={{ background: "var(--color-bg-card)" }}>i = ⌊n/2⌋</code> 倒序到 <code className="rounded px-1" style={{ background: "var(--color-bg-card)" }}>i = 0</code>（Python 0-indexed），
          对每个节点调用 <code className="rounded px-1" style={{ background: "var(--color-bg-card)" }}>MAX_HEAPIFY(i)</code>。
          叶子节点（索引 &gt; ⌊n/2⌋-1）跳过，因为叶子只有 1 个节点自然满足堆性质。
        </p>
        <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
          当前 n={n}：从 i={Math.floor(n / 2) - 1} 倒序 → 0，共调用 {Math.floor(n / 2)} 次 HEAPIFY。
        </p>
      </div>
    </div>
  );
}
