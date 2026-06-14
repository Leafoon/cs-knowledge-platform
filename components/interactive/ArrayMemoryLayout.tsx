"use client";

import React, { useState } from "react";

// ── 矩阵规格 ─────────────────────────────────────────────────────────────────
const ROWS = 4;
const COLS = 6;

function buildMatrix(): number[][] {
  const m: number[][] = [];
  let v = 0;
  for (let r = 0; r < ROWS; r++) {
    m.push([]);
    for (let c = 0; c < COLS; c++) {
      m[r].push(++v);
    }
  }
  return m;
}

const MATRIX = buildMatrix();

// 计算内存地址：行主序 base + (i*cols + j)*4
function rowMajorAddr(r: number, c: number, base = 1000): number {
  return base + (r * COLS + c) * 4;
}
// 列主序 base + (j*rows + i)*4
function colMajorAddr(r: number, c: number, base = 1000): number {
  return base + (c * ROWS + r) * 4;
}

// 行主序线性展开顺序
const ROW_MAJOR_ORDER: [number, number][] = [];
for (let r = 0; r < ROWS; r++)
  for (let c = 0; c < COLS; c++)
    ROW_MAJOR_ORDER.push([r, c]);

// 列主序线性展开顺序
const COL_MAJOR_ORDER: [number, number][] = [];
for (let c = 0; c < COLS; c++)
  for (let r = 0; r < ROWS; r++)
    COL_MAJOR_ORDER.push([r, c]);

type TraversalMode = "row" | "col";
type LayoutMode = "row-major" | "col-major";

// 颜色映射（行索引 → hue）
const ROW_COLORS = [
  { bg: "bg-blue-500", text: "text-blue-700 dark:text-blue-300", light: "bg-blue-500/15 dark:bg-blue-500/20", border: "border-blue-400/50" },
  { bg: "bg-violet-500", text: "text-violet-700 dark:text-violet-300", light: "bg-violet-500/15 dark:bg-violet-500/20", border: "border-violet-400/50" },
  { bg: "bg-emerald-500", text: "text-emerald-700 dark:text-emerald-300", light: "bg-emerald-500/15 dark:bg-emerald-500/20", border: "border-emerald-400/50" },
  { bg: "bg-amber-500", text: "text-amber-700 dark:text-amber-300", light: "bg-amber-500/15 dark:bg-amber-500/20", border: "border-amber-400/50" },
];

export default function ArrayMemoryLayout() {
  const [layout, setLayout] = useState<LayoutMode>("row-major");
  const [traversal, setTraversal] = useState<TraversalMode>("row");
  const [hoveredCell, setHoveredCell] = useState<[number, number] | null>(null);
  const [showAddr, setShowAddr] = useState(false);

  // 决定遍历顺序（哪种布局 × 哪种遍历 → 缓存是否友好）
  const isCache = (layout === "row-major" && traversal === "row") ||
                  (layout === "col-major" && traversal === "col");

  // 内存中的线性排列
  const linearOrder = layout === "row-major" ? ROW_MAJOR_ORDER : COL_MAJOR_ORDER;
  // 遍历时的访问顺序
  const accessOrder = traversal === "row" ? ROW_MAJOR_ORDER : COL_MAJOR_ORDER;

  // 为每个格子计算内存位置
  function getMemPos(r: number, c: number): number {
    return linearOrder.findIndex(([lr, lc]) => lr === r && lc === c);
  }

  function getAddr(r: number, c: number): number {
    return layout === "row-major" ? rowMajorAddr(r, c) : colMajorAddr(r, c);
  }

  // 遍历顺序中的每个格子：高亮"下一次访问跨越多少内存格"
  function getCacheJump(idx: number): number {
    if (idx >= accessOrder.length - 1) return 0;
    const [r1, c1] = accessOrder[idx];
    const [r2, c2] = accessOrder[idx + 1];
    const pos1 = getMemPos(r1, c1);
    const pos2 = getMemPos(r2, c2);
    return Math.abs(pos2 - pos1);
  }

  // 遍历步骤的平均跳跃距离（=缓存效率指标）
  const totalJump = accessOrder.reduce((sum, _, i) => sum + getCacheJump(i), 0);
  const avgJump = totalJump / (accessOrder.length - 1);

  const hovered = hoveredCell;

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-violet-500/15 dark:bg-violet-500/20 flex items-center justify-center text-xl">
          🗺️
        </div>
        <div>
          <h3 className="font-bold text-text-primary text-base">数组内存布局与缓存局部性</h3>
          <p className="text-xs text-text-secondary">行主序 vs 列主序，不同遍历方式对缓存命中的影响</p>
        </div>
      </div>

      {/* 控制 */}
      <div className="flex flex-wrap gap-3 border-t border-border-subtle pt-3">
        <div className="flex items-center gap-2 text-xs">
          <span className="text-text-secondary font-medium">内存布局：</span>
          <button
            onClick={() => setLayout("row-major")}
            className={`px-2.5 py-1 rounded-lg border text-xs font-medium transition-all ${layout === "row-major"
              ? "bg-blue-500/20 border-blue-400/60 text-blue-600 dark:text-blue-300"
              : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}
          >
            行主序（C/Python）
          </button>
          <button
            onClick={() => setLayout("col-major")}
            className={`px-2.5 py-1 rounded-lg border text-xs font-medium transition-all ${layout === "col-major"
              ? "bg-violet-500/20 border-violet-400/60 text-violet-600 dark:text-violet-300"
              : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}
          >
            列主序（Fortran/MATLAB）
          </button>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-text-secondary font-medium">遍历方向：</span>
          <button
            onClick={() => setTraversal("row")}
            className={`px-2.5 py-1 rounded-lg border text-xs transition-all ${traversal === "row"
              ? "bg-emerald-500/20 border-emerald-400/60 text-emerald-600 dark:text-emerald-300"
              : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}
          >
            按行遍历
          </button>
          <button
            onClick={() => setTraversal("col")}
            className={`px-2.5 py-1 rounded-lg border text-xs transition-all ${traversal === "col"
              ? "bg-emerald-500/20 border-emerald-400/60 text-emerald-600 dark:text-emerald-300"
              : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}
          >
            按列遍历
          </button>
        </div>
        <label className="flex items-center gap-1.5 text-xs text-text-secondary cursor-pointer ml-auto">
          <input type="checkbox" checked={showAddr} onChange={(e) => setShowAddr(e.target.checked)}
            className="accent-blue-500" />
          显示内存地址
        </label>
      </div>

      {/* 缓存友好性指示器 */}
      <div className={`rounded-xl border p-3 text-sm font-medium flex items-center gap-3 transition-colors
        ${isCache
          ? "bg-emerald-500/10 border-emerald-400/40 text-emerald-700 dark:text-emerald-300"
          : "bg-rose-500/10 border-rose-400/40 text-rose-700 dark:text-rose-300"}`}>
        <span className="text-xl">{isCache ? "✅" : "⚠️"}</span>
        <div>
          <div className="font-semibold">
            {isCache ? "缓存友好（Cache Friendly）" : "缓存不友好（Cache Unfriendly）"}
          </div>
          <div className="text-xs opacity-80 font-normal mt-0.5">
            {layout === "row-major" && traversal === "row" && "行主序 × 按行遍历：连续内存访问，每条缓存行都被充分利用"}
            {layout === "row-major" && traversal === "col" && "行主序 × 按列遍历：每步跨越整行内存，缓存行频繁失效"}
            {layout === "col-major" && traversal === "col" && "列主序 × 按列遍历：连续内存访问，每条缓存行都被充分利用"}
            {layout === "col-major" && traversal === "row" && "列主序 × 按行遍历：每步跨越整列内存，缓存行频繁失效"}
          </div>
        </div>
        <div className="ml-auto text-right text-xs">
          <div className="text-text-tertiary">平均跨越内存格数</div>
          <div className={`text-lg font-mono font-bold ${isCache ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
            {avgJump.toFixed(1)}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* 左：2D 矩阵视图 */}
        <div>
          <div className="text-xs text-text-secondary mb-2 flex items-center gap-2">
            <span>二维矩阵视图（{ROWS} × {COLS}）</span>
            <span className="text-text-tertiary">
              {traversal === "row" ? "→ 按行遍历方向" : "↓ 按列遍历方向"}
            </span>
          </div>
          <div className="border border-border-subtle rounded-xl overflow-hidden">
            {/* 列号 */}
            <div className="flex bg-bg-tertiary border-b border-border-subtle">
              <div className="w-7 text-center text-[10px] text-text-tertiary p-1" />
              {Array.from({ length: COLS }, (_, c) => (
                <div key={c} className="flex-1 text-center text-[10px] text-text-tertiary p-1 font-mono">
                  [{c}]
                </div>
              ))}
            </div>
            {MATRIX.map((row, r) => (
              <div key={r} className="flex">
                {/* 行号 */}
                <div className={`w-7 flex items-center justify-center text-[10px] font-mono border-r border-border-subtle ${ROW_COLORS[r % ROW_COLORS.length].text}`}>
                  [{r}]
                </div>
                {row.map((val, c) => {
                  const isHov = hovered?.[0] === r && hovered?.[1] === c;
                  const accessIdx = accessOrder.findIndex(([ar, ac]) => ar === r && ac === c);
                  const colors = ROW_COLORS[layout === "row-major" ? r : c % ROW_COLORS.length];
                  return (
                    <div
                      key={c}
                      onMouseEnter={() => setHoveredCell([r, c])}
                      onMouseLeave={() => setHoveredCell(null)}
                      className={`
                        flex-1 flex flex-col items-center justify-center
                        border-r border-b border-border-subtle
                        cursor-pointer transition-all duration-100 p-1
                        ${isHov
                          ? `${colors.light} border ${colors.border}`
                          : "hover:bg-bg-tertiary"
                        }
                      `}
                    >
                      <span className={`text-xs font-mono font-bold ${colors.text}`}>
                        {val}
                      </span>
                      {showAddr && (
                        <span className="text-[8px] text-text-tertiary font-mono leading-tight">
                          {getAddr(r, c)}
                        </span>
                      )}
                      <span className={`text-[8px] font-mono ${isCache ? "text-emerald-600 dark:text-emerald-400" : "text-rose-500"}`}>
                        #{accessIdx + 1}
                      </span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          {/* 悬浮信息 */}
          {hovered && (
            <div className="mt-2 px-3 py-2 rounded-lg bg-bg-tertiary border border-border-subtle text-xs font-mono text-text-secondary">
              A[{hovered[0]}][{hovered[1]}] = {MATRIX[hovered[0]][hovered[1]]} ｜
              内存地址 = {getAddr(hovered[0], hovered[1])} ｜
              线性位置 = #{getMemPos(hovered[0], hovered[1])} ｜
              遍历序号 = #{accessOrder.findIndex(([r, c]) => r === hovered[0] && c === hovered[1]) + 1}
            </div>
          )}
        </div>

        {/* 右：线性内存视图 */}
        <div>
          <div className="text-xs text-text-secondary mb-2">
            内存线性视图（{layout === "row-major" ? "行主序排列" : "列主序排列"}）
          </div>
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3">
            {/* 内存槽 */}
            <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)` }}>
              {linearOrder.map(([r, c], memIdx) => {
                const val = MATRIX[r][c];
                const accessIdx = accessOrder.findIndex(([ar, ac]) => ar === r && ac === c);
                const isHov = hovered?.[0] === r && hovered?.[1] === c;
                const colors = ROW_COLORS[layout === "row-major" ? r : c % ROW_COLORS.length];

                // 计算与下一个遍历元素的跳跃距离
                const nextAccessIdx = accessIdx + 1;
                const nextCell = accessOrder[nextAccessIdx];
                const jump = nextCell ? Math.abs(getMemPos(nextCell[0], nextCell[1]) - memIdx) : 0;

                return (
                  <div
                    key={memIdx}
                    className={`
                      rounded flex flex-col items-center justify-center p-1 border
                      transition-all duration-100 cursor-pointer
                      ${isHov
                        ? `${colors.light} ${colors.border}`
                        : `${colors.light.replace("/15", "/8").replace("/20", "/10")} border-border-subtle`
                      }
                    `}
                    onMouseEnter={() => setHoveredCell([r, c])}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    <span className={`text-xs font-bold font-mono ${colors.text}`}>{val}</span>
                    <span className="text-[8px] text-text-tertiary font-mono">[{r}][{c}]</span>
                    <span className={`text-[8px] font-mono ${isCache ? "text-emerald-600 dark:text-emerald-400" : "text-rose-500"}`}>
                      #{accessIdx + 1}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* 地址标注 */}
            {showAddr && (
              <div className="mt-2 text-[9px] text-text-tertiary font-mono flex flex-wrap gap-1">
                {linearOrder.map(([r, c], i) => (
                  <span key={i}>{getAddr(r, c)}</span>
                ))}
              </div>
            )}
          </div>

          {/* 缓存行说明 */}
          <div className="mt-3 rounded-lg bg-bg-tertiary border border-border-subtle p-3 text-xs text-text-secondary space-y-1">
            <div className="font-semibold text-text-primary">💡 缓存行（Cache Line）解释</div>
            <div>
              CPU 一次从内存加载 <strong>64 字节</strong>（含 16 个 int32 / 8 个 int64）到缓存，
              称为一条"缓存行"。
            </div>
            <div>
              {isCache
                ? "✅ 当前遍历方向与内存排列一致：每次加载的缓存行都会被后续访问重复利用，缓存命中率高。"
                : `⚠️ 当前遍历与内存排列垂直：每次跨越约 ${COLS} 个元素（${COLS * 4} 字节），频繁触发缓存缺失（Cache Miss）。`
              }
            </div>
            <div className="text-text-tertiary">
              实验证明：对 1000×1000 矩阵，缓存不友好遍历比友好遍历慢 <strong>5–10 倍</strong>。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
