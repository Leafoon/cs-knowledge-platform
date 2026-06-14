"use client";
import React, { useState, useMemo, useCallback, useRef } from "react";

// ─── Formula functions ────────────────────────────────────────────────────────
// Open addressing (linear probing approximation):
//   Unsuccessful search: 1 / (1 - α)
//   Successful search:   (1/α) * ln(1/(1-α))
// Chaining:
//   Unsuccessful search: α
//   Successful search:   1 + α/2

function oaUnsuccessful(alpha: number): number {
  if (alpha >= 1) return 999;
  return 1 / (1 - alpha);
}
function oaSuccessful(alpha: number): number {
  if (alpha >= 1 || alpha <= 0) return 1;
  return (1 / alpha) * Math.log(1 / (1 - alpha));
}
function chainUnsuccessful(alpha: number): number {
  return 1 + alpha;
}
function chainSuccessful(alpha: number): number {
  return 1 + alpha / 2;
}

// ─── SVG Chart dimensions ─────────────────────────────────────────────────────
const SVG_W = 560;
const SVG_H = 280;
const PAD = { top: 20, right: 30, bottom: 45, left: 55 };
const CHART_W = SVG_W - PAD.left - PAD.right;
const CHART_H = SVG_H - PAD.top - PAD.bottom;
const ALPHA_MAX = 0.975;
const PROBE_MAX_DISPLAY = 20;

function alphaToX(alpha: number): number {
  return PAD.left + (alpha / ALPHA_MAX) * CHART_W;
}
function probeToY(probe: number): number {
  const clamped = Math.min(probe, PROBE_MAX_DISPLAY);
  return PAD.top + CHART_H - (clamped / PROBE_MAX_DISPLAY) * CHART_H;
}

function buildPath(fn: (a: number) => number, steps = 200): string {
  const pts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const alpha = (i / steps) * ALPHA_MAX;
    const probe = fn(alpha);
    if (!isFinite(probe) || probe > PROBE_MAX_DISPLAY + 5) continue;
    const x = alphaToX(alpha);
    const y = probeToY(probe);
    pts.push(`${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return pts.join(" ");
}

// Industry thresholds
const THRESHOLDS = [
  { label: "Python dict (0.667)", alpha: 0.667, color: "#38bdf8" },
  { label: "Java HashMap (0.75)", alpha: 0.75, color: "#fb923c" },
  { label: "推荐上限 (0.70)",     alpha: 0.70, color: "#a78bfa" },
];

// ─── Component ────────────────────────────────────────────────────────────────
export default function LoadFactorPerformance() {
  const [alpha, setAlpha] = useState(0.5);
  const [mode, setMode] = useState<"open" | "chain">("open");
  const [showBoth, setShowBoth] = useState(false);
  const [showThresholds, setShowThresholds] = useState(true);
  const svgRef = useRef<SVGSVGElement>(null);

  // Compute current probe values
  const oaUnsuc = oaUnsuccessful(alpha);
  const oaSuc   = oaSuccessful(alpha);
  const chUnsuc = chainUnsuccessful(alpha);
  const chSuc   = chainSuccessful(alpha);

  // Paths
  const paths = useMemo(() => ({
    oaUnsUc:  buildPath(oaUnsuccessful),
    oaSuc:    buildPath(oaSuccessful),
    chUnsuc:  buildPath(chainUnsuccessful),
    chSuc:    buildPath(chainSuccessful),
  }), []);

  // Drag alpha on SVG
  const handleSVGClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const rawX = e.clientX - rect.left;
    const scaleX = SVG_W / rect.width;
    const xInChart = rawX * scaleX - PAD.left;
    const newAlpha = Math.max(0.01, Math.min(ALPHA_MAX - 0.01, xInChart / CHART_W * ALPHA_MAX));
    setAlpha(Math.round(newAlpha * 1000) / 1000);
  }, []);

  const curX = alphaToX(alpha);

  // Danger zone fill
  const dangerStart = alphaToX(0.7);
  const warnStart   = alphaToX(0.5);
  const critStart   = alphaToX(0.9);

  const compareRows = [
    { system: "Java HashMap",   threshold: "0.75", oaUn: oaUnsuccessful(0.75).toFixed(2), oaS: oaSuccessful(0.75).toFixed(2), chUn: chainUnsuccessful(0.75).toFixed(2), chS: chainSuccessful(0.75).toFixed(2) },
    { system: "Python dict",    threshold: "0.667", oaUn: oaUnsuccessful(0.667).toFixed(2), oaS: oaSuccessful(0.667).toFixed(2), chUn: chainUnsuccessful(0.667).toFixed(2), chS: chainSuccessful(0.667).toFixed(2) },
    { system: "推荐最大值",      threshold: "0.70", oaUn: oaUnsuccessful(0.70).toFixed(2), oaS: oaSuccessful(0.70).toFixed(2), chUn: chainUnsuccessful(0.70).toFixed(2), chS: chainSuccessful(0.70).toFixed(2) },
    { system: "当前 α",          threshold: alpha.toFixed(3), oaUn: oaUnsuc.toFixed(2), oaS: oaSuc.toFixed(2), chUn: chUnsuc.toFixed(2), chS: chSuc.toFixed(2), highlight: true },
  ];

  return (
    <div className="dark isolate rounded-2xl border border-slate-700 bg-slate-900 p-5 space-y-5 font-mono text-sm text-slate-200">
      {/* Header */}
      <div>
        <h3 className="text-base font-bold text-white">📊 负载因子 α vs 探测次数期望</h3>
        <p className="text-slate-400 text-xs mt-0.5">点击曲线图调整 α 值，查看开放寻址法与链地址法的性能差异</p>
      </div>

      {/* Mode toggles */}
      <div className="flex gap-2 flex-wrap">
        <button onClick={() => { setMode("open"); setShowBoth(false); }}
          className={`px-3 py-1.5 rounded text-xs font-bold transition-all ${mode === "open" && !showBoth ? "bg-rose-700 text-white ring-1 ring-rose-400" : "bg-slate-700 text-slate-400 hover:bg-slate-600"}`}>
          开放寻址法
        </button>
        <button onClick={() => { setMode("chain"); setShowBoth(false); }}
          className={`px-3 py-1.5 rounded text-xs font-bold transition-all ${mode === "chain" && !showBoth ? "bg-sky-700 text-white ring-1 ring-sky-400" : "bg-slate-700 text-slate-400 hover:bg-slate-600"}`}>
          链地址法
        </button>
        <button onClick={() => setShowBoth(b => !b)}
          className={`px-3 py-1.5 rounded text-xs font-bold transition-all ${showBoth ? "bg-violet-700 text-white ring-1 ring-violet-400" : "bg-slate-700 text-slate-400 hover:bg-slate-600"}`}>
          双模式对比
        </button>
        <button onClick={() => setShowThresholds(t => !t)}
          className={`px-3 py-1.5 rounded text-xs transition-all ${showThresholds ? "bg-slate-600 text-white" : "bg-slate-700 text-slate-400 hover:bg-slate-600"}`}>
          {showThresholds ? "隐藏" : "显示"}业界阈值
        </button>
      </div>

      {/* Alpha slider */}
      <div className="flex gap-3 items-center flex-wrap">
        <span className="text-slate-400 text-xs">α =</span>
        <input type="range" min="0.01" max="0.97" step="0.01" value={alpha}
          onChange={e => setAlpha(parseFloat(e.target.value))}
          className="w-40 accent-violet-500" />
        <span className={`text-base font-bold px-2 py-0.5 rounded ${alpha > 0.9 ? "text-red-400 bg-red-900/40" : alpha > 0.7 ? "text-amber-400 bg-amber-900/40" : "text-emerald-400 bg-emerald-900/40"}`}>
          {alpha.toFixed(3)}
        </span>
        <span className={`text-xs ${alpha > 0.9 ? "text-red-400" : alpha > 0.7 ? "text-amber-400" : "text-emerald-400"}`}>
          {alpha > 0.9 ? "🔴 危险区间 — 强烈建议 rehash" : alpha > 0.7 ? "🟡 警戒区间" : "🟢 健康区间"}
        </span>
      </div>

      {/* SVG Chart */}
      <div className="w-full overflow-x-auto">
        <svg ref={svgRef} viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full max-w-2xl cursor-crosshair select-none rounded-lg bg-slate-800"
          onClick={handleSVGClick}>
          {/* Danger zone backgrounds */}
          <rect x={warnStart} y={PAD.top} width={critStart - warnStart} height={CHART_H} fill="#78350f" opacity={0.15} />
          <rect x={critStart} y={PAD.top} width={alphaToX(ALPHA_MAX) - critStart} height={CHART_H} fill="#7f1d1d" opacity={0.25} />
          <text x={warnStart + 4} y={PAD.top + 12} fill="#d97706" fontSize={9} fontFamily="monospace">⚠ α{">"}0.5</text>
          <text x={critStart + 4} y={PAD.top + 12} fill="#ef4444" fontSize={9} fontFamily="monospace">🔴 α{">"}0.9</text>

          {/* Grid lines */}
          {[2, 4, 6, 8, 10, 15, 20].map(v => (
            <g key={v}>
              <line x1={PAD.left} y1={probeToY(v)} x2={PAD.left + CHART_W} y2={probeToY(v)} stroke="#475569" strokeWidth={0.5} strokeDasharray="3,3" />
              <text x={PAD.left - 4} y={probeToY(v) + 3} textAnchor="end" fill="#64748b" fontSize={9} fontFamily="monospace">{v}</text>
            </g>
          ))}
          {[0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map(v => (
            <g key={v}>
              <line x1={alphaToX(v)} y1={PAD.top} x2={alphaToX(v)} y2={PAD.top + CHART_H} stroke="#334155" strokeWidth={0.5} strokeDasharray="2,3" />
              <text x={alphaToX(v)} y={PAD.top + CHART_H + 14} textAnchor="middle" fill="#64748b" fontSize={9} fontFamily="monospace">{v}</text>
            </g>
          ))}

          {/* Axis labels */}
          <text x={PAD.left + CHART_W / 2} y={SVG_H - 3} textAnchor="middle" fill="#94a3b8" fontSize={11} fontFamily="monospace">负载因子 α = n/m</text>
          <text x={14} y={PAD.top + CHART_H / 2} textAnchor="middle" fill="#94a3b8" fontSize={11} fontFamily="monospace" transform={`rotate(-90, 14, ${PAD.top + CHART_H / 2})`}>期望探测次数</text>

          {/* Industry threshold lines */}
          {showThresholds && THRESHOLDS.map(t => (
            <g key={t.label}>
              <line x1={alphaToX(t.alpha)} y1={PAD.top} x2={alphaToX(t.alpha)} y2={PAD.top + CHART_H} stroke={t.color} strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7} />
              <text x={alphaToX(t.alpha) + 2} y={PAD.top + CHART_H - 5} fill={t.color} fontSize={8} fontFamily="monospace" opacity={0.9}>{t.label}</text>
            </g>
          ))}

          {/* Curves — Open Addressing */}
          {(showBoth || mode === "open") && (
            <>
              <path d={paths.oaUnsUc} stroke="#f43f5e" fill="none" strokeWidth={2} strokeLinecap="round" />
              <path d={paths.oaSuc}   stroke="#fb923c" fill="none" strokeWidth={2} strokeLinecap="round" />
            </>
          )}
          {/* Curves — Chaining */}
          {(showBoth || mode === "chain") && (
            <>
              <path d={paths.chUnsuc} stroke="#38bdf8" fill="none" strokeWidth={2} strokeLinecap="round" strokeDasharray="6,3" />
              <path d={paths.chSuc}   stroke="#34d399" fill="none" strokeWidth={2} strokeLinecap="round" strokeDasharray="6,3" />
            </>
          )}

          {/* Vertical cursor line */}
          <line x1={curX} y1={PAD.top} x2={curX} y2={PAD.top + CHART_H} stroke="#a78bfa" strokeWidth={1.5} strokeDasharray="3,2" />

          {/* Probe value dots */}
          {(showBoth || mode === "open") && isFinite(oaUnsuc) && oaUnsuc <= PROBE_MAX_DISPLAY && (
            <circle cx={curX} cy={probeToY(oaUnsuc)} r={4} fill="#f43f5e" stroke="#1e293b" strokeWidth={1.5} />
          )}
          {(showBoth || mode === "open") && isFinite(oaSuc) && oaSuc <= PROBE_MAX_DISPLAY && (
            <circle cx={curX} cy={probeToY(oaSuc)} r={4} fill="#fb923c" stroke="#1e293b" strokeWidth={1.5} />
          )}
          {(showBoth || mode === "chain") && (
            <circle cx={curX} cy={probeToY(chUnsuc)} r={4} fill="#38bdf8" stroke="#1e293b" strokeWidth={1.5} />
          )}
          {(showBoth || mode === "chain") && (
            <circle cx={curX} cy={probeToY(chSuc)} r={4} fill="#34d399" stroke="#1e293b" strokeWidth={1.5} />
          )}

          {/* Alpha label on cursor */}
          <text x={curX + 4} y={PAD.top + 8} fill="#a78bfa" fontSize={9} fontFamily="monospace">α={alpha.toFixed(2)}</text>
        </svg>
      </div>

      {/* Legend */}
      <div className="flex gap-3 flex-wrap text-xs">
        {(showBoth || mode === "open") && (
          <>
            <span className="flex items-center gap-1.5"><span className="w-5 h-0.5 bg-rose-500 inline-block" />OA 不成功查找 = 1/(1-α)</span>
            <span className="flex items-center gap-1.5"><span className="w-5 h-0.5 bg-orange-400 inline-block" />OA 成功查找 = (1/α)ln(1/(1-α))</span>
          </>
        )}
        {(showBoth || mode === "chain") && (
          <>
            <span className="flex items-center gap-1.5"><span className="w-5 border-t border-dashed border-sky-400 inline-block" />链地址不成功查找 = 1+α</span>
            <span className="flex items-center gap-1.5"><span className="w-5 border-t border-dashed border-emerald-400 inline-block" />链地址成功查找 = 1+α/2</span>
          </>
        )}
      </div>

      {/* Current values panel */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-lg bg-slate-800/70 p-3 space-y-1.5">
          <p className="text-rose-400 font-bold">开放寻址法 (α={alpha.toFixed(2)})</p>
          <p className="text-slate-300">不成功查找: <span className={`font-bold ${oaUnsuc > 10 ? "text-red-400" : oaUnsuc > 5 ? "text-amber-400" : "text-emerald-400"}`}>{isFinite(oaUnsuc) ? oaUnsuc.toFixed(2) : "∞"} 次</span></p>
          <p className="text-slate-300">成功查找:   <span className={`font-bold ${oaSuc > 4 ? "text-amber-400" : "text-emerald-400"}`}>{isFinite(oaSuc) ? oaSuc.toFixed(2) : "∞"} 次</span></p>
        </div>
        <div className="rounded-lg bg-slate-800/70 p-3 space-y-1.5">
          <p className="text-sky-400 font-bold">链地址法 (α={alpha.toFixed(2)})</p>
          <p className="text-slate-300">不成功查找: <span className={`font-bold ${chUnsuc > 10 ? "text-red-400" : chUnsuc > 5 ? "text-amber-400" : "text-emerald-400"}`}>{chUnsuc.toFixed(2)} 次</span></p>
          <p className="text-slate-300">成功查找:   <span className="font-bold text-emerald-400">{chSuc.toFixed(2)} 次</span></p>
        </div>
      </div>

      {/* Comparison table */}
      <div>
        <p className="text-slate-400 text-xs mb-2 font-semibold">业界标准阈值对比表</p>
        <div className="overflow-x-auto rounded-lg border border-slate-700">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-800 text-slate-400">
              <tr>
                <th className="px-3 py-2">系统/场景</th>
                <th className="px-3 py-2">α 阈值</th>
                <th className="px-3 py-2 text-rose-400">OA 不成功</th>
                <th className="px-3 py-2 text-orange-400">OA 成功</th>
                <th className="px-3 py-2 text-sky-400">链地址不成功</th>
                <th className="px-3 py-2 text-emerald-400">链地址成功</th>
              </tr>
            </thead>
            <tbody>
              {compareRows.map((row, i) => (
                <tr key={i} className={`border-t border-slate-700 ${row.highlight ? "bg-violet-950/40 text-violet-200" : "text-slate-300"}`}>
                  <td className="px-3 py-2 font-semibold">{row.system}</td>
                  <td className="px-3 py-2">{row.threshold}</td>
                  <td className="px-3 py-2 text-rose-400">{row.oaUn}</td>
                  <td className="px-3 py-2 text-orange-400">{row.oaS}</td>
                  <td className="px-3 py-2 text-sky-400">{row.chUn}</td>
                  <td className="px-3 py-2 text-emerald-400">{row.chS}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
