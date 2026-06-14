'use client';
import React, { useState, useRef, useEffect } from 'react';

// ====== 最近点对分治可视化 ======

interface Point { x: number; y: number; id: number; }

interface CPStep {
  type: 'divide' | 'conquer_left' | 'conquer_right' | 'combine' | 'done';
  midX: number;
  lo: number[];    // 左侧点集 id
  ro: number[];    // 右侧点集 id
  delta: number;
  strip: number[];  // strip 内点集 id
  closestPair: [number, number] | null;
  closestDist: number;
  desc: string;
}

function dist(p: Point, q: Point) {
  return Math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2);
}

function bruteClosest(pts: Point[]): { pair: [number, number]; d: number } {
  let best = { pair: [pts[0].id, pts[1].id] as [number, number], d: dist(pts[0], pts[1]) };
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++) {
      const d = dist(pts[i], pts[j]);
      if (d < best.d) best = { pair: [pts[i].id, pts[j].id], d };
    }
  return best;
}

function buildCPSteps(points: Point[]): { steps: CPStep[]; finalPair: [number, number]; finalDist: number } {
  const steps: CPStep[] = [];
  let globalBest: { pair: [number, number]; d: number } = { pair: [0, 1], d: Infinity };

  function solve(pts: Point[], depth: number): { pair: [number, number]; d: number } {
    if (pts.length <= 3) {
      const res = bruteClosest(pts);
      if (res.d < globalBest.d) globalBest = res;
      return res;
    }

    pts = [...pts].sort((a, b) => a.x - b.x);
    const mid = Math.floor(pts.length / 2);
    const midX = pts[mid].x;
    const leftPts = pts.slice(0, mid);
    const rightPts = pts.slice(mid);

    steps.push({
      type: 'divide',
      midX,
      lo: leftPts.map(p => p.id),
      ro: rightPts.map(p => p.id),
      delta: Infinity,
      strip: [],
      closestPair: null,
      closestDist: Infinity,
      desc: `Divide：以 x=${midX.toFixed(1)} 分割，左 ${leftPts.length} 点，右 ${rightPts.length} 点`,
    });

    const lRes = solve(leftPts, depth + 1);
    const rRes = solve(rightPts, depth + 1);
    const delta = Math.min(lRes.d, rRes.d);
    const bestPair = lRes.d < rRes.d ? lRes.pair : rRes.pair;

    // Strip check
    const strip = pts.filter(p => Math.abs(p.x - midX) < delta);
    strip.sort((a, b) => a.y - b.y);

    let stripBest = { pair: bestPair as [number, number], d: delta };
    for (let i = 0; i < strip.length; i++) {
      for (let j = i + 1; j < strip.length && strip[j].y - strip[i].y < delta; j++) {
        const d = dist(strip[i], strip[j]);
        if (d < stripBest.d) {
          stripBest = { pair: [strip[i].id, strip[j].id], d };
          if (d < globalBest.d) globalBest = stripBest;
        }
      }
    }

    steps.push({
      type: 'combine',
      midX,
      lo: leftPts.map(p => p.id),
      ro: rightPts.map(p => p.id),
      delta,
      strip: strip.map(p => p.id),
      closestPair: stripBest.pair,
      closestDist: stripBest.d,
      desc: `Combine：δ=min(${lRes.d.toFixed(2)},${rRes.d.toFixed(2)})=${delta.toFixed(2)}，strip中${strip.length}点，最近对距离=${stripBest.d.toFixed(2)}`,
    });

    return stripBest;
  }

  const final = solve(points, 0);
  steps.push({
    type: 'done',
    midX: 0,
    lo: points.map(p => p.id),
    ro: [],
    delta: final.d,
    strip: [],
    closestPair: final.pair,
    closestDist: final.d,
    desc: `完成！最近点对距离 = ${final.d.toFixed(4)}`,
  });

  return { steps, finalPair: final.pair, finalDist: final.d };
}

const PRESETS: { name: string; pts: [number, number][] }[] = [
  { name: '书中示例', pts: [[2,3],[12,30],[40,50],[5,1],[12,10],[3,4]] },
  { name: '均匀分布', pts: [[10,20],[30,50],[15,35],[45,25],[5,45],[35,15],[25,40],[50,10]] },
  { name: '近距离点', pts: [[10,10],[12,11],[30,30],[32,29],[50,50],[52,51]] },
  { name: '水平排列', pts: [[5,20],[15,20],[25,20],[35,20],[45,20],[55,20]] },
];

export default function ClosestPairVisualizer() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const preset = PRESETS[presetIdx];
  const points: Point[] = preset.pts.map(([x, y], id) => ({ x, y, id }));
  const { steps, finalPair, finalDist } = buildCPSteps(points);

  const safeIdx = Math.min(stepIdx, steps.length - 1);
  const step = steps[safeIdx];

  const CANVAS_W = 480, CANVAS_H = 340;
  const PAD = 30;

  // Normalize points to canvas
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;

  const toCanvas = (p: Point) => ({
    cx: PAD + ((p.x - minX) / rangeX) * (CANVAS_W - PAD * 2),
    cy: PAD + (1 - (p.y - minY) / rangeY) * (CANVAS_H - PAD * 2),
  });

  const toCanvasX = (x: number) => PAD + ((x - minX) / rangeX) * (CANVAS_W - PAD * 2);
  const toCanvasY = (y: number) => PAD + (1 - (y - minY) / rangeY) * (CANVAS_H - PAD * 2);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = '#18181b';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    const loSet = new Set(step?.lo || []);
    const roSet = new Set(step?.ro || []);
    const stripSet = new Set(step?.strip || []);

    // Draw divide line
    if (step?.midX && step.type !== 'done') {
      const cx = toCanvasX(step.midX);
      ctx.strokeStyle = '#71717a';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, CANVAS_H);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw strip if combine
      if (step.type === 'combine' && step.delta < Infinity) {
        const deltaPixels = (step.delta / rangeX) * (CANVAS_W - PAD * 2);
        ctx.fillStyle = 'rgba(99,102,241,0.12)';
        ctx.fillRect(cx - deltaPixels, 0, deltaPixels * 2, CANVAS_H);
        ctx.strokeStyle = '#818cf8';
        ctx.lineWidth = 1;
        ctx.strokeRect(cx - deltaPixels, 0, deltaPixels * 2, CANVAS_H);

        ctx.fillStyle = '#818cf8';
        ctx.font = '10px monospace';
        ctx.fillText(`δ = ${step.delta.toFixed(2)}`, cx + deltaPixels + 2, 18);
      }
    }

    // Draw closest pair line
    if (step?.closestPair && step.type !== 'divide') {
      const [ida, idb] = step.closestPair;
      const pa = points.find(p => p.id === ida);
      const pb = points.find(p => p.id === idb);
      if (pa && pb) {
        const a = toCanvas(pa), b = toCanvas(pb);
        ctx.strokeStyle = step.type === 'done' ? '#10b981' : '#fbbf24';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(a.cx, a.cy);
        ctx.lineTo(b.cx, b.cy);
        ctx.stroke();
      }
    }

    // Draw points
    points.forEach(p => {
      const { cx, cy } = toCanvas(p);
      const inStrip = stripSet.has(p.id);
      const isInClosest = step?.closestPair?.includes(p.id);
      const isLeft = loSet.has(p.id);
      const isRight = roSet.has(p.id);

      let color = '#6b7280';
      if (step?.type === 'done') {
        color = isInClosest ? '#10b981' : '#4f46e5';
      } else if (isInClosest) {
        color = '#f59e0b';
      } else if (isLeft) {
        color = '#3b82f6';
      } else if (isRight) {
        color = '#f97316';
      }

      if (inStrip && !isInClosest) color = '#a78bfa';

      ctx.beginPath();
      ctx.arc(cx, cy, isInClosest ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      if (isInClosest) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.fillStyle = '#d4d4d8';
      ctx.font = '10px monospace';
      ctx.fillText(`P${p.id}(${p.x},${p.y})`, cx + 8, cy - 4);
    });
  }, [step, stepIdx, points, steps]);

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* 头部 */}
      <div className="px-6 py-5 bg-slate-100 dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800">
        <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">最近点对分治可视化</h3>
        <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">Closest Pair of Points — O(n log n) 分治算法逐步演示</p>
      </div>

      <div className="p-6 space-y-6">
        {/* 预设 */}
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => { setPresetIdx(i); setStepIdx(0); }}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${presetIdx === i ? 'bg-sky-600 text-white' : 'bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 text-slate-700 dark:text-zinc-200'}`}>{p.name}</button>
          ))}
        </div>

        {/* 控制 */}
        <div className="flex gap-3 flex-wrap items-center">
          <button onClick={() => setStepIdx(s => Math.max(0, s - 1))} disabled={stepIdx === 0}
            className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">← 上一步</button>
          <button onClick={() => setStepIdx(s => Math.min(steps.length - 1, s + 1))} disabled={stepIdx >= steps.length - 1}
            className="px-4 py-2 text-sm bg-sky-600 hover:bg-sky-500 disabled:opacity-40 rounded-lg text-white transition-colors">下一步 →</button>
          <button onClick={() => setStepIdx(steps.length - 1)}
            className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white transition-colors">最终结果</button>
          <button onClick={() => setStepIdx(0)} className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">↺ 重置</button>
          <span className="text-sm text-slate-500 dark:text-zinc-400">步骤 {safeIdx+1} / {steps.length}</span>
        </div>

        {/* 步骤描述 */}
        <div className={`px-4 py-3 rounded-xl text-sm font-medium ${
          step.type === 'done' ? 'bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300' :
          step.type === 'combine' ? 'bg-purple-50 dark:bg-purple-950 border border-purple-200 dark:border-purple-700 text-purple-700 dark:text-purple-300' :
          'bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-700 text-blue-700 dark:text-blue-300'
        }`}>
          <span className="font-bold">[{step.type === 'done' ? '✅ 完成' : step.type === 'divide' ? '📤 分解' : '📥 合并'}]</span>  {step.desc}
        </div>

        {/* Canvas 可视化 */}
        <div className="rounded-xl overflow-hidden border border-slate-300 dark:border-zinc-700">
          <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} style={{ display: 'block', width: '100%', maxWidth: CANVAS_W }} />
        </div>

        {/* 图例 */}
        <div className="flex flex-wrap gap-4 text-sm text-slate-600 dark:text-zinc-400">
          <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded-full" style={{background:'#3b82f6'}} />左子集</span>
          <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded-full" style={{background:'#f97316'}} />右子集</span>
          <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded-full" style={{background:'#a78bfa'}} />Strip 内</span>
          <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded-full" style={{background:'#f59e0b'}} />当前最近对</span>
          <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded-full" style={{background:'#10b981'}} />最终结果</span>
          <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded" style={{background:'rgba(99,102,241,0.25)',border:'1px solid #818cf8'}} />Strip 区域</span>
        </div>

        {/* 最终结果 */}
        {safeIdx >= steps.length - 1 && (
          <div className="bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-700 rounded-xl p-5">
            <div className="text-sm text-slate-600 dark:text-zinc-400 mb-2">最近点对：</div>
            <div className="text-emerald-700 dark:text-emerald-400 font-mono text-base">
              P{finalPair[0]}({points[finalPair[0]].x}, {points[finalPair[0]].y}) ↔ P{finalPair[1]}({points[finalPair[1]].x}, {points[finalPair[1]].y})
            </div>
            <div className="text-emerald-600 dark:text-emerald-300 font-bold text-2xl mt-2">距离 = {finalDist.toFixed(4)}</div>
          </div>
        )}

        {/* 算法分析 */}
        <div className="bg-slate-100 dark:bg-zinc-800/60 border border-slate-200 dark:border-zinc-700 rounded-xl p-5 space-y-3">
          <div className="text-base font-semibold text-slate-700 dark:text-zinc-200">算法分析</div>
          <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm text-slate-600 dark:text-zinc-400">
            <span>暴力：<span className="text-red-500 dark:text-red-400 font-medium">O(n²)</span></span>
            <span>分治：<span className="text-green-600 dark:text-green-400 font-medium">O(n log² n)</span> 或 O(n log n)</span>
            <span>关键引理：strip 内每点最多检查 <span className="text-yellow-600 dark:text-yellow-400 font-medium">7</span> 个邻居</span>
            <span className="font-mono text-xs">T(n) = 2T(n/2) + O(n log n) → O(n log² n)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
