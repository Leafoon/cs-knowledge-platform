"use client";
import React, { useState } from "react";

/* ─── Shared types ───────────────────────────────────────────────────────── */

type SortMode = "counting" | "radix" | "bucket";

/* ═══════════════════════════════════════════════════════════════════════════
   COUNTING SORT  —  arr=[2,5,3,0,2,3,0,3], k=5
   Phases: Count → Prefix Sum → Place (right to left)
══════════════════════════════════════════════════════════════════════════════ */

const CS_ARR  = [2, 5, 3, 0, 2, 3, 0, 3];
const CS_K    = 5;

interface CSStep {
  phase: "count" | "prefix" | "place" | "done";
  phaseLabel: string;
  count: number[];
  output: (number | null)[];
  activeInput: number | null;    // index in arr being processed
  activeCount: number | null;    // index in count being updated
  activeOutput: number | null;   // index in output being written
  description: string;
}

function buildCountingSteps(): CSStep[] {
  const steps: CSStep[] = [];
  let count = Array(CS_K + 1).fill(0);
  let output: (number | null)[] = Array(CS_ARR.length).fill(null);

  // Phase 1: Count
  steps.push({ phase: "count", phaseLabel: "阶段 1：计数", count: [...count], output: [...output], activeInput: null, activeCount: null, activeOutput: null, description: "初始：count[] 全为 0，准备统计每个值的出现次数" });
  for (let i = 0; i < CS_ARR.length; i++) {
    const v = CS_ARR[i];
    count = [...count];
    count[v]++;
    steps.push({ phase: "count", phaseLabel: "阶段 1：计数", count: [...count], output: [...output], activeInput: i, activeCount: v, activeOutput: null, description: `处理 arr[${i}]=${v}：count[${v}]++ → count[${v}]=${count[v]}` });
  }
  steps.push({ phase: "count", phaseLabel: "阶段 1：计数完成", count: [...count], output: [...output], activeInput: null, activeCount: null, activeOutput: null, description: `计数完成：count = [${count.join(",")}]，count[v] = 值 v 出现的次数` });

  // Phase 2: Prefix sum
  steps.push({ phase: "prefix", phaseLabel: "阶段 2：前缀和", count: [...count], output: [...output], activeInput: null, activeCount: null, activeOutput: null, description: "对 count 求前缀和：count[i] 将表示 ≤i 的元素总数（= 值 i 最终位置的右边界）" });
  for (let i = 1; i <= CS_K; i++) {
    count = [...count];
    count[i] += count[i - 1];
    steps.push({ phase: "prefix", phaseLabel: "阶段 2：前缀和", count: [...count], output: [...output], activeInput: null, activeCount: i, activeOutput: null, description: `count[${i}] += count[${i-1}] → count[${i}] = ${count[i]}（值 ≤ ${i} 共有 ${count[i]} 个）` });
  }
  steps.push({ phase: "prefix", phaseLabel: "阶段 2：前缀和完成", count: [...count], output: [...output], activeInput: null, activeCount: null, activeOutput: null, description: `前缀和完成：count = [${count.join(",")}]。count[v] = output 中值 v 的最右允许位置（+1）` });

  // Phase 3: Place (right to left)
  steps.push({ phase: "place", phaseLabel: "阶段 3：从右向左放置", count: [...count], output: [...output], activeInput: null, activeCount: null, activeOutput: null, description: "从右向左遍历 arr，保证相同值的元素保持原始顺序（稳定性关键！）" });
  for (let i = CS_ARR.length - 1; i >= 0; i--) {
    const v = CS_ARR[i];
    const pos = count[v] - 1;
    output = [...output];
    output[pos] = v;
    count = [...count];
    count[v]--;
    steps.push({ phase: "place", phaseLabel: "阶段 3：放置", count: [...count], output: [...output], activeInput: i, activeCount: v, activeOutput: pos, description: `arr[${i}]=${v}：放到 output[count[${v}+1]-1]=output[${pos}]，count[${v}]-- → ${count[v]}` });
  }
  steps.push({ phase: "done", phaseLabel: "✅ 完成", count: [...count], output: [...output], activeInput: null, activeCount: null, activeOutput: null, description: `计数排序完成！output = [${output.join(",")}]，时间 Θ(n+k) = Θ(8+5) = Θ(13)` });
  return steps;
}

function CountingSortViz() {
  const steps = React.useMemo(buildCountingSteps, []);
  const [step, setStep] = useState(0);
  const cur = steps[step];
  const total = steps.length - 1;

  const phaseColor: Record<string, string> = {
    count:  "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800",
    prefix: "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-800",
    place:  "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800",
    done:   "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800",
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${phaseColor[cur.phase]}`}>{cur.phaseLabel}</span>
        <span className="text-xs font-mono text-slate-400 dark:text-slate-500">{step}/{total}</span>
      </div>

      {/* Input array */}
      <div>
        <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">输入 arr = [2,5,3,0,2,3,0,3]</p>
        <div className="flex gap-1.5">
          {CS_ARR.map((v, i) => (
            <div key={i} className="flex flex-col items-center gap-0.5">
              <div className={`w-9 h-9 rounded-lg border-2 flex items-center justify-center text-sm font-mono font-bold transition-all duration-200
                ${cur.activeInput === i
                  ? "bg-amber-400 dark:bg-amber-600 border-amber-500 dark:border-amber-400 text-white scale-110"
                  : "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300"}`}>
                {v}
              </div>
              <span className="text-[9px] font-mono text-slate-400 dark:text-slate-600">{i}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Count array */}
      <div>
        <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">count[0..{CS_K}]</p>
        <div className="flex gap-1.5">
          {cur.count.map((v, i) => (
            <div key={i} className="flex flex-col items-center gap-0.5">
              <div className={`w-9 h-9 rounded-lg border-2 flex items-center justify-center text-sm font-mono font-bold transition-all duration-200
                ${cur.activeCount === i
                  ? "bg-indigo-500 dark:bg-indigo-600 border-indigo-600 dark:border-indigo-400 text-white scale-110"
                  : v > 0
                    ? "bg-indigo-100 dark:bg-indigo-900/40 border-indigo-300 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300"
                    : "bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-600"}`}>
                {v}
              </div>
              <span className="text-[9px] font-mono text-slate-400 dark:text-slate-600">{i}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Output array */}
      <div>
        <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">output[]（构建中）</p>
        <div className="flex gap-1.5">
          {cur.output.map((v, i) => (
            <div key={i} className="flex flex-col items-center gap-0.5">
              <div className={`w-9 h-9 rounded-lg border-2 flex items-center justify-center text-sm font-mono font-bold transition-all duration-200
                ${cur.activeOutput === i
                  ? "bg-emerald-400 dark:bg-emerald-600 border-emerald-500 dark:border-emerald-400 text-white scale-110"
                  : v !== null
                    ? "bg-emerald-100 dark:bg-emerald-900/30 border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300"
                    : "bg-slate-50 dark:bg-slate-800/50 border-dashed border-slate-300 dark:border-slate-700 text-slate-300 dark:text-slate-700"}`}>
                {v !== null ? v : "·"}
              </div>
              <span className="text-[9px] font-mono text-slate-400 dark:text-slate-600">{i}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Description */}
      <div className={`rounded-xl border p-3 text-xs ${phaseColor[cur.phase]}`}>
        {cur.description}
      </div>

      <Controls step={step} total={total} onPrev={() => setStep(s => Math.max(0, s - 1))} onNext={() => setStep(s => Math.min(total, s + 1))} onReset={() => setStep(0)} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   RADIX SORT  —  arr=[170, 45, 75, 90, 802, 24, 2, 66], base=10, 3 rounds
══════════════════════════════════════════════════════════════════════════════ */

const RS_ARR = [170, 45, 75, 90, 802, 24, 2, 66];

interface RSStep {
  round: number;       // 0=units, 1=tens, 2=hundreds
  arr: number[];
  buckets: Record<number, number[]> | null;  // digit → elements
  activeDigit: number | null;
  description: string;
}

function buildRadixSteps(): RSStep[] {
  const steps: RSStep[] = [];
  let arr = [...RS_ARR];
  steps.push({ round: 0, arr: [...arr], buckets: null, activeDigit: null, description: "初始数组，按最低有效位（个位）开始，LSD → MSD 依次稳定排序" });

  for (let round = 0; round < 3; round++) {
    const exp = Math.pow(10, round);
    const buckets: Record<number, number[]> = {};
    for (let d = 0; d <= 9; d++) buckets[d] = [];

    for (const v of arr) {
      const d = Math.floor(v / exp) % 10;
      buckets[d].push(v);
    }
    steps.push({ round, arr: [...arr], buckets: { ...buckets }, activeDigit: round, description: `第 ${round + 1} 轮：按${["个", "十", "百"][round]}位分桶（基数 = ${exp}）` });

    // Collect from buckets
    arr = Object.values(buckets).flat();
    steps.push({ round, arr: [...arr], buckets: null, activeDigit: null, description: `第 ${round + 1} 轮收集：从桶 0→9 依次取出，稳定排序！结果 = [${arr.join(",")}]` });
  }
  steps.push({ round: 2, arr: [...arr], buckets: null, activeDigit: null, description: `✅ 基数排序完成！[${arr.join(",")}]，3 轮 × O(n+10) = O(n)` });
  return steps;
}

const DIGIT_COLORS = [
  "bg-rose-100 dark:bg-rose-900/30 border-rose-300 dark:border-rose-700 text-rose-700 dark:text-rose-300",
  "bg-orange-100 dark:bg-orange-900/30 border-orange-300 dark:border-orange-700 text-orange-700 dark:text-orange-300",
  "bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300",
];

function RadixSortViz() {
  const steps = React.useMemo(buildRadixSteps, []);
  const [step, setStep] = useState(0);
  const cur = steps[step];
  const total = steps.length - 1;
  const exp = cur.round >= 0 ? Math.pow(10, cur.round) : 1;

  const digitOf = (v: number, e: number) => Math.floor(v / e) % 10;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        {[0, 1, 2].map(r => (
          <div key={r} className={`px-2.5 py-1 rounded-full text-xs font-semibold border ${cur.round === r && cur.activeDigit !== null ? DIGIT_COLORS[r] : "bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-500"}`}>
            {["个位（exp=1）", "十位（exp=10）", "百位（exp=100）"][r]}
          </div>
        ))}
        <span className="ml-auto text-xs font-mono text-slate-400 dark:text-slate-500">{step}/{total}</span>
      </div>

      {/* Current array */}
      <div>
        <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">当前数组（排序中）</p>
        <div className="flex flex-wrap gap-1.5">
          {cur.arr.map((v, i) => {
            const d = cur.activeDigit !== null ? digitOf(v, exp) : null;
            return (
              <div key={i} className="flex flex-col items-center gap-0.5">
                <div className={`min-w-[48px] h-10 rounded-lg border-2 flex items-center justify-center text-sm font-mono font-bold transition-all duration-200 px-1
                  ${cur.activeDigit !== null ? DIGIT_COLORS[cur.round] : "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300"}`}>
                  {/* Highlight the relevant digit */}
                  {d !== null
                    ? <span>{String(v).padStart(3, ' ').split("").map((ch, ci) => (
                        <span key={ci} className={ci === 2 - cur.round ? "underline font-extrabold" : "opacity-60"}>{ch}</span>
                      ))}</span>
                    : v}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Buckets */}
      {cur.buckets && (
        <div>
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">分桶结果（桶 0 → 9）</p>
          <div className="grid grid-cols-5 gap-1.5">
            {Object.entries(cur.buckets).map(([d, vals]) => (
              <div key={d} className={`rounded-lg border p-2 min-h-[52px] ${vals.length > 0 ? DIGIT_COLORS[cur.round] : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700"}`}>
                <div className="text-[9px] font-bold mb-1 opacity-60">桶 {d}</div>
                <div className="flex flex-wrap gap-0.5">
                  {vals.map((v, i) => (
                    <span key={i} className="text-xs font-mono font-bold">{v}</span>
                  ))}
                  {vals.length === 0 && <span className="text-[10px] opacity-30">空</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Description */}
      <div className={`rounded-xl border p-3 text-xs ${DIGIT_COLORS[cur.round]}`}>
        {cur.description}
      </div>

      <Controls step={step} total={total} onPrev={() => setStep(s => Math.max(0, s - 1))} onNext={() => setStep(s => Math.min(total, s + 1))} onReset={() => setStep(0)} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   BUCKET SORT  —  10 values in [0, 1), 10 buckets
══════════════════════════════════════════════════════════════════════════════ */

const BKT_ARR = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68];

interface BktStep {
  phase: "initial" | "distribute" | "sort" | "concat";
  buckets: { elements: number[]; sorted: boolean }[];
  activeElement: number | null;
  activeBucket: number | null;
  result: number[] | null;
  description: string;
}

function buildBucketSteps(): BktStep[] {
  const steps: BktStep[] = [];
  const n = BKT_ARR.length;
  let buckets: { elements: number[]; sorted: boolean }[] = Array.from({ length: n }, () => ({ elements: [], sorted: false }));

  steps.push({ phase: "initial", buckets: buckets.map(b => ({ ...b })), activeElement: null, activeBucket: null, result: null, description: `初始：10 个桶对应 [0, 0.1)、[0.1, 0.2)、…、[0.9, 1.0)；均匀分布保证每桶期望 O(1) 个元素` });

  // Distribute
  for (let i = 0; i < BKT_ARR.length; i++) {
    const v = BKT_ARR[i];
    const bi = Math.min(Math.floor(v * n), n - 1);
    buckets = buckets.map((b, idx) => idx === bi ? { ...b, elements: [...b.elements, v] } : { ...b });
    steps.push({ phase: "distribute", buckets: buckets.map(b => ({ ...b })), activeElement: i, activeBucket: bi, result: null, description: `元素 ${v} → 桶 ${bi}（⌊${v}×${n}⌋=${bi}，属于 [${(bi/n).toFixed(1)}, ${((bi+1)/n).toFixed(1)})）` });
  }

  // Sort each bucket
  buckets = buckets.map(b => ({ elements: [...b.elements].sort((a, c) => a - c), sorted: true }));
  steps.push({ phase: "sort", buckets: buckets.map(b => ({ ...b })), activeElement: null, activeBucket: null, result: null, description: "对每个桶内部排序（少量元素用插入排序，平均 O(1) 次比较）" });

  // Concatenate
  const result = buckets.flatMap(b => b.elements);
  steps.push({ phase: "concat", buckets: buckets.map(b => ({ ...b })), activeElement: null, activeBucket: null, result, description: `✅ 拼接所有桶 → [${result.map(v => v.toFixed(2)).join(",")}]，期望 O(n)` });
  return steps;
}

function BucketSortViz() {
  const steps = React.useMemo(buildBucketSteps, []);
  const [step, setStep] = useState(0);
  const cur = steps[step];
  const total = steps.length - 1;
  const n = BKT_ARR.length;

  const BUCKET_COLORS = ["bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-800", "bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800", "bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800", "bg-lime-50 dark:bg-lime-900/20 border-lime-200 dark:border-lime-800", "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800", "bg-teal-50 dark:bg-teal-900/20 border-teal-200 dark:border-teal-800", "bg-sky-50 dark:bg-sky-900/20 border-sky-200 dark:border-sky-800", "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800", "bg-violet-50 dark:bg-violet-900/20 border-violet-200 dark:border-violet-800", "bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800"];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {(["distribute", "sort", "concat"] as const).map(ph => (
            <span key={ph} className={`w-3 h-3 rounded-full ${cur.phase === ph ? "bg-emerald-500" : "bg-slate-200 dark:bg-slate-700"}`} />
          ))}
        </div>
        <span className="text-xs font-mono text-slate-400 dark:text-slate-500">{step}/{total}</span>
      </div>

      {/* Input array */}
      <div>
        <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">输入（均匀分布 [0,1)）</p>
        <div className="flex flex-wrap gap-1">
          {BKT_ARR.map((v, i) => (
            <div key={i} className={`px-2 py-1 rounded-md border text-xs font-mono font-bold transition-all ${cur.activeElement === i ? "bg-amber-400 dark:bg-amber-600 border-amber-500 text-white scale-110" : "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300"}`}>
              {v.toFixed(2)}
            </div>
          ))}
        </div>
      </div>

      {/* Buckets grid */}
      <div className="grid grid-cols-5 gap-1.5">
        {cur.buckets.map((b, bi) => (
          <div key={bi} className={`rounded-xl border-2 p-2 min-h-[56px] transition-all duration-200 ${cur.activeBucket === bi ? "ring-2 ring-amber-400 dark:ring-amber-500 scale-105 shadow-md " + BUCKET_COLORS[bi] : BUCKET_COLORS[bi]}`}>
            <div className="text-[9px] font-bold opacity-60 mb-1">[{(bi/n).toFixed(1)},{((bi+1)/n).toFixed(1)})</div>
            <div className="flex flex-col gap-0.5">
              {b.elements.map((v, i) => (
                <span key={i} className={`text-[10px] font-mono font-bold ${b.sorted ? "text-emerald-700 dark:text-emerald-300" : "text-slate-700 dark:text-slate-300"}`}>
                  {v.toFixed(2)}{b.sorted && b.elements.length > 1 ? " ✓" : ""}
                </span>
              ))}
              {b.elements.length === 0 && <span className="text-[10px] opacity-30">·</span>}
            </div>
          </div>
        ))}
      </div>

      {/* Result */}
      {cur.result && (
        <div className="flex flex-wrap gap-1">
          {cur.result.map((v, i) => (
            <span key={i} className="px-2 py-1 rounded-md bg-emerald-100 dark:bg-emerald-900/30 border border-emerald-300 dark:border-emerald-700 text-xs font-mono font-bold text-emerald-700 dark:text-emerald-300">
              {v.toFixed(2)}
            </span>
          ))}
        </div>
      )}

      {/* Description */}
      <div className="rounded-xl border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/20 p-3 text-xs text-emerald-700 dark:text-emerald-300">
        {cur.description}
      </div>

      <Controls step={step} total={total} onPrev={() => setStep(s => Math.max(0, s - 1))} onNext={() => setStep(s => Math.min(total, s + 1))} onReset={() => setStep(0)} />
    </div>
  );
}

/* ─── Shared controls ─────────────────────────────────────────────────────── */

function Controls({ step, total, onPrev, onNext, onReset }: { step: number; total: number; onPrev: () => void; onNext: () => void; onReset: () => void }) {
  return (
    <div className="space-y-2">
      <div className="h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
        <div className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-emerald-500 transition-all duration-300" style={{ width: `${(step / total) * 100}%` }} />
      </div>
      <div className="flex items-center gap-2 justify-center">
        <button onClick={onReset} className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">重置</button>
        <button onClick={onPrev} disabled={step === 0} className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">← 上一步</button>
        <button onClick={onNext} disabled={step >= total} className="px-4 py-1.5 text-xs rounded-lg bg-indigo-500 hover:bg-indigo-600 text-white font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors">下一步 →</button>
      </div>
    </div>
  );
}

/* ─── Main export ─────────────────────────────────────────────────────────── */

const TABS: { id: SortMode; label: string; subtitle: string; color: string }[] = [
  { id: "counting", label: "计数排序", subtitle: "Θ(n+k)，无比较", color: "bg-blue-500" },
  { id: "radix",    label: "基数排序", subtitle: "Θ(d(n+b))，LSD 逐位", color: "bg-purple-500" },
  { id: "bucket",   label: "桶排序",   subtitle: "期望 O(n)，均匀分布", color: "bg-emerald-500" },
];

export default function LinearSortDemo() {
  const [mode, setMode] = useState<SortMode>("counting");

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden shadow-sm">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-700">
        <h3 className="font-bold text-slate-800 dark:text-slate-100 text-base">线性时间排序——三种算法过程对比</h3>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">不依赖元素间两两比较，利用输入数据的额外结构突破 Ω(n log n) 下界</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 dark:border-slate-700">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setMode(t.id)}
            className={`flex-1 px-4 py-3 text-left transition-colors ${mode === t.id ? "bg-slate-50 dark:bg-slate-800/70 border-b-2 border-indigo-500" : "hover:bg-slate-50 dark:hover:bg-slate-800/40"}`}
          >
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${t.color} flex-shrink-0`} />
              <span className={`text-sm font-semibold ${mode === t.id ? "text-slate-800 dark:text-slate-100" : "text-slate-500 dark:text-slate-400"}`}>{t.label}</span>
            </div>
            <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5 ml-4">{t.subtitle}</p>
          </button>
        ))}
      </div>

      <div className="p-5">
        {mode === "counting" && <CountingSortViz />}
        {mode === "radix"    && <RadixSortViz />}
        {mode === "bucket"   && <BucketSortViz />}
      </div>
    </div>
  );
}
