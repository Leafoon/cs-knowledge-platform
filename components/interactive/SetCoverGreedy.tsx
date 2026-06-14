"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Universe: elements 0..11
const U_SIZE = 12;

interface SetDef {
  name: string;
  elements: number[];
  color: {
    ring: string;
    badge: string;
    bg: string;
    text: string;
    dot: string;
  };
}

const SETS: SetDef[] = [
  {
    name: "S₀", elements: [0, 1, 2, 3, 4, 5, 6],
    color: { ring: "ring-indigo-500", badge: "bg-indigo-500", bg: "bg-indigo-50 dark:bg-indigo-900/30", text: "text-indigo-700 dark:text-indigo-300", dot: "bg-indigo-400" },
  },
  {
    name: "S₁", elements: [5, 6, 7, 8, 9, 10, 11],
    color: { ring: "ring-rose-500", badge: "bg-rose-500", bg: "bg-rose-50 dark:bg-rose-900/30", text: "text-rose-700 dark:text-rose-300", dot: "bg-rose-400" },
  },
  {
    name: "S₂", elements: [0, 1, 7, 8],
    color: { ring: "ring-amber-500", badge: "bg-amber-500", bg: "bg-amber-50 dark:bg-amber-900/30", text: "text-amber-700 dark:text-amber-300", dot: "bg-amber-400" },
  },
  {
    name: "S₃", elements: [2, 3, 9, 10],
    color: { ring: "ring-emerald-500", badge: "bg-emerald-500", bg: "bg-emerald-50 dark:bg-emerald-900/30", text: "text-emerald-700 dark:text-emerald-300", dot: "bg-emerald-400" },
  },
  {
    name: "S₄", elements: [4, 11],
    color: { ring: "ring-purple-500", badge: "bg-purple-500", bg: "bg-purple-50 dark:bg-purple-900/30", text: "text-purple-700 dark:text-purple-300", dot: "bg-purple-400" },
  },
];

interface AlgoStep {
  covered: Set<number>;
  picked: number[];           // indices of SETS chosen so far
  currentPick: number | null; // which set chosen at this step
  newlyCovered: number[];
  scores: number[];           // new coverage count for each set
  bestScore: number;
  desc: string;
  detail: string;
}

function buildSteps(): AlgoStep[] {
  const steps: AlgoStep[] = [];
  const covered = new Set<number>();
  const picked: number[] = [];

  // Compute initial scores
  function scores(cov: Set<number>): number[] {
    return SETS.map((s) => s.elements.filter((e) => !cov.has(e)).length);
  }

  // Step 0: initial
  const sc0 = scores(covered);
  steps.push({
    covered: new Set(covered),
    picked: [],
    currentPick: null,
    newlyCovered: [],
    scores: sc0,
    bestScore: Math.max(...sc0),
    desc: `初始状态：全集 U 共 ${U_SIZE} 个元素，共 ${SETS.length} 个集合`,
    detail: `贪心策略：每步选覆盖最多未覆盖元素的集合。\n◉ S₀ 和 S₁ 各覆盖 7 个元素（并列最多）。\n当前 OPT 下界：1（至少需要1个集合）。`,
  });

  // Round 1: S0 covers 7 (all 0-6, 7 new)
  {
    const s = SETS[0];
    const newCov = s.elements.filter((e) => !covered.has(e));
    newCov.forEach((e) => covered.add(e));
    picked.push(0);
    const sc = scores(covered);
    steps.push({
      covered: new Set(covered),
      picked: [...picked],
      currentPick: 0,
      newlyCovered: newCov,
      scores: sc,
      bestScore: Math.max(...sc),
      desc: `选 S₀（新覆盖 7 个元素，与 S₁ 并列最多），已覆盖 ${covered.size}/${U_SIZE}`,
      detail: `S₀ 覆盖 {0,1,2,3,4,5,6}，均为新元素。\n剩余未覆盖：{7,8,9,10,11}（5个）。\n\n再计算各集合对剩余元素的覆盖数：\nS₁ 新增 5，S₂ 新增 2，S₃ 新增 2，S₄ 新增 1。`,
    });
  }

  // Round 2: S1 covers {7,8,9,10,11} = 5 new
  {
    const s = SETS[1];
    const newCov = s.elements.filter((e) => !covered.has(e));
    newCov.forEach((e) => covered.add(e));
    picked.push(1);
    const sc = scores(covered);
    steps.push({
      covered: new Set(covered),
      picked: [...picked],
      currentPick: 1,
      newlyCovered: newCov,
      scores: sc,
      bestScore: Math.max(...sc),
      desc: `选 S₁（新覆盖 5 个元素，剩余集合中最多），已覆盖 ${covered.size}/${U_SIZE} 🎉`,
      detail: `S₁ 覆盖 {7,8,9,10,11}，均为新元素。全集已 100% 覆盖！\n\n✅ 近似比验证：\n• 贪心选了 2 个集合\n• OPT = 2（也是 S₀∪S₁）\n• 近似比 = 2/2 = 1.0（本例最优！）\n• ln(n) 上界 = ln(12) ≈ 2.49，即至多 2×2.49 ≈ 5 个集合`,
    });
  }

  return steps;
}

const STEPS = buildSteps();

const DOT_COLORS_BY_SET: Record<number, { ring: string; badge: string }> = {
  0: { ring: "ring-indigo-400", badge: "bg-indigo-500" },
  1: { ring: "ring-rose-400", badge: "bg-rose-500" },
};

export function SetCoverGreedy() {
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const cur = STEPS[step];

  useEffect(() => {
    if (auto) {
      timerRef.current = setInterval(() => {
        setStep((s) => {
          if (s >= STEPS.length - 1) { setAuto(false); return s; }
          return s + 1;
        });
      }, 1400);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [auto]);

  // Determine dot display for each element
  const getDotStyle = (e: number) => {
    const picked = cur.picked;
    if (cur.newlyCovered.includes(e)) {
      // Freshly covered this step
      const setIdx = cur.currentPick!;
      return { bg: SETS[setIdx].color.badge.replace("bg-", "bg-"), fresh: true };
    }
    if (cur.covered.has(e)) {
      // Previously covered
      const coveredBy = cur.picked.slice(0, -1).findIndex((si) => SETS[si].elements.includes(e));
      return { bg: coveredBy >= 0 ? SETS[cur.picked[coveredBy]].color.badge.replace("bg-", "bg-")+" opacity-60" : "bg-slate-400", fresh: false };
    }
    return { bg: "bg-slate-200 dark:bg-slate-600", fresh: false };
  };

  const coveragePct = Math.round((cur.covered.size / U_SIZE) * 100);

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-emerald-200 dark:border-emerald-800 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-6 py-4 flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-base font-bold text-white">集合覆盖贪心近似（ln n 近似比）</h3>
          <p className="text-xs text-emerald-100 mt-0.5">每步选覆盖未覆盖元素最多的集合，证明近似比 ≤ ln|U|</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="bg-white/20 rounded-lg px-3 py-1">
            <span className="text-xs text-white font-mono">已选: {cur.picked.length} 个集合</span>
          </div>
          <div className={`rounded-lg px-3 py-1 ${coveragePct === 100 ? "bg-emerald-400" : "bg-white/20"}`}>
            <span className="text-xs text-white font-bold font-mono">{coveragePct}% 覆盖</span>
          </div>
        </div>
      </div>

      <div className="p-5 flex flex-col gap-5">
        {/* === Universe visualization === */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
              全集 U（{U_SIZE} 个元素）
            </span>
            <span className="text-xs text-slate-400">{cur.covered.size} / {U_SIZE} 已覆盖</span>
          </div>
          <div className="grid grid-cols-12 gap-1.5">
            {Array.from({ length: U_SIZE }, (_, e) => {
              const style = getDotStyle(e);
              const isCovered = cur.covered.has(e);
              const setOwner = cur.picked.findLast?.((si: number) => SETS[si].elements.includes(e)) ?? -1;
              return (
                <div key={e} className="flex flex-col items-center gap-0.5">
                  <motion.div
                    layout
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all cursor-default
                      ${isCovered ? `${SETS[setOwner >= 0 ? setOwner : 0].color.badge} text-white shadow-sm` : "bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400"}
                      ${cur.newlyCovered.includes(e) ? "ring-4 ring-yellow-300 dark:ring-yellow-500 ring-offset-1" : ""}
                    `}
                    animate={cur.newlyCovered.includes(e) ? { scale: [1, 1.25, 1] } : { scale: 1 }}
                    transition={{ duration: 0.4 }}
                  >
                    {e}
                  </motion.div>
                  {isCovered && setOwner >= 0 && (
                    <div className={`text-[9px] font-bold ${SETS[setOwner].color.text}`}>
                      {SETS[setOwner].name}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Coverage progress bar */}
          <div className="mt-3 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full"
              animate={{ width: `${coveragePct}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        {/* === Set scores === */}
        <div>
          <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
            各集合新增覆盖数
          </div>
          <div className="grid grid-cols-5 gap-2">
            {SETS.map((s, i) => {
              const score = cur.scores[i];
              const isChosen = cur.picked.includes(i);
              const isCurrent = cur.currentPick === i;
              const isBest = score === cur.bestScore && score > 0;
              return (
                <motion.div
                  key={s.name}
                  animate={{
                    scale: isCurrent ? [1, 1.06, 1] : 1,
                    transition: { duration: 0.4 },
                  }}
                  className={`rounded-xl p-3 border-2 transition-all text-center
                    ${isCurrent ? `${s.color.bg} ${s.color.ring} ring-2 ring-offset-1` : isChosen ? `${s.color.bg} border-transparent opacity-60` : isBest ? `${s.color.bg} border-dashed ${s.color.ring.replace("ring-", "border-")}` : "bg-slate-50 dark:bg-slate-800 border-transparent opacity-50"}`
                  }
                >
                  <div className={`text-sm font-bold ${s.color.text}`}>{s.name}</div>
                  <div className="text-lg font-black text-slate-700 dark:text-slate-200">{score}</div>
                  <div className="text-[9px] text-slate-400">新增</div>
                  {(isCurrent || isChosen) && (
                    <div className={`text-[9px] mt-1 font-bold ${isCurrent ? "text-yellow-600 dark:text-yellow-400" : s.color.text}`}>
                      {isCurrent ? "✓ 本步选取" : "已选"}
                    </div>
                  )}
                  {isBest && !isCurrent && !isChosen && (
                    <div className="text-[9px] mt-1 font-bold text-amber-500">⭐ 候选</div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* === Step info === */}
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2 }}
            className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl p-4 border border-emerald-100 dark:border-emerald-800"
          >
            <p className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-1.5">{cur.desc}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed whitespace-pre-line">{cur.detail}</p>
          </motion.div>
        </AnimatePresence>

        {/* === Controls === */}
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => { setAuto(false); setStep(0); }}
            className="px-4 py-2 rounded-lg text-xs font-semibold border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            ↩ 重置
          </button>
          <button
            onClick={() => { setAuto(false); setStep((s) => Math.max(0, s - 1)); }}
            disabled={step === 0}
            className="px-4 py-2 rounded-lg text-xs font-semibold border border-emerald-200 dark:border-emerald-700 text-emerald-600 dark:text-emerald-300 disabled:opacity-30 hover:bg-emerald-50 dark:hover:bg-emerald-900/30 transition-colors"
          >
            ← 上一步
          </button>
          <button
            onClick={() => { setAuto(false); setStep((s) => Math.min(STEPS.length - 1, s + 1)); }}
            disabled={step === STEPS.length - 1}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-30 transition-colors"
          >
            下一步 →
          </button>
          <button
            onClick={() => { setStep(0); setTimeout(() => setAuto(true), 50); }}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-colors ${auto ? "bg-rose-500 hover:bg-rose-600 text-white" : "bg-slate-700 hover:bg-slate-800 text-white dark:bg-slate-600"}`}
          >
            {auto ? "⏹ 停止" : "▶ 自动演示"}
          </button>

          {/* Step dots */}
          <div className="ml-auto flex items-center gap-1.5">
            {STEPS.map((_, i) => (
              <button
                key={i}
                onClick={() => { setAuto(false); setStep(i); }}
                className={`w-2 h-2 rounded-full transition-all ${step === i ? "bg-emerald-500 w-4" : "bg-slate-300 dark:bg-slate-600"}`}
              />
            ))}
          </div>
        </div>

        {/* ln(n) bound info */}
        <div className="grid grid-cols-3 gap-3 text-center">
          {[
            { label: "ln(|U|)", value: `ln(12) ≈ 2.49`, sub: "近似比上界" },
            { label: "贪心选取", value: `${cur.picked.length} 个集合`, sub: `/ 最优解 2 个` },
            { label: "近似比", value: cur.picked.length > 0 ? `${cur.picked.length}/OPT = ${(cur.picked.length / 2).toFixed(1)}×` : "—", sub: "ln(n) 界内" },
          ].map(({ label, value, sub }) => (
            <div key={label} className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3">
              <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
              <div className="text-sm font-bold text-slate-700 dark:text-slate-200 font-mono">{value}</div>
              <div className="text-[10px] text-slate-400">{sub}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
