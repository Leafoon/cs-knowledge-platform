"use client";
import React, { useState, useCallback, useMemo } from "react";

// ─── Universal Hashing: h_{a,b}(k) = ((a*k + b) mod p) mod m ───────────────
// p is prime larger than universe U; a ∈ {1..p-1}, b ∈ {0..p-1}
const P = 1000003; // prime larger than key universe
const M_SLOTS = 17; // prime number of slots
const N_KEYS = 12;

// Pre-fixed key universe (integer keys)
const KEY_UNIVERSE: number[] = [
  7, 14, 23, 38, 45, 56, 61, 72, 89, 97, 103, 118,
];

function universalHash(key: number, a: number, b: number, m: number): number {
  return ((a * key + b) % P) % m;
}

function countCollisions(buckets: number[][]): number {
  let c = 0;
  for (const bucket of buckets) {
    const len = bucket.length;
    c += (len * (len - 1)) / 2;
  }
  return c;
}

function buildBuckets(keys: number[], a: number, b: number): number[][] {
  const buckets: number[][] = Array.from({ length: M_SLOTS }, () => []);
  for (const k of keys) {
    buckets[universalHash(k, a, b, M_SLOTS)].push(k);
  }
  return buckets;
}

// Bad hash: h(k) = k mod m — all keys map to 0 when divisible by m
function badHash(key: number): number {
  return key % M_SLOTS;
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function UniversalHashDemo() {
  const [currentA, setCurrentA] = useState(123);
  const [currentB, setCurrentB] = useState(456);
  const [trialHistory, setTrialHistory] = useState<number[]>([]);
  const [showBad, setShowBad] = useState(false);

  const currentBuckets = useMemo(() => buildBuckets(KEY_UNIVERSE, currentA, currentB), [currentA, currentB]);
  const currentCollisions = useMemo(() => countCollisions(currentBuckets), [currentBuckets]);

  // Bad hash buckets
  const badBuckets = useMemo(() => {
    const buckets: number[][] = Array.from({ length: M_SLOTS }, () => []);
    for (const k of KEY_UNIVERSE) buckets[badHash(k)].push(k);
    return buckets;
  }, []);
  const badCollisions = countCollisions(badBuckets);

  // Expected collisions: n*(n-1)/(2m)
  const expectedCollisions = (N_KEYS * (N_KEYS - 1)) / (2 * M_SLOTS);

  const runOneTrial = useCallback(() => {
    const a = Math.floor(Math.random() * (P - 1)) + 1;
    const b = Math.floor(Math.random() * P);
    setCurrentA(a);
    setCurrentB(b);
    const buckets = buildBuckets(KEY_UNIVERSE, a, b);
    const cols = countCollisions(buckets);
    setTrialHistory(prev => [cols, ...prev].slice(0, 60));
  }, []);

  const run100Trials = useCallback(() => {
    const results: number[] = [];
    for (let i = 0; i < 100; i++) {
      const a = Math.floor(Math.random() * (P - 1)) + 1;
      const b = Math.floor(Math.random() * P);
      const buckets = buildBuckets(KEY_UNIVERSE, a, b);
      results.push(countCollisions(buckets));
    }
    setTrialHistory(prev => [...results, ...prev].slice(0, 200));
    // Set last trial as current display
    const last = results[results.length - 1];
    // find a/b pair for last (just use any that gives that count)
    setCurrentA(results[0]);
    setCurrentB(results[1] ?? 1);
  }, []);

  // Histogram of trial collision counts
  const histogram = useMemo(() => {
    if (trialHistory.length === 0) return [];
    const max = Math.max(...trialHistory);
    const bins: { count: number; key: number }[] = [];
    for (let i = 0; i <= max; i++) {
      bins.push({ key: i, count: trialHistory.filter(x => x === i).length });
    }
    return bins;
  }, [trialHistory]);
  const histMax = Math.max(...(histogram.map(b => b.count)), 1);

  const avgCollisions = trialHistory.length > 0
    ? (trialHistory.reduce((a, b) => a + b, 0) / trialHistory.length).toFixed(2)
    : "—";

  const maxBucketLen = Math.max(...currentBuckets.map(b => b.length), 1);
  const badMaxBucketLen = Math.max(...badBuckets.map(b => b.length), 1);

  const renderBuckets = (buckets: number[][], isBad: boolean) => (
    <div className="flex gap-1 items-end overflow-x-auto pb-1">
      {buckets.map((bucket, idx) => {
        const len = bucket.length;
        const isWorst = len === (isBad ? badMaxBucketLen : maxBucketLen) && len > 1;
        return (
          <div key={idx} className="flex flex-col items-center gap-0.5 flex-shrink-0">
            <div className="flex flex-col-reverse gap-0.5">
              {bucket.map((key, ki) => (
                <div key={ki} className={`w-8 h-5 flex items-center justify-center rounded text-white text-xs font-bold ${
                  isWorst ? "bg-red-600" : len > 1 ? "bg-amber-600" : "bg-emerald-700"
                }`} style={{ fontSize: 9 }}>{key}</div>
              ))}
              {len === 0 && <div className="w-8 h-3 flex items-center justify-center text-slate-600 text-xs">∅</div>}
            </div>
            <span className="text-slate-500 text-xs" style={{ fontSize: 9 }}>{idx}</span>
          </div>
        );
      })}
    </div>
  );

  return (
    <div className="dark isolate rounded-2xl border border-slate-700 bg-slate-900 p-5 space-y-5 font-mono text-sm text-slate-200">
      {/* Header */}
      <div>
        <h3 className="text-base font-bold text-white">🎲 全域哈希 (Universal Hashing) 模拟器</h3>
        <p className="text-slate-400 text-xs mt-0.5">随机选择散列函数，冲突期望 ≤ n(n-1)/(2m)，对抗恶意输入</p>
      </div>

      {/* Theory box */}
      <div className="rounded-lg border border-violet-800 bg-violet-950/30 p-3 text-xs space-y-1">
        <p className="text-violet-300 font-bold">全域哈希族：h<sub>a,b</sub>(k) = ((a·k + b) mod p) mod m</p>
        <p className="text-slate-400">其中 a ∈ &lbrace;1, ..., p-1&rbrace;, b ∈ &lbrace;0, ..., p-1&rbrace;, p={P} (质数), m={M_SLOTS} (槽数)</p>
        <p className="text-slate-300">期望冲突次数 ≤ <span className="text-violet-300 font-bold">n(n-1)/(2m) = {expectedCollisions.toFixed(2)}</span>（对任意键集合成立）</p>
        <p className="text-slate-400">n={N_KEYS} 个键：[{KEY_UNIVERSE.join(", ")}]</p>
      </div>

      {/* Controls */}
      <div className="flex gap-3 flex-wrap">
        <button onClick={runOneTrial} className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold transition-colors">
          随机选一个函数 (1 次试验)
        </button>
        <button onClick={run100Trials} className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-colors">
          运行 100 次试验
        </button>
        <button onClick={() => setShowBad(!showBad)} className={`px-4 py-2 rounded-lg text-xs font-bold transition-colors ${showBad ? "bg-red-700 hover:bg-red-600 text-white" : "bg-slate-700 hover:bg-slate-600 text-slate-300"}`}>
          {showBad ? "隐藏" : "显示"} 固定差劲哈希对比
        </button>
      </div>

      {/* Current function info */}
      <div className="flex gap-3 text-xs flex-wrap items-center">
        <span className="px-2 py-1 rounded bg-slate-800 text-violet-300 font-bold">a={currentA}, b={currentB}</span>
        <span className={`px-2 py-1 rounded font-bold ${currentCollisions <= Math.ceil(expectedCollisions) ? "bg-emerald-900 text-emerald-300" : currentCollisions <= Math.ceil(expectedCollisions * 1.5) ? "bg-amber-900 text-amber-300" : "bg-red-900 text-red-300"}`}>
          当前冲突次数={currentCollisions}
        </span>
        <span className="px-2 py-1 rounded bg-slate-800 text-slate-400">期望={expectedCollisions.toFixed(2)}</span>
        {trialHistory.length > 0 && (
          <span className="px-2 py-1 rounded bg-slate-800 text-sky-300">
            已运行 {trialHistory.length} 次, 平均冲突={avgCollisions}
          </span>
        )}
      </div>

      {/* Current hash bucket visualization */}
      <div className="grid grid-cols-1 gap-4" style={{ gridTemplateColumns: showBad ? "1fr 1fr" : "1fr" }}>
        <div>
          <p className="text-slate-400 text-xs mb-2 font-semibold">
            全域哈希 h<sub>{currentA},{currentB}</sub>：冲突={currentCollisions}
            <span className={`ml-2 ${currentCollisions <= Math.ceil(expectedCollisions) ? "text-emerald-400" : "text-amber-400"}`}>
              {currentCollisions <= Math.ceil(expectedCollisions) ? "✓ 符合期望" : "△ 略高于期望"}
            </span>
          </p>
          {renderBuckets(currentBuckets, false)}
        </div>
        {showBad && (
          <div>
            <p className="text-red-400 text-xs mb-2 font-semibold">
              固定哈希 h(k)=k mod {M_SLOTS}：冲突={badCollisions}
              <span className="text-red-300 ml-2">⚠️ 可被攻击者利用</span>
            </p>
            {renderBuckets(badBuckets, true)}
          </div>
        )}
      </div>

      {/* Collision count legend */}
      <div className="flex gap-3 text-xs">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-emerald-700 inline-block" />无冲突</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-600 inline-block" />有冲突</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-600 inline-block" />最长链</span>
      </div>

      {/* Histogram */}
      {histogram.length > 0 && (
        <div>
          <p className="text-slate-400 text-xs mb-2">碰撞次数分布直方图（{trialHistory.length} 次试验）：</p>
          <div className="flex gap-0.5 items-end overflow-x-auto h-20 pb-4">
            {histogram.map(bin => (
              <div key={bin.key} className="flex flex-col items-center gap-0.5 flex-shrink-0">
                <div className={`w-5 rounded-t transition-all duration-300 ${bin.key === 0 ? "bg-emerald-600" : bin.key <= Math.ceil(expectedCollisions) ? "bg-sky-600" : "bg-red-700"}`}
                  style={{ height: `${Math.round(bin.count / histMax * 60)}px` }} />
                <span className="text-slate-500 leading-none" style={{ fontSize: 8 }}>{bin.key}</span>
              </div>
            ))}
            {/* Expected value marker */}
            <div className="absolute-placeholder ml-1 flex flex-col items-center">
              <div className="w-0.5 h-16 bg-violet-400 opacity-50" style={{ marginTop: 2 }} />
              <span className="text-violet-400 text-xs" style={{ fontSize: 8 }}>E={expectedCollisions.toFixed(1)}</span>
            </div>
          </div>
          <p className="text-slate-500 text-xs">横轴=碰撞次数, 纵轴=出现频率, 蓝色=≤期望, 红色=超出期望, 绿色=零碰撞</p>
        </div>
      )}

      {/* Insight */}
      <div className="rounded-lg bg-slate-800/60 border border-slate-700 p-3 text-xs space-y-1">
        <p className="text-slate-300 font-bold">🔑 全域哈希的意义</p>
        <p className="text-slate-400">无论输入数据什么分布，只要随机选择散列函数，期望冲突次数可以控制在 <span className="text-violet-300">O(n/m)</span> 以内。</p>
        <p className="text-slate-400">即使攻击者知道所有可能的哈希函数，也无法提前构造使所有冲突集中的输入集合。</p>
        <p className="text-slate-500 mt-1">参考：CLRS Section 11.3.3，证明 ℋ 是全域哈希族 ⟺ Pr[h(x)=h(y)] ≤ 1/m ∀x≠y</p>
      </div>
    </div>
  );
}
