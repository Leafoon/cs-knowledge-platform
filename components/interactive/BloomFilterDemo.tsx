"use client";
import React, { useState, useCallback, useMemo } from "react";

// ─── Hash functions for Bloom Filter ──────────────────────────────────────────
function bloomHash(key: string, seed: number, m: number): number {
  let h = seed * 2654435761;
  for (let i = 0; i < key.length; i++) {
    h = Math.imul(h ^ key.charCodeAt(i), 0x9e3779b9);
  }
  return ((h >>> 0) % m);
}

// ─── Types ────────────────────────────────────────────────────────────────────
interface HashResult {
  fn: number; // 0-based index of hash function
  pos: number;
}

const HASH_COLORS = ["text-rose-400", "text-sky-400", "text-violet-400", "text-emerald-400", "text-amber-400", "text-pink-400"];
const HASH_BG     = ["bg-rose-600",   "bg-sky-600",   "bg-violet-600",   "bg-emerald-600",   "bg-amber-600",   "bg-pink-600"];
const HASH_RING   = ["ring-rose-400", "ring-sky-400", "ring-violet-400", "ring-emerald-400", "ring-amber-400", "ring-pink-400"];

const M_OPTIONS = [16, 32, 64];
const PRESET_INSERTS = ["apple", "banana", "cherry", "dragonfruit", "elder"];

// ─── Component ────────────────────────────────────────────────────────────────
export default function BloomFilterDemo() {
  const [m, setM] = useState(32);
  const [k, setK] = useState(3);
  const [bits, setBits] = useState<boolean[]>(() => Array(32).fill(false));
  const [insertedSet, setInsertedSet] = useState<Set<string>>(new Set());
  const [queryKey, setQueryKey] = useState("");
  const [insertKey, setInsertKey] = useState("");
  const [queryResult, setQueryResult] = useState<null | { found: boolean; fp: boolean; positions: HashResult[] }>(null);
  const [animPositions, setAnimPositions] = useState<HashResult[]>([]);
  const [log, setLog] = useState<string[]>([]);

  // Reset when m or k changes
  const resetFilter = useCallback((newM: number, newK: number) => {
    setM(newM);
    setK(newK);
    setBits(Array(newM).fill(false));
    setInsertedSet(new Set());
    setQueryResult(null);
    setAnimPositions([]);
    setLog([`🔄 重置过滤器 (m=${newM}, k=${newK})`]);
  }, []);

  const getHashes = useCallback((key: string, _m: number, _k: number): HashResult[] => {
    return Array.from({ length: _k }, (_, i) => ({
      fn: i,
      pos: bloomHash(key, i + 1, _m),
    }));
  }, []);

  const bitsSet = useMemo(() => bits.filter(Boolean).length, [bits]);
  const n = insertedSet.size;
  // FPR = (1 - e^(-kn/m))^k
  const fpr = useMemo(() => {
    if (n === 0) return 0;
    return Math.pow(1 - Math.exp(-k * n / m), k);
  }, [k, m, n]);

  const preload = useCallback(() => {
    const newBits = Array(m).fill(false);
    const newSet = new Set<string>();
    for (const word of PRESET_INSERTS) {
      const hashes = getHashes(word, m, k);
      for (const h of hashes) newBits[h.pos] = true;
      newSet.add(word);
    }
    setBits(newBits);
    setInsertedSet(newSet);
    setQueryResult(null);
    setAnimPositions([]);
    setLog([`📦 已预加载 ${PRESET_INSERTS.length} 个元素`]);
  }, [m, k, getHashes]);

  const doInsert = useCallback(() => {
    const key = insertKey.trim();
    if (!key) return;
    const hashes = getHashes(key, m, k);
    setAnimPositions(hashes);
    setTimeout(() => setAnimPositions([]), 1800);
    const newBits = [...bits];
    for (const h of hashes) newBits[h.pos] = true;
    setBits(newBits);
    setInsertedSet(prev => new Set(prev).add(key));
    setInsertKey("");
    setQueryResult(null);
    const positions = hashes.map(h => `h${h.fn + 1}→${h.pos}`).join(", ");
    setLog(prev => [`✅ INSERT "${key}" [${positions}]`, ...prev].slice(0, 6));
  }, [insertKey, bits, m, k, getHashes]);

  const doQuery = useCallback(() => {
    const key = queryKey.trim();
    if (!key) return;
    const hashes = getHashes(key, m, k);
    const allSet = hashes.every(h => bits[h.pos]);
    const reallyIn = insertedSet.has(key);
    const fp = allSet && !reallyIn;
    setQueryResult({ found: allSet, fp, positions: hashes });
    setAnimPositions(hashes);
    setTimeout(() => setAnimPositions([]), 2000);
    const verdict = allSet ? (fp ? "🚨 误判 (false positive)" : "✅ 可能存在") : "✅ 确定不存在";
    setLog(prev => [`🔍 QUERY "${key}" → ${verdict}`, ...prev].slice(0, 6));
  }, [queryKey, bits, m, k, getHashes, insertedSet]);

  // Which positions are currently highlighted during animation
  const animPosSet = useMemo(() => new Set(animPositions.map(h => h.pos)), [animPositions]);
  const animPosColor = useMemo<Map<number, number>>(() => {
    const map = new Map<number, number>();
    for (const h of animPositions) map.set(h.pos, h.fn);
    return map;
  }, [animPositions]);

  const queryPosSet = useMemo(() => queryResult ? new Set(queryResult.positions.map(h => h.pos)) : new Set<number>(), [queryResult]);

  return (
    <div className="dark isolate rounded-2xl border border-slate-700 bg-slate-900 p-5 space-y-5 font-mono text-sm text-slate-200">
      {/* Header */}
      <div>
        <h3 className="text-base font-bold text-white">🌸 布隆过滤器 (Bloom Filter) 交互演示</h3>
        <p className="text-slate-400 text-xs mt-0.5">可视化位数组翻转、查询判断、误判率 (False Positive Rate)</p>
      </div>

      {/* Config */}
      <div className="flex gap-4 flex-wrap items-center">
        <div className="flex gap-2 items-center">
          <span className="text-slate-400 text-xs">m (位数组大小):</span>
          {M_OPTIONS.map(mv => (
            <button key={mv} onClick={() => resetFilter(mv, k)}
              className={`px-3 py-1 rounded text-xs font-bold transition-all ${m === mv ? "bg-sky-600 text-white" : "bg-slate-700 text-slate-400 hover:bg-slate-600"}`}>
              {mv}
            </button>
          ))}
        </div>
        <div className="flex gap-2 items-center">
          <span className="text-slate-400 text-xs">k (哈希函数数):</span>
          {[2, 3, 4, 5, 6].map(kv => (
            <button key={kv} onClick={() => resetFilter(m, kv)}
              className={`px-2.5 py-1 rounded text-xs font-bold transition-all ${k === kv ? "bg-violet-600 text-white" : "bg-slate-700 text-slate-400 hover:bg-slate-600"}`}>
              {kv}
            </button>
          ))}
        </div>
      </div>

      {/* Stats bar */}
      <div className="flex gap-3 flex-wrap text-xs">
        <span className="px-2 py-1 rounded bg-slate-800 text-slate-300">n={n} 已插入</span>
        <span className="px-2 py-1 rounded bg-slate-800 text-slate-300">位已置 1: {bitsSet}/{m} ({Math.round(bitsSet / m * 100)}%)</span>
        <span className={`px-2 py-1 rounded font-bold ${fpr > 0.1 ? "bg-red-900 text-red-300" : fpr > 0.01 ? "bg-amber-900 text-amber-300" : "bg-emerald-900 text-emerald-300"}`}>
          理论 FPR = {(fpr * 100).toFixed(3)}%
        </span>
      </div>

      {/* Bit array grid */}
      <div>
        <p className="text-slate-500 text-xs mb-2">位数组 (m={m} 位)：</p>
        <div className="flex flex-wrap gap-1">
          {bits.map((bit, idx) => {
            const isAnim = animPosSet.has(idx);
            const animFn = animPosColor.get(idx) ?? -1;
            const isQuery = queryPosSet.has(idx);
            const allQueryOk = queryResult?.found;
            return (
              <div key={idx} className={`relative flex flex-col items-center justify-center rounded transition-all duration-300 text-xs ${
                isAnim ? `${HASH_BG[animFn] ?? "bg-amber-600"} ring-2 ${HASH_RING[animFn] ?? "ring-amber-400"} scale-125 z-10` :
                isQuery && bit ? (allQueryOk ? (queryResult?.fp ? "bg-red-700 ring-1 ring-red-400" : "bg-emerald-700 ring-1 ring-emerald-400") : "bg-emerald-700") :
                isQuery && !bit ? "bg-slate-600 ring-1 ring-slate-400 opacity-60" :
                bit ? "bg-sky-700" :
                "bg-slate-800 border border-slate-700"
              }`} style={{ width: m <= 16 ? 36 : m <= 32 ? 28 : 18, height: m <= 16 ? 40 : m <= 32 ? 32 : 22 }}>
                <span className={`font-bold leading-none ${bit ? "text-white" : "text-slate-600"}`} style={{ fontSize: bit || isAnim ? "11px" : "10px" }}>
                  {bit ? "1" : "0"}
                </span>
                <span className="text-slate-500 leading-none" style={{ fontSize: "8px" }}>{idx}</span>
              </div>
            );
          })}
        </div>
        <div className="flex gap-3 mt-2 text-xs">
          {Array.from({ length: k }, (_, i) => (
            <span key={i} className={`flex items-center gap-1 ${HASH_COLORS[i]}`}>
              <span className={`w-2.5 h-2.5 rounded ${HASH_BG[i]}`} />h{i + 1}
            </span>
          ))}
          <span className="flex items-center gap-1 text-sky-400"><span className="w-2.5 h-2.5 rounded bg-sky-700" />已置1</span>
        </div>
      </div>

      {/* Query result */}
      {queryResult && (
        <div className={`rounded-lg px-4 py-3 border ${
          queryResult.fp ? "border-red-700 bg-red-950/60" :
          queryResult.found ? "border-emerald-700 bg-emerald-950/40" :
          "border-slate-600 bg-slate-800"
        }`}>
          <p className={`font-bold text-sm ${queryResult.fp ? "text-red-400" : queryResult.found ? "text-emerald-400" : "text-slate-300"}`}>
            {queryResult.fp ? "🔴 误判 (False Positive)！" :
             queryResult.found ? "🟡 所有位均为 1 → 可能存在" :
             "🟢 存在位为 0 → 一定不存在"}
          </p>
          <p className="text-slate-400 text-xs mt-1">
            检查位置：{queryResult.positions.map((h, i) => (
              <span key={i} className={HASH_COLORS[h.fn]}>
                h{h.fn + 1}[{h.pos}]={bits[h.pos] ? "1" : "0"}{i < queryResult!.positions.length - 1 ? ", " : ""}
              </span>
            ))}
          </p>
          {queryResult.fp && <p className="text-red-300 text-xs mt-1">⚠️ 该元素未被插入，但所有对应位恰好都是 1，产生误判！这是 Bloom Filter 的固有代价。</p>}
        </div>
      )}

      {/* Controls */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {/* INSERT */}
        <div className="bg-slate-800/50 rounded-xl p-3 space-y-2">
          <p className="text-slate-300 font-bold text-xs">INSERT</p>
          <div className="flex gap-2">
            <input value={insertKey} onChange={e => setInsertKey(e.target.value)}
              onKeyDown={e => e.key === "Enter" && doInsert()}
              placeholder="输入元素" className="bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-slate-200 text-xs flex-1 focus:outline-none focus:border-sky-500" />
            <button onClick={doInsert} className="px-3 py-1.5 rounded bg-sky-600 hover:bg-sky-500 text-white text-xs font-bold transition-colors">插入</button>
          </div>
          <button onClick={preload} className="w-full py-1.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs transition-colors">
            预加载示例 ({PRESET_INSERTS.join(", ").substring(0, 30)}...)
          </button>
        </div>
        {/* QUERY */}
        <div className="bg-slate-800/50 rounded-xl p-3 space-y-2">
          <p className="text-slate-300 font-bold text-xs">QUERY</p>
          <div className="flex gap-2">
            <input value={queryKey} onChange={e => setQueryKey(e.target.value)}
              onKeyDown={e => e.key === "Enter" && doQuery()}
              placeholder="查询元素" className="bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-slate-200 text-xs flex-1 focus:outline-none focus:border-amber-500" />
            <button onClick={doQuery} className="px-3 py-1.5 rounded bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold transition-colors">查询</button>
          </div>
          <p className="text-slate-500 text-xs">尝试查询 "mango"（未插入）触发误判演示</p>
        </div>
      </div>

      {/* Inserted set */}
      {insertedSet.size > 0 && (
        <div className="bg-slate-800/40 rounded-lg p-3">
          <p className="text-slate-400 text-xs mb-1.5">已插入元素（共 {n} 个）：</p>
          <div className="flex flex-wrap gap-1.5">
            {Array.from(insertedSet).map(word => (
              <span key={word} className="px-2 py-0.5 rounded bg-slate-700 text-sky-300 text-xs">{word}</span>
            ))}
          </div>
        </div>
      )}

      {/* Log */}
      {log.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-3 space-y-0.5">
          {log.map((entry, i) => (
            <p key={i} className={`text-xs ${i === 0 ? "text-slate-200" : "text-slate-500"}`}>{entry}</p>
          ))}
        </div>
      )}

      {/* Theory box */}
      <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3 space-y-1.5 text-xs">
        <p className="text-slate-300 font-bold">📐 理论公式</p>
        <p className="text-slate-400">误判率 (FPR) = <span className="text-violet-300">(1 - e^(-kn/m))^k</span></p>
        <p className="text-slate-400">最优 k = <span className="text-violet-300">(m/n)·ln2 ≈ 0.693·(m/n)</span></p>
        <p className="text-slate-400">当前 n={n}, m={m}, k={k}: FPR ≈ <span className={`font-bold ${fpr > 0.1 ? "text-red-400" : fpr > 0.01 ? "text-amber-400" : "text-emerald-400"}`}>{(fpr * 100).toFixed(4)}%</span></p>
        <p className="text-slate-500 text-xs mt-1">空间效率: Bloom Filter 无需存储原始数据，仅用 m 位 ≈ {Math.round(m / 8)} 字节表示 n 个元素集合</p>
      </div>
    </div>
  );
}
