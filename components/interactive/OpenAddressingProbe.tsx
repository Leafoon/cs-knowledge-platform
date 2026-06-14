"use client";
import React, { useState, useCallback, useMemo } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
type ProbeMode = "linear" | "quadratic" | "double";
type SlotStatus = "empty" | "occupied" | "deleted" | "probe-path" | "target";

interface Slot {
  status: SlotStatus;
  key?: string;
  probeOrder?: number; // which step in probe sequence is this?
}

// ─── Hash functions ────────────────────────────────────────────────────────────
function h1(key: string, m: number): number {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) % m;
  return h;
}
function h2(key: string, m: number): number {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 37 + key.charCodeAt(i)) % m;
  return 1 + (h % (m - 1));
}
function probeSlot(key: string, i: number, m: number, mode: ProbeMode): number {
  const base = h1(key, m);
  switch (mode) {
    case "linear":    return (base + i) % m;
    case "quadratic": return (base + i * i) % m;
    case "double":    return (base + i * h2(key, m)) % m;
  }
}

// ─── Preset scenario ──────────────────────────────────────────────────────────
const M = 13; // prime
const PRESET_INSERTS = ["apple", "banana", "cherry", "avocado", "blueberry", "apricot"];

function buildTable(keys: string[], mode: ProbeMode): Slot[] {
  const table: Slot[] = Array.from({ length: M }, () => ({ status: "empty" }));
  for (const key of keys) {
    for (let i = 0; i < M; i++) {
      const slot = probeSlot(key, i, M, mode);
      if (table[slot].status === "empty" || table[slot].status === "deleted") {
        table[slot] = { status: "occupied", key };
        break;
      }
    }
  }
  return table;
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function OpenAddressingProbe() {
  const [mode, setMode] = useState<ProbeMode>("linear");
  const [probeKey, setProbeKey] = useState("avocado");
  const [showProbe, setShowProbe] = useState(false);
  const [insertKey, setInsertKey] = useState("");
  const [insertedKeys, setInsertedKeys] = useState<string[]>(PRESET_INSERTS);
  const [log, setLog] = useState<string[]>([]);

  const baseTable = useMemo(() => buildTable(insertedKeys, mode), [insertedKeys, mode]);

  // Compute probe sequence for probeKey
  const probeSequence = useMemo(() => {
    if (!showProbe || !probeKey.trim()) return [];
    const seq: number[] = [];
    for (let i = 0; i < M; i++) {
      seq.push(probeSlot(probeKey, i, M, mode));
    }
    return seq;
  }, [showProbe, probeKey, mode]);

  // Build display table combining base + probe path overlay
  const displaySlots: (Slot & { probeStep?: number; isTarget?: boolean })[] = useMemo(() => {
    const slots = baseTable.map(s => ({ ...s }));
    if (!showProbe) return slots;
    let foundAt = -1;
    for (let i = 0; i < probeSequence.length; i++) {
      const s = probeSequence[i];
      if (slots[s].key === probeKey) { foundAt = i; break; }
      if (slots[s].status === "empty") break;
    }
    // Mark probe steps up to find/miss
    const limit = foundAt !== -1 ? foundAt + 1 : probeSequence.findIndex((s, i) => {
      return slots[s].status === "empty" || i === M - 1;
    }) + 1;
    for (let i = 0; i < Math.min(limit, probeSequence.length); i++) {
      const s = probeSequence[i];
      const isLast = i === limit - 1;
      (slots[s] as any).probeStep = i + 1;
      (slots[s] as any).isTarget = isLast && slots[s].key === probeKey;
      if (!isLast) (slots[s] as any).isIntermediate = true;
    }
    return slots;
  }, [baseTable, probeSequence, probeKey, showProbe]);

  const handleInsert = useCallback(() => {
    const k = insertKey.trim();
    if (!k || insertedKeys.includes(k)) return;
    if (insertedKeys.length >= M - 1) {
      setLog(prev => [`⚠️ 表快满了 (α=${(insertedKeys.length/M).toFixed(2)})，应该 rehash 了`, ...prev].slice(0, 5));
      return;
    }
    // Find where it would go with current mode
    const tmpTable = buildTable([...insertedKeys, k], mode);
    let insertedSlot = -1;
    for (let i = 0; i < M; i++) {
      const s = probeSlot(k, i, M, mode);
      if (tmpTable[s].key === k) { insertedSlot = s; break; }
    }
    setInsertedKeys(prev => [...prev, k]);
    setLog(prev => [`✅ 插入 "${k}" → 槽 ${insertedSlot} (h1=${h1(k, M)}, h2=${h2(k, M)})`, ...prev].slice(0, 5));
    setInsertKey("");
  }, [insertKey, insertedKeys, mode]);

  const handleReset = () => {
    setInsertedKeys(PRESET_INSERTS);
    setShowProbe(false);
    setLog([]);
  };

  const totalOccupied = baseTable.filter(s => s.status === "occupied").length;
  const alpha = (totalOccupied / M).toFixed(2);

  // Probe formula descriptions
  const formulas: Record<ProbeMode, { formula: string; desc: string; issue: string; color: string }> = {
    linear:    { formula: "h(k,i) = (h₁(k) + i) mod m", desc: "步长固定为 1，实现最简单", issue: "一次聚集：连续满槽越来越长，性能急剧下降", color: "text-rose-400" },
    quadratic: { formula: "h(k,i) = (h₁(k) + i²) mod m", desc: "步长为 i²，跳跃式探测缓解线性聚集", issue: "二次聚集：h₁值相同的键探测路径完全相同", color: "text-amber-400" },
    double:    { formula: "h(k,i) = (h₁(k) + i·h₂(k)) mod m", desc: "步长依赖 h₂(k)，几乎无聚集", issue: "性能最佳，是三种方式中推荐的方案", color: "text-emerald-400" },
  };

  return (
    <div className="dark isolate rounded-2xl border border-slate-700 bg-slate-900 p-5 space-y-5 font-mono text-sm text-slate-200">
      {/* Header */}
      <div>
        <h3 className="text-base font-bold text-white">🔍 开放寻址法探测序列动画</h3>
        <p className="text-slate-400 text-xs mt-0.5">Open Addressing — 三种探测策略对比 (m={M}, 质数)</p>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2 flex-wrap">
        {(Object.keys(formulas) as ProbeMode[]).map(m_ => (
          <button key={m_} onClick={() => { setMode(m_); setShowProbe(false); }}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${mode === m_ ? "bg-indigo-600 text-white ring-1 ring-indigo-400" : "bg-slate-700 text-slate-400 hover:bg-slate-600"}`}>
            {m_ === "linear" ? "线性探测" : m_ === "quadratic" ? "二次探测" : "双重哈希"}
          </button>
        ))}
      </div>

      {/* Current formula box */}
      <div className={`rounded-lg border px-3 py-2.5 ${mode === "linear" ? "border-rose-800 bg-rose-950/40" : mode === "quadratic" ? "border-amber-800 bg-amber-950/40" : "border-emerald-800 bg-emerald-950/40"}`}>
        <code className={`text-sm font-bold ${formulas[mode].color}`}>{formulas[mode].formula}</code>
        <p className="text-slate-300 text-xs mt-1">{formulas[mode].desc}</p>
        <p className={`text-xs mt-0.5 ${mode === "double" ? "text-emerald-400" : "text-amber-400"}`}>
          {mode === "double" ? "✅" : "⚠️"} {formulas[mode].issue}
        </p>
      </div>

      {/* Stats */}
      <div className="flex gap-3 text-xs flex-wrap">
        <span className="px-2 py-1 rounded bg-slate-800 text-slate-300">n={totalOccupied} / m={M}</span>
        <span className={`px-2 py-1 rounded font-bold ${parseFloat(alpha) > 0.7 ? "bg-red-900 text-red-300" : parseFloat(alpha) > 0.5 ? "bg-amber-900 text-amber-300" : "bg-emerald-900 text-emerald-300"}`}>
          α={alpha} {parseFloat(alpha) > 0.7 ? "⚠️ 建议 rehash" : parseFloat(alpha) > 0.5 ? "警戒" : "√ 健康"}
        </span>
      </div>

      {/* Insert controls */}
      <div className="flex gap-2 items-center flex-wrap">
        <input value={insertKey} onChange={e => setInsertKey(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleInsert()}
          placeholder="插入新键" className="bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-slate-200 text-xs w-36 focus:outline-none focus:border-indigo-500" />
        <button onClick={handleInsert} className="px-3 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-colors">插入</button>
        <button onClick={handleReset} className="px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs transition-colors">重置</button>
      </div>

      {/* Probe simulator */}
      <div className="flex gap-2 items-center flex-wrap">
        <span className="text-slate-400 text-xs">探测键：</span>
        <input value={probeKey} onChange={e => setProbeKey(e.target.value)}
          placeholder="输入要探测的键" className="bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-slate-200 text-xs w-36 focus:outline-none focus:border-amber-500" />
        <button onClick={() => setShowProbe(true)}
          className="px-3 py-1.5 rounded bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold transition-colors">
          显示探测路径
        </button>
        {showProbe && (
          <button onClick={() => setShowProbe(false)} className="px-3 py-1.5 rounded bg-slate-600 hover:bg-slate-500 text-slate-300 text-xs transition-colors">清除</button>
        )}
      </div>

      {/* Probe sequence display */}
      {showProbe && probeSequence.length > 0 && (
        <div className="bg-slate-800/60 rounded-lg p-3">
          <p className="text-slate-400 text-xs mb-2">"{probeKey}" 的探测序列（前 {Math.min(8, M)} 步）：</p>
          <div className="flex gap-2 flex-wrap">
            {probeSequence.slice(0, 8).map((slot, i) => {
              const s = baseTable[slot];
              const isFound = s.key === probeKey;
              const isEmpty = s.status === "empty";
              return (
                <div key={i} className={`flex flex-col items-center gap-0.5 px-2 py-1.5 rounded text-xs ${
                  isFound ? "bg-emerald-700 text-white ring-1 ring-emerald-400" :
                  isEmpty && i > 0 ? "bg-red-900/50 text-red-300 ring-1 ring-red-700" :
                  "bg-slate-700 text-slate-300"
                }`}>
                  <span className="text-slate-400">步 {i + 1}</span>
                  <span className="font-bold">槽 {slot}</span>
                  <span className={`text-xs ${isFound ? "text-emerald-300" : isEmpty ? "text-red-400" : "text-amber-300"}`}>
                    {isFound ? "✓ 命中" : isEmpty ? "↑ 停止" : `${s.key || "占用"}`}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Hash table slots grid */}
      <div>
        <p className="text-slate-500 text-xs mb-2">哈希表（m={M} 个槽）：</p>
        <div className="grid grid-cols-13 gap-1" style={{ gridTemplateColumns: `repeat(${M}, minmax(0, 1fr))` }}>
          {displaySlots.map((slot, idx) => {
            const step = (slot as any).probeStep as number | undefined;
            const isTarget = (slot as any).isTarget as boolean;
            const isIntermediate = (slot as any).isIntermediate as boolean;
            return (
              <div key={idx} className={`aspect-square flex flex-col items-center justify-center rounded text-center transition-all duration-300 text-xs min-w-0 p-0.5 ${
                isTarget ? "bg-emerald-600 ring-2 ring-emerald-300 scale-110 z-10" :
                step !== undefined ? "bg-amber-700/70 ring-1 ring-amber-500" :
                slot.status === "occupied" ? "bg-slate-700 " :
                "bg-slate-800 border border-slate-700"
              }`}>
                <span className={`font-bold text-xs leading-none ${slot.status === "occupied" ? "text-slate-200" : "text-slate-600"}`}>
                  {idx}
                </span>
                <span className="text-xs leading-none truncate w-full text-center" style={{ fontSize: "9px" }}>
                  {slot.key ? slot.key.slice(0, 4) : "∅"}
                </span>
                {step !== undefined && (
                  <span className={`text-xs font-bold leading-none ${isTarget ? "text-emerald-200" : "text-amber-300"}`} style={{ fontSize: "9px" }}>
                    {isTarget ? "✓" : `→${step}`}
                  </span>
                )}
              </div>
            );
          })}
        </div>
        <div className="flex gap-3 mt-2 text-xs flex-wrap">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-slate-700 inline-block" />已占用</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-700 inline-block" />探测路径</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-emerald-600 inline-block" />命中/终止</span>
        </div>
      </div>

      {/* Already-inserted keys */}
      <div className="bg-slate-800/60 rounded-lg p-3">
        <p className="text-slate-400 text-xs mb-1.5">已插入的键（按 {mode === "linear" ? "线性" : mode === "quadratic" ? "二次" : "双重"} 探测顺序）：</p>
        <div className="flex flex-wrap gap-1.5">
          {insertedKeys.map((k, i) => (
            <span key={i} className="px-2 py-0.5 rounded bg-slate-700 text-slate-300 text-xs">
              "{k}" → h1={h1(k, M)}{mode === "double" ? `, h2=${h2(k, M)}` : ""}
            </span>
          ))}
        </div>
      </div>

      {/* Log */}
      {log.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-3 space-y-0.5">
          {log.map((entry, i) => (
            <p key={i} className={`text-xs ${i === 0 ? "text-slate-200" : "text-slate-500"}`}>{entry}</p>
          ))}
        </div>
      )}

      {/* Clustering comparison insight */}
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        {(["linear","quadratic","double"] as ProbeMode[]).map(m_ => {
          const table = buildTable(insertedKeys, m_);
          // Compute max run length (cluster size)
          let maxRun = 0, cur = 0;
          for (let i = 0; i < M * 2; i++) {
            if (table[i % M].status === "occupied") { cur++; maxRun = Math.max(maxRun, cur); }
            else cur = 0;
          }
          return (
            <div key={m_} className={`rounded-lg p-2 ${mode === m_ ? "ring-1 ring-indigo-500 bg-slate-800" : "bg-slate-800/40"}`}>
              <p className="text-slate-400">{m_ === "linear" ? "线性探测" : m_ === "quadratic" ? "二次探测" : "双重哈希"}</p>
              <p className={`font-bold mt-1 ${maxRun >= 4 ? "text-red-400" : maxRun >= 3 ? "text-amber-400" : "text-emerald-400"}`}>
                最大簇长={maxRun}
              </p>
            </div>
          );
        })}
      </div>
      <p className="text-slate-500 text-xs text-center">最大连续占用槽数越小，聚集程度越低，性能越好</p>
    </div>
  );
}
