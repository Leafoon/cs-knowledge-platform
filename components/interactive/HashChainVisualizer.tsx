"use client";
import React, { useState, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
interface ChainNode { key: string; val: number; highlight?: boolean }
type SlotState = ChainNode[];

// ─── Hash function (simple polynomial mod m) ─────────────────────────────────
function hashKey(key: string, m: number): number {
  let h = 0;
  for (let i = 0; i < key.length; i++) {
    h = (h * 31 + key.charCodeAt(i)) % m;
  }
  return h;
}

// Prebuilt fixed example with m=8
const M_DEFAULT = 8;
const PRESET_KEYS = ["apple","banana","cat","dog","egg","fig","grape","hat","ink","jam"];
const PRESET_VALS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

// ─── Colors ──────────────────────────────────────────────────────────────────
const CHAIN_COLORS = [
  "bg-blue-500", "bg-violet-500", "bg-emerald-500", "bg-amber-500",
  "bg-rose-500", "bg-cyan-500", "bg-fuchsia-500", "bg-lime-500",
];
const CHAIN_TEXT_COLORS = [
  "text-blue-400", "text-violet-400", "text-emerald-400", "text-amber-400",
  "text-rose-400", "text-cyan-400", "text-fuchsia-400", "text-lime-400",
];

// ─── Component ────────────────────────────────────────────────────────────────
export default function HashChainVisualizer() {
  const [m, setM] = useState(M_DEFAULT);
  const [table, setTable] = useState<SlotState[]>(() => buildTable(M_DEFAULT, PRESET_KEYS, PRESET_VALS));
  const [inputKey, setInputKey] = useState("");
  const [inputVal, setInputVal] = useState("");
  const [searchKey, setSearchKey] = useState("");
  const [searchResult, setSearchResult] = useState<string | null>(null);
  const [activeSlot, setActiveSlot] = useState<number | null>(null);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [mode, setMode] = useState<"insert" | "search">("insert");
  const [log, setLog] = useState<string[]>([]);

  function buildTable(size: number, keys: string[], vals: number[]): SlotState[] {
    const t: SlotState[] = Array.from({ length: size }, () => []);
    for (let i = 0; i < keys.length; i++) {
      const h = hashKey(keys[i], size);
      t[h].push({ key: keys[i], val: vals[i] });
    }
    return t;
  }

  const handleCapacityChange = useCallback((newM: number) => {
    setM(newM);
    setTable(buildTable(newM, PRESET_KEYS, PRESET_VALS));
    setActiveSlot(null); setActiveNode(null); setSearchResult(null); setLog([]);
  }, []);

  const handleInsert = useCallback(() => {
    if (!inputKey.trim()) return;
    const k = inputKey.trim();
    const v = parseInt(inputVal) || 0;
    const h = hashKey(k, m);
    const newTable = table.map(chain => [...chain]);
    const exists = newTable[h].findIndex(n => n.key === k);
    let msg = "";
    if (exists !== -1) {
      newTable[h][exists] = { key: k, val: v };
      msg = `🔄 更新: "${k}" → 槽 ${h}（原值已更新为 ${v}）`;
    } else {
      newTable[h] = [{ key: k, val: v, highlight: true }, ...newTable[h]];
      msg = `✅ 插入: "${k}" → h("${k}")=${h}，链头插入，链长=${newTable[h].length}`;
    }
    setTable(newTable);
    setActiveSlot(h);
    setActiveNode(k);
    setLog(prev => [msg, ...prev].slice(0, 6));
    setInputKey(""); setInputVal("");
    setTimeout(() => {
      setTable(t => t.map((chain, i) => i === h ? chain.map(n => ({ ...n, highlight: false })) : chain));
      setActiveNode(null);
    }, 1500);
  }, [inputKey, inputVal, m, table]);

  const handleSearch = useCallback(() => {
    if (!searchKey.trim()) return;
    const k = searchKey.trim();
    const h = hashKey(k, m);
    const chain = table[h];
    const idx = chain.findIndex(n => n.key === k);
    setActiveSlot(h);
    if (idx !== -1) {
      setActiveNode(k);
      const comparisons = idx + 1;
      const msg = `🔍 找到 "${k}"：槽 ${h}, 比较 ${comparisons} 次（位置 ${idx + 1}/${chain.length}）`;
      setSearchResult(msg);
      setLog(prev => [msg, ...prev].slice(0, 6));
    } else {
      setActiveNode(null);
      const msg = `❌ 未找到 "${k}"：槽 ${h}, 比较 ${chain.length} 次（遍历完整链）`;
      setSearchResult(msg);
      setLog(prev => [msg, ...prev].slice(0, 6));
    }
  }, [searchKey, m, table]);

  const handleReset = () => {
    setTable(buildTable(m, PRESET_KEYS, PRESET_VALS));
    setActiveSlot(null); setActiveNode(null); setSearchResult(null); setLog([]);
  };

  // Stats
  const totalNodes = table.reduce((s, c) => s + c.length, 0);
  const alpha = (totalNodes / m).toFixed(2);
  const maxChain = Math.max(...table.map(c => c.length), 0);

  return (
    <div className="dark isolate rounded-2xl border border-slate-700 bg-slate-900 p-5 space-y-5 font-mono text-sm text-slate-200">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-base font-bold text-white">🔗 链地址法哈希表可视化</h3>
          <p className="text-slate-400 text-xs mt-0.5">Hash Table with Separate Chaining</p>
        </div>
        <div className="flex gap-2 text-xs">
          <span className="px-2 py-1 rounded bg-slate-800 text-slate-300">n={totalNodes}</span>
          <span className="px-2 py-1 rounded bg-slate-800 text-slate-300">m={m}</span>
          <span className={`px-2 py-1 rounded font-bold ${parseFloat(alpha) > 1 ? "bg-red-900 text-red-300" : parseFloat(alpha) > 0.75 ? "bg-amber-900 text-amber-300" : "bg-emerald-900 text-emerald-300"}`}>
            α={alpha}
          </span>
          <span className="px-2 py-1 rounded bg-slate-800 text-slate-300">最长链={maxChain}</span>
        </div>
      </div>

      {/* Controls Row */}
      <div className="flex flex-wrap gap-3 items-center">
        {/* Capacity selector */}
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-xs">槽数 m=</span>
          {[5, 7, 8, 11, 13].map(v => (
            <button key={v} onClick={() => handleCapacityChange(v)}
              className={`px-2 py-1 rounded text-xs font-bold transition-colors ${m === v ? "bg-indigo-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"}`}>
              {v}{[7,11,13].includes(v) ? "✓" : ""}
            </button>
          ))}
          <span className="text-slate-500 text-xs">（✓=质数）</span>
        </div>
      </div>

      {/* Mode Tabs */}
      <div className="flex gap-2">
        {(["insert","search"] as const).map(tab => (
          <button key={tab} onClick={() => { setMode(tab); setSearchResult(null); }}
            className={`px-3 py-1.5 rounded text-xs font-semibold transition-colors ${mode === tab ? "bg-indigo-600 text-white" : "bg-slate-700 text-slate-400 hover:bg-slate-600"}`}>
            {tab === "insert" ? "插入 (INSERT)" : "查找 (SEARCH)"}
          </button>
        ))}
        <button onClick={handleReset} className="ml-auto px-3 py-1.5 rounded text-xs bg-slate-700 text-slate-400 hover:bg-slate-600 transition-colors">
          重置预设数据
        </button>
      </div>

      {/* Input Area */}
      {mode === "insert" ? (
        <div className="flex gap-2 items-center flex-wrap">
          <input value={inputKey} onChange={e => setInputKey(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleInsert()}
            placeholder="键（如 hello）" className="bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-slate-200 text-xs w-32 focus:outline-none focus:border-indigo-500" />
          <input value={inputVal} onChange={e => setInputVal(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleInsert()}
            placeholder="值（整数）" className="bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-slate-200 text-xs w-24 focus:outline-none focus:border-indigo-500" />
          <button onClick={handleInsert} className="px-4 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-colors">
            插入
          </button>
          <span className="text-slate-500 text-xs">h(key) = Σ(charCode×31^i) mod m</span>
        </div>
      ) : (
        <div className="flex gap-2 items-center flex-wrap">
          <input value={searchKey} onChange={e => setSearchKey(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSearch()}
            placeholder="查找的键" className="bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-slate-200 text-xs w-36 focus:outline-none focus:border-indigo-500" />
          <button onClick={handleSearch} className="px-4 py-1.5 rounded bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold transition-colors">
            查找
          </button>
        </div>
      )}

      {/* Search Result */}
      {searchResult && (
        <div className={`px-3 py-2 rounded text-xs ${searchResult.includes("找到") ? "bg-emerald-900/50 border border-emerald-700 text-emerald-300" : "bg-red-900/40 border border-red-700 text-red-300"}`}>
          {searchResult}
        </div>
      )}

      {/* Hash Table Visualization */}
      <div className="space-y-1.5">
        {table.map((chain, slot) => {
          const isActive = slot === activeSlot;
          const chainLen = chain.length;
          const loadColor = chainLen === 0 ? "text-slate-600" : chainLen === 1 ? "text-emerald-400" : chainLen <= 2 ? "text-amber-400" : "text-red-400";
          return (
            <div key={slot} className={`flex items-center gap-2 rounded-lg px-2 py-1.5 transition-all duration-300 ${isActive ? "bg-slate-700 ring-1 ring-indigo-500" : "bg-slate-800/60"}`}>
              {/* Slot index */}
              <div className={`w-8 h-8 flex items-center justify-center rounded font-bold text-xs shrink-0 ${isActive ? "bg-indigo-600 text-white" : "bg-slate-700 text-slate-400"}`}>
                {slot}
              </div>
              {/* Slot box */}
              <div className={`w-10 h-8 border-2 flex items-center justify-center text-xs shrink-0 ${chainLen === 0 ? "border-slate-600 text-slate-600" : `border-${CHAIN_COLORS[slot].split('-')[1]}-500`} ${isActive ? "border-indigo-400" : ""}`}>
                {chainLen === 0 ? "∅" : "→"}
              </div>
              {/* Chain nodes */}
              <div className="flex items-center gap-1 overflow-x-auto flex-1">
                {chain.length === 0 ? (
                  <span className="text-slate-600 text-xs">（空槽）</span>
                ) : (
                  chain.map((node, ni) => (
                    <React.Fragment key={node.key}>
                      <div className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold shrink-0 transition-all duration-300 ${
                        node.highlight ? "ring-2 ring-yellow-400 scale-110" : ""
                      } ${activeNode === node.key ? `${CHAIN_COLORS[slot]} text-white scale-105` : "bg-slate-700 text-slate-200"}`}>
                        <span className={activeNode === node.key ? "text-white" : CHAIN_TEXT_COLORS[slot]}>{node.key}</span>
                        <span className="text-slate-400">:{node.val}</span>
                      </div>
                      {ni < chain.length - 1 && (
                        <span className="text-slate-500 text-xs shrink-0">→</span>
                      )}
                      {ni === chain.length - 1 && (
                        <span className="text-slate-600 text-xs shrink-0">∅</span>
                      )}
                    </React.Fragment>
                  ))
                )}
              </div>
              {/* Chain length indicator */}
              <div className={`shrink-0 text-xs font-bold w-12 text-right ${loadColor}`}>
                {chainLen === 0 ? "" : `len=${chainLen}`}
              </div>
            </div>
          );
        })}
      </div>

      {/* Chain length heatmap bar */}
      <div>
        <p className="text-slate-500 text-xs mb-1.5">链长热力图（颜色代表链表长度，越红越长）：</p>
        <div className="flex gap-1 items-end h-16">
          {table.map((chain, slot) => {
            const len = chain.length;
            const maxH = Math.max(maxChain, 1);
            const heightPct = (len / maxH) * 100;
            const barColor = len === 0 ? "bg-slate-700" : len === 1 ? "bg-emerald-500" : len === 2 ? "bg-amber-500" : "bg-red-500";
            return (
              <div key={slot} className="flex flex-col items-center gap-0.5 flex-1">
                <span className="text-slate-400 text-xs">{len}</span>
                <div className="w-full relative" style={{ height: "40px" }}>
                  <div className={`absolute bottom-0 w-full rounded-t transition-all duration-500 ${barColor}`}
                    style={{ height: `${Math.max(heightPct, len === 0 ? 5 : 10)}%` }} />
                </div>
                <span className="text-slate-500 text-xs">{slot}</span>
              </div>
            );
          })}
        </div>
        <div className="flex gap-4 mt-2 text-xs">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-emerald-500 inline-block" />长度=1 (理想)</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-500 inline-block" />长度=2</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500 inline-block" />长度≥3 (警告)</span>
        </div>
      </div>

      {/* Log */}
      {log.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-3 space-y-1">
          <p className="text-slate-500 text-xs mb-1">操作日志：</p>
          {log.map((entry, i) => (
            <p key={i} className={`text-xs ${i === 0 ? "text-slate-200" : "text-slate-500"}`}>{entry}</p>
          ))}
        </div>
      )}

      {/* Info box */}
      <div className="bg-slate-800/60 rounded-lg p-3 text-xs text-slate-400 space-y-1">
        <p>💡 <strong className="text-slate-300">负载因子 α = n/m</strong>：α &lt; 0.75 时性能良好，α &gt; 1 时链表变长，期望查找时间 O(1+α)</p>
        <p>💡 <strong className="text-slate-300">最坏情况</strong>：所有键哈希到同一槽 → O(n) 查找（使用全域哈希可避免）</p>
        <p>💡 <strong className="text-slate-300">m 选质数更均匀</strong>：7, 11, 13 等质数使键更均匀分布（带 ✓ 标记）</p>
      </div>
    </div>
  );
}
