"use client";

import React, { useState } from "react";

// ── LRU 内部状态 ────────────────────────────────────────────────────────────────
interface CacheNode {
  key: number;
  val: number;
  id: number;  // 稳定 key for animation
}

let gid = 1;

function useLRU(capacity: number) {
  // dll: 最近使用在前（index 0），最旧在后
  const [dll, setDll] = useState<CacheNode[]>([]);
  const [log, setLog] = useState<{ op: string; hit: boolean; evicted?: number }[]>([]);

  const addLog = (op: string, hit: boolean, evicted?: number) =>
    setLog((l) => [{ op, hit, evicted }, ...l].slice(0, 8));

  const get = (key: number): number => {
    const idx = dll.findIndex((n) => n.key === key);
    if (idx === -1) {
      addLog(`get(${key}) → -1 ❌ miss`, false);
      return -1;
    }
    const node = dll[idx];
    const next = [node, ...dll.filter((_, i) => i !== idx)];
    setDll(next);
    addLog(`get(${key}) → ${node.val} ✅ hit，移到最前`, true);
    return node.val;
  };

  const put = (key: number, val: number) => {
    const idx = dll.findIndex((n) => n.key === key);
    if (idx !== -1) {
      const node = dll[idx];
      node.val = val;
      const next = [node, ...dll.filter((_, i) => i !== idx)];
      setDll(next);
      addLog(`put(${key},${val}) 更新，移到最前`, true);
    } else {
      const newNode: CacheNode = { key, val, id: gid++ };
      let next = [newNode, ...dll];
      let evicted: number | undefined;
      if (next.length > capacity) {
        evicted = next[next.length - 1].key;
        next = next.slice(0, capacity);
        addLog(`put(${key},${val}) 新增，淘汰最旧 key=${evicted}`, false, evicted);
      } else {
        addLog(`put(${key},${val}) 新增`, false);
      }
      setDll(next);
    }
  };

  const clear = () => { setDll([]); setLog([]); };

  return { dll, log, get, put, clear };
}

// ── 预设操作序列 ────────────────────────────────────────────────────────────────
type PresetOp = { type: "get" | "put"; key: number; val?: number };
const PRESET: PresetOp[] = [
  { type: "put", key: 1, val: 10 },
  { type: "put", key: 2, val: 20 },
  { type: "put", key: 3, val: 30 },
  { type: "get", key: 1 },          // hit → 1 移最前
  { type: "put", key: 4, val: 40 }, // 淘汰 key=2（最旧）
  { type: "get", key: 2 },          // miss
  { type: "put", key: 5, val: 50 }, // 淘汰 key=3
  { type: "get", key: 1 },          // hit
];

export default function LRUCacheSim() {
  const [capacity, setCapacity] = useState(3);
  const { dll, log, get, put, clear } = useLRU(capacity);
  const [inputKey, setInputKey] = useState(1);
  const [inputVal, setInputVal] = useState(99);
  const [presetStep, setPresetStep] = useState(0);

  const runPreset = () => {
    if (presetStep < PRESET.length) {
      const op = PRESET[presetStep];
      if (op.type === "get") get(op.key);
      else put(op.key, op.val!);
      setPresetStep((s) => s + 1);
    }
  };

  const resetAll = () => { clear(); setPresetStep(0); };

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-amber-500/15 dark:bg-amber-500/20 flex items-center justify-center text-xl">💾</div>
        <div>
          <h3 className="font-bold text-text-primary text-base">LRU Cache 联动演示</h3>
          <p className="text-xs text-text-secondary">双向链表（最近→最旧）+ 哈希表，get/put 均 O(1)</p>
        </div>
      </div>

      {/* 容量控制 */}
      <div className="flex flex-wrap gap-3 items-center border-t border-border-subtle pt-3">
        <label className="flex items-center gap-2 text-xs text-text-secondary">
          容量（capacity）：
          <input type="number" min={1} max={6} value={capacity}
            onChange={(e) => { setCapacity(Number(e.target.value)); resetAll(); }}
            className="w-14 bg-bg-tertiary border border-border-subtle rounded-lg px-2 py-1 text-xs font-mono text-text-primary outline-none" />
        </label>
        <button onClick={resetAll}
          className="px-2 py-1 rounded-lg bg-bg-tertiary hover:bg-border-subtle text-text-secondary text-xs transition-colors">重置</button>
      </div>

      {/* 预设操作 */}
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-xs text-text-tertiary">预设演示：</span>
        {PRESET.map((op, i) => (
          <div key={i} className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-all ${
            i < presetStep ? "bg-emerald-500/10 border-emerald-400/30 text-emerald-700 dark:text-emerald-300" :
            i === presetStep ? "bg-amber-500/15 border-amber-400/50 text-amber-700 dark:text-amber-300 ring-1 ring-amber-400/50" :
            "bg-bg-tertiary border-border-subtle text-text-tertiary"}`}>
            {op.type}({op.key}{op.val !== undefined ? `,${op.val}` : ""})
          </div>
        ))}
        <button onClick={runPreset} disabled={presetStep >= PRESET.length}
          className="ml-auto px-3 py-1 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 disabled:opacity-40 text-amber-700 dark:text-amber-300 text-xs font-medium transition-colors">
          ▶ 下一步
        </button>
      </div>

      {/* 主视图：双向链表 + 哈希表 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* 双向链表 */}
        <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3 space-y-2">
          <div className="text-xs font-semibold text-text-primary flex items-center gap-2">
            <span>双向链表</span>
            <span className="text-text-tertiary font-normal text-[10px]">←最近 ｜ 最旧→</span>
          </div>
          {dll.length === 0 ? (
            <div className="text-xs text-text-tertiary text-center py-3">（缓存为空）</div>
          ) : (
            <div className="flex items-center gap-1 flex-wrap">
              <div className="flex flex-col items-center">
                <div className="bg-violet-500/10 border border-violet-400/30 text-violet-700 dark:text-violet-300 rounded px-2 py-1 text-[10px] font-mono">head</div>
              </div>
              <span className="text-text-tertiary text-xs">↔</span>
              {dll.map((node, i) => (
                <React.Fragment key={node.id}>
                  <div className={`flex flex-col items-center gap-0.5 transition-all duration-300`}>
                    <div className={`rounded-lg border px-3 py-2 text-xs font-bold font-mono transition-all duration-300
                      ${i === 0 ? "bg-emerald-500/20 border-emerald-400/60 text-emerald-700 dark:text-emerald-300" :
                        i === dll.length - 1 ? "bg-rose-500/15 border-rose-400/50 text-rose-700 dark:text-rose-300" :
                        "bg-bg-secondary border-border-subtle text-text-primary"}`}>
                      {node.key}:{node.val}
                    </div>
                    <span className="text-[8px] text-text-tertiary">{i === 0 ? "最近" : i === dll.length - 1 ? "最旧" : ""}</span>
                  </div>
                  {i < dll.length - 1 && <span className="text-text-tertiary text-xs">↔</span>}
                </React.Fragment>
              ))}
              <span className="text-text-tertiary text-xs">↔</span>
              <div className="flex flex-col items-center">
                <div className="bg-violet-500/10 border border-violet-400/30 text-violet-700 dark:text-violet-300 rounded px-2 py-1 text-[10px] font-mono">tail</div>
              </div>
            </div>
          )}
          {/* 容量进度 */}
          <div>
            <div className="flex justify-between text-[9px] text-text-tertiary mb-0.5">
              <span>容量使用</span>
              <span>{dll.length}/{capacity}</span>
            </div>
            <div className="h-1.5 bg-bg-secondary rounded-full overflow-hidden">
              <div className={`h-full rounded-full transition-all duration-300 ${dll.length >= capacity ? "bg-rose-500" : "bg-emerald-500"}`}
                style={{ width: `${(dll.length / capacity) * 100}%` }} />
            </div>
          </div>
        </div>

        {/* 哈希表 */}
        <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3 space-y-2">
          <div className="text-xs font-semibold text-text-primary">哈希表（key → 节点指针）</div>
          {dll.length === 0 ? (
            <div className="text-xs text-text-tertiary text-center py-3">（哈希表为空）</div>
          ) : (
            <div className="space-y-1">
              {dll.map((node) => (
                <div key={node.id} className="flex items-center gap-2">
                  <div className="bg-blue-500/10 border border-blue-400/30 text-blue-700 dark:text-blue-300 rounded px-2 py-0.5 text-[10px] font-mono w-16 text-center">
                    key={node.key}
                  </div>
                  <span className="text-text-tertiary text-xs">→</span>
                  <div className="bg-bg-secondary border border-border-subtle text-text-primary rounded px-2 py-0.5 text-[10px] font-mono">
                    DListNode({node.key},{node.val})
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 手动操作 */}
      <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3 space-y-2">
        <div className="text-xs font-semibold text-text-primary">手动操作</div>
        <div className="flex flex-wrap gap-2 items-center">
          <label className="text-xs text-text-secondary">key=</label>
          <input type="number" value={inputKey} onChange={(e) => setInputKey(Number(e.target.value))}
            className="w-14 bg-bg-secondary border border-border-subtle rounded-lg px-2 py-1 text-xs font-mono text-text-primary outline-none" />
          <button onClick={() => get(inputKey)}
            className="px-3 py-1 rounded-lg bg-blue-500/15 hover:bg-blue-500/25 text-blue-700 dark:text-blue-300 text-xs font-medium transition-colors">
            GET
          </button>
          <label className="text-xs text-text-secondary ml-2">val=</label>
          <input type="number" value={inputVal} onChange={(e) => setInputVal(Number(e.target.value))}
            className="w-14 bg-bg-secondary border border-border-subtle rounded-lg px-2 py-1 text-xs font-mono text-text-primary outline-none" />
          <button onClick={() => put(inputKey, inputVal)}
            className="px-3 py-1 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 text-amber-700 dark:text-amber-300 text-xs font-medium transition-colors">
            PUT
          </button>
        </div>
      </div>

      {/* 操作日志 */}
      {log.length > 0 && (
        <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3 space-y-1">
          <div className="text-[10px] text-text-tertiary">操作日志（最新在上）</div>
          {log.map((entry, i) => (
            <div key={i} className={`text-xs font-mono px-2 py-0.5 rounded ${
              entry.hit ? "text-emerald-700 dark:text-emerald-300" : "text-rose-600 dark:text-rose-400"}`}>
              {entry.op}
              {entry.evicted !== undefined && <span className="text-text-tertiary ml-2">（淘汰 key={entry.evicted}）</span>}
            </div>
          ))}
        </div>
      )}

      {/* 说明 */}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary px-3 py-2.5 text-xs text-text-secondary space-y-1">
        <div className="font-semibold text-text-primary">设计要点</div>
        <div>• <span className="text-text-primary">get(key)</span>：哈希表 O(1) 定位节点，双向链表将其摘下并插到最前（最近使用）</div>
        <div>• <span className="text-text-primary">put(key,val)</span>：不存在则新建放最前；容量满时删除链表尾端（最旧节点）并从哈希表移除其 key</div>
        <div>• 为什么必须用<span className="text-text-primary">双向链表</span>：删除尾节点时需要 O(1) 定位其前驱，单向链表无法做到</div>
      </div>
    </div>
  );
}
