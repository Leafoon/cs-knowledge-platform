"use client";

import React, { useState } from "react";

// ── 模拟内存访问数据 ─────────────────────────────────────────────────────────────
// 数组顺序遍历：地址连续，cache line 64 字节 = 16 个 int,
// 链表遍历：节点地址随机，每次访问几乎必然 cache miss

interface AccessPattern {
  idx: number;
  address: number;   // 模拟内存地址（基于构造散布）
  hit: boolean;      // 是否命中缓存行
  jump: number;      // 与上一次访问的地址距离
}

const CACHE_LINE_SIZE = 16; // 以 int（4字节）为单位，64字节 = 16个int

// 生成数组访问模式：地址连续
function genArrayPattern(n: number): AccessPattern[] {
  const result: AccessPattern[] = [];
  let prevAddr = 1000;
  for (let i = 0; i < n; i++) {
    const addr = 1000 + i;
    const jump = i === 0 ? 0 : addr - prevAddr;
    result.push({ idx: i, address: addr, hit: i % CACHE_LINE_SIZE !== 0, jump });
    prevAddr = addr;
  }
  return result;
}

// 生成链表访问模式：每个节点地址随机散布（模拟堆分配）
function genLinkedListPattern(n: number, seed = 42): AccessPattern[] {
  const result: AccessPattern[] = [];
  // 生成随机但固定的地址序列（简单 LCG 伪随机）
  const addresses: number[] = [];
  let x = seed;
  for (let i = 0; i < n; i++) {
    x = (x * 1103515245 + 12345) & 0x7fff;
    // 地址在 0~8000 范围内随机分布，步长自然大于 cache line
    addresses.push(100 + (x % 800) * 10);
  }
  // 确保不重复
  const seen = new Set<number>();
  let di = 0;
  for (let i = 0; i < n; i++) {
    while (seen.has(addresses[i])) addresses[i] += 1;
    seen.add(addresses[i]);
  }
  let prevAddr = addresses[0];
  for (let i = 0; i < n; i++) {
    const addr = addresses[i];
    const jump = i === 0 ? 0 : Math.abs(addr - prevAddr);
    // 命中：只有当 Math.floor(addr/16) === Math.floor(prevAddr/16) 才在同一 cache line
    const hit = i !== 0 && Math.floor(addr / CACHE_LINE_SIZE) === Math.floor(prevAddr / CACHE_LINE_SIZE);
    result.push({ idx: i, address: addr, hit, jump });
    prevAddr = addr;
  }
  return result;
}

const N_DISPLAY = 16; // 展示 16 个节点

export default function LinkedListVsArrayBenchmark() {
  const [n, setN] = useState(N_DISPLAY);
  const [showAddresses, setShowAddresses] = useState(true);
  const [tab, setTab] = useState<"visual" | "stats" | "explain">("visual");

  const arrPat = React.useMemo(() => genArrayPattern(n), [n]);
  const llPat  = React.useMemo(() => genLinkedListPattern(n), [n]);

  const arrHits  = arrPat.filter((p) => p.hit).length;
  const llHits   = llPat.filter((p) => p.hit).length;
  const arrMissRate = ((n - arrHits) / n * 100).toFixed(0);
  const llMissRate  = ((n - llHits) / n * 100).toFixed(0);

  // 相对耗时估算（每次 cache miss ≈ 额外 100 cycles）
  const arrMisses = n - arrHits;
  const llMisses  = n - llHits;
  const arrTime = n + arrMisses * 100;   // (n hits × 1) + (misses × 100)
  const llTime  = n + llMisses  * 100;
  const slowdown = (llTime / arrTime).toFixed(1);

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-teal-500/15 dark:bg-teal-500/20 flex items-center justify-center text-xl">🔬</div>
        <div>
          <h3 className="font-bold text-text-primary text-base">链表 vs 数组：缓存命中率对比</h3>
          <p className="text-xs text-text-secondary">模拟内存访问模式，展示缓存局部性对实际性能的影响</p>
        </div>
      </div>

      {/* 控制 */}
      <div className="flex flex-wrap gap-4 items-center border-t border-border-subtle pt-3">
        <label className="flex items-center gap-2 text-xs text-text-secondary">
          节点/元素数：
          <input type="range" min={8} max={32} step={4} value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-24 accent-teal-500" />
          <span className="font-mono text-text-primary w-4">{n}</span>
        </label>
        <label className="flex items-center gap-2 text-xs text-text-secondary cursor-pointer">
          <input type="checkbox" checked={showAddresses} onChange={(e) => setShowAddresses(e.target.checked)}
            className="w-4 h-4 rounded accent-teal-500" />
          显示内存地址
        </label>
        <div className="ml-auto flex gap-1">
          {(["visual", "stats", "explain"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-3 py-1 rounded-lg border text-xs font-medium transition-all ${tab === t
                ? "bg-teal-500/20 border-teal-400/60 text-teal-700 dark:text-teal-300"
                : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}>
              {t === "visual" ? "可视化" : t === "stats" ? "统计" : "原理"}
            </button>
          ))}
        </div>
      </div>

      {tab === "visual" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* 数组 */}
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3 space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-text-primary">数组（连续内存）</div>
              <span className="text-[10px] text-emerald-600 dark:text-emerald-300">✅ Cache 友好</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {arrPat.map((p) => (
                <div key={p.idx} title={showAddresses ? `地址: ${p.address * 4}` : ""}
                  className={`relative w-7 h-7 rounded text-[9px] font-mono flex items-center justify-center border transition-colors
                    ${p.hit ? "bg-emerald-500/20 border-emerald-400/40 text-emerald-700 dark:text-emerald-300"
                             : "bg-rose-500/20 border-rose-400/40 text-rose-700 dark:text-rose-300"}`}>
                  {p.idx}
                  {showAddresses && <span className="absolute -bottom-3.5 text-[7px] text-text-tertiary">
                    {(p.address * 4) % 10000}
                  </span>}
                </div>
              ))}
            </div>
            <div className="mt-3 text-[10px] text-text-secondary space-y-0.5">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-emerald-500/30 border border-emerald-400/40" />
                <span>Cache Hit（同一 cache line，0 cost）</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-rose-500/20 border border-rose-400/40" />
                <span>Cache Miss（加载新 cache line，~100 cycles）</span>
              </div>
            </div>
          </div>

          {/* 链表 */}
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3 space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-text-primary">链表（散布内存）</div>
              <span className="text-[10px] text-rose-600 dark:text-rose-400">⚠️ Cache 不友好</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {llPat.map((p) => (
                <div key={p.idx} title={showAddresses ? `地址: ${p.address * 4}, 跳跃: ${p.jump * 4}` : ""}
                  className={`relative w-7 h-7 rounded text-[9px] font-mono flex items-center justify-center border transition-colors
                    ${p.hit ? "bg-emerald-500/20 border-emerald-400/40 text-emerald-700 dark:text-emerald-300"
                             : "bg-rose-500/20 border-rose-400/40 text-rose-700 dark:text-rose-300"}`}>
                  {p.idx}
                  {showAddresses && <span className="absolute -bottom-3.5 text-[7px] text-text-tertiary">
                    {(p.address * 4) % 10000}
                  </span>}
                </div>
              ))}
            </div>
            <div className="mt-3 text-[10px] text-text-secondary">
              <div>平均跳跃距离：{Math.round(llPat.slice(1).reduce((a, p) => a + p.jump, 0) / (n - 1) * 4)} 字节</div>
              <div className="text-text-tertiary">（远超 cache line 64字节 = 必然 miss）</div>
            </div>
          </div>
        </div>
      )}

      {tab === "stats" && (
        <div className="space-y-3">
          {/* Cache Miss 率 */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "数组", missRate: arrMissRate, hits: arrHits, color: "emerald" },
              { label: "链表", missRate: llMissRate, hits: llHits, color: "rose" },
            ].map(({ label, missRate, hits, color }) => (
              <div key={label} className="rounded-xl border border-border-subtle bg-bg-tertiary p-3 text-center space-y-2">
                <div className="text-xs font-semibold text-text-primary">{label}</div>
                <div className={`text-2xl font-bold text-${color}-600 dark:text-${color}-300`}>
                  {missRate}%
                </div>
                <div className="text-[10px] text-text-tertiary">Cache Miss 率</div>
                <div className="h-2 bg-bg-secondary rounded-full overflow-hidden">
                  <div className={`h-full bg-${color}-500 rounded-full`}
                    style={{ width: `${missRate}%` }} />
                </div>
                <div className="text-[10px] text-text-secondary">
                  {n - hits} miss / {n} total
                </div>
              </div>
            ))}
          </div>

          {/* 估算耗时 */}
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3 space-y-2 text-xs text-text-secondary">
            <div className="font-semibold text-text-primary">估算相对耗时（n={n} 次访问）</div>
            <div className="text-text-tertiary text-[10px]">假设：hit=1 cycle, miss=100 cycles（L3 cache miss 参考值）</div>
            <div className="flex gap-4">
              <div>数组：<span className="text-text-primary font-mono">{arrTime}</span> cycles</div>
              <div>链表：<span className="text-text-primary font-mono">{llTime}</span> cycles</div>
              <div className="text-rose-600 dark:text-rose-400 font-semibold">链表慢约 {slowdown}×</div>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="text-[10px] w-10 text-text-tertiary shrink-0">数组</div>
                <div className="flex-1 h-3 bg-bg-secondary rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${(arrTime / llTime) * 100}%` }} />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="text-[10px] w-10 text-text-tertiary shrink-0">链表</div>
                <div className="flex-1 h-3 bg-bg-secondary rounded-full overflow-hidden">
                  <div className="h-full bg-rose-500 rounded-full" style={{ width: "100%" }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === "explain" && (
        <div className="space-y-3 text-xs text-text-secondary">
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4 space-y-2">
            <div className="font-semibold text-text-primary">什么是 Cache Line？</div>
            <div>CPU 缓存以 <span className="text-text-primary font-medium">cache line（64字节 = 16个int）</span> 为单位加载内存。
              访问任何一个地址，整个 cache line 会被加载到 L1/L2 缓存中。</div>
            <div className="bg-bg-secondary rounded-lg p-2 font-mono text-[11px] text-text-primary">
              <div>访问 arr[0] → 加载地址 0x100~0x13F（含 arr[0..15]）</div>
              <div>访问 arr[1..15] → 全部 cache <span className="text-emerald-600 dark:text-emerald-300">HIT</span> ✅</div>
            </div>
          </div>

          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4 space-y-2">
            <div className="font-semibold text-text-primary">为什么链表 Cache Miss 率高？</div>
            <div>链表节点在堆上<span className="text-text-primary">随机分配</span>，相邻节点地址差往往远大于 64 字节，
              导致访问每个节点都触发新 cache line 加载，缓存几乎形同虚设。</div>
            <div className="bg-bg-secondary rounded-lg p-2 font-mono text-[11px] text-text-primary">
              <div>node1 @ 0x100 → 加载 0x100~0x13F</div>
              <div>node2 @ 0x2080 → <span className="text-rose-600 dark:text-rose-400">MISS</span>，重新加载 0x2080~0x20BF</div>
              <div>node3 @ 0x3F40 → <span className="text-rose-600 dark:text-rose-400">MISS</span>，重新加载...</div>
            </div>
          </div>

          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4 space-y-2">
            <div className="font-semibold text-text-primary">实测数据参考（n = 10^7）</div>
            <div className="overflow-x-auto">
              <table className="w-full text-[10px] border-collapse">
                <thead>
                  <tr className="border-b border-border-subtle text-text-tertiary">
                    <th className="text-left py-1 pr-3">结构</th>
                    <th className="text-left py-1 pr-3">遍历耗时</th>
                    <th className="text-left py-1">Cache Miss 率</th>
                  </tr>
                </thead>
                <tbody className="text-text-secondary">
                  <tr className="border-b border-border-subtle/50">
                    <td className="py-1 pr-3 text-text-primary">int[] 数组</td>
                    <td className="py-1 pr-3 text-emerald-600 dark:text-emerald-300">≈20 ms</td>
                    <td className="py-1 text-emerald-600 dark:text-emerald-300">&lt; 1%</td>
                  </tr>
                  <tr>
                    <td className="py-1 pr-3 text-text-primary">LinkedList</td>
                    <td className="py-1 pr-3 text-rose-600 dark:text-rose-400">≈150 ms（7.5×）</td>
                    <td className="py-1 text-rose-600 dark:text-rose-400">≈40%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
