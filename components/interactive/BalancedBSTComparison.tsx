"use client";
import React, { useState } from "react";

/** 平衡 BST 综合对比：AVL / 红黑树 / Treap / Splay / B树 / 跳表 */

interface TreeInfo {
  name: string;
  short: string;
  color: string;
  height: string;
  search: string;
  insert: string;
  delete: string;
  rotate_insert: string;
  rotate_delete: string;
  stable: string;
  cache: string;
  usecase: string;
  example: string;
  pros: string[];
  cons: string[];
}

const TREES: TreeInfo[] = [
  {
    name: "AVL 树",
    short: "AVL",
    color: "#3b82f6",
    height: "≤ 1.44 log₂ n",
    search: "O(log n)",
    insert: "O(log n)",
    delete: "O(log n)",
    rotate_insert: "最多 1 次（单/双旋）",
    rotate_delete: "最多 O(log n) 次",
    stable: "✅ 严格平衡",
    cache: "⭐⭐⭐⭐",
    usecase: "读多写少：数据库内存索引、编译器符号表",
    example: "OpenBSD stdlib, Java AVLTree",
    pros: ["高度最低，查找最快", "精确平衡保证"],
    cons: ["插入/删除旋转次数多", "实现复杂（需维护 height/bf）"],
  },
  {
    name: "红黑树",
    short: "RB",
    color: "#ef4444",
    height: "≤ 2 log₂(n+1)",
    search: "O(log n)",
    insert: "O(log n)",
    delete: "O(log n)",
    rotate_insert: "最多 2 次",
    rotate_delete: "最多 3 次",
    stable: "✅ 近似平衡",
    cache: "⭐⭐⭐",
    usecase: "读写均衡：OS 调度、语言库 Map/Set",
    example: "Linux CFS, Java TreeMap, C++ std::map",
    pros: ["旋转次数少（写操作快）", "工程实现成熟"],
    cons: ["高度可能是 AVL 的 1.4 倍", "5 条性质 + 多情形删除复杂"],
  },
  {
    name: "Treap（树堆）",
    short: "Treap",
    color: "#8b5cf6",
    height: "期望 O(log n)",
    search: "期望 O(log n)",
    insert: "期望 O(log n)",
    delete: "期望 O(log n)",
    rotate_insert: "期望 O(log n) 次",
    rotate_delete: "期望 O(log n) 次",
    stable: "🎲 期望平衡（概率保证）",
    cache: "⭐⭐⭐",
    usecase: "竞赛编程（实现简单）、可持久化结构",
    example: "IOI/ICPC 选手实现，可持久化 Treap",
    pros: ["实现极其简单（无颜色/bf）", "支持 Split/Merge（灵活扩展）"],
    cons: ["最坏 O(n) 概率极小但存在", "性能依赖随机数质量"],
  },
  {
    name: "Splay 树",
    short: "Splay",
    color: "#f59e0b",
    height: "摊销 O(log n)",
    search: "摊销 O(log n)",
    insert: "摊销 O(log n)",
    delete: "摊销 O(log n)",
    rotate_insert: "摊销 O(log n) 次",
    rotate_delete: "摊销 O(log n) 次",
    stable: "⚡ 摊销平衡",
    cache: "⭐⭐⭐⭐⭐",
    usecase: "访问局部性强：缓存（最近用到的接近根）",
    example: "GCC STL rope, 一些编辑器底层",
    pros: ["无需存储额外字段", "访问热点自动优化（自组织）"],
    cons: ["单次操作可能 O(n)", "不适合无局部性的均匀访问"],
  },
  {
    name: "B+ 树",
    short: "B+",
    color: "#10b981",
    height: "O(log_t n)",
    search: "O(t log_t n)",
    insert: "O(t log_t n)",
    delete: "O(t log_t n)",
    rotate_insert: "1 次分裂（O(t)）",
    rotate_delete: "1 次合并（O(t)）",
    stable: "✅ 完全平衡（每层均满）",
    cache: "⭐⭐⭐⭐⭐（磁盘友好）",
    usecase: "磁盘数据库索引（减少 I/O 次数）",
    example: "MySQL InnoDB, PostgreSQL, ext4 目录",
    pros: ["磁盘 I/O 最优（宽节点）", "叶链表支持O(k)范围查询"],
    cons: ["不适合纯内存场景", "节点分裂/合并逻辑复杂"],
  },
  {
    name: "跳表",
    short: "Skip",
    color: "#64748b",
    height: "期望 O(log n) 层",
    search: "期望 O(log n)",
    insert: "期望 O(log n)",
    delete: "期望 O(log n)",
    rotate_insert: "无旋转（指针更新）",
    rotate_delete: "无旋转（指针更新）",
    stable: "🎲 概率平衡",
    cache: "⭐⭐",
    usecase: "并发（更易实现无锁）、Redis ZSet",
    example: "Redis ZSet（有序集合）、LevelDB",
    pros: ["无旋转，并发实现简单", "实现直观，代码简洁"],
    cons: ["空间开销大（额外层指针）", "缓存不友好（随机访问）"],
  },
];

type MetricKey = "height" | "search" | "insert" | "delete" | "rotate_insert" | "rotate_delete";

const METRICS: { key: MetricKey; label: string }[] = [
  { key: "height", label: "高度上界" },
  { key: "search", label: "查找时间" },
  { key: "insert", label: "插入时间" },
  { key: "delete", label: "删除时间" },
  { key: "rotate_insert", label: "插入旋转次数" },
  { key: "rotate_delete", label: "删除旋转次数" },
];

export default function BalancedBSTComparison() {
  const [selected, setSelected] = useState<string | null>(null);

  const sel = selected ? TREES.find((t) => t.short === selected) : null;

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-1">📊 平衡 BST 家族对比</h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-5">点击行查看详细特性对比。颜色代表不同树类型。</p>

      {/* Comparison table */}
      <div className="overflow-x-auto mb-5">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-slate-50 dark:bg-slate-800">
              <th className="py-2 px-3 text-left font-semibold text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700">数据结构</th>
              {METRICS.map((m) => (
                <th key={m.key} className="py-2 px-3 text-center font-semibold text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 whitespace-nowrap">{m.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {TREES.map((t) => (
              <tr
                key={t.short}
                onClick={() => setSelected(selected === t.short ? null : t.short)}
                className={`cursor-pointer transition-colors ${selected === t.short ? "bg-indigo-50 dark:bg-indigo-900/20" : "hover:bg-slate-50 dark:hover:bg-slate-800/60"}`}
              >
                <td className="py-2 px-3 border border-slate-200 dark:border-slate-700">
                  <span className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full inline-block flex-shrink-0" style={{ background: t.color }} />
                    <span className="font-medium text-slate-700 dark:text-slate-200">{t.name}</span>
                  </span>
                </td>
                {METRICS.map((m) => (
                  <td key={m.key} className="py-2 px-3 text-center text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700 whitespace-nowrap font-mono text-[11px]">
                    {t[m.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Detail panel */}
      {sel && (
        <div className="rounded-xl border-2 p-4 mb-4 transition-all" style={{ borderColor: sel.color + "60", background: sel.color + "08" }}>
          <div className="flex items-center gap-2 mb-3">
            <span className="w-4 h-4 rounded-full" style={{ background: sel.color }} />
            <span className="font-bold text-slate-800 dark:text-slate-100">{sel.name}</span>
            <span className="text-xs text-slate-500 ml-2">适配度：缓存友好性 {sel.cache}</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div>
              <div className="font-semibold text-slate-600 dark:text-slate-300 mb-1">✅ 优势</div>
              {sel.pros.map((p, i) => <div key={i} className="text-slate-600 dark:text-slate-400">· {p}</div>)}
            </div>
            <div>
              <div className="font-semibold text-slate-600 dark:text-slate-300 mb-1">❌ 劣势</div>
              {sel.cons.map((c, i) => <div key={i} className="text-slate-600 dark:text-slate-400">· {c}</div>)}
            </div>
            <div>
              <div className="font-semibold text-slate-600 dark:text-slate-300 mb-1">🎯 典型使用场景</div>
              <div className="text-slate-600 dark:text-slate-400">{sel.usecase}</div>
            </div>
            <div>
              <div className="font-semibold text-slate-600 dark:text-slate-300 mb-1">🌍 真实工程案例</div>
              <div className="text-slate-600 dark:text-slate-400">{sel.example}</div>
            </div>
          </div>
        </div>
      )}

      {/* Quick decision guide */}
      <div className="rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 p-4">
        <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">💡 选型速查</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-slate-600 dark:text-slate-400">
          <div><span className="font-semibold text-blue-600 dark:text-blue-400">读多写少</span> → AVL 树（高度最低，查找最快）</div>
          <div><span className="font-semibold text-red-600 dark:text-red-400">读写均衡</span> → 红黑树（旋转次数有上界）</div>
          <div><span className="font-semibold text-purple-600 dark:text-purple-400">竞赛/快速实现</span> → Treap（代码最少）</div>
          <div><span className="font-semibold text-amber-600 dark:text-amber-400">访问局部性强</span> → Splay 树（自组织）</div>
          <div><span className="font-semibold text-emerald-600 dark:text-emerald-400">磁盘/数据库</span> → B+ 树（减少 I/O）</div>
          <div><span className="font-semibold text-slate-600 dark:text-slate-300">并发场景</span> → 跳表（无锁实现简单）</div>
        </div>
      </div>
    </div>
  );
}
