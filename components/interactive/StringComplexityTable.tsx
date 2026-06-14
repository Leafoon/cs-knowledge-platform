"use client";

import React, { useState } from "react";

/** 字符串操作复杂度对比表：可按语言筛选，点击行展开详细说明 */

type Complexity = "O(1)" | "O(n)" | "O(n²)" | "O(log n)" | "O(n log n)" | "O(k)" | "O(m)" | "O(m+n)" | "O(n+m)" | "O(n·m)";

interface Op {
  name: string;
  category: string;
  python: { complexity: Complexity; note: string };
  cpp: { complexity: Complexity; note: string };
  detail: string;
  pitfall?: string;
  leetcode?: string;
}

const OPS: Op[] = [
  {
    name: "长度查询 len(s)",
    category: "基础",
    python: { complexity: "O(1)", note: "CPython 缓存 ob_size 字段" },
    cpp: { complexity: "O(1)", note: "std::string.size() / .length() 均 O(1)" },
    detail: "Python 的 str 对象直接存储长度字段，C++ 的 std::string 内部同样维护 size 成员，两者均为常数时间。",
    leetcode: "#3 无重复最长子串：频繁调用 len 非瓶颈",
  },
  {
    name: "索引访问 s[i]",
    category: "基础",
    python: { complexity: "O(1)", note: "内部基于 char 数组，直接偏移" },
    cpp: { complexity: "O(1)", note: "operator[] 直接指针偏移" },
    detail: "两者底层均为连续内存数组，随机访问 O(1)。注意 Python UTF-32 编码下索引仍 O(1)，但 UTF-8 字符串（如 Go/Rust）索引字节而非字符。",
    pitfall: '多字节字符的"字符数"与"字节数"不等；Python3 的 ord(s[i]) 与 C 的 s[i] 语义不同',
  },
  {
    name: "切片 s[l:r]",
    category: "基础",
    python: { complexity: "O(k)", note: "创建新字符串对象，k=r-l" },
    cpp: { complexity: "O(k)", note: "substr(l, k) 深度拷贝 k 个字符" },
    detail: "切片始终产生新字符串，长度为 k=r-l，所以是 O(k)。滑动窗口算法中尽量避免在内层循环做切片，改用双指针记录下标。",
    pitfall: "连续切片操作累积为 O(n²)，是字符串拼接外最常见的性能陷阱",
    leetcode: "#76 最小覆盖子串：结果记录下标而非直接切片",
  },
  {
    name: "拼接 s + t（单次）",
    category: "拼接",
    python: { complexity: "O(n+m)", note: "创建新字符串，复制两者" },
    cpp: { complexity: "O(n+m)", note: "operator+ 每次创建新 string" },
    detail: "单次拼接 O(n+m)，但在循环中累积 N 次则总代价 O(1+2+…+N) = O(N²)。",
    pitfall: "for 循环内 s += char 在 Python 中因 CPython 优化有时接近 O(n)，但不保证；在 C++ 中每次都是 O(n)",
  },
  {
    name: "循环拼接（N次）",
    category: "拼接",
    python: { complexity: "O(n²)", note: "''.join(list) 替代" },
    cpp: { complexity: "O(n²)", note: "改用 reserve + append/push_back" },
    detail: "每次拼接需复制当前已有字符串，第 i 次代价 O(i)，总代价 Σi = O(n²)。正确做法：Python 用 ''.join()，C++ 先 reserve 再 append/push_back，最终 O(n)。",
    pitfall: "这是最常见的字符串 TLE 原因！面试题中拼接结果前先思考是否可用 StringBuilder 思路",
  },
  {
    name: "join 拼接",
    category: "拼接",
    python: { complexity: "O(n)", note: "一次性计算总长度并分配" },
    cpp: { complexity: "O(n)", note: "reserve + append 模拟" },
    detail: "Python ''.join(lst) 内部先遍历求总长度，再分配一次内存逐个复制，整体 O(n)。C++ 可用 string result; result.reserve(total); 后逐段 append。",
    leetcode: "#6 Z 字形变换：每行用列表收集，最后 join",
  },
  {
    name: "查找 find / in",
    category: "搜索",
    python: { complexity: "O(n·m)", note: "内置朴素匹配或 Boyer-Moore-Horspool" },
    cpp: { complexity: "O(n·m)", note: "std::string::find 朴素匹配" },
    detail: "标准库的 find/in 在最坏情况均为 O(n·m)。Python 内部对某些场景有 BMH 优化，但不保证。若需 O(n) 搜索请手写 KMP 或使用正则（re 模块底层 NFA）。",
    leetcode: "#28 找出字符串中第一个匹配项下标：KMP 最佳",
  },
  {
    name: "反转 s[::-1]",
    category: "变换",
    python: { complexity: "O(n)", note: "slice with step=-1，新建对象" },
    cpp: { complexity: "O(n)", note: "std::reverse O(n) in-place" },
    detail: "Python 切片 [::-1] 返回新字符串，O(n) 时间 + O(n) 空间。C++ std::reverse 是原地 O(n)。若仅需双指针判断是否回文，可避免显式反转。",
    pitfall: "不要在回文判断中反转再比较（引入额外 O(n) 空间），双指针法可做到 O(1) 空间",
  },
  {
    name: "比较 s == t",
    category: "比较",
    python: { complexity: "O(n)", note: "逐字符比较，最坏 O(min(n,m))" },
    cpp: { complexity: "O(n)", note: "operator== 逐字节 memcmp" },
    detail: "字符串比较最坏需遍历所有字符，O(min(n,m))。Python 会先比较 id（intern 优化），若是同一对象则 O(1)，否则 O(n)。",
    pitfall: "大量字符串比较可改用哈希预处理，将 O(n) 降为均摊 O(1)",
  },
  {
    name: "哈希 hash(s)",
    category: "哈希",
    python: { complexity: "O(n)", note: "首次计算后缓存（CPython 3.3+）" },
    cpp: { complexity: "O(n)", note: "std::hash<string> 每次 O(n)" },
    detail: "Python 的 str 对象在首次哈希后将结果缓存到对象字段，后续调用 O(1)。C++ 的 std::hash<string> 无缓存，每次均为 O(n)，频繁哈希同一字符串时可手动缓存。",
    leetcode: "#49 字母异位词分组：以排序字符串或频次元组为 key",
  },
  {
    name: "转大/小写",
    category: "变换",
    python: { complexity: "O(n)", note: "s.lower() / s.upper() 新建对象" },
    cpp: { complexity: "O(n)", note: "std::transform + tolower" },
    detail: "逐字符转换，O(n) 无法避免。注意 Python 对 Unicode 的处理（如 ß.upper() = 'SS'），C++ 的 tolower 通常只处理 ASCII。",
  },
  {
    name: "分割 split",
    category: "解析",
    python: { complexity: "O(n)", note: "扫描一遍并创建子串列表" },
    cpp: { complexity: "O(n)", note: "stringstream / find_first_of 实现" },
    detail: "分割操作线性扫描字符串，O(n)，但产生的子串列表总长也为 O(n) 空间。",
  },
  {
    name: "正则匹配 re.match",
    category: "搜索",
    python: { complexity: "O(n·m)", note: "NFA 模拟，m 为正则长度" },
    cpp: { complexity: "O(n·m)", note: "std::regex 同样 NFA，常数大" },
    detail: "标准正则（NFA 模拟）复杂度为 O(n·m)，m 为正则模式长度。某些引擎可能存在指数级回溯！对于简单模式，手写双指针/KMP 更高效。面试中避免使用正则解决算法题。",
    pitfall: "ReDoS 攻击：精心构造输入让 NFA 指数级回溯，线上服务需格外小心",
  },
];

const CATEGORIES = ["全部", "基础", "拼接", "搜索", "变换", "比较", "哈希", "解析"];
const COMPLEXITY_COLORS: Record<string, string> = {
  "O(1)": "bg-green-500/20 text-green-300 border-green-500/40",
  "O(k)": "bg-blue-500/20 text-blue-300 border-blue-500/40",
  "O(m)": "bg-blue-500/20 text-blue-300 border-blue-500/40",
  "O(n)": "bg-blue-600/20 text-blue-200 border-blue-600/40",
  "O(log n)": "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
  "O(n log n)": "bg-amber-500/20 text-amber-300 border-amber-500/40",
  "O(m+n)": "bg-amber-500/20 text-amber-300 border-amber-500/40",
  "O(n·m)": "bg-red-500/20 text-red-300 border-red-500/40",
  "O(n²)": "bg-red-600/20 text-red-300 border-red-600/40",
};

type LangFilter = "both" | "python" | "cpp";

export default function StringComplexityTable() {
  const [category, setCategory] = useState("全部");
  const [lang, setLang] = useState<LangFilter>("both");
  const [expanded, setExpanded] = useState<string | null>(null);

  const filtered = OPS.filter((op) => category === "全部" || op.category === category);

  const toggle = (name: string) => setExpanded((prev) => (prev === name ? null : name));

  const Badge = ({ c }: { c: Complexity }) => (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs border font-mono font-bold ${COMPLEXITY_COLORS[c] || "bg-bg-tertiary text-text-secondary border-border-subtle"}`}>
      {c}
    </span>
  );

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 text-sm">
      <div>
        <h3 className="text-base font-bold text-text-primary">📊 字符串操作复杂度参考表</h3>
        <p className="text-xs text-text-tertiary mt-0.5">点击行查看详细说明与常见陷阱</p>
      </div>

      {/* 过滤控件 */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex gap-1 flex-wrap">
          <span className="text-xs text-text-tertiary self-center mr-1">分类：</span>
          {CATEGORIES.map((c) => (
            <button key={c} onClick={() => setCategory(c)}
              className={`px-2 py-1 rounded text-xs border transition-colors ${
                category === c ? "bg-blue-600 text-white border-blue-600" : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"
              }`}>
              {c}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          <span className="text-xs text-text-tertiary self-center mr-1">语言：</span>
          {(["both", "python", "cpp"] as LangFilter[]).map((l) => (
            <button key={l} onClick={() => setLang(l)}
              className={`px-2 py-1 rounded text-xs border transition-colors ${
                lang === l ? "bg-purple-600 text-white border-purple-600" : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-purple-400"
              }`}>
              {l === "both" ? "Python & C++" : l === "python" ? "🐍 Python" : "⚙️ C++"}
            </button>
          ))}
        </div>
      </div>

      {/* 复杂度图例 */}
      <div className="flex gap-2 flex-wrap text-xs">
        <span className="text-text-tertiary">复杂度：</span>
        {["O(1)", "O(n)", "O(n·m)", "O(n²)"].map((c) => (
          <span key={c} className={`px-2 py-0.5 rounded-full border ${COMPLEXITY_COLORS[c]}`}>{c}</span>
        ))}
      </div>

      {/* 表格 */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left py-2 pr-4 text-text-tertiary font-medium w-48">操作</th>
              <th className="text-left py-2 pr-4 text-text-tertiary font-medium w-12">类别</th>
              {(lang === "both" || lang === "python") && (
                <th className="text-left py-2 pr-4 text-text-tertiary font-medium">Python</th>
              )}
              {(lang === "both" || lang === "cpp") && (
                <th className="text-left py-2 text-text-tertiary font-medium">C++</th>
              )}
            </tr>
          </thead>
          <tbody>
            {filtered.map((op) => (
              <React.Fragment key={op.name}>
                <tr
                  onClick={() => toggle(op.name)}
                  className={`border-b border-border-subtle/40 cursor-pointer transition-colors hover:bg-bg-tertiary/60 ${expanded === op.name ? "bg-bg-tertiary" : ""}`}>
                  <td className="py-2.5 pr-4">
                    <div className="flex items-center gap-1.5">
                      <span className={`text-[10px] transition-transform ${expanded === op.name ? "rotate-90" : ""}`}>▶</span>
                      <span className="font-mono text-text-primary">{op.name}</span>
                    </div>
                  </td>
                  <td className="py-2.5 pr-4">
                    <span className="px-1.5 py-0.5 rounded bg-bg-tertiary text-text-tertiary border border-border-subtle text-[10px]">
                      {op.category}
                    </span>
                  </td>
                  {(lang === "both" || lang === "python") && (
                    <td className="py-2.5 pr-4">
                      <div className="space-y-1">
                        <Badge c={op.python.complexity} />
                        <div className="text-text-tertiary text-[10px] leading-tight">{op.python.note}</div>
                      </div>
                    </td>
                  )}
                  {(lang === "both" || lang === "cpp") && (
                    <td className="py-2.5">
                      <div className="space-y-1">
                        <Badge c={op.cpp.complexity} />
                        <div className="text-text-tertiary text-[10px] leading-tight">{op.cpp.note}</div>
                      </div>
                    </td>
                  )}
                </tr>
                {expanded === op.name && (
                  <tr className="bg-bg-tertiary border-b border-border-subtle">
                    <td colSpan={lang === "both" ? 4 : 3} className="px-4 py-3">
                      <div className="space-y-2">
                        <p className="text-text-secondary text-xs leading-relaxed">{op.detail}</p>
                        {op.pitfall && (
                          <div className="bg-red-500/10 border border-red-500/40 rounded p-2">
                            <span className="text-red-400 font-bold text-[10px]">⚠️ 常见陷阱：</span>
                            <span className="text-red-300 text-xs ml-1">{op.pitfall}</span>
                          </div>
                        )}
                        {op.leetcode && (
                          <div className="bg-blue-500/10 border border-blue-500/40 rounded p-2">
                            <span className="text-blue-400 font-bold text-[10px]">💡 LeetCode 关联：</span>
                            <span className="text-blue-300 text-xs ml-1">{op.leetcode}</span>
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {/* 汇总建议 */}
      <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
        <div className="text-xs font-bold text-text-primary mb-2">🏆 高频面试字符串性能优化拇指规则</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-text-secondary">
          <div className="flex items-start gap-1.5">
            <span className="text-green-400 mt-0.5">✓</span>
            <span>用 <span className="font-mono text-text-primary">''.join(list)</span> 代替循环 +=</span>
          </div>
          <div className="flex items-start gap-1.5">
            <span className="text-green-400 mt-0.5">✓</span>
            <span>滑动窗口用双指针，不切片</span>
          </div>
          <div className="flex items-start gap-1.5">
            <span className="text-green-400 mt-0.5">✓</span>
            <span>字符频次用 <span className="font-mono text-text-primary">int[26]</span> 代替 dict</span>
          </div>
          <div className="flex items-start gap-1.5">
            <span className="text-green-400 mt-0.5">✓</span>
            <span>回文检测用双指针，不反转</span>
          </div>
          <div className="flex items-start gap-1.5">
            <span className="text-red-400 mt-0.5">✗</span>
            <span>内层循环内做 find/in（O(n²) 风险）</span>
          </div>
          <div className="flex items-start gap-1.5">
            <span className="text-red-400 mt-0.5">✗</span>
            <span>用 re 处理可以手写的简单模式</span>
          </div>
        </div>
      </div>
    </div>
  );
}
