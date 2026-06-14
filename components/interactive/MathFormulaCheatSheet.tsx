'use client';
import React, { useState } from 'react';

// =====================================================
// MathFormulaCheatSheet — DSA Chapter 0
// 常用数学公式速查卡片，支持分类筛选
// =====================================================

type Category = '求和公式' | '对数/指数' | '渐进关系' | '概率工具' | '位运算';

interface Formula {
  category: Category;
  name: string;
  formula: string;          // LaTeX-like 人类可读表示，用 Unicode 符号
  meaning: string;          // 一句话说明
  example: string;          // 具体数值例子
  useCase: string;          // 算法中的应用场景
}

const FORMULAS: Formula[] = [
  // ── 求和公式 ──────────────────────────────────────
  {
    category: '求和公式',
    name: '等差数列求和',
    formula: 'Σk (k=1→n) = n(n+1)/2',
    meaning: '1+2+3+…+n 的总和',
    example: '1+2+…+100 = 100×101/2 = 5050',
    useCase: '分析两层嵌套循环（冒泡/选择排序）的总操作数',
  },
  {
    category: '求和公式',
    name: '平方和',
    formula: 'Σk² (k=1→n) = n(n+1)(2n+1)/6',
    meaning: '1²+2²+…+n² 的总和',
    example: '1+4+9+…+100 (n=10) = 385',
    useCase: '三层嵌套循环的操作数分析',
  },
  {
    category: '求和公式',
    name: '等比数列求和（以 r=2 为例）',
    formula: '1+2+4+…+2ⁿ = 2ⁿ⁺¹ − 1',
    meaning: '公比为 2 的等比数列之和',
    example: '1+2+4+8+16 = 31 = 2⁵−1',
    useCase: '完全二叉树节点总数；归并排序每层工作量',
  },
  {
    category: '求和公式',
    name: '调和级数',
    formula: 'Σ(1/k) (k=1→n) = Hₙ ≈ ln n + 0.5772',
    meaning: '1 + 1/2 + 1/3 + … + 1/n 趋近 ln n',
    example: 'H₁₀ ≈ 2.93, H₁₀₀ ≈ 5.19, H₁₀₀₀ ≈ 7.49',
    useCase: '快速排序平均比较次数；哈希碰撞分析；树高期望',
  },
  {
    category: '求和公式',
    name: '几何级数（|r|<1 时收敛）',
    formula: 'Σrᵏ (k=0→∞) = 1/(1−r)  (|r|<1)',
    meaning: '收敛等比数列无穷项之和',
    example: '1+1/2+1/4+… = 2',
    useCase: '摊销分析；递归树收敛性分析',
  },
  // ── 对数/指数 ──────────────────────────────────
  {
    category: '对数/指数',
    name: '换底公式',
    formula: 'log_a b = log_c b / log_c a',
    meaning: '任意底数之间相互转换（只差常数倍）',
    example: 'log₂ 8 = log₁₀ 8 / log₁₀ 2 = 0.903/0.301 = 3',
    useCase: 'O(log n) 中底数不影响渐进阶（底数只是常数因子）',
  },
  {
    category: '对数/指数',
    name: '对数乘法规则',
    formula: 'log(xy) = log x + log y',
    meaning: '乘法转加法',
    example: 'log₂(8×4) = log₂8 + log₂4 = 3+2 = 5',
    useCase: '分析带乘法结构的递归/分治算法',
  },
  {
    category: '对数/指数',
    name: '对数幂规则',
    formula: 'log(xᵇ) = b · log x',
    meaning: '指数提到前面变成系数',
    example: 'log₂(n²) = 2·log₂n',
    useCase: 'O(n log n) vs O(n²) 的区分',
  },
  {
    category: '对数/指数',
    name: 'Stirling 近似',
    formula: 'n! ≈ √(2πn)·(n/e)ⁿ  →  log(n!) ≈ n·log n',
    meaning: 'n 的阶乘的对数约等于 n log n',
    example: 'log₂(10!) ≈ 10·log₂10 ≈ 33.2  (精确值=21.8，量级正确)',
    useCase: '比较排序下界证明：n! 种排列 → 至少 Ω(n log n) 次比较',
  },
  // ── 渐进关系 ───────────────────────────────────
  {
    category: '渐进关系',
    name: '增长速度排序（慢→快）',
    formula: '1 < log n < √n < n < n·log n < n² < n³ < 2ⁿ < n!',
    meaning: '不同复杂度类别的大小关系（n→∞ 时）',
    example: 'n=100: log n≈7, √n=10, n=100, n²=10000, 2ⁿ≈10³⁰',
    useCase: '快速判断算法优劣；瓶颈定位',
  },
  {
    category: '渐进关系',
    name: '1+x ≤ eˣ',
    formula: '1+x ≤ eˣ  (对所有实数 x 成立)',
    meaning: '自然指数的基本不等式',
    example: '1+0.5 = 1.5 ≤ e⁰·⁵ ≈ 1.649',
    useCase: '概率界分析；随机算法失败概率界',
  },
  {
    category: '渐进关系',
    name: '(1−1/n)ⁿ ≈ 1/e',
    formula: 'lim(n→∞) (1−1/n)ⁿ = 1/e ≈ 0.368',
    meaning: '当 n 很大时，n 次独立成功概率为 1/n 的事件，全部失败的概率趋近 1/e',
    example: '哈希表：n 个球扔进 n 个桶，某个桶为空的概率≈1/e',
    useCase: '哈希表装载因子分析；生日悖论；随机化算法分析',
  },
  // ── 概率工具 ────────────────────────────────────
  {
    category: '概率工具',
    name: '期望的线性性',
    formula: 'E[X+Y] = E[X] + E[Y]  （X、Y 不必独立）',
    meaning: '多个随机变量的期望可以直接相加',
    example: '掷两颗骰子点数之和的期望 = 3.5+3.5 = 7',
    useCase: '随机化快速排序平均比较次数；哈希碰撞期望数',
  },
  {
    category: '概率工具',
    name: '指示随机变量',
    formula: 'E[Iₐ] = P(A)',
    meaning: '事件 A 的指示变量（0/1）的期望 = 事件发生的概率',
    example: '硬币正面 Iₐ：P=1/2，所以掷 n 次期望正面次数 = n·E[Iₐ] = n/2',
    useCase: '快速排序、随机排列分析；哈希表期望碰撞数',
  },
  {
    category: '概率工具',
    name: 'Markov 不等式',
    formula: 'P(X ≥ a) ≤ E[X] / a  （X≥0）',
    meaning: '非负随机变量超过某阈值的概率上界',
    example: '若购物平均花费 50 元，花超过 500 元的概率 ≤ 50/500 = 10%',
    useCase: '随机算法尾界分析（弱界但简单）',
  },
  {
    category: '概率工具',
    name: 'Union Bound（布尔不等式）',
    formula: 'P(A₁∪…∪Aₖ) ≤ ΣP(Aᵢ)',
    meaning: '多个事件至少一个发生，概率不超过各概率之和',
    example: 'k=3 个事件各有 1% 失败概率，总失败概率 ≤ 3%',
    useCase: '随机化算法整体失败概率界；近似算法分析',
  },
  // ── 位运算 ──────────────────────────────────────
  {
    category: '位运算',
    name: '判断奇偶',
    formula: 'n & 1 == 0 → 偶数；n & 1 == 1 → 奇数',
    meaning: '最低位是否为 1',
    example: '6 & 1 = 0（偶），7 & 1 = 1（奇）',
    useCase: '快速奇偶判断；区分两类元素',
  },
  {
    category: '位运算',
    name: '判断 2 的幂次',
    formula: 'n > 0 && (n & (n−1)) == 0',
    meaning: '2 的幂次只有 1 个 1 位，减 1 会翻转该位及更低位',
    example: '8=1000, 7=0111, 8&7=0 ✓；6=110, 5=101, 6&5=100≠0 ✗',
    useCase: '哈希表大小是否为 2 的幂；对齐检测',
  },
  {
    category: '位运算',
    name: '清除最低 1 位',
    formula: 'n & (n−1)',
    meaning: '将 n 的二进制中最右边的 1 清零',
    example: '12=1100 → 12&11 = 1100&1011 = 1000 = 8',
    useCase: '统计二进制中 1 的个数（Brian Kernighan 算法）',
  },
  {
    category: '位运算',
    name: '提取最低 1 位（lowbit）',
    formula: 'n & (−n)',
    meaning: '只保留最右边的 1，其余清零',
    example: '12=1100 → 12&(−12) = 0100 = 4',
    useCase: '树状数组（BIT/Fenwick Tree）的核心操作',
  },
];

const CATEGORIES: Category[] = ['求和公式', '对数/指数', '渐进关系', '概率工具', '位运算'];

const CATEGORY_COLORS: Record<Category, { bg: string; text: string; border: string; dot: string }> = {
  '求和公式':   { bg: 'bg-blue-500/10',   text: 'text-blue-400',   border: 'border-blue-500/30',   dot: 'bg-blue-400' },
  '对数/指数':  { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30', dot: 'bg-purple-400' },
  '渐进关系':   { bg: 'bg-emerald-500/10',text: 'text-emerald-400',border: 'border-emerald-500/30',dot: 'bg-emerald-400' },
  '概率工具':   { bg: 'bg-amber-500/10',  text: 'text-amber-400',  border: 'border-amber-500/30',  dot: 'bg-amber-400' },
  '位运算':     { bg: 'bg-rose-500/10',   text: 'text-rose-400',   border: 'border-rose-500/30',   dot: 'bg-rose-400' },
};

export default function MathFormulaCheatSheet() {
  const [selected, setSelected] = useState<Category | 'all'>('all');
  const [expanded, setExpanded] = useState<number | null>(null);
  const [search, setSearch] = useState('');

  const filtered = FORMULAS.filter(f => {
    const matchCat = selected === 'all' || f.category === selected;
    const q = search.toLowerCase();
    const matchSearch = !q || f.name.toLowerCase().includes(q) || f.meaning.toLowerCase().includes(q) || f.useCase.toLowerCase().includes(q);
    return matchCat && matchSearch;
  });

  return (
    <div className="my-8 rounded-2xl border border-border-subtle bg-bg-secondary overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border-subtle bg-bg-tertiary flex items-center gap-3">
        <span className="text-2xl">📐</span>
        <div>
          <h3 className="font-bold text-text-primary text-lg">数学公式速查卡</h3>
          <p className="text-sm text-text-tertiary">DSA 常用数学工具 · 点击展开详情</p>
        </div>
        <div className="ml-auto text-xs text-text-tertiary bg-bg-secondary px-2 py-1 rounded-full border border-border-subtle">
          {filtered.length} / {FORMULAS.length} 条
        </div>
      </div>

      {/* Controls */}
      <div className="px-6 py-3 border-b border-border-subtle space-y-3">
        {/* Search */}
        <div className="relative">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
          </svg>
          <input
            className="w-full pl-9 pr-4 py-2 rounded-lg bg-bg-tertiary border border-border-subtle text-sm text-text-primary placeholder-text-tertiary focus:outline-none focus:border-indigo-500 transition-colors"
            placeholder="搜索公式名称、用途..."
            value={search}
            onChange={e => { setSearch(e.target.value); setExpanded(null); }}
          />
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => { setSelected('all'); setExpanded(null); }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              selected === 'all'
                ? 'bg-indigo-500/20 text-indigo-400 border-indigo-500/40'
                : 'bg-bg-tertiary text-text-tertiary border-border-subtle hover:border-indigo-500/30 hover:text-text-secondary'
            }`}
          >
            全部 ({FORMULAS.length})
          </button>
          {CATEGORIES.map(cat => {
            const c = CATEGORY_COLORS[cat];
            const count = FORMULAS.filter(f => f.category === cat).length;
            return (
              <button
                key={cat}
                onClick={() => { setSelected(cat); setExpanded(null); }}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border flex items-center gap-1.5 ${
                  selected === cat
                    ? `${c.bg} ${c.text} ${c.border}`
                    : 'bg-bg-tertiary text-text-tertiary border-border-subtle hover:text-text-secondary'
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`}/>
                {cat} ({count})
              </button>
            );
          })}
        </div>
      </div>

      {/* Formula Cards */}
      <div className="p-4 space-y-2 max-h-[480px] overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="text-center py-10 text-text-tertiary text-sm">没有找到匹配的公式</div>
        ) : (
          filtered.map((f, idx) => {
            const c = CATEGORY_COLORS[f.category];
            const isOpen = expanded === idx;
            return (
              <div
                key={idx}
                className={`rounded-xl border transition-all cursor-pointer ${
                  isOpen
                    ? `${c.bg} ${c.border}`
                    : 'border-border-subtle bg-bg-tertiary hover:border-border-primary'
                }`}
                onClick={() => setExpanded(isOpen ? null : idx)}
              >
                {/* Row Header */}
                <div className="flex items-center gap-3 px-4 py-3">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${c.dot}`}/>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${c.bg} ${c.text} border ${c.border}`}>
                        {f.category}
                      </span>
                      <span className="text-sm font-semibold text-text-primary">{f.name}</span>
                    </div>
                    <code className="mt-1.5 block text-xs font-mono text-text-secondary leading-relaxed break-all">
                      {f.formula}
                    </code>
                  </div>
                  <svg
                    className={`w-4 h-4 flex-shrink-0 text-text-tertiary transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    fill="none" stroke="currentColor" viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/>
                  </svg>
                </div>

                {/* Expanded Detail */}
                {isOpen && (
                  <div className="px-4 pb-4 space-y-3 border-t border-border-subtle pt-3">
                    <div>
                      <span className="text-xs font-semibold text-text-tertiary uppercase tracking-wide">含义</span>
                      <p className="mt-1 text-sm text-text-primary">{f.meaning}</p>
                    </div>
                    <div className="rounded-lg bg-bg-primary/60 border border-border-subtle p-3">
                      <span className="text-xs font-semibold text-text-tertiary uppercase tracking-wide">具体例子</span>
                      <p className="mt-1 text-sm font-mono text-text-secondary">{f.example}</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-lg">🎯</span>
                      <div>
                        <span className="text-xs font-semibold text-text-tertiary uppercase tracking-wide">算法应用场景</span>
                        <p className="mt-0.5 text-sm text-text-primary">{f.useCase}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer hint */}
      <div className="px-6 py-3 border-t border-border-subtle flex items-center gap-2 text-xs text-text-tertiary">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        点击任意公式展开详情 · 5 大分类 · {FORMULAS.length} 条核心公式
      </div>
    </div>
  );
}
