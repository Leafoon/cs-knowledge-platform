'use client';

import { useState } from 'react';

interface LoopStep {
  label: string;
  iters: string;
  subtotal: string;
  note: string;
  highlight: boolean;
}

interface LoopExample {
  title: string;
  code: string;
  result: string;
  steps: LoopStep[];
  tags: string[];
}

const EXAMPLES: LoopExample[] = [
  {
    title: '单层循环',
    code: `for i in range(n):
    op()  # O(1)`,
    result: 'O(n)',
    tags: ['基础', '线性'],
    steps: [
      { label: '外层：for i in range(n)', iters: 'n 次', subtotal: 'n', note: 'i 从 0 到 n-1，共 n 次迭代', highlight: true },
      { label: '内部：op()', iters: '每次 O(1)', subtotal: '1', note: '常数次操作', highlight: false },
      { label: '总计', iters: 'n × 1', subtotal: 'n', note: 'T(n) = n → O(n)', highlight: true },
    ],
  },
  {
    title: '独立双层嵌套',
    code: `for i in range(n):
    for j in range(n):
        op()  # O(1)`,
    result: 'O(n²)',
    tags: ['嵌套', '平方'],
    steps: [
      { label: '外层：for i in range(n)', iters: 'n 次', subtotal: '—', note: 'i 从 0 到 n-1', highlight: false },
      { label: '内层：for j in range(n)', iters: 'n 次（每轮外层）', subtotal: '—', note: '内层与外层无关，固定 n 次', highlight: false },
      { label: '内部：op()', iters: 'O(1)', subtotal: '1', note: '', highlight: false },
      { label: '总计', iters: 'n × n × 1', subtotal: 'n²', note: 'T(n) = n² → O(n²)', highlight: true },
    ],
  },
  {
    title: '三角形双层（j<i）',
    code: `for i in range(n):
    for j in range(i):
        op()  # O(1)`,
    result: 'O(n²)',
    tags: ['上三角', '平方'],
    steps: [
      { label: '外层：for i in range(n)', iters: 'n 次', subtotal: '—', note: 'i = 0, 1, ..., n-1', highlight: false },
      { label: '内层：for j in range(i)', iters: 'i 次（随 i 变化）', subtotal: '—', note: 'i=0时0次，i=1时1次，...，i=n-1时n-1次', highlight: false },
      { label: '计算总和', iters: 'Σ(i=0→n-1) i = n(n-1)/2', subtotal: 'n(n-1)/2', note: '等差数列求和公式', highlight: true },
      { label: '总计', iters: 'n(n-1)/2', subtotal: '≈ n²/2', note: 'T(n) = n(n-1)/2 = Θ(n²)（常数 1/2 忽略）', highlight: true },
    ],
  },
  {
    title: '内层指数增长（j*=2）',
    code: `for i in range(n):
    j = 1
    while j < n:
        op()
        j *= 2`,
    result: 'O(n log n)',
    tags: ['对数', 'n log n'],
    steps: [
      { label: '外层：for i in range(n)', iters: 'n 次', subtotal: '—', note: '', highlight: false },
      { label: '内层：while j < n（j*=2）', iters: '⌊log₂n⌋+1 次', subtotal: 'log n', note: 'j = 1, 2, 4, ..., 2^⌊log n⌋ < n，共 ⌊log₂n⌋+1 步', highlight: false },
      { label: '总计', iters: 'n × log n', subtotal: 'n log n', note: 'T(n) = n·⌊log₂n⌋ = Θ(n log n)', highlight: true },
    ],
  },
  {
    title: '外层指数缩减（i//=2）',
    code: `i = n
while i > 0:
    j = 0
    while j < i:
        op()
        j += 1
    i //= 2`,
    result: 'O(n)',
    tags: ['等比数列', '线性'],
    steps: [
      { label: '外层：while i > 0（i//=2）', iters: '⌊log₂n⌋+1 次', subtotal: '—', note: 'i = n, n/2, n/4, ..., 1', highlight: false },
      { label: '内层：while j < i', iters: 'i 次（随外层 i 变化）', subtotal: '—', note: '第k轮外层：i = n/2^k', highlight: false },
      { label: '计算总和', iters: 'n + n/2 + n/4 + ... + 1', subtotal: '≤ 2n', note: '等比数列求和：n(1 - 1/2^⌊log n⌋+1)/(1-1/2) < 2n', highlight: true },
      { label: '总计', iters: '< 2n', subtotal: '2n', note: 'T(n) = Θ(n)！虽然是双层循环，但内层缩减，总和是线性的', highlight: true },
    ],
  },
  {
    title: '三层嵌套（矩阵乘法）',
    code: `for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i][j] += A[i][k]*B[k][j]`,
    result: 'O(n³)',
    tags: ['三层', '立方'],
    steps: [
      { label: '外层 i：range(n)', iters: 'n 次', subtotal: '—', note: '', highlight: false },
      { label: '中层 j：range(n)', iters: 'n 次', subtotal: '—', note: '', highlight: false },
      { label: '内层 k：range(n)', iters: 'n 次', subtotal: '—', note: '', highlight: false },
      { label: '内层操作：O(1)', iters: '1', subtotal: '1', note: '', highlight: false },
      { label: '总计', iters: 'n × n × n × 1', subtotal: 'n³', note: 'T(n) = n³ = Θ(n³)', highlight: true },
    ],
  },
];

const RESULT_COLORS: Record<string, string> = {
  'O(n)': '#22c55e',
  'O(n²)': '#f97316',
  'O(n³)': '#ef4444',
  'O(n log n)': '#f59e0b',
};

export default function NestLoopAnalyzer() {
  const [active, setActive] = useState(0);
  const [step, setStep] = useState<number | null>(null);

  const eg = EXAMPLES[active];

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-4 space-y-4">
      <h3 className="text-base font-semibold text-text-primary">🔍 循环复杂度逐步推导</h3>
      <p className="text-xs text-text-secondary">选择示例，逐步分析每层循环的迭代次数，最终累乘得出总复杂度。</p>

      {/* Example tabs */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((e, i) => (
          <button
            key={i}
            onClick={() => { setActive(i); setStep(null); }}
            className={`px-3 py-1 rounded-full text-xs border transition-colors ${i === active
              ? 'border-blue-400 bg-blue-400/10 text-blue-400'
              : 'border-border-subtle text-text-secondary hover:border-blue-400 hover:text-blue-400'
            }`}
          >
            {e.title}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        {/* Code panel */}
        <div className="rounded-lg bg-bg-primary border border-border-subtle p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs text-text-tertiary font-medium">代码示例</span>
            <div className="flex gap-1">
              {eg.tags.map(t => (
                <span key={t} className="text-xs px-1.5 py-0.5 rounded bg-bg-tertiary text-text-tertiary">{t}</span>
              ))}
            </div>
          </div>
          <pre className="text-xs font-mono text-text-primary whitespace-pre leading-relaxed">{eg.code}</pre>
          <div className="mt-3 pt-3 border-t border-border-subtle flex items-center gap-2">
            <span className="text-xs text-text-tertiary">结论：</span>
            <span className="font-mono font-bold text-sm" style={{ color: RESULT_COLORS[eg.result] ?? '#3b82f6' }}>
              {eg.result}
            </span>
          </div>
        </div>

        {/* Steps panel */}
        <div className="space-y-2">
          <p className="text-xs text-text-tertiary font-medium">逐层分析（点击每步查看说明）</p>
          {eg.steps.map((s, si) => (
            <div
              key={si}
              onClick={() => setStep(step === si ? null : si)}
              className={`rounded-lg border p-3 cursor-pointer transition-all space-y-1 ${step === si
                ? 'border-blue-400 bg-blue-400/10'
                : s.highlight
                  ? 'border-amber-400/40 bg-amber-400/5 hover:border-amber-400/70'
                  : 'border-border-subtle hover:border-border-primary'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-xs font-mono ${s.highlight ? 'text-amber-300 font-semibold' : 'text-text-secondary'}`}>
                  {s.label}
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-text-tertiary">{s.iters}</span>
                  {s.highlight && (
                    <span className="text-xs font-mono font-bold px-1.5 py-0.5 rounded bg-amber-400/20 text-amber-300">
                      ={s.subtotal}
                    </span>
                  )}
                </div>
              </div>
              {step === si && s.note && (
                <p className="text-xs text-text-tertiary pt-1 border-t border-border-subtle">{s.note}</p>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 flex items-center gap-4 flex-wrap">
        <span className="text-xs text-text-tertiary">📐 分析规则：</span>
        <span className="text-xs text-text-secondary">顺序执行 → 取最大阶</span>
        <span className="text-xs text-text-tertiary">|</span>
        <span className="text-xs text-text-secondary">嵌套循环 → 迭代次数相乘</span>
        <span className="text-xs text-text-tertiary">|</span>
        <span className="text-xs text-text-secondary">求和型内层 → 用等差/等比公式化简</span>
      </div>
    </div>
  );
}
