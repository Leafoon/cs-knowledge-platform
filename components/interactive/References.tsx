'use client';

import React, { useState } from 'react';

interface Reference {
  id: string;
  category: '论文' | '文档' | '代码' | '教程';
  title: string;
  authors?: string;
  venue?: string;
  url: string;
  desc: string;
  year?: number;
}

const references: Reference[] = [
  { id: 'r1', category: '论文', title: 'TVM: An Automated End-to-End Optimizing Compiler for Deep Learning', authors: 'Chen et al.', venue: 'OSDI 2018', url: 'https://arxiv.org/abs/1802.04799', desc: 'TVM 系统的开创性论文，提出端到端的深度学习编译优化框架', year: 2018 },
  { id: 'r2', category: '论文', title: 'Ansor: Generating High-Performance Tensor Programs for Deep Learning', authors: 'Zheng et al.', venue: 'OSDI 2020', url: 'https://arxiv.org/abs/2006.06762', desc: '基于搜索的自动调度方法，无需手动模板即可生成高性能张量程序', year: 2020 },
  { id: 'r3', category: '论文', title: 'Relay: A High-Level Compiler for Deep Learning', authors: 'Roesch et al.', venue: 'MLSys 2019', url: 'https://arxiv.org/abs/1901.04157', desc: 'Relay IR 的设计论文，支持高阶函数和控制流的深度学习 IR', year: 2019 },
  { id: 'r4', category: '论文', title: 'Learning to Optimize Tensor Programs', authors: 'Chen et al.', venue: 'NeurIPS 2018', url: 'https://arxiv.org/abs/1805.08166', desc: 'AutoTVM 的核心论文，使用机器学习模型预测最优调度参数', year: 2018 },
  { id: 'r5', category: '文档', title: 'TVM 官方文档', url: 'https://tvm.apache.org/docs/', desc: 'TVM 的完整 API 文档和使用指南' },
  { id: 'r6', category: '文档', title: 'TIR 语言参考', url: 'https://tvm.apache.org/docs/reference/langref/tir.html', desc: 'TIR（Tensor IR）的语法和语义参考文档' },
  { id: 'r7', category: '文档', title: 'Schedule 原语参考', url: 'https://tvm.apache.org/docs/reference/api/python/te.html', desc: 'TE Schedule 原语的完整参考：split/reorder/vectorize/bind 等', },
  { id: 'r8', category: '代码', title: 'Apache TVM GitHub 仓库', url: 'https://github.com/apache/tvm', desc: 'TVM 的源代码仓库，包含编译器、运行时和所有示例' },
  { id: 'r9', category: '代码', title: 'TVM 教程示例合集', url: 'https://tvm.apache.org/docs/topic/vta/tutorials/', desc: '官方教程代码，涵盖从入门到高级的各种示例' },
  { id: 'r10', category: '教程', title: 'TVM 深度学习编译器入门', url: 'https://tvm.apache.org/docs/tutorial/', desc: '从零开始学习 TVM 的系列教程' },
  { id: 'r11', category: '教程', title: 'Meta Schedule 教程', url: 'https://tvm.apache.org/docs/arch/meta_schedule.html', desc: 'Meta Schedule 自动调度框架的详细教程' },
  { id: 'r12', category: '论文', title: 'MetaSchedule: A Universal Framework for Auto-Tuning', authors: 'Shao et al.', venue: 'MLSys 2022', url: 'https://arxiv.org/abs/2203.06931', desc: '统一的自动调优框架，整合了多种搜索策略', year: 2022 },
];

const categoryColors = {
  '论文': { bg: 'bg-indigo-500/20', text: 'text-indigo-400', border: 'border-indigo-500/30' },
  '文档': { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/30' },
  '代码': { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' },
  '教程': { bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/30' },
};

const categoryIcons = { '论文': '📄', '文档': '📚', '代码': '💻', '教程': '🎓' };

export function References() {
  const [filter, setFilter] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const categories = ['论文', '文档', '代码', '教程'];
  const filtered = filter ? references.filter((r) => r.category === filter) : references;

  return (
    <div className="w-full rounded-xl border border-white/10 bg-gradient-to-br from-gray-900 via-gray-950 to-black p-6">
      <h3 className="mb-2 text-lg font-bold text-white">参考文献</h3>
      <p className="mb-4 text-sm text-gray-400">
        TVM 相关的论文、官方文档、代码仓库和学习教程汇总。
      </p>
      <div className="mb-4 flex gap-2 flex-wrap">
        <button
          onClick={() => setFilter(null)}
          className={`rounded-lg px-3 py-1 text-xs font-medium transition-all ${!filter ? 'bg-indigo-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}
        >
          全部 ({references.length})
        </button>
        {categories.map((cat) => {
          const count = references.filter((r) => r.category === cat).length;
          return (
            <button
              key={cat}
              onClick={() => setFilter(cat)}
              className={`rounded-lg px-3 py-1 text-xs font-medium transition-all ${filter === cat ? 'bg-indigo-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}
            >
              {categoryIcons[cat as keyof typeof categoryIcons]} {cat} ({count})
            </button>
          );
        })}
      </div>
      <div className="space-y-2">
        {filtered.map((ref) => {
          const c = categoryColors[ref.category];
          const isExpanded = expandedId === ref.id;
          return (
            <div
              key={ref.id}
              className={`rounded-lg border p-3 transition-all cursor-pointer ${isExpanded ? `${c.border} ${c.bg}` : 'border-white/10 bg-white/5 hover:bg-white/10'}`}
              onClick={() => setExpandedId(isExpanded ? null : ref.id)}
            >
              <div className="flex items-start gap-3">
                <span className={`mt-0.5 rounded-full border px-2 py-0.5 text-[10px] flex-shrink-0 ${c.bg} ${c.text} ${c.border}`}>
                  {ref.category}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-white">{ref.title}</div>
                  {ref.authors && (
                    <div className="text-[11px] text-gray-400">
                      {ref.authors} {ref.venue && <span>· {ref.venue}</span>} {ref.year && <span>· {ref.year}</span>}
                    </div>
                  )}
                </div>
              </div>
              {isExpanded && (
                <div className="mt-2 ml-12 space-y-2">
                  <p className="text-xs text-gray-300">{ref.desc}</p>
                  <a
                    href={ref.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-block rounded bg-indigo-600/30 px-2 py-1 text-[11px] text-indigo-300 hover:bg-indigo-600/50 break-all"
                    onClick={(e) => e.stopPropagation()}
                  >
                    🔗 {ref.url}
                  </a>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-4 text-center text-[10px] text-gray-600">
        点击展开查看详情和链接 · 共 {references.length} 条参考
      </div>
    </div>
  );
}
