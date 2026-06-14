'use client';
import React, { useState } from 'react';

// =====================================================
// LearningPathNavigator — DSA Chapter 0
// 交互式学习路径规划器：选择目标 → 推荐章节路线图
// =====================================================

type Goal = 'interview' | 'contest' | 'research' | 'fullstack';

interface Part {
  id: string;
  label: string;           // e.g. "Part I"
  title: string;
  chapters: string;        // e.g. "Ch 0–2"
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  weeks: string;
  topics: string[];
  requiredFor: Goal[];     // 哪些目标需要这个 Part
  priority: Record<Goal, 'must' | 'recommended' | 'optional' | 'skip'>;
}

const PARTS: Part[] = [
  {
    id: 'p1', label: 'Part I', title: '基础概念与复杂度分析', chapters: 'Ch 0–2', difficulty: 'beginner', weeks: '3 周',
    topics: ['渐进记号 O/Ω/Θ', '主定理', '循环不变式', '数学归纳法', '递推式求解'],
    requiredFor: ['interview', 'contest', 'research', 'fullstack'],
    priority: { interview: 'must', contest: 'must', research: 'must', fullstack: 'must' },
  },
  {
    id: 'p2', label: 'Part II', title: '线性数据结构', chapters: 'Ch 3–6', difficulty: 'beginner', weeks: '2 周',
    topics: ['数组与动态数组', '链表', '栈与队列', '双端队列'],
    requiredFor: ['interview', 'contest', 'research', 'fullstack'],
    priority: { interview: 'must', contest: 'must', research: 'must', fullstack: 'must' },
  },
  {
    id: 'p3', label: 'Part III', title: '树与优先队列', chapters: 'Ch 7–11', difficulty: 'intermediate', weeks: '3 周',
    topics: ['二叉树', 'BST', '堆/优先队列', 'AVL 树', '红黑树'],
    requiredFor: ['interview', 'contest', 'research', 'fullstack'],
    priority: { interview: 'must', contest: 'must', research: 'must', fullstack: 'recommended' },
  },
  {
    id: 'p4', label: 'Part IV', title: '哈希与集合', chapters: 'Ch 12–13', difficulty: 'intermediate', weeks: '1.5 周',
    topics: ['哈希表', '开放寻址', '链地址法', '布隆过滤器'],
    requiredFor: ['interview', 'contest', 'research', 'fullstack'],
    priority: { interview: 'must', contest: 'must', research: 'recommended', fullstack: 'must' },
  },
  {
    id: 'p5', label: 'Part V', title: '排序与搜索', chapters: 'Ch 14–17', difficulty: 'intermediate', weeks: '2 周',
    topics: ['归并/快速/堆排序', '计数/基数排序', '二分搜索', '下界证明'],
    requiredFor: ['interview', 'contest', 'research', 'fullstack'],
    priority: { interview: 'must', contest: 'must', research: 'must', fullstack: 'must' },
  },
  {
    id: 'p6', label: 'Part VI', title: '图基础与遍历', chapters: 'Ch 18–20', difficulty: 'intermediate', weeks: '2 周',
    topics: ['图表示', 'BFS', 'DFS', '拓扑排序', '连通分量'],
    requiredFor: ['interview', 'contest', 'research', 'fullstack'],
    priority: { interview: 'must', contest: 'must', research: 'must', fullstack: 'recommended' },
  },
  {
    id: 'p7', label: 'Part VII', title: '图高级算法', chapters: 'Ch 21–25', difficulty: 'advanced', weeks: '3 周',
    topics: ['Dijkstra', 'Bellman-Ford', 'Floyd-Warshall', 'MST（Prim/Kruskal）', '网络流', 'SCC'],
    requiredFor: ['interview', 'contest', 'research'],
    priority: { interview: 'recommended', contest: 'must', research: 'must', fullstack: 'optional' },
  },
  {
    id: 'p8', label: 'Part VIII', title: '算法设计范式', chapters: 'Ch 26–30', difficulty: 'advanced', weeks: '4 周',
    topics: ['分治', '动态规划', '贪心', '回溯', '分支定界'],
    requiredFor: ['interview', 'contest', 'research', 'fullstack'],
    priority: { interview: 'must', contest: 'must', research: 'must', fullstack: 'recommended' },
  },
  {
    id: 'p9', label: 'Part IX', title: '字符串算法', chapters: 'Ch 31–34', difficulty: 'advanced', weeks: '2.5 周',
    topics: ['KMP', 'Rabin-Karp', 'Boyer-Moore', '后缀数组', 'Trie/AC 自动机'],
    requiredFor: ['contest', 'research'],
    priority: { interview: 'recommended', contest: 'must', research: 'must', fullstack: 'skip' },
  },
  {
    id: 'p10', label: 'Part X', title: '高级数据结构与摊销分析', chapters: 'Ch 35–38', difficulty: 'advanced', weeks: '3 周',
    topics: ['摊销分析', '斐波那契堆', 'Splay Tree', '跳表', '并查集'],
    requiredFor: ['contest', 'research'],
    priority: { interview: 'optional', contest: 'must', research: 'must', fullstack: 'skip' },
  },
  {
    id: 'p11', label: 'Part XI', title: '计算几何基础', chapters: 'Ch 39–40', difficulty: 'advanced', weeks: '1.5 周',
    topics: ['凸包', '线段交', '计算几何框架'],
    requiredFor: ['contest', 'research'],
    priority: { interview: 'skip', contest: 'recommended', research: 'optional', fullstack: 'skip' },
  },
  {
    id: 'p12', label: 'Part XII', title: '计算复杂性与 NP', chapters: 'Ch 41–42', difficulty: 'advanced', weeks: '2 周',
    topics: ['P vs NP', 'NP 完全问题', '近似算法', '复杂性归约'],
    requiredFor: ['research'],
    priority: { interview: 'optional', contest: 'optional', research: 'must', fullstack: 'skip' },
  },
];

const GOALS: { id: Goal; label: string; icon: string; desc: string; color: string }[] = [
  { id: 'interview', label: '技术面试', icon: '💼', desc: 'FAANG / 大厂面试，LeetCode 精通', color: 'indigo' },
  { id: 'contest', label: '算法竞赛', icon: '🏆', desc: 'ICPC / NOI / Codeforces，追求卓越', color: 'amber' },
  { id: 'research', label: '学术研究', icon: '🔬', desc: '系统设计、算法理论、PhD 方向', color: 'purple' },
  { id: 'fullstack', label: '全栈工程师', icon: '⚡', desc: '工程实践，以够用为主', color: 'emerald' },
];

const PRIORITY_STYLES = {
  must:        { label: '必学', bg: 'bg-rose-500/15', text: 'text-rose-300', border: 'border-rose-500/30', dot: 'bg-rose-400' },
  recommended: { label: '推荐', bg: 'bg-amber-500/15', text: 'text-amber-300', border: 'border-amber-500/30', dot: 'bg-amber-400' },
  optional:    { label: '可选', bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/30', dot: 'bg-blue-400' },
  skip:        { label: '跳过', bg: 'bg-bg-tertiary', text: 'text-text-quaternary', border: 'border-border-subtle', dot: 'bg-neutral-600' },
};

const DIFF_BADGE = {
  beginner:     'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  intermediate: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  advanced:     'bg-rose-500/15 text-rose-400 border-rose-500/30',
};
const DIFF_LABEL = { beginner: '🟢 初级', intermediate: '🟡 中级', advanced: '🔴 高级' };

export default function LearningPathNavigator() {
  const [goal, setGoal] = useState<Goal>('interview');

  const goalMeta = GOALS.find(g => g.id === goal)!;
  const mustParts        = PARTS.filter(p => p.priority[goal] === 'must');
  const recommendedParts = PARTS.filter(p => p.priority[goal] === 'recommended');
  const optionalParts    = PARTS.filter(p => p.priority[goal] === 'optional');
  const totalWeeks = mustParts.reduce((s, p) => s + parseFloat(p.weeks), 0)
                   + recommendedParts.reduce((s, p) => s + parseFloat(p.weeks) * 0.5, 0);

  return (
    <div className="my-8 rounded-2xl border border-border-subtle bg-bg-secondary overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border-subtle bg-bg-tertiary flex items-center gap-3">
        <span className="text-2xl">🗺️</span>
        <div>
          <h3 className="font-bold text-text-primary text-lg">个性化学习路径规划器</h3>
          <p className="text-sm text-text-tertiary">选择你的目标，获取推荐学习路线与优先级</p>
        </div>
      </div>

      {/* Goal Selector */}
      <div className="px-6 py-4 border-b border-border-subtle">
        <p className="text-xs font-semibold text-text-tertiary uppercase tracking-wide mb-3">我的学习目标是？</p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {GOALS.map(g => (
            <button
              key={g.id}
              onClick={() => setGoal(g.id)}
              className={`rounded-xl border p-3 text-left transition-all ${
                goal === g.id
                  ? `bg-${g.color}-500/15 border-${g.color}-500/40`
                  : 'bg-bg-tertiary border-border-subtle hover:border-border-primary'
              }`}
            >
              <div className="text-xl mb-1">{g.icon}</div>
              <div className={`text-sm font-semibold ${goal === g.id ? `text-${g.color}-300` : 'text-text-primary'}`}>{g.label}</div>
              <div className="text-xs text-text-tertiary mt-0.5 leading-snug">{g.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Summary Banner */}
      <div className="px-6 py-3 border-b border-border-subtle flex flex-wrap gap-4 items-center bg-bg-tertiary/50">
        <div className="text-sm text-text-secondary">
          {goalMeta.icon} <strong className="text-text-primary">{goalMeta.label}</strong> 路线
        </div>
        <div className="flex items-center gap-3 ml-auto flex-wrap">
          {(['must','recommended','optional','skip'] as const).map(pri => {
            const count = PARTS.filter(p => p.priority[goal] === pri).length;
            const s = PRIORITY_STYLES[pri];
            return (
              <span key={pri} className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs border ${s.bg} ${s.text} ${s.border}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`}/>
                {s.label} × {count}
              </span>
            );
          })}
          <span className="text-xs text-text-tertiary border-l border-border-subtle pl-3">
            预计 ≈ {Math.round(totalWeeks)} 周
          </span>
        </div>
      </div>

      {/* Parts List */}
      <div className="p-4 space-y-2 max-h-[460px] overflow-y-auto">
        {PARTS.map(part => {
          const pri = part.priority[goal];
          const s = PRIORITY_STYLES[pri];
          const diff = DIFF_BADGE[part.difficulty];
          return (
            <div
              key={part.id}
              className={`rounded-xl border px-4 py-3 flex items-center gap-4 transition-all ${
                pri === 'skip' ? 'opacity-35' : `${s.bg} ${s.border}`
              }`}
            >
              {/* Priority dot */}
              <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${s.dot}`}/>

              {/* Part info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs text-text-tertiary font-mono">{part.label} · {part.chapters}</span>
                  <span className="text-sm font-semibold text-text-primary">{part.title}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded border ${diff}`}>{DIFF_LABEL[part.difficulty]}</span>
                </div>
                <div className="flex flex-wrap gap-1 mt-1.5">
                  {part.topics.map(t => (
                    <span key={t} className="text-xs bg-bg-primary/50 border border-border-subtle text-text-tertiary px-1.5 py-0.5 rounded">
                      {t}
                    </span>
                  ))}
                </div>
              </div>

              {/* Right side */}
              <div className="flex flex-col items-end gap-1 flex-shrink-0">
                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${s.bg} ${s.text} ${s.border}`}>
                  {s.label}
                </span>
                <span className="text-xs text-text-tertiary">{part.weeks}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom tip */}
      <div className="px-6 py-4 border-t border-border-subtle bg-bg-tertiary/30">
        <p className="text-xs text-text-tertiary">
          💡 <strong className="text-text-secondary">学习建议</strong>：先把所有「必学」章节吃透，再按顺序学「推荐」，保持每进入新 Part 前先完成前置依赖。遇到困难不要跳过，回到前面巩固数学基础。
        </p>
      </div>
    </div>
  );
}
