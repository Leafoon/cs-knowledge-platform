'use client';

import React, { useState } from 'react';

interface Exercise {
  id: number;
  title: string;
  difficulty: 'easy' | 'medium' | 'hard';
  category: string;
  description: string;
  hint: string;
}

const exercises: Exercise[] = [
  { id: 1, title: '实现简单的算子融合 Pass', difficulty: 'medium', category: 'Pass', description: '编写一个 Relay Pass，将连续的 element-wise 算子（如 add + relu）融合为一个函数节点。', hint: '使用 relay.transform.FuseOps 并设置 fuse_opt_level 参数' },
  { id: 2, title: '自定义 PackedFunc 注册', difficulty: 'easy', category: 'Runtime', description: '在 C++ 端注册一个自定义 PackedFunc，并从 Python 端调用它。', hint: '使用 TVM_REGISTER_GLOBAL("my_func").set_body(...)' },
  { id: 3, title: '编写 TIR 调度原语', difficulty: 'hard', category: 'TIR', description: '为一个矩阵乘法 TE 计算定义完整的调度，包含 split、reorder、vectorize 和 bind 操作。', hint: 's.split(i, factor=32); s.reorder(ko, ii, ki); s.vectorize(ji)' },
  { id: 4, title: '分析编译日志', difficulty: 'easy', category: 'Debug', description: '使用 PassContext 的 tracing 功能，记录并分析每个 Pass 的输入输出 IR。', hint: 'with tvm.transform.PassContext(config={"relay.FuseOps.cc_max_depth": 4})' },
  { id: 5, title: 'AutoTVM 调优模板', difficulty: 'hard', category: 'AutoTVM', description: '为自定义算子编写 AutoTVM 调优模板，定义搜索空间并执行调优。', hint: '@autotvm.template def schedule_conv: ... define_search_space(...)' },
  { id: 6, title: 'RPC 远程执行测试', difficulty: 'medium', category: 'RPC', description: '配置 TVM RPC 服务器，在远程设备上执行编译后的模型并验证正确性。', hint: 'remote = rpc.connect(host, port); remote.upload(lib_path)' },
];

const diffColors = {
  easy: 'bg-green-500/20 text-green-400 border-green-500/30',
  medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  hard: 'bg-red-500/20 text-red-400 border-red-500/30',
};

const diffLabels = { easy: '简单', medium: '中等', hard: '困难' };

export function PracticeExercises() {
  const [activeId, setActiveId] = useState<number | null>(null);
  const [showHint, setShowHint] = useState<Set<number>>(new Set());
  const [filter, setFilter] = useState<string | null>(null);

  const categories = [...new Set(exercises.map((e) => e.category))];
  const filtered = filter ? exercises.filter((e) => e.category === filter) : exercises;

  const toggleHint = (id: number) => {
    setShowHint((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="w-full rounded-xl border border-white/10 bg-gradient-to-br from-gray-900 via-gray-950 to-black p-6">
      <h3 className="mb-2 text-lg font-bold text-white">练习题面板</h3>
      <p className="mb-4 text-sm text-gray-400">
        通过动手练习巩固 TVM 编译器的核心概念，涵盖 Pass、Runtime、TIR、AutoTVM 等主题。
      </p>
      <div className="mb-4 flex gap-2 flex-wrap">
        <button
          onClick={() => setFilter(null)}
          className={`rounded-lg px-3 py-1 text-xs font-medium transition-all ${!filter ? 'bg-indigo-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}
        >
          全部
        </button>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            className={`rounded-lg px-3 py-1 text-xs font-medium transition-all ${filter === cat ? 'bg-indigo-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}
          >
            {cat}
          </button>
        ))}
      </div>
      <div className="space-y-2">
        {filtered.map((ex) => (
          <div
            key={ex.id}
            className={`rounded-lg border p-3 transition-all cursor-pointer ${activeId === ex.id ? 'border-indigo-500/40 bg-indigo-950/20' : 'border-white/10 bg-white/5 hover:bg-white/10'}`}
            onClick={() => setActiveId(activeId === ex.id ? null : ex.id)}
          >
            <div className="flex items-center gap-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-indigo-600 text-[11px] font-bold text-white">
                {ex.id}
              </span>
              <span className="flex-1 text-sm text-white">{ex.title}</span>
              <span className="rounded-full bg-purple-500/20 border border-purple-500/30 px-2 py-0.5 text-[10px] text-purple-300">
                {ex.category}
              </span>
              <span className={`rounded-full border px-2 py-0.5 text-[10px] ${diffColors[ex.difficulty]}`}>
                {diffLabels[ex.difficulty]}
              </span>
            </div>
            {activeId === ex.id && (
              <div className="mt-3 ml-9 space-y-2">
                <p className="text-xs text-gray-300">{ex.description}</p>
                <button
                  onClick={(e) => { e.stopPropagation(); toggleHint(ex.id); }}
                  className="rounded bg-indigo-600/30 px-2 py-1 text-[11px] text-indigo-300 hover:bg-indigo-600/50"
                >
                  {showHint.has(ex.id) ? '隐藏提示' : '💡 查看提示'}
                </button>
                {showHint.has(ex.id) && (
                  <div className="rounded bg-black/40 border border-white/5 p-2 text-xs text-yellow-300">
                    {ex.hint}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
