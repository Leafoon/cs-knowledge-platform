'use client';

import { useState } from 'react';

const checklist = [
  { category: '内存优化', items: ['合并内存访问', '使用共享内存', '减少全局内存访问', '向量化加载'] },
  { category: '计算优化', items: ['循环展开', '指令级并行', 'Tensor Core利用', '减少分支发散'] },
  { category: '并行优化', items: ['调整线程块大小', '提高占用率', '减少同步开销', '负载均衡'] },
  { category: '融合优化', items: ['逐点算子融合', '归约融合', '生产者-消费者融合', '内核自动调优'] },
];

export function OptimizationChecklist() {
  const [checked, setChecked] = useState<Record<string, boolean>>({});

  const toggle = (key: string) => {
    setChecked((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const totalItems = checklist.reduce((sum, cat) => sum + cat.items.length, 0);
  const checkedCount = Object.values(checked).filter(Boolean).length;

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">优化检查清单</h2>
      
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-600">完成进度</span>
          <span className="text-gray-800">{checkedCount}/{totalItems}</span>
        </div>
        <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-green-400 to-green-600 transition-all"
            style={{ width: `${(checkedCount / totalItems) * 100}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {checklist.map((cat) => (
          <div key={cat.category} className="p-4 bg-gray-50 rounded-xl">
            <div className="font-semibold text-gray-700 mb-3">{cat.category}</div>
            <div className="space-y-2">
              {cat.items.map((item) => {
                const key = `${cat.category}-${item}`;
                return (
                  <label
                    key={item}
                    className="flex items-center gap-2 cursor-pointer"
                    onClick={() => toggle(key)}
                  >
                    <div
                      className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                        checked[key]
                          ? 'bg-green-500 border-green-500'
                          : 'border-gray-300 hover:border-green-400'
                      }`}
                    >
                      {checked[key] && <span className="text-white text-xs">✓</span>}
                    </div>
                    <span className={`text-sm ${checked[key] ? 'text-gray-400 line-through' : 'text-gray-700'}`}>
                      {item}
                    </span>
                  </label>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}