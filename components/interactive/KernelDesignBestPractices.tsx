'use client';

import { useState } from 'react';

const practices = [
  {
    category: '内存优化',
    icon: '💾',
    color: 'blue',
    items: [
      { name: '共享内存使用', desc: '将全局内存数据加载到共享内存', priority: '高' },
      { name: '内存合并访问', desc: '确保线程束的连续内存访问', priority: '高' },
      { name: 'Bank Conflict 避免', desc: '避免共享内存访问冲突', priority: '中' },
      { name: '寄存器复用', desc: '重用寄存器减少溢出', priority: '中' },
    ],
  },
  {
    category: '计算优化',
    icon: '⚡',
    color: 'green',
    items: [
      { name: '分块计算', desc: '将大计算分解为小块', priority: '高' },
      { name: '指令级并行', desc: '增加指令级并行度', priority: '中' },
      { name: '循环展开', desc: '减少循环开销', priority: '中' },
      { name: '向量化操作', desc: '使用 SIMD 指令', priority: '高' },
    ],
  },
  {
    category: '调度策略',
    icon: '📋',
    color: 'purple',
    items: [
      { name: 'Tile 大小选择', desc: '根据硬件特性选择最优 Tile', priority: '高' },
      { name: '流水线调度', desc: '重叠计算和内存传输', priority: '高' },
      { name: '负载均衡', desc: '确保线程块间负载均衡', priority: '中' },
      { name: '占用率优化', desc: '最大化 SM 占用率', priority: '中' },
    ],
  },
  {
    category: '调试与验证',
    icon: '🔍',
    color: 'yellow',
    items: [
      { name: '数值精度检查', desc: '验证 FP16/BF16 精度', priority: '高' },
      { name: '边界条件处理', desc: '处理非整除的边界情况', priority: '高' },
      { name: '性能 Profiling', desc: '使用 profiler 定位瓶颈', priority: '中' },
      { name: '单元测试', desc: '编写全面的测试用例', priority: '中' },
    ],
  },
];

export default function KernelDesignBestPractices() {
  const [selectedCategory, setSelectedCategory] = useState<number>(0);
  const [checkedItems, setCheckedItems] = useState<Record<string, boolean>>({});

  const toggleItem = (key: string) => {
    setCheckedItems(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const totalItems = practices.reduce((sum, p) => sum + p.items.length, 0);
  const completedItems = Object.values(checkedItems).filter(Boolean).length;

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">内核设计最佳实践</h2>
      <p className="text-gray-400 text-sm mb-4">内核设计检查清单与优化指南</p>

      <div className="flex items-center gap-4 mb-6">
        <div className="flex bg-gray-800 rounded-lg p-1">
          {practices.map((p, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedCategory(idx)}
              className={`flex items-center gap-1 px-3 py-1.5 rounded text-xs font-medium transition-all ${
                selectedCategory === idx ? 'bg-blue-600' : 'text-gray-400'
              }`}
            >
              <span>{p.icon}</span>
              <span>{p.category}</span>
            </button>
          ))}
        </div>
        <div className="ml-auto text-xs text-gray-400">
          进度: <span className="text-white font-bold">{completedItems}/{totalItems}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-4">
            {practices[selectedCategory].icon} {practices[selectedCategory].category}
          </h3>
          <div className="space-y-3">
            {practices[selectedCategory].items.map((item, idx) => {
              const key = `${selectedCategory}-${idx}`;
              const isChecked = checkedItems[key] || false;

              return (
                <div
                  key={idx}
                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                    isChecked ? 'bg-green-900/30 border border-green-700' : 'bg-gray-700 hover:bg-gray-650'
                  }`}
                  onClick={() => toggleItem(key)}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                      isChecked ? 'bg-green-500 border-green-500' : 'border-gray-500'
                    }`}>
                      {isChecked && <span className="text-white text-xs">✓</span>}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-white">{item.name}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                          item.priority === '高' ? 'bg-red-900/50 text-red-400' : 'bg-yellow-900/50 text-yellow-400'
                        }`}>
                          {item.priority}
                        </span>
                      </div>
                      <div className="text-xs text-gray-400 mt-1">{item.desc}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">完成进度</h3>
            <div className="relative h-4 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 bg-green-500 transition-all"
                style={{ width: `${(completedItems / totalItems) * 100}%` }}
              />
            </div>
            <div className="text-center text-xs text-gray-400 mt-2">
              {Math.round((completedItems / totalItems) * 100)}% 完成
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">各类别进度</h3>
            <div className="space-y-3">
              {practices.map((p, idx) => {
                const catCompleted = p.items.filter((_, i) => checkedItems[`${idx}-${i}`]).length;
                return (
                  <div key={idx}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-400">{p.icon} {p.category}</span>
                      <span className="text-gray-300">{catCompleted}/{p.items.length}</span>
                    </div>
                    <div className="w-full h-1.5 bg-gray-700 rounded-full">
                      <div
                        className="h-full bg-blue-500 rounded-full"
                        style={{ width: `${(catCompleted / p.items.length) * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-2">快速提示</h3>
            <div className="text-xs text-gray-400 space-y-1">
              <p>• 优先处理标记为"高"优先级的项目</p>
              <p>• 内存优化通常带来最大性能提升</p>
              <p>• 使用 TileLang 自动调度简化实现</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
