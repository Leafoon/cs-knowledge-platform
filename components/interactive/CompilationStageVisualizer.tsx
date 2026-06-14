'use client';
import { useState, useEffect } from 'react';

interface Stage {
  name: string;
  time: number;
  color: string;
  icon: string;
  description: string;
}

const stages: Stage[] = [
  { name: '词法分析', time: 2.1, color: 'bg-blue-500', icon: '📝', description: '将源代码字符流转换为Token序列' },
  { name: '语法分析', time: 4.3, color: 'bg-purple-500', icon: '🌲', description: '构建抽象语法树(AST)' },
  { name: '语义分析', time: 3.8, color: 'bg-indigo-500', icon: '🔍', description: '类型检查和作用域分析' },
  { name: 'IR生成', time: 5.2, color: 'bg-cyan-500', icon: '⚙️', description: '将AST转换为中间表示' },
  { name: '优化', time: 12.5, color: 'bg-green-500', icon: '🚀', description: '执行各种编译优化Pass' },
  { name: '代码生成', time: 6.7, color: 'bg-orange-500', icon: '📦', description: '生成目标平台的机器码' },
];

export function CompilationStageVisualizer() {
  const [activeStage, setActiveStage] = useState<number | null>(null);
  const [animating, setAnimating] = useState<number | null>(null);
  const [completedStages, setCompletedStages] = useState<number[]>([]);

  const totalTime = stages.reduce((s, st) => s + st.time, 0);

  const startAnimation = () => {
    setCompletedStages([]);
    setAnimating(null);
    let i = 0;
    const timer = setInterval(() => {
      if (i >= stages.length) {
        clearInterval(timer);
        setAnimating(null);
        return;
      }
      setAnimating(i);
      setTimeout(() => {
        setCompletedStages(prev => [...prev, i]);
      }, stages[i].time * 80);
      i++;
    }, totalTime * 80 / stages.length + 200);
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-cyan-400">编译阶段可视化</h2>
        <button onClick={startAnimation} className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-sm font-medium transition-colors">
          ▶ 播放流水线
        </button>
      </div>

      <div className="flex items-center gap-1 mb-6">
        {stages.map((stage, i) => {
          const widthPercent = (stage.time / totalTime) * 100;
          return (
            <div key={i} className="relative group" style={{ width: `${widthPercent}%` }}>
              <div
                className={`h-12 rounded-md flex items-center justify-center text-xs font-medium cursor-pointer transition-all duration-500 ${
                  animating === i ? 'ring-2 ring-white ring-offset-2 ring-offset-gray-900 scale-105' :
                  completedStages.includes(i) ? stage.color : 'bg-gray-700'
                }`}
                onClick={() => setActiveStage(activeStage === i ? null : i)}
              >
                <span className="truncate px-1">{stage.icon}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="space-y-2">
        {stages.map((stage, i) => {
          const percent = (stage.time / totalTime) * 100;
          return (
            <div key={i}
              className={`flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors ${
                activeStage === i ? 'bg-gray-800' : 'hover:bg-gray-800/50'
              }`}
              onClick={() => setActiveStage(activeStage === i ? null : i)}
            >
              <div className={`w-3 h-3 rounded-full ${stage.color} ${animating === i ? 'animate-pulse' : ''}`} />
              <span className="w-20 text-sm text-gray-300">{stage.name}</span>
              <div className="flex-1 bg-gray-800 rounded-full h-3 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${
                    completedStages.includes(i) ? stage.color : 'bg-gray-700'
                  }`}
                  style={{ width: `${percent}%` }}
                />
              </div>
              <span className="w-16 text-right text-sm font-mono text-gray-400">{stage.time}ms</span>
              <span className="w-12 text-right text-xs text-gray-500">{percent.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>

      {activeStage !== null && (
        <div className="mt-4 p-4 bg-gray-800 rounded-lg">
          <div className="font-semibold text-cyan-300">{stages[activeStage].name}</div>
          <div className="text-sm text-gray-400 mt-1">{stages[activeStage].description}</div>
          <div className="text-xs text-gray-500 mt-2">耗时: {stages[activeStage].time}ms ({((stages[activeStage].time / totalTime) * 100).toFixed(1)}%)</div>
        </div>
      )}

      <div className="mt-4 text-xs text-gray-500">总编译时间: {totalTime.toFixed(1)}ms</div>
    </div>
  );
}
