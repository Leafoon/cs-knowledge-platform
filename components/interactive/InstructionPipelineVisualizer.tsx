'use client';

import { useState } from 'react';

const stages = ['取指', '译码', '执行', '访存', '写回'];
const stageColors = ['bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500', 'bg-red-500'];

const instructions = ['I1', 'I2', 'I3', 'I4', 'I5'];

export function InstructionPipelineVisualizer() {
  const [cycle, setCycle] = useState(0);
  const maxCycle = stages.length + instructions.length - 2;

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">指令流水线可视化</h2>
      
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setCycle(Math.max(0, cycle - 1))}
          disabled={cycle === 0}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:bg-gray-300"
        >
          ← 上一周期
        </button>
        <button
          onClick={() => setCycle(Math.min(maxCycle, cycle + 1))}
          disabled={cycle === maxCycle}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:bg-gray-300"
        >
          下一周期 →
        </button>
      </div>

      <div className="mb-4 text-center">
        <span className="px-3 py-1 bg-gray-100 rounded-full text-sm">
          时钟周期: {cycle + 1} / {maxCycle + 1}
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[500px]">
          <div className="grid gap-1" style={{ gridTemplateColumns: `60px repeat(${stages.length}, 1fr)` }}>
            <div></div>
            {stages.map((stage, i) => (
              <div key={stage} className={`text-center text-xs text-white p-1 rounded ${stageColors[i]}`}>
                {stage}
              </div>
            ))}
            
            {instructions.map((inst, row) => (
              <div key={inst} className="contents">
                <div className="text-xs text-gray-500 p-1">{inst}</div>
                {stages.map((_, col) => {
                  const startCycle = row;
                  const isActive = cycle >= startCycle && cycle < startCycle + stages.length;
                  const stageIndex = cycle - startCycle;
                  return (
                    <div
                      key={col}
                      className={`h-8 rounded transition-all ${
                        isActive && stageIndex === col
                          ? `${stageColors[col]} opacity-100`
                          : isActive
                          ? 'bg-gray-300'
                          : 'bg-gray-100'
                      }`}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}