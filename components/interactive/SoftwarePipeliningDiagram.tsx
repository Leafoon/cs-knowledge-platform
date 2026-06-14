'use client';

import React, { useState, useEffect } from 'react';

interface PipelineStage {
  name: string;
  color: string;
  tasks: string[];
}

export function SoftwarePipeliningDiagram() {
  const [currentCycle, setCurrentCycle] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const stages: PipelineStage[] = [
    { name: 'Stage 1', color: '#3B82F6', tasks: ['Load A', 'Load B', 'Compute'] },
    { name: 'Stage 2', color: '#10B981', tasks: ['Load A', 'Load B', 'Compute'] },
    { name: 'Stage 3', color: '#F59E0B', tasks: ['Load A', 'Load B', 'Compute'] },
  ];

  const totalCycles = 15;

  useEffect(() => {
    if (isPlaying) {
      const timer = setInterval(() => {
        setCurrentCycle((prev) => {
          if (prev >= totalCycles) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 1;
        });
      }, 400);
      return () => clearInterval(timer);
    }
  }, [isPlaying]);

  const getCycleStage = (cycle: number) => {
    if (cycle < 3) return 0;
    if (cycle < 6) return 1;
    if (cycle < 9) return 2;
    return -1;
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Software Pipelining 时序图</h2>
      
      {/* Timeline */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex items-center gap-4 mb-4">
          <span className="text-white font-bold">Cycle: {currentCycle}</span>
          <div className="flex-1 h-2 bg-gray-700 rounded-full">
            <div
              className="h-full bg-blue-500 rounded-full transition-all"
              style={{ width: `${(currentCycle / totalCycles) * 100}%` }}
            />
          </div>
        </div>
        
        {/* Pipeline visualization */}
        <div className="space-y-4">
          {stages.map((stage, stageIndex) => (
            <div key={stageIndex} className="flex items-center gap-2">
              <span className="w-16 text-gray-400 text-sm">{stage.name}</span>
              
              <div className="flex-1 flex gap-1">
                {Array.from({ length: totalCycles }).map((_, cycle) => {
                  const isActive = cycle >= stageIndex * 3 && cycle < (stageIndex + 1) * 3;
                  const isCurrent = cycle === currentCycle;
                  return (
                    <div
                      key={cycle}
                      className={`h-10 rounded transition-all ${
                        isActive ? 'opacity-100' : 'opacity-20'
                      } ${isCurrent ? 'ring-2 ring-white' : ''}`}
                      style={{
                        backgroundColor: isActive ? stage.color : '#374151',
                        flex: 1,
                      }}
                    />
                  );
                })}
              </div>
            </div>
          ))}
          
          {/* Operation labels */}
          <div className="flex items-center gap-2">
            <span className="w-16 text-gray-400 text-sm">Operations</span>
            <div className="flex-1 flex gap-1">
              {Array.from({ length: totalCycles }).map((_, cycle) => {
                const stageIdx = getCycleStage(cycle);
                const operations = ['Load', 'Compute', 'Store'];
                return (
                  <div
                    key={cycle}
                    className="h-8 rounded flex items-center justify-center text-[10px] text-white"
                    style={{
                      backgroundColor: stageIdx >= 0 ? stages[stageIdx].color : '#374151',
                      flex: 1,
                    }}
                  >
                    {stageIdx >= 0 ? operations[cycle % 3] : ''}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        
        {/* Cycle numbers */}
        <div className="flex items-center gap-2 mt-2">
          <span className="w-16" />
          <div className="flex-1 flex gap-1">
            {Array.from({ length: totalCycles }).map((_, cycle) => (
              <div
                key={cycle}
                className="text-center text-xs text-gray-400"
                style={{ flex: 1 }}
              >
                {cycle}
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Stage details */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {stages.map((stage, i) => (
          <div
            key={i}
            className={`p-4 rounded-lg border-2 transition-all ${
              getCycleStage(currentCycle) === i ? 'scale-105' : ''
            }`}
            style={{ borderColor: stage.color }}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded" style={{ backgroundColor: stage.color }} />
              <span className="text-white font-bold">{stage.name}</span>
            </div>
            <div className="text-gray-300 text-sm">
              {stage.tasks.map((task, j) => (
                <div key={j}>{task}</div>
              ))}
            </div>
          </div>
        ))}
      </div>
      
      {/* Current activity */}
      {currentCycle > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-bold mb-2">当前周期活动</h3>
          <div className="grid grid-cols-3 gap-4">
            {stages.map((stage, i) => {
              const isActive = getCycleStage(currentCycle) === i;
              return (
                <div
                  key={i}
                  className={`p-3 rounded-lg transition-all ${
                    isActive ? 'bg-opacity-50' : 'bg-gray-700'
                  }`}
                  style={{
                    backgroundColor: isActive ? `${stage.color}40` : undefined,
                  }}
                >
                  <div className="text-gray-400 text-sm">{stage.name}</div>
                  <div className="text-white font-bold">{isActive ? 'Active' : 'Idle'}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      
      {/* Controls */}
      <div className="mt-6 flex justify-center gap-4">
        <button
          onClick={() => {
            setCurrentCycle(0);
            setIsPlaying(true);
          }}
          disabled={isPlaying}
          className={`px-6 py-2 rounded-lg font-bold transition-all ${
            isPlaying
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          }`}
        >
          {isPlaying ? '播放中...' : '播放'}
        </button>
        <button
          onClick={() => setCurrentCycle(0)}
          className="px-6 py-2 bg-gray-700 rounded-lg text-white hover:bg-gray-600"
        >
          重置
        </button>
      </div>
    </div>
  );
}
