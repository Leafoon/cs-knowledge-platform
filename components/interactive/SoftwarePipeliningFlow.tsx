'use client';

import React, { useState, useEffect } from 'react';

interface PipelineStage {
  name: string;
  color: string;
  duration: number;
  description: string;
}

export function SoftwarePipeliningFlow() {
  const [currentCycle, setCurrentCycle] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);

  const stages: PipelineStage[] = [
    { name: 'Prologue', color: '#3B82F6', duration: 2, description: '预取第一块数据' },
    { name: 'Main Loop', color: '#10B981', duration: 4, description: '计算与加载重叠' },
    { name: 'Epilogue', color: '#F59E0B', duration: 2, description: '写回最终结果' },
  ];

  const totalCycles = 12;

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
      }, speed);
      return () => clearInterval(timer);
    }
  }, [isPlaying, speed]);

  const getCycleActivity = (cycle: number) => {
    if (cycle < 2) return { load: true, compute: false, store: false, stage: 'Prologue' };
    if (cycle < 10) return { load: true, compute: true, store: false, stage: 'Main Loop' };
    return { load: false, compute: false, store: true, stage: 'Epilogue' };
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">软件流水线流程</h2>
      
      {/* Timeline */}
      <div className="bg-gray-800 rounded-lg p-4 mb-6">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-gray-400 text-sm">Cycle:</span>
          <span className="text-white font-bold">{currentCycle}</span>
          <span className="text-gray-400 text-sm">/ {totalCycles}</span>
        </div>
        
        {/* Timeline grid */}
        <div className="overflow-x-auto">
          <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${totalCycles}, 1fr)` }}>
            {Array.from({ length: totalCycles }).map((_, cycle) => {
              const activity = getCycleActivity(cycle);
              return (
                <div
                  key={cycle}
                  className={`w-12 h-16 flex flex-col items-center justify-center rounded text-xs transition-all ${
                    cycle === currentCycle ? 'ring-2 ring-white' : ''
                  }`}
                  style={{
                    backgroundColor:
                      cycle === currentCycle
                        ? activity.load
                          ? '#3B82F6'
                          : activity.compute
                          ? '#10B981'
                          : '#F59E0B'
                        : '#374151',
                  }}
                >
                  <span className="text-white font-bold">{cycle}</span>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Activity legend */}
        <div className="mt-4 flex gap-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-blue-500" />
            <span className="text-gray-300 text-sm">Load</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-green-500" />
            <span className="text-gray-300 text-sm">Compute</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-yellow-500" />
            <span className="text-gray-300 text-sm">Store</span>
          </div>
        </div>
      </div>
      
      {/* Pipeline stages */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {stages.map((stage, i) => (
          <div
            key={i}
            className="p-4 rounded-lg border-2"
            style={{ borderColor: stage.color }}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded" style={{ backgroundColor: stage.color }} />
              <span className="text-white font-bold">{stage.name}</span>
            </div>
            <p className="text-gray-400 text-sm">{stage.description}</p>
            <div className="mt-2 text-gray-300 text-sm">
              周期: {stage.duration}
            </div>
          </div>
        ))}
      </div>
      
      {/* Current activity */}
      {currentCycle > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-bold mb-2">当前周期活动</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className={`p-3 rounded-lg ${getCycleActivity(currentCycle).load ? 'bg-blue-900/50' : 'bg-gray-700'}`}>
              <span className="text-gray-400 text-sm">Load</span>
              <div className="text-white font-bold">{getCycleActivity(currentCycle).load ? '✓' : '✗'}</div>
            </div>
            <div className={`p-3 rounded-lg ${getCycleActivity(currentCycle).compute ? 'bg-green-900/50' : 'bg-gray-700'}`}>
              <span className="text-gray-400 text-sm">Compute</span>
              <div className="text-white font-bold">{getCycleActivity(currentCycle).compute ? '✓' : '✗'}</div>
            </div>
            <div className={`p-3 rounded-lg ${getCycleActivity(currentCycle).store ? 'bg-yellow-900/50' : 'bg-gray-700'}`}>
              <span className="text-gray-400 text-sm">Store</span>
              <div className="text-white font-bold">{getCycleActivity(currentCycle).store ? '✓' : '✗'}</div>
            </div>
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
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="px-4 py-2 bg-gray-700 rounded-lg text-white"
        >
          <option value={800}>慢速</option>
          <option value={500}>正常</option>
          <option value={200}>快速</option>
        </select>
      </div>
    </div>
  );
}
