'use client';
import { useState, useEffect } from 'react';

const LANE_COUNT = 32;

interface ShuffleMode {
  name: string;
  formula: string;
  description: string;
  transform: (id: number) => number;
}

const modes: ShuffleMode[] = [
  { name: 'Exchange', formula: 'shuffle(idx, src, delta)', description: '从指定源线程获取数据', transform: (id) => (id % 2 === 0 ? id + 1 : id - 1) },
  { name: 'Up', formula: 'shuffle_up(val, delta)', description: '向低编号线程获取数据', transform: (id) => Math.max(0, id - 4) },
  { name: 'Down', formula: 'shuffle_down(val, delta)', description: '向高编号线程获取数据', transform: (id) => Math.min(31, id + 4) },
  { name: 'XOR', formula: 'shuffle_xor(val, lane_mask)', description: '与特定掩码异或位置的线程交换', transform: (id) => id ^ 3 },
];

export function WarpShuffleVisualizer() {
  const [modeIdx, setModeIdx] = useState(0);
  const [animStep, setAnimStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedLane, setSelectedLane] = useState<number | null>(null);

  const mode = modes[modeIdx];
  const data = Array.from({ length: LANE_COUNT }, (_, i) => i * 10);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      setAnimStep(prev => {
        if (prev >= 2) { setIsPlaying(false); return 0; }
        return prev + 1;
      });
    }, 800);
    return () => clearInterval(timer);
  }, [isPlaying]);

  const playAnimation = () => { setAnimStep(0); setIsPlaying(true); };

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-cyan-400">Warp Shuffle可视化</h2>
        <button onClick={playAnimation}
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-sm transition-colors">
          ▶ 播放动画
        </button>
      </div>

      <div className="flex gap-2 mb-4">
        {modes.map((m, i) => (
          <button key={i} onClick={() => { setModeIdx(i); setAnimStep(0); }}
            className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
              i === modeIdx ? 'bg-cyan-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}>{m.name}</button>
        ))}
      </div>

      <div className="text-sm text-gray-400 mb-3 font-mono">{mode.formula} — {mode.description}</div>

      {/* Lanes */}
      <div className="relative">
        <div className="grid grid-cols-16 gap-0.5 mb-1">
          {Array.from({ length: LANE_COUNT }).map((_, i) => (
            <div key={i} className="text-center text-[9px] text-gray-500">L{i}</div>
          ))}
        </div>

        {/* Source values */}
        <div className="mb-1">
          <div className="text-xs text-gray-500 mb-1">原始值:</div>
          <div className="grid grid-cols-16 gap-0.5">
            {data.map((v, i) => (
              <div key={i}
                className={`aspect-square rounded flex items-center justify-center text-[10px] font-mono cursor-pointer transition-all ${
                  selectedLane === i ? 'bg-blue-400 text-black' : 'bg-blue-900 text-blue-200'
                }`}
                onClick={() => setSelectedLane(selectedLane === i ? null : i)}>
                {v}
              </div>
            ))}
          </div>
        </div>

        {/* Shuffle arrows */}
        {animStep >= 1 && (
          <div className="my-2">
            <div className="text-xs text-gray-500 mb-1">数据交换:</div>
            <div className="grid grid-cols-16 gap-0.5">
              {Array.from({ length: LANE_COUNT }).map((_, i) => {
                const target = mode.transform(i);
                return (
                  <div key={i} className="aspect-square flex items-center justify-center">
                    <div className={`text-[8px] ${i === selectedLane ? 'text-yellow-400' : 'text-gray-500'}`}>
                      {target !== i ? `→${target}` : '—'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Result values */}
        {animStep >= 2 && (
          <div>
            <div className="text-xs text-gray-500 mb-1">结果值:</div>
            <div className="grid grid-cols-16 gap-0.5">
              {data.map((v, i) => {
                const srcIdx = mode.transform(i);
                return (
                  <div key={i}
                    className={`aspect-square rounded flex items-center justify-center text-[10px] font-mono transition-all ${
                      i === selectedLane ? 'bg-green-400 text-black' : 'bg-green-900 text-green-200'
                    }`}>
                    {data[srcIdx]}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {selectedLane !== null && (
        <div className="mt-4 p-3 bg-gray-800 rounded-lg text-sm text-gray-300">
          <span className="text-blue-400">Lane {selectedLane}</span>: 原始值 {data[selectedLane]}
          → 从 Lane {mode.transform(selectedLane)} 获取值 {data[mode.transform(selectedLane)]}
        </div>
      )}
    </div>
  );
}
