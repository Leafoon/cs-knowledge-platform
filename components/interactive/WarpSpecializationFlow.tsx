'use client';
import { useState } from 'react';

const warpStates = ['空闲', '加载数据', '计算', '存储结果', '等待同步'];

interface Warp {
  id: number;
  role: 'producer' | 'consumer';
  state: number;
  progress: number;
}

export function WarpSpecializationFlow() {
  const [warps, setWarps] = useState<Warp[]>([
    { id: 0, role: 'producer', state: 0, progress: 0 },
    { id: 1, role: 'producer', state: 0, progress: 0 },
    { id: 2, role: 'consumer', state: 0, progress: 0 },
    { id: 3, role: 'consumer', state: 0, progress: 0 },
  ]);
  const [cycle, setCycle] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const advanceCycle = () => {
    setCycle(c => c + 1);
    setWarps(prev => prev.map(w => {
      if (w.role === 'producer') {
        const newProgress = (w.progress + 25) % 100;
        const newState = newProgress === 0 ? (w.state + 1) % 5 : w.state;
        return { ...w, progress: newProgress, state: newState };
      } else {
        const delay = w.id === 2 ? 12 : 0;
        const adjusted = Math.max(0, w.progress + (cycle > delay ? 20 : 0));
        const newProgress = adjusted % 100;
        const newState = adjusted >= 100 ? (w.state + 1) % 5 : w.state;
        return { ...w, progress: Math.min(newProgress, 100), state: newState };
      }
    }));
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-cyan-400">Warp专用化模式</h2>
        <div className="flex gap-2">
          <button onClick={() => { setCycle(0); setWarps(prev => prev.map(w => ({ ...w, state: 0, progress: 0 }))); }}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm">重置</button>
          <button onClick={advanceCycle}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-sm">⚡ 推进周期</button>
        </div>
      </div>

      <div className="text-sm text-gray-400 mb-4">周期: {cycle}</div>

      {/* Pipeline diagram */}
      <div className="grid grid-cols-2 gap-6">
        {/* Producer Warps */}
        <div>
          <div className="text-sm font-medium text-orange-400 mb-3 flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-orange-500" />Producer Warps (数据搬运)
          </div>
          {warps.filter(w => w.role === 'producer').map(w => (
            <div key={w.id} className="mb-3 p-3 bg-gray-800 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-mono">Warp {w.id}</span>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  w.state === 1 ? 'bg-blue-800 text-blue-300' :
                  w.state === 2 ? 'bg-green-800 text-green-300' :
                  w.state === 3 ? 'bg-purple-800 text-purple-300' :
                  w.state === 4 ? 'bg-yellow-800 text-yellow-300' :
                  'bg-gray-700 text-gray-300'
                }`}>{warpStates[w.state]}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                <div className="h-full bg-orange-500 rounded-full transition-all duration-300"
                  style={{ width: `${w.progress}%` }} />
              </div>
              <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                <span>从HBM加载到Shared Memory</span>
                <span>{w.progress}%</span>
              </div>
            </div>
          ))}
        </div>

        {/* Consumer Warps */}
        <div>
          <div className="text-sm font-medium text-green-400 mb-3 flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-green-500" />Consumer Warps (计算)
          </div>
          {warps.filter(w => w.role === 'consumer').map(w => (
            <div key={w.id} className="mb-3 p-3 bg-gray-800 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-mono">Warp {w.id}</span>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  w.state === 1 ? 'bg-blue-800 text-blue-300' :
                  w.state === 2 ? 'bg-green-800 text-green-300' :
                  w.state === 3 ? 'bg-purple-800 text-purple-300' :
                  w.state === 4 ? 'bg-yellow-800 text-yellow-300' :
                  'bg-gray-700 text-gray-300'
                }`}>{warpStates[w.state]}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                <div className="h-full bg-green-500 rounded-full transition-all duration-300"
                  style={{ width: `${w.progress}%` }} />
              </div>
              <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                <span>从Shared Memory读取并计算</span>
                <span>{w.progress}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Synchronization */}
      <div className="mt-4 border border-gray-700 rounded-lg p-3">
        <div className="text-xs text-gray-400 mb-2">同步屏障 (bar.sync)</div>
        <div className="flex gap-2">
          {warps.map(w => (
            <div key={w.id} className={`flex-1 h-2 rounded ${
              w.progress > 80 ? 'bg-yellow-500' : 'bg-gray-700'
            }`} />
          ))}
        </div>
        <div className="text-[10px] text-gray-500 mt-1">Producer和Consumer通过barrier同步，确保数据就绪后才开始计算</div>
      </div>
    </div>
  );
}
