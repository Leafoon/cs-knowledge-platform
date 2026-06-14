'use client';
import { useState, useEffect } from 'react';

const WARP_SIZE = 8;

export function WarpReduceDemo() {
  const [step, setStep] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [values, setValues] = useState<number[]>(() => Array.from({ length: WARP_SIZE }, () => Math.floor(Math.random() * 20) + 1));

  const maxStep = Math.ceil(Math.log2(WARP_SIZE));

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setTimeout(() => {
      if (step >= maxStep) { setIsPlaying(false); return; }
      setStep(s => s + 1);
    }, 1200);
    return () => clearTimeout(timer);
  }, [isPlaying, step, maxStep]);

  const playAnimation = () => { setStep(0); setIsPlaying(true); };

  const getValue = (lane: number, currentStep: number) => {
    let v = values[lane];
    for (let s = 0; s <= Math.min(currentStep, maxStep); s++) {
      const delta = 1 << s;
      if (lane >= delta) {
        const src = lane - delta;
        if (src >= 0 && step >= s + 1) {
          v = values[lane] + values[src];
        }
      }
    }
    return v;
  };

  const getDisplayValue = (lane: number) => {
    if (step < 0) return values[lane];
    for (let s = maxStep; s >= 0; s--) {
      const delta = 1 << s;
      if (lane >= delta && step > s) {
        return values[lane] + values[lane - delta];
      }
    }
    return values[lane];
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-cyan-400">Warp Reduce 蝶形归约</h2>
        <div className="flex gap-2">
          <button onClick={() => { setValues(Array.from({ length: WARP_SIZE }, () => Math.floor(Math.random() * 20) + 1)); setStep(-1); }}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm">🔄 重置</button>
          <button onClick={playAnimation} disabled={isPlaying}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-sm disabled:opacity-50">
            ▶ 播放归约
          </button>
        </div>
      </div>

      <div className="text-sm text-gray-400 mb-4">
        步骤: {step < 0 ? '准备中' : `${step}/${maxStep}`} · 最终结果: {getDisplayValue(0)}
      </div>

      {/* Thread lanes */}
      <div className="space-y-3">
        {Array.from({ length: WARP_SIZE }).map((_, lane) => (
          <div key={lane} className="flex items-center gap-3">
            <span className="w-16 text-xs text-gray-400 font-mono">Lane {lane}</span>
            <div className="flex-1 relative">
              <div className="h-10 bg-gray-800 rounded-lg flex items-center px-4 relative overflow-hidden">
                {/* Value bar background */}
                <div className="absolute inset-0 bg-gradient-to-r from-blue-900/30 to-transparent"
                  style={{ width: `${(getDisplayValue(lane) / (Math.max(...values) * WARP_SIZE)) * 100}%` }} />

                {/* Connections */}
                {Array.from({ length: maxStep }).map((_, s) => {
                  const delta = 1 << s;
                  if (lane >= delta && step > s) {
                    return (
                      <div key={s} className="absolute -left-4 top-0 h-full">
                        <div className="w-8 h-full flex items-center">
                          <div className={`w-full h-px ${step > s ? 'bg-yellow-500' : 'bg-gray-600'}`} />
                        </div>
                      </div>
                    );
                  }
                  return null;
                })}

                <span className="relative z-10 font-mono font-bold text-lg">
                  {getDisplayValue(lane)}
                </span>

                {step > 0 && lane > 0 && step > Math.floor(Math.log2(lane)) && (
                  <span className="relative z-10 ml-2 text-xs text-yellow-400">
                    ← +{values[Math.max(0, lane - (1 << Math.floor(Math.log2(lane))))]}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Step explanation */}
      <div className="mt-6 space-y-2">
        <div className="text-sm font-medium text-gray-400">蝶形归约步骤说明:</div>
        {Array.from({ length: maxStep + 1 }).map((_, s) => (
          <div key={s} className={`flex items-center gap-2 text-xs p-2 rounded ${
            step === s ? 'bg-cyan-900/30 text-cyan-300' : step > s ? 'text-green-400' : 'text-gray-500'
          }`}>
            <span className="w-6 font-mono">{s}:</span>
            <span>stride = {1 << s}，每个线程从左侧 {1 << s} 个位置的线程获取数据并累加</span>
          </div>
        ))}
      </div>
    </div>
  );
}
