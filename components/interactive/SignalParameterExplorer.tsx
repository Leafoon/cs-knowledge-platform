"use client";
import { useState, useMemo } from "react";

export function SignalParameterExplorer() {
  const [frequency, setFrequency] = useState(2);
  const [amplitude, setAmplitude] = useState(1);
  const [phase, setPhase] = useState(0);
  const [samples] = useState(200);

  const waveform = useMemo(() => {
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i < samples; i++) {
      const t = (i / samples) * 4 * Math.PI;
      const y = amplitude * Math.sin(frequency * t + (phase * Math.PI) / 180);
      points.push({ x: i, y });
    }
    return points;
  }, [frequency, amplitude, phase, samples]);

  const maxY = amplitude * 1.2;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">信号参数探索器</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          频率 (f): <span className="text-text-primary font-mono">{frequency} Hz</span>
          <input type="range" min={0.5} max={10} step={0.5} value={frequency} onChange={(e) => setFrequency(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          振幅 (A): <span className="text-text-primary font-mono">{amplitude}</span>
          <input type="range" min={0.1} max={3} step={0.1} value={amplitude} onChange={(e) => setAmplitude(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          相位 (φ): <span className="text-text-primary font-mono">{phase}°</span>
          <input type="range" min={0} max={360} value={phase} onChange={(e) => setPhase(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <svg viewBox={`0 0 ${samples} ${maxY * 2 + 20}`} className="w-full h-32" preserveAspectRatio="none">
          <line x1="0" y1={maxY + 10} x2={samples} y2={maxY + 10} stroke="currentColor" className="text-border-subtle" strokeWidth="0.5" />
          <line x1="0" y1={10} x2="0" y2={maxY * 2 + 10} stroke="currentColor" className="text-border-subtle" strokeWidth="0.5" />
          <polyline points={waveform.map((p) => `${p.x},${maxY + 10 - p.y}`).join(" ")} fill="none" stroke="#38bdf8" strokeWidth="2" />
        </svg>
        <div className="flex justify-between text-[10px] text-text-tertiary mt-1">
          <span>0</span>
          <span>2π</span>
          <span>4π</span>
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-medium text-text-primary">信号公式</div>
        <div className="font-mono">y(t) = {amplitude} × sin({frequency}t + {(phase * Math.PI / 180).toFixed(2)})</div>
        <div>周期 T = {(1 / frequency).toFixed(3)}s，角频率 ω = {(2 * Math.PI * frequency).toFixed(2)} rad/s</div>
        <div>相位偏移 = {phase}° = {(phase * Math.PI / 180).toFixed(2)} rad</div>
      </div>
      <div className="mt-3 rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1 mb-3">
        <div className="font-medium text-text-primary">物理层应用</div>
        <div>• 频率决定信号的振荡速度，影响带宽占用</div>
        <div>• 振幅决定信号强度，影响信噪比（SNR）</div>
        <div>• 相位用于相移键控（PSK）调制，如QPSK、8PSK</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">调制技术</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• ASK: 振幅键控，调振幅</li>
            <li>• FSK: 频移键控，调频率</li>
            <li>• PSK: 相移键控，调相位</li>
            <li>• QAM: 振幅+相位联合调制</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">傅里叶分析</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 任何周期信号 = 正弦波之和</li>
            <li>• 基频 f + 谐波 2f, 3f, ...</li>
            <li>• 带宽 = 最高谐波频率 - 基频</li>
            <li>• 方波需要无穷谐波</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default SignalParameterExplorer;
