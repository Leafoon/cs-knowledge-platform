"use client";
import { useState, useMemo } from "react";

export function RTTEstimatorVisualizer() {
  const [samples, setSamples] = useState<number[]>([100, 120, 90, 150, 110, 200, 95, 105]);
  const [alpha, setAlpha] = useState(0.125);
  const [beta, setBeta] = useState(0.25);

  const computed = useMemo(() => {
    const result: { sample: number; srtt: number; rttvar: number; rto: number }[] = [];
    let srtt = samples[0];
    let rttvar = samples[0] / 2;
    result.push({ sample: samples[0], srtt, rttvar, rto: srtt + 4 * rttvar });
    for (let i = 1; i < samples.length; i++) {
      const r = samples[i];
      rttvar = (1 - beta) * rttvar + beta * Math.abs(srtt - r);
      srtt = (1 - alpha) * srtt + alpha * r;
      result.push({ sample: r, srtt, rttvar, rto: srtt + 4 * rttvar });
    }
    return result;
  }, [samples, alpha, beta]);

  const maxVal = Math.max(...samples, ...computed.map((c) => c.rto));

  const addSample = () => setSamples((s) => [...s, Math.round(80 + Math.random() * 120)]);
  const resetSamples = () => setSamples([100, 120, 90, 150, 110, 200, 95, 105]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">RTT估计器可视化</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          α (SRTT平滑因子): <span className="text-text-primary font-mono">{alpha}</span>
          <input type="range" min={0.01} max={0.5} step={0.01} value={alpha} onChange={(e) => setAlpha(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          β (RTTVAR平滑因子): <span className="text-text-primary font-mono">{beta}</span>
          <input type="range" min={0.01} max={0.5} step={0.01} value={beta} onChange={(e) => setBeta(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={addSample} className="px-3 py-1.5 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-xs font-medium hover:bg-sky-500/25 transition-colors">添加采样</button>
        <button onClick={resetSamples} className="px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:text-text-primary transition-colors">重置</button>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="flex items-end gap-1" style={{ height: 120 }}>
          {computed.map((c, i) => (
            <div key={i} className="flex-1 flex flex-col items-center justify-end gap-0.5">
              <div className="w-full bg-red-400/30 rounded-t transition-all" style={{ height: `${(c.sample / maxVal) * 100}px` }} />
              <div className="w-full bg-sky-500 rounded-t transition-all" style={{ height: `${(c.srtt / maxVal) * 100}px` }} />
              <div className="w-full bg-amber-400 rounded-t transition-all" style={{ height: `${(c.rto / maxVal) * 100}px` }} />
              <span className="text-[8px] font-mono text-text-tertiary">{i}</span>
            </div>
          ))}
        </div>
        <div className="flex items-center justify-center gap-4 mt-2 text-[10px]">
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-red-400/30 rounded" /> 采样RTT</span>
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-sky-500 rounded" /> SRTT</span>
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-amber-400 rounded" /> RTO</span>
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs font-mono space-y-1 overflow-x-auto">
        {computed.map((c, i) => (
          <div key={i} className="flex gap-3 text-text-secondary">
            <span className="text-text-tertiary w-4">#{i}</span>
            <span>RTT={c.sample}</span>
            <span className="text-sky-500">SRTT={c.srtt.toFixed(1)}</span>
            <span className="text-violet-500">RTTVAR={c.rttvar.toFixed(1)}</span>
            <span className="text-amber-500">RTO={c.rto.toFixed(1)}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 text-[10px] text-text-tertiary mb-3">RFC 6298: SRTT = (1-α)·SRTT + α·R, RTTVAR = (1-β)·RTTVAR + β·|SRTT-R|, RTO = SRTT + 4·RTTVAR</div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">参数含义</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• α=1/8: SRTT 平滑因子，新样本权重</li>
            <li>• β=1/4: RTTVAR 平滑因子，偏差权重</li>
            <li>• RTO 下限: 至少 1 秒 (RFC 2988)</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">Karn 算法</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 重传段的 ACK 不用于 RTT 采样</li>
            <li>• 避免重传歧义导致估算失准</li>
            <li>• RTO 每次超时后翻倍 (指数退避)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default RTTEstimatorVisualizer;
