"use client";
import { useState } from "react";

export function CUBICVisualizer() {
  const [Wmax, setWmax] = useState(100);
  const [C, setC] = useState(0.4);
  const [beta, setBeta] = useState(0.7);
  const [currentT, setCurrentT] = useState(0);

  const Wlast = beta * Wmax;
  const K = Math.pow((Wmax - Wlast) / C, 1 / 3);

  const getCubicW = (t: number) => C * Math.pow(t - K, 3) + Wmax;

  const points: { t: number; w: number }[] = [];
  for (let t = 0; t <= 200; t++) {
    const w = getCubicW(t);
    if (w >= 0 && w <= Wmax * 1.5) points.push({ t, w });
  }

  const maxW = Math.max(...points.map((p) => p.w), Wmax);
  const currentW = getCubicW(currentT);

  const phases = [
    { name: "快速恢复", desc: `从 Wlast=${Wlast.toFixed(1)} 快速恢复到 Wmax`, color: "text-green-600" },
    { name: "稳定探测", desc: `在 Wmax 附近缓慢变化，探测可用带宽`, color: "text-blue-600" },
    { name: "超乘增长", desc: `超过 Wmax 后加速增长`, color: "text-orange-600" },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">CUBIC 三次函数窗口增长</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div>
          <label className="text-xs text-text-secondary">Wmax = {Wmax}</label>
          <input type="range" min={20} max={200} value={Wmax} onChange={(e) => setWmax(+e.target.value)} className="w-full mt-1" />
        </div>
        <div>
          <label className="text-xs text-text-secondary">C = {C.toFixed(2)}</label>
          <input type="range" min={0.1} max={1} step={0.05} value={C} onChange={(e) => setC(+e.target.value)} className="w-full mt-1" />
        </div>
        <div>
          <label className="text-xs text-text-secondary">β = {beta.toFixed(2)}</label>
          <input type="range" min={0.3} max={0.9} step={0.05} value={beta} onChange={(e) => setBeta(+e.target.value)} className="w-full mt-1" />
        </div>
      </div>
      <div className="mb-4 h-56 flex">
        <div className="flex flex-col justify-between text-[10px] text-text-secondary pr-2 w-10">
          <span>{maxW.toFixed(0)}</span>
          <span>{(maxW / 2).toFixed(0)}</span>
          <span>0</span>
        </div>
        <svg className="flex-1" viewBox="0 0 400 200">
          <line x1="0" y1={200 - (Wmax / maxW) * 200} x2="400" y2={200 - (Wmax / maxW) * 200} stroke="#94a3b8" strokeWidth="0.5" strokeDasharray="4" />
          <line x1="0" y1={200 - (Wlast / maxW) * 200} x2="400" y2={200 - (Wlast / maxW) * 200} stroke="#94a3b8" strokeWidth="0.5" strokeDasharray="2" />
          <polyline fill="none" stroke="#3b82f6" strokeWidth="2"
            points={points.map((p) => `${(p.t / 200) * 400},${200 - (p.w / maxW) * 200}`).join(" ")} />
          <circle cx={(currentT / 200) * 400} cy={200 - (Math.max(0, currentW) / maxW) * 200} r="4" fill="#ef4444" />
          <text x="20" y={200 - (Wmax / maxW) * 200 - 5} className="fill-text-secondary text-[10px]">Wmax={Wmax}</text>
          <text x="20" y={200 - (Wlast / maxW) * 200 - 5} className="fill-text-secondary text-[10px]">Wlast={Wlast.toFixed(1)}</text>
        </svg>
      </div>
      <div className="mb-4">
        <label className="text-xs text-text-secondary">时间 t = {currentT} → W(t) = {Math.max(0, currentW).toFixed(2)}</label>
        <input type="range" min={0} max={200} value={currentT} onChange={(e) => setCurrentT(+e.target.value)} className="w-full mt-1" />
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">K (拐点)</div>
          <div className="font-bold text-text-primary">{K.toFixed(2)}</div>
        </div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">Wlast</div>
          <div className="font-bold text-text-primary">{Wlast.toFixed(1)}</div>
        </div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">当前W</div>
          <div className="font-bold text-text-primary">{Math.max(0, currentW).toFixed(1)}</div>
        </div>
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 mb-2">
        <p className="text-xs font-medium text-text-primary mb-1">公式: W(t) = C(t − K)³ + Wmax</p>
        <div className="flex gap-4 text-xs text-text-secondary">
          {phases.map((p, i) => <span key={i} className={p.color}>{p.name}: {p.desc}</span>)}
        </div>
      </div>
      <p className="text-xs text-text-secondary mt-2">CUBIC是Linux默认拥塞控制算法，用三次函数替代AIMD，在高带宽延迟积网络中更高效。</p>
    </div>
  );
}
export default CUBICVisualizer;
