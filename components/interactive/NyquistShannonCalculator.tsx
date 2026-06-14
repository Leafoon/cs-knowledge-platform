"use client";
import { useState } from "react";

export function NyquistShannonCalculator() {
  const [bandwidth, setBandwidth] = useState(4000);
  const [signalLevels, setSignalLevels] = useState(4);
  const [snr, setSnr] = useState(30);

  const nyquist = 2 * bandwidth * Math.log2(signalLevels);
  const snrLinear = Math.pow(10, snr / 10);
  const shannon = bandwidth * Math.log2(1 + snrLinear);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">📐 Nyquist/Shannon 计算器</h3>
      <p className="text-sm text-text-secondary mb-4">输入参数计算信道最大数据率</p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary">带宽 (B): {bandwidth} Hz</label>
          <input type="range" min={100} max={20000} step={100} value={bandwidth}
            onChange={e => setBandwidth(Number(e.target.value))} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">信号电平数 (L): {signalLevels}</label>
          <input type="range" min={2} max={256} value={signalLevels}
            onChange={e => setSignalLevels(Number(e.target.value))} className="w-full accent-green-500" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">信噪比 (SNR): {snr} dB</label>
          <input type="range" min={0} max={60} value={snr}
            onChange={e => setSnr(Number(e.target.value))} className="w-full accent-yellow-500" />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="bg-bg-surface rounded-lg p-4 border border-border-subtle">
          <div className="text-sm font-semibold text-blue-400 mb-2">Nyquist 定理（无噪声）</div>
          <div className="font-mono text-xs text-text-secondary mb-2">C = 2B × log₂(L)</div>
          <div className="font-mono text-xs text-text-secondary mb-2">C = 2 × {bandwidth} × log₂({signalLevels})</div>
          <div className="text-2xl font-mono font-bold text-blue-400">{nyquist.toFixed(0)} bps</div>
          <div className="text-xs text-text-secondary mt-1">= {(nyquist / 1000).toFixed(1)} kbps</div>
        </div>
        <div className="bg-bg-surface rounded-lg p-4 border border-border-subtle">
          <div className="text-sm font-semibold text-yellow-400 mb-2">Shannon 定理（有噪声）</div>
          <div className="font-mono text-xs text-text-secondary mb-2">C = B × log₂(1 + SNR)</div>
          <div className="font-mono text-xs text-text-secondary mb-2">C = {bandwidth} × log₂(1 + {snrLinear.toFixed(0)})</div>
          <div className="text-2xl font-mono font-bold text-yellow-400">{shannon.toFixed(0)} bps</div>
          <div className="text-xs text-text-secondary mt-1">= {(shannon / 1000).toFixed(1)} kbps</div>
        </div>
      </div>

      <div className="bg-bg-surface rounded-lg p-3 text-xs text-text-secondary">
        <strong className="text-text-primary">理论上限：</strong>
        Nyquist 给出无噪声信道的理论最大速率；Shannon 给出有噪声信道的理论最大速率。实际系统速率取两者中的较小值。
        SNR {snr} dB = 线性值 {snrLinear.toFixed(0)}。
      </div>
    </div>
  );
}
export default NyquistShannonCalculator;
