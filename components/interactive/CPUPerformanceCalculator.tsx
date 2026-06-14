"use client";

import { useState, useMemo } from "react";

export function CPUPerformanceCalculator() {
  const [freq, setFreq] = useState("2.4");
  const [freqUnit, setFreqUnit] = useState<"GHz" | "MHz">("GHz");
  const [cpi, setCpi] = useState("1.2");
  const [instCount, setInstCount] = useState("1000000");

  const f = parseFloat(freq) || 0;
  const c = parseFloat(cpi) || 0;
  const ic = parseFloat(instCount) || 0;
  const freqHz = freqUnit === "GHz" ? f * 1e9 : f * 1e6;

  const results = useMemo(() => {
    if (freqHz <= 0 || c <= 0 || ic <= 0) return null;
    const execTime = (ic * c) / freqHz;
    const mips = freqHz / (c * 1e6);
    const flops = freqHz / c;
    return { execTime, mips, flops };
  }, [freqHz, c, ic]);

  function formatTime(seconds: number): string {
    if (seconds < 1e-6) return `${(seconds * 1e9).toFixed(2)} ns`;
    if (seconds < 1e-3) return `${(seconds * 1e6).toFixed(2)} μs`;
    if (seconds < 1) return `${(seconds * 1e3).toFixed(2)} ms`;
    if (seconds < 60) return `${seconds.toFixed(4)} s`;
    return `${(seconds / 60).toFixed(2)} min`;
  }

  function formatFLOPS(fps: number): string {
    if (fps >= 1e12) return `${(fps / 1e12).toFixed(2)} TFLOPS`;
    if (fps >= 1e9) return `${(fps / 1e9).toFixed(2)} GFLOPS`;
    if (fps >= 1e6) return `${(fps / 1e6).toFixed(2)} MFLOPS`;
    return `${fps.toFixed(0)} FLOPS`;
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        CPU 性能计算器
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        输入时钟频率、CPI 和指令数，计算执行时间、MIPS 和 FLOPS
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        {/* frequency */}
        <div className="rounded-lg border border-border-subtle bg-bg-secondary p-4">
          <label className="block text-xs font-medium text-text-secondary mb-2">
            时钟频率 (f)
          </label>
          <div className="flex gap-2">
            <input
              type="number"
              value={freq}
              onChange={(e) => setFreq(e.target.value)}
              step="0.1"
              min="0"
              className="flex-1 px-3 py-2 rounded border border-border-subtle bg-bg-elevated text-text-primary font-mono text-sm"
            />
            <select
              value={freqUnit}
              onChange={(e) => setFreqUnit(e.target.value as "GHz" | "MHz")}
              className="px-2 py-2 rounded border border-border-subtle bg-bg-elevated text-text-primary text-sm"
            >
              <option value="GHz">GHz</option>
              <option value="MHz">MHz</option>
            </select>
          </div>
          <p className="text-xs text-text-secondary mt-1 font-mono">{freqHz.toExponential(2)} Hz</p>
        </div>

        {/* CPI */}
        <div className="rounded-lg border border-border-subtle bg-bg-secondary p-4">
          <label className="block text-xs font-medium text-text-secondary mb-2">
            CPI (每条指令时钟周期数)
          </label>
          <input
            type="number"
            value={cpi}
            onChange={(e) => setCpi(e.target.value)}
            step="0.1"
            min="0"
            className="w-full px-3 py-2 rounded border border-border-subtle bg-bg-elevated text-text-primary font-mono text-sm"
          />
          <p className="text-xs text-text-secondary mt-1">CPI 越小，每条指令执行越快</p>
        </div>

        {/* instruction count */}
        <div className="rounded-lg border border-border-subtle bg-bg-secondary p-4">
          <label className="block text-xs font-medium text-text-secondary mb-2">
            指令条数 (IC)
          </label>
          <input
            type="number"
            value={instCount}
            onChange={(e) => setInstCount(e.target.value)}
            min="0"
            className="w-full px-3 py-2 rounded border border-border-subtle bg-bg-elevated text-text-primary font-mono text-sm"
          />
          <p className="text-xs text-text-secondary mt-1">{ic.toExponential(2)} 条</p>
        </div>
      </div>

      {/* formula */}
      <div className="rounded-lg bg-bg-secondary border border-border-subtle p-3 mb-4">
        <p className="text-xs text-text-secondary">
          <span className="font-semibold text-text-primary">核心公式：</span>
          T = IC × CPI / f &nbsp;|&nbsp; MIPS = f / (CPI × 10⁶) &nbsp;|&nbsp; FLOPS ≈ f / CPI
        </p>
      </div>

      {/* results */}
      {results && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {[
            { label: "执行时间 T", value: formatTime(results.execTime), formula: `${ic} × ${c} / ${freqHz.toExponential(2)}` },
            { label: "MIPS", value: results.mips.toFixed(2), formula: `${freqHz.toExponential(2)} / (${c} × 10⁶)` },
            { label: "FLOPS", value: formatFLOPS(results.flops), formula: `${freqHz.toExponential(2)} / ${c}` },
          ].map((r) => (
            <div key={r.label} className="rounded-lg border border-border-subtle bg-bg-secondary p-4 text-center">
              <p className="text-xs text-text-secondary mb-1">{r.label}</p>
              <p className="font-mono font-bold text-lg text-text-primary">{r.value}</p>
              <p className="text-[10px] text-text-secondary mt-1 font-mono">{r.formula}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
