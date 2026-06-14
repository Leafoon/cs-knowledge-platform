"use client";
import { useState } from "react";

const algos = [
  { name: "Tahoe", color: "blue", desc: "慢启动 + 拥塞避免，超时后 cwnd=1", features: ["AIMD (加性增乘性减)", "超时 → cwnd=1", "3 dupACK → cwnd/2", "无快速恢复"] },
  { name: "Reno", color: "green", desc: "加入快速恢复，3 dupACK 不回退到慢启动", features: ["3 dupACK → cwnd/2 + 快速恢复", "超时 → cwnd=1", "快速恢复后进入拥塞避免", "比 Tahoe 更高效"] },
  { name: "CUBIC", color: "yellow", desc: "三次函数增长，Linux 默认算法", features: ["W(t) = C(t-K)³ + W_max", "丢包后三次函数恢复", "高带宽利用率", "RTT 公平性好"] },
  { name: "BBR", color: "red", desc: "基于带宽和 RTT 测量，不依赖丢包信号", features: ["测量 bottleneck BW", "测量 min RTT", "ProbeBW/ProbeRTT 状态机", "减少缓冲区膨胀"] },
];

export function TCPCongestionComparison() {
  const [selected, setSelected] = useState<string[]>(["Reno"]);
  const [step, setStep] = useState(0);
  const maxStep = 40;

  const toggle = (name: string) => {
    setSelected((s) => s.includes(name) ? s.filter((n) => n !== name) : [...s, name]);
  };

  const getCwnd = (algo: string, t: number) => {
    switch (algo) {
      case "Tahoe": return t < 6 ? Math.pow(2, t) : t < 20 ? 64 + (t - 6) * 2 : t === 20 ? 1 : t < 26 ? Math.pow(2, t - 20) : 64 + (t - 26) * 2;
      case "Reno": return t < 6 ? Math.pow(2, t) : t < 20 ? 64 + (t - 6) * 2 : t === 20 ? 32 : t < 26 ? 32 + (t - 20) * 5 : 62 + (t - 26);
      case "CUBIC": return t < 6 ? Math.pow(2, t) : t < 20 ? 64 + (t - 6) * 2 : 32 + Math.pow(t - 20, 3) * 0.1;
      case "BBR": return t < 4 ? Math.pow(2, t) : 10 + Math.sin(t * 0.5) * 3 + (t > 20 ? (t - 20) * 0.3 : 0);
      default: return 1;
    }
  };

  const maxCwnd = 100;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCP 拥塞算法对比 (Congestion Control Comparison)</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {algos.map((a) => (
          <button key={a.name} onClick={() => toggle(a.name)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${selected.includes(a.name) ? `bg-${a.color === "yellow" ? "yellow" : a.color}-500 text-white` : "border border-border-subtle text-text-muted hover:text-text-primary"}`}>
            {a.name}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
          className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">←</button>
        <span className="text-text-secondary text-sm">t={step}</span>
        <button onClick={() => setStep((s) => Math.min(maxStep, s + 1))} disabled={step === maxStep}
          className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">→</button>
        <button onClick={() => setStep(0)} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm">重置</button>
      </div>
      <div className="relative h-48 mb-4 bg-bg-primary border border-border-subtle rounded overflow-hidden">
        <div className="absolute left-0 bottom-0 w-full h-full flex items-end">
          {Array.from({ length: step + 1 }, (_, t) => (
            <div key={t} className="flex-1 flex flex-col items-center justify-end h-full">
              {selected.map((algo) => {
                const c = algos.find((a) => a.name === algo)!;
                const h = (getCwnd(algo, t) / maxCwnd) * 100;
                return <div key={algo} className={`w-1 rounded-t bg-${c.color === "yellow" ? "yellow" : c.color}-500`} style={{ height: `${Math.min(100, h)}%` }} />;
              })}
            </div>
          ))}
        </div>
        {step === 20 && <div className="absolute top-0 left-1/2 w-px h-full bg-red-500/40 border-dashed" />}
        {step >= 20 && <span className="absolute top-1 right-1 text-red-400 text-[10px]">丢包事件</span>}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        {algos.filter((a) => selected.includes(a.name)).map((a) => (
          <div key={a.name} className="p-2 rounded bg-bg-primary border border-border-subtle">
            <h4 className={`text-xs font-medium mb-1 text-${a.color === "yellow" ? "yellow" : a.color}-400`}>{a.name}</h4>
            <p className="text-text-muted text-[10px]">{a.desc}</p>
          </div>
        ))}
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <p className="text-text-muted text-xs">Tahoe/Reno 是经典 AIMD 算法。CUBIC 使用三次函数恢复窗口，是 Linux 默认算法 (2.6.19+)。BBR 不依赖丢包信号，通过测量 BtlBw 和 RTprop 控制发送速率。</p>
      </div>
    </div>
  );
}
export default TCPCongestionComparison;
