"use client";
import { useState, useMemo } from "react";

export function QUICCongestionDemo() {
  const [lossRate, setLossRate] = useState(3);
  const [rtt, setRtt] = useState(40);
  const [duration, setDuration] = useState(200);

  const data = useMemo(() => {
    const points: { t: number; tcp: number; quic: number }[] = [];
    let tcpCwnd = 1;
    let quicCwnd = 1;
    let tcpSsthresh = 64;
    let quicSsthresh = 64;
    for (let t = 0; t < duration; t++) {
      const loss = Math.random() * 100 < lossRate;
      if (tcpCwnd < tcpSsthresh) {
        tcpCwnd = Math.min(tcpCwnd * 2, 128);
      } else {
        tcpCwnd = Math.min(tcpCwnd + 1 / tcpCwnd, 128);
      }
      if (loss) {
        tcpSsthresh = Math.max(tcpCwnd / 2, 2);
        tcpCwnd = tcpSsthresh;
      }
      if (quicCwnd < quicSsthresh) {
        quicCwnd = Math.min(quicCwnd * 2, 128);
      } else {
        quicCwnd = Math.min(quicCwnd + 1 / quicCwnd, 128);
      }
      if (loss) {
        quicSsthresh = Math.max(quicCwnd * 0.85, 2);
        quicCwnd = quicSsthresh * 0.8;
      }
      points.push({ t, tcp: tcpCwnd, quic: quicCwnd });
    }
    return points;
  }, [lossRate, rtt, duration]);

  const maxCwnd = Math.max(...data.map((d) => Math.max(d.tcp, d.quic)), 1);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">QUIC拥塞控制与TCP对比</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          丢包率: <span className="text-text-primary font-mono">{lossRate}%</span>
          <input type="range" min={0} max={15} value={lossRate} onChange={(e) => setLossRate(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          RTT: <span className="text-text-primary font-mono">{rtt}ms</span>
          <input type="range" min={10} max={200} value={rtt} onChange={(e) => setRtt(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          时长: <span className="text-text-primary font-mono">{duration}RTT</span>
          <input type="range" min={50} max={500} value={duration} onChange={(e) => setDuration(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <svg viewBox={`0 0 ${duration} ${maxCwnd + 10}`} className="w-full h-40">
          <polyline points={data.map((d) => `${d.t},${maxCwnd - d.tcp + 5}`).join(" ")} fill="none" stroke="#f87171" strokeWidth="1.5" opacity="0.8" />
          <polyline points={data.map((d) => `${d.t},${maxCwnd - d.quic + 5}`).join(" ")} fill="none" stroke="#38bdf8" strokeWidth="1.5" opacity="0.8" />
        </svg>
        <div className="flex items-center justify-center gap-4 mt-2 text-xs">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-red-400 inline-block" /> TCP Reno</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-sky-400 inline-block" /> QUIC</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs">
          <div className="font-medium text-red-500 mb-1">TCP Reno</div>
          <div className="text-text-secondary">慢启动 → 拥塞避免</div>
          <div className="text-text-secondary">丢包: cwnd减半，ssthresh = cwnd/2</div>
          <div className="text-text-secondary">需要一个RTT检测丢包</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs">
          <div className="font-medium text-sky-500 mb-1">QUIC (Cubic+)</div>
          <div className="text-text-secondary">拥塞窗口恢复更快</div>
          <div className="text-text-secondary">丢包: cwnd减少更温和（~15%）</div>
          <div className="text-text-secondary">0-RTT恢复，多路复用无队头阻塞</div>
        </div>
      </div>
    </div>
  );
}
export default QUICCongestionDemo;
