"use client";
import { useState, useMemo } from "react";

export function QUICTCPBenchmark() {
  const [rtt, setRtt] = useState(50);
  const [lossRate, setLossRate] = useState(2);
  const [connections, setConnections] = useState(10);

  const metrics = useMemo(() => {
    const tcpConnect = 3 * rtt;
    const quicConnect = 1 * rtt;
    const quic0rtt = 0;
    const tcpHandshake = 2 * rtt;
    const quicHandshake = 1 * rtt;
    const tcpRecovery = 2 * rtt;
    const quicRecovery = 1 * rtt;
    const tcpHOL = connections * lossRate * 10;
    const quicHOL = 0;
    const tcpMigration = tcpConnect + tcpHandshake;
    const quicMigration = 0;
    return {
      connect: { tcp: tcpConnect, quic: quicConnect },
      zeroRTT: { tcp: "-", quic: quic0rtt },
      handshake: { tcp: tcpHandshake, quic: quicHandshake },
      recovery: { tcp: tcpRecovery, quic: quicRecovery },
      HOL: { tcp: tcpHOL, quic: quicHOL },
      migration: { tcp: tcpMigration, quic: quicMigration },
    };
  }, [rtt, lossRate, connections]);

  const benchmarks = [
    { label: "首次连接延迟", en: "Connection Setup", unit: "ms", tcp: metrics.connect.tcp, quic: metrics.connect.quic },
    { label: "TLS握手", en: "TLS Handshake", unit: "ms", tcp: metrics.handshake.tcp, quic: metrics.handshake.quic },
    { label: "丢包恢复", en: "Loss Recovery", unit: "ms", tcp: metrics.recovery.tcp, quic: metrics.recovery.quic },
    { label: "队头阻塞延迟", en: "HOL Blocking", unit: "ms", tcp: metrics.HOL.tcp, quic: metrics.HOL.quic },
    { label: "连接迁移", en: "Migration", unit: "ms", tcp: metrics.migration.tcp, quic: metrics.migration.quic },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">QUIC vs TCP 基准测试</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          RTT: <span className="text-text-primary font-mono">{rtt}ms</span>
          <input type="range" min={10} max={300} value={rtt} onChange={(e) => setRtt(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          丢包率: <span className="text-text-primary font-mono">{lossRate}%</span>
          <input type="range" min={0} max={20} value={lossRate} onChange={(e) => setLossRate(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          并发连接: <span className="text-text-primary font-mono">{connections}</span>
          <input type="range" min={1} max={100} value={connections} onChange={(e) => setConnections(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="space-y-2.5">
        {benchmarks.map((b, i) => {
          const maxVal = Math.max(b.tcp as number, b.quic as number, 1);
          return (
            <div key={i} className="rounded-lg border border-border-subtle bg-bg-tertiary p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium text-text-primary">{b.label} <span className="text-text-tertiary">{b.en}</span></span>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-red-400 w-8 shrink-0">TCP</span>
                  <div className="flex-1 h-3 bg-bg-elevated rounded-full overflow-hidden">
                    <div className="h-full bg-red-400 rounded-full transition-all" style={{ width: `${((b.tcp as number) / maxVal) * 100}%` }} />
                  </div>
                  <span className="text-[10px] font-mono text-text-secondary w-12 text-right">{b.tcp}{b.unit}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-sky-400 w-8 shrink-0">QUIC</span>
                  <div className="flex-1 h-3 bg-bg-elevated rounded-full overflow-hidden">
                    <div className="h-full bg-sky-400 rounded-full transition-all" style={{ width: `${((b.quic as number) / maxVal) * 100}%` }} />
                  </div>
                  <span className="text-[10px] font-mono text-text-secondary w-12 text-right">{b.quic}{b.unit}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-3 rounded-lg border border-sky-500/20 bg-sky-500/5 p-3 text-xs text-text-secondary">
        <span className="font-medium text-sky-600 dark:text-sky-400">0-RTT恢复:</span> QUIC支持0-RTT数据发送（延迟={metrics.zeroRTT.quic}ms），TCP无法实现。高丢包/高延迟场景QUIC优势更明显。
      </div>
    </div>
  );
}
export default QUICTCPBenchmark;
