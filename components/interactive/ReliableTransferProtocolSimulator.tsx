"use client";
import { useState, useEffect, useCallback, useRef } from "react";

interface Packet {
  id: number;
  sent: boolean;
  acked: boolean;
  lost: boolean;
  retransmit: boolean;
}

export function ReliableTransferProtocolSimulator() {
  const [windowSize, setWindowSize] = useState(4);
  const [lossRate, setLossRate] = useState(15);
  const [packets, setPackets] = useState<Packet[]>([]);
  const [base, setBase] = useState(0);
  const [nextSeq, setNextSeq] = useState(0);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const totalPackets = 16;

  const initPackets = useCallback(() => {
    setPackets(Array.from({ length: totalPackets }, (_, i) => ({ id: i, sent: false, acked: false, lost: false, retransmit: false })));
    setBase(0);
    setNextSeq(0);
    setLog([]);
  }, []);

  useEffect(() => { initPackets(); }, [initPackets]);

  const startSim = useCallback(() => {
    initPackets();
    setRunning(true);
  }, [initPackets]);

  useEffect(() => {
    if (!running) return;
    timerRef.current = setInterval(() => {
      setNextSeq((seq) => {
        setBase((b) => {
          const ackedCount = packets.filter((p, i) => i >= b && p.acked).length;
          const newBase = b + ackedCount;
          return newBase;
        });
        return seq;
      });
      setPackets((prev) => {
        const copy = [...prev];
        for (let i = base; i < Math.min(base + windowSize, totalPackets); i++) {
          if (!copy[i].sent && !copy[i].acked) {
            const lost = Math.random() * 100 < lossRate;
            copy[i] = { ...copy[i], sent: true, lost };
            setLog((l) => [...l, lost ? `[发送] #${i} - 丢失!` : `[发送] #${i}`]);
            break;
          }
          if (copy[i].sent && !copy[i].acked && !copy[i].lost) {
            copy[i] = { ...copy[i], acked: true };
            setLog((l) => [...l, `[ACK] #${i} 确认`]);
            break;
          }
          if (copy[i].lost && !copy[i].retransmit) {
            copy[i] = { ...copy[i], retransmit: true, lost: false };
            setLog((l) => [...l, `[重传] #${i}`]);
            break;
          }
          if (copy[i].retransmit && !copy[i].acked) {
            copy[i] = { ...copy[i], acked: true };
            setLog((l) => [...l, `[ACK] #${i} 重传确认`]);
            break;
          }
        }
        return copy;
      });
      setNextSeq((s) => Math.min(s + 1, totalPackets));
    }, 600);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running, base, windowSize, lossRate, packets]);

  const allDone = packets.every((p) => p.acked);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">可靠传输模拟器</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          窗口大小: <span className="text-text-primary font-mono">{windowSize}</span>
          <input type="range" min={1} max={8} value={windowSize} onChange={(e) => setWindowSize(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          丢包率: <span className="text-text-primary font-mono">{lossRate}%</span>
          <input type="range" min={0} max={40} value={lossRate} onChange={(e) => setLossRate(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={startSim} className="px-4 py-1.5 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-xs font-medium hover:bg-sky-500/25 transition-colors">
          {running ? "重新开始" : "开始传输"}
        </button>
        {allDone && <span className="text-xs text-emerald-500 flex items-center">✓ 传输完成</span>}
      </div>
      <div className="flex flex-wrap gap-1 mb-4">
        {packets.map((p, i) => {
          const inWindow = i >= base && i < base + windowSize;
          let cls = "bg-bg-tertiary border-border-subtle text-text-secondary";
          if (p.acked) cls = "bg-emerald-500/20 border-emerald-400/40 text-emerald-700 dark:text-emerald-300";
          else if (p.lost) cls = "bg-red-500/20 border-red-400/40 text-red-700 dark:text-red-300";
          else if (p.retransmit) cls = "bg-amber-500/20 border-amber-400/40 text-amber-700 dark:text-amber-300";
          else if (p.sent) cls = "bg-sky-500/20 border-sky-400/40 text-sky-700 dark:text-sky-300";
          return (
            <div key={i} className={`w-10 h-10 rounded-lg border flex flex-col items-center justify-center text-xs font-mono transition-all ${cls} ${inWindow && !p.acked ? "ring-2 ring-sky-400/50" : ""}`}>
              <span className="font-bold">{i}</span>
              <span className="text-[8px]">{p.acked ? "✓" : p.lost ? "✗" : p.sent ? "→" : "-"}</span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-3 mb-2 text-[10px]">
        <span className="text-text-tertiary">base={base}</span>
        <span className="text-text-tertiary">nextSeq={nextSeq}</span>
        <div className="flex gap-2 ml-auto">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-sky-400" /> 已发送</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-emerald-400" /> 已确认</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-red-400" /> 丢失</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-amber-400" /> 重传</span>
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 max-h-28 overflow-y-auto">
        <div className="text-[10px] font-mono space-y-0.5">
          {log.length === 0 ? <span className="text-text-tertiary">等待传输...</span> : log.slice(-10).map((l, i) => (
            <div key={i} className={l.includes("ACK") ? "text-emerald-400" : l.includes("丢失") ? "text-red-400" : l.includes("重传") ? "text-amber-400" : "text-text-secondary"}>{l}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
export default ReliableTransferProtocolSimulator;
