"use client";
import { useState, useEffect, useCallback, useRef } from "react";

interface SRPacket {
  id: number;
  sent: boolean;
  acked: boolean;
  lost: boolean;
  retransmitted: boolean;
  received: boolean;
}

export function SRSimulator() {
  const [sendWindowSize, setSendWindowSize] = useState(4);
  const [recvWindowSize, setRecvWindowSize] = useState(4);
  const [lossRate, setLossRate] = useState(15);
  const [packets, setPackets] = useState<SRPacket[]>([]);
  const [sendBase, setSendBase] = useState(0);
  const [recvBase, setRecvBase] = useState(0);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const total = 16;

  const init = useCallback(() => {
    setPackets(Array.from({ length: total }, (_, i) => ({ id: i, sent: false, acked: false, lost: false, retransmitted: false, received: false })));
    setSendBase(0);
    setRecvBase(0);
    setLog([]);
  }, []);

  useEffect(() => { init(); }, [init]);

  const startSim = useCallback(() => { init(); setRunning(true); }, [init]);

  useEffect(() => {
    if (!running) return;
    timerRef.current = setInterval(() => {
      setPackets((prev) => {
        const copy = [...prev];
        for (let i = sendBase; i < Math.min(sendBase + sendWindowSize, total); i++) {
          if (!copy[i].sent && !copy[i].acked) {
            const lost = Math.random() * 100 < lossRate;
            copy[i] = { ...copy[i], sent: true, lost };
            setLog((l) => [...l.slice(-15), lost ? `[#${i}] 发送→丢失` : `[#${i}] 发送成功`]);
            break;
          }
          if (copy[i].sent && !copy[i].acked && !copy[i].lost) {
            copy[i] = { ...copy[i], acked: true };
            setLog((l) => [...l.slice(-15), `[#${i}] 收到ACK`]);
            break;
          }
          if (copy[i].lost && !copy[i].retransmitted) {
            copy[i] = { ...copy[i], retransmitted: true, lost: false };
            setLog((l) => [...l.slice(-15), `[#${i}] 超时→选择重传`]);
            break;
          }
          if (copy[i].retransmitted && !copy[i].acked) {
            copy[i] = { ...copy[i], acked: true };
            setLog((l) => [...l.slice(-15), `[#${i}] 重传ACK`]);
            break;
          }
        }
        let newBase = sendBase;
        while (newBase < total && copy[newBase].acked) newBase++;
        if (newBase !== sendBase) setSendBase(newBase);
        return copy;
      });
    }, 500);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running, sendBase, sendWindowSize, lossRate]);

  const allDone = packets.filter((p) => p.acked).length === total;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">选择重传 (SR) 模拟器</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          发送窗口: <span className="text-text-primary font-mono">{sendWindowSize}</span>
          <input type="range" min={1} max={8} value={sendWindowSize} onChange={(e) => setSendWindowSize(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          接收窗口: <span className="text-text-primary font-mono">{recvWindowSize}</span>
          <input type="range" min={1} max={8} value={recvWindowSize} onChange={(e) => setRecvWindowSize(+e.target.value)} className="w-full mt-1" />
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
        {allDone && <span className="text-xs text-emerald-500">✓ 传输完成</span>}
      </div>
      <div className="flex flex-wrap gap-1 mb-3">
        {packets.map((p) => {
          const inSendWindow = p.id >= sendBase && p.id < sendBase + sendWindowSize;
          let cls = "bg-bg-tertiary border-border-subtle text-text-secondary";
          if (p.acked) cls = "bg-emerald-500/20 border-emerald-400/40 text-emerald-600 dark:text-emerald-400";
          else if (p.lost) cls = "bg-red-500/20 border-red-400/40 text-red-600 dark:text-red-400";
          else if (p.retransmitted) cls = "bg-amber-500/20 border-amber-400/40 text-amber-600 dark:text-amber-400";
          else if (p.sent) cls = "bg-sky-500/20 border-sky-400/40 text-sky-600 dark:text-sky-400";
          return (
            <div key={p.id} className={`w-10 h-10 rounded-lg border flex flex-col items-center justify-center text-xs font-mono transition-all ${cls} ${inSendWindow && !p.acked ? "ring-2 ring-sky-400/50" : ""}`}>
              <span className="font-bold">{p.id}</span>
              <span className="text-[8px]">{p.acked ? "✓" : p.lost ? "✗" : p.sent ? "→" : "-"}</span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-3 mb-2 text-[10px]">
        <span className="text-text-tertiary">发送base={sendBase}</span>
        <span className="text-text-tertiary">接收base={recvBase}</span>
        <div className="flex gap-2 ml-auto">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-sky-400" /> 已发送</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-emerald-400" /> 已确认</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-red-400" /> 丢失</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-amber-400" /> 重传</span>
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 max-h-24 overflow-y-auto">
        <div className="text-[10px] font-mono space-y-0.5">
          {log.length === 0 ? <span className="text-text-tertiary">等待传输...</span> : log.map((l, i) => (
            <div key={i} className={l.includes("ACK") ? "text-emerald-400" : l.includes("丢失") ? "text-red-400" : l.includes("重传") ? "text-amber-400" : "text-text-secondary"}>{l}</div>
          ))}
        </div>
      </div>
      <div className="mt-3 text-[10px] text-text-tertiary">SR协议：发送窗口大小 ≤ 2^(n-1)，接收窗口可缓存乱序帧，仅重传丢失帧（vs GBN重传所有后续帧）</div>
    </div>
  );
}
export default SRSimulator;
