"use client";
import { useState, useCallback } from "react";

interface Event { time: number; type: "send" | "ack" | "timeout" | "loss"; seq: number; msg: string }

export function StopAndWaitSimulator() {
  const [events, setEvents] = useState<Event[]>([]);
  const [seqNum, setSeqNum] = useState(0);
  const [waiting, setWaiting] = useState(false);
  const [time, setTime] = useState(0);
  const [simulateLoss, setSimulateLoss] = useState(false);

  const send = useCallback(() => {
    if (waiting) return;
    const t = time;
    const newEvents: Event[] = [...events, { time: t, type: "send", seq: seqNum % 2, msg: `发送 DATA(seq=${seqNum % 2})` }];
    setWaiting(true);
    if (simulateLoss) {
      newEvents.push({ time: t + 2, type: "loss", seq: seqNum % 2, msg: `❌ ACK 丢失！` });
      newEvents.push({ time: t + 5, type: "timeout", seq: seqNum % 2, msg: `⏰ 超时！重传 DATA(seq=${seqNum % 2})` });
      newEvents.push({ time: t + 7, type: "ack", seq: (seqNum + 1) % 2, msg: `接收 ACK(seq=${(seqNum + 1) % 2})` });
      setTime(t + 7);
    } else {
      newEvents.push({ time: t + 3, type: "ack", seq: (seqNum + 1) % 2, msg: `接收 ACK(seq=${(seqNum + 1) % 2})` });
      setTime(t + 3);
    }
    setEvents(newEvents);
    setSeqNum((s) => s + 1);
    setTimeout(() => setWaiting(false), 100);
  }, [events, seqNum, time, waiting, simulateLoss]);

  const reset = () => { setEvents([]); setSeqNum(0); setWaiting(false); setTime(0); };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">停止等待模拟器 (Stop-and-Wait)</h3>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={send} disabled={waiting}
          className="px-4 py-2 rounded bg-blue-500 text-white text-sm font-medium disabled:opacity-50 hover:bg-blue-600 transition-colors">
          发送数据
        </button>
        <label className="flex items-center gap-2 text-text-secondary text-sm">
          <input type="checkbox" checked={simulateLoss} onChange={(e) => setSimulateLoss(e.target.checked)} className="rounded" />
          模拟ACK丢失
        </label>
        <button onClick={reset} className="px-3 py-2 rounded border border-border-subtle text-text-muted text-sm hover:text-text-primary transition-colors">
          重置
        </button>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 rounded bg-blue-500/10 border border-blue-400/30">
          <span className="text-blue-400 text-sm font-medium">发送方 (Sender)</span>
          <p className="text-text-muted text-xs mt-1">下一个序列号: {seqNum % 2}</p>
          <p className="text-text-muted text-xs">状态: {waiting ? "等待 ACK..." : "就绪"}</p>
        </div>
        <div className="p-3 rounded bg-green-500/10 border border-green-400/30">
          <span className="text-green-400 text-sm font-medium">接收方 (Receiver)</span>
          <p className="text-text-muted text-xs mt-1">期望序列号: {seqNum % 2}</p>
          <p className="text-text-muted text-xs">状态: 就绪</p>
        </div>
      </div>
      <div className="relative">
        <div className="absolute left-[25%] top-0 bottom-0 w-px bg-blue-400/30" />
        <div className="absolute left-[75%] top-0 bottom-0 w-px bg-green-400/30" />
        <div className="space-y-1">
          {events.map((e, i) => (
            <div key={i} className="relative flex items-center py-1">
              <span className="absolute left-0 text-text-muted text-xs w-[12%]">t={e.time}</span>
              {e.type === "send" || e.type === "timeout" ? (
                <div className="ml-[14%] flex items-center">
                  <div className="w-[50%] flex items-center">
                    <span className={`text-xs px-2 py-1 rounded ${e.type === "timeout" ? "bg-yellow-500/10 text-yellow-400 border border-yellow-400/30" : "bg-blue-500/10 text-blue-400 border border-blue-400/30"}`}>
                      → {e.msg}
                    </span>
                  </div>
                </div>
              ) : e.type === "ack" ? (
                <div className="ml-[14%] flex items-center">
                  <div className="w-[25%]" />
                  <span className="text-xs px-2 py-1 rounded bg-green-500/10 text-green-400 border border-green-400/30">
                    ← {e.msg}
                  </span>
                </div>
              ) : (
                <div className="ml-[14%] flex items-center">
                  <div className="w-[35%]" />
                  <span className="text-xs px-2 py-1 rounded bg-red-500/10 text-red-400 border border-red-400/30">
                    ✕ {e.msg}
                  </span>
                </div>
              )}
            </div>
          ))}
          {events.length === 0 && <p className="text-text-muted text-sm text-center py-4">点击"发送数据"开始模拟</p>}
        </div>
      </div>
      <div className="mt-4 p-3 rounded bg-bg-primary border border-border-subtle">
        <p className="text-text-muted text-xs">协议特点：每发送一帧就停止等待确认，效率 = T_trans / (T_trans + RTT + T_proc)。序列号仅需 1 bit。</p>
      </div>
    </div>
  );
}
export default StopAndWaitSimulator;
