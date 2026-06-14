"use client";
import { useState } from "react";

export function SlidingWindowConcept() {
  const [sendWindowSize, setSendWindowSize] = useState(4);
  const [sendBase, setSendBase] = useState(0);
  const [totalFrames, setTotalFrames] = useState(12);
  const [acked, setAcked] = useState<boolean[]>(new Array(12).fill(false));

  const advanceSendBase = () => {
    let newBase = sendBase;
    while (newBase < totalFrames && acked[newBase]) newBase++;
    if (newBase < totalFrames) {
      const next = [...acked];
      next[newBase] = true;
      setAcked(next);
      let base = newBase;
      while (base < totalFrames && next[base]) base++;
      setSendBase(base);
    }
  };

  const reset = () => {
    setSendBase(0);
    setAcked(new Array(totalFrames).fill(false));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">滑动窗口概念演示</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          发送窗口大小: <span className="text-text-primary font-mono">{sendWindowSize}</span>
          <input type="range" min={1} max={8} value={sendWindowSize} onChange={(e) => { setSendWindowSize(+e.target.value); reset(); }} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          总帧数: <span className="text-text-primary font-mono">{totalFrames}</span>
          <input type="range" min={6} max={20} value={totalFrames} onChange={(e) => { setTotalFrames(+e.target.value); setAcked(new Array(+e.target.value).fill(false)); setSendBase(0); }} className="w-full mt-1" />
        </label>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={advanceSendBase} className="px-4 py-1.5 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-xs font-medium hover:bg-sky-500/25 transition-colors">
          发送并确认下一帧
        </button>
        <button onClick={reset} className="px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary">重置</button>
      </div>
      <div className="relative rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="text-xs text-text-secondary mb-2">发送窗口 [base={sendBase}, base+N={Math.min(sendBase + sendWindowSize, totalFrames)})</div>
        <div className="flex flex-wrap gap-1">
          {Array.from({ length: totalFrames }, (_, i) => {
            const inWindow = i >= sendBase && i < sendBase + sendWindowSize;
            const isAcked = acked[i];
            return (
              <div key={i} className={`relative w-10 h-12 rounded-lg border flex flex-col items-center justify-center text-xs font-mono transition-all
                ${isAcked ? "bg-emerald-500/20 border-emerald-400/40 text-emerald-600 dark:text-emerald-400" :
                  inWindow ? "bg-sky-500/20 border-sky-400/40 text-sky-600 dark:text-sky-400 ring-1 ring-sky-400/30" :
                  "bg-bg-elevated border-border-subtle text-text-tertiary"}`}>
                <span className="font-bold">{i}</span>
                <span className="text-[8px]">{isAcked ? "✓" : inWindow ? "待发" : "-"}</span>
                {i === sendBase && <span className="absolute -top-3 left-1/2 -translate-x-1/2 text-[8px] text-sky-500 font-bold">base</span>}
              </div>
            );
          })}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-sky-500">{sendWindowSize}</div>
          <div className="text-[10px] text-text-tertiary">窗口大小 N</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-emerald-500">{acked.filter(Boolean).length}</div>
          <div className="text-[10px] text-text-tertiary">已确认帧数</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-text-primary">{totalFrames - acked.filter(Boolean).length}</div>
          <div className="text-[10px] text-text-tertiary">剩余帧数</div>
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-medium text-text-primary">滑动窗口机制</div>
        <div>• 发送窗口限制了发送方可以连续发送的未确认帧数量</div>
        <div>• 窗口大小 N=1 退化为停等协议</div>
        <div>• 每收到一个ACK，窗口向右滑动一格</div>
        <div>• 接收方可能有接收窗口，控制可接受的帧范围</div>
      </div>
    </div>
  );
}
export default SlidingWindowConcept;
