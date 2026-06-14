"use client";
import { useState, useCallback } from "react";

interface Queue { id: number; weight: number; packets: number; served: number; color: string }

export function WFQSchedulerSimulator() {
  const [queues, setQueues] = useState<Queue[]>([
    { id: 1, weight: 4, packets: 0, served: 0, color: "blue" },
    { id: 2, weight: 2, packets: 0, served: 0, color: "green" },
    { id: 3, weight: 1, packets: 0, served: 0, color: "yellow" },
  ]);
  const [log, setLog] = useState<string[]>([]);
  const [totalServed, setTotalServed] = useState(0);

  const addPacket = (qid: number) => {
    setQueues((qs) => qs.map((q) => q.id === qid ? { ...q, packets: q.packets + 1 } : q));
  };

  const schedule = useCallback(() => {
    const totalWeight = queues.reduce((s, q) => s + (q.packets > 0 ? q.weight : 0), 0);
    if (totalWeight === 0) return;

    let selected: Queue | null = null;
    let minFinish = Infinity;
    queues.forEach((q) => {
      if (q.packets > 0) {
        const finish = (q.served + 1) / q.weight;
        if (finish < minFinish) { minFinish = finish; selected = q; }
      }
    });

    if (selected) {
      setQueues((qs) => qs.map((q) =>
        q.id === selected!.id ? { ...q, packets: q.packets - 1, served: q.served + 1 } : q
      ));
      setTotalServed((t) => t + 1);
      setLog((l) => [...l.slice(-9), `队列 ${selected!.id} (权重${selected!.weight}) 发送一个包`]);
    }
  }, [queues]);

  const reset = () => {
    setQueues((qs) => qs.map((q) => ({ ...q, packets: 0, served: 0 })));
    setLog([]); setTotalServed(0);
  };

  const totalWeight = queues.reduce((s, q) => s + q.weight, 0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">WFQ 调度器模拟器 (Weighted Fair Queue)</h3>
      <div className="space-y-3 mb-4">
        {queues.map((q) => (
          <div key={q.id} className={`p-4 rounded-lg bg-${q.color}-500/10 border border-${q.color}-400/30`}>
            <div className="flex items-center justify-between mb-2">
              <span className={`text-${q.color}-400 text-sm font-medium`}>队列 {q.id} (权重: {q.weight})</span>
              <div className="flex items-center gap-2">
                <span className="text-text-muted text-xs">带宽: {((q.weight / totalWeight) * 100).toFixed(0)}%</span>
                <button onClick={() => addPacket(q.id)}
                  className={`px-3 py-1 rounded bg-${q.color}-500 text-white text-xs hover:opacity-80`}>
                  + 包
                </button>
              </div>
            </div>
            <div className="flex gap-1">
              {Array.from({ length: 10 }, (_, i) => (
                <div key={i} className={`w-6 h-4 rounded border transition-colors ${
                  i < q.packets ? `bg-${q.color}-500 border-${q.color}-400` : "bg-gray-200 dark:bg-gray-700 border-gray-300 dark:border-gray-600"
                }`} />
              ))}
            </div>
            <div className="flex justify-between mt-1">
              <span className="text-text-muted text-xs">排队: {q.packets}</span>
              <span className="text-text-muted text-xs">已服务: {q.served}</span>
            </div>
          </div>
        ))}
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={schedule} disabled={queues.every((q) => q.packets === 0)}
          className="px-4 py-2 rounded bg-blue-500 text-white text-sm disabled:opacity-50 hover:bg-blue-600">
          调度 (WFQ)
        </button>
        <button onClick={() => { for (let i = 0; i < 5; i++) queues.forEach((q) => addPacket(q.id)); }}
          className="px-3 py-2 rounded border border-border-subtle text-text-muted text-sm">批量添加</button>
        <button onClick={reset} className="px-3 py-2 rounded border border-border-subtle text-text-muted text-sm">重置</button>
        <span className="text-text-secondary text-sm self-center">已调度: {totalServed}</span>
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle mb-3 max-h-32 overflow-y-auto">
        {log.map((l, i) => <p key={i} className="text-text-muted text-xs">{l}</p>)}
        {log.length === 0 && <p className="text-text-muted text-xs">添加数据包并点击调度</p>}
      </div>
      <p className="text-text-muted text-xs">WFQ 按虚拟完成时间排序，权重越高的队列获得更多带宽份额，保证公平性。</p>
    </div>
  );
}
export default WFQSchedulerSimulator;
