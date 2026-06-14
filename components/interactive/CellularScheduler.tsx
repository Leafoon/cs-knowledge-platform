"use client";
import { useState } from "react";

interface RB {
  id: number;
  startFreq: number;
  bandwidth: number;
  assignedTo: number | null;
  color: string;
}

interface UE {
  id: number;
  name: string;
  priority: number;
  demand: number;
  allocated: number;
  color: string;
}

type Algo = "round-robin" | "proportional-fair" | "max-rate";

export function CellularScheduler() {
  const [algorithm, setAlgorithm] = useState<Algo>("round-robin");
  const [tick, setTick] = useState(0);

  const colors = ["bg-blue-500", "bg-green-500", "bg-purple-500", "bg-orange-500", "bg-pink-500"];
  const [ues] = useState<UE[]>([
    { id: 0, name: "UE-A", priority: 3, demand: 5, allocated: 0, color: colors[0] },
    { id: 1, name: "UE-B", priority: 2, demand: 3, allocated: 0, color: colors[1] },
    { id: 2, name: "UE-C", priority: 1, demand: 4, allocated: 0, color: colors[2] },
    { id: 3, name: "UE-D", priority: 2, demand: 2, allocated: 0, color: colors[3] },
  ]);

  const [rbs, setRBs] = useState<RB[]>(() =>
    Array.from({ length: 20 }, (_, i) => ({
      id: i, startFreq: 1800 + i * 0.18, bandwidth: 0.18, assignedTo: null, color: "bg-gray-300 dark:bg-gray-600",
    }))
  );

  const schedule = () => {
    const newRBs = rbs.map((rb) => ({ ...rb, assignedTo: null as number | null, color: "bg-gray-300 dark:bg-gray-600" }));
    const alloc = [0, 0, 0, 0];

    if (algorithm === "round-robin") {
      let turn = tick % ues.length;
      for (let i = 0; i < newRBs.length; i++) {
        while (alloc[turn] >= ues[turn].demand) { turn = (turn + 1) % ues.length; }
        newRBs[i].assignedTo = turn;
        newRBs[i].color = colors[turn];
        alloc[turn]++;
        turn = (turn + 1) % ues.length;
      }
    } else if (algorithm === "proportional-fair") {
      const cqi = [10, 8, 12, 6];
      const avgRate = ues.map((_, i) => Math.max(1, alloc[i] + 1));
      for (let i = 0; i < newRBs.length; i++) {
        const pfMetric = cqi.map((c, j) => c / avgRate[j]);
        const best = pfMetric.indexOf(Math.max(...pfMetric));
        newRBs[i].assignedTo = best;
        newRBs[i].color = colors[best];
        alloc[best]++;
      }
    } else {
      const cqi = [10, 8, 12, 6];
      for (let i = 0; i < newRBs.length; i++) {
        let best = 0;
        for (let j = 1; j < ues.length; j++) {
          if (cqi[j] > cqi[best] && alloc[j] < ues[j].demand) best = j;
        }
        if (alloc[best] >= ues[best].demand) {
          best = ues.findIndex((_, j) => alloc[j] < ues[j].demand);
          if (best === -1) best = 0;
        }
        newRBs[i].assignedTo = best;
        newRBs[i].color = colors[best];
        alloc[best]++;
      }
    }

    setRBs(newRBs);
    setTick((t) => t + 1);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">LTE 资源块调度</h3>
      <div className="flex gap-2 mb-4">
        {(["round-robin", "proportional-fair", "max-rate"] as Algo[]).map((a) => (
          <button key={a} onClick={() => setAlgorithm(a)}
            className={`flex-1 py-1.5 rounded text-sm ${algorithm === a ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {a === "round-robin" ? "轮询" : a === "proportional-fair" ? "比例公平" : "最大速率"}
          </button>
        ))}
      </div>
      <div className="mb-4">
        <p className="text-xs text-text-secondary mb-2">资源块分配 (20 PRBs @ 180kHz)</p>
        <div className="flex gap-0.5">
          {rbs.map((rb) => (
            <div key={rb.id} className={`flex-1 h-10 rounded-sm ${rb.color} flex items-center justify-center text-[9px] text-white font-bold`}>
              {rb.assignedTo !== null ? ues[rb.assignedTo].name : ""}
            </div>
          ))}
        </div>
        <div className="flex justify-between text-[10px] text-text-secondary mt-1">
          <span>1800 MHz</span><span>{(1800 + 20 * 0.18).toFixed(1)} MHz</span>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2 mb-4">
        {ues.map((ue) => {
          const count = rbs.filter((rb) => rb.assignedTo === ue.id).length;
          return (
            <div key={ue.id} className={`p-2 rounded border ${count > 0 ? "border-green-300 bg-green-50 dark:bg-green-900/10" : "border-border-subtle"}`}>
              <div className="flex items-center gap-1 mb-1">
                <div className={`w-3 h-3 rounded ${ue.color}`} />
                <span className="text-xs font-bold text-text-primary">{ue.name}</span>
              </div>
              <div className="text-[10px] text-text-secondary">需求: {ue.demand} RBs</div>
              <div className="text-[10px] text-text-secondary">分配: {count} RBs</div>
            </div>
          );
        })}
      </div>
      <div className="flex gap-2">
        <button onClick={schedule} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">调度一次</button>
        <button onClick={() => setRBs(rbs.map((rb) => ({ ...rb, assignedTo: null, color: "bg-gray-300 dark:bg-gray-600" })))}
          className="px-4 py-2 bg-gray-500 text-white rounded text-sm">重置</button>
      </div>
      <p className="text-xs text-text-secondary mt-3">轮询：公平分配；比例公平：兼顾公平与信道质量；最大速率：优先信道条件好的用户。</p>
    </div>
  );
}
export default CellularScheduler;
