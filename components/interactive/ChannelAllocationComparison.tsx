"use client";
import { useState } from "react";

type Scheme = "fixed" | "dynamic" | "random";

export function ChannelAllocationComparison() {
  const [scheme, setScheme] = useState<Scheme>("fixed");
  const [totalChannels] = useState(12);
  const [cells, setCells] = useState<number[][]>([]);
  const [cellLoads, setCellLoads] = useState<number[]>([3, 5, 2, 4, 6, 3, 2, 5, 4]);
  const [allocations, setAllocations] = useState<(number | null)[]>([]);

  const initCells = () => {
    const newCells: number[][] = [];
    if (scheme === "fixed") {
      const perCell = Math.floor(totalChannels / 9);
      for (let i = 0; i < 9; i++) {
        newCells.push(Array.from({ length: perCell }, (_, j) => i * perCell + j));
      }
    } else if (scheme === "dynamic") {
      for (let i = 0; i < 9; i++) {
        const needed = Math.min(cellLoads[i], totalChannels);
        newCells.push(Array.from({ length: needed }, (_, j) => j));
      }
    } else {
      for (let i = 0; i < 9; i++) {
        const available = Math.floor(Math.random() * 6) + 2;
        const channels = new Set<number>();
        while (channels.size < available) channels.add(Math.floor(Math.random() * totalChannels));
        newCells.push([...channels].sort((a, b) => a - b));
      }
    }
    setCells(newCells);
    setAllocations(newCells.map(() => null));
  };

  const allocate = () => {
    const newAlloc = cells.map((cellChannels, i) => {
      if (cellChannels.length === 0) return null;
      const idx = Math.floor(Math.random() * cellChannels.length);
      return cellChannels[idx];
    });
    setAllocations(newAlloc);
  };

  const satisfied = allocations.filter((a, i) => a !== null && cellLoads[i] <= cells[i].length).length;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">信道分配方案对比</h3>
      <div className="flex gap-2 mb-4">
        {(["fixed", "dynamic", "random"] as Scheme[]).map((s) => (
          <button key={s} onClick={() => { setScheme(s); setCells([]); setAllocations([]); }}
            className={`flex-1 py-1.5 rounded text-sm ${scheme === s ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {s === "fixed" ? "固定分配" : s === "dynamic" ? "动态分配" : "随机分配"}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4">
        {cellLoads.map((load, i) => (
          <div key={i} className={`p-2 rounded border text-center ${allocations[i] !== null ? "border-green-400 bg-green-50 dark:bg-green-900/10" : "border-border-subtle bg-gray-50 dark:bg-gray-800"}`}>
            <div className="text-xs font-bold text-text-primary">小区 {i + 1}</div>
            <div className="text-[10px] text-text-secondary">负载: {load} 用户</div>
            <div className="text-[10px] text-text-secondary">可用: {cells[i]?.length || 0} 信道</div>
            {allocations[i] !== null && (
              <div className="text-xs font-mono text-green-600 mt-0.5">CH{allocations[i]}</div>
            )}
          </div>
        ))}
      </div>
      <div className="mb-4 p-3 rounded bg-gray-50 dark:bg-gray-800">
        <div className="flex justify-between text-xs text-text-secondary mb-2">
          <span>总信道: {totalChannels}</span>
          <span>满足: {satisfied}/{cellLoads.length}</span>
        </div>
        <div className="flex gap-0.5">
          {Array.from({ length: totalChannels }).map((_, i) => {
            const inUse = allocations.includes(i);
            return <div key={i} className={`flex-1 h-4 rounded-sm ${inUse ? "bg-blue-500" : "bg-gray-200 dark:bg-gray-700"}`} title={`CH${i}: ${inUse ? "占用" : "空闲"}`} />;
          })}
        </div>
      </div>
      <div className="flex gap-2">
        <button onClick={initCells} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">初始化</button>
        <button onClick={allocate} className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm">分配信道</button>
      </div>
      <p className="text-xs text-text-secondary mt-3">固定：每个小区分配固定信道集；动态：按需分配，需要中心控制器；随机：每个小区随机选取可用信道。</p>
    </div>
  );
}
export default ChannelAllocationComparison;
