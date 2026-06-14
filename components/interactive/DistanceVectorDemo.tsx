"use client";
import { useState } from "react";

interface RouterState {
  id: string;
  table: Record<string, { dist: number; next: string }>;
}

const INIT_TABLES: RouterState[] = [
  { id: "A", table: { A: { dist: 0, next: "-" }, B: { dist: 1, next: "B" }, C: { dist: Infinity, next: "-" }, D: { dist: Infinity, next: "-" } } },
  { id: "B", table: { A: { dist: 1, next: "A" }, B: { dist: 0, next: "-" }, C: { dist: 2, next: "C" }, D: { dist: Infinity, next: "-" } } },
  { id: "C", table: { A: { dist: Infinity, next: "-" }, B: { dist: 2, next: "B" }, C: { dist: 0, next: "-" }, D: { dist: 1, next: "D" } } },
  { id: "D", table: { A: { dist: Infinity, next: "-" }, B: { dist: Infinity, next: "-" }, C: { dist: 1, next: "C" }, D: { dist: 0, next: "-" } } },
];

function deepCopy(tables: RouterState[]): RouterState[] {
  return tables.map((r) => ({ id: r.id, table: Object.fromEntries(Object.entries(r.table).map(([k, v]) => [k, { ...v }])) }));
}

export function DistanceVectorDemo() {
  const [tables, setTables] = useState<RouterState[]>(deepCopy(INIT_TABLES));
  const [round, setRound] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const [converged, setConverged] = useState(false);

  const doRound = () => {
    const newTables = deepCopy(tables);
    const msgs: string[] = [];
    let changed = false;

    for (const router of newTables) {
      const neighbors = INIT_TABLES.find((r) => r.id === router.id)!;
      for (const [dest, entry] of Object.entries(neighbors.table)) {
        if (entry.dist === 0 || entry.dist === Infinity) continue;
        const neighborId = entry.next === "-" ? dest : entry.next;
        const neighbor = newTables.find((r) => r.id === neighborId);
        if (!neighbor) continue;
        for (const [d, e] of Object.entries(neighbor.table)) {
          const newDist = entry.dist + e.dist;
          if (newDist < router.table[d].dist) {
            msgs.push(`${router.id}: ${d} via ${neighborId} = ${newDist} (原${router.table[d].dist === Infinity ? "∞" : router.table[d].dist})`);
            router.table[d] = { dist: newDist, next: neighborId };
            changed = true;
          }
        }
      }
    }

    setTables(newTables);
    setRound(round + 1);
    setLog([...log, `--- 第${round + 1}轮 ---`, ...msgs, changed ? "" : "无更新,已收敛!"].filter(Boolean));
    if (!changed) setConverged(true);
  };

  const reset = () => {
    setTables(deepCopy(INIT_TABLES));
    setRound(0);
    setLog([]);
    setConverged(false);
  };

  const ids = ["A", "B", "C", "D"];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">距离向量路由收敛演示</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        {tables.map((r) => (
          <div key={r.id} className="bg-bg-muted rounded p-3 text-xs">
            <div className="font-mono font-bold text-text-primary mb-1">路由器 {r.id}</div>
            <table className="w-full">
              <thead><tr className="text-text-secondary"><th className="text-left">目的</th><th>距离</th><th>下一跳</th></tr></thead>
              <tbody>
                {ids.map((d) => {
                  const e = r.table[d];
                  return (
                    <tr key={d} className={e.dist === 0 ? "text-green-500" : ""}>
                      <td className="font-mono">{d}</td>
                      <td className="text-center">{e.dist === Infinity ? "∞" : e.dist}</td>
                      <td className="text-center">{e.next}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ))}
      </div>
      <div className="flex gap-3 mb-4">
        <button onClick={doRound} disabled={converged} className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 hover:bg-blue-600 text-sm">交换一轮</button>
        <button onClick={reset} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">重置</button>
        <span className="text-sm text-text-secondary self-center">轮次: {round} {converged && "✅ 已收敛"}</span>
      </div>
      {log.length > 0 && (
        <div className="bg-bg-muted rounded p-3 max-h-32 overflow-y-auto text-xs font-mono text-text-secondary">
          {log.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      )}
      <div className="text-xs text-text-secondary mt-3">
        距离向量协议: 每个路由器向邻居发送自己的路由表,邻居根据Bellman-Ford算法更新。可能出现无穷计数(count-to-infinity)问题。
      </div>
    </div>
  );
}

export default DistanceVectorDemo;
