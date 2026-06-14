"use client";
import { useState } from "react";

const routers = [
  { id: "R1", networks: ["10.1.1.0/24", "10.1.2.0/24"] },
  { id: "R2", networks: ["10.1.2.0/24", "10.1.3.0/24"] },
  { id: "R3", networks: ["10.1.3.0/24", "10.1.4.0/24"] },
];

const linkCosts: Record<string, number> = { "R1-R2": 10, "R2-R3": 20, "R1-R3": 50 };

interface LSA {
  from: string;
  seq: number;
  age: number;
  links: { to: string; cost: number }[];
}

const initialLSDB: Record<string, LSA[]> = {
  R1: [{ from: "R1", seq: 1, age: 100, links: [{ to: "R2", cost: 10 }, { to: "R3", cost: 50 }] }],
  R2: [{ from: "R2", seq: 1, age: 100, links: [{ to: "R1", cost: 10 }, { to: "R3", cost: 20 }] }],
  R3: [{ from: "R3", seq: 1, age: 100, links: [{ to: "R1", cost: 50 }, { to: "R2", cost: 20 }] }],
};

export function LSDBDemo() {
  const [selectedRouter, setSelectedRouter] = useState("R1");
  const [step, setStep] = useState(0);
  const [lsdb, setLsdb] = useState<Record<string, LSA[]>>(initialLSDB);

  const flood = () => {
    const newLsdb: Record<string, LSA[]> = { R1: [], R2: [], R3: [] };
    for (const r of routers) {
      const allLSAs: LSA[] = [];
      for (const other of routers) {
        if (other.id === r.id) {
          allLSAs.push(...lsdb[other.id]);
        } else {
          const adj = Object.keys(linkCosts).find(
            (k) => k.includes(r.id) && k.includes(other.id)
          );
          if (adj) {
            allLSAs.push(
              ...lsdb[other.id].filter(
                (l) => !allLSAs.some((a) => a.from === l.from && a.seq >= l.seq)
              )
            );
          }
        }
      }
      newLsdb[r.id] = allLSAs;
    }
    setLsdb(newLsdb);
    setStep((s) => s + 1);
  };

  const reset = () => {
    setLsdb(initialLSDB);
    setStep(0);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        LSDB Demo <span className="text-text-secondary text-sm">— OSPF链路状态数据库同步</span>
      </h3>
      <div className="flex gap-2 mb-3">
        <button onClick={flood} className="px-3 py-1 rounded bg-green-600 text-white text-sm">
          泛洪LSA (步骤 {step + 1})
        </button>
        <button onClick={reset} className="px-3 py-1 rounded bg-gray-500 text-white text-sm">
          重置
        </button>
      </div>
      <div className="text-xs text-text-secondary mb-3">
        点击"泛洪LSA"模拟链路状态通告在区域内的传播
      </div>
      <div className="flex gap-2 mb-3">
        {routers.map((r) => (
          <button
            key={r.id}
            onClick={() => setSelectedRouter(r.id)}
            className={`px-3 py-1 rounded text-sm ${selectedRouter === r.id ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {r.id}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
        <div className="font-semibold text-text-primary mb-2">{selectedRouter} 的 LSDB</div>
        <div className="space-y-2">
          {lsdb[selectedRouter].map((lsa, i) => (
            <div key={i} className="bg-white dark:bg-gray-900 p-2 rounded text-sm border border-gray-200 dark:border-gray-700">
              <div className="flex justify-between">
                <span className="font-mono text-text-primary">LSA from {lsa.from}</span>
                <span className="text-text-secondary">Seq: {lsa.seq} | Age: {lsa.age}s</span>
              </div>
              <div className="text-xs text-text-secondary mt-1">
                {lsa.links.map((l, j) => (
                  <span key={j} className="mr-3">→ {l.to} (cost: {l.cost})</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default LSDBDemo;
