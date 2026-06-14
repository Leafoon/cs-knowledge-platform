"use client";
import { useState } from "react";

interface Route {
  network: string;
  nextHop: string;
  asPath: number[];
  localPref: number;
  med: number;
  origin: string;
  weight: number;
}

const routes: Route[] = [
  { network: "10.0.0.0/8", nextHop: "192.168.1.1", asPath: [65001, 65002], localPref: 100, med: 50, origin: "IGP", weight: 0 },
  { network: "10.0.0.0/8", nextHop: "192.168.2.1", asPath: [65003], localPref: 150, med: 100, origin: "IGP", weight: 0 },
  { network: "10.0.0.0/8", nextHop: "192.168.3.1", asPath: [65004, 65005, 65006], localPref: 100, med: 30, origin: "EGP", weight: 0 },
];

const rules = [
  "① 最高 Weight",
  "② 最高 Local Preference",
  "③ 本地始发路由",
  "④ 最短 AS-PATH",
  "⑤ 最低 Origin 类型（IGP<EGP<Incomplete）",
  "⑥ 最低 MED",
  "⑦ eBGP 优于 iBGP",
  "⑧ 最近 IGP 度量",
  "⑨ 最低 Router ID",
];

export function BGPDecisionDemo() {
  const [step, setStep] = useState(0);
  const [selectedRule, setSelectedRule] = useState(0);

  const getWinner = () => {
    if (selectedRule <= 1) return 1;
    if (selectedRule === 3) return 1;
    if (selectedRule === 5) return 2;
    return 0;
  };

  const winner = getWinner();

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">BGP 选路决策演示</h3>
      <div className="mb-4 p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-text-secondary text-xs border-b border-border-subtle">
              <th className="text-left py-1 px-2">下一跳</th>
              <th className="text-left py-1 px-2">AS-PATH</th>
              <th className="text-left py-1 px-2">Local Pref</th>
              <th className="text-left py-1 px-2">MED</th>
              <th className="text-left py-1 px-2">Origin</th>
            </tr>
          </thead>
          <tbody>
            {routes.map((r, i) => (
              <tr key={i} className={`border-t border-border-subtle ${i === winner ? "bg-green-50 dark:bg-green-900/20" : ""}`}>
                <td className="py-1 px-2 font-mono text-text-primary">{r.nextHop}</td>
                <td className="py-1 px-2 font-mono text-text-primary">{r.asPath.join(" → ")}</td>
                <td className="py-1 px-2 text-text-primary">{r.localPref}</td>
                <td className="py-1 px-2 text-text-primary">{r.med}</td>
                <td className="py-1 px-2 text-text-primary">{r.origin}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mb-4">
        <p className="text-sm font-medium text-text-primary mb-2">BGP 选路规则（按优先级排序）</p>
        <div className="flex flex-wrap gap-1">
          {rules.map((r, i) => (
            <button key={i} onClick={() => setSelectedRule(i)} className={`px-2 py-1 rounded text-xs transition-colors ${selectedRule === i ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
              {r}
            </button>
          ))}
        </div>
      </div>
      <div className="p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
        <p className="text-sm font-medium text-green-700 dark:text-green-300">
          选中规则 {rules[selectedRule]}，最优路由: {routes[winner].nextHop}（AS-PATH: {routes[winner].asPath.join(" → ")}）
        </p>
        <p className="text-xs text-green-600 dark:text-green-400 mt-1">网络: {routes[winner].network} | Local Pref: {routes[winner].localPref} | MED: {routes[winner].med}</p>
      </div>
    </div>
  );
}
export default BGPDecisionDemo;
