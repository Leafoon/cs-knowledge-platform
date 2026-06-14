"use client";
import { useState } from "react";

interface Server {
  id: number;
  name: string;
  ip: string;
  weight: number;
  currentWeight: number;
  connections: number;
  healthy: boolean;
}

export function DNSLoadBalancerDemo() {
  const [servers, setServers] = useState<Server[]>([
    { id: 1, name: "Server-A", ip: "10.0.0.1", weight: 5, currentWeight: 0, connections: 0, healthy: true },
    { id: 2, name: "Server-B", ip: "10.0.0.2", weight: 3, currentWeight: 0, connections: 0, healthy: true },
    { id: 3, name: "Server-C", ip: "10.0.0.3", weight: 2, currentWeight: 0, connections: 0, healthy: true },
  ]);
  const [totalQueries, setTotalQueries] = useState(0);
  const [lastResolved, setLastResolved] = useState<string | null>(null);

  const resolve = () => {
    const healthy = servers.filter((s) => s.healthy);
    if (healthy.length === 0) { setLastResolved("所有服务器不可用!"); return; }

    const totalWeight = healthy.reduce((s, sv) => s + sv.weight, 0);
    let best: Server | null = null;
    setServers((prev) => {
      const updated = prev.map((s) => {
        if (!s.healthy) return s;
        return { ...s, currentWeight: s.currentWeight + s.weight };
      });
      const candidate = updated.filter((s) => s.healthy).sort((a, b) => b.currentWeight - a.currentWeight)[0];
      if (candidate) {
        best = candidate;
        return updated.map((s) => s.id === candidate.id ? { ...s, currentWeight: s.currentWeight - totalWeight, connections: s.connections + 1 } : s);
      }
      return updated;
    });
    setTotalQueries((t) => t + 1);
    if (best) setLastResolved(`${(best as Server).name} (${(best as Server).ip})`);
  };

  const toggleHealth = (id: number) => {
    setServers((prev) => prev.map((s) => s.id === id ? { ...s, healthy: !s.healthy } : s));
  };

  const updateWeight = (id: number, delta: number) => {
    setServers((prev) => prev.map((s) => s.id === id ? { ...s, weight: Math.max(1, Math.min(10, s.weight + delta)) } : s));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNS 加权负载均衡演示</h3>
      <p className="text-sm text-text-secondary mb-4">加权轮询(Weighted Round Robin)按权重比例分配请求。</p>
      <div className="space-y-3 mb-4">
        {servers.map((s) => (
          <div key={s.id} className={`p-3 rounded border transition-all ${s.healthy ? "border-border-subtle" : "border-red-300 dark:border-red-800 opacity-50"}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <button onClick={() => toggleHealth(s.id)}
                  className={`w-3 h-3 rounded-full ${s.healthy ? "bg-green-500" : "bg-red-500"}`} />
                <span className="font-mono text-sm text-text-primary">{s.name}</span>
                <span className="text-xs text-text-secondary">{s.ip}</span>
              </div>
              <span className="text-xs font-mono text-text-secondary">{s.connections} 次</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-text-secondary w-10">权重</span>
              <button onClick={() => updateWeight(s.id, -1)} className="px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-xs">-</button>
              <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${(s.weight / 10) * 100}%` }} />
              </div>
              <button onClick={() => updateWeight(s.id, 1)} className="px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-xs">+</button>
              <span className="font-mono text-sm text-text-primary w-6 text-center">{s.weight}</span>
            </div>
          </div>
        ))}
      </div>
      <button onClick={resolve} className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">
        发起DNS查询
      </button>
      {lastResolved && (
        <div className="mt-3 p-2 bg-gray-50 dark:bg-gray-900 rounded text-sm text-center">
          解析到: <span className="font-mono text-blue-600">{lastResolved}</span> (总查询: {totalQueries})
        </div>
      )}
    </div>
  );
}
export default DNSLoadBalancerDemo;
