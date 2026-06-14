"use client";
import { useState } from "react";

interface Lease {
  ip: string;
  mac: string;
  hostname: string;
  expiry: number;
  status: "allocated" | "expired";
}

export function DHCPPoolDemo() {
  const startIP = "192.168.1.100";
  const poolSize = 10;
  const [leases, setLeases] = useState<Lease[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  const log = (msg: string) => setLogs((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 10));

  const allocate = () => {
    const used = new Set(leases.filter((l) => l.status === "allocated").map((l) => l.ip));
    let ip = "";
    for (let i = 0; i < poolSize; i++) {
      const candidate = `192.168.1.${100 + i}`;
      if (!used.has(candidate)) { ip = candidate; break; }
    }
    if (!ip) { log("地址池已耗尽!"); return; }
    const mac = Array.from({ length: 6 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join(":");
    const hostname = `host-${leases.length + 1}`;
    const lease: Lease = { ip, mac, hostname, expiry: 3600, status: "allocated" };
    setLeases((prev) => [...prev, lease]);
    log(`分配 ${ip} → ${mac} (${hostname})`);
  };

  const expire = (idx: number) => {
    setLeases((prev) => prev.map((l, i) => i === idx ? { ...l, status: "expired" } : l));
    log(`释放 ${leases[idx].ip}`);
  };

  const reclaim = () => {
    const before = leases.length;
    setLeases((prev) => prev.filter((l) => l.status === "allocated"));
    log(`回收 ${before - leases.filter((l) => l.status === "allocated").length} 个过期地址`);
  };

  const allocatedCount = leases.filter((l) => l.status === "allocated").length;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DHCP 地址池演示</h3>
      <div className="mb-3 text-sm text-text-secondary">
        地址池: {startIP} - 192.168.1.{100 + poolSize - 1} | 已分配: {allocatedCount}/{poolSize}
      </div>
      <div className="flex flex-wrap gap-1 mb-4">
        {Array.from({ length: poolSize }, (_, i) => {
          const ip = `192.168.1.${100 + i}`;
          const lease = leases.find((l) => l.ip === ip);
          return (
            <div key={i} className={`w-10 h-10 rounded flex items-center justify-center text-xs font-mono transition-all ${lease?.status === "allocated" ? "bg-blue-500 text-white" : lease?.status === "expired" ? "bg-yellow-400 text-black" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
              .{100 + i}
            </div>
          );
        })}
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={allocate} className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm">分配地址</button>
        <button onClick={reclaim} className="flex-1 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded text-sm">回收过期</button>
      </div>
      <div className="space-y-1 mb-4 max-h-32 overflow-y-auto">
        {leases.filter((l) => l.status === "allocated").map((l, i) => (
          <div key={i} className="flex justify-between items-center text-xs bg-gray-50 dark:bg-gray-900 rounded px-3 py-1.5">
            <span className="font-mono text-text-primary">{l.ip}</span>
            <span className="text-text-secondary">{l.mac}</span>
            <button onClick={() => expire(leases.indexOf(l))} className="text-red-500 hover:text-red-700">释放</button>
          </div>
        ))}
      </div>
      <div className="bg-gray-50 dark:bg-gray-900 rounded p-2 max-h-24 overflow-y-auto">
        {logs.map((l, i) => <div key={i} className="text-xs font-mono text-text-secondary py-0.5">{l}</div>)}
      </div>
    </div>
  );
}
export default DHCPPoolDemo;
