"use client";
import { useState, useMemo } from "react";

interface Switch {
  id: number;
  mac: string;
  priority: number;
}

const switches: Switch[] = [
  { id: 1, mac: "00:1A:2B:00:00:01", priority: 32768 },
  { id: 2, mac: "00:1A:2B:00:00:02", priority: 32768 },
  { id: 3, mac: "00:1A:2B:00:00:03", priority: 40960 },
  { id: 4, mac: "00:1A:2B:00:00:04", priority: 32768 },
];

interface Link {
  from: number;
  to: number;
  cost: number;
}

const links: Link[] = [
  { from: 1, to: 2, cost: 4 },
  { from: 1, to: 3, cost: 10 },
  { from: 2, to: 3, cost: 2 },
  { from: 2, to: 4, cost: 6 },
  { from: 3, to: 4, cost: 3 },
];

export function STPSimulator() {
  const [selectedBridge, setSelectedBridge] = useState<number | null>(null);

  const rootBridge = useMemo(() => {
    return switches.reduce((min, s) => {
      const bridgeId = `${s.priority}.${s.mac}`;
      const minId = `${min.priority}.${min.mac}`;
      return bridgeId < minId ? s : min;
    });
  }, []);

  const portRoles = useMemo(() => {
    const roles: Record<string, string> = {};
    links.forEach((l) => {
      const fromIsRoot = l.from === rootBridge.id;
      const toIsRoot = l.to === rootBridge.id;
      const key1 = `${l.from}-${l.to}`;
      const key2 = `${l.to}-${l.from}`;
      if (fromIsRoot) roles[key1] = "指定端口";
      else if (toIsRoot) roles[key1] = "根端口";
      else roles[key1] = "指定端口";
      if (toIsRoot) roles[key2] = "指定端口";
      else if (fromIsRoot) roles[key2] = "根端口";
      else roles[key2] = l.cost <= links.find((ll) => (ll.from === l.to && ll.to === rootBridge.id) || (ll.to === l.to && ll.from === rootBridge.id))?.cost! ? "指定端口" : "非指定端口";
    });
    return roles;
  }, [rootBridge]);

  const getSwitchColor = (id: number) => {
    if (id === rootBridge.id) return "bg-emerald-500/20 border-emerald-400/40 text-emerald-700 dark:text-emerald-300";
    return "bg-sky-500/15 border-sky-400/40 text-sky-700 dark:text-sky-300";
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">STP生成树模拟器</h3>
      <div className="grid grid-cols-2 gap-2 mb-4">
        {switches.map((s) => (
          <button key={s.id} onClick={() => setSelectedBridge(s.id)}
            className={`px-3 py-2 rounded-lg border text-xs text-left transition-all ${selectedBridge === s.id ? "ring-2 ring-sky-400" : ""} ${getSwitchColor(s.id)}`}>
            <div className="font-medium">交换机 {s.id} {s.id === rootBridge.id ? "(根桥)" : ""}</div>
            <div className="font-mono text-[10px] opacity-70">MAC: {s.mac} | 优先级: {s.priority}</div>
          </button>
        ))}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="text-xs font-medium text-text-primary mb-2">链路与端口角色</div>
        <div className="space-y-1.5">
          {links.map((l, i) => {
            const key1 = `${l.from}-${l.to}`;
            const role1 = portRoles[key1] || "未知";
            return (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="font-mono text-text-primary">SW{l.from}</span>
                <span className="text-text-tertiary">←→</span>
                <span className="font-mono text-text-primary">SW{l.to}</span>
                <span className="text-text-tertiary">cost={l.cost}</span>
                <span className={`ml-auto px-1.5 py-0.5 rounded text-[10px] ${role1 === "根端口" ? "bg-emerald-500/15 text-emerald-500" : role1 === "指定端口" ? "bg-sky-500/15 text-sky-500" : "bg-red-500/15 text-red-500"}`}>{role1}</span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-medium text-text-primary">STP选举规则</div>
        <div>1. 根桥选举：Bridge ID最小者为根桥（优先级.MAC）</div>
        <div>2. 根端口：每个非根桥到根桥路径开销最小的端口</div>
        <div>3. 指定端口：每个网段上到根桥开销最小的端口</div>
        <div>4. 非指定端口（阻塞）：其余端口，防止环路</div>
      </div>
    </div>
  );
}
export default STPSimulator;
