"use client";
import { useState } from "react";

interface Connection {
  id: number;
  srcIP: string;
  srcPort: number;
  dstIP: string;
  dstPort: number;
  state: string;
}

const scenarios = [
  { name: "SO_REUSEADDR", desc: "允许同一端口绑定多个套接字（地址重用）", port: 8080, sockets: ["192.168.1.1:8080", "0.0.0.0:8080", "10.0.0.1:8080"] },
  { name: "端口复用（SO_REUSEPORT）", desc: "内核级负载均衡，多个进程监听同一端口", port: 80, sockets: ["进程A: 0.0.0.0:80", "进程B: 0.0.0.0:80", "进程C: 0.0.0.0:80"] },
  { name: "NAT端口复用", desc: "多个内网主机共享一个公网IP的不同端口", port: 443, sockets: ["10.0.0.2:443 → 203.0.113.1:50001", "10.0.0.3:443 → 203.0.113.1:50002", "10.0.0.4:443 → 203.0.113.1:50003"] },
];

const connections: Connection[] = [
  { id: 1, srcIP: "192.168.1.10", srcPort: 49152, dstIP: "93.184.216.34", dstPort: 80, state: "ESTABLISHED" },
  { id: 2, srcIP: "192.168.1.10", srcPort: 49153, dstIP: "93.184.216.34", dstPort: 80, state: "ESTABLISHED" },
  { id: 3, srcIP: "192.168.1.10", srcPort: 49154, dstIP: "151.101.1.140", dstPort: 443, state: "SYN_SENT" },
  { id: 4, srcIP: "10.0.0.2", srcPort: 3200, dstIP: "203.0.113.1", dstPort: 50001, state: "NAT映射" },
];

export function PortMuxSimulator() {
  const [scenario, setScenario] = useState(0);
  const [highlightConn, setHighlightConn] = useState<number | null>(null);
  const sc = scenarios[scenario];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">端口复用模拟器</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {scenarios.map((s, i) => (
          <button key={i} onClick={() => setScenario(i)}
            className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${scenario === i ? "bg-sky-500/20 border-sky-400/60 text-sky-700 dark:text-sky-300" : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}>
            {s.name}
          </button>
        ))}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="text-sm font-medium text-text-primary mb-1">{sc.name}</div>
        <div className="text-xs text-text-secondary mb-3">{sc.desc}</div>
        <div className="space-y-2">
          {sc.sockets.map((s, i) => (
            <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-elevated border border-border-subtle">
              <span className="w-2 h-2 rounded-full bg-sky-500" />
              <span className="text-xs font-mono text-text-primary">{s}</span>
              <span className="ml-auto text-[10px] text-emerald-500 dark:text-emerald-400">LISTEN</span>
            </div>
          ))}
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4">
        <div className="text-xs font-medium text-text-primary mb-2">活跃连接表</div>
        <div className="space-y-1.5">
          {connections.map((c) => (
            <div key={c.id} onClick={() => setHighlightConn(c.id)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono cursor-pointer transition-all ${highlightConn === c.id ? "bg-sky-500/15 border border-sky-400/40" : "bg-bg-elevated border border-border-subtle hover:border-sky-400/30"}`}>
              <span className="text-text-primary">{c.srcIP}:{c.srcPort}</span>
              <span className="text-text-tertiary">→</span>
              <span className="text-text-primary">{c.dstIP}:{c.dstPort}</span>
              <span className={`ml-auto px-1.5 py-0.5 rounded text-[10px] ${c.state === "ESTABLISHED" ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400" : c.state === "NAT映射" ? "bg-violet-500/15 text-violet-600 dark:text-violet-400" : "bg-amber-500/15 text-amber-600 dark:text-amber-400"}`}>
                {c.state}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-3 text-[10px] text-text-tertiary">点击连接查看四元组详情 · 端口复用允许多个连接共享同一本地端口</div>
      </div>
    </div>
  );
}
export default PortMuxSimulator;
