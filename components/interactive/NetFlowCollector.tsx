"use client";
import { useState } from "react";

interface FlowRecord {
  srcIP: string;
  dstIP: string;
  srcPort: number;
  dstPort: number;
  proto: string;
  packets: number;
  bytes: number;
  startTime: string;
  duration: number;
  status: "active" | "expired" | "exported";
}

const initialFlows: FlowRecord[] = [
  { srcIP: "10.0.1.5", dstIP: "172.16.0.10", srcPort: 48230, dstPort: 443, proto: "TCP", packets: 152, bytes: 234567, startTime: "14:01:23", duration: 35, status: "active" },
  { srcIP: "192.168.1.100", dstIP: "8.8.8.8", srcPort: 53124, dstPort: 53, proto: "UDP", packets: 2, bytes: 156, startTime: "14:02:01", duration: 0, status: "expired" },
  { srcIP: "10.0.2.15", dstIP: "10.0.3.20", srcPort: 51200, dstPort: 80, proto: "TCP", packets: 87, bytes: 45678, startTime: "14:02:15", duration: 12, status: "active" },
  { srcIP: "10.0.1.5", dstIP: "10.0.1.1", srcPort: 0, dstPort: 0, proto: "ICMP", packets: 5, bytes: 420, startTime: "14:03:00", duration: 8, status: "exported" },
];

export function NetFlowCollector() {
  const [flows, setFlows] = useState<FlowRecord[]>(initialFlows);
  const [activeTab, setActiveTab] = useState<"table" | "stats">("table");
  const [exportInterval, setExportInterval] = useState(300);

  const totalPackets = flows.reduce((s, f) => s + f.packets, 0);
  const totalBytes = flows.reduce((s, f) => s + f.bytes, 0);
  const tcpFlows = flows.filter((f) => f.proto === "TCP").length;
  const udpFlows = flows.filter((f) => f.proto === "UDP").length;

  const exportFlows = () => {
    setFlows((f) => f.map((fl) => ({ ...fl, status: "exported" as const })));
  };

  const addRandomFlow = () => {
    const protos = ["TCP", "UDP", "ICMP"];
    const proto = protos[Math.floor(Math.random() * 3)];
    const flow: FlowRecord = {
      srcIP: `10.0.${Math.floor(Math.random() * 4)}.${Math.floor(Math.random() * 255)}`,
      dstIP: `172.16.${Math.floor(Math.random() * 4)}.${Math.floor(Math.random() * 255)}`,
      srcPort: Math.floor(Math.random() * 60000) + 1024,
      dstPort: [80, 443, 53, 22, 3306][Math.floor(Math.random() * 5)],
      proto,
      packets: Math.floor(Math.random() * 500) + 1,
      bytes: Math.floor(Math.random() * 1000000) + 100,
      startTime: `14:${String(Math.floor(Math.random() * 60)).padStart(2, "0")}:${String(Math.floor(Math.random() * 60)).padStart(2, "0")}`,
      duration: Math.floor(Math.random() * 120),
      status: "active",
    };
    setFlows((f) => [...f, flow]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">NetFlow 流量采集器</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setActiveTab("table")} className={`px-3 py-1.5 rounded text-sm ${activeTab === "table" ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>流记录表</button>
        <button onClick={() => setActiveTab("stats")} className={`px-3 py-1.5 rounded text-sm ${activeTab === "stats" ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>统计</button>
        <button onClick={addRandomFlow} className="px-3 py-1.5 rounded bg-green-500 text-white text-sm">+ 新流</button>
        <button onClick={exportFlows} className="px-3 py-1.5 rounded bg-purple-500 text-white text-sm">导出全部</button>
      </div>
      {activeTab === "table" ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead><tr className="text-text-secondary"><th className="text-left py-1 pr-2">源 IP</th><th className="text-left py-1 pr-2">目的 IP</th><th className="text-left py-1 pr-2">端口</th><th className="text-left py-1 pr-2">协议</th><th className="text-right py-1 pr-2">包数</th><th className="text-right py-1 pr-2">字节</th><th className="text-left py-1">状态</th></tr></thead>
            <tbody>
              {flows.map((f, i) => (
                <tr key={i} className="border-t border-border-subtle">
                  <td className="py-1 pr-2 font-mono text-text-primary">{f.srcIP}:{f.srcPort}</td>
                  <td className="py-1 pr-2 font-mono text-text-primary">{f.dstIP}:{f.dstPort}</td>
                  <td className="py-1 pr-2 text-text-secondary">{f.dstPort}</td>
                  <td className="py-1 pr-2"><span className={`px-1.5 py-0.5 rounded text-xs ${f.proto === "TCP" ? "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300" : f.proto === "UDP" ? "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>{f.proto}</span></td>
                  <td className="py-1 pr-2 text-right text-text-primary">{f.packets}</td>
                  <td className="py-1 pr-2 text-right text-text-primary">{(f.bytes / 1024).toFixed(1)}K</td>
                  <td className="py-1"><span className={`px-1.5 py-0.5 rounded text-xs ${f.status === "active" ? "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300" : f.status === "expired" ? "bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300" : "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300"}`}>{f.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[{ l: "总流数", v: flows.length }, { l: "总包数", v: totalPackets.toLocaleString() }, { l: "总字节", v: `${(totalBytes / 1048576).toFixed(2)} MB` }, { l: "导出间隔", v: `${exportInterval}s` }, { l: "TCP 流", v: tcpFlows }, { l: "UDP 流", v: udpFlows }, { l: "活跃流", v: flows.filter((f) => f.status === "active").length }, { l: "已导出", v: flows.filter((f) => f.status === "exported").length }].map((s, i) => (
            <div key={i} className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle text-center"><span className="block text-lg font-bold text-text-primary">{s.v}</span><span className="text-xs text-text-secondary">{s.l}</span></div>
          ))}
          <div className="col-span-2 md:col-span-4">
            <label className="text-xs text-text-secondary">导出间隔: {exportInterval}s</label>
            <input type="range" min={60} max={900} step={60} value={exportInterval} onChange={(e) => setExportInterval(Number(e.target.value))} className="w-full mt-1" />
          </div>
        </div>
      )}
    </div>
  );
}
export default NetFlowCollector;
