"use client";
import { useState, useEffect, useRef } from "react";

interface Flow { srcIP: string; dstIP: string; srcPort: number; dstPort: number; proto: string; bytes: number; packets: number; anomaly?: string }

const genFlow = (): Flow => {
  const protos = ["TCP", "UDP", "ICMP"];
  const proto = protos[Math.floor(Math.random() * 3)];
  const anomalous = Math.random() < 0.15;
  return {
    srcIP: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
    dstIP: `10.0.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
    srcPort: Math.floor(Math.random() * 65535),
    dstPort: anomalous ? Math.floor(Math.random() * 100) : [80, 443, 22, 53, 8080][Math.floor(Math.random() * 5)],
    proto,
    bytes: anomalous ? Math.floor(Math.random() * 10000000) : Math.floor(Math.random() * 50000),
    packets: anomalous ? Math.floor(Math.random() * 50000) : Math.floor(Math.random() * 100),
    anomaly: anomalous ? (Math.random() > 0.5 ? "DDoS 流量突增" : "端口扫描") : undefined,
  };
};

export function TrafficAnalysisEngine() {
  const [flows, setFlows] = useState<Flow[]>([]);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState({ total: 0, anomalies: 0, bytes: 0 });
  const timerRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(() => {
        const newFlows = Array.from({ length: 3 }, genFlow);
        setFlows((f) => [...newFlows, ...f].slice(0, 15));
        setStats((s) => ({
          total: s.total + newFlows.length,
          anomalies: s.anomalies + newFlows.filter((f) => f.anomaly).length,
          bytes: s.bytes + newFlows.reduce((sum, f) => sum + f.bytes, 0),
        }));
      }, 1500);
      return () => clearInterval(timerRef.current);
    }
  }, [running]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">流量分析引擎 (NetFlow/sFlow)</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setRunning(!running)}
          className={`px-4 py-2 rounded text-sm font-medium ${running ? "bg-red-500 text-white" : "bg-green-500 text-white"}`}>
          {running ? "停止采集" : "开始采集"}
        </button>
        <button onClick={() => { setFlows([]); setStats({ total: 0, anomalies: 0, bytes: 0 }); }}
          className="px-3 py-2 rounded border border-border-subtle text-text-muted text-sm">清空</button>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-3 rounded bg-blue-500/10 border border-blue-400/30 text-center">
          <span className="text-blue-400 text-xl font-bold">{stats.total}</span><span className="text-text-muted text-xs block">总流数</span>
        </div>
        <div className="p-3 rounded bg-red-500/10 border border-red-400/30 text-center">
          <span className="text-red-400 text-xl font-bold">{stats.anomalies}</span><span className="text-text-muted text-xs block">异常流</span>
        </div>
        <div className="p-3 rounded bg-green-500/10 border border-green-400/30 text-center">
          <span className="text-green-400 text-xl font-bold">{(stats.bytes / 1000000).toFixed(1)}MB</span><span className="text-text-muted text-xs block">总流量</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left py-2 text-text-muted">源IP</th>
              <th className="text-left py-2 text-text-muted">目的IP</th>
              <th className="text-left py-2 text-text-muted">端口</th>
              <th className="text-left py-2 text-text-muted">协议</th>
              <th className="text-right py-2 text-text-muted">字节</th>
              <th className="text-right py-2 text-text-muted">包数</th>
              <th className="text-left py-2 text-text-muted">状态</th>
            </tr>
          </thead>
          <tbody>
            {flows.map((f, i) => (
              <tr key={i} className={`border-b border-border-subtle ${f.anomaly ? "bg-red-500/5" : ""}`}>
                <td className="py-1.5 font-mono text-text-primary">{f.srcIP}</td>
                <td className="py-1.5 font-mono text-text-primary">{f.dstIP}</td>
                <td className="py-1.5 text-text-secondary">{f.srcPort}→{f.dstPort}</td>
                <td className="py-1.5 text-text-secondary">{f.proto}</td>
                <td className="py-1.5 text-right text-text-primary">{f.bytes.toLocaleString()}</td>
                <td className="py-1.5 text-right text-text-primary">{f.packets}</td>
                <td className="py-1.5">{f.anomaly ? <span className="text-red-400 font-medium">{f.anomaly}</span> : <span className="text-green-400">正常</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {flows.length === 0 && <p className="text-text-muted text-sm text-center py-4">点击开始采集观察流量数据</p>}
    </div>
  );
}
export default TrafficAnalysisEngine;
