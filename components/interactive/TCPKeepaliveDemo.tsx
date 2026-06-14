"use client";
import { useState, useEffect } from "react";

export function TCPKeepaliveDemo() {
  const [status, setStatus] = useState<"idle" | "probing" | "alive" | "dead">("idle");
  const [probeCount, setProbeCount] = useState(0);
  const [maxProbes, setMaxProbes] = useState(5);
  const [interval, setInterval_] = useState(75);
  const [idleTime, setIdleTime] = useState(120);
  const [log, setLog] = useState<string[]>([]);
  const [peerAlive, setPeerAlive] = useState(true);

  const addLog = (msg: string) => setLog((l) => [...l.slice(-10), `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const startKeepalive = () => {
    setStatus("probing");
    setProbeCount(0);
    addLog(`空闲 ${idleTime}s 后启动 Keepalive 探测`);
    addLog(`发送 Keepalive probe #1`);
  };

  const sendProbe = () => {
    if (status !== "probing") return;
    const newCount = probeCount + 1;
    setProbeCount(newCount);
    if (peerAlive) {
      setStatus("alive");
      addLog(`收到 ACK 响应 → 连接存活`);
      setTimeout(() => { setStatus("idle"); setProbeCount(0); }, 1500);
    } else if (newCount >= maxProbes) {
      setStatus("dead");
      addLog(`超过最大探测次数 (${maxProbes}) → 连接断开！`);
    } else {
      addLog(`无响应 → 发送 Keepalive probe #${newCount + 1}`);
    }
  };

  useEffect(() => {
    if (status !== "probing") return;
    const timer = setTimeout(sendProbe, interval * 20);
    return () => clearTimeout(timer);
  }, [status, probeCount]);

  const reset = () => { setStatus("idle"); setProbeCount(0); setLog([]); };

  const statusConfig = {
    idle: { label: "空闲 (Idle)", color: "text-gray-400 bg-gray-500/10 border-gray-400/30" },
    probing: { label: "探测中 (Probing)", color: "text-yellow-400 bg-yellow-500/10 border-yellow-400/30" },
    alive: { label: "存活 (Alive)", color: "text-green-400 bg-green-500/10 border-green-400/30" },
    dead: { label: "断开 (Dead)", color: "text-red-400 bg-red-500/10 border-red-400/30" },
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCP Keepalive 演示 (TCP Keepalive Demo)</h3>
      <div className="flex flex-wrap gap-4 mb-4">
        <label className="text-text-secondary text-sm">空闲时间:
          <input type="range" min="30" max="600" step="30" value={idleTime} onChange={(e) => setIdleTime(+e.target.value)} className="ml-2 w-24 align-middle" />
          <span className="ml-1 text-text-primary font-mono">{idleTime}s</span>
        </label>
        <label className="text-text-secondary text-sm">探测间隔:
          <input type="range" min="10" max="300" step="10" value={interval} onChange={(e) => setInterval_(+e.target.value)} className="ml-2 w-24 align-middle" />
          <span className="ml-1 text-text-primary font-mono">{interval}s</span>
        </label>
        <label className="text-text-secondary text-sm">最大探测:
          <input type="range" min="1" max="10" value={maxProbes} onChange={(e) => setMaxProbes(+e.target.value)} className="ml-2 w-20 align-middle" />
          <span className="ml-1 text-text-primary font-mono">{maxProbes}</span>
        </label>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={startKeepalive} disabled={status !== "idle"} className="px-4 py-2 rounded bg-blue-500 text-white text-sm hover:bg-blue-600 disabled:opacity-40 transition-colors">启动探测</button>
        <label className="flex items-center gap-2 text-text-secondary text-sm">
          <input type="checkbox" checked={peerAlive} onChange={(e) => setPeerAlive(e.target.checked)} className="rounded" />
          对端存活
        </label>
        <button onClick={reset} className="px-3 py-2 rounded border border-border-subtle text-text-muted text-sm hover:text-text-primary transition-colors">重置</button>
      </div>
      <div className={`mb-4 p-3 rounded-lg border ${statusConfig[status].color}`}>
        <span className="text-sm font-medium">{statusConfig[status].label}</span>
        {status === "probing" && <span className="ml-2 text-xs">探测 #{probeCount}/{maxProbes}</span>}
      </div>
      <div className="mb-4 p-3 rounded bg-bg-primary border border-border-subtle max-h-32 overflow-y-auto">
        {log.length === 0 ? (
          <p className="text-text-muted text-xs text-center py-2">日志为空，点击"启动探测"开始</p>
        ) : (
          log.map((l, i) => <p key={i} className="text-text-secondary text-xs font-mono py-0.5">{l}</p>)
        )}
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <p className="text-text-muted text-xs">TCP Keepalive 在连接空闲 idleTime 后发送探测包。每 interval 秒发一次，超过 maxProbes 次无响应则认为连接已死。Linux 默认：idle=7200s, interval=75s, count=9。</p>
      </div>
    </div>
  );
}
export default TCPKeepaliveDemo;
