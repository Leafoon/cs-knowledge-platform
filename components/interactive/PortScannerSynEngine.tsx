"use client";
import { useState, useEffect, useCallback, useRef } from "react";

type PortState = "unknown" | "scanning" | "open" | "closed" | "filtered";

interface PortInfo {
  port: number;
  service: string;
  state: PortState;
}

const defaultPorts: { port: number; service: string }[] = [
  { port: 21, service: "FTP" }, { port: 22, service: "SSH" }, { port: 23, service: "Telnet" },
  { port: 25, service: "SMTP" }, { port: 80, service: "HTTP" }, { port: 443, service: "HTTPS" },
  { port: 3306, service: "MySQL" }, { port: 5432, service: "PostgreSQL" }, { port: 8080, service: "HTTP-Alt" },
  { port: 6379, service: "Redis" }, { port: 27017, service: "MongoDB" }, { port: 3389, service: "RDP" },
];

export function PortScannerSynEngine() {
  const [ports, setPorts] = useState<PortInfo[]>(defaultPorts.map((p) => ({ ...p, state: "unknown" as PortState })));
  const [scanning, setScanning] = useState(false);
  const [scanLog, setScanLog] = useState<string[]>([]);
  const [scanIndex, setScanIndex] = useState(-1);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stateColors: Record<PortState, string> = {
    unknown: "bg-bg-tertiary text-text-secondary",
    scanning: "bg-amber-500/15 text-amber-600 dark:text-amber-400 animate-pulse",
    open: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
    closed: "bg-red-500/15 text-red-600 dark:text-red-400",
    filtered: "bg-violet-500/15 text-violet-600 dark:text-violet-400",
  };

  const stateLabels: Record<PortState, string> = {
    unknown: "未扫描", scanning: "扫描中...", open: "OPEN", closed: "CLOSED", filtered: "FILTERED",
  };

  const handleStart = useCallback(() => {
    setPorts(defaultPorts.map((p) => ({ ...p, state: "unknown" as PortState })));
    setScanLog([]);
    setScanIndex(0);
    setScanning(true);
  }, []);

  useEffect(() => {
    if (!scanning || scanIndex < 0) return;
    if (scanIndex >= defaultPorts.length) {
      setScanning(false);
      setScanLog((l) => [...l, "── 扫描完成 ──"]);
      return;
    }
    const timer = setTimeout(() => {
      const port = defaultPorts[scanIndex];
      setPorts((prev) => prev.map((p, i) => i === scanIndex ? { ...p, state: "scanning" } : p));
      setScanLog((l) => [...l, `[→] SYN → ${port.port} (${port.service})`]);

      setTimeout(() => {
        const result: PortState = Math.random() < 0.3 ? "open" : Math.random() < 0.7 ? "closed" : "filtered";
        const flag = result === "open" ? "SYN/ACK" : result === "closed" ? "RST" : "无响应";
        setPorts((prev) => prev.map((p, i) => i === scanIndex ? { ...p, state: result } : p));
        setScanLog((l) => [...l, `[←] ${flag} ← 端口${port.port} → ${result.toUpperCase()}`]);
        setScanIndex((i) => i + 1);
      }, 400);
    }, 200);
    return () => clearTimeout(timer);
  }, [scanning, scanIndex]);

  const openCount = ports.filter((p) => p.state === "open").length;
  const closedCount = ports.filter((p) => p.state === "closed").length;
  const filteredCount = ports.filter((p) => p.state === "filtered").length;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">SYN端口扫描引擎</h3>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={handleStart} disabled={scanning}
          className="px-4 py-1.5 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-sm font-medium hover:bg-sky-500/25 disabled:opacity-50 transition-colors">
          {scanning ? "扫描中..." : "开始SYN扫描"}
        </button>
        <div className="flex gap-2 text-xs">
          <span className="text-emerald-600 dark:text-emerald-400">开放 {openCount}</span>
          <span className="text-red-500">关闭 {closedCount}</span>
          <span className="text-violet-500">过滤 {filteredCount}</span>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-4">
        {ports.map((p) => (
          <div key={p.port} className={`px-3 py-2 rounded-lg border border-border-subtle text-xs transition-all ${stateColors[p.state]}`}>
            <div className="flex items-center justify-between">
              <span className="font-mono font-medium">{p.port}</span>
              <span className="text-[10px] opacity-70">{p.service}</span>
            </div>
            <div className="text-[10px] mt-0.5 font-medium">{stateLabels[p.state]}</div>
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 max-h-32 overflow-y-auto">
        <div className="text-[10px] font-mono space-y-0.5">
          {scanLog.length === 0 ? <span className="text-text-tertiary">等待扫描...</span> : scanLog.map((l, i) => (
            <div key={i} className={l.includes("OPEN") ? "text-emerald-500" : l.includes("CLOSED") ? "text-red-400" : l.includes("FILTERED") ? "text-violet-400" : "text-text-secondary"}>{l}</div>
          ))}
        </div>
      </div>
      <div className="mt-3 text-[10px] text-text-tertiary">SYN半连接扫描：发送SYN包，根据响应判断端口状态（不完成三次握手，隐蔽性较高）</div>
    </div>
  );
}
export default PortScannerSynEngine;
