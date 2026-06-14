"use client";
import { useState } from "react";

interface ArpEntry {
  ip: string;
  mac: string;
  status: "complete" | "pending";
}

const initialEntries: ArpEntry[] = [
  { ip: "192.168.1.1", mac: "AA:BB:CC:DD:EE:01", status: "complete" },
  { ip: "192.168.1.10", mac: "AA:BB:CC:DD:EE:0A", status: "complete" },
];

export function ARPDemo() {
  const [cache, setCache] = useState<ArpEntry[]>(initialEntries);
  const [targetIP, setTargetIP] = useState("192.168.1.20");
  const [step, setStep] = useState<"idle" | "request" | "response" | "done">("idle");
  const [log, setLog] = useState<string[]>([]);

  const sendRequest = () => {
    if (cache.find((e) => e.ip === targetIP && e.status === "complete")) {
      setLog((l) => [`缓存命中: ${targetIP} → ${cache.find((e) => e.ip === targetIP)?.mac}`, ...l]);
      return;
    }
    setCache((c) => [...c, { ip: targetIP, mac: "??:??:??:??:??:??", status: "pending" }]);
    setStep("request");
    setLog((l) => [`ARP 请求广播: Who has ${targetIP}? Tell 192.168.1.100`, ...l]);

    setTimeout(() => {
      setStep("response");
      const mac = `AA:BB:CC:DD:${Math.floor(Math.random() * 256).toString(16).padStart(2, "0").toUpperCase()}:${Math.floor(Math.random() * 256).toString(16).padStart(2, "0").toUpperCase()}`;
      setLog((l) => [`ARP 响应单播: ${targetIP} is at ${mac}`, ...l]);

      setTimeout(() => {
        setCache((c) => c.map((e) => e.ip === targetIP ? { ...e, mac, status: "complete" } : e));
        setStep("done");
        setLog((l) => [`缓存已更新: ${targetIP} → ${mac}`, ...l]);
      }, 800);
    }, 1200);
  };

  const reset = () => {
    setCache(initialEntries);
    setStep("idle");
    setLog([]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">ARP 协议演示</h3>
      <div className="flex gap-2 mb-4">
        <input value={targetIP} onChange={(e) => setTargetIP(e.target.value)} placeholder="目标 IP" className="flex-1 px-3 py-1.5 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 text-text-primary text-sm" />
        <button onClick={sendRequest} disabled={step === "request" || step === "response"} className="px-4 py-1.5 rounded bg-blue-500 text-white text-sm disabled:opacity-50">发送 ARP 请求</button>
        <button onClick={reset} className="px-3 py-1.5 rounded bg-gray-200 dark:bg-gray-700 text-text-secondary text-sm">重置</button>
      </div>
      <div className="mb-4 p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <p className="text-sm font-medium text-text-primary mb-2">ARP 缓存表</p>
        <table className="w-full text-sm">
          <thead><tr className="text-text-secondary text-xs"><th className="text-left py-1">IP 地址</th><th className="text-left py-1">MAC 地址</th><th className="text-left py-1">状态</th></tr></thead>
          <tbody>
            {cache.map((e, i) => (
              <tr key={i} className={`border-t border-border-subtle ${e.ip === targetIP && step !== "idle" ? "bg-yellow-50 dark:bg-yellow-900/20" : ""}`}>
                <td className="py-1 font-mono text-text-primary">{e.ip}</td>
                <td className="py-1 font-mono text-text-primary">{e.mac}</td>
                <td className="py-1"><span className={`px-2 py-0.5 rounded text-xs ${e.status === "complete" ? "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300" : "bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300"}`}>{e.status === "complete" ? "已解析" : "待解析"}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-xs text-text-secondary max-h-28 overflow-y-auto">
        {log.length === 0 ? "点击发送 ARP 请求开始演示" : log.map((l, i) => <div key={i} className="py-0.5">{l}</div>)}
      </div>
      {step !== "idle" && (
        <div className="mt-3 flex items-center gap-2 text-xs">
          <span className={`px-2 py-1 rounded ${step === "request" ? "bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>① 广播请求</span>
          <span>→</span>
          <span className={`px-2 py-1 rounded ${step === "response" ? "bg-blue-500 text-white" : step === "done" ? "bg-gray-200 dark:bg-gray-700 text-text-secondary" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>② 单播响应</span>
          <span>→</span>
          <span className={`px-2 py-1 rounded ${step === "done" ? "bg-green-500 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>③ 缓存更新</span>
        </div>
      )}
    </div>
  );
}
export default ARPDemo;
