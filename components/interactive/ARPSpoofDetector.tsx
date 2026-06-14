"use client";
import { useState } from "react";

interface Binding {
  ip: string;
  mac: string;
  isDuplicate: boolean;
}

const initialBindings: Binding[] = [
  { ip: "192.168.1.1", mac: "AA:BB:CC:DD:EE:01", isDuplicate: false },
  { ip: "192.168.1.10", mac: "AA:BB:CC:DD:EE:0A", isDuplicate: false },
  { ip: "192.168.1.1", mac: "11:22:33:44:55:66", isDuplicate: true },
  { ip: "192.168.1.20", mac: "AA:BB:CC:DD:EE:14", isDuplicate: false },
];

export function ARPSpoofDetector() {
  const [bindings, setBindings] = useState<Binding[]>(initialBindings);
  const [newIP, setNewIP] = useState("");
  const [newMAC, setNewMAC] = useState("");
  const [alerts, setAlerts] = useState<string[]>([]);

  const detectSpoof = () => {
    const ipMacMap = new Map<string, Set<string>>();
    for (const b of bindings) {
      if (!ipMacMap.has(b.ip)) ipMacMap.set(b.ip, new Set());
      ipMacMap.get(b.ip)!.add(b.mac);
    }
    const newAlerts: string[] = [];
    ipMacMap.forEach((macs, ip) => {
      if (macs.size > 1) {
        newAlerts.push(`⚠️ 检测到 IP ${ip} 绑定了 ${macs.size} 个不同的 MAC 地址，可能存在 ARP 欺骗！`);
      }
    });
    setAlerts(newAlerts);
  };

  const addBinding = () => {
    if (!newIP || !newMAC) return;
    const isDup = bindings.some((b) => b.ip === newIP && b.mac !== newMAC);
    setBindings([...bindings, { ip: newIP, mac: newMAC, isDuplicate: isDup }]);
    setNewIP("");
    setNewMAC("");
  };

  const removeBinding = (idx: number) => {
    setBindings(bindings.filter((_, i) => i !== idx));
    setAlerts([]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">ARP 欺骗检测器</h3>
      <div className="flex gap-2 mb-4">
        <input value={newIP} onChange={(e) => setNewIP(e.target.value)} placeholder="IP 地址" className="flex-1 px-3 py-1.5 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 text-text-primary text-sm" />
        <input value={newMAC} onChange={(e) => setNewMAC(e.target.value)} placeholder="MAC 地址" className="flex-1 px-3 py-1.5 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 text-text-primary text-sm" />
        <button onClick={addBinding} className="px-3 py-1.5 rounded bg-blue-500 text-white text-sm">添加</button>
        <button onClick={detectSpoof} className="px-3 py-1.5 rounded bg-red-500 text-white text-sm">检测</button>
      </div>
      <div className="mb-4 p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <table className="w-full text-sm">
          <thead><tr className="text-text-secondary text-xs"><th className="text-left py-1">IP</th><th className="text-left py-1">MAC</th><th className="text-left py-1">状态</th><th className="text-right py-1">操作</th></tr></thead>
          <tbody>
            {bindings.map((b, i) => (
              <tr key={i} className="border-t border-border-subtle">
                <td className="py-1 font-mono text-text-primary">{b.ip}</td>
                <td className="py-1 font-mono text-text-primary">{b.mac}</td>
                <td className="py-1"><span className={`px-2 py-0.5 rounded text-xs ${b.isDuplicate ? "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300" : "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300"}`}>{b.isDuplicate ? "冲突" : "正常"}</span></td>
                <td className="py-1 text-right"><button onClick={() => removeBinding(i)} className="text-xs text-red-500 hover:underline">删除</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {alerts.length > 0 && (
        <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700 space-y-1">
          {alerts.map((a, i) => <p key={i} className="text-sm text-red-700 dark:text-red-300">{a}</p>)}
        </div>
      )}
      {alerts.length === 0 && bindings.some((b) => b.isDuplicate) && (
        <p className="text-sm text-green-600 dark:text-green-400 mt-2">✓ 未检测到 ARP 欺骗</p>
      )}
    </div>
  );
}
export default ARPSpoofDetector;
