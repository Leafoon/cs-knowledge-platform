"use client";
import { useState } from "react";

interface NatEntry {
  id: number;
  internalIP: string;
  internalPort: number;
  externalIP: string;
  externalPort: number;
  protocol: string;
  status: string;
}

const publicIP = "203.0.113.1";

const initialEntries: NatEntry[] = [
  { id: 1, internalIP: "192.168.1.10", internalPort: 50001, externalIP: publicIP, externalPort: 40001, protocol: "TCP", status: "active" },
  { id: 2, internalIP: "192.168.1.11", internalPort: 50002, externalIP: publicIP, externalPort: 40002, protocol: "TCP", status: "active" },
  { id: 3, internalIP: "192.168.1.12", internalPort: 53000, externalIP: publicIP, externalPort: 40003, protocol: "UDP", status: "active" },
];

let nextPort = 40004;

export function NATSimulator() {
  const [entries, setEntries] = useState<NatEntry[]>(initialEntries);
  const [srcIP, setSrcIP] = useState("192.168.1.15");
  const [srcPort, setSrcPort] = useState(51000);
  const [protocol, setProtocol] = useState("TCP");
  const [log, setLog] = useState<string[]>([]);

  const translate = () => {
    const existing = entries.find(
      (e) => e.internalIP === srcIP && e.internalPort === srcPort && e.protocol === protocol
    );
    if (existing) {
      setLog([...log, `已存在映射: ${srcIP}:${srcPort} → ${existing.externalIP}:${existing.externalPort}`]);
      return;
    }
    const newEntry: NatEntry = {
      id: entries.length + 1,
      internalIP: srcIP,
      internalPort: srcPort,
      externalIP: publicIP,
      externalPort: nextPort,
      protocol,
      status: "active",
    };
    nextPort++;
    setEntries([...entries, newEntry]);
    setLog([...log, `新建映射: ${srcIP}:${srcPort} → ${publicIP}:${newEntry.externalPort} (${protocol})`]);
  };

  const removeEntry = (id: number) => {
    setEntries(entries.filter((e) => e.id !== id));
    setLog([...log, `删除映射 ID=${id}`]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        NAT Simulator <span className="text-text-secondary text-sm">— 地址转换模拟器</span>
      </h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-xs text-text-secondary block mb-1">内部IP</label>
          <input
            type="text"
            value={srcIP}
            onChange={(e) => setSrcIP(e.target.value)}
            className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-primary text-sm font-mono"
          />
        </div>
        <div>
          <label className="text-xs text-text-secondary block mb-1">内部端口</label>
          <input
            type="number"
            value={srcPort}
            onChange={(e) => setSrcPort(parseInt(e.target.value))}
            className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-primary text-sm font-mono"
          />
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        {["TCP", "UDP"].map((p) => (
          <button
            key={p}
            onClick={() => setProtocol(p)}
            className={`px-3 py-1 rounded text-sm ${protocol === p ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {p}
          </button>
        ))}
        <button onClick={translate} className="px-4 py-1 rounded bg-green-600 text-white text-sm">
          转换
        </button>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left p-1 text-text-secondary">内部地址</th>
              <th className="text-left p-1 text-text-secondary">外部地址</th>
              <th className="text-left p-1 text-text-secondary">协议</th>
              <th className="text-left p-1 text-text-secondary">操作</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.id} className="border-b border-border-subtle">
                <td className="p-1 font-mono text-text-primary">{e.internalIP}:{e.internalPort}</td>
                <td className="p-1 font-mono text-text-primary">{e.externalIP}:{e.externalPort}</td>
                <td className="p-1 text-text-secondary">{e.protocol}</td>
                <td className="p-1">
                  <button onClick={() => removeEntry(e.id)} className="text-red-500 hover:text-red-700">
                    删除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {log.length > 0 && (
        <div className="bg-gray-900 p-3 rounded text-xs font-mono max-h-32 overflow-y-auto">
          {log.map((l, i) => (
            <div key={i} className="text-green-400">{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export default NATSimulator;
