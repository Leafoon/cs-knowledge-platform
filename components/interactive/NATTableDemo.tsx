"use client";
import { useState } from "react";

interface NatEntry {
  id: number;
  type: string;
  internalIP: string;
  internalPort: number;
  externalIP: string;
  externalPort: number;
  protocol: string;
  timeout: number;
}

const initialEntries: NatEntry[] = [
  { id: 1, type: "NAPT", internalIP: "192.168.1.10", internalPort: 50001, externalIP: "203.0.113.1", externalPort: 40001, protocol: "TCP", timeout: 300 },
  { id: 2, type: "NAPT", internalIP: "192.168.1.11", internalPort: 50002, externalIP: "203.0.113.1", externalPort: 40002, protocol: "TCP", timeout: 300 },
  { id: 3, type: "NAPT", internalIP: "192.168.1.10", internalPort: 53000, externalIP: "203.0.113.1", externalPort: 40003, protocol: "UDP", timeout: 60 },
];

const natTypes = [
  { name: "静态NAT", en: "Static NAT", desc: "一对一固定映射，内部IP永久对应外部IP", ratio: "1:1" },
  { name: "动态NAT", en: "Dynamic NAT", desc: "从地址池动态分配，用完释放", ratio: "N:M" },
  { name: "NAPT", en: "Network Address Port Translation", desc: "多对一映射，用端口号区分不同主机", ratio: "N:1" },
];

export function NATTableDemo() {
  const [entries, setEntries] = useState<NatEntry[]>(initialEntries);
  const [natType, setNatType] = useState(2);
  const [newInternal, setNewInternal] = useState("192.168.1.20");
  const [newPort, setNewPort] = useState(52000);

  let nextExtPort = 40010;

  const addEntry = () => {
    const extPort = nextExtPort + entries.length;
    setEntries([
      ...entries,
      {
        id: entries.length + 1,
        type: natTypes[natType].name,
        internalIP: newInternal,
        internalPort: newPort,
        externalIP: "203.0.113.1",
        externalPort: extPort,
        protocol: "TCP",
        timeout: 300,
      },
    ]);
  };

  const removeEntry = (id: number) => {
    setEntries(entries.filter((e) => e.id !== id));
  };

  const refreshTimeout = () => {
    setEntries(entries.map((e) => ({ ...e, timeout: e.timeout > 0 ? e.timeout - 10 : 0 })));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        NAT Table Demo <span className="text-text-secondary text-sm">— NAT转换表管理</span>
      </h3>
      <div className="flex gap-2 mb-4">
        {natTypes.map((t, i) => (
          <button
            key={i}
            onClick={() => setNatType(i)}
            className={`px-3 py-1 rounded text-sm ${natType === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {t.name}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded mb-4 text-sm">
        <div className="font-semibold text-text-primary">{natTypes[natType].en}</div>
        <div className="text-text-secondary">{natTypes[natType].desc}</div>
        <div className="text-xs text-text-secondary mt-1">映射比: {natTypes[natType].ratio}</div>
      </div>
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={newInternal}
          onChange={(e) => setNewInternal(e.target.value)}
          placeholder="内部IP"
          className="flex-1 p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-primary text-sm font-mono"
        />
        <input
          type="number"
          value={newPort}
          onChange={(e) => setNewPort(parseInt(e.target.value))}
          placeholder="端口"
          className="w-24 p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-primary text-sm font-mono"
        />
        <button onClick={addEntry} className="px-3 py-2 rounded bg-green-600 text-white text-sm">
          添加
        </button>
        <button onClick={refreshTimeout} className="px-3 py-2 rounded bg-yellow-600 text-white text-sm">
          刷新超时
        </button>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left p-1 text-text-secondary">内部地址:端口</th>
              <th className="text-left p-1 text-text-secondary">外部地址:端口</th>
              <th className="text-left p-1 text-text-secondary">协议</th>
              <th className="text-left p-1 text-text-secondary">超时(s)</th>
              <th className="text-left p-1 text-text-secondary">操作</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.id} className={`border-b border-border-subtle ${e.timeout <= 30 ? "bg-red-50 dark:bg-red-900/20" : ""}`}>
                <td className="p-1 font-mono text-text-primary">{e.internalIP}:{e.internalPort}</td>
                <td className="p-1 font-mono text-text-primary">{e.externalIP}:{e.externalPort}</td>
                <td className="p-1 text-text-secondary">{e.protocol}</td>
                <td className="p-1 text-text-secondary">{e.timeout}</td>
                <td className="p-1">
                  <button onClick={() => removeEntry(e.id)} className="text-red-500 hover:text-red-700">删除</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-xs text-text-secondary mt-2">共 {entries.length} 条映射记录</div>
    </div>
  );
}

export default NATTableDemo;
