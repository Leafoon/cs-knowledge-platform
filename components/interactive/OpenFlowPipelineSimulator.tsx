"use client";
import { useState } from "react";

const defaultTables = [
  {
    id: 0, name: "Table 0 (Ingress)",
    entries: [
      { match: "eth_type=0x0800, ip_dst=10.0.0.1", action: "goto Table 1", priority: 100 },
      { match: "eth_type=0x0806", action: "output:CONTROLLER", priority: 200 },
      { match: "default", action: "drop", priority: 0 },
    ],
  },
  {
    id: 1, name: "Table 1 (Routing)",
    entries: [
      { match: "ip_dst=10.0.0.0/24", action: "set_eth_dst, output:1", priority: 100 },
      { match: "ip_dst=10.0.1.0/24", action: "set_eth_dst, output:2", priority: 100 },
      { match: "default", action: "goto Table 2", priority: 0 },
    ],
  },
  {
    id: 2, name: "Table 2 (ACL)",
    entries: [
      { match: "ip_proto=6, tcp_dst=80", action: "output:3", priority: 300 },
      { match: "ip_proto=6, tcp_dst=443", action: "output:3", priority: 300 },
      { match: "default", action: "drop", priority: 0 },
    ],
  },
];

export function OpenFlowPipelineSimulator() {
  const [tables] = useState(defaultTables);
  const [currentTable, setCurrentTable] = useState(0);
  const [matchedEntry, setMatchedEntry] = useState<number | null>(null);
  const [packetInput, setPacketInput] = useState("eth_type=0x0800, ip_dst=10.0.0.1, tcp_dst=80");

  const matchPacket = () => {
    setCurrentTable(0);
    setMatchedEntry(null);
    let tableIdx = 0;
    const iv = setInterval(() => {
      const table = tables[tableIdx];
      if (!table) { clearInterval(iv); return; }
      const input = packetInput.toLowerCase();
      const matchIdx = table.entries.findIndex(e => {
        if (e.match === "default") return true;
        return e.match.split(",").every(field => {
          const [k, v] = field.trim().split("=");
          return input.includes(k.trim()) && input.includes(v.trim());
        });
      });
      setCurrentTable(tableIdx);
      setMatchedEntry(matchIdx >= 0 ? matchIdx : table.entries.length - 1);
      const entry = table.entries[matchIdx >= 0 ? matchIdx : table.entries.length - 1];
      if (entry.action.includes("goto Table")) {
        tableIdx = parseInt(entry.action.match(/\d+/)?.[0] || "0");
      } else {
        clearInterval(iv);
      }
    }, 800);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">🔧 OpenFlow 管线模拟器</h3>
      <p className="text-sm text-text-secondary mb-4">展示流表匹配和指令执行</p>

      <div className="mb-4">
        <label className="text-sm text-text-secondary">数据包头字段</label>
        <div className="flex gap-2 mt-1">
          <input value={packetInput} onChange={e => setPacketInput(e.target.value)}
            className="flex-1 bg-bg-surface border border-border-subtle rounded p-2 text-sm text-text-primary font-mono" />
          <button onClick={matchPacket}
            className="px-4 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">匹配</button>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-4 overflow-x-auto">
        {tables.map((t, i) => (
          <div key={t.id} className="flex items-center">
            <div className={`px-3 py-2 rounded-lg text-xs font-medium ${currentTable === i ? "bg-blue-600 text-white ring-2 ring-blue-400" : "bg-bg-surface border border-border-subtle text-text-secondary"}`}>
              {t.name}
            </div>
            {i < tables.length - 1 && <div className="w-4 h-0.5 bg-border-subtle mx-1" />}
          </div>
        ))}
      </div>

      <div className="space-y-3">
        {tables.map((table, ti) => (
          <div key={table.id} className={`bg-bg-surface rounded-lg p-3 border ${currentTable === ti ? "border-blue-500" : "border-border-subtle"}`}>
            <div className="text-sm font-medium text-text-primary mb-2">{table.name}</div>
            <div className="space-y-1">
              {table.entries.map((entry, ei) => (
                <div key={ei} className={`flex items-center gap-2 p-2 rounded text-xs ${currentTable === ti && matchedEntry === ei ? "bg-blue-600/30 border border-blue-500" : "border border-transparent"}`}>
                  <span className="w-8 text-right font-mono text-text-secondary">p{entry.priority}</span>
                  <span className="flex-1 font-mono text-text-primary">{entry.match}</span>
                  <span className="font-mono text-green-400">{entry.action}</span>
                  {currentTable === ti && matchedEntry === ei && <span className="text-blue-400">✓</span>}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
export default OpenFlowPipelineSimulator;
