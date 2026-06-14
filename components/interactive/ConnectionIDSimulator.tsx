"use client";
import { useState } from "react";

interface Connection {
  id: string;
  cid: string;
  srcIP: string;
  dstIP: string;
  migrated: boolean;
  migratedCID?: string;
}

export function ConnectionIDSimulator() {
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selected, setSelected] = useState<number | null>(null);

  const generateCID = () => Math.random().toString(16).slice(2, 18);

  const addConnection = () => {
    const cid = generateCID();
    setConnections((prev) => [...prev, {
      id: `conn-${prev.length + 1}`,
      cid,
      srcIP: `192.168.1.${10 + prev.length}`,
      dstIP: "93.184.216.34",
      migrated: false,
    }]);
  };

  const migrate = (idx: number) => {
    const newCID = generateCID();
    setConnections((prev) => prev.map((c, i) => i === idx ? { ...c, migrated: true, migratedCID: newCID } : c));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">QUIC 连接ID模拟器</h3>
      <p className="text-sm text-text-secondary mb-4">
        QUIC使用连接ID(Connection ID)标识连接，支持连接迁移(Connection Migration)。
      </p>
      <div className="flex gap-2 mb-4">
        <button onClick={addConnection} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">
          新建连接
        </button>
        {connections.length > 0 && (
          <button onClick={() => { setConnections([]); setSelected(null); }}
            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded text-sm text-text-secondary">
            清空
          </button>
        )}
      </div>
      <div className="space-y-2 mb-4">
        {connections.map((c, i) => (
          <div key={i}
            onClick={() => setSelected(selected === i ? null : i)}
            className={`p-3 rounded border cursor-pointer transition-colors ${selected === i ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20" : "border-border-subtle hover:border-gray-400"}`}>
            <div className="flex justify-between items-center">
              <span className="font-mono text-sm text-text-primary">{c.id}</span>
              {c.migrated && <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 px-2 py-0.5 rounded">已迁移</span>}
            </div>
            <div className="text-xs text-text-secondary mt-1">
              CID: <span className="font-mono">{c.cid}</span>
              {c.migrated && <> → <span className="font-mono text-yellow-600">{c.migratedCID}</span></>}
            </div>
            <div className="text-xs text-text-secondary">
              {c.srcIP} → {c.dstIP}
            </div>
            {selected === i && !c.migrated && (
              <button onClick={(e) => { e.stopPropagation(); migrate(i); }}
                className="mt-2 px-3 py-1 bg-yellow-500 hover:bg-yellow-600 text-white rounded text-xs">
                模拟网络切换(迁移)
              </button>
            )}
          </div>
        ))}
      </div>
      {connections.length === 0 && (
        <div className="text-center text-text-secondary py-8 text-sm">点击"新建连接"开始</div>
      )}
    </div>
  );
}
export default ConnectionIDSimulator;
