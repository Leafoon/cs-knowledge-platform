"use client";
import { useState } from "react";

interface Node {
  id: number;
  hash: number;
  label: string;
}

interface KV {
  key: string;
  value: string;
  responsibleNode: number;
}

export function DHTDemo() {
  const [nodes] = useState<Node[]>([
    { id: 0, hash: 10, label: "Node A" },
    { id: 1, hash: 35, label: "Node B" },
    { id: 2, hash: 60, label: "Node C" },
    { id: 3, hash: 85, label: "Node D" },
  ]);
  const [searchKey, setSearchKey] = useState("");
  const [searchResult, setSearchResult] = useState<string | null>(null);
  const [searchPath, setSearchPath] = useState<number[]>([]);
  const [storedKV, setStoredKV] = useState<KV[]>([
    { key: "hello", value: "world", responsibleNode: 0 },
    { key: "foo", value: "bar", responsibleNode: 1 },
    { key: "name", value: "Alice", responsibleNode: 2 },
  ]);
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");

  const simpleHash = (s: string) => {
    let h = 0;
    for (const c of s) h = (h + c.charCodeAt(0)) % 100;
    return h;
  };

  const findNode = (hash: number) => {
    const sorted = [...nodes].sort((a, b) => a.hash - b.hash);
    for (const n of sorted) {
      if (n.hash >= hash) return n;
    }
    return sorted[0];
  };

  const handleSearch = () => {
    const hash = simpleHash(searchKey);
    const path: number[] = [];
    let current = 0;
    const sorted = [...nodes].sort((a, b) => a.hash - b.hash);
    for (const n of sorted) {
      path.push(n.id);
      if (n.hash >= hash) { current = n.id; break; }
    }
    if (path.length === 0) { path.push(sorted[0].id); current = sorted[0].id; }
    setSearchPath(path);
    const kv = storedKV.find((kv) => kv.key === searchKey);
    setSearchResult(kv ? kv.value : "未找到");
  };

  const handleStore = () => {
    if (!newKey) return;
    const hash = simpleHash(newKey);
    const node = findNode(hash);
    setStoredKV((prev) => [...prev.filter((kv) => kv.key !== newKey), { key: newKey, value: newValue, responsibleNode: node.id }]);
    setNewKey("");
    setNewValue("");
  };

  const ringRadius = 110;
  const cx = 150, cy = 130;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">分布式哈希表 (Chord DHT)</h3>
      <div className="flex justify-center mb-4">
        <svg width="300" height="260" className="text-text-primary">
          <circle cx={cx} cy={cy} r={ringRadius} fill="none" stroke="currentColor" strokeWidth="1" opacity="0.2" />
          {nodes.map((n) => {
            const angle = (n.hash / 100) * 2 * Math.PI - Math.PI / 2;
            const x = cx + ringRadius * Math.cos(angle);
            const y = cy + ringRadius * Math.sin(angle);
            const highlighted = searchPath.includes(n.id);
            return (
              <g key={n.id}>
                <circle cx={x} cy={y} r={highlighted ? 18 : 14} fill={highlighted ? "#3b82f6" : "#6b7280"} className="transition-all" />
                <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="10" fontWeight="bold">{n.label}</text>
                <text x={x} y={y + 28} textAnchor="middle" fontSize="9" fill="currentColor" opacity="0.6">h={n.hash}</text>
              </g>
            );
          })}
          {searchPath.length > 1 && searchPath.map((id, i) => {
            if (i === 0) return null;
            const from = nodes.find((n) => n.id === searchPath[i - 1])!;
            const to = nodes.find((n) => n.id === id)!;
            const a1 = (from.hash / 100) * 2 * Math.PI - Math.PI / 2;
            const a2 = (to.hash / 100) * 2 * Math.PI - Math.PI / 2;
            return <line key={i} x1={cx + ringRadius * Math.cos(a1)} y1={cy + ringRadius * Math.sin(a1)}
              x2={cx + ringRadius * Math.cos(a2)} y2={cy + ringRadius * Math.sin(a2)}
              stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrow)" />;
          })}
          <defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3b82f6" /></marker></defs>
        </svg>
      </div>
      <div className="flex gap-2 mb-3">
        <input value={searchKey} onChange={(e) => setSearchKey(e.target.value)} placeholder="输入key"
          className="flex-1 px-3 py-1.5 rounded border border-border-subtle bg-bg-elevated text-text-primary text-sm" />
        <button onClick={handleSearch} className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">查找</button>
      </div>
      {searchResult !== null && (
        <div className="mb-3 text-sm bg-gray-50 dark:bg-gray-900 rounded p-2">
          路由路径: {searchPath.map((id) => nodes.find((n) => n.id === id)?.label).join(" → ")} | 值: <span className="font-mono text-text-primary">{searchResult}</span>
        </div>
      )}
      <div className="flex gap-2 mb-3">
        <input value={newKey} onChange={(e) => setNewKey(e.target.value)} placeholder="key"
          className="flex-1 px-3 py-1.5 rounded border border-border-subtle bg-bg-elevated text-text-primary text-sm" />
        <input value={newValue} onChange={(e) => setNewValue(e.target.value)} placeholder="value"
          className="flex-1 px-3 py-1.5 rounded border border-border-subtle bg-bg-elevated text-text-primary text-sm" />
        <button onClick={handleStore} className="px-4 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded text-sm">存储</button>
      </div>
      <div className="text-xs text-text-secondary">
        已存储: {storedKV.map((kv) => `${kv.key}→${kv.value}@${nodes.find((n) => n.id === kv.responsibleNode)?.label}`).join(", ")}
      </div>
    </div>
  );
}
export default DHTDemo;
