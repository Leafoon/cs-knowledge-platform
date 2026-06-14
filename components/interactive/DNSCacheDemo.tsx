"use client";
import { useState } from "react";

interface CacheEntry {
  domain: string;
  type: string;
  value: string;
  ttl: number;
  remaining: number;
  hits: number;
}

const initialCache: CacheEntry[] = [
  { domain: "www.google.com", type: "A", value: "142.250.80.4", ttl: 300, remaining: 245, hits: 12 },
  { domain: "github.com", type: "A", value: "140.82.121.3", ttl: 60, remaining: 15, hits: 8 },
  { domain: "cdn.example.com", type: "CNAME", value: "d1234.cloudfront.net", ttl: 3600, remaining: 3200, hits: 3 },
];

export function DNSCacheDemo() {
  const [cache, setCache] = useState<CacheEntry[]>(initialCache);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [resultType, setResultType] = useState<"hit" | "miss" | null>(null);

  const queryDomain = () => {
    const entry = cache.find((e) => e.domain === query && e.remaining > 0);
    if (entry) {
      setResult(`缓存命中: ${entry.value} (TTL剩余: ${entry.remaining}s)`);
      setResultType("hit");
      setCache((prev) => prev.map((e) => e === entry ? { ...e, hits: e.hits + 1 } : e));
    } else {
      setResult(`缓存未命中，递归查询 → 142.250.80.${Math.floor(Math.random() * 255)}`);
      setResultType("miss");
      if (query) {
        setCache((prev) => [...prev, { domain: query, type: "A", value: `142.250.80.${Math.floor(Math.random() * 255)}`, ttl: 300, remaining: 300, hits: 1 }]);
      }
    }
  };

  const tick = () => {
    setCache((prev) => prev.map((e) => ({ ...e, remaining: Math.max(0, e.remaining - 30) })));
  };

  const purge = () => {
    setCache((prev) => prev.filter((e) => e.remaining > 0));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNS 缓存演示</h3>
      <div className="flex gap-2 mb-4">
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="输入域名"
          className="flex-1 px-3 py-2 rounded border border-border-subtle bg-bg-elevated text-text-primary text-sm" onKeyDown={(e) => e.key === "Enter" && queryDomain()} />
        <button onClick={queryDomain} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">查询</button>
        <button onClick={tick} className="px-3 py-2 bg-gray-200 dark:bg-gray-700 rounded text-sm text-text-secondary">⏱ -30s</button>
        <button onClick={purge} className="px-3 py-2 bg-red-100 dark:bg-red-900/20 text-red-600 rounded text-sm">清除过期</button>
      </div>
      {result && (
        <div className={`mb-4 p-3 rounded text-sm ${resultType === "hit" ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300" : "bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 text-yellow-700 dark:text-yellow-300"}`}>
          {result}
        </div>
      )}
      <div className="space-y-2">
        {cache.map((e, i) => (
          <div key={i} className={`flex items-center gap-3 p-2 rounded border text-xs transition-all ${e.remaining === 0 ? "opacity-40 border-red-200 dark:border-red-800" : "border-border-subtle"}`}>
            <span className="font-mono text-text-primary flex-1">{e.domain}</span>
            <span className="text-text-secondary">{e.type}</span>
            <span className="font-mono text-text-primary">{e.value}</span>
            <div className="w-20 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${e.remaining < 30 ? "bg-red-400" : "bg-green-400"}`} style={{ width: `${(e.remaining / e.ttl) * 100}%` }} />
            </div>
            <span className="font-mono text-text-secondary w-12 text-right">{e.remaining}s</span>
            <span className="text-text-secondary">命中{e.hits}次</span>
          </div>
        ))}
      </div>
    </div>
  );
}
export default DNSCacheDemo;
