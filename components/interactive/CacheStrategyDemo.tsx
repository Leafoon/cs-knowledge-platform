"use client";
import { useState } from "react";

type Strategy = "LRU" | "LFU" | "FIFO";

export function CacheStrategyDemo() {
  const [strategy, setStrategy] = useState<Strategy>("LRU");
  const [cacheSize, setCacheSize] = useState(4);
  const [sequence, setSequence] = useState("ABCBDACEFD");
  const [pos, setPos] = useState(-1);
  const [cache, setCache] = useState<string[]>([]);
  const [hits, setHits] = useState(0);
  const [misses, setMisses] = useState(0);
  const [history, setHistory] = useState<{ item: string; hit: boolean; cache: string[] }[]>([]);
  const [freq, setFreq] = useState<Record<string, number>>({});

  const reset = () => { setPos(-1); setCache([]); setHits(0); setMisses(0); setHistory([]); setFreq({}); };

  const step = () => {
    const nextPos = pos + 1;
    if (nextPos >= sequence.length) return;
    const item = sequence[nextPos];

    const isHit = cache.includes(item);
    let newCache: string[];

    if (isHit) {
      newCache = [...cache];
      if (strategy === "LRU") {
        newCache = newCache.filter((c) => c !== item);
        newCache.push(item);
      }
      setHits((h) => h + 1);
    } else {
      setMisses((m) => m + 1);
      if (cache.length >= cacheSize) {
        if (strategy === "LRU") {
          newCache = [...cache.slice(1), item];
        } else if (strategy === "FIFO") {
          newCache = [...cache.slice(1), item];
        } else {
          const minFreq = Math.min(...cache.map((c) => freq[c] || 0));
          const evict = cache.find((c) => (freq[c] || 0) === minFreq) || cache[0];
          newCache = cache.filter((c) => c !== evict);
          newCache.push(item);
        }
      } else {
        newCache = [...cache, item];
      }
    }

    setFreq((prev) => ({ ...prev, [item]: (prev[item] || 0) + 1 }));
    setCache(newCache);
    setHistory((prev) => [...prev, { item, hit: isHit, cache: newCache }]);
    setPos(nextPos);
  };

  const runAll = () => {
    reset();
    let curCache: string[] = [];
    let curHits = 0;
    let curMisses = 0;
    let curFreq: Record<string, number> = {};
    const hist: { item: string; hit: boolean; cache: string[] }[] = [];

    for (const item of sequence) {
      const isHit = curCache.includes(item);
      let newCache: string[];
      if (isHit) {
        newCache = [...curCache];
        if (strategy === "LRU") { newCache = newCache.filter((c) => c !== item); newCache.push(item); }
        curHits++;
      } else {
        curMisses++;
        if (curCache.length >= cacheSize) {
          if (strategy === "LRU" || strategy === "FIFO") { newCache = [...curCache.slice(1), item]; }
          else {
            const evict = curCache.reduce((a, b) => (curFreq[a] || 0) <= (curFreq[b] || 0) ? a : b);
            newCache = curCache.filter((c) => c !== evict); newCache.push(item);
          }
        } else { newCache = [...curCache, item]; }
      }
      curFreq[item] = (curFreq[item] || 0) + 1;
      curCache = newCache;
      hist.push({ item, hit: isHit, cache: newCache });
    }
    setCache(curCache); setHits(curHits); setMisses(curMisses); setHistory(hist); setFreq(curFreq); setPos(sequence.length - 1);
  };

  const hitRate = (hits + misses) > 0 ? ((hits / (hits + misses)) * 100).toFixed(1) : "0.0";

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">缓存替换策略对比</h3>
      <div className="flex gap-2 mb-4">
        {(["LRU", "LFU", "FIFO"] as Strategy[]).map((s) => (
          <button key={s} onClick={() => { setStrategy(s); reset(); }}
            className={`flex-1 py-1.5 rounded text-sm ${strategy === s ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {s}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs text-text-secondary">缓存大小: {cacheSize}</label>
          <input type="range" min={2} max={8} value={cacheSize} onChange={(e) => { setCacheSize(+e.target.value); reset(); }} className="w-full mt-1" />
        </div>
        <div>
          <label className="text-xs text-text-secondary">访问序列</label>
          <input value={sequence} onChange={(e) => { setSequence(e.target.value.toUpperCase().replace(/[^A-Z]/g, "")); reset(); }}
            className="w-full mt-1 px-2 py-1 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 font-mono text-sm text-text-primary" />
        </div>
      </div>
      <div className="mb-4">
        <div className="flex gap-0.5 mb-2">
          {sequence.split("").map((c, i) => (
            <span key={i} className={`w-7 h-7 flex items-center justify-center text-xs font-mono rounded ${i === pos ? "bg-blue-600 text-white" : i < pos ? "bg-gray-300 dark:bg-gray-600 text-text-primary" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
              {c}
            </span>
          ))}
        </div>
      </div>
      <div className="mb-4 flex gap-1">
        {Array.from({ length: cacheSize }).map((_, i) => (
          <div key={i} className={`flex-1 h-10 rounded flex items-center justify-center text-sm font-mono font-bold ${i < cache.length ? "bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300 border border-green-300" : "bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-text-secondary"}`}>
            {cache[i] || "—"}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-2 rounded bg-green-50 dark:bg-green-900/20 text-center">
          <div className="text-xs text-green-600">命中</div><div className="font-bold text-green-700 dark:text-green-300">{hits}</div>
        </div>
        <div className="p-2 rounded bg-red-50 dark:bg-red-900/20 text-center">
          <div className="text-xs text-red-600">缺失</div><div className="font-bold text-red-700 dark:text-red-300">{misses}</div>
        </div>
        <div className="p-2 rounded bg-blue-50 dark:bg-blue-900/20 text-center">
          <div className="text-xs text-blue-600">命中率</div><div className="font-bold text-blue-700 dark:text-blue-300">{hitRate}%</div>
        </div>
      </div>
      <div className="flex gap-2">
        <button onClick={step} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">单步</button>
        <button onClick={runAll} className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm">全部执行</button>
        <button onClick={reset} className="px-4 py-2 bg-gray-500 text-white rounded text-sm">重置</button>
      </div>
      <p className="text-xs text-text-secondary mt-3">LRU淘汰最久未用；LFU淘汰使用频率最低；FIFO先进先出淘汰。</p>
    </div>
  );
}
export default CacheStrategyDemo;
