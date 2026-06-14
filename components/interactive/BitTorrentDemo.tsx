"use client";
import { useState, useEffect } from "react";

interface Peer {
  id: number;
  name: string;
  chunks: boolean[];
  downloadSpeed: number;
  uploadSpeed: number;
  choked: boolean;
  interested: boolean;
}

const TOTAL_CHUNKS = 16;

export function BitTorrentDemo() {
  const [peers, setPeers] = useState<Peer[]>([]);
  const [strategy, setStrategy] = useState<"rarest" | "random">("rarest");
  const [tick, setTick] = useState(0);
  const [running, setRunning] = useState(false);

  const initPeers = () => {
    const names = ["Alice", "Bob", "Carol", "Dave", "Eve"];
    setPeers(names.map((name, id) => ({
      id,
      name,
      chunks: Array.from({ length: TOTAL_CHUNKS }, () => Math.random() > 0.7),
      downloadSpeed: 5 + Math.floor(Math.random() * 10),
      uploadSpeed: 3 + Math.floor(Math.random() * 8),
      choked: false,
      interested: false,
    })));
    setTick(0);
  };

  useEffect(() => { initPeers(); }, []);

  const getChunkRarity = (chunkIdx: number) => {
    return peers.filter((p) => p.chunks[chunkIdx]).length;
  };

  const step = () => {
    setPeers((prev) => {
      const next = prev.map((p) => ({ ...p, chunks: [...p.chunks] }));
      for (const peer of next) {
        const missing = peer.chunks.map((has, i) => (!has ? i : -1)).filter((i) => i >= 0);
        if (missing.length === 0) continue;
        let targetIdx: number;
        if (strategy === "rarest") {
          const rarity = missing.map((i) => ({ idx: i, count: getChunkRarity(i) }));
          rarity.sort((a, b) => a.count - b.count);
          targetIdx = rarity[0].idx;
        } else {
          targetIdx = missing[Math.floor(Math.random() * missing.length)];
        }
        const hasSource = next.filter((p) => p.id !== peer.id && p.chunks[targetIdx]);
        if (hasSource.length > 0 && !peer.choked) {
          peer.chunks[targetIdx] = true;
        }
      }
      return next;
    });
    setTick((t) => t + 1);
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 600);
    return () => clearInterval(id);
  }, [running, peers, strategy]);

  const allComplete = peers.every((p) => p.chunks.every(Boolean));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">BitTorrent 分片下载演示</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStrategy("rarest")}
          className={`flex-1 py-1.5 rounded text-sm ${strategy === "rarest" ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
          稀有优先 (Rarest First)
        </button>
        <button onClick={() => setStrategy("random")}
          className={`flex-1 py-1.5 rounded text-sm ${strategy === "random" ? "bg-purple-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
          随机选择
        </button>
      </div>
      <div className="space-y-2 mb-4">
        {peers.map((peer) => (
          <div key={peer.id} className="flex items-center gap-3">
            <span className="w-14 text-xs font-medium text-text-primary text-right">{peer.name}</span>
            <div className="flex gap-0.5 flex-1">
              {peer.chunks.map((has, i) => (
                <div key={i} className={`h-5 flex-1 rounded-sm ${has ? "bg-green-500" : "bg-gray-200 dark:bg-gray-700"}`}
                  title={`Chunk ${i}: ${has ? "已下载" : "缺失"} | 稀有度: ${getChunkRarity(i)}`} />
              ))}
            </div>
            <span className="text-xs text-text-secondary w-12 text-right">{peer.chunks.filter(Boolean).length}/{TOTAL_CHUNKS}</span>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">轮次</div>
          <div className="font-bold text-text-primary">{tick}</div>
        </div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">策略</div>
          <div className="font-bold text-text-primary">{strategy === "rarest" ? "稀有优先" : "随机"}</div>
        </div>
        <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">最稀有分片</div>
          <div className="font-bold text-text-primary">{Math.min(...Array.from({ length: TOTAL_CHUNKS }, (_, i) => getChunkRarity(i)))} 人拥有</div>
        </div>
      </div>
      <div className="flex gap-2">
        <button onClick={step} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">单步</button>
        <button onClick={() => setRunning(!running)} className={`flex-1 py-2 rounded text-sm font-medium ${running ? "bg-red-600 hover:bg-red-700 text-white" : "bg-green-600 hover:bg-green-700 text-white"}`}>
          {running ? "暂停" : "自动运行"}
        </button>
        <button onClick={initPeers} className="flex-1 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded text-sm font-medium">重置</button>
      </div>
      {allComplete && <div className="mt-3 p-3 rounded bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 text-sm font-medium text-center">所有对等体已完成下载!</div>}
      <p className="text-xs text-text-secondary mt-3">BitTorrent使用分片传输，稀有优先策略使网络中稀缺的分片优先被下载，提高整体分发效率。</p>
    </div>
  );
}
export default BitTorrentDemo;
