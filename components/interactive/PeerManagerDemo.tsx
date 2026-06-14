"use client";
import { useState } from "react";

interface Peer { id: number; ip: string; port: number; state: string; speed: string; pieces: number; choked: boolean; }

const peerStates = [
  { name: "connecting", label: "连接中", color: "text-yellow-400" },
  { name: "handshaking", label: "握手", color: "text-blue-400" },
  { name: "downloading", label: "下载中", color: "text-green-400" },
  { name: "seeding", label: "做种", color: "text-purple-400" },
  { name: "choked", label: "被阻塞", color: "text-red-400" },
];

export function PeerManagerDemo() {
  const [peers, setPeers] = useState<Peer[]>([
    { id: 1, ip: "10.0.1.50", port: 6881, state: "downloading", speed: "1.2 MB/s", pieces: 342, choked: false },
    { id: 2, ip: "10.0.2.100", port: 51413, state: "seeding", speed: "0 B/s", pieces: 500, choked: false },
    { id: 3, ip: "10.0.3.77", port: 6881, state: "choked", speed: "0 B/s", pieces: 156, choked: true },
    { id: 4, ip: "10.0.4.200", port: 6881, state: "connecting", speed: "—", pieces: 0, choked: false },
  ]);
  const [optimisticUnchoke, setOptimisticUnchoke] = useState(false);

  const addPeer = () => {
    const states = ["connecting", "handshaking", "downloading", "seeding"];
    const state = states[Math.floor(Math.random() * states.length)];
    setPeers(prev => [...prev, {
      id: Date.now(),
      ip: `10.0.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
      port: 6881 + Math.floor(Math.random() * 100),
      state,
      speed: state === "downloading" ? `${(Math.random() * 5).toFixed(1)} MB/s` : "0 B/s",
      pieces: Math.floor(Math.random() * 500),
      choked: Math.random() > 0.7,
    }].slice(0, 20));
  };

  const toggleChoke = (id: number) => {
    setPeers(prev => prev.map(p => p.id === id ? { ...p, choked: !p.choked, state: p.choked ? "downloading" : "choked" } : p));
  };

  const unchokeBest = () => {
    setOptimisticUnchoke(true);
    setPeers(prev => {
      const sorted = [...prev].sort((a, b) => b.pieces - a.pieces);
      return prev.map(p => {
        if (p.id === sorted[0]?.id) return { ...p, choked: false, state: "downloading" };
        return p;
      });
    });
    setTimeout(() => setOptimisticUnchoke(false), 2000);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">🤝 对等节点管理器</h3>
      <p className="text-sm text-text-secondary mb-4">展示 BitTorrent 的 peer 连接和管理</p>

      <div className="flex flex-wrap gap-2 mb-4">
        <button onClick={addPeer}
          className="px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">发现新 Peer</button>
        <button onClick={unchokeBest}
          className={`px-3 py-1.5 rounded text-sm ${optimisticUnchoke ? "bg-yellow-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-yellow-400"}`}>
          乐观阻塞解除
        </button>
        <div className="ml-auto flex items-center gap-2 text-xs text-text-secondary">
          <span>Peers: {peers.length}</span>
          <span>做种: {peers.filter(p => p.state === "seeding").length}</span>
          <span>下载: {peers.filter(p => p.state === "downloading").length}</span>
        </div>
      </div>

      <div className="space-y-1.5 mb-4">
        {peers.map(p => (
          <div key={p.id} className={`flex items-center gap-3 p-3 rounded-lg bg-bg-surface border ${p.choked ? "border-red-700" : "border-border-subtle"}`}>
            <div className={`w-2 h-2 rounded-full ${p.state === "downloading" ? "bg-green-500 animate-pulse" : p.state === "seeding" ? "bg-purple-500" : p.choked ? "bg-red-500" : "bg-yellow-500"}`} />
            <div className="font-mono text-xs text-text-primary w-28">{p.ip}:{p.port}</div>
            <div className={`text-xs w-16 ${peerStates.find(s => s.name === p.state)?.color || "text-text-secondary"}`}>
              {peerStates.find(s => s.name === p.state)?.label || p.state}
            </div>
            <div className="text-xs text-text-secondary w-20">{p.pieces} pieces</div>
            <div className="text-xs font-mono text-green-400 w-20">{p.speed}</div>
            <button onClick={() => toggleChoke(p.id)}
              className={`ml-auto px-2 py-1 rounded text-[10px] ${p.choked ? "bg-red-700 text-red-200" : "bg-green-700 text-green-200"}`}>
              {p.choked ? "Choked" : "Unchoked"}
            </button>
          </div>
        ))}
      </div>

      <div className="bg-bg-surface rounded-lg p-3 text-xs text-text-secondary">
        <strong className="text-text-primary">Tit-for-Tat：</strong>BitTorrent 的阻塞算法优先服务上传速率最高的 peer。每 10 秒进行一次乐观阻塞解除（Optimistic Unchoke），随机选择一个 peer 解除阻塞以发现更好的下载伙伴。
      </div>
    </div>
  );
}
export default PeerManagerDemo;
