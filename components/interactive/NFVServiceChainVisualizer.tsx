"use client";
import { useState } from "react";

interface VNF {
  id: string;
  name: string;
  zh: string;
  icon: string;
  cpu: number;
  mem: number;
}

const availableVNFs: VNF[] = [
  { id: "fw", name: "Firewall", zh: "防火墙", icon: "🛡", cpu: 2, mem: 4 },
  { id: "lb", name: "Load Balancer", zh: "负载均衡", icon: "⚖", cpu: 1, mem: 2 },
  { id: "ids", name: "IDS/IPS", zh: "入侵检测", icon: "🔍", cpu: 4, mem: 8 },
  { id: "nat", name: "NAT Gateway", zh: "NAT网关", icon: "🔄", cpu: 1, mem: 2 },
  { id: "vpn", name: "VPN Gateway", zh: "VPN网关", icon: "🔒", cpu: 2, mem: 4 },
  { id: "wan", name: "WAN Optimizer", zh: "广域网优化", icon: "🚀", cpu: 2, mem: 4 },
];

interface ChainItem {
  vnf: VNF;
  instanceId: number;
  status: "running" | "stopped" | "scaling";
}

export function NFVServiceChainVisualizer() {
  const [chain, setChain] = useState<ChainItem[]>([
    { vnf: availableVNFs[0], instanceId: 1, status: "running" },
    { vnf: availableVNFs[1], instanceId: 1, status: "running" },
    { vnf: availableVNFs[2], instanceId: 1, status: "running" },
  ]);
  const [selectedVNF, setSelectedVNF] = useState<string | null>(null);

  const addVNF = (vnf: VNF) => {
    setChain([...chain, { vnf, instanceId: chain.filter((c) => c.vnf.id === vnf.id).length + 1, status: "running" }]);
  };

  const removeVNF = (idx: number) => {
    setChain(chain.filter((_, i) => i !== idx));
  };

  const toggleStatus = (idx: number) => {
    const next = [...chain];
    next[idx].status = next[idx].status === "running" ? "stopped" : "running";
    setChain(next);
  };

  const totalCPU = chain.reduce((s, c) => s + c.vnf.cpu, 0);
  const totalMem = chain.reduce((s, c) => s + c.vnf.mem, 0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        NFV Service Chain <span className="text-text-secondary text-sm">— VNF服务链编排</span>
      </h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {availableVNFs.map((v) => (
          <button
            key={v.id}
            onClick={() => addVNF(v)}
            className="px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-sm text-text-primary hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            {v.icon} {v.zh}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="flex items-center gap-2 mb-3 text-sm text-text-secondary">
          <span>流量入口</span>
          <span>→</span>
          {chain.map((c, i) => (
            <span key={i} className="flex items-center gap-1">
              <button
                onClick={() => setSelectedVNF(selectedVNF === `${c.vnf.id}-${i}` ? null : `${c.vnf.id}-${i}`)}
                className={`px-2 py-1 rounded text-sm ${c.status === "running" ? "bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-200" : "bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-200"} ${selectedVNF === `${c.vnf.id}-${i}` ? "ring-2 ring-blue-400" : ""}`}
              >
                {c.vnf.icon} {c.vnf.zh}#{c.instanceId}
              </button>
              {i < chain.length - 1 && <span>→</span>}
            </span>
          ))}
          <span>→</span>
          <span>流量出口</span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
            <div className="text-text-secondary">总CPU</div>
            <div className="font-bold text-text-primary">{totalCPU} vCPU</div>
          </div>
          <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
            <div className="text-text-secondary">总内存</div>
            <div className="font-bold text-text-primary">{totalMem} GB</div>
          </div>
        </div>
      </div>
      {selectedVNF && (
        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
          {(() => {
            const idx = chain.findIndex((c) => `${c.vnf.id}-${chain.filter((x, xi) => xi <= idx && x.vnf.id === c.vnf.id).length - 1}` === selectedVNF);
            const item = chain.find((c, i) => `${c.vnf.id}-${i}` === selectedVNF);
            if (!item) return null;
            const realIdx = chain.findIndex((c, i) => `${c.vnf.id}-${i}` === selectedVNF);
            return (
              <>
                <div className="font-semibold text-text-primary mb-2">
                  {item.vnf.icon} {item.vnf.name} ({item.vnf.zh})
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs mb-3">
                  <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
                    <div className="text-text-secondary">状态</div>
                    <div className={`font-bold ${item.status === "running" ? "text-green-600" : "text-red-600"}`}>{item.status}</div>
                  </div>
                  <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
                    <div className="text-text-secondary">CPU</div>
                    <div className="font-bold text-text-primary">{item.vnf.cpu} vCPU</div>
                  </div>
                  <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
                    <div className="text-text-secondary">内存</div>
                    <div className="font-bold text-text-primary">{item.vnf.mem} GB</div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => toggleStatus(realIdx)}
                    className={`px-3 py-1 rounded text-sm ${item.status === "running" ? "bg-yellow-600 text-white" : "bg-green-600 text-white"}`}
                  >
                    {item.status === "running" ? "停止" : "启动"}
                  </button>
                  <button
                    onClick={() => { removeVNF(realIdx); setSelectedVNF(null); }}
                    className="px-3 py-1 rounded bg-red-600 text-white text-sm"
                  >
                    移除
                  </button>
                </div>
              </>
            );
          })()}
        </div>
      )}
      <div className="text-xs text-text-secondary">
        点击VNF查看详情，使用上方按钮添加新功能到服务链
      </div>
    </div>
  );
}

export default NFVServiceChainVisualizer;
