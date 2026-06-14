"use client";
import { useState } from "react";

interface Interface {
  name: string;
  type: "north" | "south" | "east-west";
  protocol: string;
  desc: string;
}

const interfaces: Interface[] = [
  { name: "北向接口 (Northbound)", type: "north", protocol: "REST API / gRPC", desc: "控制器向应用层暴露的编程接口，应用通过此接口下发网络策略" },
  { name: "南向接口 (Southbound)", type: "south", protocol: "OpenFlow / NETCONF", desc: "控制器向下管理交换机/路由器的接口，下发流表和配置" },
  { name: "东西向接口 (East-West)", type: "east-west", protocol: "控制器间协议", desc: "多控制器集群间的协调接口，同步网络状态和分担负载" },
];

export function SDNControllerVisualizer() {
  const [activeInterface, setActiveInterface] = useState(0);
  const [showFlow, setShowFlow] = useState(false);
  const iface = interfaces[activeInterface];

  const typeColors = {
    north: "bg-blue-500/15 border-blue-400/40 text-blue-700 dark:text-blue-300",
    south: "bg-emerald-500/15 border-emerald-400/40 text-emerald-700 dark:text-emerald-300",
    "east-west": "bg-violet-500/15 border-violet-400/40 text-violet-700 dark:text-violet-300",
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">SDN控制器可视化 (SDN Controller)</h3>
      <div className="space-y-2 mb-4">
        <div className="flex items-center justify-center">
          <div className="px-4 py-2 rounded-lg bg-blue-500/15 border border-blue-400/40 text-xs font-medium text-blue-700 dark:text-blue-300">
            应用层（网络应用）
          </div>
        </div>
        <div className="flex justify-center text-text-tertiary text-lg">↕ 北向接口</div>
        <div className="flex items-center justify-center">
          <div className="px-6 py-3 rounded-lg bg-amber-500/15 border border-amber-400/40 text-sm font-bold text-amber-700 dark:text-amber-300">
            SDN 控制器
          </div>
        </div>
        <div className="flex justify-center text-text-tertiary text-lg">↕ 南向接口</div>
        <div className="flex justify-center gap-4">
          <div className="px-3 py-2 rounded-lg bg-emerald-500/15 border border-emerald-400/40 text-xs text-emerald-700 dark:text-emerald-300">交换机 A</div>
          <div className="px-3 py-2 rounded-lg bg-emerald-500/15 border border-emerald-400/40 text-xs text-emerald-700 dark:text-emerald-300">交换机 B</div>
          <div className="px-3 py-2 rounded-lg bg-emerald-500/15 border border-emerald-400/40 text-xs text-emerald-700 dark:text-emerald-300">交换机 C</div>
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        {interfaces.map((inf, i) => (
          <button key={i} onClick={() => setActiveInterface(i)}
            className={`flex-1 px-2 py-1.5 rounded-lg border text-[10px] font-medium transition-all ${activeInterface === i ? typeColors[inf.type] : "bg-bg-tertiary border-border-subtle text-text-secondary"}`}>
            {inf.name.split(" (")[0]}
          </button>
        ))}
      </div>
      <div className={`rounded-lg border p-4 mb-4 ${typeColors[iface.type]}`}>
        <div className="text-sm font-medium mb-1">{iface.name}</div>
        <div className="text-xs opacity-80 mb-1">协议: {iface.protocol}</div>
        <div className="text-xs opacity-70">{iface.desc}</div>
      </div>
      <button onClick={() => setShowFlow(!showFlow)} className="w-full px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:text-text-primary transition-colors mb-3">
        {showFlow ? "隐藏" : "显示"}SDN数据流
      </button>
      {showFlow && (
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 text-xs space-y-2 mb-3">
          <div className="flex items-center gap-2"><span className="text-blue-500">1.</span><span className="text-text-secondary">应用通过REST API下发意图（如"禁止A到B的流量"）</span></div>
          <div className="flex items-center gap-2"><span className="text-amber-500">2.</span><span className="text-text-secondary">控制器将意图翻译为流表规则</span></div>
          <div className="flex items-center gap-2"><span className="text-emerald-500">3.</span><span className="text-text-secondary">通过OpenFlow将流表下发到交换机</span></div>
          <div className="flex items-center gap-2"><span className="text-violet-500">4.</span><span className="text-text-secondary">交换机按流表匹配转发数据包</span></div>
        </div>
      )}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">OpenDaylight</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• Java 实现，模块化架构</li>
            <li>• 支持 OpenFlow/NETCONF</li>
            <li>• 企业级功能丰富</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">ONOS</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 分布式集群架构</li>
            <li>• 面向运营商网络</li>
            <li>• 高可用和可扩展</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">Ryu / Floodlight</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 轻量级，适合实验</li>
            <li>• Python (Ryu) / Java</li>
            <li>• 教学和原型开发</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default SDNControllerVisualizer;
