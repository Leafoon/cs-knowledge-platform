"use client";
import { useState } from "react";

type Stage = "dns" | "edge" | "origin" | "done";

export function CDNArchitectureDemo() {
  const [stage, setStage] = useState<Stage>("dns");
  const [userLocation, setUserLocation] = useState<"北京" | "上海" | "纽约">("北京");

  const edgeServers: Record<string, { name: string; ip: string; latency: string }> = {
    北京: { name: "北京边缘节点", ip: "103.21.244.1", latency: "5ms" },
    上海: { name: "上海边缘节点", ip: "103.22.201.2", latency: "3ms" },
    纽约: { name: "纽约边缘节点", ip: "103.23.250.3", latency: "120ms" },
  };

  const origin = { name: "源站 (US-East)", ip: "203.0.113.50", latency: "200ms" };

  const stageInfo: Record<Stage, { title: string; desc: string; action: string }> = {
    dns: { title: "DNS 重定向", desc: "用户请求 www.example.com → CDN权威DNS返回最近边缘节点IP", action: "解析域名" },
    edge: { title: "边缘节点查询", desc: `请求到达 ${edgeServers[userLocation].name}，检查缓存是否命中`, action: "查询缓存" },
    origin: { title: "回源请求", desc: "缓存未命中，边缘节点向源站请求内容并缓存", action: "回源获取" },
    done: { title: "内容交付", desc: "内容从边缘节点返回给用户，后续请求直接命中缓存", action: "重新演示" },
  };

  const nextStage = () => {
    const order: Stage[] = ["dns", "edge", "origin", "done"];
    const idx = order.indexOf(stage);
    setStage(order[(idx + 1) % order.length]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">CDN 架构演示</h3>
      <div className="flex gap-2 mb-4">
        {(["北京", "上海", "纽约"] as const).map((loc) => (
          <button key={loc} onClick={() => { setUserLocation(loc); setStage("dns"); }}
            className={`px-3 py-1.5 rounded text-sm ${userLocation === loc ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            用户: {loc}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2 mb-4">
        {(["dns", "edge", "origin", "done"] as Stage[]).map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${stage === s ? "bg-blue-600 text-white" : i < ["dns", "edge", "origin", "done"].indexOf(stage) ? "bg-green-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
              {i + 1}
            </div>
            {i < 3 && <div className={`w-8 h-0.5 ${i < ["dns", "edge", "origin", "done"].indexOf(stage) ? "bg-green-500" : "bg-gray-200 dark:bg-gray-700"}`} />}
          </div>
        ))}
      </div>
      <div className="p-4 rounded bg-gray-50 dark:bg-gray-800 border border-border-subtle mb-4">
        <h4 className="text-sm font-bold text-text-primary mb-2">{stageInfo[stage].title}</h4>
        <p className="text-xs text-text-secondary mb-3">{stageInfo[stage].desc}</p>
        {stage === "dns" && (
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2 py-1 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-mono">www.example.com</span>
            <span className="text-text-secondary">→</span>
            <span className="px-2 py-1 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 font-mono">{edgeServers[userLocation].ip}</span>
          </div>
        )}
        {stage === "edge" && (
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 rounded bg-white dark:bg-gray-900 text-center">
              <div className="text-xs text-text-secondary">缓存状态</div>
              <div className="text-sm font-bold text-orange-600">MISS</div>
            </div>
            <div className="p-2 rounded bg-white dark:bg-gray-900 text-center">
              <div className="text-xs text-text-secondary">边缘延迟</div>
              <div className="text-sm font-bold text-text-primary">{edgeServers[userLocation].latency}</div>
            </div>
          </div>
        )}
        {stage === "origin" && (
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2 py-1 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">{edgeServers[userLocation].name}</span>
            <span className="text-text-secondary">→ {origin.latency} →</span>
            <span className="px-2 py-1 rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">{origin.name}</span>
          </div>
        )}
        {stage === "done" && (
          <div className="p-2 rounded bg-green-50 dark:bg-green-900/20 text-center text-sm text-green-700 dark:text-green-300 font-medium">
            内容已缓存，后续请求延迟仅 {edgeServers[userLocation].latency}
          </div>
        )}
      </div>
      <button onClick={nextStage} className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">
        {stageInfo[stage].action}
      </button>
      <p className="text-xs text-text-secondary mt-3">CDN通过DNS将用户导向最近的边缘节点，减少延迟；缓存未命中时回源获取。</p>
    </div>
  );
}
export default CDNArchitectureDemo;
