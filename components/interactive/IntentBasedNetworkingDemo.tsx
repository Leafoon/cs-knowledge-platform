"use client";
import { useState } from "react";

const intents = [
  {
    id: 1,
    intent: "VoIP流量优先于Web流量",
    config: ["access-list 101 permit udp any any range 16384 32767", "class-map match-any VOIP", "match access-group 101", "policy-map QOS_POLICY", "class VOIP", "priority percent 30", "interface GigabitEthernet0/0", "service-policy output QOS_POLICY"],
    layer: "QoS策略",
  },
  {
    id: 2,
    intent: "隔离研发部门与财务部门网络",
    config: ["ip access-list extended BLOCK_FINANCE", "deny ip 10.1.0.0 0.0.255.255 10.2.0.0 0.0.255.255", "permit ip any any", "interface Vlan10", "ip access-group BLOCK_FINANCE in"],
    layer: "安全策略",
  },
  {
    id: 3,
    intent: "所有分支通过主数据中心访问互联网",
    config: ["ip route 0.0.0.0 0.0.0.0 10.0.0.1", "router bgp 65001", "neighbor 10.0.0.1 remote-as 65000", "network 0.0.0.0"],
    layer: "路由策略",
  },
];

export function IntentBasedNetworkingDemo() {
  const [selected, setSelected] = useState(0);
  const [phase, setPhase] = useState(0);

  const phases = ["意图输入", "翻译引擎", "配置生成", "验证部署"];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Intent-Based Networking <span className="text-text-secondary text-sm">— 意图驱动网络</span>
      </h3>
      <div className="flex gap-2 mb-4">
        {intents.map((it, i) => (
          <button
            key={i}
            onClick={() => { setSelected(i); setPhase(0); }}
            className={`px-3 py-1 rounded text-sm ${selected === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            示例 {i + 1}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-1 mb-4">
        {phases.map((p, i) => (
          <div key={i} className="flex items-center">
            <button
              onClick={() => setPhase(i)}
              className={`px-3 py-1 rounded text-sm ${phase === i ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
            >
              {p}
            </button>
            {i < phases.length - 1 && <span className="mx-1 text-text-secondary">→</span>}
          </div>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
        {phase === 0 && (
          <div>
            <div className="text-sm text-text-secondary mb-1">业务意图:</div>
            <div className="text-lg font-semibold text-text-primary">&ldquo;{intents[selected].intent}&rdquo;</div>
            <div className="text-xs text-text-secondary mt-2">策略域: {intents[selected].layer}</div>
          </div>
        )}
        {phase === 1 && (
          <div>
            <div className="text-sm text-text-secondary mb-2">翻译引擎解析意图为结构化策略...</div>
            <div className="text-text-primary">分析意图关键词 → 匹配策略模板 → 生成配置参数</div>
          </div>
        )}
        {phase === 2 && (
          <div>
            <div className="text-sm text-text-secondary mb-2">生成的设备配置:</div>
            <pre className="text-xs font-mono text-text-primary bg-gray-200 dark:bg-gray-900 p-3 rounded overflow-x-auto">
              {intents[selected].config.join("\n")}
            </pre>
          </div>
        )}
        {phase === 3 && (
          <div>
            <div className="text-green-600 dark:text-green-400 font-semibold mb-1">✓ 配置验证通过</div>
            <div className="text-sm text-text-secondary">无冲突规则 | 语法正确 | 资源充足 | 已部署到 3 台设备</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default IntentBasedNetworkingDemo;
