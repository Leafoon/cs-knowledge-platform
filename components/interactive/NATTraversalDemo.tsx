"use client";
import { useState } from "react";

const methods = [
  {
    name: "STUN",
    full: "Session Traversal Utilities for NAT",
    zh: "NAT会话穿透工具",
    desc: "客户端通过STUN服务器发现自己的公网IP和端口，以及NAT类型",
    steps: ["客户端发送Binding Request到STUN服务器", "STUN服务器回复公网IP:端口", "客户端用公网地址与对端通信"],
    works: true,
    limitation: "对称NAT(Symmetric NAT)下失败",
  },
  {
    name: "TURN",
    full: "Traversal Using Relays around NAT",
    zh: "中继穿透",
    desc: "当STUN失败时，TURN服务器作为中继转发所有数据",
    steps: ["客户端连接TURN服务器分配中继地址", "对端通过TURN服务器发送数据", "TURN服务器双向中继转发"],
    works: true,
    limitation: "带宽成本高，增加延迟",
  },
  {
    name: "ICE",
    full: "Interactive Connectivity Establishment",
    zh: "交互式连接建立",
    desc: "综合STUN和TURN，收集所有候选地址并选择最优路径",
    steps: ["收集主机候选(host)、服务器反射(srflx)、中继(relay)", "交换候选地址(SDP信令)", "连通性检查(STUN Binding)", "选择最低延迟的可用路径"],
    works: true,
    limitation: "实现复杂，需信令服务器配合",
  },
];

export function NATTraversalDemo() {
  const [selected, setSelected] = useState(2);
  const [step, setStep] = useState(0);
  const [natType, setNatType] = useState("full-cone");

  const method = methods[selected];

  const natTypes = [
    { id: "full-cone", name: "Full Cone", zh: "全锥形", stun: true },
    { id: "restricted", name: "Restricted Cone", zh: "受限锥形", stun: true },
    { id: "port-restricted", name: "Port Restricted", zh: "端口受限锥形", stun: true },
    { id: "symmetric", name: "Symmetric", zh: "对称形", stun: false },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        NAT Traversal <span className="text-text-secondary text-sm">— STUN/TURN/ICE穿透</span>
      </h3>
      <div className="flex gap-2 mb-3">
        {methods.map((m, i) => (
          <button
            key={i}
            onClick={() => { setSelected(i); setStep(0); }}
            className={`px-3 py-1 rounded text-sm font-mono ${selected === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {m.name}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">{method.name} — {method.full}</div>
        <div className="text-sm text-text-secondary mb-2">{method.zh}: {method.desc}</div>
        <div className="text-xs text-yellow-600 dark:text-yellow-400">局限: {method.limitation}</div>
      </div>
      <div className="flex gap-1 mb-3 flex-wrap">
        {method.steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`px-2 py-1 rounded text-xs ${step === i ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
          >
            步骤 {i + 1}
          </button>
        ))}
      </div>
      <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">步骤 {step + 1}</div>
        <div className="text-sm text-text-secondary">{method.steps[step]}</div>
      </div>
      <div className="text-xs text-text-secondary mb-2">NAT类型测试:</div>
      <div className="flex gap-2 flex-wrap">
        {natTypes.map((nt) => (
          <button
            key={nt.id}
            onClick={() => setNatType(nt.id)}
            className={`px-2 py-1 rounded text-xs ${natType === nt.id ? "bg-purple-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {nt.name} ({nt.zh})
          </button>
        ))}
      </div>
      <div className="mt-2 text-xs">
        <span className="text-text-secondary">STUN在{natTypes.find((n) => n.id === natType)?.name}下: </span>
        <span className={natTypes.find((n) => n.id === natType)?.stun ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>
          {natTypes.find((n) => n.id === natType)?.stun ? "✓ 可用" : "✗ 不可用，需TURN"}
        </span>
      </div>
    </div>
  );
}

export default NATTraversalDemo;
