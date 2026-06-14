"use client";
import { useState } from "react";

interface Rule {
  id: number;
  srcIp: string;
  dstIp: string;
  srcPort: string;
  dstPort: string;
  protocol: string;
  action: "permit" | "deny";
  priority: number;
}

const INIT_RULES: Rule[] = [
  { id: 1, srcIp: "10.0.0.0/8", dstIp: "any", srcPort: "any", dstPort: "80", protocol: "TCP", action: "permit", priority: 1 },
  { id: 2, srcIp: "any", dstIp: "192.168.1.0/24", srcPort: "any", dstPort: "22", protocol: "TCP", action: "permit", priority: 2 },
  { id: 3, srcIp: "172.16.0.0/16", dstIp: "any", srcPort: "any", dstPort: "any", protocol: "any", action: "deny", priority: 3 },
  { id: 4, srcIp: "any", dstIp: "any", srcPort: "any", dstPort: "any", protocol: "any", action: "permit", priority: 4 },
];

interface Packet {
  srcIp: string;
  dstIp: string;
  srcPort: number;
  dstPort: number;
  protocol: string;
}

const SAMPLE_PACKETS: Packet[] = [
  { srcIp: "10.0.0.5", dstIp: "8.8.8.8", srcPort: 12345, dstPort: 80, protocol: "TCP" },
  { srcIp: "192.168.1.100", dstIp: "192.168.1.1", srcPort: 54321, dstPort: 22, protocol: "TCP" },
  { srcIp: "172.16.0.10", dstIp: "10.0.0.1", srcPort: 9999, dstPort: 443, protocol: "TCP" },
  { srcIp: "8.8.8.8", dstIp: "10.0.0.5", srcPort: 80, dstPort: 12345, protocol: "TCP" },
];

export function FirewallTCAMengine() {
  const [rules] = useState<Rule[]>(INIT_RULES);
  const [packetIdx, setPacketIdx] = useState(0);
  const [matchResult, setMatchResult] = useState<{ rule: Rule; matched: boolean } | null>(null);

  const matchPacket = () => {
    const pkt = SAMPLE_PACKETS[packetIdx];
    for (const rule of rules) {
      if (matchRule(rule, pkt)) {
        setMatchResult({ rule, matched: true });
        return;
      }
    }
    setMatchResult(null);
  };

  const matchRule = (rule: Rule, pkt: Packet): boolean => {
    if (rule.protocol !== "any" && rule.protocol !== pkt.protocol) return false;
    if (rule.dstPort !== "any" && parseInt(rule.dstPort) !== pkt.dstPort) return false;
    return true;
  };

  const pkt = SAMPLE_PACKETS[packetIdx];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">防火墙TCAM规则匹配引擎</h3>
      <div className="bg-bg-muted rounded-lg p-3 mb-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-text-secondary border-b border-border-subtle">
              <th className="text-left p-1">优先级</th><th className="text-left p-1">源IP</th><th className="text-left p-1">目的IP</th>
              <th className="text-left p-1">协议</th><th className="text-left p-1">目的端口</th><th className="text-left p-1">动作</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((r) => (
              <tr key={r.id} className={`border-b border-border-subtle ${matchResult?.rule.id === r.id ? "bg-blue-100 dark:bg-blue-900/30" : ""}`}>
                <td className="p-1 font-mono">{r.priority}</td><td className="p-1 font-mono">{r.srcIp}</td><td className="p-1 font-mono">{r.dstIp}</td>
                <td className="p-1 font-mono">{r.protocol}</td><td className="p-1 font-mono">{r.dstPort}</td>
                <td className={`p-1 font-bold ${r.action === "permit" ? "text-green-500" : "text-red-500"}`}>{r.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="bg-bg-muted rounded-lg p-3 mb-4">
        <div className="text-sm text-text-secondary mb-2">测试数据包:</div>
        <div className="flex gap-2 mb-2">
          {SAMPLE_PACKETS.map((_, i) => (
            <button key={i} onClick={() => { setPacketIdx(i); setMatchResult(null); }}
              className={`px-2 py-1 rounded text-xs ${packetIdx === i ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
              包{i + 1}
            </button>
          ))}
        </div>
        <div className="font-mono text-xs text-text-primary">
          {pkt.srcIp}:{pkt.srcPort} → {pkt.dstIp}:{pkt.dstPort} [{pkt.protocol}]
        </div>
      </div>
      <button onClick={matchPacket} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm mb-4">
        TCAM匹配
      </button>
      {matchResult && (
        <div className={`p-3 rounded-lg text-sm ${matchResult.rule.action === "permit" ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300" : "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"}`}>
          匹配规则 #{matchResult.rule.id}: <strong>{matchResult.rule.action.toUpperCase()}</strong> (优先级 {matchResult.rule.priority})
        </div>
      )}
      <div className="text-xs text-text-secondary mt-3">
        TCAM(三态内容寻可存储器)支持通配符匹配,O(1)时间完成规则查找。规则按优先级排列,首个匹配决定动作。
      </div>
    </div>
  );
}

export default FirewallTCAMengine;
