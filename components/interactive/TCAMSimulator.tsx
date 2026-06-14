"use client";
import { useState } from "react";

interface Rule { id: number; match: string; action: string; priority: number }

export function TCAMSimulator() {
  const [rules, setRules] = useState<Rule[]>([
    { id: 1, match: "192.168.1.0/24", action: "转发 eth0", priority: 100 },
    { id: 2, match: "10.0.0.0/8", action: "转发 eth1", priority: 200 },
    { id: 3, match: "172.16.0.0/12", action: "丢弃", priority: 150 },
    { id: 4, match: "0.0.0.0/0", action: "默认路由 eth2", priority: 0 },
  ]);
  const [lookupIP, setLookupIP] = useState("192.168.1.100");
  const [result, setResult] = useState<{ rule: Rule; cycles: number } | null>(null);
  const [newMatch, setNewMatch] = useState("");
  const [newAction, setNewAction] = useState("");
  const [newPrio, setNewPrio] = useState(100);

  const lookup = () => {
    const ipParts = lookupIP.split(".").map(Number);
    let best: Rule | null = null;
    for (const rule of rules) {
      const [prefix, lenStr] = rule.match.split("/");
      const len = parseInt(lenStr);
      const prefixParts = prefix.split(".").map(Number);
      const bytesToCheck = Math.floor(len / 8);
      const bitsRem = len % 8;
      let match = true;
      for (let i = 0; i < bytesToCheck; i++) {
        if (ipParts[i] !== prefixParts[i]) { match = false; break; }
      }
      if (match && bitsRem > 0 && bytesToCheck < 4) {
        const mask = 0xFF << (8 - bitsRem);
        if ((ipParts[bytesToCheck] & mask) !== (prefixParts[bytesToCheck] & mask)) match = false;
      }
      if (match && (!best || rule.priority > best.priority)) best = rule;
    }
    setResult(best ? { rule: best, cycles: 1 } : null);
  };

  const addRule = () => {
    if (!newMatch || !newAction) return;
    setRules((r) => [...r, { id: Date.now(), match: newMatch, action: newAction, priority: newPrio }]);
    setNewMatch(""); setNewAction("");
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCAM 模拟器 (Ternary CAM)</h3>
      <div className="flex gap-2 mb-4">
        <input type="text" value={lookupIP} onChange={(e) => setLookupIP(e.target.value)} placeholder="IP 地址"
          className="flex-1 p-2 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm font-mono" />
        <button onClick={lookup} className="px-4 py-2 rounded bg-blue-500 text-white text-sm hover:bg-blue-600 transition-colors">并行查找</button>
      </div>
      {result && (
        <div className="mb-4 p-3 rounded bg-green-500/10 border border-green-400/30">
          <p className="text-green-400 text-sm font-medium">命中规则 #{result.rule.id}: {result.rule.match}</p>
          <p className="text-text-primary text-sm">动作: {result.rule.action}</p>
          <p className="text-text-muted text-xs">查找周期: {result.cycles} (TCAM 并行匹配所有规则)</p>
        </div>
      )}
      <div className="overflow-x-auto mb-4">
        <table className="w-full text-xs">
          <thead><tr className="text-text-muted border-b border-border-subtle"><th className="p-2 text-left">ID</th><th className="p-2 text-left">匹配规则</th><th className="p-2 text-left">动作</th><th className="p-2 text-left">优先级</th></tr></thead>
          <tbody>
            {rules.sort((a, b) => b.priority - a.priority).map((r) => (
              <tr key={r.id} className={`border-t border-border-subtle ${result?.rule.id === r.id ? "bg-green-500/10" : ""}`}>
                <td className="p-2 text-text-muted">{r.id}</td>
                <td className="p-2 text-text-primary font-mono">{r.match}</td>
                <td className="p-2 text-text-secondary">{r.action}</td>
                <td className="p-2 text-text-secondary">{r.priority}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex gap-2 mb-4">
        <input type="text" value={newMatch} onChange={(e) => setNewMatch(e.target.value)} placeholder="10.0.0.0/8"
          className="p-2 rounded border border-border-subtle bg-bg-primary text-text-primary text-xs font-mono w-36" />
        <input type="text" value={newAction} onChange={(e) => setNewAction(e.target.value)} placeholder="转发 eth3"
          className="p-2 rounded border border-border-subtle bg-bg-primary text-text-primary text-xs w-28" />
        <input type="number" value={newPrio} onChange={(e) => setNewPrio(+e.target.value)} placeholder="优先级"
          className="p-2 rounded border border-border-subtle bg-bg-primary text-text-primary text-xs w-20" />
        <button onClick={addRule} className="px-3 py-2 rounded bg-green-500 text-white text-xs hover:bg-green-600 transition-colors">添加</button>
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <p className="text-text-muted text-xs">TCAM (三态内容寻址存储器) 每个表项包含 (值, 掩码) 对，支持通配符匹配。所有表项并行比较，1 个时钟周期返回最高优先级匹配结果。现代交换芯片 TCAM 容量约 1-16M 条目。</p>
      </div>
    </div>
  );
}
export default TCAMSimulator;
