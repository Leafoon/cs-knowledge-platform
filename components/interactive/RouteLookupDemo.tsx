"use client";
import { useState, useMemo } from "react";

interface Route {
  prefix: string;
  len: number;
  nextHop: string;
  iface: string;
}

const routes: Route[] = [
  { prefix: "0.0.0.0", len: 0, nextHop: "10.0.0.1", iface: "eth0 (默认)" },
  { prefix: "10.0.0.0", len: 8, nextHop: "直连", iface: "eth0" },
  { prefix: "172.16.0.0", len: 12, nextHop: "直连", iface: "eth1" },
  { prefix: "192.168.1.0", len: 24, nextHop: "10.0.0.2", iface: "eth0" },
  { prefix: "192.168.1.128", len: 25, nextHop: "10.0.0.3", iface: "eth0" },
  { prefix: "192.168.2.0", len: 24, nextHop: "172.16.0.1", iface: "eth1" },
  { prefix: "203.0.113.0", len: 24, nextHop: "10.0.0.4", iface: "eth0" },
];

function ipToBin(ip: string): string {
  return ip.split(".").map((o) => parseInt(o).toString(2).padStart(8, "0")).join("");
}

export function RouteLookupDemo() {
  const [destIP, setDestIP] = useState("192.168.1.100");
  const [showTrie, setShowTrie] = useState(false);

  const matchResult = useMemo(() => {
    const destBin = ipToBin(destIP);
    let bestMatch: Route | null = null;
    let bestLen = -1;
    const checked: { route: Route; match: boolean; bits: number }[] = [];
    for (const route of routes) {
      const prefixBin = ipToBin(route.prefix);
      let match = true;
      for (let i = 0; i < route.len; i++) {
        if (destBin[i] !== prefixBin[i]) { match = false; break; }
      }
      checked.push({ route, match, bits: route.len });
      if (match && route.len > bestLen) {
        bestLen = route.len;
        bestMatch = route;
      }
    }
    return { bestMatch, checked };
  }, [destIP]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">路由查找演示（最长前缀匹配）</h3>
      <label className="text-xs text-text-secondary block mb-4">
        目的IP: <input type="text" value={destIP} onChange={(e) => setDestIP(e.target.value)}
          className="ml-2 px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-text-primary font-mono text-sm w-48" placeholder="如 192.168.1.100" />
      </label>
      <div className="space-y-1.5 mb-4">
        {matchResult.checked.map((c, i) => (
          <div key={i} className={`flex items-center gap-3 px-3 py-2 rounded-lg border text-xs font-mono transition-all ${c.match ? "bg-emerald-500/10 border-emerald-500/30" : "bg-bg-tertiary border-border-subtle opacity-60"}`}>
            <span className="w-32">{c.route.prefix}/{c.route.len}</span>
            <span className="text-text-tertiary">→</span>
            <span className="text-text-primary">{c.route.nextHop}</span>
            <span className="text-text-tertiary ml-auto">{c.route.iface}</span>
            {c.match && <span className="text-[10px] text-emerald-500">前缀匹配 {c.bits}位</span>}
          </div>
        ))}
      </div>
      {matchResult.bestMatch && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs">
          <span className="font-medium text-emerald-600 dark:text-emerald-400">最长前缀匹配结果：</span>
          <span className="font-mono text-text-primary ml-2">{destIP} → {matchResult.bestMatch.prefix}/{matchResult.bestMatch.len} → {matchResult.bestMatch.nextHop} ({matchResult.bestMatch.iface})</span>
        </div>
      )}
      <button onClick={() => setShowTrie(!showTrie)} className="mt-3 text-xs text-sky-600 dark:text-sky-400 hover:underline">
        {showTrie ? "隐藏" : "显示"}Trie树查找说明
      </button>
      {showTrie && (
        <div className="mt-2 rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
          <div className="font-medium text-text-primary">Trie（前缀树）加速路由查找</div>
          <div>• 每个节点代表IP地址的一位（0或1）</div>
          <div>• 从根到叶子的路径对应一个前缀</div>
          <div>• 查找时沿Trie树向下，沿途记录最长匹配前缀</div>
          <div>• 时间复杂度 O(W)，W=地址位数（IPv4=32），不受路由表大小影响</div>
          <div>• 实际使用压缩Trie（Patricia树）减少内存和跳转次数</div>
        </div>
      )}
    </div>
  );
}
export default RouteLookupDemo;
