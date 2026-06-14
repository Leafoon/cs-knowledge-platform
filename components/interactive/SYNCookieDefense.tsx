"use client";
import { useState } from "react";

export function SYNCookieDefense() {
  const [attackRate, setAttackRate] = useState(1000);
  const [cookieEnabled, setCookieEnabled] = useState(true);
  const [connections, setConnections] = useState(0);
  const [halfOpen, setHalfOpen] = useState(0);
  const [dropped, setDropped] = useState(0);

  const simulate = () => {
    const legit = Math.floor(attackRate * 0.05);
    const malicious = attackRate - legit;
    if (cookieEnabled) {
      setConnections(legit);
      setHalfOpen(0);
      setDropped(malicious);
    } else {
      setConnections(legit);
      setHalfOpen(Math.min(malicious, 10000));
      setDropped(Math.max(0, malicious - 10000));
    }
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">SYN Cookie防御机制</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          SYN包速率: <span className="text-text-primary font-mono">{attackRate}/s</span>
          <input type="range" min={100} max={10000} step={100} value={attackRate} onChange={(e) => setAttackRate(+e.target.value)} className="w-full mt-1" />
        </label>
        <div className="flex items-end">
          <button onClick={() => setCookieEnabled(!cookieEnabled)}
            className={`px-4 py-1.5 rounded-lg border text-xs font-medium transition-all ${cookieEnabled ? "bg-emerald-500/20 border-emerald-400/40 text-emerald-700 dark:text-emerald-300" : "bg-red-500/20 border-red-400/40 text-red-700 dark:text-red-300"}`}>
            SYN Cookie: {cookieEnabled ? "开启" : "关闭"}
          </button>
        </div>
      </div>
      <button onClick={simulate} className="w-full px-4 py-2 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-sm font-medium hover:bg-sky-500/25 transition-colors mb-4">
        模拟SYN Flood攻击
      </button>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-emerald-500">{connections}</div>
          <div className="text-[10px] text-text-tertiary">合法连接</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className={`text-lg font-bold ${halfOpen > 0 ? "text-red-500" : "text-gray-400"}`}>{halfOpen}</div>
          <div className="text-[10px] text-text-tertiary">半开连接</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-center">
          <div className="text-lg font-bold text-amber-500">{dropped}</div>
          <div className="text-[10px] text-text-tertiary">丢弃包</div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className={`rounded-lg border p-3 text-xs space-y-1 ${cookieEnabled ? "border-emerald-500/30 bg-emerald-500/5" : "border-border-subtle bg-bg-tertiary"}`}>
          <div className="font-medium text-emerald-600 dark:text-emerald-400">SYN Cookie工作原理</div>
          <div className="text-text-secondary">1. 不分配TCB资源，仅计算Cookie</div>
          <div className="text-text-secondary">2. Cookie = f(srcIP, srcPort, dstIP, dstPort, seq)</div>
          <div className="text-text-secondary">3. 收到ACK时验证Cookie，合法才分配资源</div>
          <div className="text-text-secondary">4. 无状态，不消耗服务器内存</div>
        </div>
        <div className={`rounded-lg border p-3 text-xs space-y-1 ${!cookieEnabled ? "border-red-500/30 bg-red-500/5" : "border-border-subtle bg-bg-tertiary"}`}>
          <div className="font-medium text-red-600 dark:text-red-400">无Cookie防护</div>
          <div className="text-text-secondary">1. 每个SYN分配一个TCB（传输控制块）</div>
          <div className="text-text-secondary">2. 半开连接队列很快被填满</div>
          <div className="text-text-secondary">3. 合法连接无法建立（SYN队列溢出）</div>
          <div className="text-text-secondary">4. 内耗尽导致服务器拒绝服务</div>
        </div>
      </div>
      <div className="text-[10px] text-text-tertiary mb-3">注意：SYN Cookie会禁用TCP的一些扩展选项（如窗口缩放、时间戳）</div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">SYN Cookie 代价</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 禁用窗口缩放 (Window Scale)</li>
            <li>• 禁用 SACK 选项</li>
            <li>• 时间戳编码部分信息</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">替代防御方案</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• SYN Cache: 有限状态缓存表</li>
            <li>• SYN Proxy: 代理三次握手</li>
            <li>• 防火墙 rate limiting</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default SYNCookieDefense;
