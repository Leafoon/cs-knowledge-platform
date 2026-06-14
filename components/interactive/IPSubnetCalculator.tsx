"use client";
import { useState } from "react";

export function IPSubnetCalculator() {
  const [ip, setIp] = useState("192.168.1.100");
  const [prefix, setPrefix] = useState(24);

  const parseIp = (ipStr: string): number[] | null => {
    const parts = ipStr.split(".").map(Number);
    if (parts.length !== 4 || parts.some((p) => isNaN(p) || p < 0 || p > 255)) return null;
    return parts;
  };

  const ipParts = parseIp(ip);
  const valid = ipParts !== null;

  const mask = valid ? Array(4).fill(0).map((_, i) => {
    const bitsInOctet = Math.min(8, Math.max(0, prefix - i * 8));
    return 256 - (1 << (8 - bitsInOctet));
  }) : [0, 0, 0, 0];

  const network = valid ? ipParts.map((p, i) => p & mask[i]) : [0, 0, 0, 0];
  const broadcast = valid ? network.map((p, i) => p | (~mask[i] & 255)) : [0, 0, 0, 0];
  const firstHost = valid ? [...network] : [0, 0, 0, 0];
  const lastHost = valid ? [...broadcast] : [0, 0, 0, 0];

  if (prefix < 31) {
    firstHost[3] += 1;
    lastHost[3] -= 1;
  } else if (prefix === 31) {
    firstHost[3] = network[3];
    lastHost[3] = broadcast[3];
  }

  const hostBits = 32 - prefix;
  const totalHosts = prefix >= 31 ? 2 : Math.max(0, (1 << hostBits) - 2);

  const toBin = (parts: number[]) => parts.map((p) => p.toString(2).padStart(8, "0")).join(".");

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">IP子网计算器</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary mb-1 block">IP地址:</label>
          <input type="text" value={ip} onChange={(e) => setIp(e.target.value)}
            className={`w-full px-3 py-2 rounded border font-mono text-sm ${valid ? "border-border-subtle bg-bg-subtle text-text-primary" : "border-red-500 bg-red-50 dark:bg-red-900/20 text-red-500"}`} />
        </div>
        <div>
          <label className="text-sm text-text-secondary mb-1 block">前缀长度: /{prefix}</label>
          <input type="range" min={8} max={30} value={prefix} onChange={(e) => setPrefix(Number(e.target.value))} className="w-full" />
        </div>
      </div>
      {valid && (
        <div className="space-y-3">
          <div className="bg-bg-muted rounded-lg p-4">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div><strong className="text-text-primary">子网掩码:</strong> <span className="font-mono text-text-secondary">{mask.join(".")}</span></div>
              <div><strong className="text-text-primary">掩码二进制:</strong> <span className="font-mono text-xs text-text-secondary">{toBin(mask)}</span></div>
              <div><strong className="text-text-primary">网络地址:</strong> <span className="font-mono text-blue-500">{network.join(".")}/{prefix}</span></div>
              <div><strong className="text-text-primary">广播地址:</strong> <span className="font-mono text-red-500">{broadcast.join(".")}</span></div>
              <div><strong className="text-text-primary">第一个主机:</strong> <span className="font-mono text-green-500">{firstHost.join(".")}</span></div>
              <div><strong className="text-text-primary">最后一个主机:</strong> <span className="font-mono text-green-500">{lastHost.join(".")}</span></div>
              <div><strong className="text-text-primary">可用主机数:</strong> <span className="font-mono text-text-secondary">{totalHosts}</span></div>
              <div><strong className="text-text-primary">主机位数:</strong> <span className="font-mono text-text-secondary">{hostBits} 位</span></div>
            </div>
          </div>
          <div className="bg-bg-subtle rounded p-3 font-mono text-xs">
            <div className="text-text-secondary mb-1">IP二进制: {toBin(ipParts)}</div>
            <div className="text-text-secondary">
              网络部分: <span className="text-blue-500">{toBin(ipParts).slice(0, prefix + Math.floor(prefix / 8))}</span>
              主机部分: <span className="text-green-500">{toBin(ipParts).slice(prefix + Math.floor(prefix / 8))}</span>
            </div>
          </div>
        </div>
      )}
      <div className="text-xs text-text-secondary mt-3">
        子网划分: 将IP地址分为网络部分和主机部分。/24表示前24位为网络号,后8位为主机号,可容纳254台主机(减去网络地址和广播地址)。
      </div>
    </div>
  );
}

export default IPSubnetCalculator;
