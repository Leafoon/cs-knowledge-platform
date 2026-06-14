"use client";
import { useState } from "react";

const features = [
  { name: "传输协议", http1: "TCP", http11: "TCP", http2: "TCP", http3: "QUIC (UDP)" },
  { name: "头部压缩", http1: "无", http11: "无", http2: "HPACK", http3: "QPACK" },
  { name: "多路复用", http1: "×", http11: "×", http2: "✓ 流级", http3: "✓ 流级" },
  { name: "队头阻塞", http1: "有", http11: "有", http2: "TCP层有", http3: "无" },
  { name: "服务器推送", http1: "×", http11: "×", http2: "✓", http3: "✓" },
  { name: "二进制协议", http1: "×", http11: "×", http2: "✓", http3: "✓" },
  { name: "连接迁移", http1: "×", http11: "×", http2: "×", http3: "✓ Connection ID" },
  { name: "0-RTT建立", http1: "×", http11: "×", http2: "×", http3: "✓" },
  { name: "TLS要求", http1: "可选", http11: "可选", http2: "实践要求", http3: "强制加密" },
  { name: "流控制", http1: "TCP级", http11: "TCP级", http2: "流级", http3: "流级" },
];

const versions = ["http1", "http11", "http2", "http3"] as const;
const labels: Record<string, string> = { http1: "HTTP/1.0", http11: "HTTP/1.1", http2: "HTTP/2", http3: "HTTP/3" };

export function HTTPVersionComparison() {
  const [highlight, setHighlight] = useState<string | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">HTTP 版本对比</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left py-2 px-2 text-text-secondary font-medium">特性</th>
              {versions.map((v) => (
                <th key={v} className={`py-2 px-2 text-center font-medium cursor-pointer transition-all ${
                  highlight === v ? "text-blue-400 bg-blue-500/10" : "text-text-secondary"
                }`} onMouseEnter={() => setHighlight(v)} onMouseLeave={() => setHighlight(null)}>
                  {labels[v]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((f) => (
              <tr key={f.name} className="border-b border-border-subtle/50 hover:bg-bg-subtle">
                <td className="py-2 px-2 text-text-primary font-medium">{f.name}</td>
                {versions.map((v) => {
                  const val = f[v];
                  const isPositive = val.includes("✓") || val.includes("无");
                  const isNegative = val === "×" || val === "有" || val.includes("TCP层");
                  return (
                    <td key={v} className={`py-2 px-2 text-center font-mono text-xs ${
                      isPositive ? "text-green-400" : isNegative ? "text-red-400" : "text-text-secondary"
                    } ${highlight === v ? "bg-blue-500/5" : ""}`}>
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
export default HTTPVersionComparison;
