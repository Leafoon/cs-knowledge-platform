"use client";
import { useState } from "react";

const ipv4Fields = [
  { name: "Version", bits: 4, note: "4" },
  { name: "IHL", bits: 4, note: "头部长度" },
  { name: "DSCP/ECN", bits: 8, note: "QoS" },
  { name: "Total Length", bits: 16, note: "总长度" },
  { name: "ID", bits: 16, note: "标识" },
  { name: "Flags+Offset", bits: 16, note: "分片" },
  { name: "TTL", bits: 8, note: "生存时间" },
  { name: "Protocol", bits: 8, note: "上层协议" },
  { name: "Checksum", bits: 16, note: "校验和" },
  { name: "Src IP", bits: 32, note: "32位源地址" },
  { name: "Dst IP", bits: 32, note: "32位目的地址" },
];

const ipv6Fields = [
  { name: "Version", bits: 4, note: "6" },
  { name: "Traffic Class", bits: 8, note: "流量类别" },
  { name: "Flow Label", bits: 20, note: "流标签(新)" },
  { name: "Payload Length", bits: 16, note: "载荷长度" },
  { name: "Next Header", bits: 8, note: "下一头部" },
  { name: "Hop Limit", bits: 8, note: "跳数限制" },
  { name: "Src IP", bits: 128, note: "128位源地址" },
  { name: "Dst IP", bits: 128, note: "128位目的地址" },
];

const diffs = [
  { feat: "头部长度", v4: "可变(20-60字节)", v6: "固定40字节" },
  { feat: "分片", v4: "路由器+端系统", v6: "仅端系统" },
  { feat: "校验和", v4: "有(每跳计算)", v6: "无(上层负责)" },
  { feat: "地址长度", v4: "32位", v6: "128位" },
  { feat: "IPSec", v4: "可选", v6: "内置支持" },
  { feat: "流标签", v4: "无", v6: "有(20位)" },
];

export function IPv6HeaderParser() {
  const [hoveredV4, setHoveredV4] = useState<number | null>(null);
  const [hoveredV6, setHoveredV6] = useState<number | null>(null);
  const totalV4 = ipv4Fields.reduce((s, f) => s + f.bits, 0);
  const totalV6 = ipv6Fields.reduce((s, f) => s + f.bits, 0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        IPv6 vs IPv4 Header <span className="text-text-secondary text-sm">— 头部对比</span>
      </h3>
      <div className="mb-4">
        <div className="text-sm font-semibold text-text-secondary mb-1">IPv4 Header ({totalV4 / 8} bytes)</div>
        <div className="flex flex-wrap gap-0.5">
          {ipv4Fields.map((f, i) => (
            <button
              key={i}
              onMouseEnter={() => setHoveredV4(i)}
              onMouseLeave={() => setHoveredV4(null)}
              className="text-xs p-1 rounded bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 transition-all"
              style={{ flex: `${Math.max(f.bits, 4)} 0 0` }}
            >
              {f.name}
            </button>
          ))}
        </div>
      </div>
      <div className="mb-4">
        <div className="text-sm font-semibold text-text-secondary mb-1">IPv6 Header ({totalV6 / 8} bytes)</div>
        <div className="flex flex-wrap gap-0.5">
          {ipv6Fields.map((f, i) => (
            <button
              key={i}
              onMouseEnter={() => setHoveredV6(i)}
              onMouseLeave={() => setHoveredV6(null)}
              className="text-xs p-1 rounded bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 transition-all"
              style={{ flex: `${Math.max(f.bits, 4)} 0 0` }}
            >
              {f.name}
            </button>
          ))}
        </div>
      </div>
      {hoveredV4 !== null && (
        <div className="bg-blue-50 dark:bg-blue-900/30 p-2 rounded text-sm mb-2">
          <strong>IPv4 - {ipv4Fields[hoveredV4].name}:</strong> {ipv4Fields[hoveredV4].bits} bits, {ipv4Fields[hoveredV4].note}
        </div>
      )}
      {hoveredV6 !== null && (
        <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded text-sm mb-2">
          <strong>IPv6 - {ipv6Fields[hoveredV6].name}:</strong> {ipv6Fields[hoveredV6].bits} bits, {ipv6Fields[hoveredV6].note}
        </div>
      )}
      <table className="w-full text-sm mt-4">
        <thead>
          <tr className="border-b border-border-subtle">
            <th className="text-left p-2 text-text-secondary">特性</th>
            <th className="text-left p-2 text-text-secondary">IPv4</th>
            <th className="text-left p-2 text-text-secondary">IPv6</th>
          </tr>
        </thead>
        <tbody>
          {diffs.map((d, i) => (
            <tr key={i} className="border-b border-border-subtle">
              <td className="p-2 text-text-primary">{d.feat}</td>
              <td className="p-2 text-text-secondary">{d.v4}</td>
              <td className="p-2 text-text-secondary">{d.v6}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default IPv6HeaderParser;
