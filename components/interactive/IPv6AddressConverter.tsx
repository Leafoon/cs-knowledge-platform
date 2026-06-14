"use client";
import { useState } from "react";

function expandIPv6(addr: string): string {
  if (addr.includes("::")) {
    const parts = addr.split("::");
    const left = parts[0] ? parts[0].split(":") : [];
    const right = parts[1] ? parts[1].split(":") : [];
    const missing = 8 - left.length - right.length;
    const expanded = [...left, ...Array(missing).fill("0000"), ...right];
    return expanded.map((g) => g.padStart(4, "0")).join(":");
  }
  return addr
    .split(":")
    .map((g) => g.padStart(4, "0"))
    .join(":");
}

function compressIPv6(addr: string): string {
  const groups = addr.split(":").map((g) => g.replace(/^0+/, "") || "0");
  let bestStart = -1,
    bestLen = 0,
    curStart = -1,
    curLen = 0;
  for (let i = 0; i <= groups.length; i++) {
    if (i < groups.length && groups[i] === "0") {
      if (curStart === -1) curStart = i;
      curLen = i - curStart + 1;
    } else {
      if (curLen > bestLen) {
        bestStart = curStart;
        bestLen = curLen;
      }
      curStart = -1;
      curLen = 0;
    }
  }
  if (bestLen < 2) return groups.join(":");
  const before = groups.slice(0, bestStart);
  const after = groups.slice(bestStart + bestLen);
  return (before.length ? before.join(":") : "") + "::" + (after.length ? after.join(":") : "");
}

function getIPv6Type(addr: string): string {
  const expanded = expandIPv6(addr).replace(/:/g, "");
  if (expanded.startsWith("fe80")) return "Link-Local 链路本地";
  if (expanded.startsWith("fc") || expanded.startsWith("fd"))
    return "Unique Local 唯一本地";
  if (expanded.startsWith("ff")) return "Multicast 组播";
  if (expanded === "00000000000000000000000000000001") return "Loopback 回环";
  if (expanded.startsWith("2001")) return "Global Unicast 全球单播";
  if (expanded.startsWith("2002")) return "6to4 隧道";
  return "Global Unicast 全球单播";
}

export function IPv6AddressConverter() {
  const [input, setInput] = useState("2001:db8::1");
  const [error, setError] = useState("");

  const expanded = (() => {
    try {
      setError("");
      return expandIPv6(input);
    } catch {
      setError("格式无效");
      return "";
    }
  })();

  const compressed = expanded ? compressIPv6(expanded) : "";
  const addrType = expanded ? getIPv6Type(input) : "";

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        IPv6 Address Converter <span className="text-text-secondary text-sm">— 地址压缩/展开/类型</span>
      </h3>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-primary font-mono mb-3"
        placeholder="输入IPv6地址"
      />
      {error && <div className="text-red-500 text-sm mb-2">{error}</div>}
      {!error && expanded && (
        <div className="space-y-2 text-sm">
          <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded">
            <span className="text-text-secondary">展开: </span>
            <span className="font-mono text-text-primary">{expanded}</span>
          </div>
          <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded">
            <span className="text-text-secondary">压缩: </span>
            <span className="font-mono text-text-primary">{compressed}</span>
          </div>
          <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded">
            <span className="text-text-secondary">类型: </span>
            <span className="text-blue-600 dark:text-blue-400 font-semibold">{addrType}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default IPv6AddressConverter;
