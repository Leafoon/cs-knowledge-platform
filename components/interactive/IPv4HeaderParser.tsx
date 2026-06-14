"use client";
import { useState } from "react";

const fields = [
  { name: "Version", bits: 4, val: "4", desc: "IP版本号，IPv4固定为4" },
  { name: "IHL", bits: 4, val: "5", desc: "头部长度（32位字数），最小5=20字节" },
  { name: "DSCP", bits: 6, val: "0", desc: "差分服务代码点，用于QoS分类" },
  { name: "ECN", bits: 2, val: "0", desc: "显式拥塞通知" },
  { name: "Total Length", bits: 16, val: "1500", desc: "整个IP包长度（字节），最大65535" },
  { name: "Identification", bits: 16, val: "0x1A2B", desc: "分片标识，同一包的分片共享此值" },
  { name: "Flags", bits: 3, val: "010", desc: "DF=不分片, MF=还有分片" },
  { name: "Fragment Offset", bits: 13, val: "0", desc: "分片偏移（8字节为单位）" },
  { name: "TTL", bits: 8, val: "64", desc: "生存时间，每经过一跳减1，防环路" },
  { name: "Protocol", bits: 8, val: "6", desc: "上层协议：6=TCP, 17=UDP, 1=ICMP" },
  { name: "Header Checksum", bits: 16, val: "0x3A5C", desc: "头部校验和，每跳重新计算" },
  { name: "Source IP", bits: 32, val: "192.168.1.1", desc: "源IP地址" },
  { name: "Destination IP", bits: 32, val: "10.0.0.1", desc: "目的IP地址" },
];

const hexByteView = "45 00 05 DC 1A 2B 40 00 40 06 3A 5C C0 A8 01 01 0A 00 00 01";

const protocolMap: Record<string, string> = {
  "1": "ICMP",
  "6": "TCP",
  "17": "UDP",
  "89": "OSPF",
};

const flagMeanings: Record<string, string> = {
  "000": "保留位，允许分片",
  "010": "Don't Fragment (DF)",
  "001": "More Fragments (MF)",
  "100": "保留",
};

export function IPv4HeaderParser() {
  const [selected, setSelected] = useState<number | null>(null);
  const [showHex, setShowHex] = useState(false);
  const totalBits = fields.reduce((s, f) => s + f.bits, 0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        IPv4 Header Parser <span className="text-text-secondary text-sm">— 逐字段解析</span>
      </h3>
      <div className="flex flex-wrap gap-0.5 mb-4">
        {fields.map((f, i) => {
          const width = (f.bits / totalBits) * 100;
          return (
            <button
              key={i}
              onClick={() => setSelected(selected === i ? null : i)}
              className={`text-xs p-1.5 rounded transition-all ${selected === i ? "bg-blue-600 text-white ring-2 ring-blue-400" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600"}`}
              style={{ minWidth: `${Math.max(width, 5)}%`, flex: `${f.bits} 0 0` }}
            >
              {f.name}
            </button>
          );
        })}
      </div>
      <div className="text-xs text-text-secondary mb-2">
        共 {totalBits / 8} 字节（{totalBits} 位）
      </div>
      {selected !== null && (
        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
          <div className="font-semibold text-text-primary">{fields[selected].name}</div>
          <div className="text-sm text-text-secondary">
            位数: {fields[selected].bits} bits | 值: {fields[selected].val}
          </div>
          <div className="text-sm text-text-secondary mt-1">{fields[selected].desc}</div>
          {fields[selected].name === "Protocol" && (
            <div className="mt-2 text-xs">
              <span className="text-text-secondary">常用协议: </span>
              {Object.entries(protocolMap).map(([k, v]) => (
                <span key={k} className="mr-2 px-1 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-text-primary">
                  {k}={v}
                </span>
              ))}
            </div>
          )}
          {fields[selected].name === "Flags" && (
            <div className="mt-2 text-xs">
              <span className="text-text-secondary">标志位含义: </span>
              <span className="text-text-primary">{flagMeanings[fields[selected].val] || "自定义"}</span>
            </div>
          )}
          {fields[selected].name === "TTL" && (
            <div className="mt-2 text-xs text-text-secondary">
              常见默认值: Windows=128, Linux=64, macOS=64
            </div>
          )}
        </div>
      )}
      {selected === null && (
        <div className="text-sm text-text-secondary italic">点击任意字段查看详情</div>
      )}
      <div className="mt-4">
        <button
          onClick={() => setShowHex(!showHex)}
          className="px-3 py-1 rounded bg-gray-600 text-white text-sm"
        >
          {showHex ? "隐藏" : "显示"}十六进制视图
        </button>
        {showHex && (
          <div className="mt-2 bg-gray-900 p-3 rounded font-mono text-xs text-green-400">
            {hexByteView}
          </div>
        )}
      </div>
    </div>
  );
}

export default IPv4HeaderParser;
