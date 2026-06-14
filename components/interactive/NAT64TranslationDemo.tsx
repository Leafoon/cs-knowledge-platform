"use client";
import { useState } from "react";

interface TranslationEntry {
  ipv6Src: string;
  ipv6Dst: string;
  ipv4Src: string;
  ipv4Dst: string;
  protocol: string;
}

const pool = "203.0.113.1";

const examples: TranslationEntry[] = [
  {
    ipv6Src: "2001:db8::10",
    ipv6Dst: "64:ff9b::192.168.1.1",
    ipv4Src: pool,
    ipv4Dst: "192.168.1.1",
    protocol: "TCP",
  },
  {
    ipv6Src: "2001:db8::20",
    ipv6Dst: "64:ff9b::10.0.0.5",
    ipv4Src: pool,
    ipv4Dst: "10.0.0.5",
    protocol: "UDP",
  },
  {
    ipv6Src: "fd00::30",
    ipv6Dst: "64:ff9b::172.16.0.1",
    ipv4Src: pool,
    ipv4Dst: "172.16.0.1",
    protocol: "ICMP",
  },
];

export function NAT64TranslationDemo() {
  const [selected, setSelected] = useState(0);
  const [step, setStep] = useState(0);

  const entry = examples[selected];

  const steps = [
    { label: "IPv6数据包入站", detail: `源: ${entry.ipv6Src} → 目的: ${entry.ipv6Dst}` },
    { label: "DNS64合成AAAA记录", detail: "将A记录的IPv4地址嵌入64:ff9b::/96前缀" },
    { label: "NAT64地址转换", detail: `IPv6源 → IPv4: ${entry.ipv6Src} → ${entry.ipv4Src}` },
    { label: "协议头转换", detail: "IPv6头 → IPv4头，ICMPv6 → ICMP" },
    { label: "IPv4数据包出站", detail: `源: ${entry.ipv4Src} → 目的: ${entry.ipv4Dst}` },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        NAT64 Translation <span className="text-text-secondary text-sm">— IPv6→IPv4 协议转换</span>
      </h3>
      <div className="flex gap-2 mb-4">
        {examples.map((_, i) => (
          <button
            key={i}
            onClick={() => { setSelected(i); setStep(0); }}
            className={`px-3 py-1 rounded text-sm ${selected === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            示例 {i + 1}
          </button>
        ))}
      </div>
      <div className="flex gap-1 mb-3 flex-wrap">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`px-2 py-1 rounded text-xs ${step === i ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
          >
            {i + 1}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">步骤 {step + 1}: {steps[step].label}</div>
        <div className="text-sm text-text-secondary font-mono">{steps[step].detail}</div>
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-blue-50 dark:bg-blue-900/30 p-3 rounded">
          <div className="font-semibold text-blue-700 dark:text-blue-300 mb-1">IPv6侧</div>
          <div className="text-text-secondary">源: <span className="font-mono">{entry.ipv6Src}</span></div>
          <div className="text-text-secondary">目的: <span className="font-mono">{entry.ipv6Dst}</span></div>
        </div>
        <div className="bg-green-50 dark:bg-green-900/30 p-3 rounded">
          <div className="font-semibold text-green-700 dark:text-green-300 mb-1">IPv4侧</div>
          <div className="text-text-secondary">源: <span className="font-mono">{entry.ipv4Src}</span></div>
          <div className="text-text-secondary">目的: <span className="font-mono">{entry.ipv4Dst}</span></div>
        </div>
      </div>
      <div className="mt-3 text-xs text-text-secondary">
        Well-Known Prefix: 64:ff9b::/96 | 地址池: {pool} | 协议: {entry.protocol}
      </div>
    </div>
  );
}

export default NAT64TranslationDemo;
