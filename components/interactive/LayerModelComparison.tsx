"use client";
import { useState } from "react";

const osiLayers = [
  { num: 7, name: "Application", zh: "应用层", desc: "用户接口与网络服务", protocols: "HTTP, FTP, SMTP, DNS" },
  { num: 6, name: "Presentation", zh: "表示层", desc: "数据格式转换、加密、压缩", protocols: "SSL/TLS, JPEG, ASCII" },
  { num: 5, name: "Session", zh: "会话层", desc: "建立、管理、终止会话", protocols: "NetBIOS, RPC" },
  { num: 4, name: "Transport", zh: "传输层", desc: "端到端可靠传输", protocols: "TCP, UDP" },
  { num: 3, name: "Network", zh: "网络层", desc: "路由与逻辑寻址", protocols: "IP, ICMP, OSPF" },
  { num: 2, name: "Data Link", zh: "数据链路层", desc: "帧封装与MAC寻址", protocols: "Ethernet, Wi-Fi, PPP" },
  { num: 1, name: "Physical", zh: "物理层", desc: "比特流传输", protocols: "RJ45,光纤,无线电" },
];

const tcpipLayers = [
  { num: 4, name: "Application", zh: "应用层", desc: "合并了OSI上三层", protocols: "HTTP, DNS, SMTP, FTP" },
  { num: 3, name: "Transport", zh: "传输层", desc: "端到端传输", protocols: "TCP, UDP" },
  { num: 2, name: "Internet", zh: "网际层", desc: "IP路由与寻址", protocols: "IP, ICMP, ARP" },
  { num: 1, name: "Network Access", zh: "网络接入层", desc: "物理+数据链路", protocols: "Ethernet, Wi-Fi" },
];

export function LayerModelComparison() {
  const [highlightOSI, setHighlightOSI] = useState<number | null>(null);
  const [highlightTCPIP, setHighlightTCPIP] = useState<number | null>(null);

  const mapping: Record<number, number[]> = { 1: [1], 2: [1, 2], 3: [2, 3], 4: [3, 4], 5: [4], 6: [4], 7: [4] };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        OSI vs TCP/IP <span className="text-text-secondary text-sm">— 并排对比</span>
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm font-semibold text-text-secondary mb-2 text-center">OSI 7层模型</div>
          {osiLayers.map((l) => {
            const active = highlightOSI === l.num || (highlightTCPIP !== null && mapping[highlightTCPIP]?.includes(l.num));
            return (
              <button
                key={l.num}
                onMouseEnter={() => setHighlightOSI(l.num)}
                onMouseLeave={() => setHighlightOSI(null)}
                className={`w-full text-left p-2 mb-1 rounded text-sm transition-all ${active ? "bg-blue-100 dark:bg-blue-900/40 ring-1 ring-blue-400" : "bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"}`}
              >
                <span className="font-mono text-text-secondary mr-2">L{l.num}</span>
                <span className="text-text-primary font-medium">{l.name}</span>
                <span className="text-text-secondary ml-1">({l.zh})</span>
              </button>
            );
          })}
        </div>
        <div>
          <div className="text-sm font-semibold text-text-secondary mb-2 text-center">TCP/IP 4层模型</div>
          {tcpipLayers.map((l) => {
            const heights = [4, 1, 1, 2];
            const active = highlightTCPIP === l.num || (highlightOSI !== null && mapping[highlightOSI]?.includes(l.num));
            return (
              <button
                key={l.num}
                onMouseEnter={() => setHighlightTCPIP(l.num)}
                onMouseLeave={() => setHighlightTCPIP(null)}
                className={`w-full text-left p-2 mb-1 rounded text-sm transition-all ${active ? "bg-green-100 dark:bg-green-900/40 ring-1 ring-green-400" : "bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"}`}
                style={{ minHeight: `${heights[l.num - 1] * 36}px` }}
              >
                <span className="font-mono text-text-secondary mr-2">L{l.num}</span>
                <span className="text-text-primary font-medium">{l.name}</span>
                <span className="text-text-secondary ml-1">({l.zh})</span>
                <div className="text-xs text-text-secondary mt-1">{l.desc}</div>
              </button>
            );
          })}
        </div>
      </div>
      {(highlightOSI !== null || highlightTCPIP !== null) && (
        <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded mt-3 text-sm">
          {(highlightOSI !== null ? osiLayers.find((l) => l.num === highlightOSI) : tcpipLayers.find((l) => l.num === highlightTCPIP)) && (
            <>
              <div className="text-text-primary font-medium">
                {(highlightOSI !== null ? osiLayers.find((l) => l.num === highlightOSI)! : tcpipLayers.find((l) => l.num === highlightTCPIP)!).desc}
              </div>
              <div className="text-text-secondary text-xs mt-1">
                协议: {(highlightOSI !== null ? osiLayers.find((l) => l.num === highlightOSI)! : tcpipLayers.find((l) => l.num === highlightTCPIP)!).protocols}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default LayerModelComparison;
