"use client";
import { useState } from "react";

export function PacketCaptureEngine() {
  const [filter, setFilter] = useState("tcp port 80");
  const [capturing, setCapturing] = useState(false);
  const [packets, setPackets] = useState<{ id: number; time: string; src: string; dst: string; proto: string; len: number; info: string }[]>([]);

  const samplePackets = [
    { src: "192.168.1.100", dst: "142.250.80.46", proto: "TCP", len: 66, info: "SYN Seq=0 Win=65535" },
    { src: "142.250.80.46", dst: "192.168.1.100", proto: "TCP", len: 66, info: "SYN, ACK Seq=0 Ack=1" },
    { src: "192.168.1.100", dst: "142.250.80.46", proto: "TCP", len: 54, info: "ACK Seq=1 Ack=1" },
    { src: "192.168.1.100", dst: "142.250.80.46", proto: "HTTP", len: 517, info: "GET / HTTP/1.1" },
    { src: "142.250.80.46", dst: "192.168.1.100", proto: "HTTP", len: 1460, info: "HTTP/1.1 200 OK (data)" },
    { src: "142.250.80.46", dst: "192.168.1.100", proto: "HTTP", len: 256, info: "HTTP/1.1 200 OK (end)" },
    { src: "192.168.1.100", dst: "8.8.8.8", proto: "DNS", len: 72, info: "Standard query A google.com" },
    { src: "8.8.8.8", dst: "192.168.1.100", proto: "DNS", len: 128, info: "Standard response A 142.250.80.46" },
  ];

  const startCapture = () => {
    setCapturing(true);
    setPackets([]);
    let i = 0;
    const iv = setInterval(() => {
      if (i < samplePackets.length) {
        const pkt = samplePackets[i];
        setPackets(prev => [...prev, {
          id: prev.length + 1, time: `${(Date.now() / 1000).toFixed(6)}`,
          ...pkt,
        }]);
        i++;
      } else {
        clearInterval(iv);
        setCapturing(false);
      }
    }, 400);
  };

  const filters = ["tcp port 80", "udp port 53", "host 192.168.1.100", "tcp[tcpflags] & tcp-syn != 0", "ip proto icmp"];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">🔍 抓包引擎演示</h3>
      <p className="text-sm text-text-secondary mb-4">展示 BPF 过滤和数据包捕获过程</p>

      <div className="flex gap-2 mb-3">
        <input value={filter} onChange={e => setFilter(e.target.value)}
          className="flex-1 bg-bg-surface border border-border-subtle rounded p-2 text-sm text-text-primary font-mono" placeholder="BPF 过滤器" />
        <button onClick={startCapture} disabled={capturing}
          className={`px-4 py-2 rounded text-sm font-medium ${capturing ? "bg-red-600 text-white" : "bg-green-600 text-white hover:bg-green-700"}`}>
          {capturing ? "⏹ 停止" : "▶ 开始抓包"}
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5 mb-4">
        {filters.map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className="px-2 py-1 bg-bg-surface border border-border-subtle rounded text-[10px] font-mono text-text-secondary hover:border-blue-400">
            {f}
          </button>
        ))}
      </div>

      <div className="mb-4 bg-bg-surface rounded-lg border border-border-subtle overflow-hidden">
        <div className="flex items-center gap-1 p-2 bg-bg-elevated border-b border-border-subtle text-[10px] text-text-secondary">
          <span className="w-8">#</span>
          <span className="w-24">时间</span>
          <span className="flex-1">源地址</span>
          <span className="flex-1">目的地址</span>
          <span className="w-12">协议</span>
          <span className="w-12 text-right">长度</span>
          <span className="flex-1">信息</span>
        </div>
        <div className="max-h-48 overflow-y-auto">
          {packets.length === 0 && (
            <div className="p-4 text-center text-xs text-text-secondary">{capturing ? "等待数据包..." : "点击开始抓包"}</div>
          )}
          {packets.map((p, i) => (
            <div key={i} className={`flex items-center gap-1 p-1.5 text-[10px] font-mono border-b border-border-subtle/50 ${p.proto === "HTTP" ? "bg-green-900/10" : p.proto === "DNS" ? "bg-yellow-900/10" : "bg-bg-surface"}`}>
              <span className="w-8 text-text-secondary">{p.id}</span>
              <span className="w-24 text-text-secondary">{p.time}</span>
              <span className="flex-1 text-blue-300 truncate">{p.src}</span>
              <span className="flex-1 text-green-300 truncate">{p.dst}</span>
              <span className="w-12"><span className={`px-1 py-0.5 rounded text-[9px] ${p.proto === "HTTP" ? "bg-green-700 text-green-200" : p.proto === "DNS" ? "bg-yellow-700 text-yellow-200" : "bg-blue-700 text-blue-200"}`}>{p.proto}</span></span>
              <span className="w-12 text-right text-text-primary">{p.len}</span>
              <span className="flex-1 text-text-secondary truncate">{p.info}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-bg-surface rounded-lg p-3 text-xs text-text-secondary">
        <strong className="text-text-primary">BPF (Berkeley Packet Filter)：</strong>在内核态执行的过滤器字节码，只将匹配的数据包复制到用户空间，大幅降低抓包开销。
      </div>
    </div>
  );
}
export default PacketCaptureEngine;
