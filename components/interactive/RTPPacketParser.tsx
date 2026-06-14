"use client";
import { useState } from "react";

interface RTPField {
  name: string;
  bits: number;
  value: string;
  desc: string;
  color: string;
}

const defaultFields: RTPField[] = [
  { name: "V", bits: 2, value: "10", desc: "版本号 = 2", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "P", bits: 1, value: "0", desc: "填充位（无填充）", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "X", bits: 1, value: "0", desc: "扩展位（无扩展头）", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "CC", bits: 4, value: "0001", desc: "CSRC计数 = 1", color: "bg-emerald-500/20 border-emerald-400/40" },
  { name: "M", bits: 1, value: "0", desc: "标记位", color: "bg-amber-500/20 border-amber-400/40" },
  { name: "PT", bits: 7, value: "1100000", desc: "载荷类型 = 96（动态）", color: "bg-amber-500/20 border-amber-400/40" },
  { name: "Sequence Number", bits: 16, value: "0000000100101100", desc: "序列号 = 300（用于排序和丢包检测）", color: "bg-violet-500/20 border-violet-400/40" },
  { name: "Timestamp", bits: 32, value: "00001001011000010101110011000000", desc: "时间戳 = 157506624（采样时钟）", color: "bg-rose-500/20 border-rose-400/40" },
  { name: "SSRC", bits: 32, value: "01010101010101010101010101010101", desc: "同步源标识 = 0x55555555", color: "bg-pink-500/20 border-pink-400/40" },
];

export function RTPPacketParser() {
  const [fields] = useState<RTPField[]>(defaultFields);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [seqNum, setSeqNum] = useState(300);
  const [timestamp, setTimestamp] = useState(157506624);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">RTP报文解析器</h3>
      <div className="flex flex-wrap gap-0.5 mb-4">
        {fields.map((f, i) => (
          <div key={i} onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)}
            className={`px-2 py-2 border text-[10px] font-mono text-center cursor-pointer transition-all ${f.color} ${hoverIdx === i ? "ring-2 ring-sky-400 scale-105 z-10" : ""}`}
            style={{ minWidth: `${Math.max(32, f.bits * 2.5)}px`, flex: `${Math.max(f.bits, 2)} 0 auto` }}>
            <div className="text-text-primary font-medium">{f.name}</div>
            <div className="text-text-tertiary">{f.bits}b</div>
          </div>
        ))}
      </div>
      {hoverIdx !== null && (
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 mb-4">
          <div className="text-xs font-medium text-text-primary">{fields[hoverIdx].name} ({fields[hoverIdx].bits} bits)</div>
          <div className="text-[10px] font-mono text-sky-600 dark:text-sky-400 mt-1">值: {fields[hoverIdx].value}</div>
          <div className="text-xs text-text-secondary mt-1">{fields[hoverIdx].desc}</div>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          序列号: <span className="text-text-primary font-mono">{seqNum}</span>
          <input type="range" min={0} max={65535} value={seqNum} onChange={(e) => setSeqNum(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          时间戳: <span className="text-text-primary font-mono">{timestamp}</span>
          <input type="range" min={0} max={1000000} step={160} value={timestamp} onChange={(e) => setTimestamp(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1 mb-3">
        <div className="font-medium text-text-primary">RTP头部分析</div>
        <div>• 序列号 {seqNum}：接收端用于检测丢包和重排序</div>
        <div>• 时间戳 {timestamp}：标识第一个采样的时刻，用于同步和抖动缓冲</div>
        <div>• SSRC 0x55555555：标识此RTP流的同步源</div>
        <div>• 载荷类型 96：动态类型（通常用于H.264/H.265视频）</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">常见 Payload Type</h4>
          <div className="text-text-muted text-xs space-y-0.5">
            <div>0 = PCMU (G.711 μ-law)</div>
            <div>8 = PCMA (G.711 A-law)</div>
            <div>9 = G.722</div>
            <div>96+ = 动态 (H.264/VP8/Opus)</div>
          </div>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">RTP vs RTCP</h4>
          <div className="text-text-muted text-xs space-y-0.5">
            <div>• RTP: 媒体数据传输 (偶数端口)</div>
            <div>• RTCP: 控制/统计 (奇数端口 = RTP+1)</div>
            <div>• RTCP 报告丢包率、抖动、RTT</div>
            <div>• SR/RR 报告用于 QoS 监控</div>
          </div>
        </div>
      </div>
    </div>
  );
}
export default RTPPacketParser;
