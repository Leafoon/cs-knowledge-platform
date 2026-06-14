"use client";
import { useState } from "react";

interface SONETField {
  name: string;
  bytes: number;
  desc: string;
  color: string;
}

const fields: SONETField[] = [
  { name: "A1 A2 (帧同步)", bytes: 2, desc: "帧定界字节，A1=0xF6, A2=0x28，用于接收端帧同步", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "J0 (追踪)", bytes: 1, desc: "Section Trace字节，用于验证连接的连续性", color: "bg-emerald-500/20 border-emerald-400/40" },
  { name: "B1 (比特交错奇偶校验)", bytes: 1, desc: "Section层误码监测，BIP-8校验", color: "bg-amber-500/20 border-amber-400/40" },
  { name: "E1 (公务联络)", bytes: 1, desc: "工程勤务线（Orderwire），用于维护人员语音通信", color: "bg-violet-500/20 border-violet-400/40" },
  { name: "F1 (用户通道)", bytes: 1, desc: "留给网络运营者的专用通道", color: "bg-rose-500/20 border-rose-400/40" },
  { name: "D1-D3 (数据通信)", bytes: 3, desc: "Section层DCC通道，用于网管通信（192kbps）", color: "bg-pink-500/20 border-pink-400/40" },
  { name: "H1-H2 (指针)", bytes: 2, desc: "Payload指针，指示SPE在帧中的起始位置", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "H3 (指针动作)", bytes: 1, desc: "用于频率调整的负偏移字节", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "B2 (BIP)", bytes: 1, desc: "Line层误码监测，BIP-N×24", color: "bg-amber-500/20 border-amber-400/40" },
  { name: "K1-K2 (APS)", bytes: 2, desc: "自动保护倒换信令，实现50ms故障恢复", color: "bg-emerald-500/20 border-emerald-400/40" },
  { name: "D4-D12 (DCC)", bytes: 9, desc: "Line层DCC通道，用于网管通信（576kbps）", color: "bg-pink-500/20 border-pink-400/40" },
  { name: "S1/Z1-Z2", bytes: 3, desc: "同步状态/增长字节", color: "bg-gray-500/20 border-gray-400/40" },
  { name: "E2 (公务)", bytes: 1, desc: "Line层公务联络通道", color: "bg-violet-500/20 border-violet-400/40" },
];

export function SONETFrameExplorer() {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [showInfo, setShowInfo] = useState(false);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">SONET帧探索器 (STS-1)</h3>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="text-xs text-text-secondary mb-2">STS-1 帧 = 810字节（9行 × 90列），传输速率 = 51.84 Mbps</div>
        <div className="flex flex-wrap gap-0.5">
          {fields.map((f, i) => (
            <div key={i} onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)}
              className={`px-2 py-1.5 border text-[9px] font-mono text-center cursor-pointer transition-all ${f.color} ${hoverIdx === i ? "ring-2 ring-sky-400 scale-110 z-10" : ""}`}
              style={{ minWidth: `${Math.max(40, f.bytes * 20)}px`, flex: `${Math.max(f.bytes, 1)} 0 auto` }}>
              <div className="text-text-primary font-medium truncate">{f.name}</div>
              <div className="text-text-tertiary">{f.bytes}B</div>
            </div>
          ))}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <div className="flex-1 h-8 rounded bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-[10px] text-emerald-600 dark:text-emerald-400">
            SPE (Payload) = 783字节（87列 × 9行）
          </div>
        </div>
      </div>
      {hoverIdx !== null && (
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 mb-4">
          <div className="text-xs font-medium text-text-primary">{fields[hoverIdx].name}</div>
          <div className="text-xs text-text-secondary mt-1">{fields[hoverIdx].desc}</div>
          <div className="text-[10px] text-text-tertiary mt-1">大小: {fields[hoverIdx].bytes} 字节</div>
        </div>
      )}
      <button onClick={() => setShowInfo(!showInfo)} className="w-full px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:text-text-primary transition-colors mb-3">
        {showInfo ? "隐藏" : "显示"}SONET层次结构
      </button>
      {showInfo && (
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1 mb-3">
          <div className="font-medium text-text-primary">SONET分层</div>
          <div>• Section层：光中继器之间，负责帧同步和段误码监测</div>
          <div>• Line层：复用器之间，负责复用和保护倒换</div>
          <div>• Path层：端到端，负责SPE的组装和拆分</div>
          <div className="text-text-tertiary">SDH对应：Section→再生段(RS)，Line→复用段(MS)，Path→通道</div>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">SONET 速率等级</h4>
          <div className="text-text-muted text-xs space-y-0.5">
            <div>STS-1 = 51.84 Mbps (基本信号)</div>
            <div>STS-3 = OC-3 = 155.52 Mbps</div>
            <div>STS-12 = OC-12 = 622.08 Mbps</div>
            <div>STS-48 = OC-48 = 2.488 Gbps</div>
          </div>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">保护倒换 (APS)</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• K1/K2 字节承载倒换请求</li>
            <li>• 1+1 保护: 双发选收 (&lt;50ms)</li>
            <li>• 1:N 保护: 共享备用通道</li>
            <li>• 比 IP 路由收敛快得多</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default SONETFrameExplorer;
