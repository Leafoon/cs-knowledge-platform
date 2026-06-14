"use client";
import { useState } from "react";

interface Field {
  name: string;
  bits: number;
  desc: string;
  color: string;
}

const longHeader: Field[] = [
  { name: "Header Form (1)", bits: 1, desc: "1=长头，0=短头", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "Fixed Bit", bits: 1, desc: "固定为1", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "Long Packet Type", bits: 2, desc: "00=Initial, 01=0-RTT, 10=Handshake, 11=Retry", color: "bg-emerald-500/20 border-emerald-400/40" },
  { name: "Type-Specific Bits", bits: 4, desc: "类型相关字段", color: "bg-amber-500/20 border-amber-400/40" },
  { name: "Version", bits: 32, desc: "QUIC版本号（如0x00000001）", color: "bg-violet-500/20 border-violet-400/40" },
  { name: "DCID Len", bits: 8, desc: "目的Connection ID长度", color: "bg-rose-500/20 border-rose-400/40" },
  { name: "Destination CID", bits: 0, desc: "目的Connection ID（0-20字节）", color: "bg-rose-500/20 border-rose-400/40" },
  { name: "SCID Len", bits: 8, desc: "源Connection ID长度", color: "bg-pink-500/20 border-pink-400/40" },
  { name: "Source CID", bits: 0, desc: "源Connection ID（0-20字节）", color: "bg-pink-500/20 border-pink-400/40" },
];

const shortHeader: Field[] = [
  { name: "Header Form (0)", bits: 1, desc: "0=短头", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "Fixed Bit", bits: 1, desc: "固定为1", color: "bg-sky-500/20 border-sky-400/40" },
  { name: "Spin Bit", bits: 1, desc: "RTT测量旋转位", color: "bg-emerald-500/20 border-emerald-400/40" },
  { name: "Reserved", bits: 2, desc: "保留位", color: "bg-gray-500/20 border-gray-400/40" },
  { name: "Key Phase", bits: 1, desc: "密钥更新相位", color: "bg-amber-500/20 border-amber-400/40" },
  { name: "Packet Number Len", bits: 2, desc: "包号长度（1-4字节）", color: "bg-violet-500/20 border-violet-400/40" },
  { name: "Destination CID", bits: 0, desc: "目的Connection ID（连接建立后固定）", color: "bg-rose-500/20 border-rose-400/40" },
  { name: "Packet Number", bits: 0, desc: "包号（1-4字节）", color: "bg-pink-500/20 border-pink-400/40" },
];

export function QUICPacketFormatDemo() {
  const [headerType, setHeaderType] = useState<"long" | "short">("long");
  const [hoverField, setHoverField] = useState<number | null>(null);
  const fields = headerType === "long" ? longHeader : shortHeader;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">QUIC报文格式</h3>
      <div className="flex gap-2 mb-4">
        {(["long", "short"] as const).map((t) => (
          <button key={t} onClick={() => { setHeaderType(t); setHoverField(null); }}
            className={`px-4 py-1.5 rounded-lg border text-xs font-medium transition-all ${headerType === t ? "bg-sky-500/20 border-sky-400/60 text-sky-700 dark:text-sky-300" : "bg-bg-tertiary border-border-subtle text-text-secondary"}`}>
            {t === "long" ? "长头 Long Header" : "短头 Short Header"}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-0.5 mb-4">
        {fields.map((f, i) => (
          <div key={i} onMouseEnter={() => setHoverField(i)} onMouseLeave={() => setHoverField(null)}
            className={`px-2 py-2 border text-[10px] font-mono text-center cursor-pointer transition-all ${f.color} ${hoverField === i ? "ring-2 ring-sky-400 scale-105 z-10" : ""}`}
            style={{ minWidth: `${Math.max(40, f.bits * 3)}px`, flex: f.bits === 0 ? "1 1 auto" : `${Math.max(f.bits, 2)} 0 auto` }}>
            <div className="text-text-primary font-medium truncate">{f.name.split(" (")[0]}</div>
            {f.bits > 0 && <div className="text-text-tertiary">{f.bits}b</div>}
          </div>
        ))}
      </div>
      {hoverField !== null && (
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 mb-3">
          <div className="text-xs font-medium text-text-primary">{fields[hoverField].name}</div>
          <div className="text-xs text-text-secondary mt-1">{fields[hoverField].desc}</div>
          {fields[hoverField].bits > 0 && <div className="text-[10px] text-text-tertiary mt-1">长度: {fields[hoverField].bits} bit(s)</div>}
        </div>
      )}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-medium text-text-primary">{headerType === "long" ? "长头用途" : "短头用途"}</div>
        {headerType === "long" ? (
          <>
            <div>• Initial包: 连接建立（含TLS ClientHello）</div>
            <div>• 0-RTT包: 早期数据传输</div>
            <div>• Handshake包: TLS握手完成</div>
            <div>• Retry包: 地址验证</div>
          </>
        ) : (
          <>
            <div>• 1-RTT包: 连接建立后的常规数据传输</div>
            <div>• 更短的头部，更高的传输效率</div>
            <div>• Spin Bit用于被动RTT测量</div>
            <div>• Key Phase支持密钥轮换</div>
          </>
        )}
      </div>
    </div>
  );
}
export default QUICPacketFormatDemo;
