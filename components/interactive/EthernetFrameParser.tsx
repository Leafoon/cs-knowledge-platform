"use client";
import { useState } from "react";

const SAMPLE_FRAME = "FFFFFFFFFFFF001A2B3C4D5E08004500003C1C4600004006B1E6C0A80101C0A80102";

const FIELDS = [
  { name: "目的MAC", bytes: 6, desc: "FF:FF:FF:FF:FF:FF (广播)", color: "bg-red-500" },
  { name: "源MAC", bytes: 6, desc: "00:1A:2B:3C:4D:5E", color: "bg-orange-500" },
  { name: "类型", bytes: 2, desc: "0x0800 = IPv4", color: "bg-yellow-500" },
  { name: "IP首部", bytes: 20, desc: "IPv4首部 (20字节)", color: "bg-green-500" },
  { name: "TCP/数据", bytes: 0, desc: "上层数据", color: "bg-blue-500" },
];

export function EthernetFrameParser() {
  const [frame, setFrame] = useState(SAMPLE_FRAME);
  const [hoverField, setHoverField] = useState(-1);

  const validFrame = frame.replace(/[^0-9a-fA-F]/g, "");
  let offset = 0;
  const parsed = FIELDS.map((f, i) => {
    const size = i === FIELDS.length - 1 ? Math.max(0, (validFrame.length / 2) - 14) : f.bytes;
    const hex = validFrame.slice(offset * 2, (offset + size) * 2);
    offset += size;
    return { ...f, hex, offset: offset - size, size };
  });

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">以太网帧十六进制解析</h3>
      <div className="mb-4">
        <label className="text-sm text-text-secondary mb-1 block">帧数据 (十六进制):</label>
        <input
          type="text"
          value={frame}
          onChange={(e) => setFrame(e.target.value.replace(/[^0-9a-fA-F]/g, "").slice(0, 120))}
          className="w-full px-3 py-2 rounded border border-border-subtle bg-bg-subtle text-text-primary font-mono text-xs"
        />
      </div>
      <div className="flex flex-wrap gap-0.5 mb-4 font-mono text-xs">
        {validFrame.match(/.{2}/g)?.map((byte, i) => {
          const fieldIdx = parsed.findIndex((f) => i >= f.offset && i < f.offset + f.size);
          return (
            <span
              key={i}
              onMouseEnter={() => setHoverField(fieldIdx)}
              onMouseLeave={() => setHoverField(-1)}
              className={`px-1 py-0.5 rounded cursor-pointer ${
                hoverField === fieldIdx ? `${parsed[fieldIdx]?.color || "bg-gray-500"} text-white` : "bg-bg-muted text-text-primary"
              }`}
            >
              {byte}
            </span>
          );
        })}
      </div>
      <div className="space-y-2 mb-4">
        {parsed.map((f, i) => (
          <div
            key={i}
            onMouseEnter={() => setHoverField(i)}
            onMouseLeave={() => setHoverField(-1)}
            className={`flex items-center gap-3 p-2 rounded transition-all cursor-pointer ${
              hoverField === i ? `${f.color} text-white` : "bg-bg-muted"
            }`}
          >
            <span className="w-20 text-xs font-bold">{f.name}</span>
            <span className="w-8 text-xs text-right">{f.size > 0 ? `${f.size}B` : "可变"}</span>
            <span className="flex-1 text-xs font-mono truncate">{f.hex || "-"}</span>
            <span className="text-xs">{f.desc}</span>
          </div>
        ))}
      </div>
      <div className="text-xs text-text-secondary">
        以太网帧结构: 目的MAC(6B) + 源MAC(6B) + 类型(2B) + 数据(46-1500B) + FCS(4B)。最小64字节,最大1518字节。
      </div>
    </div>
  );
}

export default EthernetFrameParser;
