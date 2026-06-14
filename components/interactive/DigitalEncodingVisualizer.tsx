"use client";
import { useState } from "react";

const ENCODINGS = ["NRZ-L", "NRZ-I", "Manchester", "Differential Manchester", "4B5B"] as const;
type Encoding = (typeof ENCODINGS)[number];

function generateWaveform(data: string, encoding: Encoding): { high: boolean; label: string }[] {
  const bits = data.split("").map((b) => parseInt(b));
  const result: { high: boolean; label: string }[] = [];
  let prevHigh = true;

  for (let i = 0; i < bits.length; i++) {
    const bit = bits[i];
    if (encoding === "NRZ-L") {
      result.push({ high: bit === 1, label: `${bit}` });
    } else if (encoding === "NRZ-I") {
      if (bit === 1) prevHigh = !prevHigh;
      result.push({ high: prevHigh, label: `${bit}` });
    } else if (encoding === "Manchester") {
      result.push({ high: bit === 1, label: `${bit}↑` });
      result.push({ high: bit === 0, label: `${bit}↓` });
    } else if (encoding === "Differential Manchester") {
      if (bit === 0) prevHigh = !prevHigh;
      result.push({ high: prevHigh, label: `${bit}↑` });
      prevHigh = !prevHigh;
      result.push({ high: prevHigh, label: `${bit}↓` });
    }
  }
  return result;
}

const MAP_4B5B: Record<string, string> = {
  "0000": "11110", "0001": "01001", "0010": "10100", "0011": "10101",
  "0100": "01010", "0101": "01011", "0110": "01110", "0111": "01111",
  "1000": "10010", "1001": "10011", "1010": "10110", "1011": "10111",
  "1100": "11010", "1101": "11011", "1110": "11100", "1111": "11101",
};

export function DigitalEncodingVisualizer() {
  const [data, setData] = useState("10110");
  const [encoding, setEncoding] = useState<Encoding>("NRZ-L");

  const validData = data.replace(/[^01]/g, "");
  const waveform = encoding !== "4B5B" ? generateWaveform(validData, encoding) : [];
  const encoded4B5B = encoding === "4B5B"
    ? validData.match(/.{1,4}/g)?.map((g) => MAP_4B5B[g.padEnd(4, "0")] || "?????").join("") || ""
    : "";

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">数字编码波形可视化</h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {ENCODINGS.map((enc) => (
          <button
            key={enc}
            onClick={() => setEncoding(enc)}
            className={`px-3 py-1.5 rounded text-sm transition-all ${
              encoding === enc
                ? "bg-blue-500 text-white"
                : "bg-bg-subtle text-text-secondary hover:bg-bg-muted"
            }`}
          >
            {enc}
          </button>
        ))}
      </div>
      <div className="mb-4">
        <label className="text-sm text-text-secondary mb-1 block">输入比特序列:</label>
        <input
          type="text"
          value={data}
          onChange={(e) => setData(e.target.value.replace(/[^01]/g, "").slice(0, 16))}
          className="w-full px-3 py-2 rounded border border-border-subtle bg-bg-subtle text-text-primary font-mono"
          placeholder="输入0和1,如 10110"
        />
      </div>
      {encoding !== "4B5B" && waveform.length > 0 && (
        <div className="mb-4 p-4 bg-bg-muted rounded-lg overflow-x-auto">
          <svg width={waveform.length * 40 + 20} height={80} className="block">
            {waveform.map((w, i) => {
              const y = w.high ? 15 : 55;
              const nextY = i < waveform.length - 1 ? (waveform[i + 1].high ? 15 : 55) : y;
              return (
                <g key={i}>
                  <line x1={i * 40 + 10} y1={y} x2={(i + 1) * 40 + 10} y2={y} stroke="#3b82f6" strokeWidth={2} />
                  {i < waveform.length - 1 && y !== nextY && (
                    <line x1={(i + 1) * 40 + 10} y1={y} x2={(i + 1) * 40 + 10} y2={nextY} stroke="#3b82f6" strokeWidth={2} />
                  )}
                  <text x={i * 40 + 25} y={75} textAnchor="middle" fill="#6b7280" fontSize={10}>{w.label}</text>
                </g>
              );
            })}
            <line x1={10} y1={35} x2={waveform.length * 40 + 10} y2={35} stroke="#d1d5db" strokeWidth={0.5} strokeDasharray="4" />
          </svg>
        </div>
      )}
      {encoding === "4B5B" && (
        <div className="mb-4 p-4 bg-bg-muted rounded-lg">
          <p className="text-sm text-text-secondary mb-2">4B5B编码映射:</p>
          <div className="font-mono text-sm text-text-primary">
            {validData.match(/.{1,4}/g)?.map((group, i) => (
              <span key={i} className="inline-block mr-3 mb-1">
                <span className="text-blue-500">{group.padEnd(4, "0")}</span> → <span className="text-green-500">{MAP_4B5B[group.padEnd(4, "0")] || "????"}</span>
              </span>
            ))}
          </div>
          <p className="text-xs text-text-secondary mt-2">编码后: {encoded4B5B}</p>
        </div>
      )}
      <div className="text-xs text-text-secondary">
        {encoding === "NRZ-L" && "NRZ-L: 1=高电平,0=低电平。简单但连续相同bit无同步。"}
        {encoding === "NRZ-I" && "NRZ-I: 遇1翻转,遇0保持。差分编码,抗极性反转。"}
        {encoding === "Manchester" && "Manchester: 每个bit周期中间有跳变。1=先低后高,0=先高后低。以太网使用。"}
        {encoding === "Differential Manchester" && "差分Manchester: 每bit中间必跳变。0=bit起始跳变,1=bit起始不变。Token Ring使用。"}
        {encoding === "4B5B" && "4B5B: 4位数据映射为5位码,确保足够跳变密度,配合NRZ-I使用(Fast Ethernet)。"}
      </div>
    </div>
  );
}

export default DigitalEncodingVisualizer;
