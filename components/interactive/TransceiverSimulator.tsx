"use client";
import { useState } from "react";

const encodingTypes = [
  { id: "nrz", name: "NRZ (Non-Return-to-Zero)", desc: "高电平=1，低电平=0，简单但无同步能力" },
  { id: "manchester", name: "Manchester", desc: "位中间跳变：↑=1, ↓=0，自同步" },
  { id: "4b5b", name: "4B/5B", desc: "4位数据映射为5位码字，保证足够跳变" },
];

const bitPatterns: Record<string, number[]> = {
  nrz: [1, 0, 1, 1, 0, 0, 1, 0],
  manchester: [1, 0, 1, 1, 0, 0, 1, 0],
  "4b5b": [1, 0, 1, 1, 0, 0, 1, 0],
};

export function TransceiverSimulator() {
  const [encoding, setEncoding] = useState("nrz");
  const [formFactor, setFormFactor] = useState<"SFP" | "QSFP">("SFP");
  const [bits] = useState([1, 0, 1, 1, 0, 0, 1, 0]);

  const drawWaveform = (type: string) => {
    const W = 400; const H = 60; const segW = W / bits.length;
    let d = "";
    bits.forEach((bit, i) => {
      if (type === "nrz") {
        const y = bit ? 15 : 45;
        d += `${i === 0 ? "M" : "L"} ${i * segW} ${y} L ${(i + 1) * segW} ${y}`;
      } else if (type === "manchester") {
        const y1 = bit ? 15 : 45;
        const y2 = bit ? 45 : 15;
        d += `${i === 0 ? "M" : "L"} ${i * segW} ${y1} L ${i * segW + segW / 2} ${y1} L ${i * segW + segW / 2} ${y2} L ${(i + 1) * segW} ${y2}`;
      } else {
        const maps: Record<number, string> = { 0: "11110", 1: "01001" };
        const code = maps[bit] || "00000";
        code.split("").forEach((c, j) => {
          const y = c === "1" ? 15 : 45;
          const x = i * segW + j * (segW / 5);
          d += `${i === 0 && j === 0 ? "M" : "L"} ${x} ${y}`;
        });
      }
    });
    return d;
  };

  const specs: Record<string, { speed: string; distance: string; fiber: string }> = {
    SFP: { speed: "1 Gbps", distance: "100m - 80km", fiber: "单模/多模" },
    QSFP: { speed: "40-400 Gbps", distance: "100m - 10km", fiber: "多模/单模" },
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">收发器模拟器 (Transceiver)</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setFormFactor("SFP")}
          className={`px-4 py-2 rounded text-sm ${formFactor === "SFP" ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          SFP
        </button>
        <button onClick={() => setFormFactor("QSFP")}
          className={`px-4 py-2 rounded text-sm ${formFactor === "QSFP" ? "bg-green-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          QSFP
        </button>
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4">
        {Object.entries(specs[formFactor]).map(([k, v]) => (
          <div key={k} className="p-2 rounded bg-bg-primary border border-border-subtle text-center">
            <span className="text-text-muted text-xs block">{k === "speed" ? "速率" : k === "distance" ? "距离" : "光纤"}</span>
            <span className="text-text-primary text-sm font-medium">{v}</span>
          </div>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <div className="flex items-center gap-4 mb-3">
          <div className="p-3 rounded bg-blue-500/10 border border-blue-400/30 text-center">
            <span className="text-blue-400 text-xs block">电信号 (输入)</span>
            <span className="text-text-primary font-mono text-sm">{bits.join(" ")}</span>
          </div>
          <svg width="40" height="20"><path d="M0 10 L30 10 M25 5 L30 10 L25 15" stroke="#60a5fa" strokeWidth="2" fill="none" /></svg>
          <div className="p-3 rounded bg-yellow-500/10 border border-yellow-400/30 text-center flex-1">
            <span className="text-yellow-400 text-xs block">光信号 (光纤传输)</span>
            <div className="flex justify-center gap-2 mt-1">
              {bits.map((b, i) => (
                <div key={i} className={`w-3 h-3 rounded-full ${b ? "bg-yellow-400 shadow-lg shadow-yellow-400/50" : "bg-gray-400"}`} />
              ))}
            </div>
          </div>
          <svg width="40" height="20"><path d="M0 10 L30 10 M25 5 L30 10 L25 15" stroke="#4ade80" strokeWidth="2" fill="none" /></svg>
          <div className="p-3 rounded bg-green-500/10 border border-green-400/30 text-center">
            <span className="text-green-400 text-xs block">电信号 (输出)</span>
            <span className="text-text-primary font-mono text-sm">{bits.join(" ")}</span>
          </div>
        </div>
      </div>
      <h4 className="text-text-secondary text-sm mb-2">信号编码波形</h4>
      <div className="flex gap-2 mb-3">
        {encodingTypes.map((e) => (
          <button key={e.id} onClick={() => setEncoding(e.id)}
            className={`px-3 py-1 rounded text-xs ${encoding === e.id ? "bg-purple-500 text-white" : "border border-border-subtle text-text-muted"}`}>
            {e.name}
          </button>
        ))}
      </div>
      <svg viewBox="0 0 400 70" className="w-full max-w-md bg-bg-primary rounded border border-border-subtle p-2">
        <path d={drawWaveform(encoding)} fill="none" stroke="#a78bfa" strokeWidth="2" />
        {bits.map((_, i) => (
          <text key={i} x={i * 50 + 25} y={65} textAnchor="middle" className="fill-text-muted text-[10px]">{bits[i]}</text>
        ))}
      </svg>
      <p className="text-text-muted text-xs mt-2">{encodingTypes.find((e) => e.id === encoding)?.desc}</p>
    </div>
  );
}
export default TransceiverSimulator;
