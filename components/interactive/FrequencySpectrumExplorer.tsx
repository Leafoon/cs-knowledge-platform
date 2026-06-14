"use client";
import { useState } from "react";

const bands = [
  { name: "ELF", range: "3–30 Hz", use: "潜艇通信", color: "bg-red-500" },
  { name: "VLF", range: "30–300 kHz", use: "导航信标", color: "bg-orange-500" },
  { name: "LF", range: "30–300 kHz", use: "AM广播", color: "bg-yellow-500" },
  { name: "MF", range: "300 kHz–3 MHz", use: "AM广播/海事", color: "bg-green-500" },
  { name: "HF", range: "3–30 MHz", use: "短波广播/Ham Radio", color: "bg-teal-500" },
  { name: "VHF", range: "30–300 MHz", use: "FM广播/电视/WiFi", color: "bg-blue-500" },
  { name: "UHF", range: "300 MHz–3 GHz", use: "移动通信/GPS/WiFi", color: "bg-indigo-500" },
  { name: "SHF", range: "3–30 GHz", use: "卫星通信/5G毫米波", color: "bg-purple-500" },
  { name: "EHF", range: "30–300 GHz", use: "射电天文/6G研究", color: "bg-pink-500" },
];

export function FrequencySpectrumExplorer() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">频谱探索器</h3>
      <p className="text-sm text-text-secondary mb-4">电磁频谱分配与各频段的网络应用</p>
      <div className="flex flex-wrap gap-2 mb-4">
        {bands.map((b, i) => (
          <button
            key={b.name}
            onClick={() => setSelected(selected === i ? null : i)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-all ${
              selected === i
                ? `${b.color} text-white scale-105`
                : "bg-bg-subtle text-text-secondary hover:bg-bg-muted"
            }`}
          >
            {b.name}
          </button>
        ))}
      </div>
      {selected !== null && (
        <div className="p-4 rounded-lg bg-bg-muted border border-border-subtle">
          <div className="flex items-center gap-3 mb-2">
            <span className={`w-3 h-3 rounded-full ${bands[selected].color}`} />
            <span className="font-semibold text-text-primary">{bands[selected].name} 频段</span>
          </div>
          <p className="text-sm text-text-secondary">频率范围: {bands[selected].range}</p>
          <p className="text-sm text-text-secondary">典型应用: {bands[selected].use}</p>
        </div>
      )}
      <div className="mt-4 flex h-6 rounded overflow-hidden">
        {bands.map((b, i) => (
          <div
            key={b.name}
            className={`${b.color} cursor-pointer transition-all ${
              selected === i ? "opacity-100" : "opacity-60 hover:opacity-80"
            }`}
            style={{ flex: i + 1 }}
            onClick={() => setSelected(selected === i ? null : i)}
            title={b.name}
          />
        ))}
      </div>
      <div className="flex justify-between text-xs text-text-muted mt-1">
        <span>低频 (3 Hz)</span>
        <span>高频 (300 GHz)</span>
      </div>
    </div>
  );
}
export default FrequencySpectrumExplorer;
