"use client";

import { useState, useMemo } from "react";

const units = [
  { name: "B", factor: 1 },
  { name: "KB", factor: 1024 },
  { name: "MB", factor: 1024 ** 2 },
  { name: "GB", factor: 1024 ** 3 },
  { name: "TB", factor: 1024 ** 4 },
];

export function StorageCapacityCalculator() {
  const [marBits, setMarBits] = useState(16);
  const [mdrBits, setMdrBits] = useState(8);
  const [customAddr, setCustomAddr] = useState("");
  const [customData, setCustomData] = useState("");

  const addrBits = customAddr ? parseInt(customAddr) || 0 : marBits;
  const dataBits = customData ? parseInt(customData) || 0 : mdrBits;

  const results = useMemo(() => {
    if (addrBits <= 0 || dataBits <= 0) return null;
    const numCells = 2 ** addrBits;
    const totalBits = numCells * dataBits;
    const totalBytes = totalBits / 8;
    return { numCells, totalBits, totalBytes };
  }, [addrBits, dataBits]);

  function formatBytes(bytes: number): string {
    for (let i = units.length - 1; i >= 0; i--) {
      if (bytes >= units[i].factor) {
        const val = bytes / units[i].factor;
        return val % 1 === 0 ? `${val} ${units[i].name}` : `${val.toFixed(2)} ${units[i].name}`;
      }
    }
    return `${bytes} B`;
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        存储容量计算器
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        输入 MAR（地址寄存器）和 MDR（数据寄存器）的位数，自动计算存储器容量
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        <div className="rounded-lg border border-border-subtle bg-bg-secondary p-4">
          <label className="block text-xs font-medium text-text-secondary mb-2">
            MAR 位数 (地址线宽度)
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={1}
              max={64}
              value={customAddr || marBits}
              onChange={(e) => {
                setCustomAddr("");
                setMarBits(Number(e.target.value));
              }}
              className="flex-1 accent-accent-primary"
            />
            <input
              type="number"
              value={customAddr || marBits}
              onChange={(e) => setCustomAddr(e.target.value)}
              min={1}
              max={64}
              className="w-20 px-2 py-1 rounded border border-border-subtle bg-bg-elevated text-text-primary font-mono text-sm text-center"
            />
            <span className="text-xs text-text-secondary">位</span>
          </div>
          <p className="text-xs text-text-secondary mt-2">
            可寻址单元数 = 2<sup>{customAddr || marBits}</sup> = {(2 ** (customAddr ? parseInt(customAddr) || 0 : marBits)).toLocaleString()}
          </p>
        </div>

        <div className="rounded-lg border border-border-subtle bg-bg-secondary p-4">
          <label className="block text-xs font-medium text-text-secondary mb-2">
            MDR 位数 (数据线宽度)
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={1}
              max={128}
              value={customData || mdrBits}
              onChange={(e) => {
                setCustomData("");
                setMdrBits(Number(e.target.value));
              }}
              className="flex-1 accent-accent-primary"
            />
            <input
              type="number"
              value={customData || mdrBits}
              onChange={(e) => setCustomData(e.target.value)}
              min={1}
              max={128}
              className="w-20 px-2 py-1 rounded border border-border-subtle bg-bg-elevated text-text-primary font-mono text-sm text-center"
            />
            <span className="text-xs text-text-secondary">位</span>
          </div>
          <p className="text-xs text-text-secondary mt-2">
            每个存储单元 {customData || mdrBits} 位 = {((customData ? parseInt(customData) || 0 : mdrBits) / 8).toFixed(1)} 字节
          </p>
        </div>
      </div>

      {/* results */}
      {results && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "存储单元数", value: results.numCells.toLocaleString(), unit: "个" },
            { label: "总容量(位)", value: results.totalBits.toLocaleString(), unit: "bit" },
            { label: "总容量(字节)", value: results.totalBytes.toLocaleString(), unit: "B" },
            { label: "可读表示", value: formatBytes(results.totalBytes), unit: "" },
          ].map((r) => (
            <div key={r.label} className="rounded-lg border border-border-subtle bg-bg-secondary p-3 text-center">
              <p className="text-xs text-text-secondary mb-1">{r.label}</p>
              <p className="font-mono font-bold text-text-primary text-sm">
                {r.value} <span className="text-xs text-text-secondary font-normal">{r.unit}</span>
              </p>
            </div>
          ))}
        </div>
      )}

      {/* formula */}
      <div className="mt-4 rounded-lg bg-bg-secondary border border-border-subtle p-3">
        <p className="text-xs text-text-secondary">
          <span className="font-semibold text-text-primary">计算公式：</span>
          存储器容量 = 2<sup>MAR位数</sup> × MDR位数 (bit)
          {results && (
            <span className="ml-2 font-mono">
              = 2<sup>{customAddr || marBits}</sup> × {customData || mdrBits} = {results.totalBits.toLocaleString()} bit = {formatBytes(results.totalBytes)}
            </span>
          )}
        </p>
      </div>
    </div>
  );
}
