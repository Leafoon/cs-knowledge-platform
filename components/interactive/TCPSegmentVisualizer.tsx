"use client";
import { useState } from "react";

interface Field { name: string; bits: number; value: string; color: string; desc: string; }

const defaultFields: Field[] = [
  { name: "源端口 Source Port", bits: 16, value: "12345", color: "bg-blue-200 dark:bg-blue-800", desc: "发送方端口号，范围 0-65535" },
  { name: "目的端口 Dest Port", bits: 16, value: "80", color: "bg-blue-200 dark:bg-blue-800", desc: "接收方端口号，标识应用服务" },
  { name: "序列号 Sequence Number", bits: 32, value: "1000000", color: "bg-green-200 dark:bg-green-800", desc: "本报文段数据的首字节序号" },
  { name: "确认号 Ack Number", bits: 32, value: "2000000", color: "bg-green-200 dark:bg-green-800", desc: "期望收到的下一个字节序号 (ACK=1 时有效)" },
  { name: "数据偏移 Data Offset", bits: 4, value: "5", color: "bg-yellow-200 dark:bg-yellow-800", desc: "头部长度，单位为 32-bit 字 (5=20 bytes)" },
  { name: "保留 Reserved", bits: 3, value: "0", color: "bg-gray-200 dark:bg-gray-700", desc: "保留字段，必须为 0" },
  { name: "标志位 Flags", bits: 9, value: "0x012", color: "bg-red-200 dark:bg-red-800", desc: "CWR ECE URG ACK PSH RST SYN FIN" },
  { name: "窗口大小 Window", bits: 16, value: "65535", color: "bg-purple-200 dark:bg-purple-800", desc: "接收窗口大小，用于流量控制" },
  { name: "校验和 Checksum", bits: 16, value: "0xABCD", color: "bg-orange-200 dark:bg-orange-800", desc: "覆盖头部+数据+伪头部的校验和" },
  { name: "紧急指针 Urgent Pointer", bits: 16, value: "0", color: "bg-gray-200 dark:bg-gray-700", desc: "URG=1 时指向紧急数据末尾" },
];

const flagBits = [
  { name: "CWR", bit: 8, desc: "Congestion Window Reduced" },
  { name: "ECE", bit: 7, desc: "ECN Echo" },
  { name: "URG", bit: 6, desc: "紧急指针有效" },
  { name: "ACK", bit: 5, desc: "确认号有效" },
  { name: "PSH", bit: 4, desc: "推送，尽快交付数据" },
  { name: "RST", bit: 3, desc: "重置连接" },
  { name: "SYN", bit: 2, desc: "同步序列号" },
  { name: "FIN", bit: 1, desc: "发送方数据发送完毕" },
];

export function TCPSegmentVisualizer() {
  const [fields, setFields] = useState<Field[]>(defaultFields);
  const [hovered, setHovered] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [flagValue, setFlagValue] = useState(0x012);

  const updateField = (idx: number, value: string) => {
    setFields((f) => f.map((field, i) => i === idx ? { ...field, value } : field));
  };

  const toggleFlag = (bit: number) => {
    setFlagValue((v) => v ^ (1 << bit));
  };

  const totalBits = fields.reduce((s, f) => s + f.bits, 0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCP 报文段结构</h3>
      <div className="text-xs text-text-secondary mb-3">总长度: {totalBits} bits ({totalBits / 8} bytes) | 数据偏移: 5 (20 bytes，无选项)</div>
      <div className="flex flex-wrap gap-px mb-4">
        {fields.map((f, i) => (
          <div key={i} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)} onClick={() => setSelected(selected === i ? null : i)}
            className={`${f.color} rounded px-2 py-2 cursor-pointer transition-all border-2 ${hovered === i || selected === i ? "border-blue-500 ring-1 ring-blue-300 scale-105 z-10" : "border-transparent"}`}
            style={{ flex: `${f.bits} ${f.bits}`, minWidth: "50px" }}>
            <div className="text-xs font-medium text-text-primary truncate">{f.name.split(" ")[0]}</div>
            <div className="text-xs font-mono text-text-secondary">{f.bits} bits</div>
          </div>
        ))}
      </div>
      {selected !== null && (
        <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle mb-4">
          <div className="text-sm font-medium text-text-primary mb-1">{fields[selected].name}</div>
          <div className="text-xs text-text-secondary mb-2">{fields[selected].desc}</div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-xs text-text-secondary mb-1">位宽: {fields[selected].bits} bits ({(fields[selected].bits / 8).toFixed(1)} bytes)</div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-text-secondary">值:</span>
                <input value={fields[selected].value} onChange={(e) => updateField(selected, e.target.value)} className="px-2 py-1 rounded border border-border-subtle bg-white dark:bg-gray-900 text-sm font-mono text-text-primary w-32" />
              </div>
            </div>
            <div className="text-xs text-text-secondary">偏移: {fields.slice(0, selected).reduce((s, f) => s + f.bits, 0)} bits from start</div>
          </div>
        </div>
      )}
      <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle mb-3">
        <div className="text-xs text-text-secondary mb-2">标志位编辑器 (当前值: 0x{flagValue.toString(16).padStart(3, "0")}):</div>
        <div className="flex flex-wrap gap-1">
          {flagBits.map((f) => (
            <button key={f.name} onClick={() => toggleFlag(f.bit)} className={`px-2 py-1 rounded text-xs font-mono border transition-colors ${flagValue & (1 << f.bit) ? "bg-red-500 text-white border-red-600" : "bg-gray-200 dark:bg-gray-700 text-text-secondary border-transparent"}`}>
              {f.name}
            </button>
          ))}
        </div>
        <div className="mt-2 text-xs text-text-secondary">
          {flagBits.filter((f) => flagValue & (1 << f.bit)).map((f) => <span key={f.name} className="mr-2">{f.name}: {f.desc}</span>)}
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
        {fields.map((f, i) => (
          <div key={i} className="flex items-center gap-1">
            <div className={`w-3 h-3 rounded ${f.color}`} />
            <span className="text-text-secondary truncate">{f.name.split(" ")[0]}: <span className="font-mono text-text-primary">{f.value}</span></span>
          </div>
        ))}
      </div>
    </div>
  );
}
export default TCPSegmentVisualizer;
