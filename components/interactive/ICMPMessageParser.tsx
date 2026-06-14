"use client";
import { useState } from "react";

const ICMP_TYPES: Record<number, { name: string; codes: Record<number, string>; desc: string }> = {
  0: { name: "Echo Reply", codes: { 0: "回显应答" }, desc: "ping应答,响应Echo Request" },
  3: {
    name: "Destination Unreachable",
    codes: { 0: "网络不可达", 1: "主机不可达", 2: "协议不可达", 3: "端口不可达", 4: "需要分片但DF置位", 13: "通信被过滤" },
    desc: "目标不可达,路由器或主机返回",
  },
  8: { name: "Echo Request", codes: { 0: "回显请求" }, desc: "ping请求,测试主机可达性" },
  11: {
    name: "Time Exceeded",
    codes: { 0: "TTL超时", 1: "分片重组超时" },
    desc: "TTL=0时路由器丢包并返回,traceroute利用此报文",
  },
  12: { name: "Parameter Problem", codes: { 0: "IP首部错误" }, desc: "IP首部参数错误" },
  4: { name: "Source Quench", codes: { 0: "源站抑制" }, desc: "拥塞控制(已废弃)" },
  5: { name: "Redirect", codes: { 0: "网络重定向", 1: "主机重定向" }, desc: "路由重定向" },
};

export function ICMPMessageParser() {
  const [type, setType] = useState(8);
  const [code, setCode] = useState(0);
  const [hexData, setHexData] = useState("08004d5a00010001");

  const icmp = ICMP_TYPES[type];
  const codes = icmp?.codes || {};
  const codeKeys = Object.keys(codes).map(Number);

  const fields = [
    { name: "Type", value: type.toString(), hex: type.toString(16).padStart(2, "0").toUpperCase(), bits: 8 },
    { name: "Code", value: code.toString(), hex: code.toString(16).padStart(2, "0").toUpperCase(), bits: 8 },
    { name: "Checksum", value: "自动计算", hex: hexData.slice(4, 8) || "0000", bits: 16 },
    { name: "数据", value: "取决于类型", hex: hexData.slice(8) || "", bits: 32 },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">ICMP报文类型解析</h3>
      <div className="grid grid-cols-4 gap-2 mb-4">
        {Object.keys(ICMP_TYPES).map(Number).map((t) => (
          <button key={t} onClick={() => { setType(t); setCode(0); }}
            className={`px-2 py-1.5 rounded text-xs ${type === t ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
            Type {t}: {ICMP_TYPES[t].name.split(" ")[0]}
          </button>
        ))}
      </div>
      {codeKeys.length > 1 && (
        <div className="flex gap-2 mb-4 flex-wrap">
          {codeKeys.map((c) => (
            <button key={c} onClick={() => setCode(c)}
              className={`px-2 py-1 rounded text-xs ${code === c ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
              Code {c}: {codes[c]}
            </button>
          ))}
        </div>
      )}
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        <div className="font-semibold text-text-primary mb-2">{icmp.name} (Type={type}, Code={code})</div>
        <p className="text-sm text-text-secondary mb-3">{icmp.desc}</p>
        <p className="text-xs text-text-secondary">含义: {codes[code] || "无"}</p>
      </div>
      <div className="bg-bg-subtle rounded-lg p-3 mb-4 font-mono text-xs">
        <div className="text-text-secondary mb-2">报文结构 (8字节首部):</div>
        <div className="flex gap-0.5">
          {fields.map((f, i) => (
            <div key={i} className="flex-1 text-center">
              <div className="bg-blue-100 dark:bg-blue-900/30 p-2 rounded mb-1">
                <div className="text-text-primary font-bold">{f.name}</div>
                <div className="text-text-secondary">{f.bits}bit</div>
              </div>
              <div className="text-text-secondary">{f.hex || "-"}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="text-xs text-text-secondary">
        ICMP用于网络诊断和差错报告。常见应用: ping(Echo Request/Reply), traceroute(Time Exceeded)。
      </div>
    </div>
  );
}

export default ICMPMessageParser;
