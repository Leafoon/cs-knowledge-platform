"use client";
import { useState } from "react";

export function DNSMessageParser() {
  const [hexInput, setHexInput] = useState("0001818000010003000000000377777706676f6f676c6503636f6d0000010001c00c000100010000012c0004d83af004c00c000100010000012c0004d83af00ec00c000100010000012c0004d83af008");

  const parseHex = (hex: string) => {
    const clean = hex.replace(/\s/g, "").toLowerCase();
    if (clean.length < 24) return null;
    const getU16 = (off: number) => parseInt(clean.slice(off * 2, off * 2 + 4), 16);
    const id = getU16(0);
    const flags = getU16(2);
    const qr = (flags >> 15) & 1;
    const opcode = (flags >> 11) & 0xf;
    const aa = (flags >> 10) & 1;
    const tc = (flags >> 9) & 1;
    const rd = (flags >> 8) & 1;
    const ra = (flags >> 7) & 1;
    const rcode = flags & 0xf;
    const qdcount = getU16(4);
    const ancount = getU16(6);
    const nscount = getU16(8);
    const arcount = getU16(10);

    const rcodeNames: Record<number, string> = { 0: "NoError", 1: "FormErr", 2: "ServFail", 3: "NXDomain", 4: "NotImp", 5: "Refused" };

    return {
      id: `0x${id.toString(16).padStart(4, "0")}`,
      qr: qr ? "响应(Response)" : "查询(Query)",
      opcode: opcode.toString(),
      aa: aa ? "是" : "否",
      tc: tc ? "是(已截断)" : "否",
      rd: rd ? "是(期望递归)" : "否",
      ra: ra ? "是(支持递归)" : "否",
      rcode: rcodeNames[rcode] || rcode.toString(),
      qdcount, ancount, nscount, arcount,
    };
  };

  const parsed = parseHex(hexInput);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNS 消息解析器</h3>
      <label className="text-sm text-text-secondary block mb-2">输入DNS报文十六进制数据:</label>
      <textarea value={hexInput} onChange={(e) => setHexInput(e.target.value)}
        className="w-full h-20 px-3 py-2 rounded border border-border-subtle bg-bg-elevated text-text-primary text-xs font-mono resize-none mb-4"
        placeholder="粘贴十六进制DNS报文..." />
      {parsed ? (
        <div className="space-y-2">
          <div className="text-sm font-semibold text-text-primary mb-2">Header 部分</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {([
              ["Transaction ID", parsed.id],
              ["QR", parsed.qr],
              ["Opcode", parsed.opcode],
              ["AA (权威应答)", parsed.aa],
              ["TC (截断)", parsed.tc],
              ["RD (递归)", parsed.rd],
              ["RA (递归可用)", parsed.ra],
              ["RCODE", parsed.rcode],
              ["Question数", parsed.qdcount.toString()],
              ["Answer数", parsed.ancount.toString()],
              ["Authority数", parsed.nscount.toString()],
              ["Additional数", parsed.arcount.toString()],
            ] as [string, string][]).map(([k, v]) => (
              <div key={k} className="flex justify-between bg-gray-50 dark:bg-gray-900 rounded px-3 py-1.5">
                <span className="text-text-secondary">{k}</span>
                <span className="font-mono text-text-primary">{v}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="text-sm text-text-secondary text-center py-4">请输入至少12字节(24个十六进制字符)的DNS报文</div>
      )}
    </div>
  );
}
export default DNSMessageParser;
