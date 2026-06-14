"use client";
import { useState } from "react";

const contentTypes = [
  { type: "text/plain", desc: "纯文本", example: "Hello, World!" },
  { type: "text/html", desc: "HTML文档", example: "<h1>Hello</h1>" },
  { type: "image/jpeg", desc: "JPEG图片", example: "[二进制图片数据]" },
  { type: "multipart/mixed", desc: "混合多部分", example: "--boundary\nContent-Type: text/plain\n\nHello" },
  { type: "application/pdf", desc: "PDF附件", example: "%PDF-1.4 二进制内容" },
];

const encodings = [
  { name: "7bit", desc: "ASCII文本，每行不超过1000字符" },
  { name: "8bit", desc: "支持扩展字符集，非ASCII" },
  { name: "base64", desc: "将二进制数据编码为ASCII字符" },
  { name: "quoted-printable", desc: "可打印字符编码，保留可读性" },
];

export function MIMEDemo() {
  const [input, setInput] = useState("Hello MIME!");
  const [ctIdx, setCtIdx] = useState(0);
  const [encIdx, setEncIdx] = useState(2);

  const toBase64 = (s: string) => {
    try { return btoa(unescape(encodeURIComponent(s))); } catch { return "[编码错误]"; }
  };
  const fromBase64 = (s: string) => {
    try { return decodeURIComponent(escape(atob(s))); } catch { return "[解码错误]"; }
  };

  const encoded = toBase64(input);
  const decoded = fromBase64(encoded);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">✉️ MIME 演示</h3>
      <p className="text-sm text-text-secondary mb-4">展示邮件的 Content-Type 和 Base64 编码</p>

      <div className="mb-4">
        <label className="text-sm font-medium text-text-secondary">Content-Type</label>
        <div className="flex flex-wrap gap-2 mt-2">
          {contentTypes.map((ct, i) => (
            <button key={ct.type} onClick={() => setCtIdx(i)}
              className={`px-2.5 py-1.5 rounded text-xs font-mono ${ctIdx === i ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
              {ct.type}
            </button>
          ))}
        </div>
        <p className="text-xs text-text-secondary mt-1">{contentTypes[ctIdx].desc} — {contentTypes[ctIdx].example}</p>
      </div>

      <div className="mb-4">
        <label className="text-sm font-medium text-text-secondary">Content-Transfer-Encoding</label>
        <div className="flex flex-wrap gap-2 mt-2">
          {encodings.map((e, i) => (
            <button key={e.name} onClick={() => setEncIdx(i)}
              className={`px-2.5 py-1.5 rounded text-xs font-mono ${encIdx === i ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
              {e.name}
            </button>
          ))}
        </div>
        <p className="text-xs text-text-secondary mt-1">{encodings[encIdx].desc}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm font-medium text-text-secondary">输入内容</label>
          <textarea value={input} onChange={e => setInput(e.target.value)} rows={3}
            className="w-full mt-1 bg-bg-surface border border-border-subtle rounded p-2 text-sm text-text-primary font-mono" />
        </div>
        <div>
          <label className="text-sm font-medium text-text-secondary">Base64 编码结果</label>
          <div className="mt-1 bg-bg-surface border border-border-subtle rounded p-2 text-sm font-mono text-green-400 break-all min-h-[76px]">
            {encIdx === 2 ? encoded : input}
          </div>
        </div>
      </div>

      <div className="bg-bg-surface rounded-lg p-4 border border-border-subtle mb-4">
        <div className="text-sm font-medium text-text-primary mb-2">完整 MIME 头部</div>
        <pre className="text-xs font-mono text-text-secondary whitespace-pre-wrap">
{`MIME-Version: 1.0
Content-Type: ${contentTypes[ctIdx].type}; charset="UTF-8"
Content-Transfer-Encoding: ${encodings[encIdx].name}
Content-Disposition: ${ctIdx >= 2 ? 'attachment; filename="file.dat"' : "inline"}

${encIdx === 2 ? encoded : input}`}
        </pre>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "原始字节", value: `${new TextEncoder().encode(input).length} B` },
          { label: "编码后字节", value: `${(encIdx === 2 ? encoded : input).length} B` },
          { label: "膨胀率", value: `${((encIdx === 2 ? encoded.length : input.length) / Math.max(input.length, 1) * 100).toFixed(0)}%` },
        ].map(s => (
          <div key={s.label} className="bg-bg-surface rounded-lg p-2 text-center">
            <div className="text-xs text-text-secondary">{s.label}</div>
            <div className="font-mono text-sm font-bold text-text-primary">{s.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
export default MIMEDemo;
