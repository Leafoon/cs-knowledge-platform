"use client";
import { useState } from "react";

export function UDPDemo() {
  const [messages, setMessages] = useState<{ id: number; text: string; status: "sent" | "received" | "lost" }[]>([]);
  const [input, setInput] = useState("");
  const [lossRate, setLossRate] = useState(10);
  const [showHeader, setShowHeader] = useState(false);
  const [nextId, setNextId] = useState(1);

  const send = () => {
    if (!input.trim()) return;
    const lost = Math.random() * 100 < lossRate;
    setMessages((m) => [...m, { id: nextId, text: input, status: lost ? "lost" : "sent" }]);
    if (!lost) {
      setTimeout(() => {
        setMessages((m) => m.map((msg) => msg.id === nextId ? { ...msg, status: "received" } : msg));
      }, 500);
    }
    setNextId((n) => n + 1);
    setInput("");
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">UDP 演示</h3>
      <div className="flex gap-2 mb-4">
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()}
          className="flex-1 px-3 py-2 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm"
          placeholder="输入消息..." />
        <button onClick={send} className="px-4 py-2 rounded bg-blue-500 text-white text-sm hover:bg-blue-600">发送</button>
        <button onClick={() => setMessages([])} className="px-3 py-2 rounded border border-border-subtle text-text-muted text-sm">清空</button>
      </div>
      <div className="mb-4">
        <label className="text-text-muted text-xs">丢包率: {lossRate}%</label>
        <input type="range" min="0" max="50" value={lossRate} onChange={(e) => setLossRate(Number(e.target.value))}
          className="w-full accent-red-500" />
      </div>
      <div className="flex items-center gap-4 mb-4">
        <button onClick={() => setShowHeader(!showHeader)} className="text-xs text-blue-400 hover:underline">
          {showHeader ? "隐藏" : "显示"} UDP 头部
        </button>
      </div>
      {showHeader && (
        <div className="p-3 rounded-lg bg-bg-primary border border-border-subtle mb-4">
          <h4 className="text-text-primary text-xs font-medium mb-2">UDP 头部 (8 字节)</h4>
          <div className="flex gap-px">
            {["源端口 (16bit)", "目的端口 (16bit)", "长度 (16bit)", "校验和 (16bit)"].map((f) => (
              <div key={f} className="flex-1 p-2 rounded bg-blue-500/10 border border-blue-400/30 text-center">
                <span className="text-blue-400 text-[10px]">{f}</span>
              </div>
            ))}
          </div>
          <p className="text-text-muted text-xs mt-2">无序列号、无确认号、无连接状态 — 极简设计</p>
        </div>
      )}
      <div className="space-y-1 mb-4 max-h-48 overflow-y-auto">
        {messages.map((m) => (
          <div key={m.id} className={`flex items-center gap-2 p-2 rounded text-sm ${
            m.status === "sent" ? "bg-blue-500/10" : m.status === "received" ? "bg-green-500/10" : "bg-red-500/10"
          }`}>
            <span className={`text-xs ${m.status === "sent" ? "text-blue-400" : m.status === "received" ? "text-green-400" : "text-red-400"}`}>
              {m.status === "sent" ? "📤" : m.status === "received" ? "✅" : "❌"}
            </span>
            <span className="text-text-primary">{m.text}</span>
            <span className="text-text-muted text-xs ml-auto">
              {m.status === "lost" ? "数据包丢失 (无重传)" : m.status === "sent" ? "已发送..." : "已接收"}
            </span>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-2 rounded bg-green-500/10 border border-green-400/30">
          <span className="text-green-400 text-xs font-medium">UDP 优势</span>
          <ul className="text-text-muted text-xs mt-1 space-y-0.5">
            <li>• 无连接建立延迟</li>
            <li>• 头部仅 8 字节</li>
            <li>• 支持广播/多播</li>
          </ul>
        </div>
        <div className="p-2 rounded bg-red-500/10 border border-red-400/30">
          <span className="text-red-400 text-xs font-medium">UDP 限制</span>
          <ul className="text-text-muted text-xs mt-1 space-y-0.5">
            <li>• 不可靠交付（丢包不重传）</li>
            <li>• 无流量/拥塞控制</li>
            <li>• 无顺序保证</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default UDPDemo;
