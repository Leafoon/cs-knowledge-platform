"use client";
import { useState, useRef } from "react";

interface WsMessage { id: number; from: "client" | "server"; data: string; opcode: string; time: string; }

const opcodeNames: Record<number, string> = { 0x0: "Continuation", 0x1: "Text", 0x2: "Binary", 0x8: "Close", 0x9: "Ping", 0xA: "Pong" };

export function WebSocketDemo() {
  const [messages, setMessages] = useState<WsMessage[]>([]);
  const [input, setInput] = useState("Hello WebSocket!");
  const [connected, setConnected] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [msgType, setMsgType] = useState<"text" | "binary" | "ping">("text");
  const idRef = useRef(0);

  const connect = () => {
    setLog((l) => [...l, "→ GET /chat HTTP/1.1", "  Upgrade: websocket", "  Connection: Upgrade", "  Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==", "← HTTP/1.1 101 Switching Protocols", "  Upgrade: websocket", "  Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", "✓ WebSocket 全双工连接建立完成"]);
    setConnected(true);
  };

  const disconnect = () => {
    setLog((l) => [...l, "→ Close Frame (opcode=0x8)", "← Close Frame (opcode=0x8)", "✗ 连接已正常关闭"]);
    setConnected(false);
  };

  const sendMessage = (from: "client" | "server") => {
    const id = ++idRef.current;
    const time = new Date().toLocaleTimeString();
    const data = from === "client" ? input : `服务器回复: ${input}`;
    const opcode = msgType === "text" ? "0x1 Text" : msgType === "binary" ? "0x2 Binary" : "0x9 Ping";

    if (msgType === "ping") {
      setMessages((m) => [...m.slice(-19), { id, from, data: "Ping!", opcode: "0x9 Ping", time }]);
      setLog((l) => [...l, `${from === "client" ? "→" : "←"} Ping Frame (opcode=0x9)`]);
      setTimeout(() => {
        setMessages((m) => [...m.slice(-19), { id: id + 1, from: from === "client" ? "server" : "client", data: "Pong!", opcode: "0xA Pong", time }]);
        setLog((l) => [...l, `${from === "client" ? "←" : "→"} Pong Frame (opcode=0xA)`]);
      }, 200);
    } else {
      setMessages((m) => [...m.slice(-19), { id, from, data, opcode, time }]);
      setLog((l) => [...l, `${from === "client" ? "→" : "←"} ${opcode} Frame (${data.length} bytes): ${data.slice(0, 50)}`]);
    }
  };

  const clearAll = () => { setMessages([]); setLog([]); idRef.current = 0; };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">WebSocket 全双工通信演示</h3>
      <div className="flex items-center gap-3 mb-4 p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <span className={`w-3 h-3 rounded-full ${connected ? "bg-green-500" : "bg-gray-400"}`} />
        <span className="text-sm text-text-secondary">{connected ? "已连接 Connected" : "未连接 Disconnected"}</span>
        <div className="flex-1" />
        <button onClick={connected ? disconnect : connect} className={`px-4 py-1.5 rounded text-sm ${connected ? "bg-red-600 hover:bg-red-700 text-white" : "bg-green-600 hover:bg-green-700 text-white"}`}>
          {connected ? "断开连接" : "建立连接"}
        </button>
      </div>
      <div className="flex gap-2 mb-4">
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="消息内容" disabled={!connected} className="flex-1 px-3 py-1.5 rounded border border-border-subtle bg-gray-50 dark:bg-gray-800 text-sm text-text-primary disabled:opacity-50" />
        <select value={msgType} onChange={(e) => setMsgType(e.target.value as "text" | "binary" | "ping")} className="px-2 py-1.5 rounded border border-border-subtle bg-white dark:bg-gray-900 text-xs text-text-primary">
          <option value="text">Text Frame</option>
          <option value="binary">Binary Frame</option>
          <option value="ping">Ping/Pong</option>
        </select>
        <button onClick={() => sendMessage("client")} disabled={!connected} className="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-700 text-white text-sm disabled:opacity-50">客户端发送</button>
        <button onClick={() => sendMessage("server")} disabled={!connected} className="px-3 py-1.5 rounded bg-green-600 hover:bg-green-700 text-white text-sm disabled:opacity-50">服务端发送</button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle max-h-48 overflow-y-auto">
          <div className="text-xs text-text-secondary mb-2">消息面板 Messages</div>
          {messages.length === 0 ? (
            <div className="text-xs text-text-secondary text-center py-4">建立连接后发送消息</div>
          ) : messages.map((m) => (
            <div key={m.id} className={`flex gap-2 py-1 text-xs border-b border-border-subtle last:border-0 ${m.from === "client" ? "" : "flex-row-reverse"}`}>
              <span className={`px-1.5 py-0.5 rounded shrink-0 ${m.from === "client" ? "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300" : "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300"}`}>{m.from === "client" ? "Client" : "Server"}</span>
              <span className="text-text-primary truncate">{m.data}</span>
              <span className="text-text-secondary ml-auto shrink-0">{m.time}</span>
            </div>
          ))}
        </div>
        <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle max-h-48 overflow-y-auto">
          <div className="text-xs text-text-secondary mb-2">协议日志 Protocol Log</div>
          {log.map((l, i) => <div key={i} className="text-xs font-mono py-0.5 text-text-secondary">{l}</div>)}
        </div>
      </div>
      <button onClick={clearAll} className="px-4 py-1.5 rounded bg-gray-200 dark:bg-gray-700 text-text-secondary text-sm">清空全部</button>
      <div className="mt-3 text-xs text-text-secondary text-center">WebSocket: 全双工 | 持久连接 | 2-14B 帧头 | 通过 HTTP Upgrade 握手</div>
    </div>
  );
}
export default WebSocketDemo;
