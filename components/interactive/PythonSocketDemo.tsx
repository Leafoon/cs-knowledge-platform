"use client";
import { useState } from "react";

type Proto = "tcp" | "udp";

const codeSnippets: Record<Proto, { title: string; code: string }[]> = {
  tcp: [
    { title: "TCP 服务端", code: `import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 8080))
server.listen(5)

while True:
    conn, addr = server.accept()
    data = conn.recv(1024)
    conn.send(b'HTTP/1.1 200 OK\\r\\n\\r\\nHello')
    conn.close()` },
    { title: "TCP 客户端", code: `import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 8080))
client.send(b'GET / HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n')
response = client.recv(4096)
print(response.decode())
client.close()` },
  ],
  udp: [
    { title: "UDP 服务端", code: `import socket

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(('0.0.0.0', 9090))

while True:
    data, addr = server.recvfrom(1024)
    print(f"来自 {addr}: {data.decode()}")
    server.sendto(b"Pong!", addr)` },
    { title: "UDP 客户端", code: `import socket

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.sendto(b"Hello UDP!", ('127.0.0.1', 9090))
response, addr = client.recvfrom(1024)
print(f"回复: {response.decode()}")
client.close()` },
  ],
};

export function PythonSocketDemo() {
  const [proto, setProto] = useState<Proto>("tcp");
  const [activeTab, setActiveTab] = useState(0);
  const snippets = codeSnippets[proto];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">Python套接字编程演示</h3>
      <div className="flex gap-2 mb-4">
        {(["tcp", "udp"] as const).map((p) => (
          <button key={p} onClick={() => { setProto(p); setActiveTab(0); }}
            className={`px-4 py-1.5 rounded-lg border text-xs font-medium transition-all ${proto === p ? "bg-sky-500/20 border-sky-400/60 text-sky-700 dark:text-sky-300" : "bg-bg-tertiary border-border-subtle text-text-secondary"}`}>
            {p.toUpperCase()}
          </button>
        ))}
      </div>
      <div className="flex gap-2 mb-3">
        {snippets.map((s, i) => (
          <button key={i} onClick={() => setActiveTab(i)}
            className={`px-3 py-1 rounded-lg text-xs border transition-all ${activeTab === i ? "bg-violet-500/15 border-violet-400/40 text-violet-700 dark:text-violet-300" : "bg-bg-tertiary border-border-subtle text-text-secondary"}`}>
            {s.title}
          </button>
        ))}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <pre className="text-xs font-mono text-text-primary whitespace-pre-wrap leading-relaxed overflow-x-auto">{snippets[activeTab].code}</pre>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-medium text-text-primary">{proto === "tcp" ? "TCP 套接字特点" : "UDP 套接字特点"}</div>
        {proto === "tcp" ? (
          <>
            <div>• SOCK_STREAM: 面向连接的可靠传输</div>
            <div>• listen() / accept() / connect(): 三次握手建立连接</div>
            <div>• 内置流量控制和拥塞控制</div>
            <div>• 适合：Web、文件传输、邮件</div>
          </>
        ) : (
          <>
            <div>• SOCK_DGRAM: 无连接的不可靠传输</div>
            <div>• sendto() / recvfrom(): 直接指定目标地址</div>
            <div>• 无连接建立开销，低延迟</div>
            <div>• 适合：DNS查询、视频流、游戏</div>
          </>
        )}
      </div>
    </div>
  );
}
export default PythonSocketDemo;
