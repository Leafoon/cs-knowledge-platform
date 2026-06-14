"use client";
import { useState } from "react";

const templates = [
  {
    lang: "C",
    code: `// UDP 服务器
#include <sys/socket.h>
#include <netinet/in.h>

int sock = socket(AF_INET, SOCK_DGRAM, 0);

struct sockaddr_in addr = {
  .sin_family = AF_INET,
  .sin_port = htons(53),
  .sin_addr.s_addr = INADDR_ANY
};

bind(sock, (struct sockaddr*)&addr, sizeof(addr));

char buf[512];
struct sockaddr_in client;
socklen_t len = sizeof(client);

// 接收数据报
int n = recvfrom(sock, buf, sizeof(buf), 0,
  (struct sockaddr*)&client, &len);

// 发送响应
sendto(sock, response, resp_len, 0,
  (struct sockaddr*)&client, len);

close(sock);`,
  },
  {
    lang: "Python",
    code: `# UDP 服务器
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 53))

# 接收数据报
data, addr = sock.recvfrom(512)
print(f"收到来自 {addr}: {data}")

# 发送响应
sock.sendto(response, addr)

sock.close()

# --- UDP 客户端 ---
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.sendto(b'query', ('127.0.0.1', 53))
data, server = client.recvfrom(512)
client.close()`,
  },
];

const apiFlow = [
  { step: "socket()", desc: "创建 UDP 套接字 (SOCK_DGRAM)", color: "blue" },
  { step: "bind()", desc: "绑定本地地址和端口（服务器必需）", color: "green" },
  { step: "sendto() / recvfrom()", desc: "发送/接收数据报（无需建立连接）", color: "yellow" },
  { step: "close()", desc: "关闭套接字", color: "red" },
];

export function UDPProgrammingPlayground() {
  const [lang, setLang] = useState(0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">UDP 编程沙盒</h3>
      <div className="flex gap-2 mb-4">
        {templates.map((t, i) => (
          <button key={t.lang} onClick={() => setLang(i)}
            className={`px-4 py-2 rounded text-sm font-medium ${lang === i ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted"}`}>
            {t.lang}
          </button>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <pre className="text-text-primary text-xs font-mono whitespace-pre-wrap overflow-x-auto">{templates[lang].code}</pre>
      </div>
      <h4 className="text-text-secondary text-sm font-medium mb-2">Socket API 流程</h4>
      <div className="space-y-2 mb-4">
        {apiFlow.map((a, i) => {
          const colorMap: Record<string, string> = { blue: "border-blue-400 bg-blue-500/10", green: "border-green-400 bg-green-500/10", yellow: "border-yellow-400 bg-yellow-500/10", red: "border-red-400 bg-red-500/10" };
          const textMap: Record<string, string> = { blue: "text-blue-400", green: "text-green-400", yellow: "text-yellow-400", red: "text-red-400" };
          return (
            <div key={i} className="flex items-center gap-3">
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${textMap[a.color]} bg-bg-elevated border ${colorMap[a.color].split(" ")[0]}`}>{i + 1}</span>
              <div className={`flex-1 p-3 rounded-lg border ${colorMap[a.color]}`}>
                <span className={`font-mono text-sm ${textMap[a.color]}`}>{a.step}</span>
                <p className="text-text-secondary text-xs mt-0.5">{a.desc}</p>
              </div>
            </div>
          );
        })}
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-secondary text-xs font-medium mb-1">UDP vs TCP 套接字对比</h4>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><span className="text-blue-400 font-mono">SOCK_DGRAM</span><span className="text-text-muted"> → UDP 数据报服务</span></div>
          <div><span className="text-green-400 font-mono">SOCK_STREAM</span><span className="text-text-muted"> → TCP 字节流服务</span></div>
        </div>
      </div>
    </div>
  );
}
export default UDPProgrammingPlayground;
