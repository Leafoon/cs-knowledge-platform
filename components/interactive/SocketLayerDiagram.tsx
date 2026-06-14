"use client";
import { useState } from "react";

const layers = [
  { id: "fd", label: "文件描述符 (File Descriptor)", color: "blue", desc: "整数标识符，进程通过它引用打开的套接字。fd 0/1/2 分别为 stdin/stdout/stderr。", items: ["fd=3 (TCP socket)", "fd=4 (UDP socket)", "fd=5 (raw socket)"] },
  { id: "socket", label: "套接字层 (Socket Layer)", color: "green", desc: "BSD Socket API 提供统一接口，屏蔽底层协议差异。支持 SOCK_STREAM/SOCK_DGRAM/SOCK_RAW。", items: ["socket()", "bind()", "listen()/accept()", "connect()", "send()/recv()"] },
  { id: "proto", label: "协议栈 (Protocol Stack)", color: "yellow", desc: "实现传输层和网络层协议，处理分段、重组、拥塞控制、路由等。", items: ["TCP (可靠/有序)", "UDP (无连接/轻量)", "ICMP (差错报告)", "IP (寻址/路由)"] },
  { id: "nic", label: "网卡 (NIC / Network Interface)", color: "red", desc: "网络接口控制器，负责帧的发送与接收，完成电信号/光信号与数字数据的转换。", items: ["以太网卡 (eth0)", "WiFi (wlan0)", "环回接口 (lo)"] },
];

export function SocketLayerDiagram() {
  const [active, setActive] = useState<string | null>(null);
  const colorMap: Record<string, { bg: string; border: string; text: string }> = {
    blue: { bg: "bg-blue-500/10", border: "border-blue-400", text: "text-blue-400" },
    green: { bg: "bg-green-500/10", border: "border-green-400", text: "text-green-400" },
    yellow: { bg: "bg-yellow-500/10", border: "border-yellow-400", text: "text-yellow-400" },
    red: { bg: "bg-red-500/10", border: "border-red-400", text: "text-red-400" },
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">套接字层架构图</h3>
      <p className="text-text-secondary text-sm mb-4">点击各层查看详细说明：文件描述符 → 套接字 → 协议栈 → 网卡</p>
      <div className="flex flex-col items-center gap-2">
        {layers.map((layer, i) => {
          const c = colorMap[layer.color];
          const isActive = active === layer.id;
          return (
            <div key={layer.id} className="w-full max-w-lg">
              <button
                onClick={() => setActive(isActive ? null : layer.id)}
                className={`w-full p-4 rounded-lg border-2 transition-all cursor-pointer ${c.bg} ${isActive ? c.border + " shadow-lg scale-[1.02]" : "border-border-subtle hover:border-gray-400"}`}
              >
                <span className={`font-semibold text-base ${c.text}`}>{layer.label}</span>
              </button>
              {i < layers.length - 1 && (
                <div className="flex justify-center py-1">
                  <svg width="24" height="20" viewBox="0 0 24 20"><path d="M12 0 L12 14 M6 10 L12 16 L18 10" stroke="currentColor" strokeWidth="2" fill="none" className="text-text-muted" /></svg>
                </div>
              )}
              {isActive && (
                <div className={`mt-1 p-4 rounded-lg ${c.bg} border ${c.border}`}>
                  <p className="text-text-primary text-sm mb-2">{layer.desc}</p>
                  <div className="flex flex-wrap gap-2">
                    {layer.items.map((item) => (
                      <span key={item} className={`text-xs px-2 py-1 rounded ${c.bg} ${c.text} border ${c.border}`}>{item}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-4 p-3 rounded bg-bg-primary border border-border-subtle mb-3">
        <p className="text-text-muted text-xs">
          数据流向：应用进程 → fd(文件描述符) → Socket API → TCP/UDP协议处理 → IP路由 → NIC(网卡) → 物理网络
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">阻塞 vs 非阻塞</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 阻塞: recv() 无数据时进程挂起</li>
            <li>• 非阻塞: 立即返回 EAGAIN/EWOULDBLOCK</li>
            <li>• 多路复用: select/poll/epoll 监听多个 fd</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">Socket 类型</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• SOCK_STREAM: TCP，可靠有序字节流</li>
            <li>• SOCK_DGRAM: UDP，无连接数据报</li>
            <li>• SOCK_RAW: 原始套接字，直接构造 IP 包</li>
          </ul>
        </div>
      </div>
      <div className="mt-3 p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-primary text-xs font-medium mb-1">系统调用开销</h4>
        <p className="text-text-muted text-xs">每次 read()/write() 需要用户态↔内核态切换 (上下文切换 ~1μs)。零拷贝技术 (sendfile/splice/mmap) 可减少数据拷贝次数，提升 I/O 性能。</p>
      </div>
    </div>
  );
}
export default SocketLayerDiagram;
