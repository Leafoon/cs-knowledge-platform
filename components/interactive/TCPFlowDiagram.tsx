"use client";
import { useState } from "react";

export function TCPFlowDiagram() {
  const [step, setStep] = useState(0);
  const maxStep = 8;

  const flows = [
    { time: 0, from: "client", to: "server", seq: 0, ack: 0, data: "SYN", color: "yellow" },
    { time: 1, from: "server", to: "client", seq: 0, ack: 1, data: "SYN-ACK", color: "yellow" },
    { time: 2, from: "client", to: "server", seq: 1, ack: 1, data: "ACK (三次握手)", color: "green" },
    { time: 3, from: "client", to: "server", seq: 1, ack: 1, data: "DATA [100B]", color: "blue" },
    { time: 4, from: "client", to: "server", seq: 101, ack: 1, data: "DATA [200B]", color: "blue" },
    { time: 5, from: "server", to: "client", seq: 1, ack: 101, data: "ACK", color: "green" },
    { time: 6, from: "server", to: "client", seq: 1, ack: 301, data: "ACK (累积确认)", color: "green" },
    { time: 7, from: "client", to: "server", seq: 301, ack: 1, data: "FIN", color: "red" },
    { time: 8, from: "server", to: "client", seq: 1, ack: 302, data: "FIN-ACK", color: "red" },
  ];

  const colorMap: Record<string, string> = {
    yellow: "bg-yellow-500/20 border-yellow-400 text-yellow-400",
    green: "bg-green-500/20 border-green-400 text-green-400",
    blue: "bg-blue-500/20 border-blue-400 text-blue-400",
    red: "bg-red-500/20 border-red-400 text-red-400",
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCP 流量图 (TCP Flow Diagram)</h3>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
          className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">←</button>
        <span className="text-text-secondary text-sm">{step + 1}/{maxStep + 1}</span>
        <button onClick={() => setStep((s) => Math.min(maxStep, s + 1))} disabled={step === maxStep}
          className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">→</button>
        <button onClick={() => setStep(0)} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm">重置</button>
      </div>
      <div className="relative mb-4">
        <div className="flex justify-between mb-2">
          <span className="text-blue-400 text-sm font-medium w-24 text-center">客户端</span>
          <span className="text-green-400 text-sm font-medium w-24 text-center">服务器</span>
        </div>
        <div className="flex justify-between mb-1">
          <div className="w-px h-4 bg-blue-400/30 mx-auto" />
          <div className="w-px h-4 bg-green-400/30 mx-auto" />
        </div>
        <div className="flex justify-between">
          <div className="w-px flex-1 bg-blue-400/20" style={{ minHeight: `${(step + 1) * 48}px` }} />
          <div className="w-px flex-1 bg-green-400/20" style={{ minHeight: `${(step + 1) * 48}px` }} />
        </div>
        <div className="absolute top-12 left-0 right-0 space-y-2">
          {flows.slice(0, step + 1).map((f, i) => {
            const isFromClient = f.from === "client";
            return (
              <div key={i} className={`flex ${isFromClient ? "justify-start" : "justify-end"} px-4`}>
                <div className={`relative px-3 py-1.5 rounded border ${colorMap[f.color]}`}>
                  <span className="text-xs font-mono">{f.data}</span>
                  <div className="flex gap-3 text-[10px] text-text-muted mt-0.5">
                    <span>seq={f.seq}</span>
                    <span>ack={f.ack}</span>
                  </div>
                  <div className={`absolute top-1/2 ${isFromClient ? "right-0 translate-x-full" : "left-0 -translate-x-full"} w-4 h-px bg-current`} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle mb-3">
        <p className="text-text-muted text-xs">TCP seq/ack 编号按字节计数。seq 表示发送数据的起始字节号，ack 表示期望收到的下一个字节号。三次握手建立连接，四次挥手 (简化为两次) 关闭连接。</p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">连接状态机</h4>
          <div className="text-text-muted text-xs space-y-0.5">
            <div>CLOSED → SYN_SENT → ESTABLISHED</div>
            <div>LISTEN → SYN_RCVD → ESTABLISHED</div>
            <div>ESTABLISHED → FIN_WAIT_1 → ...</div>
            <div>TIME_WAIT (2MSL) → CLOSED</div>
          </div>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">关键机制</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 累积确认: ACK N 表示 N 之前全部收到</li>
            <li>• 滑动窗口: 流量控制 (rwnd)</li>
            <li>• 拥塞控制: cwnd 慢启动/拥塞避免</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default TCPFlowDiagram;
