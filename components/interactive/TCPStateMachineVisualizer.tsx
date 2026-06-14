"use client";
import { useState } from "react";

const states = [
  "CLOSED", "LISTEN", "SYN_SENT", "SYN_RCVD", "ESTABLISHED",
  "FIN_WAIT_1", "FIN_WAIT_2", "CLOSING", "TIME_WAIT", "CLOSE_WAIT", "LAST_ACK",
];

const transitions: { from: string; to: string; event: string; color: string }[] = [
  { from: "CLOSED", to: "LISTEN", event: "被动 open()", color: "blue" },
  { from: "CLOSED", to: "SYN_SENT", event: "主动 connect()", color: "blue" },
  { from: "LISTEN", to: "SYN_RCVD", event: "收到 SYN", color: "green" },
  { from: "LISTEN", to: "SYN_SENT", event: "发送 SYN", color: "blue" },
  { from: "SYN_SENT", to: "ESTABLISHED", event: "收到 SYN+ACK, 发送 ACK", color: "green" },
  { from: "SYN_SENT", to: "CLOSED", event: "超时/关闭", color: "red" },
  { from: "SYN_RCVD", to: "ESTABLISHED", event: "收到 ACK", color: "green" },
  { from: "SYN_RCVD", to: "CLOSED", event: "RST/超时", color: "red" },
  { from: "ESTABLISHED", to: "FIN_WAIT_1", event: "主动 close(), 发送 FIN", color: "yellow" },
  { from: "ESTABLISHED", to: "CLOSE_WAIT", event: "收到 FIN, 发送 ACK", color: "yellow" },
  { from: "FIN_WAIT_1", to: "FIN_WAIT_2", event: "收到 ACK", color: "green" },
  { from: "FIN_WAIT_1", to: "CLOSING", event: "收到 FIN", color: "yellow" },
  { from: "FIN_WAIT_1", to: "TIME_WAIT", event: "收到 FIN+ACK", color: "green" },
  { from: "FIN_WAIT_2", to: "TIME_WAIT", event: "收到 FIN, 发送 ACK", color: "green" },
  { from: "CLOSING", to: "TIME_WAIT", event: "收到 ACK", color: "green" },
  { from: "CLOSE_WAIT", to: "LAST_ACK", event: "应用 close(), 发送 FIN", color: "yellow" },
  { from: "LAST_ACK", to: "CLOSED", event: "收到 ACK", color: "green" },
  { from: "TIME_WAIT", to: "CLOSED", event: "2MSL 超时", color: "red" },
];

export function TCPStateMachineVisualizer() {
  const [current, setCurrent] = useState("CLOSED");
  const [log, setLog] = useState<string[]>([]);

  const available = transitions.filter((t) => t.from === current);

  const fire = (t: typeof transitions[0]) => {
    setCurrent(t.to);
    setLog((l) => [...l.slice(-8), `${t.from} → ${t.to}: ${t.event}`]);
  };

  const colorMap: Record<string, string> = {
    blue: "border-blue-400 bg-blue-500/10 text-blue-400",
    green: "border-green-400 bg-green-500/10 text-green-400",
    yellow: "border-yellow-400 bg-yellow-500/10 text-yellow-400",
    red: "border-red-400 bg-red-500/10 text-red-400",
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCP 状态机可视化</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {states.map((s) => (
          <span key={s} className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium border-2 transition-all ${
            current === s ? "border-blue-400 bg-blue-500/20 text-blue-400 shadow-md" : "border-border-subtle text-text-muted"
          }`}>{s}</span>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <h4 className="text-text-primary text-sm font-medium mb-2">当前状态: <span className="text-blue-400">{current}</span></h4>
        {available.length > 0 ? (
          <div className="space-y-2">
            {available.map((t, i) => (
              <button key={i} onClick={() => fire(t)}
                className={`w-full text-left p-3 rounded-lg border-2 transition-all cursor-pointer ${colorMap[t.color]} hover:shadow-md`}>
                <span className="text-sm font-medium">{t.event}</span>
                <span className="text-text-muted text-xs ml-2">→ {t.to}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="text-text-muted text-sm">无可用转换（终态）</p>
        )}
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-secondary text-xs font-medium mb-1">转换日志</h4>
        {log.map((l, i) => <p key={i} className="text-text-muted text-xs font-mono">{l}</p>)}
        {log.length === 0 && <p className="text-text-muted text-xs">点击上方按钮触发状态转换</p>}
      </div>
      <p className="text-text-muted text-xs mt-3">TIME_WAIT 持续 2MSL (约 4 分钟)，确保最后的 ACK 到达且旧报文消亡。</p>
    </div>
  );
}
export default TCPStateMachineVisualizer;
