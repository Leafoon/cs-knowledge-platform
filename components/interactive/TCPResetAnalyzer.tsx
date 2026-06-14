"use client";
import { useState } from "react";

interface Scenario { id: string; title: string; desc: string; trigger: string; severity: "low" | "medium" | "high"; rstType: string; flags: string; }

const scenarios: Scenario[] = [
  { id: "port", title: "端口未监听", desc: "客户端尝试连接未开放的端口，服务器立即回复 RST", trigger: "SYN → RST", severity: "medium", rstType: "RST (无 ACK)", flags: "RST=1, ACK=0" },
  { id: "abort", title: "应用层中止连接", desc: "应用进程异常退出，内核发送 RST 清理连接", trigger: "RST 发送", severity: "high", rstType: "RST+ACK", flags: "RST=1, ACK=1" },
  { id: "timeout", title: "半开连接检测", desc: "长时间无数据后收到报文，对端已重启", trigger: "数据 → RST", severity: "medium", rstType: "RST+ACK", flags: "RST=1, ACK=1" },
  { id: "invalid", title: "无效序列号", desc: "收到的 TCP 报文序列号不在接收窗口内", trigger: "无效 SEQ → RST", severity: "low", rstType: "RST+ACK", flags: "RST=1, ACK=1" },
  { id: "security", title: "防火墙拒绝", desc: "安全设备主动发送 RST 阻断非法连接", trigger: "SYN → RST", severity: "high", rstType: "RST (无 ACK)", flags: "RST=1, ACK=0" },
  { id: "accept", title: "accept 队列满", desc: "SYN 队列已满，新连接被 RST 拒绝", trigger: "SYN → RST", severity: "high", rstType: "RST (无 ACK)", flags: "RST=1, ACK=0" },
];

export function TCPResetAnalyzer() {
  const [selected, setSelected] = useState<string>("port");
  const [log, setLog] = useState<string[]>([]);
  const [step, setStep] = useState(0);
  const active = scenarios.find((s) => s.id === selected)!;

  const simulate = () => {
    setLog([]);
    setStep(0);
    const steps = [
      `1. 触发条件: ${active.trigger}`,
      `2. 场景描述: ${active.desc}`,
      `3. RST 报文类型: ${active.rstType}`,
      `4. 标志位: ${active.flags}`,
      `5. 发送 RST 报文 → 对端收到后连接状态 → CLOSED`,
      `6. 应用层收到通知: ECONNRESET (Connection reset by peer)`,
      `7. 如果是 accept 队列满，可通过增大 backlog 或调整 somaxconn 处理`,
    ];
    steps.forEach((s, i) => {
      setTimeout(() => {
        setLog((l) => [...l, s]);
        setStep(i + 1);
      }, (i + 1) * 600);
    });
  };

  const severityColor: Record<string, string> = { low: "bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300", medium: "bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-300", high: "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300" };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">TCP RST 分析器</h3>
      <div className="text-xs text-text-secondary mb-3">分析 TCP RST (Reset) 报文的触发场景和处理流程</div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 mb-4">
        {scenarios.map((s) => (
          <button key={s.id} onClick={() => { setSelected(s.id); setLog([]); setStep(0); }} className={`p-2 rounded-lg border text-left transition-colors ${selected === s.id ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20" : "border-border-subtle bg-gray-50 dark:bg-gray-800 hover:border-gray-400"}`}>
            <div className="text-xs font-medium text-text-primary">{s.title}</div>
            <span className={`inline-block mt-1 text-xs px-1.5 py-0.5 rounded ${severityColor[s.severity]}`}>{s.severity === "low" ? "低" : s.severity === "medium" ? "中" : "高"}</span>
          </button>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle mb-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm font-medium text-text-primary">{active.title}</span>
          <span className={`px-2 py-0.5 rounded text-xs ${severityColor[active.severity]}`}>严重性: {active.severity === "low" ? "低" : active.severity === "medium" ? "中" : "高"}</span>
        </div>
        <div className="text-sm text-text-secondary mb-2">{active.desc}</div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="p-2 rounded bg-white dark:bg-gray-900 border border-border-subtle text-center">
            <div className="text-text-secondary">触发模式</div>
            <div className="font-mono text-text-primary">{active.trigger}</div>
          </div>
          <div className="p-2 rounded bg-white dark:bg-gray-900 border border-border-subtle text-center">
            <div className="text-text-secondary">RST 类型</div>
            <div className="font-mono text-text-primary">{active.rstType}</div>
          </div>
          <div className="p-2 rounded bg-white dark:bg-gray-900 border border-border-subtle text-center">
            <div className="text-text-secondary">标志位</div>
            <div className="font-mono text-text-primary">{active.flags}</div>
          </div>
        </div>
      </div>
      <button onClick={simulate} className="w-full py-2 rounded bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors mb-3">模拟 RST 过程</button>
      {log.length > 0 && (
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-xs font-mono space-y-1">
          {log.map((l, i) => <div key={i} className={`py-0.5 ${l.includes("RST") ? "text-red-500" : l.includes("ECONNRESET") ? "text-orange-500" : "text-text-secondary"}`}>{l}</div>)}
        </div>
      )}
      {step > 0 && (
        <div className="mt-3 flex items-center gap-1 text-xs">
          {Array.from({ length: 7 }, (_, i) => (
            <div key={i} className={`flex-1 h-2 rounded-full ${i < step ? "bg-blue-500" : "bg-gray-200 dark:bg-gray-700"}`} />
          ))}
        </div>
      )}
    </div>
  );
}
export default TCPResetAnalyzer;
