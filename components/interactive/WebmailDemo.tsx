"use client";
import { useState } from "react";

interface Component { name: string; role: string; protocol: string; desc: string; port: string; steps?: string[]; examples?: string[]; details?: string }

const components: Component[] = [
  {
    name: "MUA (Mail User Agent)", role: "邮件客户端", protocol: "SMTP/IMAP", port: "—",
    desc: "用户撰写、阅读邮件的界面",
    examples: ["Outlook", "Thunderbird", "Gmail Web", "Apple Mail"],
    details: "MUA 通过 SMTP 提交邮件到 MSA，通过 IMAP/POP3 从邮箱服务器获取邮件",
  },
  {
    name: "MSA (Mail Submission Agent)", role: "邮件提交", protocol: "SMTP", port: "587",
    desc: "接收 MUA 提交的邮件，进行格式验证和身份认证",
    examples: ["Postfix submission", "Exim", "Dovecot Submission"],
    details: "MSA 与 MTA 类似但运行在 587 端口，要求身份认证，用于防止开放中继",
  },
  {
    name: "MTA (Mail Transfer Agent)", role: "邮件传输", protocol: "SMTP", port: "25",
    desc: "在邮件服务器之间转发邮件，通过 MX 记录路由",
    examples: ["Postfix", "Sendmail", "Exim", "Exchange"],
    details: "MTA 通过 DNS MX 记录查找收件人域的邮件服务器，使用 SMTP 协议在 25 端口传输邮件",
  },
  {
    name: "MDA (Mail Delivery Agent)", role: "邮件投递", protocol: "LMTP", port: "—",
    desc: "将邮件投递到用户邮箱存储 (Maildir/Mbox)",
    examples: ["Dovecot LDA", "procmail", "Maildrop"],
    details: "MDA 负责最终投递，支持 Maildir (每邮件一文件) 或 Mbox (单文件存储) 格式",
  },
  {
    name: "Webmail Server", role: "Web 界面", protocol: "HTTP(S)", port: "443",
    desc: "提供浏览器端邮件访问，后端调用 IMAP/SMTP",
    examples: ["Roundcube", "SOGo", "Zimbra", "Gmail"],
    details: "Webmail 前端通过 HTTPS 提供界面，后端通过 IMAP 读取邮件、SMTP 发送邮件",
  },
];

export function WebmailDemo() {
  const [active, setActive] = useState(0);
  const [step, setStep] = useState(0);
  const [mode, setMode] = useState<"send" | "receive">("send");

  const sendSteps = [
    { label: "浏览器 → Web 服务器", desc: "HTTP POST /api/sendmail (MIME 编码的邮件内容)", protocol: "HTTPS" },
    { label: "Web 服务器 → SMTP", desc: "EHLO → MAIL FROM → RCPT TO: user@domain → DATA", protocol: "SMTP" },
    { label: "SMTP → DNS MX 查询", desc: "查找目标域的邮件交换服务器地址", protocol: "DNS" },
    { label: "SMTP → 远程 MTA", desc: "建立 SMTP 连接，传输邮件到目标服务器", protocol: "SMTP" },
    { label: "远程 MTA → 收件箱", desc: "MDA 将邮件投递到目标用户的邮箱存储", protocol: "LMTP" },
  ];

  const recvSteps = [
    { label: "浏览器 → Web 服务器", desc: "HTTP GET /api/inbox (请求邮件列表)", protocol: "HTTPS" },
    { label: "Web 服务器 → IMAP", desc: "LOGIN → SELECT INBOX → FETCH 1:* (ENVELOPE)", protocol: "IMAP" },
    { label: "IMAP → Web 服务器", desc: "返回邮件列表 (发件人/主题/日期/大小)", protocol: "IMAP" },
    { label: "Web 服务器 → 浏览器", desc: "JSON 响应渲染为 HTML 邮件列表视图", protocol: "HTTPS" },
  ];

  const currentSteps = mode === "send" ? sendSteps : recvSteps;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">Webmail 演示</h3>
      <div className="p-4 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <h4 className="text-text-secondary text-xs font-medium mb-2">架构概览</h4>
        <div className="flex items-center gap-2 justify-center flex-wrap">
          {["浏览器", "Web服务器", "IMAP/SMTP", "DNS"].map((c, i) => (
            <div key={c} className="flex items-center gap-2">
              <span className="px-3 py-1.5 rounded bg-blue-500/10 border border-blue-400/30 text-blue-400 text-xs">{c}</span>
              {i < 3 && <span className="text-text-muted">⇄</span>}
            </div>
          ))}
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => { setMode("send"); setStep(0); }}
          className={`px-4 py-2 rounded text-sm font-medium ${mode === "send" ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          发送流程
        </button>
        <button onClick={() => { setMode("receive"); setStep(0); }}
          className={`px-4 py-2 rounded text-sm font-medium ${mode === "receive" ? "bg-green-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          接收流程
        </button>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep((s) => Math.min(s + 1, currentSteps.length - 1))} disabled={step >= currentSteps.length - 1}
          className="px-3 py-1 rounded bg-blue-500 text-white text-sm disabled:opacity-50">下一步</button>
        <button onClick={() => setStep((s) => Math.max(s - 1, 0))} disabled={step === 0}
          className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">上一步</button>
        <button onClick={() => setStep(0)} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm">重置</button>
        <span className="text-text-secondary text-sm self-center">步骤 {step + 1}/{currentSteps.length}</span>
      </div>
      <div className="space-y-2 mb-4">
        {currentSteps.map((s, i) => (
          <div key={i} className={`flex items-center gap-3 p-3 rounded-lg border transition-all ${
            i < step ? "bg-green-500/10 border-green-400/30" : i === step ? "bg-blue-500/10 border-blue-400/30" : "border-border-subtle opacity-40"
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
              i < step ? "bg-green-500 text-white" : i === step ? "bg-blue-500 text-white" : "bg-gray-300 dark:bg-gray-600 text-text-muted"
            }`}>{i + 1}</span>
            <div className="flex-1">
              <span className="text-text-primary text-sm">{s.label}</span>
              <p className="text-text-muted text-xs">{s.desc}</p>
            </div>
            <span className="px-1.5 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-text-secondary text-[10px] shrink-0">{s.protocol}</span>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mb-3">
        {components.map((c, i) => (
          <button key={c.name} onClick={() => setActive(i)}
            className={`p-3 rounded-lg border-2 text-left transition-all cursor-pointer ${
              active === i ? "border-blue-400 bg-blue-500/10" : "border-border-subtle hover:border-gray-400"
            }`}>
            <span className={`text-xs font-medium ${active === i ? "text-blue-400" : "text-text-primary"}`}>{c.name}</span>
            <p className="text-text-muted text-[10px] mt-0.5">{c.role}</p>
          </button>
        ))}
      </div>
      {active !== null && (
        <div className="p-3 rounded-lg bg-bg-primary border border-border-subtle">
          <span className="text-text-primary text-sm font-medium">{components[active].name}</span>
          <p className="text-text-secondary text-xs mt-1">{components[active].desc}</p>
          <p className="text-text-muted text-xs mt-1">端口: {components[active].port} | 协议: {components[active].protocol}</p>
        </div>
      )}
      <div className="text-xs text-text-secondary text-center mt-3">发送: SMTP (25/587) | 接收: IMAP (143/993) | Web: HTTP(S)</div>
    </div>
  );
}
export default WebmailDemo;
