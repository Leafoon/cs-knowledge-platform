"use client";
import { useState } from "react";

interface SMTPStep {
  command: string;
  response: string;
  desc: string;
  sender: "client" | "server";
}

const steps: SMTPStep[] = [
  { command: "", response: "220 mail.example.com ESMTP Postfix", desc: "服务器就绪，等待连接", sender: "server" },
  { command: "EHLO client.example.com", response: "250-mail.example.com\n250-STARTTLS\n250 SIZE 10485760", desc: "客户端标识自己，服务器返回支持的扩展", sender: "client" },
  { command: "AUTH LOGIN", response: "334 VXNlcm5hbWU6", desc: "请求认证（Base64编码的Username:提示）", sender: "client" },
  { command: "dXNlcm5hbWU=", response: "334 UGFzc3dvcmQ6", desc: "发送Base64编码的用户名", sender: "client" },
  { command: "cGFzc3dvcmQ=", response: "235 Authentication successful", desc: "发送Base64编码的密码，认证成功", sender: "client" },
  { command: "MAIL FROM:<sender@example.com>", response: "250 OK", desc: "指定发件人地址", sender: "client" },
  { command: "RCPT TO:<receiver@example.com>", response: "250 OK", desc: "指定收件人地址", sender: "client" },
  { command: "DATA", response: "354 End data with <CR><LF>.<CR><LF>", desc: "开始发送邮件正文", sender: "client" },
  { command: "Subject: 测试邮件\nFrom: sender@example.com\nTo: receiver@example.com\n\n这是一封测试邮件。\n.", response: "250 OK: Message queued", desc: "发送邮件内容，单行.表示结束", sender: "client" },
  { command: "QUIT", response: "221 Bye", desc: "关闭连接", sender: "client" },
];

export function SMTPFlowDemo() {
  const [activeStep, setActiveStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);

  const step = steps[activeStep];

  const handleAutoPlay = () => {
    setAutoPlay(true);
    setActiveStep(0);
    let i = 0;
    const timer = setInterval(() => {
      i++;
      if (i >= steps.length) { clearInterval(timer); setAutoPlay(false); return; }
      setActiveStep(i);
    }, 1500);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">SMTP流程演示</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setActiveStep(Math.max(0, activeStep - 1))} disabled={activeStep === 0}
          className="px-3 py-1 rounded-lg bg-bg-tertiary border border-border-subtle text-xs disabled:opacity-30">←</button>
        <button onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))} disabled={activeStep === steps.length - 1}
          className="px-3 py-1 rounded-lg bg-bg-tertiary border border-border-subtle text-xs disabled:opacity-30">→</button>
        <button onClick={handleAutoPlay} disabled={autoPlay}
          className="px-3 py-1 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-xs font-medium hover:bg-sky-500/25 disabled:opacity-50 transition-colors">
          {autoPlay ? "播放中..." : "▶ 自动播放"}
        </button>
        <span className="text-xs text-text-tertiary ml-auto">{activeStep + 1}/{steps.length}</span>
      </div>
      <div className="flex gap-2 mb-4">
        {steps.map((s, i) => (
          <button key={i} onClick={() => setActiveStep(i)}
            className={`flex-1 h-1.5 rounded-full transition-all ${i <= activeStep ? "bg-sky-500" : "bg-bg-tertiary"}`} />
        ))}
      </div>
      <div className={`rounded-lg border p-4 mb-4 ${step.sender === "server" ? "border-emerald-500/30 bg-emerald-500/5" : "border-sky-500/30 bg-sky-500/5"}`}>
        <div className="flex items-center gap-2 mb-2">
          <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${step.sender === "server" ? "bg-emerald-500/20 text-emerald-600 dark:text-emerald-400" : "bg-sky-500/20 text-sky-600 dark:text-sky-400"}`}>
            {step.sender === "server" ? "← 服务器" : "→ 客户端"}
          </span>
        </div>
        {step.sender === "client" && step.command && (
          <div className="mb-2">
            <div className="text-[10px] text-text-tertiary mb-0.5">命令:</div>
            <pre className="text-xs font-mono text-sky-700 dark:text-sky-300 whitespace-pre-wrap">{step.command}</pre>
          </div>
        )}
        <div>
          <div className="text-[10px] text-text-tertiary mb-0.5">响应:</div>
          <pre className="text-xs font-mono text-emerald-700 dark:text-emerald-300 whitespace-pre-wrap">{step.response}</pre>
        </div>
        <div className="text-xs text-text-secondary mt-2">{step.desc}</div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-medium text-text-primary">SMTP关键点</div>
        <div>• 基于文本的命令-响应协议，默认端口25（提交）/587（加密提交）</div>
        <div>• DATA命令后邮件内容以单独一行.结束</div>
        <div>• 扩展命令（EHLO）支持认证、大小限制、STARTTLS等</div>
      </div>
    </div>
  );
}
export default SMTPFlowDemo;
