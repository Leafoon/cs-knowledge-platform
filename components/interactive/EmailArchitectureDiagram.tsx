"use client";
import { useState } from "react";

interface Stage {
  name: string;
  role: string;
  protocol: string;
  desc: string;
  color: string;
}

const STAGES: Stage[] = [
  { name: "发件人UA", role: "用户代理", protocol: "SMTP", desc: "用户撰写邮件,通过UA提交", color: "bg-blue-500" },
  { name: "发件MTA", role: "邮件传输代理", protocol: "SMTP", desc: "UA将邮件发送到本地MTA", color: "bg-green-500" },
  { name: "中继MTA", role: "邮件中继", protocol: "SMTP", desc: "可能经过多个MTA中继转发", color: "bg-yellow-500" },
  { name: "收件MTA", role: "目的MTA", protocol: "SMTP", desc: "邮件到达目的域的MTA", color: "bg-orange-500" },
  { name: "MDA", role: "邮件投递代理", protocol: "LMTP", desc: "MTA将邮件交给MDA投递到邮箱", color: "bg-purple-500" },
  { name: "收件人UA", role: "用户代理", protocol: "IMAP/POP3", desc: "收件人通过UA收取并阅读邮件", color: "bg-red-500" },
];

export function EmailArchitectureDiagram() {
  const [active, setActive] = useState(-1);
  const [animate, setAnimate] = useState(false);

  const autoPlay = async () => {
    setAnimate(true);
    for (let i = 0; i < STAGES.length; i++) {
      setActive(i);
      await new Promise((r) => setTimeout(r, 800));
    }
    setAnimate(false);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">邮件传输架构 (UA→MTA→MDA)</h3>
      <div className="flex items-center gap-1 mb-4 overflow-x-auto pb-2">
        {STAGES.map((s, i) => (
          <div key={i} className="flex items-center flex-shrink-0">
            <div
              onClick={() => setActive(i)}
              className={`cursor-pointer px-3 py-3 rounded-lg text-center transition-all min-w-[100px] ${
                active === i ? `${s.color} text-white shadow-lg scale-105` : "bg-bg-muted text-text-secondary hover:bg-bg-subtle"
              }`}
            >
              <div className="text-xs font-bold">{s.name}</div>
              <div className="text-[10px] mt-0.5">{s.protocol}</div>
            </div>
            {i < STAGES.length - 1 && (
              <div className={`w-6 h-0.5 flex-shrink-0 ${i < active ? "bg-gray-500" : "bg-gray-300 dark:bg-gray-700"}`} />
            )}
          </div>
        ))}
      </div>
      {active >= 0 && (
        <div className="bg-bg-muted rounded-lg p-4 mb-4">
          <div className="flex items-center gap-2 mb-2">
            <span className={`w-3 h-3 rounded-full ${STAGES[active].color}`} />
            <span className="font-semibold text-text-primary">{STAGES[active].name}</span>
            <span className="text-xs text-text-secondary">({STAGES[active].role})</span>
          </div>
          <p className="text-sm text-text-secondary">{STAGES[active].desc}</p>
          <p className="text-xs text-text-secondary mt-1">协议: {STAGES[active].protocol}</p>
        </div>
      )}
      <div className="flex gap-3 mb-4">
        <button onClick={autoPlay} disabled={animate} className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 hover:bg-blue-600 text-sm">
          {animate ? "播放中..." : "自动演示"}
        </button>
        <button onClick={() => setActive(-1)} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">重置</button>
      </div>
      <div className="text-xs text-text-secondary">
        邮件传输流程: 发件人UA → SMTP → 发件MTA → SMTP → (中继MTA) → SMTP → 收件MTA → MDA → IMAP/POP3 → 收件人UA
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">邮件系统要点</div>
        <div>• MTA使用SMTP协议在服务器之间转发邮件</div>
        <div>• MDA将邮件投递到收件人的邮箱(Maildir/mbox)</div>
        <div>• 用户通过IMAP(在线管理)或POP3(下载)读取邮件</div>
      </div>
    </div>
  );
}

export default EmailArchitectureDiagram;
