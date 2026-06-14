"use client";
import { useState } from "react";

const mtaServers = [
  {
    name: "发送方MUA",
    en: "Mail User Agent",
    desc: "用户撰写邮件（如Outlook、Thunderbird）",
    role: "sender",
  },
  {
    name: "发送方MTA",
    en: "Mail Transfer Agent",
    desc: "接收邮件并通过SMTP转发",
    role: "mta",
  },
  {
    name: "DNS MX查询",
    en: "MX Record Lookup",
    desc: "查询目的域名的邮件交换记录",
    role: "dns",
  },
  {
    name: "中继MTA",
    en: "Relay MTA",
    desc: "可选的中间转发服务器",
    role: "mta",
  },
  {
    name: "接收方MTA",
    en: "Destination MTA",
    desc: "最终目的邮件服务器",
    role: "mta",
  },
  {
    name: "接收方MDA",
    en: "Mail Delivery Agent",
    desc: "将邮件投递到用户邮箱",
    role: "mda",
  },
];

const mxRecords = [
  { domain: "gmail.com", mx: "gmail-smtp-in.l.google.com", priority: 10 },
  { domain: "outlook.com", mx: "outlook-com.olc.protection.outlook.com", priority: 10 },
  { domain: "example.com", mx: "mail.example.com", priority: 5 },
];

export function MTARoutingDemo() {
  const [step, setStep] = useState(0);
  const [email, setEmail] = useState("user@example.com");
  const [showMX, setShowMX] = useState(false);

  const domain = email.split("@")[1] || "";

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        MTA Routing <span className="text-text-secondary text-sm">— 邮件路由查找</span>
      </h3>
      <div className="mb-4">
        <label className="text-sm text-text-secondary mb-1 block">收件人地址:</label>
        <input
          type="text"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-primary font-mono text-sm"
        />
      </div>
      <div className="flex gap-1 mb-4 flex-wrap">
        {mtaServers.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`px-2 py-1 rounded text-xs ${step === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            步骤 {i + 1}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">
          {mtaServers[step].name} ({mtaServers[step].en})
        </div>
        <p className="text-sm text-text-secondary">{mtaServers[step].desc}</p>
        {step === 2 && (
          <div className="mt-2">
            <button
              onClick={() => setShowMX(!showMX)}
              className="px-2 py-1 rounded bg-purple-600 text-white text-xs mb-2"
            >
              {showMX ? "隐藏" : "查询"} MX 记录
            </button>
            {showMX && (
              <div className="text-xs font-mono bg-gray-200 dark:bg-gray-900 p-2 rounded">
                <div className="text-text-secondary">$ dig MX {domain}</div>
                {mxRecords
                  .filter((r) => domain.includes(r.domain.replace("example", "")) || domain === r.domain)
                  .map((r, i) => (
                    <div key={i} className="text-text-primary">
                      {r.domain}. IN MX {r.priority} {r.mx}
                    </div>
                  ))}
                {mxRecords.every((r) => !domain.includes(r.domain.replace("example", "")) && domain !== r.domain) && (
                  <div className="text-text-primary">
                    {domain}. IN MX 10 mail.{domain}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="flex items-center gap-1 text-xs text-text-secondary">
        {mtaServers.map((s, i) => (
          <span key={i} className="flex items-center">
            <span className={i <= step ? "text-blue-600 dark:text-blue-400 font-semibold" : ""}>{s.name}</span>
            {i < mtaServers.length - 1 && <span className="mx-1">→</span>}
          </span>
        ))}
      </div>
    </div>
  );
}

export default MTARoutingDemo;
