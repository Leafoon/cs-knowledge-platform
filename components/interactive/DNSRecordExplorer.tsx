"use client";
import { useState } from "react";

const recordTypes = [
  { type: "A", name: "地址记录", desc: "域名→IPv4地址", example: "example.com → 93.184.216.34", ttl: 300 },
  { type: "AAAA", name: "IPv6地址记录", desc: "域名→IPv6地址", example: "example.com → 2606:2800:220:1:248:1893:25c8:1946", ttl: 300 },
  { type: "CNAME", name: "规范名称", desc: "域名别名指向", example: "www.example.com → example.com", ttl: 3600 },
  { type: "MX", name: "邮件交换", desc: "指定邮件服务器", example: "example.com → 10 mail.example.com", ttl: 3600 },
  { type: "NS", name: "名称服务器", desc: "指定权威DNS服务器", example: "example.com → ns1.example.com", ttl: 86400 },
  { type: "TXT", name: "文本记录", desc: "存储任意文本(常用于SPF/DKIM)", example: "example.com → \"v=spf1 include:_spf.google.com ~all\"", ttl: 300 },
  { type: "SOA", name: "起始授权", desc: "区域的管理信息", example: "ns1.example.com admin.example.com 2024010101 3600 900 604800 86400", ttl: 86400 },
  { type: "PTR", name: "指针记录", desc: "IP→域名(反向DNS)", example: "34.216.184.93.in-addr.arpa → example.com", ttl: 3600 },
];

export function DNSRecordExplorer() {
  const [selected, setSelected] = useState(0);
  const [queryInput, setQueryInput] = useState("example.com");
  const [showQuery, setShowQuery] = useState(false);
  const record = recordTypes[selected];

  const simulatedQuery = `;; QUESTION SECTION:\n;${queryInput}.            IN      ${record.type}\n\n;; ANSWER SECTION:\n${queryInput}.      ${record.ttl}  IN      ${record.type}      ${record.example.split("→")[1]?.trim() || record.example.split("→")[0].trim()}`;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNS 记录类型探索器</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {recordTypes.map((r, i) => (
          <button key={r.type} onClick={() => setSelected(i)}
            className={`px-3 py-1.5 rounded text-sm font-mono transition-colors ${selected === i ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700"}`}>
            {r.type}
          </button>
        ))}
      </div>
      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 mb-4">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl font-mono font-bold text-blue-600">{record.type}</span>
          <div>
            <div className="text-text-primary font-medium">{record.name}</div>
            <div className="text-sm text-text-secondary">{record.desc}</div>
          </div>
        </div>
        <div className="mb-3">
          <div className="text-xs text-text-secondary mb-1">示例</div>
          <code className="block bg-gray-100 dark:bg-gray-800 rounded px-3 py-2 text-sm font-mono text-text-primary break-all">
            {record.example}
          </code>
        </div>
        <div className="flex gap-4 text-xs text-text-secondary">
          <span>默认TTL: <span className="font-mono text-text-primary">{record.ttl}s</span></span>
          <span>类别: <span className="font-mono text-text-primary">IN (Internet)</span></span>
        </div>
      </div>
      <div className="mb-4">
        <button onClick={() => setShowQuery(!showQuery)}
          className="px-3 py-1.5 rounded text-sm bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700">
          {showQuery ? "隐藏" : "模拟"} dig 查询
        </button>
        {showQuery && (
          <div className="mt-2">
            <div className="flex gap-2 mb-2">
              <input value={queryInput} onChange={(e) => setQueryInput(e.target.value)}
                className="flex-1 px-2 py-1 rounded border border-border-subtle bg-white dark:bg-gray-800 font-mono text-xs text-text-primary" />
            </div>
            <pre className="p-3 rounded bg-gray-100 dark:bg-gray-800 text-xs font-mono text-text-primary overflow-x-auto whitespace-pre-wrap">
              {simulatedQuery}
            </pre>
          </div>
        )}
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800">
        <p className="text-xs font-medium text-text-primary mb-2">记录类型分类</p>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="p-2 rounded bg-blue-50 dark:bg-blue-900/20">
            <div className="font-medium text-blue-700 dark:text-blue-300">地址记录</div>
            <div className="text-text-secondary">A, AAAA</div>
          </div>
          <div className="p-2 rounded bg-green-50 dark:bg-green-900/20">
            <div className="font-medium text-green-700 dark:text-green-300">别名/权威</div>
            <div className="text-text-secondary">CNAME, NS, SOA</div>
          </div>
          <div className="p-2 rounded bg-purple-50 dark:bg-purple-900/20">
            <div className="font-medium text-purple-700 dark:text-purple-300">服务/其他</div>
            <div className="text-text-secondary">MX, TXT, PTR</div>
          </div>
        </div>
      </div>
    </div>
  );
}
export default DNSRecordExplorer;
