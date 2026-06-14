"use client";
import { useState } from "react";

export function DoHDemo() {
  const [showComparison, setShowComparison] = useState(false);
  const [encryptTraditional, setEncryptTraditional] = useState(false);

  const traditionalSteps = [
    { step: "用户输入域名", visible: true },
    { step: "构造DNS查询报文", visible: true },
    { step: "通过UDP 53端口发送", visible: !encryptTraditional },
    { step: "ISP路由器转发", visible: !encryptTraditional },
    { step: "DNS服务器解析", visible: !encryptTraditional },
    { step: "返回IP地址", visible: !encryptTraditional },
  ];

  const dohSteps = [
    { step: "用户输入域名", visible: true },
    { step: "构造DNS查询报文", visible: true },
    { step: "加密为HTTPS请求", visible: true },
    { step: "通过TCP 443端口发送", visible: true },
    { step: "与普通HTTPS流量混合", visible: true },
    { step: "DoH服务器解密并解析", visible: true },
    { step: "加密返回IP地址", visible: true },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNS over HTTPS (DoH) 隐私对比</h3>
      <div className="flex gap-3 mb-4">
        <button onClick={() => setShowComparison(!showComparison)} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm">
          {showComparison ? "隐藏对比" : "显示对比"}
        </button>
        <button onClick={() => setEncryptTraditional(!encryptTraditional)} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">
          传统DNS: {encryptTraditional ? "已加密(DNSSEC)" : "明文"}
        </button>
      </div>
      <div className={`grid ${showComparison ? "grid-cols-2" : "grid-cols-1"} gap-4`}>
        <div className="bg-bg-muted rounded-lg p-4">
          <h4 className="font-semibold text-text-primary mb-3">传统DNS (UDP 53)</h4>
          {traditionalSteps.map((s, i) => (
            <div key={i} className={`flex items-center gap-2 mb-2 p-2 rounded text-sm ${s.visible ? "bg-yellow-100 dark:bg-yellow-900/30 text-text-primary" : "bg-green-100 dark:bg-green-900/30 text-text-secondary"}`}>
              <span className="w-6 h-6 flex items-center justify-center rounded-full bg-bg-subtle text-xs">{i + 1}</span>
              <span>{s.step}</span>
              {!s.visible && <span className="ml-auto text-xs text-green-500">🔒</span>}
            </div>
          ))}
        </div>
        {showComparison && (
          <div className="bg-bg-muted rounded-lg p-4">
            <h4 className="font-semibold text-text-primary mb-3">DoH (HTTPS 443)</h4>
            {dohSteps.map((s, i) => (
              <div key={i} className="flex items-center gap-2 mb-2 p-2 rounded text-sm bg-green-100 dark:bg-green-900/30 text-text-primary">
                <span className="w-6 h-6 flex items-center justify-center rounded-full bg-bg-subtle text-xs">{i + 1}</span>
                <span>{s.step}</span>
                <span className="ml-auto text-xs text-green-500">🔒</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2 text-xs text-text-secondary">
        <div className="p-2 bg-bg-muted rounded"><strong>加密</strong>: DoH全程TLS加密,传统DNS明文</div>
        <div className="p-2 bg-bg-muted rounded"><strong>端口</strong>: DoH用443,传统DNS用53</div>
        <div className="p-2 bg-bg-muted rounded"><strong>隐蔽性</strong>: DoH与HTTPS流量混合不可区分</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">DoH vs DoT对比</div>
        <div>• DoH: 端口443，与HTTPS流量混合，难以封锁</div>
        <div>• DoT (DNS over TLS): 端口853，独立TLS连接，易被识别封锁</div>
        <div>• 隐私权衡: DoH更强隐私，但可能绕过企业/家长控制</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">主要DoH提供商</div>
        <div>• Cloudflare: https://1.1.1.1/dns-query</div>
        <div>• Google: https://dns.google/dns-query</div>
        <div>• Quad9: https://dns.quad9.net/dns-query</div>
      </div>
    </div>
  );
}

export default DoHDemo;
