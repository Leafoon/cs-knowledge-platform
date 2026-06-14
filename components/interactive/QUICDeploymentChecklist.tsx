"use client";
import { useState } from "react";

interface CheckItem {
  id: number;
  category: string;
  item: string;
  detail: string;
  required: boolean;
}

const checklist: CheckItem[] = [
  { id: 1, category: "服务端", item: "启用UDP端口443监听", detail: "QUIC运行在UDP之上，需要开放UDP 443端口", required: true },
  { id: 2, category: "服务端", item: "安装TLS 1.3证书", detail: "QUIC强制使用TLS 1.3，需要有效的证书和私钥", required: true },
  { id: 3, category: "服务端", item: "配置Connection ID长度", detail: "建议8-20字节，支持连接迁移", required: false },
  { id: 4, category: "服务端", item: "启用HTTP/3协议", detail: "配置Alt-Svc响应头标识H3支持", required: true },
  { id: 5, category: "网络", item: "确认防火墙放行UDP", detail: "中间设备（防火墙/NAT）需允许UDP 443流量", required: true },
  { id: 6, category: "网络", item: "配置负载均衡器", detail: "LB需支持QUIC的Connection ID路由", required: false },
  { id: 7, category: "网络", item: "检查MTU设置", detail: "QUIC对PMTUD敏感，建议MTU≥1200", required: false },
  { id: 8, category: "客户端", item: "浏览器兼容性检查", detail: "Chrome 87+、Firefox 114+、Edge 87+原生支持", required: true },
  { id: 9, category: "客户端", item: "配置Alt-Svc回退", detail: "不支持QUIC时回退到HTTP/2 over TCP", required: true },
  { id: 10, category: "监控", item: "部署QUIC流量监控", detail: "监控连接建立成功率、0-RTT使用率等指标", required: false },
  { id: 11, category: "监控", item: "配置日志和故障排查", detail: "记录QUIC握手失败、版本协商等日志", required: false },
  { id: 12, category: "安全", item: "配置速率限制", detail: "防止QUIC反射放大攻击", required: true },
];

export function QUICDeploymentChecklist() {
  const [checked, setChecked] = useState<Set<number>>(new Set());
  const toggle = (id: number) => setChecked((prev) => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const categories = [...new Set(checklist.map((c) => c.category))];
  const requiredDone = checklist.filter((c) => c.required).every((c) => checked.has(c.id));
  const progress = (checked.size / checklist.length) * 100;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">QUIC部署清单</h3>
      <div className="flex items-center gap-3 mb-4">
        <div className="flex-1 h-2 bg-bg-tertiary rounded-full overflow-hidden">
          <div className={`h-full rounded-full transition-all ${requiredDone ? "bg-emerald-500" : "bg-sky-500"}`} style={{ width: `${progress}%` }} />
        </div>
        <span className="text-xs font-mono text-text-secondary">{checked.size}/{checklist.length}</span>
        {requiredDone && <span className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">必需项已完成</span>}
      </div>
      {categories.map((cat) => (
        <div key={cat} className="mb-4">
          <div className="text-xs font-medium text-text-primary mb-2 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-sky-500" />{cat}
          </div>
          <div className="space-y-1.5">
            {checklist.filter((c) => c.category === cat).map((item) => (
              <div key={item.id} onClick={() => toggle(item.id)}
                className={`flex items-start gap-3 px-3 py-2.5 rounded-lg border cursor-pointer transition-all ${checked.has(item.id) ? "bg-emerald-500/10 border-emerald-500/30" : "bg-bg-tertiary border-border-subtle hover:border-sky-400/30"}`}>
                <span className={`mt-0.5 w-4 h-4 rounded border flex items-center justify-center text-[10px] shrink-0 ${checked.has(item.id) ? "bg-emerald-500 border-emerald-500 text-white" : "border-border-subtle"}`}>
                  {checked.has(item.id) ? "✓" : ""}
                </span>
                <div className="flex-1 min-w-0">
                  <div className={`text-xs font-medium ${checked.has(item.id) ? "text-emerald-700 dark:text-emerald-300 line-through" : "text-text-primary"}`}>
                    {item.item}
                    {item.required && <span className="ml-1 text-red-500">*</span>}
                  </div>
                  <div className="text-[10px] text-text-tertiary mt-0.5">{item.detail}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
export default QUICDeploymentChecklist;
