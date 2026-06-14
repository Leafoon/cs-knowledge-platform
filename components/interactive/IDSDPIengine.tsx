"use client";
import { useState } from "react";

interface Signature {
  id: number;
  name: string;
  pattern: string;
  severity: "low" | "medium" | "high" | "critical";
  desc: string;
}

const SIGNATURES: Signature[] = [
  { id: 1, name: "SQL注入", pattern: "' OR 1=1 --", severity: "critical", desc: "SQL注入攻击尝试" },
  { id: 2, name: "XSS攻击", pattern: "<script>alert(1)</script>", severity: "high", desc: "跨站脚本注入" },
  { id: 3, name: "目录遍历", pattern: "../../etc/passwd", severity: "high", desc: "路径遍历攻击" },
  { id: 4, name: "端口扫描", pattern: "SYN flood", severity: "medium", desc: "TCP SYN扫描" },
  { id: 5, name: "异常大包", pattern: "payload > 1400B", severity: "low", desc: "可疑大数据包" },
];

export function IDSDPIengine() {
  const [traffic, setTraffic] = useState("");
  const [alerts, setAlerts] = useState<Signature[]>([]);
  const [inspecting, setInspecting] = useState(false);

  const inspect = () => {
    setInspecting(true);
    const found: Signature[] = [];
    for (const sig of SIGNATURES) {
      if (traffic.toLowerCase().includes(sig.pattern.toLowerCase().slice(0, 6))) {
        found.push(sig);
      }
    }
    setTimeout(() => {
      setAlerts(found);
      setInspecting(false);
    }, 500);
  };

  const severityColor = {
    low: "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300",
    medium: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300",
    high: "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300",
    critical: "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300",
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">IDS深度包检测(DPI)引擎</h3>
      <div className="mb-4">
        <label className="text-sm text-text-secondary mb-1 block">模拟网络流量/请求:</label>
        <textarea
          value={traffic}
          onChange={(e) => setTraffic(e.target.value)}
          className="w-full px-3 py-2 rounded border border-border-subtle bg-bg-subtle text-text-primary font-mono text-sm h-20"
          placeholder="输入HTTP请求内容,如: GET /page?id=1' OR 1=1 --"
        />
      </div>
      <div className="flex gap-3 mb-4">
        <button onClick={inspect} disabled={inspecting}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 hover:bg-blue-600 text-sm">
          {inspecting ? "检测中..." : "DPI深度检测"}
        </button>
        <button onClick={() => { setAlerts([]); setTraffic(""); }}
          className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">清除</button>
      </div>
      {alerts.length > 0 && (
        <div className="space-y-2 mb-4">
          <div className="text-sm font-semibold text-red-500">⚠ 检测到 {alerts.length} 条告警:</div>
          {alerts.map((a) => (
            <div key={a.id} className={`p-3 rounded-lg ${severityColor[a.severity]}`}>
              <div className="flex items-center justify-between">
                <span className="font-bold">{a.name}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-white/20">{a.severity.toUpperCase()}</span>
              </div>
              <div className="text-xs mt-1">模式: {a.pattern}</div>
              <div className="text-xs mt-1">{a.desc}</div>
            </div>
          ))}
        </div>
      )}
      {alerts.length === 0 && traffic && !inspecting && (
        <div className="p-3 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-lg text-sm">
          ✓ 未检测到已知攻击模式
        </div>
      )}
      <div className="mt-4">
        <div className="text-sm text-text-secondary mb-2">已知特征库:</div>
        <div className="grid grid-cols-5 gap-1 text-xs">
          {SIGNATURES.map((s) => (
            <div key={s.id} className={`p-1.5 rounded ${severityColor[s.severity]}`}>
              {s.name}
            </div>
          ))}
        </div>
      </div>
      <div className="text-xs text-text-secondary mt-3">
        IDS(入侵检测系统)通过DPI深度包检测,匹配已知攻击特征(Signature-based)或检测异常行为(Anomaly-based)。
      </div>
    </div>
  );
}

export default IDSDPIengine;
