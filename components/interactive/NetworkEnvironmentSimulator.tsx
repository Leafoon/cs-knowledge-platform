"use client";
import { useState } from "react";

const environments = [
  { name: "LAN", label: "局域网", latency: "<1ms", bandwidth: "1-10 Gbps", jitter: "<0.1ms", loss: "<0.01%", color: "bg-green-600" },
  { name: "WiFi", label: "无线局域网", latency: "1-5ms", bandwidth: "50-600 Mbps", jitter: "1-10ms", loss: "0.1-1%", color: "bg-blue-600" },
  { name: "4G LTE", label: "4G 移动网络", latency: "20-50ms", bandwidth: "10-100 Mbps", jitter: "5-20ms", loss: "0.5-2%", color: "bg-yellow-600" },
  { name: "5G", label: "5G 移动网络", latency: "1-10ms", bandwidth: "100-1000 Mbps", jitter: "1-5ms", loss: "<0.5%", color: "bg-purple-600" },
  { name: "Satellite", label: "卫星网络", latency: "500-800ms", bandwidth: "10-100 Mbps", jitter: "50-100ms", loss: "1-5%", color: "bg-red-600" },
];

const appImpacts = [
  { name: "VoIP", icon: "📞", lan: "优秀", wifi: "良好", "4g": "可接受", "5g": "良好", sat: "差" },
  { name: "网页浏览", icon: "🌐", lan: "优秀", wifi: "优秀", "4g": "良好", "5g": "优秀", sat: "可接受" },
  { name: "视频流", icon: "📺", lan: "优秀", wifi: "良好", "4g": "良好", "5g": "优秀", sat: "可接受" },
  { name: "游戏", icon: "🎮", lan: "优秀", wifi: "可接受", "4g": "差", "5g": "良好", sat: "不可用" },
  { name: "大文件传输", icon: "📁", lan: "优秀", wifi: "良好", "4g": "可接受", "5g": "良好", sat: "差" },
];

const ratingColor: Record<string, string> = {
  "优秀": "text-green-400", "良好": "text-blue-400", "可接受": "text-yellow-400", "差": "text-orange-400", "不可用": "text-red-400",
};

export function NetworkEnvironmentSimulator() {
  const [envIdx, setEnvIdx] = useState(0);
  const env = environments[envIdx];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">🌍 网络环境模拟器</h3>
      <p className="text-sm text-text-secondary mb-4">模拟不同网络条件对应用的影响</p>

      <div className="flex gap-2 mb-4 flex-wrap">
        {environments.map((e, i) => (
          <button key={e.name} onClick={() => setEnvIdx(i)}
            className={`px-3 py-2 rounded text-sm font-medium ${envIdx === i ? `${e.color} text-white` : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
            {e.name}
          </button>
        ))}
      </div>

      <div className={`${env.color}/10 border ${env.color.replace("bg-", "border-")} rounded-lg p-4 mb-4`}>
        <div className="font-semibold text-text-primary mb-3">{env.name} — {env.label}</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "延迟", value: env.latency },
            { label: "带宽", value: env.bandwidth },
            { label: "抖动", value: env.jitter },
            { label: "丢包率", value: env.loss },
          ].map(s => (
            <div key={s.label} className="bg-bg-surface rounded-lg p-2 text-center">
              <div className="text-xs text-text-secondary">{s.label}</div>
              <div className="font-mono text-sm font-bold text-text-primary">{s.value}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="text-sm font-medium text-text-secondary mb-2">应用体验评估</div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left p-2 text-text-secondary">应用</th>
              {environments.map(e => (
                <th key={e.name} className={`text-center p-2 ${envIdx === environments.indexOf(e) ? "text-text-primary font-bold" : "text-text-secondary"}`}>{e.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {appImpacts.map(app => (
              <tr key={app.name} className="border-b border-border-subtle">
                <td className="p-2 text-text-primary">{app.icon} {app.name}</td>
                {(["lan", "wifi", "4g", "5g", "sat"] as const).map((key, i) => (
                  <td key={key} className={`p-2 text-center font-medium ${ratingColor[app[key]]} ${envIdx === i ? "bg-bg-surface" : ""}`}>
                    {app[key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
export default NetworkEnvironmentSimulator;
