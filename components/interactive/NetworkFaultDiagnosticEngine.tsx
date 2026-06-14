"use client";
import { useState } from "react";

const layers = [
  {
    name: "物理层", en: "Physical", icon: "🔌",
    faults: [
      { issue: "线缆断裂/松动", cmd: "ethtool eth0", symptom: "Link 状态 DOWN" },
      { issue: "光模块故障", cmd: "ethtool -m eth0", symptom: "光功率异常" },
      { issue: "端口协商失败", cmd: "ethtool eth0 | grep Speed", symptom: "速率/双工不匹配" },
    ],
  },
  {
    name: "数据链路层", en: "Data Link", icon: "🔗",
    faults: [
      { issue: "MAC 地址冲突", cmd: "arp -a", symptom: "地址翻动" },
      { issue: "VLAN 配置错误", cmd: "show vlan brief", symptom: "VLAN 不匹配" },
      { issue: "STP 环路", cmd: "show spanning-tree", symptom: "广播风暴" },
    ],
  },
  {
    name: "网络层", en: "Network", icon: "🗺️",
    faults: [
      { issue: "路由缺失", cmd: "ip route / traceroute", symptom: "目的不可达" },
      { issue: "ACL 拦截", cmd: "show access-list", symptom: "特定流量被丢弃" },
      { issue: "MTU 不匹配", cmd: "ping -f -l 1472", symptom: "分片/丢包" },
    ],
  },
  {
    name: "传输层", en: "Transport", icon: "🚚",
    faults: [
      { issue: "端口未监听", cmd: "ss -tlnp / netstat", symptom: "连接被拒绝" },
      { issue: "防火墙拦截", cmd: "iptables -L", symptom: "SYN 超时" },
      { issue: "连接数耗尽", cmd: "ss -s", symptom: "新连接失败" },
    ],
  },
  {
    name: "应用层", en: "Application", icon: "💻",
    faults: [
      { issue: "DNS 解析失败", cmd: "dig / nslookup", symptom: "域名不可达" },
      { issue: "证书过期", cmd: "openssl s_client", symptom: "TLS 握手失败" },
      { issue: "服务进程崩溃", cmd: "systemctl status", symptom: "服务不可用" },
    ],
  },
];

export function NetworkFaultDiagnosticEngine() {
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [selectedFault, setSelectedFault] = useState<number | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">🔧 网络故障诊断引擎</h3>
      <p className="text-sm text-text-secondary mb-4">交互式分层排查网络问题</p>

      <div className="flex gap-2 mb-4 flex-wrap">
        {layers.map((l, i) => (
          <button key={l.en} onClick={() => { setSelectedLayer(selectedLayer === i ? null : i); setSelectedFault(null); }}
            className={`px-3 py-2 rounded text-sm flex items-center gap-1.5 ${selectedLayer === i ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
            <span>{l.icon}</span> {l.name}
          </button>
        ))}
      </div>

      {selectedLayer !== null && (
        <div className="mb-4">
          <div className="text-sm font-medium text-text-secondary mb-2">{layers[selectedLayer].icon} {layers[selectedLayer].name} ({layers[selectedLayer].en}) 常见故障</div>
          <div className="space-y-1.5">
            {layers[selectedLayer].faults.map((f, i) => (
              <button key={i} onClick={() => setSelectedFault(selectedFault === i ? null : i)}
                className={`w-full flex items-center justify-between p-3 rounded-lg text-sm text-left transition-all ${selectedFault === i ? "bg-red-900/30 border border-red-600" : "bg-bg-surface border border-border-subtle hover:border-red-400"}`}>
                <span className="text-text-primary">{f.issue}</span>
                <span className="text-xs text-text-secondary">{f.symptom}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {selectedLayer !== null && selectedFault !== null && (
        <div className="bg-bg-surface rounded-lg p-4 border border-border-subtle">
          <div className="font-medium text-text-primary mb-2">{layers[selectedLayer].faults[selectedFault].issue}</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <div className="text-xs text-text-secondary mb-1">症状</div>
              <div className="text-sm text-red-400">{layers[selectedLayer].faults[selectedFault].symptom}</div>
            </div>
            <div>
              <div className="text-xs text-text-secondary mb-1">诊断命令</div>
              <code className="text-sm font-mono text-green-400 bg-bg-elevated px-2 py-1 rounded">{layers[selectedLayer].faults[selectedFault].cmd}</code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
export default NetworkFaultDiagnosticEngine;
