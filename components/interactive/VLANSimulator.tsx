"use client";
import { useState } from "react";

interface Host { name: string; vlan: number; port: string; mac: string; switch_: string; }

const hosts: Host[] = [
  { name: "PC-A", vlan: 10, port: "Fa0/1", mac: "AA:AA:AA:00:00:01", switch_: "SW1" },
  { name: "PC-B", vlan: 10, port: "Fa0/2", mac: "AA:AA:AA:00:00:02", switch_: "SW1" },
  { name: "PC-C", vlan: 20, port: "Fa0/3", mac: "AA:AA:AA:00:00:03", switch_: "SW1" },
  { name: "PC-D", vlan: 20, port: "Fa0/1", mac: "AA:AA:AA:00:00:04", switch_: "SW2" },
  { name: "PC-E", vlan: 10, port: "Fa0/2", mac: "AA:AA:AA:00:00:05", switch_: "SW2" },
];

export function VLANSimulator() {
  const [src, setSrc] = useState(0);
  const [dst, setDst] = useState(4);
  const [result, setResult] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const sendFrame = () => {
    const s = hosts[src];
    const d = hosts[dst];
    setLog([]);

    if (s.vlan !== d.vlan) {
      setResult(`❌ 发送失败: ${s.name} (VLAN ${s.vlan}) 和 ${d.name} (VLAN ${d.vlan}) 不在同一 VLAN，广播域隔离`);
      setLog([
        `${s.name} 发送广播帧 (VLAN ${s.vlan})`,
        `交换机 ${s.switch_} 在 VLAN ${s.vlan} 内广播`,
        `${d.name} 属于 VLAN ${d.vlan}，不在广播域内`,
        `帧被丢弃 — VLAN 隔离生效`,
      ]);
    } else if (s.switch_ === d.switch_) {
      setResult(`✓ 发送成功: ${s.name} → ${d.name} (同交换机 VLAN ${s.vlan})`);
      setLog([
        `${s.name} 发送帧到 ${d.name} (VLAN ${s.vlan})`,
        `交换机 ${s.switch_} 查找 MAC 表`,
        `从端口 ${d.port} 转发 (同一 VLAN 内)`,
        `${d.name} 收到帧`,
      ]);
    } else {
      setResult(`✓ 发送成功: ${s.name} → ${d.name} (跨交换机 VLAN ${s.vlan}，通过 Trunk)`);
      setLog([
        `${s.name} 发送帧到 ${d.name} (VLAN ${s.vlan})`,
        `交换机 ${s.switch_} 查找 MAC 表，目的在远端`,
        `通过 Trunk 端口发送 (标记 VLAN ${s.vlan})`,
        `交换机 ${d.switch_} 收到并剥离 VLAN 标签`,
        `从端口 ${d.port} 转发到 ${d.name}`,
      ]);
    }
  };

  const vlanColor = (v: number) => v === 10 ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20" : "border-green-500 bg-green-50 dark:bg-green-900/20";

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">VLAN 隔离模拟器</h3>
      <div className="text-xs text-text-secondary mb-3">点击选择发送方和接收方，测试 VLAN 隔离效果</div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-xs text-text-secondary mb-2">交换机 SW1</div>
          <div className="space-y-2">
            {hosts.filter((h) => h.switch_ === "SW1").map((h, i) => {
              const idx = hosts.indexOf(h);
              return (
                <button key={i} onClick={() => setSrc(idx)} className={`w-full text-left p-2 rounded-lg border-2 transition-colors ${vlanColor(h.vlan)} ${src === idx ? "ring-2 ring-blue-500" : dst === idx ? "ring-2 ring-red-500" : ""}`}>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-text-primary">{h.name}</span>
                    <span className="text-xs text-text-secondary">VLAN {h.vlan}</span>
                  </div>
                  <div className="text-xs text-text-secondary">Port {h.port} | {h.mac.slice(-5)}</div>
                </button>
              );
            })}
          </div>
        </div>
        <div>
          <div className="text-xs text-text-secondary mb-2">交换机 SW2</div>
          <div className="space-y-2">
            {hosts.filter((h) => h.switch_ === "SW2").map((h, i) => {
              const idx = hosts.indexOf(h);
              return (
                <button key={i} onClick={() => setDst(idx)} className={`w-full text-left p-2 rounded-lg border-2 transition-colors ${vlanColor(h.vlan)} ${src === idx ? "ring-2 ring-blue-500" : dst === idx ? "ring-2 ring-red-500" : ""}`}>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-text-primary">{h.name}</span>
                    <span className="text-xs text-text-secondary">VLAN {h.vlan}</span>
                  </div>
                  <div className="text-xs text-text-secondary">Port {h.port} | {h.mac.slice(-5)}</div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
      <div className="flex items-center justify-center gap-4 mb-4 text-sm p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <span className="text-blue-500 font-medium">发送: {hosts[src].name} (VLAN {hosts[src].vlan})</span>
        <span className="text-text-secondary">→</span>
        <span className="text-red-500 font-medium">接收: {hosts[dst].name} (VLAN {hosts[dst].vlan})</span>
      </div>
      <button onClick={sendFrame} className="w-full py-2 rounded bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors mb-3">发送帧</button>
      {result && (
        <div className={`p-3 rounded text-sm mb-3 ${result.startsWith("✓") ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300" : "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300"}`}>
          {result}
        </div>
      )}
      {log.length > 0 && (
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-xs font-mono space-y-1">
          {log.map((l, i) => <div key={i} className="text-text-secondary">{l}</div>)}
        </div>
      )}
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <div className="p-2 rounded bg-blue-50 dark:bg-blue-900/20 border border-border-subtle text-center"><span className="text-blue-500 font-medium">VLAN 10</span>: PC-A, PC-B, PC-E</div>
        <div className="p-2 rounded bg-green-50 dark:bg-green-900/20 border border-border-subtle text-center"><span className="text-green-500 font-medium">VLAN 20</span>: PC-C, PC-D</div>
      </div>
    </div>
  );
}
export default VLANSimulator;
