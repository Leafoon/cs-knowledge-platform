"use client";
import { useState } from "react";

const DSCP_MAP: Record<string, { phb: string; desc: string; color: string }> = {
  "CS0": { phb: "BE", desc: "尽力服务 (Best Effort)", color: "bg-gray-400" },
  "CS1": { phb: "AF1", desc: "确保转发 低", color: "bg-blue-400" },
  "CS2": { phb: "AF2", desc: "确保转发 中", color: "bg-green-400" },
  "CS3": { phb: "AF3", desc: "确保转发 高", color: "bg-yellow-400" },
  "CS4": { phb: "AF4", desc: "确保转发 最高", color: "bg-orange-400" },
  "CS5": { phb: "EF", desc: "加速转发 (Expedited)", color: "bg-red-400" },
  "CS6": { phb: "CS6", desc: "网络控制", color: "bg-purple-400" },
  "CS7": { phb: "CS7", desc: "网络控制", color: "bg-pink-400" },
};

const DSCP_VALUES: Record<string, number> = {
  "CS0": 0, "CS1": 8, "CS2": 16, "CS3": 24, "CS4": 32, "CS5": 40, "CS6": 48, "CS7": 56,
};

export function DiffServClassifier() {
  const [dscp, setDscp] = useState("CS0");
  const [traffic, setTraffic] = useState("");
  const entry = DSCP_MAP[dscp];
  const dscpBits = DSCP_VALUES[dscp];
  const binary = dscpBits.toString(2).padStart(6, "0");

  const trafficTypes = [
    { label: "网页浏览", dscp: "CS0" },
    { label: "视频流", dscp: "CS4" },
    { label: "VoIP语音", dscp: "CS5" },
    { label: "网络管理", dscp: "CS6" },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DiffServ DSCP标记和PHB映射</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        {Object.keys(DSCP_MAP).map((key) => (
          <button
            key={key}
            onClick={() => { setDscp(key); setTraffic(""); }}
            className={`px-3 py-2 rounded text-sm font-mono transition-all ${
              dscp === key
                ? "ring-2 ring-blue-500 bg-blue-100 dark:bg-blue-900 text-text-primary"
                : "bg-bg-subtle text-text-secondary hover:bg-bg-muted"
            }`}
          >
            {key} ({DSCP_VALUES[key]})
          </button>
        ))}
      </div>
      <div className="mb-4 p-4 rounded-lg bg-bg-muted">
        <div className="flex items-center gap-3 mb-2">
          <span className={`w-4 h-4 rounded-full ${entry.color}`} />
          <span className="font-mono text-lg text-text-primary">{entry.phb}</span>
          <span className="text-text-secondary">— {entry.desc}</span>
        </div>
        <div className="font-mono text-sm text-text-secondary">
          DSCP 6位: <span className="text-blue-500">{binary}</span> | IP ToS字节: <span className="text-green-500">{binary}00</span> = {dscpBits << 2}
        </div>
      </div>
      <div className="mb-4">
        <p className="text-sm text-text-secondary mb-2">常见流量分类:</p>
        <div className="flex gap-2 flex-wrap">
          {trafficTypes.map((t) => (
            <button
              key={t.label}
              onClick={() => { setDscp(t.dscp); setTraffic(t.label); }}
              className={`px-3 py-1.5 rounded text-sm transition-all ${
                traffic === t.label
                  ? "bg-blue-500 text-white"
                  : "bg-bg-subtle text-text-secondary hover:bg-bg-muted"
              }`}
            >
              {t.label} → {t.dscp}
            </button>
          ))}
        </div>
      </div>
      <div className="text-xs text-text-secondary">
        DiffServ将IP首部ToS字节的高6位作为DSCP,定义每跳行为(PHB),替代旧式IP优先级分类。
      </div>
    </div>
  );
}

export default DiffServClassifier;
