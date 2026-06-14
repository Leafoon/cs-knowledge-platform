"use client";
import { useState } from "react";

const slices = [
  {
    name: "eMBB",
    label: "增强移动宽带",
    desc: "大带宽、高吞吐量场景",
    bandwidth: "10 Gbps",
    latency: "10 ms",
    reliability: "99.9%",
    useCases: ["4K/8K 视频流", "AR/VR 应用", "高速下载"],
    color: "bg-blue-500",
  },
  {
    name: "URLLC",
    label: "超可靠低延迟通信",
    desc: "超低延迟、超高可靠性场景",
    bandwidth: "100 Mbps",
    latency: "0.5 ms",
    reliability: "99.999%",
    useCases: ["远程手术", "自动驾驶", "工业控制"],
    color: "bg-red-500",
  },
  {
    name: "mMTC",
    label: "大规模机器通信",
    desc: "海量设备连接场景",
    bandwidth: "1 Mbps",
    latency: "100 ms",
    reliability: "99%",
    useCases: ["智能城市", "IoT 传感器", "智能农业"],
    color: "bg-green-500",
  },
];

export function GSliceExplorer() {
  const [active, setActive] = useState(0);
  const slice = slices[active];
  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">5G 网络切片探索器</h3>
      <div className="flex gap-2 mb-4">
        {slices.map((s, i) => (
          <button
            key={s.name}
            onClick={() => setActive(i)}
            className={`px-4 py-2 rounded font-mono text-sm transition-colors ${
              active === i
                ? `${s.color} text-white`
                : "bg-gray-200 dark:bg-gray-700 text-text-secondary hover:bg-gray-300 dark:hover:bg-gray-600"
            }`}
          >
            {s.name}
          </button>
        ))}
      </div>
      <div className="mb-4 p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-border-subtle">
        <p className="text-lg font-semibold text-text-primary">{slice.name} — {slice.label}</p>
        <p className="text-sm text-text-secondary mt-1">{slice.desc}</p>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">带宽</div>
          <div className="text-lg font-bold text-text-primary">{slice.bandwidth}</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">延迟</div>
          <div className="text-lg font-bold text-text-primary">{slice.latency}</div>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">可靠性</div>
          <div className="text-lg font-bold text-text-primary">{slice.reliability}</div>
        </div>
      </div>
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">典型应用场景：</p>
        <div className="flex flex-wrap gap-2">
          {slice.useCases.map((uc) => (
            <span key={uc} className="px-2 py-1 text-xs rounded-full bg-gray-200 dark:bg-gray-700 text-text-secondary">
              {uc}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
export default GSliceExplorer;
