"use client";
import { useState } from "react";

const trafficTypes = [
  {
    name: "VoIP",
    zh: "语音通话",
    bandwidth: "64-128 Kbps",
    latency: "< 150ms",
    jitter: "< 30ms",
    loss: "< 1%",
    protocol: "RTP/UDP",
    priority: "高",
    color: "bg-red-500",
  },
  {
    name: "Video Streaming",
    zh: "视频流",
    bandwidth: "1-25 Mbps",
    latency: "< 400ms",
    jitter: "< 50ms",
    loss: "< 2%",
    protocol: "RTP/UDP",
    priority: "高",
    color: "bg-orange-500",
  },
  {
    name: "Web Browsing",
    zh: "网页浏览",
    bandwidth: "1-10 Mbps",
    latency: "< 2s",
    jitter: "可容忍",
    loss: "可恢复",
    protocol: "HTTP/TCP",
    priority: "中",
    color: "bg-blue-500",
  },
  {
    name: "File Transfer",
    zh: "文件传输",
    bandwidth: "尽力而为",
    latency: "不敏感",
    jitter: "不敏感",
    loss: "可重传",
    protocol: "FTP/TCP",
    priority: "低",
    color: "bg-gray-500",
  },
  {
    name: "IoT Sensors",
    zh: "物联网传感器",
    bandwidth: "1-100 Kbps",
    latency: "可变",
    jitter: "可变",
    loss: "可接受",
    protocol: "MQTT/CoAP",
    priority: "低",
    color: "bg-green-500",
  },
];

export function MultimediaTrafficAnalyzer() {
  const [selected, setSelected] = useState(0);
  const [showAll, setShowAll] = useState(false);

  const traffic = trafficTypes[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Multimedia Traffic Analyzer <span className="text-text-secondary text-sm">— 音视频流量需求</span>
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {trafficTypes.map((t, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`px-3 py-1 rounded text-sm text-white ${t.color} ${selected === i ? "ring-2 ring-offset-1 ring-blue-400" : "opacity-60"}`}
          >
            {t.name}
          </button>
        ))}
        <button
          onClick={() => setShowAll(!showAll)}
          className={`px-3 py-1 rounded text-sm ${showAll ? "bg-purple-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
        >
          {showAll ? "单条" : "对比"}
        </button>
      </div>
      {showAll ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="text-left p-2 text-text-secondary">类型</th>
                <th className="text-left p-2 text-text-secondary">带宽</th>
                <th className="text-left p-2 text-text-secondary">延迟</th>
                <th className="text-left p-2 text-text-secondary">抖动</th>
                <th className="text-left p-2 text-text-secondary">丢包</th>
                <th className="text-left p-2 text-text-secondary">协议</th>
              </tr>
            </thead>
            <tbody>
              {trafficTypes.map((t, i) => (
                <tr key={i} className="border-b border-border-subtle">
                  <td className="p-2 font-medium text-text-primary">{t.zh}</td>
                  <td className="p-2 text-text-secondary">{t.bandwidth}</td>
                  <td className="p-2 text-text-secondary">{t.latency}</td>
                  <td className="p-2 text-text-secondary">{t.jitter}</td>
                  <td className="p-2 text-text-secondary">{t.loss}</td>
                  <td className="p-2 text-text-secondary font-mono">{t.protocol}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
          <div className="flex items-center gap-2 mb-3">
            <span className={`w-3 h-3 rounded-full ${traffic.color}`} />
            <span className="font-semibold text-text-primary">{traffic.name} — {traffic.zh}</span>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-white dark:bg-gray-900 p-3 rounded">
              <div className="text-xs text-text-secondary">带宽需求</div>
              <div className="font-bold text-text-primary">{traffic.bandwidth}</div>
            </div>
            <div className="bg-white dark:bg-gray-900 p-3 rounded">
              <div className="text-xs text-text-secondary">延迟要求</div>
              <div className="font-bold text-text-primary">{traffic.latency}</div>
            </div>
            <div className="bg-white dark:bg-gray-900 p-3 rounded">
              <div className="text-xs text-text-secondary">抖动容忍</div>
              <div className="font-bold text-text-primary">{traffic.jitter}</div>
            </div>
            <div className="bg-white dark:bg-gray-900 p-3 rounded">
              <div className="text-xs text-text-secondary">丢包容忍</div>
              <div className="font-bold text-text-primary">{traffic.loss}</div>
            </div>
            <div className="bg-white dark:bg-gray-900 p-3 rounded">
              <div className="text-xs text-text-secondary">传输协议</div>
              <div className="font-bold text-text-primary font-mono">{traffic.protocol}</div>
            </div>
            <div className="bg-white dark:bg-gray-900 p-3 rounded">
              <div className="text-xs text-text-secondary">QoS优先级</div>
              <div className="font-bold text-text-primary">{traffic.priority}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default MultimediaTrafficAnalyzer;
