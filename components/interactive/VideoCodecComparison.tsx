"use client";
import { useState } from "react";

const codecs = [
  {
    name: "H.264 (AVC)", year: 2003, compression: "中等", bitrate: "4-8 Mbps (1080p)", latency: "低", royalty: "有专利费", efficiency: 60, color: "blue",
    features: ["DCT 变换", "帧内/帧间预测", "CABAC 熵编码", "宏块 16×16", "最广泛硬件兼容"],
    pros: ["最广泛兼容", "硬件解码普及", "编码速度快", "生态成熟"],
    cons: ["压缩效率不如新标准", "高分辨率码率较大"],
  },
  {
    name: "H.265 (HEVC)", year: 2013, compression: "高", bitrate: "2-4 Mbps (1080p)", latency: "中", royalty: "有专利费 (复杂)", efficiency: 80, color: "green",
    features: ["CTU 最大 64×64", "35 种帧内模式", "高级运动补偿", "并行处理优化", "支持 8K 分辨率"],
    pros: ["比 H.264 节省 ~50% 带宽", "支持 8K 分辨率", "适合 4K 流媒体"],
    cons: ["专利费复杂且昂贵", "编码计算量大", "部分旧设备不支持"],
  },
  {
    name: "AV1", year: 2018, compression: "极高", bitrate: "1.5-3 Mbps (1080p)", latency: "高", royalty: "免费开源", efficiency: 95, color: "purple",
    features: ["超级块 128×128", "方向帧内预测", "Film Grain 合成", "环路恢复滤波", "免专利费 (AOMedia)"],
    pros: ["免专利费", "比 HEVC 节省 ~30%", "Google/Apple/Mozilla 支持"],
    cons: ["编码速度慢 (5-10x)", "硬件解码器普及中", "实时编码仍需优化"],
  },
];

export function VideoCodecComparison() {
  const [selected, setSelected] = useState<number | null>(null);
  const [metric, setMetric] = useState<"efficiency" | "year">("efficiency");

  const maxVal = Math.max(...codecs.map((c) => c[metric]));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">视频编码对比</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setMetric("efficiency")}
          className={`px-3 py-1.5 rounded text-xs font-medium ${metric === "efficiency" ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          压缩效率
        </button>
        <button onClick={() => setMetric("year")}
          className={`px-3 py-1.5 rounded text-xs font-medium ${metric === "year" ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted"}`}>
          发布年份
        </button>
      </div>
      <div className="space-y-3 mb-4">
        {codecs.map((c, i) => {
          const pct = (c[metric] / maxVal) * 100;
          const bgMap: Record<string, string> = { blue: "bg-blue-500", green: "bg-green-500", purple: "bg-purple-500" };
          const borderMap: Record<string, string> = { blue: "border-blue-400", green: "border-green-400", purple: "border-purple-400" };
          const textMap: Record<string, string> = { blue: "text-blue-400", green: "text-green-400", purple: "text-purple-400" };
          return (
            <div key={c.name}>
              <button onClick={() => setSelected(selected === i ? null : i)}
                className={`w-full text-left p-3 rounded-lg border-2 transition-all cursor-pointer ${
                  selected === i ? `${borderMap[c.color]} bg-${c.color}-500/10` : "border-border-subtle hover:border-gray-400"
                }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-medium ${textMap[c.color]}`}>{c.name}</span>
                  <div className="flex gap-2 text-xs text-text-muted">
                    <span>{c.year}</span>
                    <span>{c.bitrate}</span>
                    <span className={c.royalty.includes("免费") ? "text-green-400" : "text-red-400"}>{c.royalty}</span>
                  </div>
                </div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div className={`h-full ${bgMap[c.color]} rounded-full transition-all flex items-center justify-end pr-2`}
                    style={{ width: `${Math.max(pct, 15)}%` }}>
                    <span className="text-white text-[10px] font-bold">{c[metric]}{metric === "efficiency" ? "%" : ""}</span>
                  </div>
                </div>
              </button>
              {selected === i && (
                <div className="mt-1 p-4 rounded-lg bg-bg-primary border border-border-subtle">
                  <div className="grid grid-cols-3 gap-2 mb-3 text-xs">
                    <div><span className="text-text-muted">压缩率:</span> <span className="text-text-primary">{c.compression}</span></div>
                    <div><span className="text-text-muted">延迟:</span> <span className="text-text-primary">{c.latency}</span></div>
                    <div><span className="text-text-muted">专利:</span> <span className={c.royalty.includes("免费") ? "text-green-400" : "text-red-400"}>{c.royalty}</span></div>
                  </div>
                  <h5 className="text-text-secondary text-xs font-medium mb-1">技术特性</h5>
                  <div className="flex flex-wrap gap-1 mb-2">
                    {c.features.map((f) => (
                      <span key={f} className="px-2 py-0.5 rounded bg-bg-elevated border border-border-subtle text-text-secondary text-[10px]">{f}</span>
                    ))}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <h5 className="text-green-400 text-[10px] font-medium mb-0.5">优势</h5>
                      {c.pros.map((p) => <p key={p} className="text-text-muted text-[10px]">✓ {p}</p>)}
                    </div>
                    <div>
                      <h5 className="text-red-400 text-[10px] font-medium mb-0.5">劣势</h5>
                      {c.cons.map((p) => <p key={p} className="text-text-muted text-[10px]">✗ {p}</p>)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-secondary text-xs font-medium mb-1">编码效率公式</h4>
        <p className="text-text-muted text-xs">压缩率 = 原始比特率 / 编码比特率。H.265 约比 H.264 节省 50% 带宽，AV1 再节省约 30%。</p>
        <p className="text-text-muted text-xs mt-1">H.264 最广泛兼容 | H.265 节省 50% 带宽 | AV1 开源免费，编码复杂度最高</p>
      </div>
    </div>
  );
}
export default VideoCodecComparison;
