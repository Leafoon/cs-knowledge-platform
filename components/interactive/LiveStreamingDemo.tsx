"use client";
import { useState } from "react";

const codecs = [
  { name: "H.264", latency: 80, bitrate: "2-8 Mbps", desc: "应用最广泛，兼容性最好" },
  { name: "H.265", latency: 120, bitrate: "1-4 Mbps", desc: "压缩率更高，编码复杂度更高" },
  { name: "VP9", latency: 100, bitrate: "1.5-6 Mbps", desc: "Google 开源方案，YouTube 常用" },
  { name: "AV1", latency: 150, bitrate: "0.8-3 Mbps", desc: "下一代开源编码，压缩率最优" },
];

const steps = [
  { id: 0, label: "采集", en: "Capture", icon: "📷", delay: 5 },
  { id: 1, label: "编码", en: "Encode", icon: "⚙️", delay: 0 },
  { id: 2, label: "封装", en: "Packetize", icon: "📦", delay: 3 },
  { id: 3, label: "CDN分发", en: "CDN Relay", icon: "🌐", delay: 15 },
  { id: 4, label: "解码", en: "Decode", icon: "🔓", delay: 20 },
  { id: 5, label: "播放", en: "Playback", icon: "▶️", delay: 10 },
];

export function LiveStreamingDemo() {
  const [codec, setCodec] = useState(0);
  const [activeStep, setActiveStep] = useState(-1);
  const [playing, setPlaying] = useState(false);

  const totalLatency = steps.reduce((s, st) => s + st.delay, 0) + codecs[codec].latency;

  const startStream = () => {
    setPlaying(true);
    setActiveStep(0);
    let i = 0;
    const iv = setInterval(() => {
      i++;
      if (i < steps.length) setActiveStep(i);
      else { clearInterval(iv); setPlaying(false); }
    }, 600);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">📺 直播流媒体演示</h3>
      <p className="text-sm text-text-secondary mb-4">展示实时视频流的编码→传输→播放流程</p>

      <div className="mb-4">
        <label className="text-sm font-medium text-text-secondary">选择编码器：</label>
        <div className="flex gap-2 mt-2 flex-wrap">
          {codecs.map((c, i) => (
            <button key={c.name} onClick={() => setCodec(i)}
              className={`px-3 py-1.5 rounded text-sm font-mono ${codec === i ? "bg-blue-600 text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-blue-400"}`}>
              {c.name}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-bg-surface rounded-lg p-3 mb-4 text-sm text-text-secondary">
        <span className="font-medium text-text-primary">{codecs[codec].name}</span> — {codecs[codec].desc}<br />
        码率: {codecs[codec].bitrate} | 编码延迟: {codecs[codec].latency}ms
      </div>

      <div className="flex items-center gap-1 overflow-x-auto py-4 mb-4">
        {steps.map((s, i) => (
          <div key={s.id} className="flex items-center">
            <div onClick={() => setActiveStep(i)}
              className={`flex flex-col items-center cursor-pointer px-3 py-2 rounded-lg min-w-[70px] transition-all ${activeStep === i ? "bg-blue-600/20 border border-blue-500 scale-110" : "border border-transparent"}`}>
              <span className="text-2xl">{s.icon}</span>
              <span className="text-xs font-medium text-text-primary mt-1">{s.label}</span>
              <span className="text-[10px] text-text-secondary">{s.en}</span>
              <span className="text-[10px] text-text-secondary">{s.delay}ms</span>
            </div>
            {i < steps.length - 1 && (
              <div className={`w-8 h-0.5 mx-1 ${activeStep > i ? "bg-blue-500" : "bg-border-subtle"}`} />
            )}
          </div>
        ))}
      </div>

      {activeStep >= 0 && (
        <div className="bg-bg-surface rounded-lg p-3 mb-4 border border-border-subtle">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg">{steps[activeStep].icon}</span>
            <span className="font-semibold text-text-primary">{steps[activeStep].label} ({steps[activeStep].en})</span>
          </div>
          <p className="text-sm text-text-secondary">
            {activeStep === 0 && "摄像头/屏幕采集原始帧数据，分辨率通常为 1080p@30fps 或 720p@60fps"}
            {activeStep === 1 && `使用 ${codecs[codec].name} 编码器压缩原始帧，降低带宽需求，引入 ${codecs[codec].latency}ms 编码延迟`}
            {activeStep === 2 && "将编码帧封装为 FLV/RTMP 或 MPEG-TS 分段，添加时间戳和元数据"}
            {activeStep === 3 && "通过 CDN 边缘节点进行全球分发，RTMP/SRT 协议传输，增加 15-50ms 延迟"}
            {activeStep === 4 && "客户端拉流后进行解码还原为原始帧，使用硬件加速降低延迟"}
            {activeStep === 5 && "视频帧经过渲染管线输出到显示器，加入缓冲区平滑抖动"}
          </p>
        </div>
      )}

      <div className="flex items-center gap-4">
        <button onClick={startStream} disabled={playing}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
          {playing ? "传输中..." : "开始直播"}
        </button>
        <span className="text-sm text-text-secondary">端到端延迟: <span className="font-mono text-text-primary">{totalLatency}ms</span></span>
      </div>
    </div>
  );
}
export default LiveStreamingDemo;
