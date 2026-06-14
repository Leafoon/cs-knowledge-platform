"use client";
import { useState } from "react";

const frames = [
  { type: "HEADERS", code: 0x1, desc: "打开流并发送HTTP头部", fields: [":method: GET", ":path: /index.html", ":scheme: https", "host: example.com"], color: "blue" },
  { type: "DATA", code: 0x0, desc: "传输请求/响应体数据", fields: ["长度: 1024字节", "END_STREAM标志"], color: "green" },
  { type: "RST_STREAM", code: 0x3, desc: "立即终止某个流", fields: ["错误码: CANCEL (0x8)", "流ID: 3"], color: "red" },
  { type: "SETTINGS", code: 0x4, desc: "协商连接级参数", fields: ["SETTINGS_MAX_CONCURRENT_STREAMS", "SETTINGS_INITIAL_WINDOW_SIZE", "SETTINGS_HEADER_TABLE_SIZE"], color: "purple" },
  { type: "PING", code: 0x6, desc: "测量往返时间和连接活性", fields: ["8字节负载", "ACK标志"], color: "yellow" },
  { type: "GOAWAY", code: 0x7, desc: "优雅关闭连接", fields: ["最后处理的流ID", "错误码: NO_ERROR"], color: "orange" },
  { type: "WINDOW_UPDATE", code: 0x8, desc: "流量控制：更新接收窗口", fields: ["窗口增量: 65535", "流级/连接级"], color: "teal" },
  { type: "PUSH_PROMISE", code: 0x5, desc: "服务器推送资源", fields: ["承诺的流ID", "响应头部字段"], color: "indigo" },
];

export function HTTP2FrameDemo() {
  const [selected, setSelected] = useState(0);
  const frame = frames[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">HTTP/2 帧演示</h3>
      <div className="grid grid-cols-4 gap-1.5 mb-4">
        {frames.map((f, i) => (
          <button key={f.type} onClick={() => setSelected(i)}
            className={`px-2 py-1.5 rounded text-xs font-mono transition-all ${
              i === selected
                ? `bg-${f.color}-500/20 border border-${f.color}-500 text-${f.color}-400`
                : "bg-bg-subtle text-text-secondary hover:bg-bg-muted"
            }`}>
            {f.type}
          </button>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-bg-muted border border-border-subtle">
        <div className="flex items-center gap-2 mb-2">
          <span className={`font-mono text-sm text-${frame.color}-400 font-semibold`}>{frame.type}</span>
          <span className="text-xs text-text-muted font-mono">0x{frame.code.toString(16).padStart(2, "0")}</span>
        </div>
        <p className="text-sm text-text-secondary mb-3">{frame.desc}</p>
        <div className="space-y-1">
          {frame.fields.map((f, i) => (
            <div key={i} className="font-mono text-xs bg-bg-subtle px-2 py-1 rounded text-text-muted">{f}</div>
          ))}
        </div>
      </div>
      <div className="mt-3 grid grid-cols-9 gap-0.5 text-[10px] font-mono text-center">
        {["Length(24)", "Type(8)", "Flags(8)", "R+StreamID(31)", "Frame Payload..."].map((h, i) => (
          <div key={h} className={`col-span-${i < 4 ? 2 : 1} bg-bg-subtle px-1 py-0.5 rounded text-text-muted`}>{h}</div>
        ))}
      </div>
    </div>
  );
}
export default HTTP2FrameDemo;
