"use client";
import { useState } from "react";

interface Stream { id: number; name: string; color: string; data: string[] }

const initialStreams: Stream[] = [
  { id: 1, name: "HTML", color: "blue", data: ["<html>...", "</html>"] },
  { id: 2, name: "CSS", color: "green", data: ["body{...}", "h1{...}"] },
  { id: 3, name: "JS", color: "purple", data: ["app.js", "util.js"] },
  { id: 4, name: "IMG", color: "yellow", data: ["logo.png", "bg.jpg"] },
];

export function StreamMultiplexingDemo() {
  const [mode, setMode] = useState<"http2" | "quic">("http2");
  const [step, setStep] = useState(0);
  const maxSteps = 6;

  const next = () => setStep((s) => Math.min(s + 1, maxSteps));
  const prev = () => setStep((s) => Math.max(s - 1, 0));

  const colorMap: Record<string, string> = { blue: "bg-blue-500", green: "bg-green-500", purple: "bg-purple-500", yellow: "bg-yellow-500" };
  const textColorMap: Record<string, string> = { blue: "text-blue-400", green: "text-green-400", purple: "text-purple-400", yellow: "text-yellow-400" };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">流多路复用演示 (Stream Multiplexing)</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => { setMode("http2"); setStep(0); }}
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${mode === "http2" ? "bg-blue-500 text-white" : "border border-border-subtle text-text-muted hover:text-text-primary"}`}>
          HTTP/2
        </button>
        <button onClick={() => { setMode("quic"); setStep(0); }}
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${mode === "quic" ? "bg-purple-500 text-white" : "border border-border-subtle text-text-muted hover:text-text-primary"}`}>
          QUIC
        </button>
      </div>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={prev} disabled={step === 0} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">← 上一步</button>
        <span className="text-text-secondary text-sm">步骤 {step}/{maxSteps}</span>
        <button onClick={next} disabled={step === maxSteps} className="px-3 py-1 rounded border border-border-subtle text-text-muted text-sm disabled:opacity-40">下一步 →</button>
      </div>
      <div className="relative p-4 rounded-lg bg-bg-primary border border-border-subtle min-h-[200px]">
        {mode === "http2" ? (
          <div>
            <p className="text-text-secondary text-sm mb-3">HTTP/2: 所有流共享一条 TCP 连接，一个流阻塞会影响所有流</p>
            <div className="flex items-center gap-4 mb-3">
              <span className="text-text-primary text-sm font-medium w-20">TCP连接</span>
              <div className="flex-1 h-10 bg-gray-200 dark:bg-gray-700 rounded relative overflow-hidden flex">
                {initialStreams.map((s, si) => {
                  const visible = si < step;
                  return visible ? (
                    <div key={s.id} className={`h-full ${colorMap[s.color]} opacity-70 flex items-center justify-center text-white text-xs font-medium`}
                      style={{ width: `${100 / step}%`, animation: "slideIn 0.3s ease" }}>
                      {s.name}
                    </div>
                  ) : null;
                })}
              </div>
            </div>
            {step >= 4 && (
              <div className="p-2 rounded bg-red-500/10 border border-red-400/30 mt-2">
                <span className="text-red-400 text-xs">⚠ 队头阻塞 (Head-of-Line Blocking): 一个 TCP 段丢失 → 所有流等待重传</span>
              </div>
            )}
          </div>
        ) : (
          <div>
            <p className="text-text-secondary text-sm mb-3">QUIC: 每个流有独立的可靠传输，流之间互不影响</p>
            <div className="space-y-2">
              {initialStreams.map((s, si) => {
                const visible = si < step;
                return (
                  <div key={s.id} className="flex items-center gap-4">
                    <span className={`text-sm font-medium w-20 ${textColorMap[s.color]}`}>{s.name}</span>
                    <div className="flex-1 h-6 bg-gray-200 dark:bg-gray-700 rounded relative overflow-hidden">
                      {visible && (
                        <div className={`h-full ${colorMap[s.color]} opacity-70 flex items-center justify-center text-white text-xs`}
                          style={{ width: "60%" }}>
                          {s.data[0]}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            {step >= 4 && (
              <div className="p-2 rounded bg-green-500/10 border border-green-400/30 mt-2">
                <span className="text-green-400 text-xs">✅ 独立流：一个流丢失不影响其他流，消除队头阻塞</span>
              </div>
            )}
          </div>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3 mt-4">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">HTTP/2</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 单 TCP 连接，多流复用</li>
            <li>• 流有优先级和权重</li>
            <li>• TCP 层队头阻塞</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">QUIC</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 基于 UDP，内建 TLS 1.3</li>
            <li>• 独立流，无队头阻塞</li>
            <li>• 连接迁移 (Connection ID)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default StreamMultiplexingDemo;
