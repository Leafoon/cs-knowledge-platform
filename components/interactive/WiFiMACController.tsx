"use client";
import { useState } from "react";

export function WiFiMACController() {
  const [step, setStep] = useState(0);
  const [backoff, setBackoff] = useState(5);
  const [collision, setCollision] = useState(false);
  const [cwMin, setCwMin] = useState(15);
  const [log, setLog] = useState<string[]>([]);
  const [channelBusy, setChannelBusy] = useState(false);

  const rtsctsSteps = [
    { label: "信道空闲等待 DIFS", desc: "源站检测信道，等待 DIFS (DCF Interframe Space) 时间", time: "DIFS" },
    { label: "退避计数器", desc: `退避计数器从 ${backoff} 开始递减，信道空闲时每时隙减1`, time: "Backoff" },
    { label: "发送 RTS", desc: "源站发送 RTS (Request to Send)，包含 duration 字段", time: "SIFS" },
    { label: "接收 CTS", desc: "目的站等待 SIFS 后回复 CTS (Clear to Send)", time: "SIFS" },
    { label: "发送数据帧", desc: "源站收到 CTS 后等待 SIFS，发送数据帧", time: "SIFS" },
    { label: "接收 ACK", desc: "目的站正确接收后等待 SIFS，回复 ACK", time: "ACK" },
  ];

  const next = () => {
    if (step === 1 && collision) {
      setCwMin((cw) => Math.min(cw * 2 + 1, 1023));
      setBackoff(Math.floor(Math.random() * (cwMin * 2 + 1)));
      setLog((l) => [...l, `碰撞！竞争窗口增大到 ${cwMin * 2 + 1}`]);
      setCollision(false);
      return;
    }
    if (step < rtsctsSteps.length - 1) {
      setStep((s) => s + 1);
      setLog((l) => [...l, `✅ ${rtsctsSteps[step].label}`]);
    }
  };

  const simCollision = () => {
    setCollision(true);
    setChannelBusy(true);
    setLog((l) => [...l, "⚠️ 检测到碰撞！其他站点同时发送"]);
    setTimeout(() => setChannelBusy(false), 1000);
  };

  const reset = () => { setStep(0); setBackoff(5); setCollision(false); setCwMin(15); setLog([]); setChannelBusy(false); };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">WiFi MAC 控制器 (CSMA/CA)</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-text-muted text-xs block mb-1">初始退避: {backoff}</label>
          <input type="range" min="1" max="15" value={backoff} onChange={(e) => setBackoff(Number(e.target.value))} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="text-text-muted text-xs block mb-1">CWmin: {cwMin}</label>
          <input type="range" min="15" max="255" value={cwMin} onChange={(e) => setCwMin(Number(e.target.value))} className="w-full accent-green-500" />
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={next} disabled={step >= rtsctsSteps.length - 1}
          className="px-4 py-2 rounded bg-blue-500 text-white text-sm disabled:opacity-50 hover:bg-blue-600">下一步</button>
        <button onClick={simCollision} disabled={step !== 1}
          className="px-3 py-2 rounded bg-red-500 text-white text-sm disabled:opacity-50 hover:bg-red-600">模拟碰撞</button>
        <button onClick={reset} className="px-3 py-2 rounded border border-border-subtle text-text-muted text-sm">重置</button>
      </div>
      <div className={`p-3 rounded-lg mb-4 ${channelBusy ? "bg-red-500/10 border border-red-400/30" : "bg-green-500/10 border border-green-400/30"}`}>
        <span className={channelBusy ? "text-red-400 text-sm" : "text-green-400 text-sm"}>
          信道状态: {channelBusy ? "🔴 忙碌 (碰撞中)" : "🟢 空闲"}
        </span>
      </div>
      <div className="space-y-2 mb-4">
        {rtsctsSteps.map((s, i) => (
          <div key={i} className={`flex items-center gap-3 p-3 rounded-lg border transition-all ${
            i < step ? "bg-green-500/10 border-green-400/30" : i === step ? "bg-blue-500/10 border-blue-400/30" : "border-border-subtle opacity-40"
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              i < step ? "bg-green-500 text-white" : i === step ? "bg-blue-500 text-white" : "bg-gray-300 dark:bg-gray-600 text-text-muted"
            }`}>{i + 1}</span>
            <div className="flex-1">
              <span className="text-text-primary text-sm">{s.label}</span>
              <p className="text-text-muted text-xs">{s.desc}</p>
            </div>
            <span className="text-text-muted text-xs px-2 py-0.5 rounded bg-bg-primary border border-border-subtle">{s.time}</span>
          </div>
        ))}
      </div>
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-secondary text-xs font-medium mb-1">CSMA/CA 时序</h4>
        <div className="flex items-center gap-1 text-xs">
          {["DIFS", "Backoff", "RTS", "SIFS", "CTS", "SIFS", "Data", "SIFS", "ACK"].map((t, i) => (
            <span key={i} className={`px-1.5 py-0.5 rounded ${t === "DIFS" || t === "Backoff" ? "bg-blue-500/10 text-blue-400" : t === "SIFS" ? "bg-yellow-500/10 text-yellow-400" : "bg-green-500/10 text-green-400"}`}>{t}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
export default WiFiMACController;
