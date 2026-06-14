"use client";
import { useState, useEffect } from "react";

interface StationState {
  id: number;
  data: boolean;
  collision: boolean;
  backoff: number;
  attempts: number;
  transmitting: boolean;
  jamming: boolean;
}

export function CSMA_CDSimulator() {
  const [stations, setStations] = useState<StationState[]>([]);
  const [channel, setChannel] = useState<"idle" | "busy" | "collision">("idle");
  const [log, setLog] = useState<string[]>([]);
  const [tick, setTick] = useState(0);
  const [running, setRunning] = useState(false);
  const [slotTime, setSlotTime] = useState(2);

  useEffect(() => {
    setStations(Array.from({ length: 4 }, (_, i) => ({
      id: i, data: i < 3, collision: false, backoff: 0, attempts: 0, transmitting: false, jamming: false,
    })));
  }, []);

  const addLog = (msg: string) => setLog((prev) => [...prev.slice(-20), `[T${tick}] ${msg}`]);

  const step = () => {
    setStations((prev) => {
      const next = prev.map((s) => ({ ...s }));
      const ready = next.filter((s) => s.data && s.backoff === 0 && !s.transmitting);

      if (channel === "idle" && ready.length > 0) {
        if (ready.length >= 2) {
          ready.forEach((s) => {
            s.transmitting = true;
            s.collision = true;
          });
          setChannel("collision");
          addLog(`碰撞检测! ${ready.length}个站点同时发送，发送Jam信号`);
        } else {
          ready[0].transmitting = true;
          setChannel("busy");
          addLog(`Station ${ready[0].id}: 检测到空闲，开始发送`);
        }
      } else if (channel === "collision") {
        next.filter((s) => s.collision).forEach((s) => {
          s.attempts++;
          const maxSlot = Math.min(Math.pow(2, s.attempts), 1024);
          s.backoff = Math.floor(Math.random() * maxSlot) * slotTime;
          s.collision = false;
          s.transmitting = false;
          s.jamming = false;
          addLog(`Station ${s.id}: 二进制退避，等待 ${s.backoff} 个时隙 (k≤${maxSlot})`);
        });
        setChannel("idle");
      } else if (channel === "busy") {
        next.filter((s) => s.transmitting).forEach((s) => {
          s.transmitting = false;
          s.data = false;
          addLog(`Station ${s.id}: 传输完成`);
        });
        setChannel("idle");
      }

      next.filter((s) => s.backoff > 0).forEach((s) => s.backoff--);
      return next;
    });
    setTick((t) => t + 1);
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 700);
    return () => clearInterval(id);
  }, [running, stations, channel, tick]);

  const resetAll = () => {
    setStations(Array.from({ length: 4 }, (_, i) => ({
      id: i, data: i < 3, collision: false, backoff: 0, attempts: 0, transmitting: false, jamming: false,
    })));
    setChannel("idle");
    setLog([]);
    setTick(0);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">CSMA/CD 碰撞检测 + 二进制退避</h3>
      <div className="mb-4">
        <label className="text-xs text-text-secondary">时隙长度: {slotTime} μs</label>
        <input type="range" min={1} max={10} value={slotTime} onChange={(e) => setSlotTime(+e.target.value)} className="w-full mt-1" />
      </div>
      <div className="grid grid-cols-4 gap-3 mb-4">
        {stations.map((s) => (
          <div key={s.id} className={`p-3 rounded-lg border-2 text-center transition-all ${s.transmitting && s.collision ? "border-red-500 bg-red-50 dark:bg-red-900/20" : s.transmitting ? "border-green-500 bg-green-50 dark:bg-green-900/20" : s.backoff > 0 ? "border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20" : "border-border-subtle bg-gray-50 dark:bg-gray-800"}`}>
            <div className="text-sm font-bold text-text-primary">站点 {s.id}</div>
            <div className={`text-xs mt-1 ${s.collision ? "text-red-600" : s.transmitting ? "text-green-600" : s.backoff > 0 ? "text-yellow-600" : s.data ? "text-blue-600" : "text-text-secondary"}`}>
              {s.collision ? "碰撞!" : s.transmitting ? "发送中" : s.backoff > 0 ? `退避:${s.backoff}` : s.data ? "就绪" : "空闲"}
            </div>
            <div className="text-[10px] text-text-secondary mt-1">尝试: {s.attempts}</div>
          </div>
        ))}
      </div>
      <div className="mb-4">
        <div className={`w-full h-6 rounded flex items-center justify-center text-xs font-medium ${channel === "collision" ? "bg-red-500 text-white" : channel === "busy" ? "bg-green-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
          信道: {channel === "idle" ? "空闲" : channel === "busy" ? "忙碌" : "碰撞!"}
        </div>
      </div>
      <div className="mb-4 p-2 rounded bg-gray-50 dark:bg-gray-800 h-32 overflow-y-auto">
        {log.map((l, i) => <div key={i} className="text-[10px] text-text-secondary font-mono">{l}</div>)}
        {log.length === 0 && <div className="text-xs text-text-secondary text-center">点击单步或自动运行开始模拟</div>}
      </div>
      <div className="flex gap-2">
        <button onClick={step} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">单步</button>
        <button onClick={() => setRunning(!running)} className={`flex-1 py-2 rounded text-sm ${running ? "bg-red-600 text-white" : "bg-green-600 text-white"}`}>
          {running ? "暂停" : "自动运行"}
        </button>
        <button onClick={resetAll} className="px-4 py-2 bg-gray-500 text-white rounded text-sm">重置</button>
      </div>
      <p className="text-xs text-text-secondary mt-3">CSMA/CD: 先听后发，边发边听。检测到碰撞后发送Jam信号，采用截断二进制指数退避算法。</p>
    </div>
  );
}
export default CSMA_CDSimulator;
