"use client";
import { useState, useEffect, useRef } from "react";

type Protocol = "1-persistent" | "non-persistent" | "p-persistent";

interface Station {
  id: number;
  hasData: boolean;
  waiting: boolean;
  transmitting: boolean;
  collisions: number;
  backoff: number;
}

export function CSMAComparison() {
  const [protocol, setProtocol] = useState<Protocol>("1-persistent");
  const [stations, setStations] = useState<Station[]>([]);
  const [channelBusy, setChannelBusy] = useState(false);
  const [pValue, setPValue] = useState(0.3);
  const [log, setLog] = useState<string[]>([]);
  const [tick, setTick] = useState(0);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    setStations(Array.from({ length: 5 }, (_, i) => ({
      id: i, hasData: Math.random() > 0.5, waiting: false, transmitting: false, collisions: 0, backoff: 0,
    })));
  }, []);

  const addLog = (msg: string) => setLog((prev) => [...prev.slice(-15), `[${tick}] ${msg}`]);

  const step = () => {
    setStations((prev) => {
      const next = prev.map((s) => ({ ...s }));
      const activeStations = next.filter((s) => s.hasData && s.backoff === 0);
      
      if (protocol === "1-persistent") {
        if (!channelBusy && activeStations.length > 0) {
          if (activeStations.length === 1) {
            activeStations[0].transmitting = true;
            activeStations[0].hasData = false;
            setChannelBusy(true);
            addLog(`Station ${activeStations[0].id}: 检测到空闲，立即发送`);
          } else {
            activeStations.forEach((s) => { s.collisions++; s.backoff = Math.floor(Math.random() * 4) + 1; });
            addLog(`碰撞! ${activeStations.length}个站点同时发送`);
          }
        }
      } else if (protocol === "non-persistent") {
        if (!channelBusy && activeStations.length > 0) {
          if (activeStations.length === 1) {
            activeStations[0].transmitting = true;
            activeStations[0].hasData = false;
            setChannelBusy(true);
            addLog(`Station ${activeStations[0].id}: 信道空闲，发送数据`);
          } else {
            activeStations.forEach((s) => { s.collisions++; s.backoff = Math.floor(Math.random() * 5) + 1; });
            addLog(`碰撞! 随机退避后重试`);
          }
        }
        next.filter((s) => s.hasData && s.backoff > 0).forEach((s) => s.backoff--);
      } else {
        if (!channelBusy && activeStations.length > 0) {
          activeStations.forEach((s) => {
            if (Math.random() < pValue) {
              if (!channelBusy) {
                s.transmitting = true;
                s.hasData = false;
                setChannelBusy(true);
                addLog(`Station ${s.id}: 以p=${pValue}概率发送`);
              }
            } else {
              s.backoff = 1;
              addLog(`Station ${s.id}: 等待下一时隙`);
            }
          });
        }
      }

      next.filter((s) => s.transmitting).forEach((s) => {
        s.transmitting = false;
        setChannelBusy(false);
        addLog(`Station ${s.id}: 传输完成`);
      });
      next.filter((s) => s.backoff > 0).forEach((s) => s.backoff--);
      return next;
    });
    setTick((t) => t + 1);
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 800);
    return () => clearInterval(id);
  }, [running, stations, protocol, tick]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">CSMA 协议对比</h3>
      <div className="flex gap-2 mb-4">
        {(["1-persistent", "non-persistent", "p-persistent"] as Protocol[]).map((p) => (
          <button key={p} onClick={() => setProtocol(p)}
            className={`flex-1 py-1.5 rounded text-sm ${protocol === p ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {p}
          </button>
        ))}
      </div>
      {protocol === "p-persistent" && (
        <div className="mb-4">
          <label className="text-xs text-text-secondary">发送概率 p = {pValue.toFixed(2)}</label>
          <input type="range" min={0.1} max={1} step={0.05} value={pValue} onChange={(e) => setPValue(+e.target.value)} className="w-full mt-1" />
        </div>
      )}
      <div className="grid grid-cols-5 gap-2 mb-4">
        {stations.map((s) => (
          <div key={s.id} className={`p-2 rounded border text-center transition-all ${s.transmitting ? "border-green-500 bg-green-50 dark:bg-green-900/20" : s.backoff > 0 ? "border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20" : "border-border-subtle bg-gray-50 dark:bg-gray-800"}`}>
            <div className="text-xs font-bold text-text-primary">S{s.id}</div>
            <div className={`text-[10px] mt-1 ${s.transmitting ? "text-green-600" : s.backoff > 0 ? "text-yellow-600" : s.hasData ? "text-blue-600" : "text-text-secondary"}`}>
              {s.transmitting ? "发送中" : s.backoff > 0 ? `退避${s.backoff}` : s.hasData ? "就绪" : "空闲"}
            </div>
            <div className="text-[10px] text-red-500">碰撞:{s.collisions}</div>
          </div>
        ))}
      </div>
      <div className="mb-4">
        <div className={`w-full h-4 rounded ${channelBusy ? "bg-green-500" : "bg-gray-200 dark:bg-gray-700"}`}>
          <span className="text-[10px] text-white px-2 leading-4">{channelBusy ? "信道忙" : "信道空闲"}</span>
        </div>
      </div>
      <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 h-28 overflow-y-auto mb-4">
        {log.map((l, i) => <div key={i} className="text-[10px] text-text-secondary font-mono">{l}</div>)}
      </div>
      <div className="flex gap-2">
        <button onClick={step} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">单步</button>
        <button onClick={() => setRunning(!running)} className={`flex-1 py-2 rounded text-sm ${running ? "bg-red-600 text-white" : "bg-green-600 text-white"}`}>
          {running ? "暂停" : "自动"}
        </button>
      </div>
      <p className="text-xs text-text-secondary mt-3">1-persistent：一直监听，空闲就发；non-persistent：不忙就发，忙就随机等；p-persistent：以概率p发送。</p>
    </div>
  );
}
export default CSMAComparison;
