"use client";
import { useState, useRef, useEffect, useCallback } from "react";

export default function ARQTimerDemo() {
  const [timeoutVal, setTimeoutVal] = useState(3000);
  const [rtt, setRTT] = useState(1000);
  const [events, setEvents] = useState<string[]>([]);
  const [timerRunning, setTimerRunning] = useState(false);
  const [timerProgress, setTimerProgress] = useState(0);
  const [ackReceived, setAckReceived] = useState(false);
  const timerRef = useRef<any>(null);
  const startRef = useRef(0);

  const cleanup = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  const sendFrame = useCallback(() => {
    cleanup();
    setEvents(["发送数据帧 Seq=0", "启动重传定时器..."]);
    setTimerRunning(true);
    setTimerProgress(0);
    setAckReceived(false);
    startRef.current = Date.now();

    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - startRef.current;
      const progress = Math.min(100, (elapsed / timeoutVal) * 100);
      setTimerProgress(progress);

      if (elapsed >= rtt && !ackReceived) {
        setAckReceived(true);
        setEvents((e: string[]) => [...e, "收到 ACK (RTT=" + rtt + "ms)", "确认成功，取消定时器"]);
        setTimerRunning(false);
        cleanup();
      }

      if (elapsed >= timeoutVal && !ackReceived) {
        setEvents((e: string[]) => [...e, "定时器超时！(" + timeoutVal + "ms)", "重传数据帧 Seq=0", "重启定时器..."]);
        startRef.current = Date.now();
        setTimerProgress(0);
      }
    }, 50);
  }, [timeoutVal, rtt, cleanup]);

  const simulateLoss = useCallback(() => {
    cleanup();
    setEvents(["发送数据帧 Seq=0", "启动重传定时器...", "数据帧丢失！等待超时..."]);
    setTimerRunning(true);
    setTimerProgress(0);
    setAckReceived(false);
    startRef.current = Date.now();

    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - startRef.current;
      setTimerProgress(Math.min(100, (elapsed / timeoutVal) * 100));
      if (elapsed >= timeoutVal) {
        setEvents((e: string[]) => [...e, "定时器超时！(" + timeoutVal + "ms)", "重传数据帧 Seq=0"]);
        setTimerRunning(false);
        cleanup();
      }
    }, 50);
  }, [timeoutVal, cleanup]);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">ARQ 定时器演示</h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">模拟可靠传输协议中的超时重传机制</p>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-xs text-slate-500 dark:text-gray-400">超时时间: {timeoutVal}ms</label>
          <input type="range" min={1000} max={8000} step={500} value={timeoutVal} onChange={(e) => setTimeoutVal(+e.target.value)} className="w-full mt-1" />
        </div>
        <div>
          <label className="text-xs text-slate-500 dark:text-gray-400">RTT: {rtt}ms</label>
          <input type="range" min={500} max={7000} step={500} value={rtt} onChange={(e) => setRTT(+e.target.value)} className="w-full mt-1" />
        </div>
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={sendFrame} disabled={timerRunning} className="px-4 py-1.5 rounded bg-blue-500 text-white text-sm disabled:opacity-50">发送帧（正常）</button>
        <button onClick={simulateLoss} disabled={timerRunning} className="px-4 py-1.5 rounded bg-red-500 text-white text-sm disabled:opacity-50">发送帧（模拟丢包）</button>
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-xs mb-1 text-slate-500 dark:text-gray-400">
          <span>定时器进度</span>
          <span>{timerRunning ? timerProgress.toFixed(0) + "%" : "停止"}</span>
        </div>
        <div className="w-full h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div className={"h-full transition-all duration-75 rounded-full " + (timerProgress > 80 ? "bg-red-500" : "bg-blue-500")} style={{ width: timerProgress + "%" }} />
        </div>
        {rtt < timeoutVal && <p className="text-xs text-green-600 dark:text-green-400 mt-1">RTT {"<"} 超时时间，正常情况不会超时</p>}
        {rtt >= timeoutVal && <p className="text-xs text-red-600 dark:text-red-400 mt-1">RTT ≥ 超时时间，会导致不必要的重传</p>}
      </div>

      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-xs text-slate-600 dark:text-gray-300 max-h-28 overflow-y-auto space-y-0.5">
        {events.length === 0 ? "点击按钮开始演示" : events.map((e, i) => <div key={i}>{e}</div>)}
      </div>
    </div>
  );
}

export { ARQTimerDemo };
