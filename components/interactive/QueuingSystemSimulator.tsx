"use client";
import { useState, useEffect, useRef, useCallback } from "react";

interface Customer {
  id: number;
  arrivalTime: number;
  serviceTime: number;
  startTime: number;
  finishTime: number;
}

export function QueuingSystemSimulator() {
  const [lambda, setLambda] = useState(3);
  const [mu, setMu] = useState(5);
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [queue, setQueue] = useState<number[]>([]);
  const [serverBusy, setServerBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [time, setTime] = useState(0);
  const [stats, setStats] = useState({ avgWait: 0, avgQueueLen: 0, utilization: 0 });
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const nextId = useRef(1);
  const queueRef = useRef<number[]>([]);

  const start = useCallback(() => {
    setCustomers([]);
    setQueue([]);
    setServerBusy(false);
    setTime(0);
    setStats({ avgWait: 0, avgQueueLen: 0, utilization: 0 });
    nextId.current = 1;
    queueRef.current = [];
    setRunning(true);
  }, []);

  useEffect(() => {
    if (!running) return;
    timerRef.current = setInterval(() => {
      setTime((t) => {
        const newTime = t + 1;
        if (Math.random() < lambda / 10) {
          const id = nextId.current++;
          const serviceTime = Math.ceil(-Math.log(1 - Math.random()) / (mu / 10)) + 1;
          const cust: Customer = { id, arrivalTime: newTime, serviceTime, startTime: -1, finishTime: -1 };
          setCustomers((prev) => [...prev, cust]);
          queueRef.current.push(id);
          setQueue([...queueRef.current]);
        }
        return newTime;
      });
    }, 300);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running, lambda, mu]);

  useEffect(() => {
    if (!running || queueRef.current.length === 0 || serverBusy) return;
    const id = queueRef.current.shift()!;
    setQueue([...queueRef.current]);
    setServerBusy(true);
    setCustomers((prev) => prev.map((c) => c.id === id ? { ...c, startTime: time } : c));
    const cust = customers.find((c) => c.id === id);
    const svcTime = cust ? cust.serviceTime : 3;
    setTimeout(() => {
      setCustomers((prev) => prev.map((c) => c.id === id ? { ...c, finishTime: time + svcTime } : c));
      setServerBusy(false);
    }, svcTime * 300);
  }, [customers, time, running, serverBusy]);

  useEffect(() => {
    if (customers.length > 1) {
      const completed = customers.filter((c) => c.finishTime >= 0);
      if (completed.length > 0) {
        const avgWait = completed.reduce((s, c) => s + (c.startTime - c.arrivalTime), 0) / completed.length;
        const avgLen = customers.reduce((s, c) => s + (c.finishTime - c.arrivalTime), 0) / customers.length;
        setStats({ avgWait, avgQueueLen: avgLen / 10, utilization: serverBusy ? 100 : stats.utilization });
      }
    }
  }, [customers, serverBusy]);

  const rho = lambda / mu;
  const theoreticalLq = (rho * rho) / (1 - rho);
  const theoreticalWq = theoreticalLq / lambda;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">M/M/1排队系统模拟器</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          到达率 λ: <span className="text-text-primary font-mono">{lambda}/10</span>
          <input type="range" min={1} max={9} value={lambda} onChange={(e) => setLambda(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          服务率 μ: <span className="text-text-primary font-mono">{mu}/10</span>
          <input type="range" min={2} max={10} value={mu} onChange={(e) => setMu(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={start} className="px-4 py-1.5 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-sm font-medium hover:bg-sky-500/25 transition-colors">
          {running ? "重新开始" : "开始模拟"}
        </button>
        <button onClick={() => { if (timerRef.current) clearInterval(timerRef.current); setRunning(false); }}
          className="px-4 py-1.5 rounded-lg bg-red-500/15 text-red-600 dark:text-red-400 text-sm font-medium hover:bg-red-500/25 transition-colors">
          停止
        </button>
        <span className="text-xs text-text-tertiary ml-auto">时间: {time}</span>
      </div>
      <div className="flex items-center gap-3 mb-4 p-3 rounded-lg border border-border-subtle bg-bg-tertiary">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold ${serverBusy ? "bg-red-500/20 text-red-500" : "bg-emerald-500/20 text-emerald-500"}`}>
          {serverBusy ? "忙" : "空"}
        </div>
        <div className="flex-1">
          <div className="text-xs text-text-secondary mb-1">队列 ({queue.length})</div>
          <div className="flex gap-1 overflow-x-auto">
            {queue.map((id, i) => (
              <span key={i} className="px-2 py-0.5 rounded bg-sky-500/15 text-sky-700 dark:text-sky-300 text-[10px] font-mono shrink-0">#{id}</span>
            ))}
            {queue.length === 0 && <span className="text-[10px] text-text-tertiary">空</span>}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs space-y-1">
          <div className="font-medium text-text-primary">理论值 (M/M/1)</div>
          <div className="text-text-secondary">ρ = λ/μ = {rho.toFixed(2)}</div>
          <div className="text-text-secondary">Lq = {theoreticalLq.toFixed(2)}（平均队列长度）</div>
          <div className="text-text-secondary">Wq = {theoreticalWq.toFixed(2)}（平均等待时间）</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs space-y-1">
          <div className="font-medium text-text-primary">模拟统计</div>
          <div className="text-text-secondary">已到达: {customers.length} 人</div>
          <div className="text-text-secondary">平均等待: {stats.avgWait.toFixed(1)}</div>
          <div className="text-text-secondary">完成: {customers.filter((c) => c.finishTime >= 0).length} 人</div>
        </div>
      </div>
      <div className="mt-3 text-[10px] text-text-tertiary">M/M/1: 泊松到达 + 指数服务时间 + 单服务台 · 要求 ρ &lt; 1（λ &lt; μ）否则队列无限增长</div>
    </div>
  );
}
export default QueuingSystemSimulator;
