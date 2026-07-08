"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Bell, BellOff, ArrowRight, CheckCircle } from "lucide-react";

export function EventDemo() {
  const [eventSet, setEventSet] = useState(false);
  const [consumerState, setConsumerState] = useState<"阻塞" | "就绪" | "完成">("阻塞");
  const [producerState, setProducerState] = useState<"等待" | "发送" | "完成">("等待");
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const reset = () => {
    setRunning(false); setEventSet(false); setConsumerState("阻塞"); setProducerState("等待"); setStep(0);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  useEffect(() => {
    if (!running) return;
    intervalRef.current = setInterval(() => { setStep((s) => { if (s >= 6) { setRunning(false); return s; } return s + 1; }); }, 800);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running]);

  useEffect(() => {
    if (step === 1) setProducerState("发送");
    if (step === 2) { setEventSet(true); setProducerState("完成"); }
    if (step === 3) setConsumerState("就绪");
    if (step === 4) setConsumerState("完成");
  }, [step]);

  const Card = ({ title, state, color, children }: { title: string; state: string; color: string; children: React.ReactNode }) => (
    <motion.div layout className={`rounded-xl border p-4 border-${color}-400 bg-${color}-50 dark:bg-${color}-900/20`}>
      <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">{title}</div>
      <div className="flex items-center gap-2 mb-3">
        <div className={`w-3 h-3 rounded-full bg-${color}-500 ${state === "运行中" || state === "发送" ? "animate-pulse" : ""}`} />
        <span className="text-sm text-slate-600 dark:text-slate-400">{state}</span>
      </div>
      {children}
    </motion.div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Bell className="w-5 h-5 text-amber-500" /> 事件同步演示 — 生产者/消费者
      </h3>
      <div className="flex flex-wrap gap-3 mb-6">
        <button onClick={() => { reset(); setRunning(true); }} disabled={running}
          className="px-4 py-2 rounded-lg bg-amber-600 text-white font-medium text-sm flex items-center gap-2 disabled:opacity-50">
          <Play className="w-4 h-4" /> 运行演示
        </button>
        <button onClick={() => { setEventSet(!eventSet); if (!eventSet) setConsumerState("就绪"); }}
          className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm flex items-center gap-2">
          {eventSet ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />} 切换事件: {eventSet ? "已设置" : "未设置"}
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm"><RotateCcw className="w-4 h-4" /></button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card title="生产者" state={producerState} color={producerState === "发送" ? "orange" : producerState === "完成" ? "green" : "slate"}>
          {producerState === "完成" && <div className="text-xs text-green-600 dark:text-green-400 flex items-center gap-1"><CheckCircle className="w-3 h-3" /> 事件已发送</div>}
        </Card>
        <motion.div layout className={`rounded-xl border p-4 ${eventSet ? "border-amber-400 bg-amber-50 dark:bg-amber-900/20" : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"}`}>
          <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">事件对象</div>
          <div className="flex items-center justify-center">
            <motion.div animate={{ scale: eventSet ? [1, 1.3, 1] : 1 }} transition={{ duration: 0.3 }}
              className={`w-16 h-16 rounded-full flex items-center justify-center ${eventSet ? "bg-amber-400 dark:bg-amber-600" : "bg-slate-200 dark:bg-slate-600"}`}>
              {eventSet ? <Bell className="w-8 h-8 text-white" /> : <BellOff className="w-8 h-8 text-slate-400" />}
            </motion.div>
          </div>
          <div className="text-center text-xs mt-2 text-slate-500">状态: {eventSet ? "SET" : "CLEAR"}</div>
        </motion.div>
        <Card title="消费者" state={consumerState} color={consumerState === "阻塞" ? "red" : consumerState === "就绪" ? "blue" : "green"}>
          {consumerState === "阻塞" && <div className="text-xs text-red-600 dark:text-red-400">await event.wait() 阻塞中...</div>}
          {consumerState === "就绪" && <div className="text-xs text-blue-600 dark:text-blue-400 flex items-center gap-1"><ArrowRight className="w-3 h-3" /> 事件已触发</div>}
          {consumerState === "完成" && <div className="text-xs text-green-600 dark:text-green-400 flex items-center gap-1"><CheckCircle className="w-3 h-3" /> 处理完成</div>}
        </Card>
      </div>
      <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-4 text-sm font-mono text-slate-600 dark:text-slate-400">
        {[{ s: 1, t: "生产者: 准备发送事件" }, { s: 2, t: "生产者: event.set() → 事件变为 SET" }, { s: 3, t: "消费者: event.wait() 检测到事件，解除阻塞" }, { s: 4, t: "消费者: 开始处理数据" }].map(({ s, t }) => (
          <motion.div key={s} initial={{ opacity: 0 }} animate={{ opacity: step >= s ? 1 : 0.3 }} className="flex items-center gap-2 py-1">
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${step >= s ? "bg-amber-500 text-white" : "bg-slate-300 dark:bg-slate-600 text-slate-500"}`}>{s}</span>{t}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
