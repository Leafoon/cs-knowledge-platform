"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, Globe, Server, Clock, ArrowRight, Loader2, CheckCircle } from "lucide-react";

interface Step {
  name: string;
  duration: number;
  async: boolean;
  desc: string;
}

const STEPS: Step[] = [
  { name: "DNS 查询", duration: 800, async: true, desc: "解析域名 → IP 地址" },
  { name: "TCP 连接", duration: 600, async: true, desc: "三次握手建立连接" },
  { name: "发送请求", duration: 300, async: false, desc: "构造并发送 HTTP 请求" },
  { name: "等待响应", duration: 2000, async: true, desc: "服务器处理请求 (I/O 等待)" },
  { name: "接收数据", duration: 500, async: true, desc: "接收并缓存响应数据" },
  { name: "处理数据", duration: 400, async: false, desc: "解析 JSON / 渲染页面" },
];

export function AsyncHttpRequestFlow() {
  const [running, setRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [stepProgress, setStepProgress] = useState(0);
  const [completed, setCompleted] = useState<boolean[]>(Array(STEPS.length).fill(false));
  const [totalElapsed, setTotalElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const reset = () => {
    setRunning(false);
    setCurrentStep(-1);
    setStepProgress(0);
    setCompleted(Array(STEPS.length).fill(false));
    setTotalElapsed(0);
    if (timerRef.current) clearInterval(timerRef.current);
    if (stepTimerRef.current) clearInterval(stepTimerRef.current);
  };

  const start = () => {
    reset();
    setRunning(true);
    setCurrentStep(0);
  };

  useEffect(() => {
    if (!running || currentStep < 0 || currentStep >= STEPS.length) return;
    const step = STEPS[currentStep];
    const interval = 50;
    setStepProgress(0);

    stepTimerRef.current = setInterval(() => {
      setStepProgress((p) => {
        const next = p + (interval / step.duration) * 100;
        if (next >= 100) {
          if (stepTimerRef.current) clearInterval(stepTimerRef.current);
          setCompleted((c) => { const n = [...c]; n[currentStep] = true; return n; });
          if (currentStep + 1 < STEPS.length) {
            setCurrentStep(currentStep + 1);
          } else {
            setRunning(false);
          }
          return 100;
        }
        return next;
      });
      setTotalElapsed((t) => t + interval);
    }, interval);

    return () => { if (stepTimerRef.current) clearInterval(stepTimerRef.current); };
  }, [running, currentStep]);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Globe className="w-5 h-5 text-cyan-500" />
        HTTP 请求异步流程
      </h3>

      <div className="flex gap-3 mb-6">
        <button onClick={start} disabled={running} className="px-4 py-2 rounded-lg bg-cyan-600 text-white font-medium text-sm flex items-center gap-2 disabled:opacity-50">
          <Play className="w-4 h-4" /> 发送请求
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm flex items-center gap-2">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
        <div className="ml-auto flex items-center gap-2 text-sm text-slate-500">
          <Clock className="w-4 h-4" /> {(totalElapsed / 1000).toFixed(1)}s
        </div>
      </div>

      <div className="flex items-center gap-2 mb-6 overflow-x-auto pb-2">
        {STEPS.map((step, i) => (
          <React.Fragment key={i}>
            <motion.div
              animate={{
                scale: i === currentStep ? 1.05 : 1,
                opacity: completed[i] ? 0.7 : 1,
              }}
              className={`flex-shrink-0 rounded-xl border p-3 min-w-[140px] ${
                i === currentStep && running
                  ? "border-cyan-400 bg-cyan-50 dark:bg-cyan-900/20 shadow-lg shadow-cyan-200/50 dark:shadow-cyan-900/30"
                  : completed[i]
                  ? "border-green-300 bg-green-50 dark:bg-green-900/20"
                  : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"
              }`}
            >
              <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">{step.name}</div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 mb-2">{step.desc}</div>
              <div className="h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full rounded-full ${completed[i] ? "bg-green-500" : "bg-cyan-500"}`}
                  animate={{ width: completed[i] ? "100%" : `${stepProgress}%` }}
                />
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${step.async ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300" : "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300"}`}>
                  {step.async ? "异步" : "同步"}
                </span>
                <span className="text-[10px] text-slate-400">{step.duration}ms</span>
              </div>
            </motion.div>
            {i < STEPS.length - 1 && <ArrowRight className="w-4 h-4 text-slate-300 flex-shrink-0" />}
          </React.Fragment>
        ))}
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-4">
        <div className="text-xs font-medium text-slate-500 mb-2">异步关键点</div>
        <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-500" />
            <span><strong>异步步骤</strong> (DNS、连接、等待、接收): 执行器可以在等待期间切换到其他协程</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-orange-500" />
            <span><strong>同步步骤</strong> (发送、处理): CPU 密集型操作，会阻塞当前协程</span>
          </div>
        </div>
      </div>
    </div>
  );
}
