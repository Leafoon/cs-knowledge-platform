"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Zap, AlertTriangle } from "lucide-react";

export default function SignalHandlingDemo() {
  const [step, setStep] = useState(0);

  const steps = [
    { title: "进程正常运行", desc: "进程在用户态执行代码", state: "running" },
    { title: "内核发送信号 SIGINT", desc: "用户按 Ctrl+C，内核设置进程 PCB 中的信号待处理位", state: "signal_pending" },
    { title: "返回用户态前检查", desc: "系统调用/中断返回时，内核检查 pending signals", state: "checking" },
    { title: "执行信号处理函数", desc: "跳转到用户注册的 signal_handler()", state: "handling" },
    { title: "恢复执行", desc: "处理函数返回，继续原程序执行", state: "resumed" }
  ];

  const current = steps[step];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-yellow-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Zap className="w-7 h-7 text-yellow-600" />
        信号处理演示
      </h3>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h4 className="font-bold text-lg text-slate-800">{current.title}</h4>
            <p className="text-sm text-slate-600 mt-1">{current.desc}</p>
          </div>
          <div className="text-2xl font-bold text-indigo-600">步骤 {step + 1}/5</div>
        </div>

        <motion.div key={step} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="bg-gradient-to-r from-indigo-100 to-purple-100 rounded-lg p-6 mb-6 border-2 border-indigo-300">
          {current.state === "running" && (
            <div className="text-center">
              <div className="bg-green-500 text-white px-6 py-3 rounded-lg inline-block font-bold">进程运行中 (用户态)</div>
              <div className="mt-4 text-sm text-slate-700">执行 main() 函数，CPU 处于 Ring 3</div>
            </div>
          )}
          {current.state === "signal_pending" && (
            <div className="space-y-3">
              <div className="bg-yellow-500 text-white px-6 py-3 rounded-lg text-center font-bold flex items-center justify-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                信号待处理 (SIGINT)
              </div>
              <div className="bg-white p-4 rounded border border-yellow-300">
                <div className="text-xs text-slate-600 mb-2">PCB 信号位图</div>
                <div className="font-mono text-sm">pending: 0b0000000000000010 (SIGINT = bit 1)</div>
              </div>
            </div>
          )}
          {current.state === "checking" && (
            <div className="space-y-3">
              <div className="bg-blue-500 text-white px-6 py-3 rounded-lg text-center font-bold">内核检查信号</div>
              <div className="bg-white p-4 rounded border border-blue-300">
                <div className="text-sm text-slate-700">从内核态返回用户态前：</div>
                <div className="font-mono text-xs mt-2 bg-slate-100 p-2 rounded">if (pending_signals & ~blocked_signals)</div>
                <div className="text-xs text-slate-600 mt-2">发现 SIGINT 待处理且未被阻塞</div>
              </div>
            </div>
          )}
          {current.state === "handling" && (
            <div className="space-y-3">
              <div className="bg-purple-500 text-white px-6 py-3 rounded-lg text-center font-bold">执行信号处理函数</div>
              <div className="bg-white p-4 rounded border border-purple-300">
                <div className="text-sm text-slate-700 mb-2">跳转到用户定义的处理函数：</div>
                <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs">
{`void signal_handler(int sig) {
    printf("收到信号 %d\\n", sig);
    exit(0);
}`}
                </pre>
              </div>
            </div>
          )}
          {current.state === "resumed" && (
            <div className="text-center">
              <div className="bg-green-500 text-white px-6 py-3 rounded-lg inline-block font-bold">恢复执行 / 进程终止</div>
              <div className="mt-4 text-sm text-slate-700">若处理函数调用 exit()，进程终止；否则恢复原程序执行</div>
            </div>
          )}
        </motion.div>

        <div className="flex justify-center gap-4">
          <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} className="px-6 py-2 bg-slate-300 rounded-lg font-semibold disabled:opacity-50">上一步</button>
          <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step === steps.length - 1} className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-semibold disabled:opacity-50">下一步</button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">常见信号</h5>
          <table className="w-full text-sm">
            <thead><tr className="border-b"><th className="text-left py-2">信号</th><th className="text-left py-2">默认行为</th></tr></thead>
            <tbody>
              <tr className="border-b"><td className="py-2 font-mono">SIGINT (2)</td><td>终止 (Ctrl+C)</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">SIGKILL (9)</td><td>强制终止（不可捕获）</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">SIGSEGV (11)</td><td>段错误终止</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">SIGCHLD (17)</td><td>子进程状态改变（忽略）</td></tr>
              <tr><td className="py-2 font-mono">SIGUSR1 (10)</td><td>用户自定义</td></tr>
            </tbody>
          </table>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">信号处理示例</h5>
          <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs overflow-x-auto">
{`#include <signal.h>

void handler(int sig) {
    printf("收到 %d\\n", sig);
}

int main() {
    signal(SIGINT, handler);
    while (1) {
        pause(); // 等待信号
    }
}`}
          </pre>
        </div>
      </div>

      <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h5 className="font-bold text-amber-800 mb-2">关键点</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>异步</strong>：信号可在任意时刻到达</li>
          <li><strong>不可靠</strong>：多个相同信号可能丢失（非实时信号）</li>
          <li><strong>不可重入</strong>：处理函数中应避免调用非安全函数（如 printf、malloc）</li>
          <li><strong>SIGKILL/SIGSTOP</strong> 无法捕获或忽略</li>
        </ul>
      </div>
    </div>
  );
}
