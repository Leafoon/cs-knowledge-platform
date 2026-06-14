"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Skull, Baby, AlertTriangle, Info } from "lucide-react";

export default function ZombieOrphanDemo() {
  const [mode, setMode] = useState<"zombie" | "orphan">("zombie");
  const [step, setStep] = useState(0);

  const zombieSteps = [
    {
      step: 0,
      title: "父进程和子进程运行中",
      parent: { pid: 1234, state: "running", code: "// 父进程正常运行" },
      child: { pid: 1235, state: "running", code: "// 子进程正常运行" },
      description: "父子进程都在正常运行"
    },
    {
      step: 1,
      title: "子进程调用 exit(0)",
      parent: { pid: 1234, state: "running", code: "// 父进程继续运行\n// 但未调用 wait()" },
      child: { pid: 1235, state: "exiting", code: "exit(0);  // 子进程退出" },
      description: "子进程终止，释放内存但保留 PCB"
    },
    {
      step: 2,
      title: "子进程变成僵尸",
      parent: { pid: 1234, state: "running", code: "sleep(60);  // 不调用 wait()" },
      child: { pid: 1235, state: "zombie", code: "// 僵尸状态 (Z)\n// PCB 仍存在" },
      description: "子进程成为僵尸进程，占用 PID 资源"
    },
    {
      step: 3,
      title: "父进程调用 wait()",
      parent: { pid: 1234, state: "running", code: "wait(&status);  // 回收子进程" },
      child: { pid: 1235, state: "cleaned", code: "// PCB 被删除\n// PID 被释放" },
      description: "父进程回收子进程，僵尸消失"
    }
  ];

  const orphanSteps = [
    {
      step: 0,
      title: "父进程和子进程运行中",
      parent: { pid: 1234, state: "running", code: "// 父进程正常运行" },
      child: { pid: 1235, state: "running", code: "sleep(10);  // 子进程睡眠" },
      init: { visible: false },
      description: "父子进程都在正常运行"
    },
    {
      step: 1,
      title: "父进程先退出",
      parent: { pid: 1234, state: "exiting", code: "exit(0);  // 父进程退出" },
      child: { pid: 1235, state: "running", code: "sleep(10);  // 子进程仍在运行" },
      init: { visible: false },
      description: "父进程终止，子进程成为孤儿"
    },
    {
      step: 2,
      title: "子进程被 init 收养",
      parent: { pid: 1234, state: "terminated", code: "// 父进程已终止" },
      child: { pid: 1235, state: "running", code: "// 父进程变为 init (PID 1)\ngetppid();  // 返回 1" },
      init: { visible: true, pid: 1, state: "managing" },
      description: "init 进程成为孤儿进程的新父进程"
    },
    {
      step: 3,
      title: "子进程终止，init 自动回收",
      parent: { pid: 1234, state: "terminated", code: "// 父进程已终止" },
      child: { pid: 1235, state: "exiting", code: "exit(0);  // 子进程退出" },
      init: { visible: true, pid: 1, state: "reaping", code: "wait(&status);  // 自动回收" },
      description: "init 进程自动调用 wait() 回收孤儿进程"
    }
  ];

  const currentSteps = mode === "zombie" ? zombieSteps : orphanSteps;
  const currentData: any = currentSteps[step];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center">
        僵尸进程 vs 孤儿进程演示
      </h3>

      {/* Mode Toggle */}
      <div className="flex justify-center mb-6 gap-4">
        <button
          onClick={() => { setMode("zombie"); setStep(0); }}
          className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${
            mode === "zombie"
              ? "bg-red-600 text-white shadow-lg scale-105"
              : "bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Skull className="w-5 h-5" />
          僵尸进程（Zombie）
        </button>
        <button
          onClick={() => { setMode("orphan"); setStep(0); }}
          className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${
            mode === "orphan"
              ? "bg-blue-600 text-white shadow-lg scale-105"
              : "bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Baby className="w-5 h-5" />
          孤儿进程（Orphan）
        </button>
      </div>

      {/* Step Controls */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          上一步
        </button>
        <div className="px-6 py-2 bg-white rounded-lg border-2 border-slate-300 font-semibold">
          步骤 {step + 1} / {currentSteps.length}
        </div>
        <button
          onClick={() => setStep(Math.min(currentSteps.length - 1, step + 1))}
          disabled={step === currentSteps.length - 1}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          下一步
        </button>
      </div>

      {/* Visualization */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-slate-800 mb-4">{currentData.title}</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Parent Process */}
          <motion.div
            key={`parent-${step}`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: currentData.parent.state === "terminated" ? 0.3 : 1, scale: 1 }}
            className={`p-4 rounded-lg border-2 ${
              currentData.parent.state === "running" ? "bg-green-100 border-green-400" :
              currentData.parent.state === "exiting" ? "bg-yellow-100 border-yellow-400" :
              "bg-gray-100 border-gray-400"
            }`}
          >
            <div className="font-bold text-slate-800 mb-2">
              父进程 (PID {currentData.parent.pid})
            </div>
            <div className="text-xs text-slate-600 mb-2">
              状态: {currentData.parent.state}
            </div>
            <pre className="text-xs bg-slate-900 text-green-400 p-2 rounded overflow-x-auto">
              {currentData.parent.code}
            </pre>
          </motion.div>

          {/* Child Process */}
          <motion.div
            key={`child-${step}`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`p-4 rounded-lg border-2 ${
              currentData.child.state === "running" ? "bg-blue-100 border-blue-400" :
              currentData.child.state === "zombie" ? "bg-red-100 border-red-400" :
              currentData.child.state === "exiting" ? "bg-yellow-100 border-yellow-400" :
              "bg-gray-100 border-gray-400"
            }`}
          >
            <div className="font-bold text-slate-800 mb-2 flex items-center gap-2">
              子进程 (PID {currentData.child.pid})
              {currentData.child.state === "zombie" && (
                <Skull className="w-4 h-4 text-red-600" />
              )}
            </div>
            <div className="text-xs text-slate-600 mb-2">
              状态: {currentData.child.state}
            </div>
            <pre className="text-xs bg-slate-900 text-green-400 p-2 rounded overflow-x-auto">
              {currentData.child.code}
            </pre>
          </motion.div>

          {/* Init Process (orphan mode only) */}
          {mode === "orphan" && currentData.init?.visible && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="p-4 rounded-lg border-2 bg-purple-100 border-purple-400"
            >
              <div className="font-bold text-slate-800 mb-2">
                init 进程 (PID 1)
              </div>
              <div className="text-xs text-slate-600 mb-2">
                状态: {currentData.init.state}
              </div>
              {currentData.init.code && (
                <pre className="text-xs bg-slate-900 text-green-400 p-2 rounded overflow-x-auto">
                  {currentData.init.code}
                </pre>
              )}
            </motion.div>
          )}
        </div>
      </div>

      {/* Description */}
      <div className={`p-4 rounded-lg border-l-4 ${
        mode === "zombie" ? "bg-red-50 border-red-400" : "bg-blue-50 border-blue-400"
      }`}>
        <div className="flex items-center gap-2 mb-2">
          <Info className="w-5 h-5" />
          <h5 className="font-bold text-slate-800">当前状态说明</h5>
        </div>
        <p className="text-sm text-slate-700">{currentData.description}</p>
      </div>

      {/* Warning/Info Box */}
      <div className={`mt-6 p-4 rounded-lg border-l-4 ${
        mode === "zombie"
          ? "bg-orange-50 border-orange-400"
          : "bg-green-50 border-green-400"
      }`}>
        {mode === "zombie" ? (
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-slate-700 font-semibold mb-1">僵尸进程的危害</p>
              <ul className="text-sm text-slate-600 space-y-1">
                <li>• 占用 PID 资源（系统 PID 有限）</li>
                <li>• 大量僵尸进程可能耗尽 PID</li>
                <li>• 修复方法：父进程调用 wait() 或使用 SIGCHLD 信号处理器</li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="flex items-start gap-2">
            <Info className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-slate-700 font-semibold mb-1">孤儿进程的处理</p>
              <ul className="text-sm text-slate-600 space-y-1">
                <li>• init 进程周期性调用 wait() 回收所有孤儿进程</li>
                <li>• 孤儿进程不会产生僵尸进程</li>
                <li>• 这是 Unix/Linux 的正常机制，无害</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
