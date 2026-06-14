"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { RefreshCw, Play, Pause } from "lucide-react";

export default function ExecMemoryReplacement() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const steps = [
    {
      title: "exec() 调用前",
      description: "进程运行原程序（如 shell），内存包含代码、数据、堆、栈",
      memory: [
        { region: "内核空间", address: "0xFFFFFFFF", color: "red", retained: true },
        { region: "栈", address: "0x7FFF0000", content: "shell 栈", color: "purple", retained: false },
        { region: "堆", address: "0x700000", content: "shell malloc() 数据", color: "orange", retained: false },
        { region: "BSS", address: "0x602000", content: "shell 未初始化全局变量", color: "yellow", retained: false },
        { region: "数据段", address: "0x601000", content: "shell 全局变量", color: "green", retained: false },
        { region: "代码段", address: "0x400000", content: "shell 指令", color: "blue", retained: false }
      ],
      pid: 1234,
      programName: "shell"
    },
    {
      title: "内核检查权限与文件",
      description: "验证可执行文件存在、有执行权限、格式正确（ELF）",
      memory: [
        { region: "内核空间", address: "0xFFFFFFFF", color: "red", retained: true },
        { region: "栈", address: "0x7FFF0000", content: "shell 栈", color: "purple", retained: false, fading: true },
        { region: "堆", address: "0x700000", content: "shell malloc() 数据", color: "orange", retained: false, fading: true },
        { region: "BSS", address: "0x602000", content: "shell 未初始化全局变量", color: "yellow", retained: false, fading: true },
        { region: "数据段", address: "0x601000", content: "shell 全局变量", color: "green", retained: false, fading: true },
        { region: "代码段", address: "0x400000", content: "shell 指令", color: "blue", retained: false, fading: true }
      ],
      pid: 1234,
      programName: "shell → ls"
    },
    {
      title: "释放旧内存",
      description: "释放代码段、数据段、BSS、堆、栈的物理页（内核空间保留）",
      memory: [
        { region: "内核空间", address: "0xFFFFFFFF", color: "red", retained: true },
        { region: "（已释放）", address: "0x400000", content: "", color: "slate", retained: false, empty: true }
      ],
      pid: 1234,
      programName: "ls（内存已清空）"
    },
    {
      title: "加载新程序代码段",
      description: "从可执行文件（ls）加载代码段到内存",
      memory: [
        { region: "内核空间", address: "0xFFFFFFFF", color: "red", retained: true },
        { region: "代码段", address: "0x400000", content: "ls 指令（ELF 加载）", color: "blue", retained: false, newContent: true }
      ],
      pid: 1234,
      programName: "ls（代码段加载）"
    },
    {
      title: "加载数据段与 BSS",
      description: "加载初始化数据段，分配 BSS 段（未初始化全局变量，清零）",
      memory: [
        { region: "内核空间", address: "0xFFFFFFFF", color: "red", retained: true },
        { region: "BSS", address: "0x602000", content: "ls 未初始化全局变量（清零）", color: "yellow", retained: false, newContent: true },
        { region: "数据段", address: "0x601000", content: "ls 全局变量", color: "green", retained: false, newContent: true },
        { region: "代码段", address: "0x400000", content: "ls 指令", color: "blue", retained: false }
      ],
      pid: 1234,
      programName: "ls（数据段加载）"
    },
    {
      title: "重新分配堆与栈",
      description: "分配新栈，初始化堆（brk 指向数据段末尾）",
      memory: [
        { region: "内核空间", address: "0xFFFFFFFF", color: "red", retained: true },
        { region: "栈", address: "0x7FFF0000", content: "ls 栈（argc, argv, envp）", color: "purple", retained: false, newContent: true },
        { region: "堆", address: "0x700000", content: "（空，brk 初始化）", color: "orange", retained: false, newContent: true },
        { region: "BSS", address: "0x602000", content: "ls 未初始化全局变量", color: "yellow", retained: false },
        { region: "数据段", address: "0x601000", content: "ls 全局变量", color: "green", retained: false },
        { region: "代码段", address: "0x400000", content: "ls 指令", color: "blue", retained: false }
      ],
      pid: 1234,
      programName: "ls（堆栈初始化）"
    },
    {
      title: "exec() 成功返回",
      description: "跳转到新程序入口点（PC = entry point），进程继续运行（执行 ls）",
      memory: [
        { region: "内核空间", address: "0xFFFFFFFF", color: "red", retained: true },
        { region: "栈", address: "0x7FFF0000", content: "ls 栈", color: "purple", retained: false },
        { region: "堆", address: "0x700000", content: "ls 堆", color: "orange", retained: false },
        { region: "BSS", address: "0x602000", content: "ls 未初始化全局变量", color: "yellow", retained: false },
        { region: "数据段", address: "0x601000", content: "ls 全局变量", color: "green", retained: false },
        { region: "代码段", address: "0x400000", content: "ls 指令", color: "blue", retained: false }
      ],
      pid: 1234,
      programName: "ls（执行中）",
      completed: true
    }
  ];

  const handleNext = () => {
    if (step < steps.length - 1) {
      setStep(step + 1);
    }
  };

  const handlePrev = () => {
    if (step > 0) {
      setStep(step - 1);
    }
  };

  const handleReset = () => {
    setStep(0);
    setIsPlaying(false);
  };

  React.useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && step < steps.length - 1) {
      interval = setInterval(() => {
        setStep(prev => {
          if (prev < steps.length - 1) {
            return prev + 1;
          } else {
            setIsPlaying(false);
            return prev;
          }
        });
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, step]);

  const currentStep = steps[step];

  const getColorClass = (color: string) => {
    const map: Record<string, string> = {
      red: "bg-red-500 text-white",
      purple: "bg-purple-500 text-white",
      orange: "bg-orange-500 text-white",
      yellow: "bg-yellow-500 text-white",
      green: "bg-green-500 text-white",
      blue: "bg-blue-500 text-white",
      slate: "bg-slate-300 text-slate-600"
    };
    return map[color] || "bg-slate-400 text-white";
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <RefreshCw className="w-7 h-7 text-orange-600" />
        exec() 内存替换可视化
      </h3>

      {/* Controls */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={handlePrev}
          disabled={step === 0}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          上一步
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={`px-6 py-2 rounded-lg font-semibold text-white transition-all flex items-center gap-2 ${
            isPlaying ? "bg-orange-600 hover:bg-orange-700" : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {isPlaying ? (
            <>
              <Pause className="w-5 h-5" />
              暂停
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              自动播放
            </>
          )}
        </button>
        <button
          onClick={handleNext}
          disabled={step === steps.length - 1}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          下一步
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 flex items-center gap-2"
        >
          <RefreshCw className="w-5 h-5" />
          重置
        </button>
      </div>

      {/* Progress */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-slate-600 mb-2">
          <span>步骤 {step + 1} / {steps.length}</span>
          <span>{Math.round((step / (steps.length - 1)) * 100)}%</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-2">
          <motion.div
            className="bg-orange-600 h-2 rounded-full"
            animate={{ width: `${((step) / (steps.length - 1)) * 100}%` }}
          />
        </div>
      </div>

      {/* Memory Visualization */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h4 className="font-bold text-slate-800">{currentStep.title}</h4>
            <p className="text-sm text-slate-600">{currentStep.description}</p>
          </div>
          <div className="text-right">
            <div className="text-xs text-slate-600">进程</div>
            <div className="font-bold text-slate-800">{currentStep.programName}</div>
            <div className="text-xs text-slate-600">PID: {currentStep.pid}</div>
          </div>
        </div>

        {/* Memory Layout */}
        <div className="space-y-2">
          {currentStep.memory.map((region, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{
                opacity: (region as any).fading ? 0.3 : (region as any).empty ? 0.5 : 1,
                x: 0,
                scale: (region as any).newContent ? [1, 1.05, 1] : 1
              }}
              transition={{ duration: 0.5 }}
              className={`p-3 rounded-lg ${getColorClass(region.color)} ${
                (region as any).newContent ? "ring-4 ring-green-400" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="font-semibold">{region.region}</div>
                <div className="font-mono text-sm">{region.address}</div>
              </div>
              {region.content && (
                <div className="text-sm mt-1 opacity-90">{region.content}</div>
              )}
              {region.retained && (
                <div className="text-xs mt-1 bg-white bg-opacity-30 inline-block px-2 py-1 rounded">
                  保留（不变）
                </div>
              )}
              {(region as any).newContent && (
                <div className="text-xs mt-1 bg-green-600 inline-block px-2 py-1 rounded">
                  ✨ 新加载
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {currentStep.completed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 bg-green-100 border-2 border-green-400 p-3 rounded-lg text-center"
          >
            <div className="font-bold text-green-800">✅ exec() 成功！进程已完全替换为新程序</div>
          </motion.div>
        )}
      </div>

      {/* Key Points */}
      <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-3">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
          <li>
            <strong>PID 不变</strong>：exec() 不创建新进程，只替换当前进程的内存映像，PID、父进程、权限等保持不变。
          </li>
          <li>
            <strong>内核空间保留</strong>：内核空间（页表、PCB、内核栈）不受影响，只替换用户空间。
          </li>
          <li>
            <strong>文件描述符默认保留</strong>：除非设置 FD_CLOEXEC 标志（close-on-exec），否则打开的文件描述符会继承到新程序。
          </li>
          <li>
            <strong>exec() 不返回</strong>：如果成功，exec() 不会返回（因为代码段已被替换）；如果失败才返回 -1。
          </li>
          <li>
            <strong>Shell 的实现</strong>：Shell 执行外部命令时，先 fork() 创建子进程，子进程再 exec() 加载命令，
            父进程 wait() 等待子进程结束。
          </li>
        </ul>
      </div>
    </div>
  );
}
