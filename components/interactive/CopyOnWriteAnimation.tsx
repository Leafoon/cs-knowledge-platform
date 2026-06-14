"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Edit3, AlertTriangle } from "lucide-react";

export default function CopyOnWriteAnimation() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const steps = [
    {
      title: "fork() 时：页表共享",
      description: "子进程的页表指向与父进程相同的物理页，所有可写页标记为只读",
      parentPage: { physical: "0x1000", writable: false, refCount: 2 },
      childPage: { physical: "0x1000", writable: false, refCount: 2 },
      shared: true,
      faulted: false
    },
    {
      title: "子进程尝试写入",
      description: "子进程执行 *p = 200，CPU 检测到页表项为只读，触发缺页异常（Page Fault）",
      parentPage: { physical: "0x1000", writable: false, refCount: 2 },
      childPage: { physical: "0x1000", writable: false, refCount: 2 },
      shared: true,
      faulted: true
    },
    {
      title: "内核分配新页并复制",
      description: "内核分配新物理页 0x2000，复制原页内容（*p = 100）",
      parentPage: { physical: "0x1000", writable: false, refCount: 2 },
      childPage: { physical: "0x2000 (新分配)", writable: false, refCount: 1 },
      shared: false,
      copying: true,
      faulted: false
    },
    {
      title: "更新页表并恢复可写",
      description: "子进程页表指向新页 0x2000，恢复可写权限，原页引用计数 -1",
      parentPage: { physical: "0x1000", writable: false, refCount: 1 },
      childPage: { physical: "0x2000", writable: true, refCount: 1 },
      shared: false,
      copying: false,
      faulted: false
    },
    {
      title: "写入成功",
      description: "重新执行写指令 *p = 200，写入成功，父子进程各有独立的页",
      parentPage: { physical: "0x1000 (*p = 100)", writable: false, refCount: 1 },
      childPage: { physical: "0x2000 (*p = 200)", writable: true, refCount: 1 },
      shared: false,
      copying: false,
      faulted: false,
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

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Edit3 className="w-7 h-7 text-green-600" />
        写时复制（COW）动画演示
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
          <RotateCcw className="w-5 h-5" />
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
            className="bg-blue-600 h-2 rounded-full"
            animate={{ width: `${((step) / (steps.length - 1)) * 100}%` }}
          />
        </div>
      </div>

      {/* Animation */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-slate-800 mb-2">{currentStep.title}</h4>
        <p className="text-sm text-slate-600 mb-6">{currentStep.description}</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Parent Process */}
          <motion.div
            animate={{ scale: currentStep.shared ? 1 : 0.95 }}
            className={`p-4 rounded-lg border-2 transition-all ${
              currentStep.shared ? "bg-yellow-50 border-yellow-400" : "bg-green-50 border-green-400"
            }`}
          >
            <h5 className="font-bold text-green-700 mb-3">父进程 (PID 1234)</h5>
            <div className="space-y-2">
              <div className="bg-white p-3 rounded border border-slate-200">
                <div className="text-xs text-slate-600">虚拟地址</div>
                <div className="font-mono text-sm">0x600000</div>
              </div>
              <div className={`p-3 rounded border-2 ${
                currentStep.parentPage.writable ? "bg-green-100 border-green-400" : "bg-red-100 border-red-400"
              }`}>
                <div className="text-xs text-slate-600">物理页</div>
                <div className="font-mono text-sm font-bold">{currentStep.parentPage.physical}</div>
                <div className="text-xs mt-1">
                  {currentStep.parentPage.writable ? "可写" : "只读 (COW 标志)"}
                </div>
                <div className="text-xs">引用计数: {currentStep.parentPage.refCount}</div>
              </div>
            </div>
          </motion.div>

          {/* Child Process */}
          <motion.div
            animate={{ scale: currentStep.faulted ? 1.05 : currentStep.copying ? 1.05 : 1 }}
            className={`p-4 rounded-lg border-2 transition-all ${
              currentStep.faulted ? "bg-red-100 border-red-600 ring-4 ring-red-300" :
              currentStep.copying ? "bg-yellow-100 border-yellow-600" :
              currentStep.shared ? "bg-yellow-50 border-yellow-400" :
              "bg-purple-50 border-purple-400"
            }`}
          >
            <h5 className="font-bold text-purple-700 mb-3 flex items-center gap-2">
              子进程 (PID 1235)
              {currentStep.faulted && <AlertTriangle className="w-4 h-4 text-red-600 animate-pulse" />}
            </h5>
            <div className="space-y-2">
              <div className="bg-white p-3 rounded border border-slate-200">
                <div className="text-xs text-slate-600">虚拟地址</div>
                <div className="font-mono text-sm">0x600000</div>
              </div>
              <div className={`p-3 rounded border-2 ${
                currentStep.childPage.writable ? "bg-green-100 border-green-400" : "bg-red-100 border-red-400"
              }`}>
                <div className="text-xs text-slate-600">物理页</div>
                <div className="font-mono text-sm font-bold">{currentStep.childPage.physical}</div>
                <div className="text-xs mt-1">
                  {currentStep.childPage.writable ? "可写" : "只读 (COW 标志)"}
                </div>
                <div className="text-xs">引用计数: {currentStep.childPage.refCount}</div>
              </div>
              {currentStep.faulted && (
                <div className="bg-red-200 p-2 rounded text-xs text-red-800 font-semibold">
                  ⚠️ Page Fault 异常！
                </div>
              )}
              {currentStep.copying && (
                <div className="bg-yellow-200 p-2 rounded text-xs text-yellow-800 font-semibold animate-pulse">
                  🔄 正在复制页内容...
                </div>
              )}
              {currentStep.completed && (
                <div className="bg-green-200 p-2 rounded text-xs text-green-800 font-semibold">
                  ✅ 写入成功！
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Shared Indicator */}
        {currentStep.shared && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 text-center"
          >
            <div className="inline-block bg-yellow-200 px-4 py-2 rounded-lg border-2 border-yellow-400">
              <div className="text-sm font-semibold text-yellow-800">
                ↔️ 父子进程共享同一物理页 (0x1000)
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Code Example */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <h4 className="font-bold text-slate-800 mb-3">伪代码流程</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto">
{`// fork() 时
void fork_cow() {
  for (each page in parent) {
    child.pagetable[vpn] = parent.pagetable[vpn];  // 共享物理页
    parent.pagetable[vpn].writable = false;        // 标记为只读
    child.pagetable[vpn].writable = false;
    page_refcount[pfn]++;                          // 引用计数 +1
  }
}

// 缺页异常处理
void page_fault_handler(addr) {
  pte = lookup_pte(addr);
  if (pte.cow_flag) {  // 写时复制页
    new_page = alloc_page();                // 分配新页
    copy_page(pte.pfn, new_page);           // 复制内容
    pte.pfn = new_page;                     // 更新页表
    pte.writable = true;                    // 恢复可写
    page_refcount[old_pfn]--;               // 旧页引用计数 -1
  }
}`}
        </pre>
      </div>
    </div>
  );
}
