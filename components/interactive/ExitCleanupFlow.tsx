"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { XCircle, Play, Pause } from "lucide-react";

export default function ExitCleanupFlow() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const steps = [
    {
      title: "进程调用 exit(status)",
      description: "进程调用 exit() 或 main() 返回，传入退出码",
      actions: ["调用 exit(123)"],
      state: "运行中 → 退出流程",
      color: "blue"
    },
    {
      title: "执行 atexit() 注册的清理函数",
      description: "按注册逆序执行 atexit() 回调函数",
      actions: ["执行清理函数1", "执行清理函数2", "执行清理函数3"],
      state: "清理用户资源",
      color: "purple"
    },
    {
      title: "刷新并关闭标准 I/O 流",
      description: "刷新 stdout/stderr 缓冲区，关闭所有打开的 FILE* 流",
      actions: ["fflush(stdout)", "fclose() 所有 FILE*"],
      state: "清理 I/O 缓冲",
      color: "green"
    },
    {
      title: "进入内核态：sys_exit()",
      description: "陷入内核，执行系统调用 exit()",
      actions: ["trap → sys_exit()"],
      state: "用户态 → 内核态",
      color: "orange"
    },
    {
      title: "关闭文件描述符",
      description: "关闭所有打开的文件描述符（0-NOFILE），引用计数 -1",
      actions: ["close(fd[0])", "close(fd[1])", "...", "close(fd[NOFILE-1])"],
      state: "释放文件资源",
      color: "red"
    },
    {
      title: "释放内存资源",
      description: "释放用户空间页表、物理页（代码段、数据段、堆、栈）",
      actions: ["uvmfree(pagetable)", "释放所有用户态物理页"],
      state: "释放内存",
      color: "yellow"
    },
    {
      title: "释放 inode（当前目录）",
      description: "释放当前工作目录的 inode 引用",
      actions: ["iput(cwd)"],
      state: "释放目录资源",
      color: "teal"
    },
    {
      title: "唤醒父进程",
      description: "唤醒可能在 wait() 中阻塞的父进程",
      actions: ["wakeup(parent)"],
      state: "通知父进程",
      color: "indigo"
    },
    {
      title: "将子进程过继给 init",
      description: "将所有子进程的 parent 指针改为 init（PID 1）",
      actions: ["for (child) { child->parent = init; }"],
      state: "过继子进程",
      color: "pink"
    },
    {
      title: "设置为僵尸状态",
      description: "设置进程状态为 ZOMBIE，保存退出码",
      actions: ["p->state = ZOMBIE", "p->xstate = 123"],
      state: "僵尸状态",
      color: "slate"
    },
    {
      title: "调度器切换",
      description: "调用 sched() 让出 CPU，永不返回",
      actions: ["sched()", "（不再返回）"],
      state: "永久阻塞",
      color: "gray"
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
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isPlaying, step]);

  const currentStep = steps[step];

  const getColorClass = (color: string) => {
    const map: Record<string, string> = {
      blue: "bg-blue-100 border-blue-400",
      purple: "bg-purple-100 border-purple-400",
      green: "bg-green-100 border-green-400",
      orange: "bg-orange-100 border-orange-400",
      red: "bg-red-100 border-red-400",
      yellow: "bg-yellow-100 border-yellow-400",
      teal: "bg-teal-100 border-teal-400",
      indigo: "bg-indigo-100 border-indigo-400",
      pink: "bg-pink-100 border-pink-400",
      slate: "bg-slate-100 border-slate-400",
      gray: "bg-gray-100 border-gray-400"
    };
    return map[color] || "bg-slate-100 border-slate-400";
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <XCircle className="w-7 h-7 text-red-600" />
        exit() 清理流程
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
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700"
        >
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
            className="bg-red-600 h-2 rounded-full"
            animate={{ width: `${((step) / (steps.length - 1)) * 100}%` }}
          />
        </div>
      </div>

      {/* Flowchart */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex flex-col gap-3">
          {steps.map((s, idx) => (
            <React.Fragment key={idx}>
              <motion.div
                animate={{
                  scale: idx === step ? 1.02 : idx < step ? 0.98 : 1,
                  opacity: idx === step ? 1 : idx < step ? 0.6 : 0.4
                }}
                className={`p-4 rounded-lg border-2 transition-all ${
                  idx === step
                    ? getColorClass(s.color)
                    : idx < step
                    ? "bg-slate-100 border-slate-300"
                    : "bg-white border-slate-200"
                }`}
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${
                      idx === step
                        ? "bg-red-600"
                        : idx < step
                        ? "bg-green-600"
                        : "bg-slate-400"
                    }`}
                  >
                    {idx < step ? "✓" : idx + 1}
                  </div>
                  <div className="flex-1">
                    <div className="font-bold text-slate-800">{s.title}</div>
                    <div className="text-sm text-slate-600">{s.description}</div>
                    {idx === step && (
                      <div className="mt-2 space-y-1">
                        {s.actions.map((action, aidx) => (
                          <motion.div
                            key={aidx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: aidx * 0.2 }}
                            className="text-xs font-mono bg-white bg-opacity-60 px-2 py-1 rounded"
                          >
                            • {action}
                          </motion.div>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className={`text-sm font-semibold px-3 py-1 rounded ${
                    idx === step ? "bg-white bg-opacity-60" : ""
                  }`}>
                    {idx === step ? s.state : ""}
                  </div>
                </div>
              </motion.div>

              {idx < steps.length - 1 && (
                <div className="flex justify-center">
                  <div className={`w-1 h-6 transition-all ${
                    idx < step ? "bg-green-400" : "bg-slate-300"
                  }`}></div>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Current Step Detail */}
      <motion.div
        key={step}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`p-6 rounded-lg border-2 ${getColorClass(currentStep.color)}`}
      >
        <h4 className="font-bold text-slate-800 mb-2">{currentStep.title}</h4>
        <p className="text-sm text-slate-700 mb-3">{currentStep.description}</p>
        <div className="bg-white bg-opacity-60 p-3 rounded">
          <div className="text-xs font-semibold text-slate-600 mb-1">当前状态</div>
          <div className="font-bold text-slate-800">{currentStep.state}</div>
        </div>
      </motion.div>

      {/* xv6 Code */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-4">
        <h4 className="font-bold text-slate-800 mb-3">xv6 exit() 源码（简化）</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto">
{`// xv6-riscv/kernel/proc.c
void exit(int status) {
  struct proc *p = myproc();

  // 1. 关闭所有文件描述符
  for (int fd = 0; fd < NOFILE; fd++) {
    if (p->ofile[fd]) {
      struct file *f = p->ofile[fd];
      fileclose(f);       // 引用计数 -1
      p->ofile[fd] = 0;
    }
  }

  // 2. 释放当前目录 inode
  iput(p->cwd);
  p->cwd = 0;

  // 3. 唤醒父进程（可能在 wait() 中阻塞）
  wakeup(p->parent);

  // 4. 将子进程过继给 init
  for (struct proc *np = proc; np < &proc[NPROC]; np++) {
    if (np->parent == p) {
      np->parent = initproc;
      wakeup(initproc);  // 唤醒 init
    }
  }

  // 5. 设置为僵尸状态，保存退出码
  p->xstate = status;
  p->state = ZOMBIE;

  // 6. 让出 CPU，永不返回
  sched();  // 切换到调度器
  panic("zombie exit");  // 不应执行到这里
}`}
        </pre>
      </div>

      {/* Key Points */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-3">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
          <li>
            <strong>exit() 不返回</strong>：exit() 最终调用 sched() 让出 CPU，进程永久变为僵尸状态，不会返回用户态。
          </li>
          <li>
            <strong>资源分阶段释放</strong>：用户态先释放部分资源（atexit、I/O），
            内核态释放文件、内存，但 PCB 保留给父进程（存储退出码）。
          </li>
          <li>
            <strong>父进程负责最终回收</strong>：僵尸进程的 PCB、PID、退出码由父进程调用 wait() 释放。
          </li>
          <li>
            <strong>子进程过继给 init</strong>：防止父进程先退出导致子进程变孤儿无人回收，
            init 会定期调用 wait() 回收孤儿进程。
          </li>
          <li>
            <strong>exit() vs _exit()</strong>：exit() 是 C 库函数（执行 atexit、刷新 I/O），
            _exit() 是系统调用（直接进入内核）。
          </li>
        </ul>
      </div>
    </div>
  );
}
