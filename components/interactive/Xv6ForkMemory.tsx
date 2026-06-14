"use client";
import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, ChevronLeft, RotateCcw } from "lucide-react";

interface PageMapping {
  va: string;
  pa: string;
  flags: string;
  shared: boolean;
}

interface Step {
  title: string;
  description: string;
  parentPages: PageMapping[];
  childPages: PageMapping[];
  physicalPages: { id: string; owner: string; refcount: number }[];
  highlight: "parent" | "child" | "physical" | "all" | null;
}

const steps: Step[] = [
  {
    title: "fork 前：父进程地址空间",
    description: "父进程拥有独立的用户地址空间，包含代码段、数据段和栈。",
    parentPages: [
      { va: "0x0000", pa: "PA_0", flags: "R-X", shared: false },
      { va: "0x1000", pa: "PA_1", flags: "RW-", shared: false },
      { va: "0x2000", pa: "PA_2", flags: "RW-", shared: false },
    ],
    childPages: [],
    physicalPages: [
      { id: "PA_0", owner: "父进程", refcount: 1 },
      { id: "PA_1", owner: "父进程", refcount: 1 },
      { id: "PA_2", owner: "父进程", refcount: 1 },
    ],
    highlight: "parent",
  },
  {
    title: "fork：创建子进程页表",
    description: "allocproc 分配新的进程结构体，创建空的用户页表。",
    parentPages: [
      { va: "0x0000", pa: "PA_0", flags: "R-X", shared: false },
      { va: "0x1000", pa: "PA_1", flags: "RW-", shared: false },
      { va: "0x2000", pa: "PA_2", flags: "RW-", shared: false },
    ],
    childPages: [],
    physicalPages: [
      { id: "PA_0", owner: "父进程", refcount: 1 },
      { id: "PA_1", owner: "父进程", refcount: 1 },
      { id: "PA_2", owner: "父进程", refcount: 1 },
    ],
    highlight: "child",
  },
  {
    title: "uvmcopy：复制页表和物理页面",
    description: "遍历父进程页表，为每个页面分配新的物理页面，复制内容。",
    parentPages: [
      { va: "0x0000", pa: "PA_0", flags: "R-X", shared: false },
      { va: "0x1000", pa: "PA_1", flags: "RW-", shared: false },
      { va: "0x2000", pa: "PA_2", flags: "RW-", shared: false },
    ],
    childPages: [
      { va: "0x0000", pa: "PA_3", flags: "R-X", shared: false },
      { va: "0x1000", pa: "PA_4", flags: "RW-", shared: false },
      { va: "0x2000", pa: "PA_5", flags: "RW-", shared: false },
    ],
    physicalPages: [
      { id: "PA_0", owner: "父进程", refcount: 1 },
      { id: "PA_1", owner: "父进程", refcount: 1 },
      { id: "PA_2", owner: "父进程", refcount: 1 },
      { id: "PA_3", owner: "子进程", refcount: 1 },
      { id: "PA_4", owner: "子进程", refcount: 1 },
      { id: "PA_5", owner: "子进程", refcount: 1 },
    ],
    highlight: "physical",
  },
  {
    title: "fork 完成：父子进程独立",
    description: "fork 返回：父进程返回子进程 PID，子进程返回 0。两个进程拥有独立的物理页面副本。",
    parentPages: [
      { va: "0x0000", pa: "PA_0", flags: "R-X", shared: false },
      { va: "0x1000", pa: "PA_1", flags: "RW-", shared: false },
      { va: "0x2000", pa: "PA_2", flags: "RW-", shared: false },
    ],
    childPages: [
      { va: "0x0000", pa: "PA_3", flags: "R-X", shared: false },
      { va: "0x1000", pa: "PA_4", flags: "RW-", shared: false },
      { va: "0x2000", pa: "PA_5", flags: "RW-", shared: false },
    ],
    physicalPages: [
      { id: "PA_0", owner: "父进程", refcount: 1 },
      { id: "PA_1", owner: "父进程", refcount: 1 },
      { id: "PA_2", owner: "父进程", refcount: 1 },
      { id: "PA_3", owner: "子进程", refcount: 1 },
      { id: "PA_4", owner: "子进程", refcount: 1 },
      { id: "PA_5", owner: "子进程", refcount: 1 },
    ],
    highlight: "all",
  },
];

const cowSteps: Step[] = [
  {
    title: "COW fork：标记共享",
    description: "fork 时不复制物理页面，父子进程共享同一组物理页面，但标记为只读。",
    parentPages: [
      { va: "0x0000", pa: "PA_0", flags: "R--", shared: true },
      { va: "0x1000", pa: "PA_1", flags: "R--", shared: true },
      { va: "0x2000", pa: "PA_2", flags: "R--", shared: true },
    ],
    childPages: [
      { va: "0x0000", pa: "PA_0", flags: "R--", shared: true },
      { va: "0x1000", pa: "PA_1", flags: "R--", shared: true },
      { va: "0x2000", pa: "PA_2", flags: "R--", shared: true },
    ],
    physicalPages: [
      { id: "PA_0", owner: "共享", refcount: 2 },
      { id: "PA_1", owner: "共享", refcount: 2 },
      { id: "PA_2", owner: "共享", refcount: 2 },
    ],
    highlight: "all",
  },
  {
    title: "父进程写入 VA 0x1000 → 页故障",
    description: "父进程尝试写入只读页面，触发页故障。内核分配新页面 PA_3，复制 PA_1 的内容，更新父进程页表。",
    parentPages: [
      { va: "0x0000", pa: "PA_0", flags: "R--", shared: true },
      { va: "0x1000", pa: "PA_3", flags: "RW-", shared: false },
      { va: "0x2000", pa: "PA_2", flags: "R--", shared: true },
    ],
    childPages: [
      { va: "0x0000", pa: "PA_0", flags: "R--", shared: true },
      { va: "0x1000", pa: "PA_1", flags: "R--", shared: true },
      { va: "0x2000", pa: "PA_2", flags: "R--", shared: true },
    ],
    physicalPages: [
      { id: "PA_0", owner: "共享", refcount: 2 },
      { id: "PA_1", owner: "子进程", refcount: 1 },
      { id: "PA_2", owner: "共享", refcount: 2 },
      { id: "PA_3", owner: "父进程", refcount: 1 },
    ],
    highlight: "parent",
  },
  {
    title: "子进程写入 VA 0x0000 → 页故障",
    description: "子进程尝试写入只读的代码段页面，触发页故障。内核分配新页面 PA_4，复制 PA_0，PA_0 引用计数归零被释放。",
    parentPages: [
      { va: "0x0000", pa: "PA_0", flags: "R--", shared: true },
      { va: "0x1000", pa: "PA_3", flags: "RW-", shared: false },
      { va: "0x2000", pa: "PA_2", flags: "R--", shared: true },
    ],
    childPages: [
      { va: "0x0000", pa: "PA_4", flags: "RW-", shared: false },
      { va: "0x1000", pa: "PA_1", flags: "R--", shared: true },
      { va: "0x2000", pa: "PA_2", flags: "R--", shared: true },
    ],
    physicalPages: [
      { id: "PA_0", owner: "父进程", refcount: 1 },
      { id: "PA_1", owner: "共享", refcount: 1 },
      { id: "PA_2", owner: "共享", refcount: 2 },
      { id: "PA_3", owner: "父进程", refcount: 1 },
      { id: "PA_4", owner: "子进程", refcount: 1 },
    ],
    highlight: "child",
  },
];

export default function Xv6ForkMemory() {
  const [mode, setMode] = useState<"normal" | "cow">("normal");
  const [step, setStep] = useState(0);
  const currentSteps = mode === "normal" ? steps : cowSteps;
  const current = currentSteps[step];

  const canPrev = step > 0;
  const canNext = step < currentSteps.length - 1;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
        xv6 fork 内存复制过程
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
        逐步演示 fork 如何复制页表和物理页面
      </p>

      <div className="flex items-center gap-2 mb-4">
        <button
          onClick={() => { setMode("normal"); setStep(0); }}
          className={`px-3 py-1.5 text-sm rounded-lg font-medium transition-colors ${
            mode === "normal"
              ? "bg-blue-600 text-white"
              : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
          }`}
        >
          原始 fork
        </button>
        <button
          onClick={() => { setMode("cow"); setStep(0); }}
          className={`px-3 py-1.5 text-sm rounded-lg font-medium transition-colors ${
            mode === "cow"
              ? "bg-blue-600 text-white"
              : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
          }`}
        >
          写时复制 (COW)
        </button>
      </div>

      <div className="flex items-center gap-2 mb-4">
        <button
          onClick={() => setStep((s) => s - 1)}
          disabled={!canPrev}
          className="p-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40"
        >
          <ChevronLeft size={18} />
        </button>
        <span className="text-sm text-slate-500 dark:text-slate-400 font-mono">
          {step + 1} / {currentSteps.length}
        </span>
        <button
          onClick={() => setStep((s) => s + 1)}
          disabled={!canNext}
          className="p-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40"
        >
          <ChevronRight size={18} />
        </button>
        <button
          onClick={() => setStep(0)}
          className="p-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
        >
          <RotateCcw size={16} />
        </button>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={`${mode}-${step}`}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4 mb-4 border border-slate-200 dark:border-slate-700">
            <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-1">
              {current.title}
            </h4>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {current.description}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
              <h5 className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-2">
                父进程页表
              </h5>
              {current.parentPages.length === 0 ? (
                <div className="text-xs text-slate-400 text-center py-4">
                  无
                </div>
              ) : (
                <div className="space-y-1">
                  {current.parentPages.map((p) => (
                    <div
                      key={p.va}
                      className={`flex items-center justify-between text-xs font-mono px-2 py-1.5 rounded ${
                        p.shared
                          ? "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300"
                          : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                      } border border-slate-200 dark:border-slate-700`}
                    >
                      <span>VA {p.va}</span>
                      <span>→ {p.pa}</span>
                      <span className="text-[10px] px-1 rounded bg-slate-200 dark:bg-slate-700">
                        {p.flags}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 border border-green-200 dark:border-green-800">
              <h5 className="text-sm font-semibold text-green-700 dark:text-green-300 mb-2">
                子进程页表
              </h5>
              {current.childPages.length === 0 ? (
                <div className="text-xs text-slate-400 text-center py-4">
                  尚未创建
                </div>
              ) : (
                <div className="space-y-1">
                  {current.childPages.map((p) => (
                    <div
                      key={p.va}
                      className={`flex items-center justify-between text-xs font-mono px-2 py-1.5 rounded ${
                        p.shared
                          ? "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300"
                          : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                      } border border-slate-200 dark:border-slate-700`}
                    >
                      <span>VA {p.va}</span>
                      <span>→ {p.pa}</span>
                      <span className="text-[10px] px-1 rounded bg-slate-200 dark:bg-slate-700">
                        {p.flags}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
              <h5 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                物理页面
              </h5>
              <div className="space-y-1">
                {current.physicalPages.map((p) => (
                  <div
                    key={p.id}
                    className={`flex items-center justify-between text-xs font-mono px-2 py-1.5 rounded border ${
                      p.refcount > 1
                        ? "bg-yellow-100 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-700 text-yellow-700 dark:text-yellow-300"
                        : p.owner === "父进程"
                        ? "bg-blue-100 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300"
                        : p.owner === "子进程"
                        ? "bg-green-100 dark:bg-green-900/20 border-green-300 dark:border-green-700 text-green-700 dark:text-green-300"
                        : "bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300"
                    }`}
                  >
                    <span>{p.id}</span>
                    <span>{p.owner}</span>
                    <span className="text-[10px]">
                      ref:{p.refcount}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
