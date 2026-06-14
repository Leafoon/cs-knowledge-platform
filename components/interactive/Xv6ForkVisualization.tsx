"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { GitBranch } from "lucide-react";

export default function Xv6ForkVisualization() {
  const [step, setStep] = useState(0);

  const steps = [
    { title: "初始状态", parentState: "RUNNING", childState: null, desc: "父进程正在运行" },
    { title: "调用 fork()", parentState: "RUNNING", childState: null, desc: "父进程执行 fork() 系统调用" },
    { title: "分配子进程", parentState: "BLOCKED", childState: "EMBRYO", desc: "从 proc[] 数组分配新 PCB" },
    { title: "复制页表", parentState: "BLOCKED", childState: "EMBRYO", desc: "copyuvm() 复制父进程页表" },
    { title: "复制文件描述符", parentState: "BLOCKED", childState: "EMBRYO", desc: "复制 ofile[]，增加引用计数" },
    { title: "设置返回值", parentState: "BLOCKED", childState: "RUNNABLE", desc: "父进程返回子 PID，子进程返回 0" },
    { title: "fork 完成", parentState: "RUNNING", childState: "RUNNABLE", desc: "父子进程都可调度运行" }
  ];

  const current = steps[step];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-sky-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <GitBranch className="w-7 h-7 text-sky-600" />
        xv6 fork() 可视化
      </h3>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h4 className="font-bold text-xl text-slate-800">{current.title}</h4>
            <p className="text-sm text-slate-600 mt-1">{current.desc}</p>
          </div>
          <div className="text-2xl font-bold text-sky-600">步骤 {step + 1}/7</div>
        </div>

        <motion.div key={step} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* 父进程 */}
          <div className="bg-gradient-to-br from-blue-100 to-cyan-100 rounded-lg p-6 border-2 border-blue-300">
            <div className="text-center mb-4">
              <div className="text-xl font-bold text-blue-800 mb-2">父进程 (PID=1)</div>
              <div className={`inline-block px-6 py-3 rounded-lg font-bold text-white ${
                current.parentState === "RUNNING" ? "bg-green-600" :
                current.parentState === "BLOCKED" ? "bg-orange-600" :
                "bg-slate-400"
              }`}>
                {current.parentState}
              </div>
            </div>
            <div className="bg-white p-4 rounded border border-blue-200 space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-600">页表:</span>
                <span className="font-mono text-blue-700">0x80000</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">返回值:</span>
                <span className="font-mono text-green-700">{step >= 5 ? "2 (子 PID)" : "-"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">文件描述符:</span>
                <span className="font-mono text-blue-700">[0, 1, 2]</span>
              </div>
            </div>
          </div>

          {/* 子进程 */}
          <div className={`rounded-lg p-6 border-2 ${
            current.childState ? "bg-gradient-to-br from-green-100 to-emerald-100 border-green-300" : "bg-slate-100 border-slate-300 opacity-50"
          }`}>
            <div className="text-center mb-4">
              <div className="text-xl font-bold text-green-800 mb-2">子进程 (PID=2)</div>
              {current.childState ? (
                <div className={`inline-block px-6 py-3 rounded-lg font-bold text-white ${
                  current.childState === "EMBRYO" ? "bg-yellow-600" :
                  current.childState === "RUNNABLE" ? "bg-blue-600" :
                  current.childState === "RUNNING" ? "bg-green-600" :
                  "bg-slate-400"
                }`}>
                  {current.childState}
                </div>
              ) : (
                <div className="text-slate-500 font-semibold">未创建</div>
              )}
            </div>
            {current.childState && (
              <div className="bg-white p-4 rounded border border-green-200 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-600">页表:</span>
                  <span className="font-mono text-green-700">{step >= 3 ? "0x90000 (复制)" : "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">返回值:</span>
                  <span className="font-mono text-green-700">{step >= 5 ? "0" : "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">文件描述符:</span>
                  <span className="font-mono text-green-700">{step >= 4 ? "[0, 1, 2] (共享)" : "-"}</span>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        <div className="flex justify-center gap-4">
          <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} className="px-6 py-2 bg-slate-300 rounded-lg font-semibold disabled:opacity-50">上一步</button>
          <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step === steps.length - 1} className="px-6 py-2 bg-sky-600 text-white rounded-lg font-semibold disabled:opacity-50">下一步</button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h5 className="font-bold text-slate-800 mb-3">核心代码（proc.c: fork()）</h5>
        <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto">
{`int fork(void) {
  struct proc *np;
  struct proc *curproc = myproc();

  // 1. 分配新进程
  if ((np = allocproc()) == 0)
    return -1;

  // 2. 复制页表
  if ((np->pgdir = copyuvm(curproc->pgdir, curproc->sz)) == 0) {
    kfree(np->kstack);
    np->state = UNUSED;
    return -1;
  }
  np->sz = curproc->sz;
  np->parent = curproc;
  *np->tf = *curproc->tf;  // 复制 trapframe

  // 3. 子进程返回 0
  np->tf->eax = 0;

  // 4. 复制文件描述符
  for (int i = 0; i < NOFILE; i++)
    if (curproc->ofile[i])
      np->ofile[i] = filedup(curproc->ofile[i]);
  np->cwd = idup(curproc->cwd);

  // 5. 复制进程名
  safestrcpy(np->name, curproc->name, sizeof(curproc->name));

  int pid = np->pid;

  // 6. 设置为可运行
  acquire(&ptable.lock);
  np->state = RUNNABLE;
  release(&ptable.lock);

  return pid;  // 父进程返回子 PID
}`}
        </pre>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">关键数据结构</h5>
          <div className="space-y-2 text-sm">
            <div className="bg-blue-50 p-3 rounded border border-blue-200">
              <div className="font-semibold text-blue-800 mb-1">proc (PCB)</div>
              <div className="text-xs text-slate-600">包含 pid, state, pgdir, parent, ofile[], tf 等</div>
            </div>
            <div className="bg-green-50 p-3 rounded border border-green-200">
              <div className="font-semibold text-green-800 mb-1">pgdir (页表)</div>
              <div className="text-xs text-slate-600">虚拟地址 → 物理地址映射</div>
            </div>
            <div className="bg-purple-50 p-3 rounded border border-purple-200">
              <div className="font-semibold text-purple-800 mb-1">trapframe</div>
              <div className="text-xs text-slate-600">保存寄存器状态（包括返回值 eax）</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">进程状态</h5>
          <table className="w-full text-sm">
            <thead><tr className="border-b"><th className="text-left py-2">状态</th><th className="text-left py-2">含义</th></tr></thead>
            <tbody>
              <tr className="border-b"><td className="py-2 font-mono">UNUSED</td><td>PCB 空闲</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">EMBRYO</td><td>正在创建</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">RUNNABLE</td><td>就绪队列</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">RUNNING</td><td>正在运行</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">SLEEPING</td><td>等待事件</td></tr>
              <tr><td className="py-2 font-mono">ZOMBIE</td><td>已终止，等待父进程回收</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-sky-50 border-l-4 border-sky-400 p-4 rounded">
        <h5 className="font-bold text-sky-800 mb-2">关键点</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>写时复制（COW）</strong>：xv6 直接复制页表，现代 OS 使用 COW 延迟复制</li>
          <li><strong>返回值差异</strong>：父进程 fork() 返回子 PID，子进程返回 0</li>
          <li><strong>文件描述符共享</strong>：父子进程共享打开的文件（filedup 增加引用计数）</li>
          <li><strong>调度</strong>：子进程设为 RUNNABLE 后可被调度器选中运行</li>
          <li>xv6 最多 64 个进程（NPROC=64）</li>
        </ul>
      </div>
    </div>
  );
}
