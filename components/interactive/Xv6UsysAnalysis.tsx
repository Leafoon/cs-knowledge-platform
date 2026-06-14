"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { FileCode } from "lucide-react";

export default function Xv6UsysAnalysis() {
  const [selectedSyscall, setSelectedSyscall] = useState("fork");

  const syscalls = [
    { name: "fork", num: 1, asm: "movl $SYS_fork, %eax; int $T_SYSCALL; ret" },
    { name: "exit", num: 2, asm: "movl $SYS_exit, %eax; int $T_SYSCALL; ret" },
    { name: "wait", num: 3, asm: "movl $SYS_wait, %eax; int $T_SYSCALL; ret" },
    { name: "pipe", num: 4, asm: "movl $SYS_pipe, %eax; int $T_SYSCALL; ret" },
    { name: "read", num: 5, asm: "movl $SYS_read, %eax; int $T_SYSCALL; ret" },
    { name: "write", num: 16, asm: "movl $SYS_write, %eax; int $T_SYSCALL; ret" },
    { name: "exec", num: 7, asm: "movl $SYS_exec, %eax; int $T_SYSCALL; ret" },
    { name: "open", num: 15, asm: "movl $SYS_open, %eax; int $T_SYSCALL; ret" }
  ];

  const current = syscalls.find(s => s.name === selectedSyscall)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <FileCode className="w-7 h-7 text-cyan-600" />
        xv6 usys.S 系统调用入口分析
      </h3>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-lg mb-4">usys.S 宏定义</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto">
{`#define SYSCALL(name) \\
  .globl name; \\
  name: \\
    movl $SYS_##name, %eax; \\
    int $T_SYSCALL; \\
    ret

SYSCALL(fork)
SYSCALL(exit)
SYSCALL(wait)
// ... 其他系统调用`}
        </pre>
        <div className="mt-4 text-sm text-slate-700">
          <strong>作用</strong>：为每个系统调用生成汇编入口代码
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-6">
        {syscalls.map(sc => (
          <button key={sc.name} onClick={() => setSelectedSyscall(sc.name)} className={`px-3 py-2 rounded-lg font-semibold text-sm ${selectedSyscall === sc.name ? "bg-cyan-600 text-white" : "bg-slate-200"}`}>
            {sc.name}()
          </button>
        ))}
      </div>

      <motion.div key={selectedSyscall} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-xl mb-4">{current.name}() 系统调用</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="bg-blue-50 p-4 rounded border border-blue-200">
            <div className="text-xs text-slate-600 mb-2">系统调用编号</div>
            <div className="font-bold text-2xl text-blue-700">SYS_{current.name} = {current.num}</div>
          </div>
          <div className="bg-green-50 p-4 rounded border border-green-200">
            <div className="text-xs text-slate-600 mb-2">生成的汇编代码</div>
            <div className="font-mono text-xs text-slate-800 bg-white p-2 rounded">{current.asm}</div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-cyan-100 to-blue-100 p-6 rounded-lg border-2 border-cyan-300">
          <h5 className="font-bold text-cyan-900 mb-3">执行流程</h5>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <div className="bg-cyan-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">1</div>
              <div className="flex-1">
                <div className="font-semibold text-slate-800">设置系统调用编号</div>
                <div className="text-sm text-slate-600 font-mono">movl $SYS_{current.name}, %eax</div>
                <div className="text-xs text-slate-600 mt-1">将系统调用编号 {current.num} 放入 %eax 寄存器</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="bg-cyan-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">2</div>
              <div className="flex-1">
                <div className="font-semibold text-slate-800">触发软中断</div>
                <div className="text-sm text-slate-600 font-mono">int $T_SYSCALL (int $64)</div>
                <div className="text-xs text-slate-600 mt-1">触发 64 号中断，CPU 切换到内核态</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="bg-cyan-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">3</div>
              <div className="flex-1">
                <div className="font-semibold text-slate-800">内核处理</div>
                <div className="text-sm text-slate-600">跳转到 trap → syscall() → sys_{current.name}()</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="bg-cyan-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">4</div>
              <div className="flex-1">
                <div className="font-semibold text-slate-800">返回用户态</div>
                <div className="text-sm text-slate-600 font-mono">ret</div>
                <div className="text-xs text-slate-600 mt-1">返回值在 %eax 中</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h5 className="font-bold text-slate-800 mb-3">完整调用链</h5>
        <div className="flex items-center justify-center gap-2 text-sm flex-wrap">
          <div className="bg-blue-100 px-4 py-2 rounded font-semibold">用户代码 fork()</div>
          <div>→</div>
          <div className="bg-green-100 px-4 py-2 rounded font-semibold">usys.S: fork</div>
          <div>→</div>
          <div className="bg-purple-100 px-4 py-2 rounded font-semibold">int $64</div>
          <div>→</div>
          <div className="bg-orange-100 px-4 py-2 rounded font-semibold">trap.c: trap()</div>
          <div>→</div>
          <div className="bg-red-100 px-4 py-2 rounded font-semibold">syscall.c: syscall()</div>
          <div>→</div>
          <div className="bg-pink-100 px-4 py-2 rounded font-semibold">sysproc.c: sys_fork()</div>
        </div>
      </div>

      <div className="bg-cyan-50 border-l-4 border-cyan-400 p-4 rounded">
        <h5 className="font-bold text-cyan-800 mb-2">关键点</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>usys.S</strong> 是用户空间系统调用的汇编入口（xv6 特有）</li>
          <li><strong>%eax</strong> 用于传递系统调用编号和返回值</li>
          <li><strong>int $64</strong> 触发中断，切换到内核态（Ring 0）</li>
          <li>现代 x86-64 使用 <code>syscall</code> 指令，比 <code>int</code> 更快</li>
        </ul>
      </div>
    </div>
  );
}
