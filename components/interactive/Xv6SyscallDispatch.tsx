"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { GitBranch } from "lucide-react";

export default function Xv6SyscallDispatch() {
  const [step, setStep] = useState(0);

  const steps = [
    { title: "用户调用 fork()", code: "fork();", layer: "user", desc: "用户程序调用 fork()" },
    { title: "usys.S 入口", code: "movl $SYS_fork, %eax\nint $64", layer: "usys", desc: "设置 %eax=1，触发中断" },
    { title: "trap() 接收中断", code: "void trap(struct trapframe *tf) {\n  if (tf->trapno == T_SYSCALL) {\n    syscall();\n  }\n}", layer: "trap", desc: "检测到系统调用中断" },
    { title: "syscall() 分发", code: "void syscall(void) {\n  int num = myproc()->tf->eax;\n  syscalls[num]();\n}", layer: "syscall", desc: "根据 %eax 查表调用" },
    { title: "sys_fork() 执行", code: "int sys_fork(void) {\n  return fork();\n}", layer: "sysproc", desc: "真正的 fork 实现" },
    { title: "返回用户态", code: "// %eax = 返回值\nret", layer: "return", desc: "返回值通过 %eax 传回" }
  ];

  const current = steps[step];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <GitBranch className="w-7 h-7 text-teal-600" />
        xv6 系统调用分发机制
      </h3>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h4 className="font-bold text-lg text-slate-800">{current.title}</h4>
            <p className="text-sm text-slate-600 mt-1">{current.desc}</p>
          </div>
          <div className="text-2xl font-bold text-teal-600">步骤 {step + 1}/6</div>
        </div>

        <motion.div key={step} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="bg-gradient-to-r from-teal-100 to-cyan-100 rounded-lg p-6 mb-6 border-2 border-teal-300">
          <div className="flex items-center justify-center mb-4">
            <div className={`px-6 py-3 rounded-lg font-bold text-white ${
              current.layer === "user" ? "bg-blue-600" :
              current.layer === "usys" ? "bg-green-600" :
              current.layer === "trap" ? "bg-purple-600" :
              current.layer === "syscall" ? "bg-orange-600" :
              current.layer === "sysproc" ? "bg-red-600" :
              "bg-teal-600"
            }`}>
              {current.layer === "user" ? "用户态 (Ring 3)" :
               current.layer === "return" ? "返回用户态" :
               "内核态 (Ring 0)"}
            </div>
          </div>
          <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto whitespace-pre-wrap">{current.code}</pre>
        </motion.div>

        <div className="flex justify-center gap-4">
          <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} className="px-6 py-2 bg-slate-300 rounded-lg font-semibold disabled:opacity-50">上一步</button>
          <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step === steps.length - 1} className="px-6 py-2 bg-teal-600 text-white rounded-lg font-semibold disabled:opacity-50">下一步</button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h5 className="font-bold text-slate-800 mb-4">系统调用表（syscalls[]）</h5>
        <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto">
{`static int (*syscalls[])(void) = {
[SYS_fork]    sys_fork,
[SYS_exit]    sys_exit,
[SYS_wait]    sys_wait,
[SYS_pipe]    sys_pipe,
[SYS_read]    sys_read,
[SYS_kill]    sys_kill,
[SYS_exec]    sys_exec,
[SYS_fstat]   sys_fstat,
[SYS_chdir]   sys_chdir,
[SYS_dup]     sys_dup,
[SYS_getpid]  sys_getpid,
[SYS_sbrk]    sys_sbrk,
[SYS_sleep]   sys_sleep,
[SYS_uptime]  sys_uptime,
[SYS_open]    sys_open,
[SYS_write]   sys_write,
[SYS_mknod]   sys_mknod,
[SYS_unlink]  sys_unlink,
[SYS_link]    sys_link,
[SYS_mkdir]   sys_mkdir,
[SYS_close]   sys_close,
};`}
        </pre>
        <div className="mt-4 text-sm text-slate-700">
          <strong>机制</strong>：根据 %eax 中的系统调用编号（如 SYS_fork=1）索引数组，调用对应函数指针
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">syscall() 核心代码</h5>
          <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs overflow-x-auto">
{`void syscall(void) {
  int num;
  struct proc *curproc = myproc();

  num = curproc->tf->eax;
  if(num > 0 && num < NELEM(syscalls) 
     && syscalls[num]) {
    curproc->tf->eax = syscalls[num]();
  } else {
    cprintf("unknown syscall %d\\n", num);
    curproc->tf->eax = -1;
  }
}`}
          </pre>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">关键步骤</h5>
          <div className="space-y-2 text-sm text-slate-700">
            <div className="flex items-start gap-2">
              <div className="bg-teal-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold flex-shrink-0">1</div>
              <div>从 trapframe 读取 %eax（系统调用编号）</div>
            </div>
            <div className="flex items-start gap-2">
              <div className="bg-teal-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold flex-shrink-0">2</div>
              <div>检查编号合法性</div>
            </div>
            <div className="flex items-start gap-2">
              <div className="bg-teal-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold flex-shrink-0">3</div>
              <div>调用 syscalls[num]()</div>
            </div>
            <div className="flex items-start gap-2">
              <div className="bg-teal-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold flex-shrink-0">4</div>
              <div>将返回值写回 trapframe{'->'}eax</div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-teal-50 border-l-4 border-teal-400 p-4 rounded">
        <h5 className="font-bold text-teal-800 mb-2">设计优势</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>表驱动</strong>：通过数组索引快速分发，O(1) 复杂度</li>
          <li><strong>可扩展</strong>：添加新系统调用只需扩展数组</li>
          <li><strong>类型安全</strong>：函数指针类型统一</li>
          <li><strong>简洁</strong>：xv6 核心设计哲学——简单即美</li>
        </ul>
      </div>
    </div>
  );
}
