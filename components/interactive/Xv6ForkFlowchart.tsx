"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { GitBranch, Code, ChevronRight } from "lucide-react";

export default function Xv6ForkFlowchart() {
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  const steps = [
    {
      id: 1,
      title: "sys_fork()",
      description: "系统调用入口（usys.S → syscall.c → sysproc.c）",
      code: `int sys_fork(void) {
  return fork();  // 调用内核 fork()
}`,
      color: "blue"
    },
    {
      id: 2,
      title: "fork()",
      description: "分配新进程 PCB（从 proc[] 数组中找空闲项）",
      code: `int fork(void) {
  struct proc *np = allocproc();
  if (np == 0) return -1;  // 分配失败
  // ...
}`,
      color: "green"
    },
    {
      id: 3,
      title: "allocproc()",
      description: "初始化 PCB，设置状态为 EMBRYO，分配 PID、内核栈",
      code: `static struct proc* allocproc(void) {
  // 遍历 proc[] 找 UNUSED 项
  for (p = proc; p < &proc[NPROC]; p++) {
    if (p->state == UNUSED) {
      p->state = EMBRYO;
      p->pid = nextpid++;
      // 分配内核栈 + trapframe
      p->kstack = kalloc();
      p->trapframe = (struct trapframe*)kalloc();
      return p;
    }
  }
  return 0;
}`,
      color: "yellow"
    },
    {
      id: 4,
      title: "复制页表",
      description: "uvmcopy() 复制父进程页表到子进程（COW 实现需修改）",
      code: `// 复制父进程页表
if (uvmcopy(curproc->pagetable, np->pagetable,
            curproc->sz) < 0) {
  freeproc(np);
  return -1;
}
np->sz = curproc->sz;  // 复制进程大小`,
      color: "purple"
    },
    {
      id: 5,
      title: "复制文件描述符",
      description: "复制父进程打开的文件（引用计数 +1）",
      code: `// 复制父进程打开的文件
for (i = 0; i < NOFILE; i++) {
  if (curproc->ofile[i])
    np->ofile[i] = filedup(curproc->ofile[i]);
}
np->cwd = idup(curproc->cwd);  // 复制当前目录`,
      color: "orange"
    },
    {
      id: 6,
      title: "复制 trapframe",
      description: "复制父进程 trapframe（保存寄存器状态）",
      code: `// 复制父进程 trapframe
*np->trapframe = *curproc->trapframe;

// 关键：设置子进程返回值为 0
np->trapframe->a0 = 0;`,
      color: "red"
    },
    {
      id: 7,
      title: "设置子进程状态",
      description: "设置父进程指针、进程名，状态改为 RUNNABLE",
      code: `np->parent = curproc;
safestrcpy(np->name, curproc->name, sizeof(curproc->name));

pid = np->pid;

// 设置为就绪状态
np->state = RUNNABLE;`,
      color: "teal"
    },
    {
      id: 8,
      title: "返回子进程 PID",
      description: "父进程返回子进程 PID，子进程返回 0（通过 trapframe->a0）",
      code: `// 父进程返回子进程 PID
return pid;

// 子进程通过 trapframe->a0 = 0 返回 0`,
      color: "indigo"
    }
  ];

  const getColorClass = (color: string) => {
    const map: Record<string, string> = {
      blue: "bg-blue-100 border-blue-400 text-blue-800",
      green: "bg-green-100 border-green-400 text-green-800",
      yellow: "bg-yellow-100 border-yellow-400 text-yellow-800",
      purple: "bg-purple-100 border-purple-400 text-purple-800",
      orange: "bg-orange-100 border-orange-400 text-orange-800",
      red: "bg-red-100 border-red-400 text-red-800",
      teal: "bg-teal-100 border-teal-400 text-teal-800",
      indigo: "bg-indigo-100 border-indigo-400 text-indigo-800"
    };
    return map[color] || "bg-slate-100 border-slate-400 text-slate-800";
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <GitBranch className="w-7 h-7 text-indigo-600" />
        xv6 fork() 实现流程
      </h3>

      {/* Flowchart */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex flex-col gap-3">
          {steps.map((step, idx) => (
            <React.Fragment key={step.id}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedStep(step.id)}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedStep === step.id
                    ? getColorClass(step.color)
                    : "bg-slate-50 border-slate-200 hover:border-slate-300"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${
                      step.color === "blue" ? "bg-blue-600" :
                      step.color === "green" ? "bg-green-600" :
                      step.color === "yellow" ? "bg-yellow-600" :
                      step.color === "purple" ? "bg-purple-600" :
                      step.color === "orange" ? "bg-orange-600" :
                      step.color === "red" ? "bg-red-600" :
                      step.color === "teal" ? "bg-teal-600" :
                      "bg-indigo-600"
                    }`}>
                      {idx + 1}
                    </div>
                    <div>
                      <div className="font-bold text-slate-800">{step.title}</div>
                      <div className="text-sm text-slate-600">{step.description}</div>
                    </div>
                  </div>
                  <ChevronRight className={`w-5 h-5 transition-transform ${
                    selectedStep === step.id ? "rotate-90" : ""
                  }`} />
                </div>
              </motion.div>

              {idx < steps.length - 1 && (
                <div className="flex justify-center">
                  <div className="w-1 h-6 bg-slate-300"></div>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Code Detail */}
      {selectedStep !== null && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
            <Code className="w-5 h-5" />
            {steps.find(s => s.id === selectedStep)?.title} - 源码
          </h4>
          <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
            {steps.find(s => s.id === selectedStep)?.code}
          </pre>
        </motion.div>
      )}

      {/* Key Points */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-3">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
          <li>
            <strong>allocproc()：</strong>从全局 proc[] 数组中找到一个 UNUSED 的 PCB，
            设置为 EMBRYO 状态，分配 PID、内核栈、trapframe。
          </li>
          <li>
            <strong>uvmcopy()：</strong>复制父进程页表到子进程。xv6 默认是完全复制（eager copy），
            COW 需要修改此函数（标记页表项为只读，设置 COW 标志）。
          </li>
          <li>
            <strong>子进程返回值为 0：</strong>通过设置 <code>np-&gt;trapframe-&gt;a0 = 0</code> 实现。
            trapframe 保存了用户态寄存器，a0 是 RISC-V 的返回值寄存器，
            子进程从内核返回用户态时会恢复 a0 = 0。
          </li>
          <li>
            <strong>文件描述符共享：</strong>通过 filedup() 增加文件引用计数，
            父子进程共享文件表项（包括文件偏移）。
          </li>
          <li>
            <strong>状态转换：</strong>UNUSED → EMBRYO（allocproc）→ RUNNABLE（fork 结束），
            子进程随后可被调度器选中运行。
          </li>
        </ul>
      </div>

      {/* Full Code Reference */}
      <div className="mt-6 bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
        <h4 className="font-bold text-blue-800 mb-2">完整源码参考</h4>
        <p className="text-sm text-slate-700 mb-2">
          xv6-riscv 完整 fork() 实现位于 <code>kernel/proc.c</code>：
        </p>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto">
{`// xv6-riscv/kernel/proc.c
int fork(void) {
  int i, pid;
  struct proc *np;
  struct proc *p = myproc();

  // 1. 分配新进程
  if ((np = allocproc()) == 0) return -1;

  // 2. 复制页表
  if (uvmcopy(p->pagetable, np->pagetable, p->sz) < 0) {
    freeproc(np);
    return -1;
  }
  np->sz = p->sz;

  // 3. 复制 trapframe（设置子进程返回值为 0）
  *np->trapframe = *p->trapframe;
  np->trapframe->a0 = 0;

  // 4. 复制文件描述符
  for (i = 0; i < NOFILE; i++)
    if (p->ofile[i])
      np->ofile[i] = filedup(p->ofile[i]);
  np->cwd = idup(p->cwd);

  // 5. 设置父进程、进程名
  safestrcpy(np->name, p->name, sizeof(p->name));
  np->parent = p;

  // 6. 设置为就绪状态
  np->state = RUNNABLE;
  pid = np->pid;

  return pid;  // 父进程返回子进程 PID
}`}
        </pre>
      </div>
    </div>
  );
}
