"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  RotateCcw,
  ChevronRight,
  ChevronLeft,
  Terminal,
  Cpu,
  ArrowRight,
  Layers,
  Code,
  User,
  Zap,
} from "lucide-react";

interface Step {
  id: number;
  title: string;
  mode: string;
  icon: React.ReactNode;
  description: string;
  code: string;
  memoryState: string;
  cpuAction: string;
}

const STEPS: Step[] = [
  {
    id: 0,
    title: "userinit() 创建进程",
    mode: "S-mode",
    icon: <User className="w-5 h-5" />,
    description: "内核调用 userinit()，分配一个进程结构体（PID=1），创建用户页表，将 initcode 的机器码复制到用户地址空间的虚拟地址 0 处。",
    code: `struct proc *p = allocproc();   // 分配进程
initproc = p;                   // 保存为全局变量
p->pagetable = proc_pagetable(p); // 创建用户页表
p->sz = PGSIZE;                 // 进程大小 = 4KB
uvminit(p->pagetable, initcode, sizeof(initcode));
// ↑ 将 initcode 复制到用户地址 0x0`,
    memoryState: "用户地址空间：虚拟地址 0x0 包含 initcode 的机器码（ecall + exec 参数）",
    cpuAction: "CPU 仍在 S-mode，执行内核代码",
  },
  {
    id: 1,
    title: "设置 trapframe 寄存器",
    mode: "S-mode",
    icon: <Layers className="w-5 h-5" />,
    description: "内核在 trapframe 中设置用户态寄存器：epc=0（用户程序入口地址），sp=PGSIZE（用户栈顶）。这些值将在返回用户态时恢复到 CPU 寄存器。",
    code: `p->trapframe->epc = 0;     // 用户程序 PC = 0
p->trapframe->sp = PGSIZE; // 用户栈指针 = 4096

safestrcpy(p->name, "initcode", 16);
p->cwd = namei("/");       // 当前目录 = 根目录`,
    memoryState: "trapframe->epc = 0x0 (initcode 入口)\ntrapframe->sp = 0x1000 (用户栈顶)",
    cpuAction: "CPU 仍在 S-mode",
  },
  {
    id: 2,
    title: "进程标记为 RUNNABLE",
    mode: "S-mode",
    icon: <Zap className="w-5 h-5" />,
    description: "将进程状态设为 RUNNABLE，表示它可以被调度器选中执行。main() 继续执行其他初始化，最终调用 scheduler()。",
    code: `p->state = RUNNABLE;  // 进程可以被调度了
release(&p->lock);

// main() 继续执行...
started = 1;  // 通知其他 hart
scheduler();  // 进入调度器主循环（永不返回）`,
    memoryState: "进程 proc[1]: state = RUNNABLE, pid = 1",
    cpuAction: "CPU 调用 scheduler()，开始寻找 RUNNABLE 进程",
  },
  {
    id: 3,
    title: "scheduler() 选中 init 进程",
    mode: "S-mode",
    icon: <Cpu className="w-5 h-5" />,
    description: "调度器遍历进程表，找到唯一的 RUNNABLE 进程（init）。设置状态为 RUNNING，然后调用 swtch() 切换到该进程的内核上下文。",
    code: `for(p = proc; p < &proc[NPROC]; p++) {
  acquire(&p->lock);
  if(p->state == RUNNABLE) {
    p->state = RUNNING;
    c->proc = p;
    swtch(&c->context, &p->context);
    // ↑ 切换到 init 的内核上下文
    c->proc = 0;
  }
  release(&p->lock);
}`,
    memoryState: "proc[1].state: RUNNABLE → RUNNING\nproc[1].context: 被恢复到 CPU",
    cpuAction: "swtch() 保存调度器上下文，恢复 init 的内核上下文",
  },
  {
    id: 4,
    title: "usertrapret() 准备返回用户态",
    mode: "S-mode",
    icon: <ArrowRight className="w-5 h-5" />,
    description: "从 swtch() 返回后，进入 usertrapret()。该函数设置陷阱处理的下一跳：将 stvec 设为 uservec，配置 trapframe 中的内核信息，最后跳转到 trampoline 中的 userret。",
    code: `void usertrapret(void) {
  struct proc *p = myproc();
  w_stvec(TRAMPOLINE + (uservec - trampoline));
  p->trapframe->kernel_satp = r_satp();
  p->trapframe->kernel_sp = p->kstack + PGSIZE;
  p->trapframe->kernel_trap = (uint64)usertrap;
  p->trapframe->kernel_hartid = r_tp();
  w_sepc(p->trapframe->epc); // sepc = 0
  // ... 设置 sstatus, 跳转到 userret
}`,
    memoryState: "stvec = uservec (trampoline 页)\nsepc = 0 (用户程序入口)\ntrapframe 保存了内核页表、内核栈等信息",
    cpuAction: "CPU 准备切换到 U-mode",
  },
  {
    id: 5,
    title: "userret → sret → 用户态",
    mode: "S-mode → U-mode",
    icon: <Terminal className="w-5 h-5" />,
    description: "userret 汇编代码加载用户页表，恢复所有用户态寄存器，然后执行 sret 指令。sret 将 sepc 加载到 PC（=0），切换到 U-mode。CPU 开始执行 initcode。",
    code: `userret:
  # a0 = user page table
  csrw satp, a0       # 加载用户页表
  sfence.vma          # 刷新 TLB
  # 从 trapframe 恢复所有用户寄存器
  ld ra, 40(a1)
  ld sp, 48(a1)
  ld a0, 104(a1)
  # ... 恢复所有寄存器
  sret                # 返回用户态！PC = sepc = 0`,
    memoryState: "satp = 用户页表\nCPU 寄存器 = trapframe 中保存的值\nPC = 0 (initcode 开始执行)",
    cpuAction: "sret: S-mode → U-mode, PC = 0",
  },
  {
    id: 6,
    title: "initcode.S 执行 exec",
    mode: "U-mode",
    icon: <Code className="w-5 h-5" />,
    description: "initcode 是一个极简的汇编程序，只做一件事：调用 exec 系统调用，加载 /init 程序到当前进程的地址空间。",
    code: `# user/initcode.S
.globl start
start:
  la a0, init      # a0 = "/init" 字符串
  la a1, argv       # a1 = argv 数组
  li a7, SYS_exec   # 系统调用号 = exec
  ecall             # 触发系统调用！

init:
  .string "/init\\0"
argv:
  .long init        # argv[0] = "/init"
  .long 0           # argv[1] = NULL`,
    memoryState: "用户地址 0x0: initcode 的 ecall 指令\na0 = \"/init\" 字符串地址\na7 = SYS_exec (7)",
    cpuAction: "ecall: U-mode → S-mode (系统调用)",
  },
  {
    id: 7,
    title: "内核执行 sys_exec",
    mode: "S-mode",
    icon: <Layers className="w-5 h-5" />,
    description: "内核的 syscall() 分发到 sys_exec()。sys_exec 查找 /init 的 inode，读取 ELF 头部，分配新的页表，加载代码段和数据段到用户地址空间，替换旧的页表。",
    code: `// sys_exec → exec
// 1. namei("/init") → 获取 inode
// 2. readi(ip, &elf, 0, sizeof(elf))
// 3. 验证 ELF_MAGIC
// 4. 新页表 = proc_pagetable(p)
// 5. for each program header:
//      loadseg(新页表, vaddr, ip, off, filesz)
// 6. 设置用户栈（2 页：栈 + guard page）
// 7. p->pagetable = 新页表
// 8. p->trapframe->epc = elf.entry
// 9. 释放旧页表`,
    memoryState: "旧页表被释放\n新页表：/init 的代码和数据已加载\ntrapframe->epc = /init 入口地址",
    cpuAction: "内核完成 exec，准备返回用户态",
  },
  {
    id: 8,
    title: "init.c 开始执行",
    mode: "U-mode",
    icon: <Terminal className="w-5 h-5" />,
    description: "/init（init.c）开始执行。它打开控制台设备作为 stdin/stdout/stderr，然后 fork 子进程来执行 shell。",
    code: `// user/init.c
char *argv[] = { "sh", 0 };
int main(void) {
  int pid;
  // 打开控制台设备
  open("console", O_RDWR);  // fd 0 = stdin
  dup(0);                     // fd 1 = stdout
  dup(0);                     // fd 2 = stderr
  // 循环：如果 shell 退出，重启
  for(;;) {
    printf("init: starting sh\\n");
    pid = fork();
    if(pid == 0) {
      exec("sh", argv);  // 子进程执行 shell
    }
    wait(0);  // 父进程等待
  }
}`,
    memoryState: "fd 0 → console (stdin)\nfd 1 → console (stdout)\nfd 2 → console (stderr)",
    cpuAction: "fork() 创建子进程，子进程 exec(\"sh\")",
  },
  {
    id: 9,
    title: "shell 启动！",
    mode: "U-mode",
    icon: <Terminal className="w-5 h-5" />,
    description: "shell（sh.c）启动，显示提示符 xv6$，等待用户输入命令。至此，xv6 的启动流程全部完成！",
    code: `// user/sh.c
int main(void) {
  // 读取并执行命令
  while(getcmd(buf, sizeof(buf)) >= 0) {
    if(buf[0] == 'c' && buf[1] == 'd') { ... }
    if(fork() == 0)
      runcmd(parsecmd(buf));
    wait(0);
  }
}

// 用户看到：
// xv6$ _`,
    memoryState: "init 进程 (PID 1): wait() 等待 shell\nshell 进程 (PID 2): 读取用户输入",
    cpuAction: "shell 调用 read() 等待键盘输入",
  },
];

export default function Xv6FirstProcess() {
  const [current, setCurrent] = useState(0);
  const [autoPlaying, setAutoPlaying] = useState(false);

  const step = STEPS[current];

  const next = useCallback(() => setCurrent((c) => Math.min(c + 1, STEPS.length - 1)), []);
  const prev = useCallback(() => setCurrent((c) => Math.max(c - 1, 0)), []);
  const reset = useCallback(() => { setCurrent(0); setAutoPlaying(false); }, []);

  useEffect(() => {
    if (!autoPlaying) return;
    const timer = setInterval(() => {
      setCurrent((c) => {
        if (c >= STEPS.length - 1) { setAutoPlaying(false); return c; }
        return c + 1;
      });
    }, 2000);
    return () => clearInterval(timer);
  }, [autoPlaying]);

  const togglePlay = () => {
    if (autoPlaying) { setAutoPlaying(false); }
    else {
      if (current >= STEPS.length - 1) setCurrent(0);
      setAutoPlaying(true);
    }
  };

  const getModeColor = (mode: string) => {
    if (mode.includes("U-mode")) return "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300 border-green-300 dark:border-green-700";
    if (mode.includes("S-mode")) return "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300 border-blue-300 dark:border-blue-700";
    return "bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600";
  };

  return (
    <div className="max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-1">
        第一个进程的创建与启动
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        从 userinit() 到 shell 提示符的完整过程
      </p>

      <div className="flex items-center gap-2 mb-4">
        <button onClick={togglePlay} className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-violet-500 text-white hover:bg-violet-600 transition-all flex items-center gap-1">
          {autoPlaying ? <><Pause className="w-3 h-3" /> 暂停</> : <><Play className="w-3 h-3" /> 播放</>}
        </button>
        <button onClick={prev} disabled={current === 0} className="px-2 py-1.5 text-xs rounded-lg bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-300 dark:hover:bg-gray-600 transition-all">
          <ChevronLeft className="w-4 h-4" />
        </button>
        <span className="text-xs font-mono text-slate-500 dark:text-slate-400 min-w-[60px] text-center">
          {current + 1} / {STEPS.length}
        </span>
        <button onClick={next} disabled={current === STEPS.length - 1} className="px-2 py-1.5 text-xs rounded-lg bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-300 dark:hover:bg-gray-600 transition-all">
          <ChevronRight className="w-4 h-4" />
        </button>
        <button onClick={reset} className="px-2 py-1.5 text-xs rounded-lg bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-gray-600 transition-all">
          <RotateCcw className="w-3 h-3" />
        </button>

        <div className="ml-auto flex gap-1">
          {STEPS.map((_, i) => (
            <button
              key={i}
              onClick={() => { setCurrent(i); setAutoPlaying(false); }}
              className={`w-2.5 h-2.5 rounded-full transition-all ${
                i === current
                  ? "bg-violet-500 scale-125"
                  : i < current
                    ? "bg-violet-300 dark:bg-violet-600"
                    : "bg-slate-200 dark:bg-gray-600"
              }`}
            />
          ))}
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={step.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          <div className="bg-white dark:bg-gray-700 rounded-xl border border-slate-200 dark:border-gray-600 overflow-hidden shadow-sm">
            <div className="p-4 border-b border-slate-100 dark:border-gray-600">
              <div className="flex items-center gap-2 mb-2">
                <span className="p-1.5 rounded-lg bg-violet-100 dark:bg-violet-900/40 text-violet-600 dark:text-violet-400">
                  {step.icon}
                </span>
                <span className="text-xs font-mono text-slate-400 dark:text-slate-500">
                  Step {step.id + 1}/{STEPS.length}
                </span>
                <span className={`ml-auto text-xs px-2 py-0.5 rounded-full border ${getModeColor(step.mode)}`}>
                  {step.mode}
                </span>
              </div>
              <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                {step.title}
              </h4>
              <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                {step.description}
              </p>
            </div>

            <div className="p-4">
              <pre className="bg-slate-900 dark:bg-black text-green-400 text-xs p-4 rounded-lg overflow-x-auto font-mono leading-relaxed">
                {step.code}
              </pre>
            </div>

            <div className="px-4 pb-4 grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                <h5 className="text-xs font-bold text-blue-700 dark:text-blue-300 mb-1 flex items-center gap-1">
                  <Layers className="w-3 h-3" /> 内存状态
                </h5>
                <p className="text-xs text-blue-600 dark:text-blue-400 whitespace-pre-line">
                  {step.memoryState}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                <h5 className="text-xs font-bold text-amber-700 dark:text-amber-300 mb-1 flex items-center gap-1">
                  <Cpu className="w-3 h-3" /> CPU 动作
                </h5>
                <p className="text-xs text-amber-600 dark:text-amber-400 whitespace-pre-line">
                  {step.cpuAction}
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="mt-4 p-3 bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800 rounded-lg">
        <p className="text-xs text-violet-800 dark:text-violet-300 text-center">
          <strong>核心流程：</strong>userinit → RUNNABLE → scheduler → swtch → usertrapret → sret → initcode → exec("/init") → init.c → fork → exec("sh") → shell
        </p>
      </div>
    </div>
  );
}
