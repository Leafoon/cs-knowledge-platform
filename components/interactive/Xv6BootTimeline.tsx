"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Cpu,
  Terminal,
  ChevronDown,
  ChevronUp,
  Zap,
  HardDrive,
  Users,
  Shield,
  Layers,
  Play,
  RotateCcw,
} from "lucide-react";

interface BootStep {
  id: number;
  title: string;
  subtitle: string;
  mode: "M-mode" | "S-mode" | "U-mode";
  icon: React.ReactNode;
  color: string;
  darkColor: string;
  description: string;
  code: string;
  details: string[];
}

const BOOT_STEPS: BootStep[] = [
  {
    id: 1,
    title: "QEMU 加载内核",
    subtitle: "加载 kernel ELF 到物理内存",
    mode: "M-mode",
    icon: <HardDrive className="w-5 h-5" />,
    color: "from-red-500 to-red-600",
    darkColor: "dark:from-red-600 dark:to-red-700",
    description: "QEMU 模拟器创建虚拟 RISC-V 硬件，将内核 ELF 文件的代码段加载到物理地址 0x80000000，所有 hart 的 PC 设为入口地址。",
    code: `qemu-system-riscv64 \\
  -machine virt \\
  -bios none \\
  -kernel kernel/kernel \\
  -m 128M \\
  -smp 3 \\
  -nographic`,
    details: [
      "创建 3 个 hart（CPU 核心）、128MB 物理内存",
      "加载 kernel ELF 的 .text 段到 0x80000000",
      "所有 hart 初始运行在 Machine Mode（M-mode）",
      "不加载 BIOS，直接从内核入口启动",
    ],
  },
  {
    id: 2,
    title: "entry.S 汇编入口",
    subtitle: "设置栈指针，跳转到 start()",
    mode: "M-mode",
    icon: <Zap className="w-5 h-5" />,
    color: "from-orange-500 to-orange-600",
    darkColor: "dark:from-orange-600 dark:to-orange-700",
    description: "CPU 从物理地址 0x80000000 开始执行 entry.S。每个 hart 计算自己的栈指针（基于 hart ID），然后调用 start() 函数。",
    code: `.globl _entry
_entry:
  la sp, stack0
  li a0, 1024*4
  csrr a1, mhartid
  addi a1, a1, 1
  mul a0, a0, a1
  add sp, sp, a0
  call start`,
    details: [
      "此时还未启用分页，所有地址都是物理地址",
      "每个 hart 使用独立的 4KB 栈空间",
      "栈地址 = stack0 + 4096 * (hartid + 1)",
      "hart 0 调用 start()，其他 hart 进入自旋等待",
    ],
  },
  {
    id: 3,
    title: "start() M-mode 配置",
    subtitle: "配置硬件，切换到 S-mode",
    mode: "M-mode",
    icon: <Shield className="w-5 h-5" />,
    color: "from-yellow-500 to-yellow-600",
    darkColor: "dark:from-yellow-600 dark:to-yellow-700",
    description: "start() 在 M-mode 下配置硬件：禁用中断、设置陷阱向量、配置 PMP、委托中断给 S-mode，最后通过 mret 切换到 S-mode。",
    code: `void start() {
  w_mstatus(r_mstatus() & ~MSTATUS_MIE);
  w_mepc((uint64)main);
  w_mtvec((uint64)timervec);
  w_satp(0);
  w_medeleg(0xffff);
  w_mideleg(0xffff);
  w_pmpaddr0(0x3fffffffffffffull);
  w_pmpcfg0(0xf);
  timerinit();
  w_tp(r_mhartid());
  asm volatile("mret");
}`,
    details: [
      "mepc = main：mret 后跳转到 main()",
      "mret 将 mepc 加载到 PC，同时切换 M→S 模式",
      "medeleg/mideleg：将异常和中断委托给 S-mode",
      "PMP 允许 S-mode 访问全部物理内存",
    ],
  },
  {
    id: 4,
    title: "main() 初始化 - 基础设施",
    subtitle: "console / printf / 物理页分配器 / 页表",
    mode: "S-mode",
    icon: <Cpu className="w-5 h-5" />,
    color: "from-green-500 to-green-600",
    darkColor: "dark:from-green-600 dark:to-green-700",
    description: "hart 0 的 main() 开始初始化内核子系统。首先初始化控制台和 printf，然后初始化物理页分配器（kinit），创建内核页表（kvminit），最后启用分页（kvminithart）。",
    code: `void main() {
  if(cpuid() == 0) {
    consoleinit();
    printfinit();
    printf("xv6 kernel is booting\\n");
    kinit();
    kvminit();
    kvminithart();
    ...
  }
}`,
    details: [
      "consoleinit()：初始化 UART 串口和控制台驱动",
      "printfinit()：初始化 printf 的自旋锁",
      "kinit()：将空闲物理页加入链表分配器",
      "kvminit()：创建内核页表（恒等映射）",
      "kvminithart()：将页表加载到 satp，启用 Sv39 分页",
    ],
  },
  {
    id: 5,
    title: "main() 初始化 - 进程与中断",
    subtitle: "进程表 / trap / PLIC",
    mode: "S-mode",
    icon: <Layers className="w-5 h-5" />,
    color: "from-teal-500 to-teal-600",
    darkColor: "dark:from-teal-600 dark:to-teal-700",
    description: "继续初始化：进程表（procinit）、陷阱处理（trapinit/trapinithart）、PLIC 中断控制器（plicinit/plicinithart）。",
    code: `    procinit();
    trapinit();
    trapinithart();
    plicinit();
    plicinithart();`,
    details: [
      "procinit()：初始化进程表（最多 64 个进程）",
      "trapinit()：设置系统调用函数指针表",
      "trapinithart()：设置 stvec = kernelvec",
      "plicinit()：配置 PLIC 中断优先级",
      "plicinithart()：使能当前 hart 的外部中断",
    ],
  },
  {
    id: 6,
    title: "main() 初始化 - 文件系统与磁盘",
    subtitle: "缓冲区缓存 / inode / 文件表 / virtio",
    mode: "S-mode",
    icon: <HardDrive className="w-5 h-5" />,
    color: "from-blue-500 to-blue-600",
    darkColor: "dark:from-blue-600 dark:to-blue-700",
    description: "初始化文件系统相关组件：块 I/O 缓冲区缓存（binit）、inode 缓存（iinit）、文件表（fileinit），最后初始化 virtio 磁盘驱动。",
    code: `    binit();
    iinit();
    fileinit();
    virtio_disk_init();`,
    details: [
      "binit()：初始化 NBUF 个缓冲区的 LRU 链表",
      "iinit()：初始化 inode 缓存表",
      "fileinit()：初始化全局文件表的锁",
      "virtio_disk_init()：初始化 virtio 磁盘设备（复位→协商→就绪）",
    ],
  },
  {
    id: 7,
    title: "userinit() 创建第一个进程",
    subtitle: "PID 1 的 init 进程诞生",
    mode: "S-mode",
    icon: <Users className="w-5 h-5" />,
    color: "from-purple-500 to-purple-600",
    darkColor: "dark:from-purple-600 dark:to-purple-700",
    description: "调用 userinit() 创建第一个用户进程（PID 1）。将 initcode 的机器码复制到用户地址空间 0x0，设置 epc=0（程序入口），将进程标记为 RUNNABLE。",
    code: `void userinit(void) {
  struct proc *p = allocproc();
  initproc = p;
  p->pagetable = proc_pagetable(p);
  p->sz = PGSIZE;
  uvminit(p->pagetable, initcode, sizeof(initcode));
  p->trapframe->epc = 0;
  p->trapframe->sp = PGSIZE;
  p->state = RUNNABLE;
}`,
    details: [
      "allocproc()：分配进程结构体，创建内核栈",
      "uvminit()：将 initcode 复制到用户页 0x0",
      "epc=0：用户程序从虚拟地址 0 开始执行",
      "sp=PGSIZE：用户栈从一页顶部开始",
    ],
  },
  {
    id: 8,
    title: "scheduler() 调度到 init 进程",
    subtitle: "上下文切换，进入用户态",
    mode: "S-mode",
    icon: <Play className="w-5 h-5" />,
    color: "from-indigo-500 to-indigo-600",
    darkColor: "dark:from-indigo-600 dark:to-indigo-700",
    description: "main() 最后调用 scheduler()，这是调度器主循环。找到唯一的 RUNNABLE 进程（init），通过 swtch() 切换到它的内核上下文，最终 usertrapret() 返回用户态。",
    code: `void scheduler(void) {
  for(;;) {
    for(p = proc; p < &proc[NPROC]; p++) {
      acquire(&p->lock);
      if(p->state == RUNNABLE) {
        p->state = RUNNING;
        c->proc = p;
        swtch(&c->context, &p->context);
        c->proc = 0;
      }
      release(&p->lock);
    }
  }
}`,
    details: [
      "scheduler() 是一个永不返回的死循环",
      "swtch() 保存当前上下文，恢复 init 的上下文",
      "usertrapret() 设置 stvec=uservec，准备 sret",
      "sret 将 CPU 切换到 U-mode，PC = epc = 0",
    ],
  },
  {
    id: 9,
    title: "initcode → exec → init.c → shell",
    subtitle: "用户态启动链",
    mode: "U-mode",
    icon: <Terminal className="w-5 h-5" />,
    color: "from-pink-500 to-pink-600",
    darkColor: "dark:from-pink-600 dark:to-pink-700",
    description: "initcode.S 执行 exec(\"/init\") 系统调用，加载 init.c 程序。init.c 打开控制台文件描述符，fork 创建子进程，子进程 exec(\"sh\") 启动 shell。用户看到 xv6$ 提示符！",
    code: `// init.c
int main(void) {
  open("console", O_RDWR);
  dup(0);  // stdout
  dup(0);  // stderr
  for(;;) {
    pid = fork();
    if(pid == 0) {
      exec("sh", argv);
      exit(1);
    }
    wait(0);
  }
}`,
    details: [
      "initcode.S：最小程序，只调用 exec(\"/init\")",
      "init.c：打开 fd 0/1/2，启动 shell",
      "fork() + exec(\"sh\")：创建 shell 子进程",
      "如果 shell 退出，init 会重新启动一个",
      "用户看到 xv6$ 提示符，启动完成！",
    ],
  },
];

const MODE_COLORS = {
  "M-mode": { bg: "bg-red-100 dark:bg-red-900/40", text: "text-red-700 dark:text-red-300", border: "border-red-300 dark:border-red-700" },
  "S-mode": { bg: "bg-blue-100 dark:bg-blue-900/40", text: "text-blue-700 dark:text-blue-300", border: "border-blue-300 dark:border-blue-700" },
  "U-mode": { bg: "bg-green-100 dark:bg-green-900/40", text: "text-green-700 dark:text-green-300", border: "border-green-300 dark:border-green-700" },
};

export default function Xv6BootTimeline() {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [activeMode, setActiveMode] = useState<string | null>(null);
  const [animStep, setAnimStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const toggle = (id: number) => setExpanded(expanded === id ? null : id);

  const filtered = activeMode
    ? BOOT_STEPS.filter((s) => s.mode === activeMode)
    : BOOT_STEPS;

  const startAnimation = () => {
    setPlaying(true);
    setAnimStep(0);
    let step = 0;
    const interval = setInterval(() => {
      step++;
      setAnimStep(step);
      if (step >= BOOT_STEPS.length - 1) {
        clearInterval(interval);
        setPlaying(false);
      }
    }, 800);
  };

  const resetAnimation = () => {
    setAnimStep(0);
    setPlaying(false);
  };

  return (
    <div className="max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
        xv6 启动流程时间线
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        从 QEMU 加载内核到 shell 提示符的完整启动过程
      </p>

      <div className="flex flex-wrap gap-2 mb-4">
        {(["M-mode", "S-mode", "U-mode"] as const).map((m) => {
          const c = MODE_COLORS[m];
          const active = activeMode === m;
          return (
            <button
              key={m}
              onClick={() => setActiveMode(active ? null : m)}
              className={`px-3 py-1 rounded-full text-xs font-semibold border transition-all ${
                active
                  ? `${c.bg} ${c.text} ${c.border}`
                  : "bg-white dark:bg-gray-700 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-gray-600"
              }`}
            >
              {m}
            </button>
          );
        })}
        <button
          onClick={playing ? resetAnimation : startAnimation}
          disabled={playing}
          className="ml-auto px-3 py-1 rounded-full text-xs font-semibold bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-50 transition-all flex items-center gap-1"
        >
          {playing ? "播放中..." : <><Play className="w-3 h-3" /> 自动播放</>}
        </button>
        {animStep > 0 && (
          <button onClick={resetAnimation} className="px-2 py-1 rounded text-xs text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-gray-700 transition-all">
            <RotateCcw className="w-3 h-3" />
          </button>
        )}
      </div>

      <div className="relative">
        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gradient-to-b from-red-400 via-blue-400 to-green-400 dark:from-red-600 dark:via-blue-600 dark:to-green-600" />

        <AnimatePresence>
          {filtered.map((step, idx) => {
            const isExpanded = expanded === step.id;
            const mc = MODE_COLORS[step.mode];
            const isVisible = animStep === 0 || idx <= animStep;

            return (
              <motion.div
                key={step.id}
                initial={animStep > 0 ? { opacity: 0, x: -20 } : false}
                animate={isVisible ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
                transition={{ duration: 0.3, delay: animStep > 0 ? 0.05 : 0 }}
                className="relative pl-14 pb-4"
              >
                <div className={`absolute left-4 w-5 h-5 rounded-full border-2 border-white dark:border-gray-800 flex items-center justify-center bg-gradient-to-br ${step.color} ${step.darkColor} shadow-sm z-10`}>
                  <div className="w-1.5 h-1.5 rounded-full bg-white" />
                </div>

                <button
                  onClick={() => toggle(step.id)}
                  className={`w-full text-left p-3 rounded-lg border transition-all ${
                    isExpanded
                      ? "bg-white dark:bg-gray-700 border-slate-300 dark:border-gray-600 shadow-md"
                      : "bg-white/60 dark:bg-gray-800/60 border-slate-200 dark:border-gray-700 hover:bg-white dark:hover:bg-gray-700 hover:shadow-sm"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`p-1 rounded bg-gradient-to-br ${step.color} ${step.darkColor} text-white`}>
                      {step.icon}
                    </span>
                    <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${mc.bg} ${mc.text} ${mc.border} border`}>
                      {step.mode}
                    </span>
                    <span className="text-xs text-slate-400 dark:text-slate-500 font-mono">
                      #{step.id}
                    </span>
                    <span className="ml-auto text-slate-400 dark:text-slate-500">
                      {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </span>
                  </div>
                  <p className="font-semibold text-sm text-slate-700 dark:text-slate-200">
                    {step.title}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {step.subtitle}
                  </p>
                </button>

                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="mt-2 p-4 bg-white dark:bg-gray-700 rounded-lg border border-slate-200 dark:border-gray-600">
                        <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                          {step.description}
                        </p>
                        <pre className="bg-slate-900 dark:bg-black text-green-400 text-xs p-3 rounded-lg overflow-x-auto mb-3 font-mono">
                          {step.code}
                        </pre>
                        <ul className="space-y-1">
                          {step.details.map((d, i) => (
                            <li key={i} className="text-xs text-slate-600 dark:text-slate-400 flex items-start gap-2">
                              <span className="w-1 h-1 rounded-full bg-blue-400 dark:bg-blue-500 mt-1.5 flex-shrink-0" />
                              {d}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      <div className="mt-4 flex gap-3 justify-center">
        {(["M-mode", "S-mode", "U-mode"] as const).map((m) => {
          const c = MODE_COLORS[m];
          return (
            <div key={m} className="flex items-center gap-1.5">
              <div className={`w-3 h-3 rounded-full ${c.bg} ${c.border} border`} />
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {m === "M-mode" ? "Machine Mode" : m === "S-mode" ? "Supervisor Mode" : "User Mode"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
