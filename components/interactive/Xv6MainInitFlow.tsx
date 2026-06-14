"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronRight,
  ChevronDown,
  Monitor,
  Printer,
  MemoryStick,
  Layers,
  Cpu,
  Shield,
  Wifi,
  HardDrive,
  FileText,
  Database,
  Users,
  Play,
  RotateCcw,
  ArrowRight,
} from "lucide-react";

interface InitFunction {
  id: string;
  name: string;
  phase: string;
  phaseColor: string;
  icon: React.ReactNode;
  description: string;
  whatItDoes: string[];
  depends: string[];
  code: string;
  whyThisOrder: string;
}

const INIT_FUNCTIONS: InitFunction[] = [
  {
    id: "consoleinit",
    name: "consoleinit()",
    phase: "基础 I/O",
    phaseColor: "bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300",
    icon: <Monitor className="w-4 h-4" />,
    description: "初始化控制台设备和 UART 串口硬件，建立内核的 I/O 通道。",
    whatItDoes: [
      "初始化控制台自旋锁 cons.lock",
      "注册设备驱动的 read/write 函数（consoleread/consolewrite）",
      "初始化 UART 硬件（uartinit）",
      "连接 stdin/stdout 到控制台设备",
    ],
    depends: [],
    code: `void consoleinit(void) {
  initlock(&cons.lock, "console");
  devsw[CONSOLE].read = consoleread;
  devsw[CONSOLE].write = consolewrite;
  uartinit();
}`,
    whyThisOrder: "必须最先初始化，因为后续所有函数都可能调用 printf 输出调试信息。",
  },
  {
    id: "printfinit",
    name: "printfinit()",
    phase: "基础 I/O",
    phaseColor: "bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300",
    icon: <Printer className="w-4 h-4" />,
    description: "初始化内核 printf 函数使用的自旋锁，防止多核输出交错。",
    whatItDoes: [
      "初始化 printf 锁 pr.lock",
      "设置 pr.locking = 1（启用锁保护）",
    ],
    depends: [],
    code: `void printfinit(void) {
  initlock(&pr.lock, "pr");
  pr.locking = 1;
}`,
    whyThisOrder: "紧接 consoleinit 之后，因为 printf 依赖控制台已初始化。",
  },
  {
    id: "kinit",
    name: "kinit()",
    phase: "内存管理",
    phaseColor: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
    icon: <MemoryStick className="w-4 h-4" />,
    description: "初始化物理页分配器，将内核 ELF 结束到 PHYSTOP 的所有空闲物理页加入链表。",
    whatItDoes: [
      "初始化分配器自旋锁 kmem.lock",
      "调用 freerange(end, PHYSTOP)",
      "将每一页调用 kfree() 加入空闲链表",
      "每个空闲页的起始处存储指向下一个空闲页的指针",
    ],
    depends: [],
    code: `void kinit() {
  initlock(&kmem.lock, "kmem");
  freerange(end, (void*)PHYSTOP);
}

void freerange(void *pa_start, void *pa_end) {
  char *p = (char*)PGROUNDUP((uint64)pa_start);
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
    kfree(p);
}`,
    whyThisOrder: "必须在 kvminit 之前，因为 kvminit 需要调用 kalloc() 分配物理页存储页表。",
  },
  {
    id: "kvminit",
    name: "kvminit()",
    phase: "内存管理",
    phaseColor: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
    icon: <Layers className="w-4 h-4" />,
    description: "创建内核页表，建立虚拟地址到物理地址的恒等映射。映射 UART、PLIC、内核代码段、数据段、trampoline。",
    whatItDoes: [
      "分配一个物理页存储根页表",
      "映射 UART0 → UART0（R/W）",
      "映射 VIRTIO0 → VIRTIO0（R/W）",
      "映射 PLIC → PLIC（R/W, 4MB）",
      "映射内核代码段 KERNBASE→etext（R/X）",
      "映射内核数据段 etext→PHYSTOP（R/W）",
      "映射 TRAMPOLINE（R/X, 最高地址）",
    ],
    depends: ["kinit"],
    code: `pagetable_t kvminit(void) {
  pagetable_t kpagetable = (pagetable_t) kalloc();
  memset(kpagetable, 0, PGSIZE);
  kvmmap(kpagetable, UART0, UART0, PGSIZE, PTE_R|PTE_W);
  kvmmap(kpagetable, VIRTIO0, VIRTIO0, PGSIZE, PTE_R|PTE_W);
  kvmmap(kpagetable, PLIC, PLIC, 0x400000, PTE_R|PTE_W);
  kvmmap(kpagetable, KERNBASE, KERNBASE,
         (uint64)etext - KERNBASE, PTE_R|PTE_X);
  kvmmap(kpagetable, (uint64)etext, (uint64)etext,
         PHYSTOP - (uint64)etext, PTE_R|PTE_W);
  kvmmap(kpagetable, TRAMPOLINE, (uint64)trampoline,
         PGSIZE, PTE_R|PTE_X);
  return kpagetable;
}`,
    whyThisOrder: "使用恒等映射（虚拟=物理），所以可以在分页启用前用物理地址创建。",
  },
  {
    id: "kvminithart",
    name: "kvminithart()",
    phase: "内存管理",
    phaseColor: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
    icon: <Layers className="w-4 h-4" />,
    description: "将内核页表加载到 satp 寄存器，启用 Sv39 虚拟地址翻译。由于使用恒等映射，代码继续正常执行。",
    whatItDoes: [
      "写入 satp = SATP_SV39 | (pagetable >> 12)",
      "执行 sfence_vma() 刷新 TLB",
      "此后 CPU 使用虚拟地址访问内存",
    ],
    depends: ["kvminit"],
    code: `void kvminithart() {
  w_satp(MAKE_SATP(kernel_pagetable));
  sfence_vma();
}`,
    whyThisOrder: "必须在 kvminit 之后（页表已创建），必须在所有需要虚拟地址的操作之前。",
  },
  {
    id: "procinit",
    name: "procinit()",
    phase: "进程管理",
    phaseColor: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
    icon: <Users className="w-4 h-4" />,
    description: "初始化进程表（proc[] 数组），为每个进程分配内核栈虚拟地址。",
    whatItDoes: [
      "初始化 PID 分配器的锁",
      "遍历 proc[0..NPROC-1]，设置 state = UNUSED",
      "为每个进程计算内核栈地址 KSTACK(i)",
      "初始化每个进程的锁",
    ],
    depends: ["kinit"],
    code: `void procinit(void) {
  initlock(&pid_lock, "nextpid");
  nextpid = 1;
  for(struct proc *p = proc; p < &proc[NPROC]; p++) {
    initlock(&p->lock, "proc");
    p->state = UNUSED;
    p->kstack = KSTACK((int)(p - proc));
  }
}`,
    whyThisOrder: "需要内存系统可用（进程结构体在 BSS 段，但内核栈地址需要页表映射）。",
  },
  {
    id: "trapinit",
    name: "trapinit()",
    phase: "中断/陷阱",
    phaseColor: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300",
    icon: <Shield className="w-4 h-4" />,
    description: "初始化陷阱处理相关的全局数据结构，包括系统调用函数指针表和时钟锁。",
    whatItDoes: [
      "初始化时钟中断锁 tickslock",
      "系统调用表在 syscall.c 中静态初始化",
    ],
    depends: [],
    code: `void trapinit(void) {
  initlock(&tickslock, "time");
}`,
    whyThisOrder: "全局初始化只需一次，与 per-hart 的 trapinithart 分开。",
  },
  {
    id: "trapinithart",
    name: "trapinithart()",
    phase: "中断/陷阱",
    phaseColor: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300",
    icon: <Shield className="w-4 h-4" />,
    description: "为当前 hart 设置 S-mode 陷阱向量寄存器 stvec，指向 kernelvec。",
    whatItDoes: [
      "设置 stvec = kernelvec（汇编入口地址）",
      "kernelvec 保存寄存器后调用 kerneltrap()",
      "每个 hart 都需要独立设置自己的 stvec",
    ],
    depends: ["trapinit"],
    code: `void trapinithart(void) {
  w_stvec((uint64)kernelvec);
}`,
    whyThisOrder: "必须在 PLIC 和设备初始化之前，因为中断可能随时触发。",
  },
  {
    id: "plicinit",
    name: "plicinit()",
    phase: "中断控制器",
    phaseColor: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300",
    icon: <Wifi className="w-4 h-4" />,
    description: "初始化 PLIC（Platform-Level Interrupt Controller），设置外部中断的优先级。",
    whatItDoes: [
      "设置 virtio 磁盘中断的优先级 = 1",
      "设置 UART 中断的优先级 = 1",
      "配置 PLIC 的内存映射寄存器",
    ],
    depends: ["kvminithart"],
    code: `void plicinit(void) {
  *(uint32*)(PLIC + IRQ_VIRTIO*4) = 1;
  *(uint32*)(PLIC + IRQ_UART*4) = 1;
}`,
    whyThisOrder: "需要页表已启用（PLIC 地址是内存映射的），必须在设备驱动之前。",
  },
  {
    id: "plicinithart",
    name: "plicinithart()",
    phase: "中断控制器",
    phaseColor: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300",
    icon: <Wifi className="w-4 h-4" />,
    description: "为当前 hart 配置 PLIC：使能 virtio 和 UART 中断，设置中断阈值为 0（接受所有优先级）。",
    whatItDoes: [
      "设置 SENABLE 寄存器：使能 virtio 和 UART 中断",
      "设置 SPRIORITY 阈值 = 0",
      "在 sstatus 中启用外部中断（SIE_SEIE）",
    ],
    depends: ["plicinit", "trapinithart"],
    code: `void plicinithart(void) {
  int hart = cpuid();
  *(uint32*)PLIC_SENABLE(hart) =
    (1 << IRQ_VIRTIO) | (1 << IRQ_UART);
  *(uint32*)PLIC_SPRIORITY(hart) = 0;
  w_sie(r_sie() | SIE_SEIE);
}`,
    whyThisOrder: "必须在 trapinithart 之后（stvec 已设置），必须在设备驱动之前。",
  },
  {
    id: "binit",
    name: "binit()",
    phase: "文件系统",
    phaseColor: "bg-teal-100 text-teal-700 dark:bg-teal-900/40 dark:text-teal-300",
    icon: <Database className="w-4 h-4" />,
    description: "初始化块 I/O 缓冲区缓存（buffer cache），将 NBUF 个缓冲区组织成 LRU 双向链表。",
    whatItDoes: [
      "初始化缓冲区缓存锁",
      "创建 LRU 循环双向链表",
      "将 NBUF 个缓冲区加入链表",
      "head 是哨兵节点",
    ],
    depends: ["kinit"],
    code: `void binit(void) {
  struct buf *b;
  initlock(&bcache.lock, "bcache");
  bcache.head.prev = &bcache.head;
  bcache.head.next = &bcache.head;
  for(b = bcache.buf; b < bcache.buf + NBUF; b++) {
    b->next = bcache.head.next;
    b->prev = &bcache.head;
    bcache.head.next->prev = b;
    bcache.head.next = b;
  }
}`,
    whyThisOrder: "文件系统的基础设施，必须在 inode 和文件操作之前。",
  },
  {
    id: "iinit",
    name: "iinit()",
    phase: "文件系统",
    phaseColor: "bg-teal-100 text-teal-700 dark:bg-teal-900/40 dark:text-teal-300",
    icon: <FileText className="w-4 h-4" />,
    description: "初始化 inode 缓存表（itable），所有 inode 初始为 UNUSED 状态。",
    whatItDoes: [
      "初始化 inode 缓存的锁",
      "为每个 inode 条目初始化睡眠锁",
    ],
    depends: ["binit"],
    code: `void iinit(void) {
  initlock(&itable.lock, "itable");
  for(int i = 0; i < NINODE; i++) {
    initsleeplock(&itable.inode[i].lock, "inode");
  }
}`,
    whyThisOrder: "依赖缓冲区缓存（binit）已初始化。",
  },
  {
    id: "fileinit",
    name: "fileinit()",
    phase: "文件系统",
    phaseColor: "bg-teal-100 text-teal-700 dark:bg-teal-900/40 dark:text-teal-300",
    icon: <FileText className="w-4 h-4" />,
    description: "初始化全局文件表（ftable）的锁。文件表跟踪所有打开的文件。",
    whatItDoes: [
      "初始化文件表锁 ftable.lock",
      "文件表是全局共享的 struct file 数组",
    ],
    depends: [],
    code: `void fileinit(void) {
  initlock(&ftable.lock, "file");
}`,
    whyThisOrder: "简单的锁初始化，为文件描述符操作做准备。",
  },
  {
    id: "virtio_disk_init",
    name: "virtio_disk_init()",
    phase: "设备驱动",
    phaseColor: "bg-pink-100 text-pink-700 dark:bg-pink-900/40 dark:text-pink-300",
    icon: <HardDrive className="w-4 h-4" />,
    description: "初始化 virtio 磁盘设备驱动，按照 virtio 规范完成设备协商和队列配置。",
    whatItDoes: [
      "复位 virtio 设备",
      "设置 ACKNOWLEDGE 和 DRIVER 状态位",
      "协商设备特性（features）",
      "设置 FEATURES_OK",
      "配置 virtqueue（描述符表、可用环、已用环）",
      "设置 DRIVER_OK，设备就绪",
    ],
    depends: ["plicinithart", "binit"],
    code: `void virtio_disk_init(void) {
  uint32 status = 0;
  *R(VIRTIO_MMIO_STATUS) = status;
  status |= VIRTIO_CONFIG_S_ACKNOWLEDGE;
  *R(VIRTIO_MMIO_STATUS) = status;
  status |= VIRTIO_CONFIG_S_DRIVER;
  *R(VIRTIO_MMIO_STATUS) = status;
  // ... 协商特性、配置队列
  status |= VIRTIO_CONFIG_S_DRIVER_OK;
  *R(VIRTIO_MMIO_STATUS) = status;
}`,
    whyThisOrder: "需要中断系统和缓冲区缓存都已就绪。",
  },
  {
    id: "userinit",
    name: "userinit()",
    phase: "第一个进程",
    phaseColor: "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300",
    icon: <Play className="w-4 h-4" />,
    description: "创建第一个用户进程（PID 1），加载 initcode 到用户地址空间，将进程标记为 RUNNABLE。这是整个启动流程的终点和用户程序的起点。",
    whatItDoes: [
      "调用 allocproc() 分配进程结构体",
      "创建用户页表（proc_pagetable）",
      "将 initcode 复制到用户地址空间 0x0",
      "设置 trapframe->epc = 0（用户程序入口）",
      "设置 trapframe->sp = PGSIZE（用户栈）",
      "设置当前目录为根目录 /",
      "将进程状态设为 RUNNABLE",
    ],
    depends: ["all previous"],
    code: `void userinit(void) {
  struct proc *p = allocproc();
  initproc = p;
  p->pagetable = proc_pagetable(p);
  p->sz = PGSIZE;
  uvminit(p->pagetable, initcode, sizeof(initcode));
  p->trapframe->epc = 0;
  p->trapframe->sp = PGSIZE;
  safestrcpy(p->name, "initcode", sizeof(p->name));
  p->cwd = namei("/");
  p->state = RUNNABLE;
}`,
    whyThisOrder: "最后初始化，因为第一个进程需要所有子系统（内存、中断、文件系统、磁盘）都已就绪。",
  },
];

const PHASE_ORDER = ["基础 I/O", "内存管理", "进程管理", "中断/陷阱", "中断控制器", "文件系统", "设备驱动", "第一个进程"];

export default function Xv6MainInitFlow() {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [showDeps, setShowDeps] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);

  const toggle = (id: string) => setExpanded(expanded === id ? null : id);

  const stepForward = () => {
    setCurrentStep((prev) => Math.min(prev + 1, INIT_FUNCTIONS.length - 1));
  };

  const reset = () => {
    setCurrentStep(-1);
    setExpanded(null);
  };

  const autoPlay = () => {
    setCurrentStep(0);
    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step >= INIT_FUNCTIONS.length) {
        clearInterval(interval);
        return;
      }
      setCurrentStep(step);
    }, 600);
  };

  const visibleFunctions = currentStep >= 0
    ? INIT_FUNCTIONS.slice(0, currentStep + 1)
    : INIT_FUNCTIONS;

  return (
    <div className="max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-1">
        xv6 main() 初始化流程
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        按顺序调用的 15 个初始化函数，从 consoleinit 到 userinit
      </p>

      <div className="flex flex-wrap gap-2 mb-4">
        {PHASE_ORDER.map((phase) => {
          const funcs = INIT_FUNCTIONS.filter((f) => f.phase === phase);
          return (
            <span key={phase} className="text-xs px-2 py-1 rounded-full bg-white dark:bg-gray-700 border border-slate-200 dark:border-gray-600 text-slate-500 dark:text-slate-400">
              {phase} ({funcs.length})
            </span>
          );
        })}
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={autoPlay}
          className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-500 text-white hover:bg-emerald-600 transition-all flex items-center gap-1"
        >
          <Play className="w-3 h-3" /> 自动播放
        </button>
        <button
          onClick={stepForward}
          disabled={currentStep >= INIT_FUNCTIONS.length - 1}
          className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-40 transition-all flex items-center gap-1"
        >
          <ArrowRight className="w-3 h-3" /> 下一步
        </button>
        <button
          onClick={reset}
          className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-gray-600 transition-all flex items-center gap-1"
        >
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
        <label className="ml-auto flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={showDeps}
            onChange={(e) => setShowDeps(e.target.checked)}
            className="rounded"
          />
          显示依赖关系
        </label>
      </div>

      <div className="relative">
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gradient-to-b from-slate-300 via-emerald-400 to-violet-400 dark:from-slate-600 dark:via-emerald-600 dark:to-violet-600" />

        <AnimatePresence>
          {visibleFunctions.map((func, idx) => {
            const isExpanded = expanded === func.id;
            const isLatest = currentStep >= 0 && idx === currentStep;

            return (
              <motion.div
                key={func.id}
                initial={currentStep >= 0 ? { opacity: 0, x: -15, scale: 0.95 } : false}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                transition={{ duration: 0.2 }}
                className="relative pl-10 mb-2"
              >
                <div className={`absolute left-2.5 w-4 h-4 rounded-full border-2 border-white dark:border-gray-800 z-10 transition-all ${
                  isLatest
                    ? "bg-emerald-500 shadow-lg shadow-emerald-300 dark:shadow-emerald-700"
                    : "bg-slate-300 dark:bg-slate-600"
                }`} />

                <button
                  onClick={() => toggle(func.id)}
                  className={`w-full text-left p-3 rounded-lg border transition-all ${
                    isExpanded
                      ? "bg-white dark:bg-gray-700 border-slate-300 dark:border-gray-600 shadow-md"
                      : isLatest
                        ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800 shadow-sm"
                        : "bg-white/60 dark:bg-gray-800/60 border-slate-200 dark:border-gray-700 hover:bg-white dark:hover:bg-gray-700"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-slate-400 dark:text-slate-500 w-5 text-right">
                      {idx + 1}.
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${func.phaseColor}`}>
                      {func.phase}
                    </span>
                    <span className="font-mono text-sm font-bold text-slate-700 dark:text-slate-200">
                      {func.name}
                    </span>
                    {showDeps && func.depends.length > 0 && (
                      <span className="text-xs text-amber-600 dark:text-amber-400">
                        ← {func.depends.join(", ")}
                      </span>
                    )}
                    <span className="ml-auto text-slate-400 dark:text-slate-500">
                      {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 ml-7">
                    {func.description.slice(0, 60)}...
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
                          {func.description}
                        </p>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                          <div>
                            <h5 className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-1">具体操作：</h5>
                            <ul className="space-y-1 mb-3">
                              {func.whatItDoes.map((item, i) => (
                                <li key={i} className="text-xs text-slate-600 dark:text-slate-400 flex items-start gap-1.5">
                                  <span className="w-1 h-1 rounded-full bg-emerald-400 mt-1.5 flex-shrink-0" />
                                  {item}
                                </li>
                              ))}
                            </ul>
                            <div className="p-2 bg-amber-50 dark:bg-amber-900/20 rounded border border-amber-200 dark:border-amber-800">
                              <p className="text-xs text-amber-800 dark:text-amber-300">
                                <strong>为什么这个顺序：</strong>{func.whyThisOrder}
                              </p>
                            </div>
                          </div>
                          <div>
                            <h5 className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-1">源码：</h5>
                            <pre className="bg-slate-900 dark:bg-black text-green-400 text-xs p-3 rounded-lg overflow-x-auto font-mono">
                              {func.code}
                            </pre>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}
