# 操作系统完整学习大纲

> **Version**: Based on OSTEP, OS Concepts 11th, Modern OS 5th, xv6-riscv (2026年1月)  
> **Target Audience**: 计算机科学研究生、系统工程师、操作系统研究人员  
> **Prerequisite**: C 语言编程、计算机组成原理、数据结构与算法基础

---

## 📚 **课程结构概览**

```
Part I: 操作系统基础与抽象 (Chapters 0-2)
Part II: 虚拟化：进程抽象 (Chapters 3-5)
Part III: 虚拟化：内存管理 (Chapters 6-9)
Part IV: CPU 调度机制 (Chapters 10-12)
Part V: 并发与同步 (Chapters 13-16)
Part VI: 死锁问题 (Chapters 17-18)
Part VII: 持久化：文件系统 (Chapters 19-22)
Part VIII: 持久化：I/O 与存储 (Chapters 23-24)
Part IX: xv6 内核深度剖析 (Chapters 25-28)
Part X: 现代操作系统主题 (Chapters 29-33)
Part XI: 高级主题与研究方向 (Chapters 34-36)
```

---

## Part I: 操作系统基础与抽象 (Foundation)

### **Chapter 0: 操作系统概论**
- 0.1 操作系统的定义与作用
  - 0.1.1 什么是操作系统？资源管理器 vs 虚拟机
  - 0.1.2 操作系统的历史演进（批处理 → 分时 → 个人计算 → 云计算）
  - 0.1.3 操作系统在计算机系统中的位置
  - 0.1.4 为什么需要操作系统？硬件抽象与资源复用
- 0.2 操作系统的核心功能
  - 0.2.1 虚拟化（Virtualization）：CPU、内存
  - 0.2.2 并发（Concurrency）：多线程与同步
  - 0.2.3 持久化（Persistence）：文件系统与 I/O
  - 0.2.4 安全与保护（Security & Protection）
- 0.3 操作系统的设计目标
  - 0.3.1 性能（Performance）：吞吐量、响应时间、利用率
  - 0.3.2 可靠性（Reliability）：容错、恢复、一致性
  - 0.3.3 安全性（Security）：隔离、权限、审计
  - 0.3.4 可扩展性（Scalability）：多核、分布式
  - 0.3.5 易用性（Usability）：API 设计、兼容性
- 0.4 操作系统类型与架构
  - 0.4.1 单体内核（Monolithic Kernel）：Linux、Unix
  - 0.4.2 微内核（Microkernel）：Minix、L4、seL4
  - 0.4.3 混合内核（Hybrid Kernel）：Windows NT、macOS
  - 0.4.4 外核（Exokernel）与 Unikernel
  - 0.4.5 架构对比与权衡分析

**交互式组件**：
- `SystemLayersVisualization` - 操作系统层次结构动画
- `KernelArchitectureComparison` - 内核架构对比图
- `ComputerEvolutionTimeline` - 操作系统历史演进时间线

---

### **Chapter 1: 硬件基础与系统启动**
- 1.1 冯·诺依曼架构回顾
  - 1.1.1 CPU、内存、I/O 设备的基本结构
  - 1.1.2 指令执行周期（Fetch-Decode-Execute）
  - 1.1.3 寄存器、缓存、主存层次结构
  - 1.1.4 总线与设备互连
- 1.2 处理器模式与特权级
  - 1.2.1 用户态（User Mode）vs 内核态（Kernel Mode）
  - 1.2.2 特权指令与系统调用
  - 1.2.3 模式切换的硬件机制
  - 1.2.4 保护环（Protection Rings）：x86 的 Ring 0-3
- 1.3 中断与异常机制
  - 1.3.1 中断的分类：硬件中断、软件中断、异常
  - 1.3.2 中断描述符表（IDT）与中断向量
  - 1.3.3 中断处理流程：保存上下文 → 处理 → 恢复
  - 1.3.4 中断优先级与中断嵌套
  - 1.3.5 中断延迟与响应时间分析
- 1.4 内存管理单元（MMU）
  - 1.4.1 地址翻译基础：虚拟地址 → 物理地址
  - 1.4.2 TLB（Translation Lookaside Buffer）原理
  - 1.4.3 页表基址寄存器（PTBR）
  - 1.4.4 内存保护位（Read/Write/Execute）
- 1.5 系统启动流程
  - 1.5.1 BIOS/UEFI → Bootloader → 内核加载
  - 1.5.2 xv6 启动流程详解
  - 1.5.3 Linux 启动流程（GRUB → vmlinuz → init/systemd）
  - 1.5.4 内核初始化：内存检测、设备枚举、进程 0/1 创建

**交互式组件**：
- `VonNeumannArchitecture` - 冯·诺依曼架构交互式示意图
- `InstructionCycleSimulator` - 指令周期模拟器
- `InterruptHandlingFlow` - 中断处理流程动画
- `SystemBootVisualizer` - 系统启动流程可视化

---

### **Chapter 2: 系统调用与操作系统接口**
- 2.1 系统调用概述
  - 2.1.1 什么是系统调用？用户态与内核态的桥梁
  - 2.1.2 系统调用 vs 库函数调用
  - 2.1.3 POSIX 标准与操作系统兼容性
  - 2.1.4 系统调用的性能开销分析
- 2.2 系统调用机制
  - 2.2.1 软件中断（int 0x80、syscall 指令）
  - 2.2.2 系统调用号与参数传递
  - 2.2.3 系统调用表（syscall table）
  - 2.2.4 返回值与错误处理（errno）
- 2.3 系统调用分类
  - 2.3.1 进程控制：fork()、exec()、wait()、exit()
  - 2.3.2 文件操作：open()、read()、write()、close()
  - 2.3.3 内存管理：brk()、mmap()、munmap()
  - 2.3.4 进程通信：pipe()、socket()、shmget()
  - 2.3.5 信号处理：signal()、kill()、sigaction()
- 2.4 xv6 系统调用实现
  - 2.4.1 用户空间：usys.S 汇编代码分析
  - 2.4.2 内核空间：syscall.c 分发逻辑
  - 2.4.3 参数获取：argint()、argaddr()、argstr()
  - 2.4.4 完整示例：sys_fork() 实现剖析
- 2.5 Linux 系统调用深入
  - 2.5.1 系统调用约定：寄存器传参（x86-64）
  - 2.5.2 vDSO（Virtual Dynamic Shared Object）优化
  - 2.5.3 系统调用追踪：strace 工具使用
  - 2.5.4 新系统调用的添加流程

**交互式组件**：
- `SystemCallFlow` - 系统调用完整流程动画
- `SyscallTableViewer` - 系统调用表交互式查看器
- `Xv6SyscallTracer` - xv6 系统调用追踪可视化
- `StraceOutputAnalyzer` - strace 输出分析工具

---

## Part II: 虚拟化：进程抽象 (Process Virtualization)

### **Chapter 3: 进程抽象基础**
- 3.1 进程的概念
  - 3.1.1 程序 vs 进程：静态 vs 动态
  - 3.1.2 进程的定义：执行中的程序实例
  - 3.1.3 进程的组成：代码、数据、堆、栈、PCB
  - 3.1.4 进程抽象的价值：并发与隔离
- 3.2 进程控制块（PCB）
  - 3.2.1 PCB 的数据结构设计
  - 3.2.2 关键字段：PID、状态、寄存器、内存指针、打开文件
  - 3.2.3 xv6 的 struct proc 详解
  - 3.2.4 Linux 的 task_struct 核心字段
- 3.3 进程状态模型
  - 3.3.1 五状态模型：新建、就绪、运行、阻塞、终止
  - 3.3.2 七状态模型：增加挂起就绪、挂起阻塞
  - 3.3.3 状态转换条件与触发事件
  - 3.3.4 僵尸进程（Zombie）与孤儿进程（Orphan）
- 3.4 进程创建
  - 3.4.1 fork() 系统调用详解
  - 3.4.2 父子进程的关系：内存复制 vs 共享
  - 3.4.3 写时复制（Copy-on-Write）优化
  - 3.4.4 fork() 的返回值语义
  - 3.4.5 xv6 fork() 实现源码剖析
- 3.5 进程执行与终止
  - 3.5.1 exec() 系列系统调用：execl()、execv()、execve()
  - 3.5.2 exec() 的内存替换过程
  - 3.5.3 wait() 与 waitpid()：父进程等待子进程
  - 3.5.4 exit() 与进程清理
  - 3.5.5 进程终止的资源回收

**交互式组件**：
- `ProcessStateVisualization` - 进程状态机交互式动画
- `PCBStructureExplorer` - PCB 数据结构交互式查看器
- `ForkProcessTree` - fork() 进程树生成可视化
- `CopyOnWriteDemo` - 写时复制机制动画

---

### **Chapter 4: 上下文切换与进程调度基础**
- 4.1 上下文切换概述
  - 4.1.1 什么是上下文？CPU 寄存器、栈指针、程序计数器
  - 4.1.2 上下文切换的必要性：时间片轮转、I/O 阻塞
  - 4.1.3 上下文切换的开销：直接成本 + 间接成本
  - 4.1.4 上下文切换频率与系统性能
- 4.2 上下文切换机制
  - 4.2.1 保存旧进程上下文：寄存器 → PCB
  - 4.2.2 选择新进程：调度算法
  - 4.2.3 加载新进程上下文：PCB → 寄存器
  - 4.2.4 切换地址空间：更新页表基址寄存器
  - 4.2.5 TLB 刷新与缓存失效
- 4.3 xv6 上下文切换实现
  - 4.3.1 struct context 数据结构
  - 4.3.2 swtch() 汇编代码详解（kernel/swtch.S）
  - 4.3.3 sched() 与 scheduler() 协作
  - 4.3.4 用户态 → 内核态 → 调度器 → 新进程的完整路径
- 4.4 Linux 上下文切换
  - 4.4.1 schedule() 函数调用链
  - 4.4.2 context_switch() 实现
  - 4.4.3 进程上下文 vs 中断上下文
  - 4.4.4 性能优化：ASID（Address Space ID）
- 4.5 测量上下文切换
  - 4.5.1 lmbench 微基准测试
  - 4.5.2 perf 工具分析上下文切换
  - 4.5.3 vmstat、pidstat 监控

**交互式组件**：
- `ContextSwitchAnimation` - 上下文切换完整流程动画
- `RegisterSaveRestoreVisualizer` - 寄存器保存/恢复过程
- `SchedulerFlowDiagram` - 调度器调用流程图
- `ContextSwitchOverheadCalculator` - 上下文切换开销计算器

---

### **Chapter 5: 线程与多线程编程**
- 5.1 线程的概念
  - 5.1.1 为什么需要线程？进程的局限性
  - 5.1.2 线程的定义：轻量级进程
  - 5.1.3 线程 vs 进程：共享内存 vs 独立内存
  - 5.1.4 线程的优势：快速创建、低开销切换、共享数据
- 5.2 线程模型
  - 5.2.1 用户级线程（User-Level Threads）
  - 5.2.2 内核级线程（Kernel-Level Threads）
  - 5.2.3 混合线程模型（M:N 模型）
  - 5.2.4 三种模型的性能对比与权衡
- 5.3 线程控制块（TCB）
  - 5.3.1 TCB vs PCB：共享与独立部分
  - 5.3.2 线程私有数据：栈、寄存器、线程局部存储（TLS）
  - 5.3.3 线程共享数据：代码、数据、堆、文件描述符
- 5.4 POSIX 线程（Pthreads）
  - 5.4.1 线程创建：pthread_create()
  - 5.4.2 线程终止：pthread_exit()、pthread_join()
  - 5.4.3 线程分离：pthread_detach()
  - 5.4.4 线程取消：pthread_cancel()
  - 5.4.5 线程属性：pthread_attr_t
- 5.5 多线程问题
  - 5.5.1 竞态条件（Race Condition）
  - 5.5.2 临界区（Critical Section）
  - 5.5.3 数据竞争（Data Race）
  - 5.5.4 线程安全（Thread Safety）与可重入
- 5.6 Linux 线程实现
  - 5.6.1 clone() 系统调用：统一进程与线程
  - 5.6.2 NPTL（Native POSIX Thread Library）
  - 5.6.3 线程组（Thread Group）与 TGID
  - 5.6.4 线程调度：O(1)、CFS

**交互式组件**：
- `ThreadVsProcessComparison` - 线程与进程对比图
- `ThreadModelComparison` - 用户级/内核级/混合线程模型对比
- `PthreadAPIExplorer` - Pthread API 交互式演示
- `RaceConditionDemo` - 竞态条件可视化演示

---

## Part III: 虚拟化：内存管理 (Memory Virtualization)

### **Chapter 6: 内存抽象与地址空间**
- 6.1 内存抽象的必要性
  - 6.1.1 早期系统：直接物理内存访问的问题
  - 6.1.2 多道程序设计的内存挑战
  - 6.1.3 地址空间抽象的价值：隔离、保护、共享
- 6.2 地址空间概念
  - 6.2.1 虚拟地址空间（Virtual Address Space）
  - 6.2.2 物理地址空间（Physical Address Space）
  - 6.2.3 地址翻译：虚拟地址 → 物理地址
  - 6.2.4 地址空间布局：代码段、数据段、堆、栈
- 6.3 内存管理目标
  - 6.3.1 透明性（Transparency）：对程序透明
  - 6.3.2 效率（Efficiency）：时间效率 + 空间效率
  - 6.3.3 保护（Protection）：进程隔离与访问控制
  - 6.3.4 共享（Sharing）：代码共享、数据共享
- 6.4 早期内存管理技术
  - 6.4.1 基址寄存器与界限寄存器
  - 6.4.2 动态重定位（Dynamic Relocation）
  - 6.4.3 内存碎片问题：外部碎片 vs 内部碎片
  - 6.4.4 内存压缩（Memory Compaction）
- 6.5 连续内存分配
  - 6.5.1 固定分区（Fixed Partitioning）
  - 6.5.2 可变分区（Variable Partitioning）
  - 6.5.3 分配算法：首次适应、最佳适应、最坏适应
  - 6.5.4 性能对比与碎片分析

**交互式组件**：
- `AddressSpaceLayout` - 地址空间布局交互式图
- `AddressTranslationVisualizer` - 地址翻译过程动画
- `MemoryAllocationSimulator` - 内存分配算法模拟器
- `FragmentationVisualizer` - 内存碎片可视化

---

### **Chapter 7: 分页机制**
- 7.1 分页基础
  - 7.1.1 分页的动机：解决外部碎片
  - 7.1.2 页（Page）与页框（Page Frame）
  - 7.1.3 页表（Page Table）：虚拟页号 → 物理页框号
  - 7.1.4 页大小的权衡：4KB、2MB、1GB
- 7.2 地址翻译机制
  - 7.2.1 虚拟地址结构：VPN + Offset
  - 7.2.2 物理地址结构：PFN + Offset
  - 7.2.3 页表项（PTE）结构：PFN + 标志位
  - 7.2.4 标志位：有效位、保护位、修改位、访问位
  - 7.2.5 地址翻译公式与示例计算
- 7.3 页表实现
  - 7.3.1 线性页表（Linear Page Table）
  - 7.3.2 页表的内存开销分析
  - 7.3.3 页表存储位置：内核内存
  - 7.3.4 页表基址寄存器（PTBR / CR3）
- 7.4 TLB（Translation Lookaside Buffer）
  - 7.4.1 TLB 的必要性：加速地址翻译
  - 7.4.2 TLB 结构：全相联缓存
  - 7.4.3 TLB 查找流程：命中 vs 未命中
  - 7.4.4 TLB 替换策略：LRU、Random
  - 7.4.5 TLB 一致性：上下文切换时的刷新
  - 7.4.6 ASID（Address Space Identifier）优化
- 7.5 页表访问优化
  - 7.5.1 TLB 命中率分析：时间局部性 + 空间局部性
  - 7.5.2 TLB 覆盖范围（TLB Reach）
  - 7.5.3 大页（Huge Pages）：2MB、1GB 页
  - 7.5.4 性能测量：TLB 未命中代价

**交互式组件**：
- `PagingMechanismVisualizer` - 分页机制完整流程动画
- `PageTableWalker` - 页表查找步进演示
- `TLBSimulator` - TLB 命中/未命中模拟器
- `PageSizeComparison` - 不同页大小性能对比

---

### **Chapter 8: 高级分页技术**
- 8.1 多级页表
  - 8.1.1 线性页表的空间问题
  - 8.1.2 二级页表结构：页目录 + 页表
  - 8.1.3 三级、四级页表：x86-64 的四级分页
  - 8.1.4 多级页表的时间-空间权衡
  - 8.1.5 地址翻译开销：TLB 的重要性
- 8.2 反向页表（Inverted Page Table）
  - 8.2.1 设计动机：减少页表空间
  - 8.2.2 反向页表结构：按物理页框索引
  - 8.2.3 哈希表加速查找
  - 8.2.4 优缺点分析
- 8.3 页表缓存优化
  - 8.3.1 页表缓存（Page Table Cache）
  - 8.3.2 页表预取（Page Table Prefetching）
  - 8.3.3 页表压缩技术
- 8.4 x86-64 分页机制
  - 8.4.1 四级页表：PML4 → PDPT → PD → PT
  - 8.4.2 48 位虚拟地址空间
  - 8.4.3 大页支持：2MB（PSE）、1GB（1GB Pages）
  - 8.4.4 CR3 寄存器与页表基址
  - 8.4.5 页表项格式详解
- 8.5 ARM 分页机制
  - 8.5.1 ARMv8 地址翻译
  - 8.5.2 TTBR0 与 TTBR1：用户/内核地址空间
  - 8.5.3 页表遍历过程

**交互式组件**：
- `MultiLevelPageTableVisualizer` - 多级页表遍历动画
- `PageTableMemoryCalculator` - 页表内存开销计算器
- `X86PageTableWalker` - x86-64 页表遍历模拟器
- `InvertedPageTableDemo` - 反向页表查找演示

---

### **Chapter 9: 分段与段页式管理**
- 9.1 分段机制
  - 9.1.1 分段的动机：逻辑划分与保护
  - 9.1.2 段（Segment）的概念：代码段、数据段、堆段、栈段
  - 9.1.3 段表（Segment Table）与段描述符
  - 9.1.4 段基址与段界限
  - 9.1.5 段地址翻译过程
- 9.2 分段的优缺点
  - 9.2.1 优点：逻辑分离、共享、保护
  - 9.2.2 缺点：外部碎片、可变大小管理困难
  - 9.2.3 分段 vs 分页对比
- 9.3 段页式管理
  - 9.3.1 结合分段与分页的优势
  - 9.3.2 两级地址翻译：段 → 页
  - 9.3.3 x86 保护模式的段页式结构
  - 9.3.4 GDT（Global Descriptor Table）与 LDT
- 9.4 现代系统的分段
  - 9.4.1 x86-64 的平坦内存模型
  - 9.4.2 分段的退化：段基址为 0
  - 9.4.3 FS/GS 段寄存器的特殊用途：TLS

**交互式组件**：
- `SegmentationVisualizer` - 分段机制可视化
- `SegmentPageTranslation` - 段页式地址翻译动画
- `SegmentationVsPaging` - 分段与分页对比表

---

## Part IV: CPU 调度机制 (CPU Scheduling)

### **Chapter 10: CPU 调度基础**
- 10.1 调度概述
  - 10.1.1 为什么需要调度？CPU 利用率最大化
  - 10.1.2 调度时机：进程阻塞、时间片耗尽、创建/终止
  - 10.1.3 抢占式 vs 非抢占式调度
  - 10.1.4 调度器的位置：短期、中期、长期调度器
- 10.2 调度评价指标
  - 10.2.1 CPU 利用率（CPU Utilization）
  - 10.2.2 吞吐量（Throughput）：单位时间完成进程数
  - 10.2.3 周转时间（Turnaround Time）：完成时间 - 到达时间
  - 10.2.4 等待时间（Waiting Time）：在就绪队列等待的总时间
  - 10.2.5 响应时间（Response Time）：首次响应时间
  - 10.2.6 公平性（Fairness）：资源分配的公平程度
- 10.3 工作负载假设
  - 10.3.1 进程运行时间已知 vs 未知
  - 10.3.2 进程同时到达 vs 不同时到达
  - 10.3.3 CPU 密集型 vs I/O 密集型
  - 10.3.4 批处理 vs 交互式工作负载
- 10.4 调度算法分类
  - 10.4.1 批处理系统调度算法
  - 10.4.2 交互式系统调度算法
  - 10.4.3 实时系统调度算法
  - 10.4.4 多处理器调度算法

**交互式组件**：
- `SchedulingMetricsCalculator` - 调度指标计算器
- `WorkloadCharacteristics` - 工作负载特性可视化
- `SchedulerTimingDiagram` - 调度时序图生成器

---

### **Chapter 11: 经典调度算法**
- 11.1 先来先服务（FCFS / FIFO）
  - 11.1.1 算法描述：按到达顺序执行
  - 11.1.2 实现：FIFO 队列
  - 11.1.3 护航效应（Convoy Effect）
  - 11.1.4 性能分析：平均等待时间计算
  - 11.1.5 优缺点与适用场景
- 11.2 最短作业优先（SJF）
  - 11.2.1 算法描述：选择执行时间最短的进程
  - 11.2.2 证明：SJF 最小化平均周转时间（最优性证明）
  - 11.2.3 问题：如何预测执行时间？
  - 11.2.4 非抢占式 SJF 示例与计算
  - 11.2.5 饥饿问题（Starvation）
- 11.3 最短剩余时间优先（SRTF / STCF）
  - 11.3.1 SJF 的抢占式版本
  - 11.3.2 算法流程：新进程到达时比较剩余时间
  - 11.3.3 性能优势：更低的平均周转时间
  - 11.3.4 响应时间问题
  - 11.3.5 实现复杂度
- 11.4 轮转调度（Round Robin / RR）
  - 11.4.1 算法描述：时间片轮转
  - 11.4.2 时间片（Time Quantum）的选择：10ms - 100ms
  - 11.4.3 时间片大小的权衡：响应时间 vs 上下文切换开销
  - 11.4.4 平均响应时间分析
  - 11.4.5 RR 的公平性
- 11.5 优先级调度
  - 11.5.1 静态优先级 vs 动态优先级
  - 11.5.2 优先级队列实现：堆、多级队列
  - 11.5.3 优先级反转（Priority Inversion）
  - 11.5.4 优先级继承（Priority Inheritance）协议
  - 11.5.5 老化（Aging）技术防止饥饿
- 11.6 算法综合对比
  - 11.6.1 性能指标对比表
  - 11.6.2 工作负载适应性分析
  - 11.6.3 实现复杂度对比
  - 11.6.4 实际系统的选择

**交互式组件**：
- `FCFSSchedulerSimulator` - FCFS 调度模拟器
- `SJFOptimalityProof` - SJF 最优性证明可视化
- `RoundRobinSimulator` - 轮转调度交互式模拟器
- `SchedulingAlgorithmComparison` - 调度算法综合对比工具

---

### **Chapter 12: 高级调度算法与真实系统**
- 12.1 多级反馈队列（MLFQ）
  - 12.1.1 设计目标：优化周转时间 + 响应时间
  - 12.1.2 基本规则：
    - 规则 1：优先级高的先运行
    - 规则 2：同优先级 RR
    - 规则 3：新进程进入最高优先级
    - 规则 4：时间片用完降低优先级
    - 规则 5：I/O 后保持优先级
  - 12.1.3 防止游戏（Gaming）：周期性提升
  - 12.1.4 参数调优：队列数量、时间片、提升周期
  - 12.1.5 MLFQ 在 BSD、Windows 中的实现
- 12.2 完全公平调度器（CFS）
  - 12.2.1 Linux CFS 设计哲学：理想的多任务处理器
  - 12.2.2 虚拟运行时间（vruntime）
  - 12.2.3 红黑树（rbtree）实现就绪队列
  - 12.2.4 调度延迟（Scheduling Latency）与最小粒度
  - 12.2.5 Nice 值与权重映射
  - 12.2.6 CFS 公平性证明
  - 12.2.7 组调度（Group Scheduling）
- 12.3 实时调度
  - 12.3.1 硬实时 vs 软实时
  - 12.3.2 速率单调调度（RMS）
  - 12.3.3 最早截止时间优先（EDF）
  - 12.3.4 可调度性分析：Liu & Layland 定理
  - 12.3.5 Linux 实时调度：SCHED_FIFO、SCHED_RR、SCHED_DEADLINE
- 12.4 多处理器调度
  - 12.4.1 单队列 vs 多队列
  - 12.4.2 负载均衡（Load Balancing）
  - 12.4.3 处理器亲和性（CPU Affinity）
  - 12.4.4 缓存热度（Cache Warmth）
  - 12.4.5 Linux 多核调度：每 CPU 运行队列
  - 12.4.6 NUMA 感知调度
- 12.5 xv6 调度器分析
  - 12.5.1 简单的 RR 调度器实现
  - 12.5.2 scheduler() 函数代码剖析
  - 12.5.3 进程切换：yield() → sched() → swtch()
  - 12.5.4 锁的使用：ptable.lock

**交互式组件**：
- `MLFQSimulator` - MLFQ 交互式模拟器
- `CFSVirtualRuntime` - CFS vruntime 演进动画
- `CFSRedBlackTree` - CFS 红黑树可视化
- `RealTimeSchedulabilityAnalyzer` - 实时调度可调度性分析工具
- `MultiProcessorLoadBalancer` - 多处理器负载均衡动画

---

## Part V: 并发与同步 (Concurrency & Synchronization)

### **Chapter 13: 并发编程基础**
- 13.1 并发的挑战
  - 13.1.1 为什么并发很难？不确定性
  - 13.1.2 竞态条件（Race Condition）示例
  - 13.1.3 原子性（Atomicity）问题
  - 13.1.4 不可控的调度
- 13.2 临界区问题
  - 13.2.1 临界区（Critical Section）定义
  - 13.2.2 临界区问题的三个要求：
    - 互斥（Mutual Exclusion）
    - 进步（Progress）
    - 有限等待（Bounded Waiting）
  - 13.2.3 单处理器解决方案：禁用中断
  - 13.2.4 禁用中断的问题：不适用于多处理器、特权指令
- 13.3 软件解决方案
  - 13.3.1 Peterson 算法（两进程）
    - 算法描述与伪代码
    - 正确性证明：互斥、进步、有限等待
    - 现代处理器的问题：内存乱序
  - 13.3.2 Dekker 算法
  - 13.3.3 面包店算法（Bakery Algorithm，多进程）
  - 13.3.4 软件方案的局限性
- 13.4 硬件原子指令
  - 13.4.1 测试并设置（Test-and-Set / TSL）
  - 13.4.2 比较并交换（Compare-and-Swap / CAS）
  - 13.4.3 获取并增加（Fetch-and-Add）
  - 13.4.4 Load-Linked / Store-Conditional (LL/SC)
  - 13.4.5 原子指令的实现：缓存一致性协议
- 13.5 自旋锁（Spinlock）
  - 13.5.1 基于 TSL/CAS 的自旋锁实现
  - 13.5.2 自旋等待的问题：CPU 浪费
  - 13.5.3 适用场景：临界区很短、多处理器
  - 13.5.4 自旋锁优化：
    - Test-and-Test-and-Set
    - 指数退避（Exponential Backoff）
    - 队列自旋锁（MCS Lock、Ticket Lock）

**交互式组件**：
- `RaceConditionDemo` - 竞态条件交互式演示
- `PetersonAlgorithmVisualizer` - Peterson 算法执行过程动画
- `AtomicInstructionComparison` - 原子指令对比演示
- `SpinlockPerformance` - 自旋锁性能分析工具

---

### **Chapter 14: 锁与互斥**
- 14.1 锁（Lock）抽象
  - 14.1.1 锁的接口：lock() / unlock()
  - 14.1.2 锁的语义：互斥访问
  - 14.1.3 锁的评价标准：正确性、公平性、性能
- 14.2 简单锁实现
  - 14.2.1 禁用中断实现（单处理器）
  - 14.2.2 自旋锁实现（多处理器）
  - 14.2.3 yield() 自旋锁：降低 CPU 浪费
- 14.3 睡眠锁（Sleeping Lock）
  - 14.3.1 设计动机：避免自旋浪费 CPU
  - 14.3.2 队列 + 睡眠/唤醒机制
  - 14.3.3 Linux mutex 实现
  - 14.3.4 futex（Fast Userspace Mutex）
    - 用户态快速路径
    - 内核态慢速路径
    - 性能优势
- 14.4 xv6 自旋锁
  - 14.4.1 struct spinlock 数据结构
  - 14.4.2 acquire() 与 release() 实现
  - 14.4.3 禁用中断防止死锁
  - 14.4.4 锁持有者跟踪（调试）
- 14.5 xv6 睡眠锁
  - 14.5.1 struct sleeplock 数据结构
  - 14.5.2 acquiresleep() 与 releasesleep()
  - 14.5.3 sleep() 与 wakeup() 原语
  - 14.5.4 Lost Wakeup 问题与解决
- 14.6 读写锁（Reader-Writer Lock）
  - 14.6.1 设计动机：多读者单写者
  - 14.6.2 读者优先 vs 写者优先
  - 14.6.3 公平读写锁
  - 14.6.4 Linux rwlock 与 rwsem
- 14.7 锁的性能优化
  - 14.7.1 细粒度锁（Fine-Grained Locking）
  - 14.7.2 锁分拆（Lock Splitting）
  - 14.7.3 顺序锁（Seqlock）
  - 14.7.4 无锁数据结构（Lock-Free）

**交互式组件**：
- `LockImplementationComparison` - 锁实现方式对比
- `FutexMechanism` - futex 快速/慢速路径动画
- `SleeplockWaitQueue` - 睡眠锁等待队列可视化
- `ReaderWriterLockDemo` - 读写锁并发控制演示
- `LockGranularityTradeoff` - 锁粒度权衡分析

---

### **Chapter 15: 信号量与条件变量**
- 15.1 信号量（Semaphore）
  - 15.1.1 Dijkstra 的信号量设计
  - 15.1.2 信号量的定义：整数 + 两个原子操作
  - 15.1.3 P（wait / down）与 V（signal / up）操作
  - 15.1.4 二值信号量 vs 计数信号量
  - 15.1.5 信号量的语义：资源计数
- 15.2 信号量实现
  - 15.2.1 基于自旋锁 + 等待队列
  - 15.2.2 伪代码实现
  - 15.2.3 POSIX 信号量：sem_init()、sem_wait()、sem_post()
  - 15.2.4 Linux 信号量：struct semaphore
- 15.3 信号量使用模式
  - 15.3.1 互斥锁（初值 = 1）
  - 15.3.2 条件同步（初值 = 0）
  - 15.3.3 资源计数（初值 = N）
- 15.4 条件变量（Condition Variable）
  - 15.4.1 设计动机：wait 特定条件
  - 15.4.2 条件变量的接口：wait()、signal()、broadcast()
  - 15.4.3 条件变量 + 锁的配合使用
  - 15.4.4 Mesa 语义 vs Hoare 语义
  - 15.4.5 虚假唤醒（Spurious Wakeup）与 while 循环
- 15.5 Pthread 条件变量
  - 15.5.1 pthread_cond_t 类型
  - 15.5.2 pthread_cond_wait() 的原子性
  - 15.5.3 pthread_cond_signal() vs pthread_cond_broadcast()
  - 15.5.4 使用示例与常见错误
- 15.6 xv6 sleep/wakeup
  - 15.6.1 sleep() 与 wakeup() 原语
  - 15.6.2 睡眠通道（sleep channel）
  - 15.6.3 Lost Wakeup 问题
  - 15.6.4 锁的传递与重新获取

**交互式组件**：
- `SemaphoreVisualizer` - 信号量操作可视化
- `ConditionVariableFlow` - 条件变量等待/唤醒流程动画
- `MesaVsHoareSemantics` - Mesa 与 Hoare 语义对比
- `SleepWakeupMechanism` - xv6 sleep/wakeup 机制演示

---

### **Chapter 16: 经典同步问题**
- 16.1 生产者-消费者问题
  - 16.1.1 问题描述：有界缓冲区
  - 16.1.2 信号量解决方案
    - empty：空槽位计数
    - full：满槽位计数
    - mutex：互斥访问缓冲区
  - 16.1.3 完整伪代码与正确性分析
  - 16.1.4 条件变量解决方案
  - 16.1.5 xv6 pipe 实现分析
- 16.2 读者-写者问题
  - 16.2.1 问题描述：多读者单写者
  - 16.2.2 第一类读者-写者问题：读者优先
    - 解决方案与伪代码
    - 写者饥饿问题
  - 16.2.3 第二类读者-写者问题:写者优先
  - 16.2.4 公平解决方案
  - 16.2.5 Linux RCU（Read-Copy-Update）机制
- 16.3 哲学家就餐问题
  - 16.3.1 问题描述：资源竞争与死锁
  - 16.3.2 错误方案：简单获取两个叉子（死锁）
  - 16.3.3 破坏循环等待：非对称方案
  - 16.3.4 最多 N-1 个哲学家同时就餐
  - 16.3.5 条件变量方案
  - 16.3.6 Chandy-Misra 方案
- 16.4 睡理发师问题
  - 16.4.1 问题描述：等待队列管理
  - 16.4.2 信号量解决方案
  - 16.4.3 伪代码实现
- 16.5 吸烟者问题
  - 16.5.1 问题描述与约束
  - 16.5.2 信号量解决方案

**交互式组件**：
- `ProducerConsumerSimulator` - 生产者-消费者动画模拟器
- `ReaderWriterVisualizer` - 读者-写者并发访问可视化
- `DiningPhilosophersAnimation` - 哲学家就餐问题动画
- `BarberShopSimulation` - 睡理发师问题模拟

---

## Part VI: 死锁问题 (Deadlock)

### **Chapter 17: 死锁基础**
- 17.1 死锁的概念
  - 17.1.1 死锁的定义：永久性循环等待
  - 17.1.2 死锁示例：两个进程、两个锁
  - 17.1.3 死锁 vs 活锁 vs 饥饿
- 17.2 死锁的必要条件
  - 17.2.1 互斥（Mutual Exclusion）
  - 17.2.2 持有并等待（Hold and Wait）
  - 17.2.3 非抢占（No Preemption）
  - 17.2.4 循环等待（Circular Wait）
  - 17.2.5 四个条件的必要性证明
- 17.3 资源分配图
  - 17.3.1 图的表示：进程节点、资源节点、边
  - 17.3.2 请求边 vs 分配边
  - 17.3.3 死锁检测：环的存在
  - 17.3.4 单实例资源 vs 多实例资源
  - 17.3.5 图的演进示例
- 17.4 死锁处理策略
  - 17.4.1 预防（Prevention）：破坏必要条件
  - 17.4.2 避免（Avoidance）：动态检查安全性
  - 17.4.3 检测与恢复（Detection & Recovery）
  - 17.4.4 忽略（Ostrich Algorithm）：鸵鸟算法

**交互式组件**：
- `DeadlockScenarioDemo` - 死锁场景交互式演示
- `ResourceAllocationGraph` - 资源分配图动态生成器
- `DeadlockConditionAnalyzer` - 死锁必要条件分析工具

---

### **Chapter 18: 死锁预防、避免与检测**
- 18.1 死锁预防
  - 18.1.1 破坏互斥：不可行（某些资源必须互斥）
  - 18.1.2 破坏持有并等待：
    - 方案 1：一次性请求所有资源
    - 方案 2：释放已持有资源再请求
    - 缺点：资源利用率低、可能饥饿
  - 18.1.3 破坏非抢占：
    - 抢占已分配资源
    - 适用于可保存/恢复状态的资源
  - 18.1.4 破坏循环等待：
    - 资源排序（Resource Ordering）
    - 所有进程按顺序请求资源
    - 证明：无环
- 18.2 死锁避免
  - 18.2.1 安全状态 vs 不安全状态
  - 18.2.2 安全序列（Safe Sequence）
  - 18.2.3 银行家算法（Banker's Algorithm）
    - 数据结构：Available、Max、Allocation、Need
    - 安全性检查算法
    - 资源请求算法
    - 完整示例与手算
    - 时间复杂度：O(m × n²)
  - 18.2.4 银行家算法的缺点：
    - 需要预先知道最大需求
    - 进程数量固定
    - 资源数量固定
- 18.3 死锁检测
  - 18.3.1 单实例资源：等待图（Wait-For Graph）
    - 环检测算法：DFS
    - 时间复杂度：O(n²)
  - 18.3.2 多实例资源：类似银行家算法
  - 18.3.3 检测时机：定期 vs 每次请求
  - 18.3.4 检测开销分析
- 18.4 死锁恢复
  - 18.4.1 进程终止
    - 终止所有死锁进程
    - 逐个终止直到解除死锁
    - 选择牺牲进程：优先级、运行时间、资源持有量
  - 18.4.2 资源抢占
    - 选择牺牲进程
    - 回滚（Rollback）
    - 饥饿问题：限制抢占次数
- 18.5 实际系统的选择
  - 18.5.1 数据库：死锁检测 + 事务回滚
  - 18.5.2 操作系统：资源排序（预防）
  - 18.5.3 嵌入式系统：静态分析

**交互式组件**：
- `BankerAlgorithmSimulator` - 银行家算法交互式模拟器
- `SafeSequenceFinder` - 安全序列查找可视化
- `WaitForGraphDetector` - 等待图死锁检测动画
- `DeadlockRecoveryStrategy` - 死锁恢复策略对比

---

## Part VII: 持久化：文件系统 (Persistence: File Systems)

### **Chapter 19: 文件系统基础**
- 19.1 持久化存储概述
  - 19.1.1 为什么需要持久化？数据超越进程生命周期
  - 19.1.2 存储设备：HDD、SSD、NVMe
  - 19.1.3 文件系统的作用：组织、命名、保护、共享
- 19.2 文件抽象
  - 19.2.1 文件的定义：字节序列 vs 记录序列
  - 19.2.2 文件类型：普通文件、目录、设备文件、链接
  - 19.2.3 文件属性：名称、大小、时间戳、权限、所有者
  - 19.2.4 文件操作：创建、打开、读、写、关闭、删除、定位
- 19.3 目录抽象
  - 19.3.1 目录的作用：文件命名与组织
  - 19.3.2 单级目录 vs 多级目录
  - 19.3.3 树形目录结构
  - 19.3.4 路径：绝对路径 vs 相对路径
  - 19.3.5 当前工作目录（cwd）
  - 19.3.6 目录操作：创建、删除、遍历
- 19.4 文件系统接口
  - 19.4.1 POSIX 文件 API：
    - open()、close()、read()、write()
    - lseek()、fsync()、stat()
  - 19.4.2 文件描述符（File Descriptor）
  - 19.4.3 文件偏移量（File Offset）
  - 19.4.4 打开文件表（Open File Table）
    - 进程级：文件描述符表
    - 系统级：打开文件表
    - inode 表
- 19.5 文件共享
  - 19.5.1 多进程共享文件
  - 19.5.2 fork() 后的文件描述符继承
  - 19.5.3 dup() 与 dup2()：文件描述符复制
  - 19.5.4 文件锁：advisory vs mandatory

**交互式组件**：
- `FileSystemAbstraction` - 文件系统抽象层次图
- `DirectoryTreeVisualizer` - 目录树交互式可视化
- `FileDescriptorTable` - 文件描述符表结构演示
- `FileOffsetSimulator` - 文件偏移量操作模拟器

---

### **Chapter 20: 文件系统实现**
- 20.1 文件系统布局
  - 20.1.1 磁盘分区与文件系统
  - 20.1.2 引导块（Boot Block）
  - 20.1.3 超级块（Superblock）：文件系统元数据
  - 20.1.4 inode 区域
  - 20.1.5 数据块区域
  - 20.1.6 空闲空间管理
- 20.2 inode（索引节点）
  - 20.2.1 inode 的作用：文件元数据
  - 20.2.2 inode 结构：
    - 文件类型与权限
    - 链接计数
    - 所有者与组
    - 文件大小
    - 时间戳：atime、mtime、ctime
    - 数据块指针
  - 20.2.3 inode 编号与 inode 表
  - 20.2.4 inode vs 文件名的分离
- 20.3 数据块索引
  - 20.3.1 直接指针（Direct Pointers）：12 个
  - 20.3.2 间接指针（Indirect Pointer）：单级间接
  - 20.3.3 二级间接指针（Double Indirect）
  - 20.3.4 三级间接指针（Triple Indirect）
  - 20.3.5 最大文件大小计算
  - 20.3.6 多级索引的优缺点
- 20.4 目录实现
  - 20.4.1 目录 = 特殊文件
  - 20.4.2 目录项（Directory Entry）：inode 号 + 文件名
  - 20.4.3 线性列表 vs 哈希表
  - 20.4.4 目录查找过程：路径解析
  - 20.4.5 . 与 .. 特殊目录项
- 20.5 空闲空间管理
  - 20.5.1 位图（Bitmap）：inode 位图 + 数据块位图
  - 20.5.2 空闲链表（Free List）
  - 20.5.3 分组（Grouping）
  - 20.5.4 计数（Counting）
  - 20.5.5 性能对比

**交互式组件**：
- `FileSystemLayoutVisualizer` - 文件系统磁盘布局图
- `InodeStructureExplorer` - inode 结构交互式查看器
- `MultiLevelIndexing` - 多级索引寻址动画
- `DirectoryLookup` - 目录查找路径解析可视化

---

### **Chapter 21: xv6 文件系统剖析**
- 21.1 xv6 文件系统概述
  - 21.1.1 设计目标：简单、教学友好
  - 21.1.2 文件系统层次：
    - 磁盘层（disk layer）
    - 缓冲层（buffer cache layer）
    - 日志层（logging layer）
    - inode 层（inode layer）
    - 目录层（directory layer）
    - 路径名层（pathname layer）
    - 文件描述符层（file descriptor layer）
- 21.2 磁盘布局
  - 21.2.1 块大小：512 字节
  - 21.2.2 超级块（block 1）
  - 21.2.3 日志块（log blocks）
  - 21.2.4 inode 块
  - 21.2.5 位图块（bitmap blocks）
  - 21.2.6 数据块（data blocks）
- 21.3 缓冲区缓存（Buffer Cache）
  - 21.3.1 struct buf 数据结构
  - 21.3.2 bread() 与 bwrite()
  - 21.3.3 LRU 替换策略
  - 21.3.4 缓冲区锁（buf.lock）
  - 21.3.5 缓冲区缓存的作用：
    - 同步访问磁盘块
    - 缓存常用块
    - 协调块的修改
- 21.4 inode 层
  - 21.4.1 struct dinode（磁盘 inode）
  - 21.4.2 struct inode（内存 inode）
  - 21.4.3 ialloc() 与 iget()
  - 21.4.4 ilock() 与 iunlock()
  - 21.4.5 bmap()：逻辑块号 → 物理块号
  - 21.4.6 readi() 与 writei()
- 21.5 目录层
  - 21.5.1 struct dirent 目录项
  - 21.5.2 dirlookup()：目录查找
  - 21.5.3 dirlink()：添加目录项
- 21.6 路径名层
  - 21.6.1 namei() 与 nameiparent()
  - 21.6.2 路径解析过程
  - 21.6.3 符号链接处理（xv6 不支持）
- 21.7 文件描述符层
  - 21.7.1 struct file 数据结构
  - 21.7.2 sys_open() 实现
  - 21.7.3 sys_read() 与 sys_write()
  - 21.7.4 sys_close()

**交互式组件**：
- `Xv6FileSystemLayers` - xv6 文件系统层次图
- `Xv6DiskLayout` - xv6 磁盘布局可视化
- `BufferCacheSimulator` - 缓冲区缓存 LRU 模拟器
- `Xv6InodeTraversal` - xv6 inode 操作追踪
- `Xv6PathResolution` - xv6 路径解析步进演示

---

### **Chapter 22: 崩溃一致性与日志**
- 22.1 崩溃一致性问题
  - 22.1.1 什么是崩溃一致性？文件系统的完整性
  - 22.1.2 崩溃场景：断电、系统崩溃、内核 panic
  - 22.1.3 不一致的例子：
    - 数据块已写，inode/位图未更新
    - inode 已更新，数据块/位图未更新
    - 位图已更新，inode/数据块未更新
  - 22.1.4 文件系统检查工具：fsck
    - 超级块检查
    - inode 检查
    - 块分配检查
    - 目录检查
    - fsck 的局限性：慢、可能丢失数据
- 22.2 日志结构文件系统（LFS）
  - 22.2.1 设计动机：优化写性能
  - 22.2.2 核心思想：顺序写日志
  - 22.2.3 inode map 与 checkpoint region
  - 22.2.4 垃圾回收（Garbage Collection）
  - 22.2.5 段清理（Segment Cleaning）
- 22.3 写前日志（WAL / Journaling）
  - 22.3.1 日志的基本思想：先记录意图
  - 22.3.2 日志流程：
    - 日志写（Journal Write）
    - 日志提交（Journal Commit）
    - 检查点（Checkpoint）：写入实际位置
    - 释放日志
  - 22.3.3 恢复过程：重放日志
  - 22.3.4 日志模式：
    - 数据日志（Data Journaling）
    - 元数据日志（Metadata Journaling）
    - 有序模式（Ordered Mode）
  - 22.3.5 性能优化：批处理、异步提交
- 22.4 xv6 日志系统
  - 22.4.1 struct log 数据结构
  - 22.4.2 begin_op() 与 end_op()
  - 22.4.3 log_write()：延迟写入
  - 22.4.4 commit()：提交事务
  - 22.4.5 recover_from_log()：启动时恢复
  - 22.4.6 组提交（Group Commit）
- 22.5 现代文件系统的崩溃一致性
  - 22.5.1 ext3/ext4 的日志
  - 22.5.2 XFS 的日志
  - 22.5.3 Btrfs 的写时复制（Copy-on-Write）
  - 22.5.4 ZFS 的事务对象

**交互式组件**：
- `CrashConsistencyScenarios` - 崩溃一致性场景演示
- `JournalingProtocol` - 日志写入流程动画
- `Xv6LogMechanism` - xv6 日志机制可视化
- `WALRecoveryProcess` - WAL 恢复过程模拟器

---

## Part VIII: 持久化：I/O 与存储 (Persistence: I/O & Storage)

### **Chapter 23: I/O 设备与驱动**
- 23.1 I/O 设备基础
  - 23.1.1 设备分类：块设备 vs 字符设备
  - 23.1.2 设备特性：速度、传输单位、访问方式
  - 23.1.3 I/O 接口：寄存器、数据、控制、状态
- 23.2 I/O 方式
  - 23.2.1 轮询（Polling）
    - 优点：简单
    - 缺点：CPU 浪费
    - 适用场景：快速设备
  - 23.2.2 中断驱动（Interrupt-Driven）
    - 中断处理流程
    - 优点：CPU 利用率高
    - 缺点：中断开销、活锁（livelock）
  - 23.2.3 直接内存访问（DMA）
    - DMA 控制器
    - 减少 CPU 参与
    - 适用场景：大块数据传输
  - 23.2.4 三种方式的对比
- 23.3 设备驱动程序
  - 23.3.1 驱动程序的作用：硬件抽象
  - 23.3.2 驱动程序接口：read()、write()、ioctl()
  - 23.3.3 Linux 设备驱动模型
  - 23.3.4 xv6 设备驱动示例：UART、磁盘
- 23.4 I/O 缓冲
  - 23.4.1 单缓冲、双缓冲、环形缓冲
  - 23.4.2 缓冲的作用：匹配速度差异
  - 23.4.3 缓冲池管理
- 23.5 I/O 调度
  - 23.5.1 I/O 请求队列
  - 23.5.2 调度目标：吞吐量、延迟、公平性

**交互式组件**：
- `IOMethodComparison` - I/O 方式对比动画
- `DMATransferVisualizer` - DMA 传输过程可视化
- `DeviceDriverInterface` - 设备驱动接口示意图
- `IOBufferingDemo` - I/O 缓冲机制演示

---

### **Chapter 24: 磁盘与存储管理**
- 24.1 硬盘驱动器（HDD）
  - 24.1.1 HDD 物理结构：盘片、磁道、扇区、柱面
  - 24.1.2 磁盘臂（Disk Arm）与寻道
  - 24.1.3 旋转延迟（Rotational Delay）
  - 24.1.4 访问时间计算：寻道 + 旋转 + 传输
  - 24.1.5 磁盘性能特性
- 24.2 磁盘调度算法
  - 24.2.1 FCFS（First-Come First-Served）
  - 24.2.2 SSTF（Shortest Seek Time First）
    - 贪心策略
    - 饥饿问题
  - 24.2.3 SCAN（电梯算法）
    - 单向扫描
    - 公平性改进
  - 24.2.4 C-SCAN（Circular SCAN）
    - 单向服务
    - 更好的公平性
  - 24.2.5 LOOK 与 C-LOOK
    - 不到端点折返
  - 24.2.6 调度算法性能对比
    - 平均寻道时间
    - 方差（公平性）
    - 吞吐量
- 24.3 固态硬盘（SSD）
  - 24.3.1 闪存（Flash Memory）基础
  - 24.3.2 读、写、擦除操作
  - 24.3.3 写放大（Write Amplification）
  - 24.3.4 闪存转换层（FTL）
  - 24.3.5 磨损均衡（Wear Leveling）
  - 24.3.6 垃圾回收（Garbage Collection）
  - 24.3.7 TRIM 命令
  - 24.3.8 SSD vs HDD 性能对比
- 24.4 RAID（磁盘阵列）
  - 24.4.1 RAID 的目标：性能 + 可靠性
  - 24.4.2 RAID 0（条带化）：性能
  - 24.4.3 RAID 1（镜像）：可靠性
  - 24.4.4 RAID 4（专用校验盘）
  - 24.4.5 RAID 5（分布式校验）
  - 24.4.6 RAID 6（双校验）
  - 24.4.7 RAID 10（1+0）
  - 24.4.8 RAID 级别对比表
- 24.5 Linux I/O 栈
  - 24.5.1 VFS（Virtual File System）
  - 24.5.2 Page Cache
  - 24.5.3 Block Layer
  - 24.5.4 I/O Scheduler：CFQ、Deadline、Noop、BFQ
  - 24.5.5 Device Driver Layer

**交互式组件**：
- `HDDStructureVisualizer` - HDD 物理结构 3D 可视化
- `DiskSchedulingSimulator` - 磁盘调度算法模拟器（磁盘臂移动动画）
- `SSDOperationDemo` - SSD 读写擦除操作演示
- `RAIDConfigurationComparison` - RAID 级别配置对比工具
- `LinuxIOStackDiagram` - Linux I/O 栈层次图

---

## Part IX: xv6 内核深度剖析 (xv6 Kernel Deep Dive)

### **Chapter 25: xv6 启动与初始化**
- 25.1 xv6 概述
  - 25.1.1 xv6 的定位：教学操作系统
  - 25.1.2 基于 Unix v6 与 ANSI C
  - 25.1.3 RISC-V 架构
  - 25.1.4 源码结构：kernel/、user/、mkfs/
- 25.2 RISC-V 架构基础
  - 25.2.1 RISC-V 寄存器：x0-x31、pc
  - 25.2.2 特权模式：M、S、U
  - 25.2.3 控制状态寄存器（CSR）：
    - stvec：trap 向量
    - sepc：异常 PC
    - scause：异常原因
    - sscratch：临时寄存器
    - satp：页表基址
  - 25.2.4 页表格式：Sv39
- 25.3 启动流程
  - 25.3.1 QEMU 加载 kernel
  - 25.3.2 entry.S：启动代码
    - 设置栈指针
    - 跳转到 start()
  - 25.3.3 start()（kernel/start.c）
    - 设置 M 模式 CSR
    - 进入 S 模式
    - 跳转到 main()
  - 25.3.4 main()（kernel/main.c）
    - consoleinit()：控制台
    - printfinit()：printf 锁
    - kinit()：物理内存分配器
    - kvminit()：内核页表
    - kvminithart()：启用分页
    - procinit()：进程表
    - trapinit() / trapinithart()：trap
    - plicinit() / plicinithart()：中断控制器
    - binit()：缓冲区缓存
    - iinit()：inode 缓存
    - fileinit()：文件表
    - virtio_disk_init()：磁盘
    - userinit()：第一个用户进程
    - scheduler()：永不返回
- 25.4 第一个进程
  - 25.4.1 userinit() 创建 initcode
  - 25.4.2 initcode.S：exec("/init")
  - 25.4.3 init.c：启动 shell

**交互式组件**：
- `RISCV_ArchitectureOverview` - RISC-V 架构概览
- `Xv6BootSequence` - xv6 启动流程时间线
- `MainInitializationFlow` - main() 初始化流程图
- `FirstProcessCreation` - 第一个进程创建动画

---

### **Chapter 26: xv6 陷阱与系统调用**
- 26.1 陷阱（Trap）机制
  - 26.1.1 陷阱的分类：系统调用、异常、中断
  - 26.1.2 陷阱处理目标：
    - 保存状态
    - 进入内核模式
    - 执行处理程序
    - 恢复状态
    - 返回用户态
- 26.2 陷阱处理流程
  - 26.2.1 用户态触发 trap
  - 26.2.2 硬件操作：
    - 切换到 S 模式
    - 保存 pc 到 sepc
    - 写 scause（原因）
    - 跳转到 stvec
  - 26.2.3 uservec（kernel/trampoline.S）
    - 保存用户寄存器到 trapframe
    - 加载内核栈指针
    - 跳转到 usertrap()
  - 26.2.4 usertrap()（kernel/trap.c）
    - 分发：系统调用 vs 异常 vs 中断
  - 26.2.5 usertrapret()
    - 准备返回用户态
    - 调用 userret
  - 26.2.6 userret（kernel/trampoline.S）
    - 恢复用户寄存器
    - sret 返回用户态
- 26.3 系统调用实现
  - 26.3.1 用户态：usys.S
    - ecall 指令
    - 系统调用号在 a7
  - 26.3.2 内核态：syscall()（kernel/syscall.c）
    - 从 a7 获取系统调用号
    - 查表调用对应函数
    - 返回值写入 a0
  - 26.3.3 参数获取：argint()、argaddr()、argstr()
  - 26.3.4 完整示例：sys_write()
- 26.4 trapframe 结构
  - 26.4.1 struct trapframe 字段
  - 26.4.2 每个进程的 trapframe 页
  - 26.4.3 trampoline 页：共享代码
- 26.5 内核态 trap
  - 26.5.1 kernelvec 与 kerneltrap
  - 26.5.2 内核栈上的 trap

**交互式组件**：
- `TrapMechanismFlow` - Trap 完整流程动画
- `TrapframeStructure` - Trapframe 数据结构可视化
- `SystemCallPath` - 系统调用路径追踪
- `UserKernelTransition` - 用户态/内核态切换动画

---

### **Chapter 27: xv6 内存管理**
- 27.1 物理内存管理
  - 27.1.1 空闲链表（free list）
  - 27.1.2 struct run：链表节点
  - 27.1.3 kinit()：初始化空闲链表
  - 27.1.4 kalloc()：分配页
  - 27.1.5 kfree()：释放页
  - 27.1.6 kmem.lock：保护空闲链表
- 27.2 内核页表
  - 27.2.1 直接映射（Direct Mapping）
  - 27.2.2 内核地址空间布局：
    - UART、VIRTIO、PLIC 等设备
    - 内核代码与数据
    - 直接映射物理内存
    - trampoline 页
  - 27.2.3 kvminit()：创建内核页表
  - 27.2.4 kvmmap()：添加映射
  - 27.2.5 walk()：页表遍历
- 27.3 用户页表
  - 27.3.1 每个进程独立页表
  - 27.3.2 用户地址空间布局：
    - 代码段（0 地址开始）
    - 数据段与堆
    - 栈（高地址）
    - trapframe 页
    - trampoline 页
  - 27.3.3 proc_pagetable()：创建用户页表
  - 27.3.4 uvmalloc() 与 uvmdealloc()
  - 27.3.5 copyout() 与 copyin()：跨地址空间复制
- 27.4 fork() 与写时复制
  - 27.4.1 fork() 实现：allocproc() + uvmcopy()
  - 27.4.2 uvmcopy()：复制页表
  - 27.4.3 写时复制优化（xv6 未实现）：
    - 标记页为只读
    - 页故障时复制
    - 引用计数
- 27.5 exec() 内存管理
  - 27.5.1 加载 ELF 文件
  - 27.5.2 uvmalloc()：分配内存
  - 27.5.3 loadseg()：加载段
  - 27.5.4 构建用户栈：argc、argv
  - 27.5.5 切换页表：oldpagetable → newpagetable

**交互式组件**：
- `Xv6PhysicalMemoryAllocator` - xv6 物理内存分配器可视化
- `KernelPageTableLayout` - 内核页表布局图
- `UserPageTableLayout` - 用户页表布局图
- `ForkMemoryCopy` - fork() 内存复制过程动画
- `ExecELFLoading` - exec() ELF 加载流程

---

### **Chapter 28: xv6 进程调度与同步**
- 28.1 进程结构
  - 28.1.1 struct proc 详解
    - 状态：UNUSED、USED、SLEEPING、RUNNABLE、RUNNING、ZOMBIE
    - context：保存寄存器
    - trapframe：用户寄存器
    - pagetable：页表
    - kstack：内核栈
    - chan：睡眠通道
  - 28.1.2 进程表：proc[NPROC]
  - 28.1.3 ptable.lock
- 28.2 调度器
  - 28.2.1 scheduler()（kernel/proc.c）
    - 无限循环
    - 遍历进程表
    - 选择 RUNNABLE 进程
    - swtch() 切换上下文
  - 28.2.2 每个 CPU 的 scheduler
  - 28.2.3 简单的轮转策略
- 28.3 上下文切换
  - 28.3.1 struct context：ra、sp、s0-s11
  - 28.3.2 swtch.S 汇编实现
    - 保存旧 context
    - 加载新 context
    - ret 返回
  - 28.3.3 进程 → scheduler → 新进程
- 28.4 sleep 与 wakeup
  - 28.4.1 sleep(chan, lock)
    - 释放 lock
    - 设置 chan 与 SLEEPING
    - 调用 sched()
    - 唤醒后重新获取 lock
  - 28.4.2 wakeup(chan)
    - 遍历进程表
    - 唤醒匹配 chan 的进程
  - 28.4.3 lost wakeup 问题与解决
- 28.5 锁的使用
  - 28.5.1 ptable.lock：保护进程表
  - 28.5.2 持有锁时不能 sleep
  - 28.5.3 锁顺序：避免死锁
- 28.6 kill 与 wait
  - 28.6.1 kill()：设置 p->killed
  - 28.6.2 在 trap 中检查 killed
  - 28.6.3 wait()：等待 ZOMBIE 子进程
  - 28.6.4 回收子进程资源

**交互式组件**：
- `Xv6ProcessStateTransition` - xv6 进程状态转换动画
- `Xv6SchedulerLoop` - xv6 调度器循环可视化
- `Xv6ContextSwitch` - xv6 上下文切换步进演示
- `SleepWakeupProtocol` - sleep/wakeup 协议动画
- `Xv6LockingRules` - xv6 锁规则与死锁预防

---

## Part X: 现代操作系统主题 (Modern OS Topics)

### **Chapter 29: 多核与并发扩展性**
- 29.1 多核架构
  - 29.1.1 SMP（Symmetric Multi-Processing）
  - 29.1.2 NUMA（Non-Uniform Memory Access）
  - 29.1.3 缓存一致性协议：MESI、MOESI
  - 29.1.4 False Sharing 问题
- 29.2 并发扩展性挑战
  - 29.2.1 Amdahl 定律
  - 29.2.2 锁竞争（Lock Contention）
  - 29.2.3 可扩展性瓶颈分析
- 29.3 细粒度锁
  - 29.3.1 锁分拆（Lock Splitting）
  - 29.3.2 锁分层（Lock Striping）
  - 29.3.3 每 CPU 数据结构
  - 29.3.4 Linux 内核示例：per-CPU 变量
- 29.4 无锁数据结构
  - 29.4.1 无锁队列（Lock-Free Queue）
  - 29.4.2 无锁栈（Lock-Free Stack）
  - 29.4.3 ABA 问题与解决
  - 29.4.4 内存屏障（Memory Barriers）
- 29.5 RCU（Read-Copy-Update）
  - 29.5.1 RCU 的设计思想
  - 29.5.2 读端：rcu_read_lock() / rcu_read_unlock()
  - 29.5.3 写端：同步 RCU
  - 29.5.4 Grace Period
  - 29.5.5 Linux RCU 实现
  - 29.5.6 RCU 适用场景：读多写少

**交互式组件**：
- `NUMAArchitecture` - NUMA 架构示意图
- `CacheCoherenceProtocol` - 缓存一致性协议动画
- `LockContentionAnalyzer` - 锁竞争分析工具
- `RCUMechanism` - RCU 读-复制-更新动画
- `LockFreeQueueDemo` - 无锁队列操作演示

---

### **Chapter 30: 虚拟化技术**
- 30.1 虚拟化概述
  - 30.1.1 虚拟化的定义与动机
  - 30.1.2 虚拟化类型：
    - 全虚拟化（Full Virtualization）
    - 半虚拟化（Paravirtualization）
    - 硬件辅助虚拟化
  - 30.1.3 VMM / Hypervisor：Type 1 vs Type 2
- 30.2 CPU 虚拟化
  - 30.2.1 敏感指令（Sensitive Instructions）
  - 30.2.2 二进制翻译（Binary Translation）
  - 30.2.3 Intel VT-x / AMD-V
    - VMX（Virtual Machine Extensions）
    - VMCS（Virtual Machine Control Structure）
    - VM Entry / VM Exit
  - 30.2.4 Guest 模式与 Host 模式
- 30.3 内存虚拟化
  - 30.3.1 影子页表（Shadow Page Tables）
  - 30.3.2 EPT（Extended Page Table）/ NPT（Nested Page Table）
  - 30.3.3 二级地址翻译：GVA → GPA → HPA
  - 30.3.4 内存气球（Memory Ballooning）
  - 30.3.5 页共享（KSM）
- 30.4 I/O 虚拟化
  - 30.4.1 设备模拟（Device Emulation）
  - 30.4.2 半虚拟化驱动：virtio
  - 30.4.3 SR-IOV（Single Root I/O Virtualization）
  - 30.4.4 直通（Passthrough）：VFIO
- 30.5 容器与轻量级虚拟化
  - 30.5.1 容器 vs 虚拟机
  - 30.5.2 Namespace：隔离视图
    - PID namespace
    - Network namespace
    - Mount namespace
    - UTS namespace
    - IPC namespace
    - User namespace
  - 30.5.3 cgroup：资源限制
    - CPU、内存、I/O 限制
    - cgroup v1 vs v2
  - 30.5.4 Docker 与容器化
  - 30.5.5 安全隔离：seccomp、AppArmor、SELinux

**交互式组件**：
- `VirtualizationArchitecture` - 虚拟化架构对比图
- `EPTTranslation` - EPT 二级地址翻译动画
- `NamespaceIsolation` - Namespace 隔离演示
- `CgroupResourceLimits` - cgroup 资源限制可视化
- `ContainerVsVM` - 容器与虚拟机对比

---

### **Chapter 31: 操作系统安全基础**
- 31.1 安全威胁模型
  - 31.1.1 威胁类型：恶意软件、特权提升、信息泄露
  - 31.1.2 攻击面（Attack Surface）
  - 31.1.3 最小权限原则（Principle of Least Privilege）
- 31.2 访问控制
  - 31.2.1 自主访问控制（DAC）：Unix 权限
  - 31.2.2 强制访问控制（MAC）：SELinux、AppArmor
  - 31.2.3 基于角色的访问控制（RBAC）
  - 31.2.4 Capabilities：细粒度权限
- 31.3 隔离机制
  - 31.3.1 进程隔离：地址空间、权限
  - 31.3.2 用户隔离：UID/GID
  - 31.3.3 容器隔离：Namespace、cgroup
  - 31.3.4 虚拟机隔离：硬件虚拟化
- 31.4 常见漏洞与防护
  - 31.4.1 缓冲区溢出（Buffer Overflow）
    - 栈溢出、堆溢出
    - 防护：ASLR、Stack Canary、DEP/NX
  - 31.4.2 竞态条件（TOCTTOU）
  - 31.4.3 特权提升（Privilege Escalation）
  - 31.4.4 侧信道攻击（Side-Channel）：Spectre、Meltdown
- 31.5 安全增强
  - 31.5.1 ASLR（Address Space Layout Randomization）
  - 31.5.2 KASLR（Kernel ASLR）
  - 31.5.3 Secure Boot
  - 31.5.4 Trusted Execution Environment (TEE)
  - 31.5.5 硬件安全：TPM、SGX

**交互式组件**：
- `AccessControlModels` - 访问控制模型对比
- `BufferOverflowDemo` - 缓冲区溢出攻击演示
- `ASLRVisualization` - ASLR 地址随机化可视化
- `SecurityMechanismsComparison` - 安全机制综合对比

---

### **Chapter 32: 性能分析与优化**
- 32.1 性能指标
  - 32.1.1 吞吐量（Throughput）
  - 32.1.2 延迟（Latency）
  - 32.1.3 利用率（Utilization）
  - 32.1.4 可扩展性（Scalability）
- 32.2 性能分析工具
  - 32.2.1 top、htop：系统概览
  - 32.2.2 vmstat：虚拟内存统计
  - 32.2.3 iostat：I/O 统计
  - 32.2.4 perf：性能事件采样
    - perf stat：计数器
    - perf record / report：采样
    - perf top：实时监控
  - 32.2.5 strace：系统调用追踪
  - 32.2.6 ltrace：库函数调用
  - 32.2.7 火焰图（Flame Graphs）
- 32.3 性能瓶颈定位
  - 32.3.1 CPU 瓶颈：高 CPU 使用率
  - 32.3.2 内存瓶颈：频繁页故障、交换
  - 32.3.3 I/O 瓶颈：高 iowait
  - 32.3.4 锁竞争：高上下文切换率
- 32.4 优化技术
  - 32.4.1 算法优化：更好的时间复杂度
  - 32.4.2 缓存优化：局部性、预取
  - 32.4.3 并发优化：减少锁竞争
  - 32.4.4 I/O 优化：批处理、异步
  - 32.4.5 编译器优化：-O2、-O3、PGO
- 32.5 Linux 性能调优
  - 32.5.1 调度器参数
  - 32.5.2 内存管理参数：vm.swappiness
  - 32.5.3 网络栈调优
  - 32.5.4 文件系统挂载选项

**交互式组件**：
- `PerformanceMetricsDashboard` - 性能指标仪表盘
- `PerfToolsWorkflow` - perf 工具使用流程
- `FlameGraphInteractive` - 交互式火焰图
- `BottleneckIdentifier` - 性能瓶颈识别工具

---

### **Chapter 33: 分布式操作系统概念**
- 33.1 分布式系统基础
  - 33.1.1 分布式系统的定义与挑战
  - 33.1.2 网络通信：消息传递
  - 33.1.3 部分失败（Partial Failure）
  - 33.1.4 CAP 定理：一致性、可用性、分区容错
- 33.2 分布式文件系统
  - 33.2.1 NFS（Network File System）
    - 设计目标：透明性
    - 无状态协议
    - 缓存一致性
  - 33.2.2 AFS（Andrew File System）
    - 全文件缓存
    - 回调机制
  - 33.2.3 GFS（Google File System）
    - Master-Chunk 架构
    - 大文件、顺序写优化
  - 33.2.4 HDFS（Hadoop Distributed File System）
- 33.3 时钟与时间
  - 33.3.1 物理时钟同步：NTP
  - 33.3.2 逻辑时钟：Lamport 时钟
  - 33.3.3 向量时钟（Vector Clocks）
  - 33.3.4 因果关系（Causality）
- 33.4 一致性模型
  - 33.4.1 强一致性（Strong Consistency）
  - 33.4.2 最终一致性（Eventual Consistency）
  - 33.4.3 因果一致性（Causal Consistency）
- 33.5 分布式协调
  - 33.5.1 共识问题（Consensus）
  - 33.5.2 Paxos 算法
  - 33.5.3 Raft 算法
  - 33.5.4 ZooKeeper：分布式协调服务

**交互式组件**：
- `DistributedFileSystemArchitecture` - 分布式文件系统架构图
- `LamportClockVisualization` - Lamport 时钟可视化
- `RaftConsensusAnimation` - Raft 共识算法动画
- `CAPTheoremExplorer` - CAP 定理交互式演示

---

## Part XI: 高级主题与研究方向 (Advanced Topics & Research)

### **Chapter 34: 实时操作系统**
- 34.1 实时系统特性
  - 34.1.1 硬实时 vs 软实时
  - 34.1.2 确定性（Determinism）
  - 34.1.3 截止时间（Deadline）
  - 34.1.4 抖动（Jitter）
- 34.2 实时调度算法
  - 34.2.1 速率单调调度（RMS）
  - 34.2.2 最早截止时间优先（EDF）
  - 34.2.3 可调度性分析：Liu & Layland 边界
  - 34.2.4 优先级反转与优先级继承
- 34.3 实时操作系统示例
  - 34.3.1 VxWorks
  - 34.3.2 FreeRTOS
  - 34.3.3 QNX
  - 34.3.4 PREEMPT_RT Linux
- 34.4 实时系统设计
  - 34.4.1 中断延迟最小化
  - 34.4.2 可抢占内核
  - 34.4.3 内存锁定
  - 34.4.4 CPU 亲和性与隔离

**交互式组件**：
- `RealTimeSchedulingSimulator` - 实时调度模拟器
- `SchedulabilityAnalysis` - 可调度性分析工具
- `PriorityInversionDemo` - 优先级反转演示
- `RTOSComparison` - 实时操作系统对比表

---

### **Chapter 35: 操作系统研究前沿**
- 35.1 新硬件架构
  - 35.1.1 非易失性内存（NVM）：持久内存
  - 35.1.2 FPGA 加速
  - 35.1.3 异构计算：CPU + GPU + TPU
  - 35.1.4 量子计算的操作系统需求
- 35.2 内核架构演进
  - 35.2.1 Unikernel：库操作系统
  - 35.2.2 用户态驱动
  - 35.2.3 eBPF：安全可编程内核
  - 35.2.4 内核旁路（Kernel Bypass）：DPDK、SPDK
- 35.3 形式化验证
  - 35.3.1 seL4：经过形式化验证的微内核
  - 35.3.2 Coq、Isabelle/HOL 定理证明器
  - 35.3.3 可信计算基（TCB）最小化
- 35.4 能效与绿色计算
  - 35.4.1 动态电压频率调整（DVFS）
  - 35.4.2 CPU 休眠状态（C-States）
  - 35.4.3 能耗感知调度
- 35.5 人工智能与操作系统
  - 35.5.1 机器学习辅助的系统优化
  - 35.5.2 智能资源管理
  - 35.5.3 自适应系统

**交互式组件**：
- `NVMArchitecture` - 持久内存架构示意图
- `Unikernel VsTraditional` - Unikernel 与传统内核对比
- `eBPFPipeline` - eBPF 执行流程可视化
- `EnergyAwareScheduling` - 能效感知调度演示

---

### **Chapter 36: 操作系统设计哲学与未来**
- 36.1 设计原则回顾
  - 36.1.1 简单性（Simplicity）：KISS 原则
  - 36.1.2 模块化（Modularity）
  - 36.1.3 机制与策略分离
  - 36.1.4 端到端论证（End-to-End Argument）
- 36.2 Unix 哲学
  - 36.2.1 一切皆文件
  - 36.2.2 小工具组合
  - 36.2.3 文本流接口
- 36.3 经典论文导读
  - 36.3.1 "The UNIX Time-Sharing System" (Ritchie & Thompson, 1974)
  - 36.3.2 "Lottery Scheduling" (Waldspurger & Weihl, 1994)
  - 36.3.3 "The Google File System" (Ghemawat et al., 2003)
  - 36.3.4 "seL4: Formal Verification of an OS Kernel" (Klein et al., 2009)
- 36.4 未来趋势
  - 36.4.1 软硬件协同设计
  - 36.4.2 边缘计算操作系统
  - 36.4.3 量子操作系统
  - 36.4.4 人机共生系统
- 36.5 学习建议与资源
  - 36.5.1 推荐阅读：教材、论文、博客
  - 36.5.2 实践项目：xv6 labs、OS competitions
  - 36.5.3 开源社区参与
  - 36.5.4 面试与职业发展

**交互式组件**：
- `DesignPrinciplesMap` - 操作系统设计原则思维导图
- `ClassicPapersTimeline` - 经典论文时间线
- `FutureTrendsVisualization` - 未来趋势可视化
- `LearningPathGuide` - 学习路径指南

---

## 附录（Appendices）

### **Appendix A: 常见问题与调试技巧**
- A.1 xv6 常见问题
  - A.1.1 编译错误
  - A.1.2 QEMU 启动失败
  - A.1.3 Panic 调试
  - A.1.4 GDB 调试技巧
- A.2 系统编程陷阱
  - A.2.1 并发 Bug：竞态条件、死锁
  - A.2.2 内存错误：泄漏、越界、悬空指针
  - A.2.3 文件描述符泄漏
- A.3 性能调试
  - A.3.1 CPU 占用高
  - A.3.2 内存泄漏定位：valgrind
  - A.3.3 死锁检测

---

### **Appendix B: 面试高频题库**
- B.1 进程与线程
  - B.1.1 进程与线程的区别？
  - B.1.2 fork() 的实现原理？
  - B.1.3 僵尸进程与孤儿进程？
  - B.1.4 写时复制的优势？
- B.2 内存管理
  - B.2.1 虚拟内存的作用？
  - B.2.2 页表的多级设计为什么？
  - B.2.3 TLB 的作用？
  - B.2.4 页面置换算法对比？
- B.3 并发与同步
  - B.3.1 互斥锁与信号量的区别？
  - B.3.2 死锁的四个必要条件？
  - B.3.3 哲学家就餐问题解法？
  - B.3.4 读写锁的实现？
- B.4 文件系统
  - B.4.1 inode 的作用？
  - B.4.2 硬链接与软链接的区别？
  - B.4.3 日志文件系统的优势？
- B.5 调度算法
  - B.5.1 RR 与 MLFQ 的区别？
  - B.5.2 CFS 如何实现公平性？
  - B.5.3 实时调度算法？

---

### **Appendix C: 系统设计题精选**
- C.1 设计一个支持 COW 的 fork
- C.2 实现一个简单的内存分配器
- C.3 设计一个高性能的文件系统缓存
- C.4 实现一个读写锁
- C.5 设计一个多级页表系统
- C.6 实现 LRU 页面置换算法
- C.7 设计一个死锁检测算法

---

### **Appendix D: 扩展阅读资源**
- D.1 经典教材
  - OSTEP（在线免费）
  - Operating System Concepts（恐龙书）
  - Modern Operating Systems（Tanenbaum）
- D.2 在线课程
  - MIT 6.S081 (xv6)
  - UC Berkeley CS162
  - Stanford CS140
- D.3 论文与博客
  - SOSP、OSDI 会议论文
  - LWN.net（Linux 内核新闻）
  - Brendan Gregg 的性能博客
- D.4 实践资源
  - xv6 源码与 labs
  - Linux 内核源码
  - OSDev Wiki

---

### **Appendix E: xv6 源码导读**
- E.1 源码结构
  - E.1.1 kernel/ 目录
  - E.1.2 user/ 目录
  - E.1.3 关键文件列表
- E.2 代码阅读路径
  - E.2.1 启动流程
  - E.2.2 系统调用路径
  - E.2.3 进程调度
  - E.2.4 文件系统
- E.3 代码风格与约定
  - E.3.1 命名规范
  - E.3.2 锁的使用规则
  - E.3.3 注释约定

---

**交互式组件总结**：
- 每个章节配备 2-5 个交互式可视化组件
- 总计约 150+ 个交互式组件
- 涵盖动画演示、模拟器、分析工具、对比表等多种形式
- 支持步进执行、参数调整、实时反馈

**教学目标**：
- 从零基础到系统研究水平的完整路径
- 理论与实践深度结合：教材 + xv6 + Linux
- 强调可视化理解复杂机制
- 包含大量面试题、系统设计题、手算题
- 培养系统思维与工程实践能力
