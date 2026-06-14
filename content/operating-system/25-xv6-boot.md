---
title: "Chapter 25: xv6 启动与初始化"
description: "深入理解 RISC-V 架构基础与特权模式，掌握 xv6 从 entry.S 到 main() 的完整启动流程"
updated: "2026-06-11"
---

# Chapter 25: xv6 启动与初始化

> **本章目标**：
> - 理解 xv6 操作系统的定位与源码结构
> - 掌握 RISC-V 架构基础：寄存器、特权模式、CSR 寄存器、Sv39 页表
> - 深入理解从 QEMU 加载内核到 `main()` 函数的完整启动流程
> - 掌握 `main()` 中所有初始化函数的作用与顺序
> - 理解第一个进程（init 进程）的创建与启动过程

---

## 25.1 xv6 概述

### 25.1.1 什么是 xv6

xv6 是由 MIT（麻省理工学院）开发的一个**教学用操作系统**，其设计目标是让操作系统课程的学生能够通过阅读和修改一个简单但完整的操作系统源码来深入理解操作系统的内部工作原理。

xv6 的核心特征：

| 特征 | 说明 |
|------|------|
| 基础 | 基于 Unix V6（1976 年的 Unix 第 6 版）重新实现 |
| 架构 | RISC-V 64 位（2020 年后版本，早期版本为 x86） |
| 代码量 | 约 10,000 行 C 代码 + 汇编 |
| 许可 | MIT 开源许可 |
| 教学 | MIT 6.S081（操作系统工程）课程的核心实验平台 |

**为什么选择 xv6 来学习操作系统？**

1. **代码量小**：整个内核可以用几天时间通读，不像 Linux 内核有数百万行
2. **设计经典**：保留了 Unix 的核心设计思想，这些思想至今仍在现代 OS 中使用
3. **可实验**：MIT 课程提供了 lab（xv6 labs），学生可以直接修改内核代码
4. **RISC-V**：现代 ISA，比 x86 简洁得多，更适合教学

### 25.1.2 Unix V6 与 xv6 的关系

Unix V6 是 1976 年在 PDP-11 上运行的 Unix 版本，由 Ken Thompson 和 Dennis Ritchie 编写。xv6 用现代 C 语言和 RISC-V 架构重新实现了 Unix V6 的核心功能：

```
Unix V6 (1976)                    xv6 (2006+, 2020+)
─────────────────                  ─────────────────────
语言：C (K&R)                      语言：ANSI C
架构：PDP-11                       架构：x86 → RISC-V
功能：Unix 核心系统调用             功能：Unix 核心系统调用
教学：Unix 源码本身是教材            教学：MIT 6.S081 课程教材

继承关系：
- 进程模型（fork/exec/wait）
- 文件系统（inode、目录、路径名）
- 文件描述符（fd table）
- Shell（命令行解释器）
- 管道（pipe）
```

### 25.1.3 源码结构

xv6-riscv 的源码组织非常清晰，主要分为三个顶层目录：

```
xv6-riscv/
├── kernel/              # 内核代码（所有内核态运行的代码）
│   ├── entry.S          # 汇编入口，启动第一个 CPU
│   ├── main.c           # C 语言入口，调用所有初始化函数
│   ├── start.c          # 早期硬件设置（时钟中断等）
│   ├── proc.c           # 进程管理（fork, exit, wait, sched）
│   ├── trap.c           # 陷阱处理（中断、异常）
│   ├── vm.c             # 虚拟内存管理（页表、mappages）
│   ├── syscall.c        # 系统调用分发
│   ├── sysfile.c        # 文件系统相关系统调用
│   ├── sysproc.c        # 进程相关系统调用
│   ├── file.c           # 文件描述符层
│   ├── fs.c             # 文件系统核心（inode, 目录, 路径名）
│   ├── log.c            # 日志层（crash consistency）
│   ├── bio.c            # 块 I/O 缓冲区缓存
│   ├── ide.c / virtio_disk.c  # 磁盘驱动
│   ├── plic.c           # 平台级中断控制器
│   ├── console.c        # 控制台驱动
│   ├── uart.c           # UART 串口驱动
│   ├── swtch.S          # 上下文切换汇编
│   ├── spinlock.c       # 自旋锁
│   ├── sleeplock.c      # 睡眠锁
│   ├── string.c         # 字符串工具函数
│   ├── printf.c         # 内核 printf
│   ├── defs.h           # 函数声明
│   ├── types.h          # 类型定义
│   ├── param.h          # 系统参数
│   ├── memlayout.h      # 内存布局
│   ├── riscv.h          # RISC-V 特殊指令/CPU 操作
│   └── Makefile         # 构建系统
│
├── user/                # 用户程序
│   ├── init.c           # 第一个用户进程（init 进程）
│   ├── initcode.S       # init 进程的汇编启动代码
│   ├── sh.c             # Shell（命令行解释器）
│   ├── cat.c            # cat 命令
│   ├── ls.c             # ls 命令
│   ├── echo.c           # echo 命令
│   ├── grep.c           # grep 命令
│   ├── rm.c             # rm 命令
│   ├── mkdir.c          # mkdir 命令
│   ├── ulib.c           # 用户库函数
│   ├── umalloc.c        # 用户态 malloc
│   └── user.h           # 用户态头文件
│
├── mkfs/                # 文件系统镜像工具
│   └── mkfs.c           # 构建初始文件系统镜像（fs.img）
│
├── Makefile             # 顶层构建脚本
└── README.md            # 说明文档
```

**kernel/ 目录**：包含所有在内核态（Supervisor mode）运行的代码。这是 OS 的核心。

**user/ 目录**：包含所有在用户态（User mode）运行的程序，包括 init 进程、shell 和各种命令。

**mkfs/ 目录**：`mkfs.c` 是一个运行在宿主机上的工具程序，用于构建初始的文件系统镜像 `fs.img`。它将 user/ 目录中的程序打包进一个模拟的磁盘镜像中。

### 25.1.4 构建与运行

xv6 使用 Makefile 构建，核心目标：

```bash
# 编译内核和用户程序，生成 fs.img 和 kernel
$ make

# 在 QEMU 中启动 xv6
$ make qemu

# 调试模式（GDB 可连接）
$ make qemu-gdb
```

构建产物：

```
kernel          → 内核 ELF 可执行文件
kernel.asm      → 内核反汇编（调试用）
fs.img          → 文件系统镜像（包含 user/ 中的程序）
```

---

## 25.2 RISC-V 架构基础

### 25.2.1 RISC-V 寄存器

RISC-V 是一种精简指令集（RISC）架构，xv6 使用的是 RV64（64 位）变体。RISC-V 有 32 个通用整数寄存器（x0-x31）和一个程序计数器（PC）：

```
┌────────────────────────────────────────────────────────────┐
│               RISC-V 通用寄存器 (RV64)                     │
├───────┬──────────┬─────────────────────────────────────────┤
│ 寄存器 │ ABI 别名  │ 用途                                    │
├───────┼──────────┼─────────────────────────────────────────┤
│  x0   │  zero    │ 硬连线为 0，写入无效                       │
│  x1   │  ra      │ 返回地址（Return Address）                 │
│  x2   │  sp      │ 栈指针（Stack Pointer）                   │
│  x3   │  gp      │ 全局指针（Global Pointer）                 │
│  x4   │  tp      │ 线程指针（Thread Pointer）                 │
│  x5-7 │  t0-t2   │ 临时寄存器（caller-saved）                 │
│  x8   │  s0/fp   │ 保存寄存器 / 帧指针（callee-saved）         │
│  x9   │  s1      │ 保存寄存器（callee-saved）                 │
│ x10-11│  a0-a1   │ 函数参数 / 返回值                         │
│ x12-17│  a2-a7   │ 函数参数                                  │
│ x18-27│  s2-s11  │ 保存寄存器（callee-saved）                 │
│ x28-31│  t3-t6   │ 临时寄存器（caller-saved）                 │
├───────┼──────────┼─────────────────────────────────────────┤
│  PC   │   —      │ 程序计数器（不在通用寄存器组中）              │
└───────┴──────────┴─────────────────────────────────────────┘
```

**caller-saved vs callee-saved**：

- **caller-saved**（t0-t6, a0-a7）：被调用函数可以自由使用，调用者负责保存
- **callee-saved**（s0-s11）：被调用函数必须保存恢复，调用者无需关心

这在上下文切换中非常重要——切换时只需保存 callee-saved 寄存器（加上 ra, sp 等），因为 caller-saved 寄存器已经被调用者保存在栈上了。

### 25.2.2 特权模式

RISC-V 定义了三种（或更多）特权模式，xv6 使用其中三种：

```
┌─────────────────────────────────────────────────────────────┐
│                    RISC-V 特权模式                           │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │  Machine Mode (M-mode)              │  最高特权          │
│  │  - 固件/引导加载程序运行于此           │                    │
│  │  - 可以访问所有硬件                   │                    │
│  │  - xv6 中：start.c 的 early init     │                    │
│  └──────────────┬──────────────────────┘                    │
│                 │ mret 切换                                  │
│                 ▼                                            │
│  ┌─────────────────────────────────────┐                    │
│  │  Supervisor Mode (S-mode)           │  内核态             │
│  │  - 操作系统内核运行于此              │                    │
│  │  - 可以管理页表、处理中断             │                    │
│  │  - xv6 中：kernel/ 的所有代码         │                    │
│  └──────────────┬──────────────────────┘                    │
│                 │ sret 切换                                  │
│                 ▼                                            │
│  ┌─────────────────────────────────────┐                    │
│  │  User Mode (U-mode)                 │  用户态             │
│  │  - 应用程序运行于此                  │                    │
│  │  - 受限的指令集和内存访问             │                    │
│  │  - xv6 中：user/ 的所有程序           │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

**特权模式切换的触发条件**：

| 从 → 到 | 触发方式 | 说明 |
|---------|---------|------|
| U → S | 系统调用（ecall）或中断/异常 | 用户程序请求内核服务 |
| S → U | sret 指令 | 内核返回用户程序 |
| M → S | mret 指令 | 固件将控制权交给内核 |
| U → M | 中断（如果委托给 M-mode） | 极少在 xv6 中使用 |

### 25.2.3 CSR 寄存器

CSR（Control and Status Register，控制与状态寄存器）是 RISC-V 中用于配置和监控处理器状态的特殊寄存器。xv6 内核大量使用以下 CSR：

```
┌──────────────────────────────────────────────────────────────┐
│              Supervisor Mode CSR（关键寄存器）                 │
├───────────┬──────────────────────────────────────────────────┤
│ CSR 名称   │ 用途                                             │
├───────────┼──────────────────────────────────────────────────┤
│  stvec     │ 陷阱向量基地址                                    │
│            │ 存放陷阱处理程序的入口地址                         │
│            │ 当 trap 发生时，PC 跳转到 stvec 指向的地址         │
│            │                                                │
│  sepc      │ 异常程序计数器                                    │
│            │ trap 发生时，硬件自动保存用户程序的 PC 到 sepc      │
│            │ 内核可通过 sepc 知道用户程序执行到哪里               │
│            │                                                │
│  scause    │ 陷阱原因                                         │
│            │ 记录 trap 的原因（中断 or 异常 + 具体类型）          │
│            │ 最高位=1 表示中断，=0 表示异常                      │
│            │ 低 bit 表示具体编号                                │
│            │                                                │
│  sscratch  │ 临时寄存器                                       │
│            │ xv6 在初始化时将内核栈地址写入 sscratch              │
│            │ trap 入口汇编读取 sscratch 获取内核栈指针            │
│            │                                                │
│  stval     │ 陷阱值                                           │
│            │ 存放附加信息（如 page fault 的地址）                 │
│            │                                                │
│  satp      │ 地址翻译与保护                                    │
│            │ 控制页表：存放根页表的物理页号                       │
│            │ satp.MODE = 8 表示 Sv39 模式                      │
│            │ satp.PPN = 根页表的物理页号                        │
│            │                                                │
│  sstatus   │ 状态寄存器                                       │
│            │ 包含全局中断使能位 SIE 等                          │
│            │ SIE=1 允许中断，SIE=0 禁止中断                    │
└───────────┴──────────────────────────────────────────────────┘
```

**CSR 的读写指令**：

```c
// RISC-V 提供专门的 CSR 读写指令：
csrr  rd, csr      // 读取 CSR 到 rd:  rd = csr
csrw  csr, rs      // 写入 rs 到 CSR:  csr = rs
csrs  csr, rs      // 设置 CSR 的位:   csr |= rs
csrc  csr, rs      // 清除 CSR 的位:   csr &= ~rs

// xv6 中的宏定义（kernel/riscv.h）：
#define r_stvec()       ({ unsigned long x; asm volatile("csrr %0, stvec" : "=r" (x)); x; })
#define w_stvec(x)      asm volatile("csrw stvec, %0" : : "r" (x))
#define r_sepc()        ({ unsigned long x; asm volatile("csrr %0, sepc" : "=r" (x)); x; })
#define w_sepc(x)       asm volatile("csrw sepc, %0" : : "r" (x))
#define r_scause()      ({ unsigned long x; asm volatile("csrr %0, scause" : "=r" (x)); x; })
#define r_sscratch()    ({ unsigned long x; asm volatile("csrr %0, sscratch" : "=r" (x)); x; })
#define w_sscratch(x)   asm volatile("csrw sscratch, %0" : : "r" (x))
#define r_satp()        ({ unsigned long x; asm volatile("csrr %0, satp" : "=r" (x)); x; })
#define w_satp(x)       asm volatile("csrw satp, %0" : : "r" (x))
```

### 25.2.4 Sv39 页表

xv6 使用 RISC-V 的 **Sv39** 虚拟内存方案。Sv39 意味着虚拟地址有 39 位有效位（使用 9+9+9 三级页表）：

```
Sv39 虚拟地址格式（64 位，但只有低 39 位有效）：
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ 保留(25) │ VPN[2]  │ VPN[1]  │ VPN[0]  │ Offset  │
│  63-39   │  38-30  │  29-21  │  20-12  │  11-0   │
│  9 bits  │ 9 bits  │ 9 bits  │ 9 bits  │ 12 bits │
└─────────┴─────────┴─────────┴─────────┴─────────┘

Sv39 物理地址（56 位）：
┌─────────────────────┬──────────────────────────┐
│   PPN（44 位）        │      Offset（12 位）      │
│   55-12              │      11-0                │
└─────────────────────┴──────────────────────────┘
```

**三级页表结构**：

```
satp 寄存器
│
│  satp.PPN → 根页表（Level 2）的物理地址
│
▼
┌──────────────────────────┐
│   Level 2 页表（512 项）   │  ← 用 VPN[2] 索引
│   PTE[VPN[2]]            │
└──────────┬───────────────┘
           │ PTE.PPN → Level 1 页表的物理地址
           ▼
┌──────────────────────────┐
│   Level 1 页表（512 项）   │  ← 用 VPN[1] 索引
│   PTE[VPN[1]]            │
└──────────┬───────────────┘
           │ PTE.PPN → Level 0 页表的物理地址
           ▼
┌──────────────────────────┐
│   Level 0 页表（512 项）   │  ← 用 VPN[0] 索引
│   PTE[VPN[0]]            │
└──────────┬───────────────┘
           │ PTE.PPN → 物理页帧
           ▼
┌──────────────────────────┐
│   物理页帧（4KB）          │  ← Offset 索引具体字节
│   Physical Memory        │
└──────────────────────────┘
```

**页表项（PTE）格式**：

```
  63      54 53        28 27        19 18        10 9   8 7 6 5 4 3 2 1 0
┌─────────┬──────────────┬──────────────┬──────────────┬─────┬─────┬────┬───┬───┬───┬───┬───┐
│ Reserved │   PPN[2]    │    PPN[1]    │    PPN[0]    │ RSV │ RSV │ D  │ A │ G │ U │ X │ W │ R │ V │
│ (10bit)  │  (26 bit)   │   (9 bit)   │   (9 bit)    │     │     │    │   │   │   │   │   │   │   │
└─────────┴──────────────┴──────────────┴──────────────┴─────┴─────┴────┴───┴───┴───┴───┴───┘

V：Valid（有效位）
R：Read（可读）
W：Write（可写）
X：Execute（可执行）
U：User（用户态可访问）
G：Global（全局映射，不刷新 TLB）
A：Accessed（已访问，软件可用来实现替换策略）
D：Dirty（已写入）
```

**权限组合**：

| R | W | X | 含义 |
|---|---|---|------|
| 0 | 0 | 0 | 指向下一级页表（非叶子节点） |
| 1 | 0 | 0 | 只读页面 |
| 1 | 1 | 0 | 可读写页面（数据段） |
| 1 | 0 | 1 | 可读可执行页面（代码段） |
| 0 | 0 | 1 | 仅执行页面（罕见） |

### 25.2.5 RISC-V 与 x86 的对比

| 特性 | RISC-V | x86-64 |
|------|--------|--------|
| 指令集风格 | RISC（精简） | CISC（复杂） |
| 寄存器数量 | 32 个通用 | 16 个通用 |
| 特权级 | M/S/U | Ring 0-3（实际用 0 和 3） |
| 页表 | Sv39 三级页表 | 四级页表 |
| 中断控制器 | PLIC + CLINT | APIC |
| 系统调用 | ecall | syscall |
| 特权切换 | sret/mret | sysret/iret |
| 向量扩展 | V 扩展（可选） | AVX/SSE |

RISC-V 的设计哲学是"简单胜于复杂"，这使得 xv6 的代码比同等功能的 x86 版本更加清晰。

---

## 25.3 启动流程

xv6 的启动流程是一个从硬件到软件、从 M-mode 到 S-mode、从汇编到 C 的层层递进过程。

### 25.3.1 启动全景图

```
QEMU 模拟器启动
│
├─ 加载 kernel ELF 到内存 0x80000000
├─ 设置初始 PC = 0x80000000（entry.S）
│
▼
┌──────────────────────────────────────────────────────────┐
│ Phase 1: M-mode 早期初始化 (entry.S)                     │
│ - 多核启动：每个 hart 检查 hartid                          │
│ - hart 0 跳转到 start()                                  │
│ - 其他 hart 自旋等待                                      │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 2: M-mode 硬件配置 (start.c)                       │
│ - 设置 M-mode 异常处理                                    │
│ - 配置时钟中断 (timervec)                                 │
│ - 切换到 S-mode (mret)                                   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 3: S-mode 内核初始化 (main.c)                      │
│ - consoleinit, printfinit                                │
│ - kinit, kvminit, kvminithart                            │
│ - procinit, trapinit, trapinithart                       │
│ - plicinit, plicinithart                                 │
│ - binit, iinit, fileinit                                 │
│ - virtio_disk_init                                       │
│ - userinit                                               │
│ - scheduler()（永不返回）                                 │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 4: 第一个用户进程                                    │
│ - userinit 创建 init 进程                                 │
│ - initcode.S → exec("/init")                             │
│ - init.c → 启动 shell                                    │
└──────────────────────────────────────────────────────────┘
```

### 25.3.2 QEMU 如何加载 xv6 内核

当你执行 `make qemu` 时，Makefile 调用 QEMU 的命令大致如下：

```bash
qemu-system-riscv64 \
  -machine virt \
  -bios none \
  -kernel kernel/kernel \
  -m 128M \
  -smp 3 \
  -nographic \
  -drive file=fs.img,if=none,format=raw,id=x0 \
  -device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0
```

QEMU 做了以下事情：

1. **创建虚拟 RISC-V 硬件**：3 个 hart（CPU 核心）、128MB 内存、virtio 磁盘设备
2. **加载内核 ELF**：将 `kernel/kernel` 的代码段加载到物理地址 `0x80000000`
3. **设置初始 PC**：所有 hart 的 PC 都设为 `0x80000000`
4. **设置 M-mode**：CPU 从 Machine mode 开始运行

### 25.3.3 entry.S：汇编入口

`kernel/entry.S` 是 CPU 执行的第一段代码：

```asm
# kernel/entry.S
        .section .text
        .globl _entry
_entry:
        # 此时运行在 M-mode，物理地址 0x80000000
        # 所有 hart 都从这里开始执行

        # 设置栈指针：每个 hart 使用独立的栈
        # stack0 是在 start.c 中定义的栈空间数组
        # 每个栈大小为 4096 字节（2 * PGSIZE）
        la sp, stack0
        li a0, 1024*4
        csrr a1, mhartid         # 读取当前 hart ID
        addi a1, a1, 1           # hartid + 1
        mul a0, a0, a1           # 栈偏移 = 4096 * (hartid + 1)
        add sp, sp, a0           # sp = stack0 + 偏移

        # 调用 start()（在 start.c 中）
        call start

spin:
        # start() 不应返回，但如果返回了就无限循环
        j spin
```

**栈布局示意**：

```
高地址
┌──────────────────┐
│                  │
│   stack0[2]      │  ← hart 2 的栈顶 (4096 * 3)
│   4096 字节      │
├──────────────────┤
│   stack0[1]      │  ← hart 1 的栈顶 (4096 * 2)
│   4096 字节      │
├──────────────────┤
│   stack0[0]      │  ← hart 0 的栈顶 (4096 * 1)
│   4096 字节      │
├──────────────────┤
│   stack0 起始     │
└──────────────────┘
低地址
```

**注意**：此时还没有启用页表（虚拟内存），所有地址都是**物理地址**。

### 25.3.4 start.c：M-mode 配置

`kernel/start.c` 中的 `start()` 函数在 M-mode 下运行，负责配置硬件：

```c
// kernel/start.c
#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "defs.h"

void start() {
    // 1. 设置 M-mode 的状态寄存器
    unsigned long x = r_mstatus();
    x &= ~MSTATUS_MIE;         // 禁用 M-mode 全局中断
    w_mstatus(x);

    // 2. 设置 M-mode 异常处理程序
    w_mepc((uint64)main);       // 设置 mepc 为 main 函数地址
                                 // mret 后将跳转到 main()
    w_mtvec((uint64)timervec);  // M-mode 异常向量 → timervec

    // 3. 禁用分页（此时使用物理寻址）
    w_satp(0);

    // 4. 委托中断和异常给 S-mode
    w_medeleg(0xffff);          // 委托所有异常给 S-mode
    w_mideleg(0xffff);          // 委托所有中断给 S-mode
    w_sie(r_sie() | SIE_SEIE | SIE_STIE | SIE_SSIE);

    // 5. 配置物理内存保护（允许 S-mode 访问所有物理内存）
    w_pmpaddr0(0x3fffffffffffffull);
    w_pmpcfg0(0xf);

    // 6. 配置时钟中断
    timerinit();

    // 7. 切换到 S-mode
    int id = r_mhartid();
    w_tp(id);                   // 设置 tp 寄存器为 hart ID
    // 以下指令将：
    // - 设置 sstatus（S-mode 状态）
    // - 将 mepc 中的地址（main）加载到 PC
    // - 从 M-mode 切换到 S-mode
    asm volatile("mret");
}
```

**关键点**：

1. `w_mepc((uint64)main)` + `mret`：这是一个巧妙的技巧。`mret` 指令会将 `mepc` 的值加载到 PC，同时从 M-mode 切换到 S-mode。所以这行代码的效果是"跳转到 main()，同时切换特权级"。

2. `w_medeleg(0xffff)`：将异常委托给 S-mode。这样当用户态发生 page fault、illegal instruction 等异常时，直接交给 S-mode 的内核处理，不需要经过 M-mode。

3. `w_pmpaddr0` + `w_pmpcfg0`：PMP（Physical Memory Protection）配置，允许 S-mode 访问所有物理内存。如果不配置 PMP，S-mode 默认不能访问物理内存。

### 25.3.5 main()：内核初始化全景

`main()` 是在 S-mode 下运行的 C 语言入口函数，它按特定顺序调用所有初始化函数：

```c
// kernel/main.c
#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "defs.h"

volatile static int started = 0;

// hart 0 的 main()：负责所有初始化
// 其他 hart 的 main()：等待 hart 0 完成后启动
void main() {
    if(cpuid() == 0) {
        // === hart 0 的初始化流程 ===

        consoleinit();    // 1. 控制台初始化
        printfinit();     // 2. printf 锁初始化
        printf("\n");
        printf("xv6 kernel is booting\n");
        printf("\n");

        kinit();          // 3. 物理页分配器初始化
        kvminit();        // 4. 创建内核页表
        kvminithart();    // 5. 启用分页（加载内核页表到 satp）

        procinit();       // 6. 进程表初始化
        trapinit();       // 7. 中断向量表初始化
        trapinithart();   // 8. 设置 stvec（S-mode trap 入口）

        plicinit();       // 9. PLIC 中断控制器初始化
        plicinithart();   // 10. PLIC 当前 hart 初始化

        binit();          // 11. 块 I/O 缓冲区缓存初始化
        iinit();          // 12. inode 缓存初始化
        fileinit();       // 13. 文件表初始化

        virtio_disk_init(); // 14. virtio 磁盘驱动初始化

        userinit();       // 15. 创建第一个用户进程！

        __sync_synchronize();
        started = 1;      // 通知其他 hart 可以启动了
    } else {
        // === 其他 hart 等待 hart 0 完成 ===
        while(started == 0)
            ;
        __sync_synchronize();
        printf("hart %d starting\n", cpuid());
        kvminithart();    // 启用自己的分页
        trapinithart();   // 设置自己的 stvec
        plicinithart();   // 设置自己的 PLIC
    }

    // 所有 hart 最终都进入调度器
    scheduler();          // 永不返回！
}
```

**初始化顺序非常重要**！后面的函数依赖前面的函数设置的基础设施：

```
consoleinit()   ← 最先，因为后续所有函数都可能调用 printf
printfinit()    ← 初始化 printf 的锁
│
kinit()         ← 物理页分配器（必须在任何内存分配之前）
kvminit()       ← 创建内核页表（需要物理页）
kvminithart()   ← 启用分页（需要页表已就绪）
│
procinit()      ← 进程表（需要内存分配可用）
trapinit()      ← 异常处理表
trapinithart()  ← stvec（需要知道 trap 处理函数地址）
│
plicinit()      ← 中断控制器
plicinithart()  ← PLIC 当前核心配置
│
binit()         ← 缓冲区缓存（需要内存和锁）
iinit()         ← inode 缓存（需要缓冲区缓存）
fileinit()      ← 文件表
│
virtio_disk_init() ← 磁盘驱动（需要中断和缓冲区）
│
userinit()      ← 创建第一个进程（需要以上所有基础设施）
```

下面逐一详细讲解每个初始化函数。

---

### 25.3.6 consoleinit()：控制台初始化

```c
// kernel/console.c
void consoleinit(void) {
    initlock(&cons.lock, "console");

    // 将控制台设备连接到 read/write 函数
    devsw[CONSOLE].read = consoleread;
    devsw[CONSOLE].write = consolewrite;

    // 初始化 UART 硬件
    uartinit();

    // 将 console 设备连接到 stdin 和 stdout
    // 这样用户程序的 printf 和 read/write 才能工作
}
```

`consoleinit` 的作用：
1. 初始化控制台的锁
2. 设置设备驱动的读写函数指针（设备表 `devsw`）
3. 初始化 UART 硬件（串口），这是 xv6 的终端 I/O 通道

### 25.3.7 printfinit()：打印系统初始化

```c
// kernel/printf.c
void printfinit(void) {
    initlock(&pr.lock, "pr");
    pr.locking = 1;
}
```

很简单——初始化 `printf` 函数使用的自旋锁。内核的 `printf` 需要加锁，因为多个 CPU 可能同时调用 `printf`，如果不加锁，输出会交错混乱。

### 25.3.8 kinit()：物理页分配器初始化

```c
// kernel/kalloc.c
void kinit() {
    initlock(&kmem.lock, "kmem");
    freerange(end, (void*)PHYSTOP);
}

void freerange(void *pa_start, void *pa_end) {
    char *p;
    p = (char*)PGROUNDUP((uint64)pa_start);
    for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
        kfree(p);
}
```

`kinit` 初始化物理页分配器：
1. 初始化分配器的锁
2. 调用 `freerange`，将从 `end`（内核 ELF 结束地址）到 `PHYSTOP`（物理内存顶部）的所有物理页加入空闲链表

```
物理内存布局：
0x80000000 ┌──────────────────┐
           │  内核代码和数据    │  ← kernel ELF 加载到这里
           │  ...             │
end →      ├──────────────────┤
           │  空闲物理页       │  ← kinit 将这些页加入空闲链表
           │  (kfree 每页)     │
           │  ...             │
PHYSTOP →  ├──────────────────┤
           │  ...             │
           └──────────────────┘
```

每个空闲物理页被组织成一个**链表**：每页的起始处存储指向下一个空闲页的指针。

### 25.3.9 kvminit()：创建内核页表

```c
// kernel/vm.c
pagetable_t kvminit(void) {
    pagetable_t kpagetable = (pagetable_t) kalloc();

    // 清零页表
    memset(kpagetable, 0, PGSIZE);

    // 映射 UART 寄存器
    kvmmap(kpagetable, UART0, UART0, PGSIZE, PTE_R | PTE_W);

    // 映射 virtio 磁盘接口
    kvmmap(kpagetable, VIRTIO0, VIRTIO0, PGSIZE, PTE_R | PTE_W);

    // 映射 PLIC 中断控制器
    kvmmap(kpagetable, PLIC, PLIC, 0x400000, PTE_R | PTE_W);

    // 映射内核代码段（可读可执行）
    kvmmap(kpagetable, KERNBASE, KERNBASE,
           (uint64)etext - KERNBASE, PTE_R | PTE_X);

    // 映射内核数据段及物理内存剩余部分（可读可写）
    kvmmap(kpagetable, (uint64)etext, (uint64)etext,
           PHYSTOP - (uint64)etext, PTE_R | PTE_W);

    // 映射 trampoline（trap 进出内核的跳板页）
    kvmmap(kpagetable, TRAMPOLINE, (uint64)trampoline,
           PGSIZE, PTE_R | PTE_X);

    return kpagetable;
}
```

**内核页表的映射**：

```
虚拟地址                    物理地址                  权限
─────────────────────────────────────────────────────────
0x0000_0000_0000_0000      （未映射，访问会 fault）

0x0000_0000_0200_0000      → 0x0200_0000 (PLIC)       R/W
  (PLIC, 4MB)

0x0000_0000_1000_0000      → 0x1000_0000 (VIRTIO0)    R/W
  (Virtio 磁盘, 4KB)

0x0000_0000_1000_0000      → 0x1000_0000 (UART0)      R/W
  (UART, 4KB)

0x0000_0000_8000_0000      → 0x8000_0000 (KERNBASE)   R/X
  (内核代码段, 到 etext)      （只读可执行）

0x0000_0000_8000_xxxx      → 0x8000_xxxx (etext)      R/W
  (内核数据+剩余内存)         （可读写）

0xFFFF_FFFF_C000_0000      → trampoline 物理页         R/X
  (TRAMPOLINE, 4KB)          （trap 跳板）
```

**关键设计**：内核页表使用**恒等映射**（identity mapping）——虚拟地址 = 物理地址。这简化了早期启用分页时的代码，因为不需要重新计算地址。

`TRAMPOLINE` 是一个特殊的映射——它映射到所有地址空间（内核和用户页表），位于虚拟地址空间的顶部。这是 trap 处理时进出内核的关键。

### 25.3.10 kvminithart()：启用分页

```c
// kernel/vm.c
void kvminithart() {
    // 将内核页表加载到 satp 寄存器
    w_satp(MAKE_SATP(kernel_pagetable));
    // 刷新 TLB（快表）
    sfence_vma();
}
```

`MAKE_SATP` 宏将页表物理地址编码为 satp 值：

```c
#define MAKE_SATP(pagetable) \
    (SATP_SV39 | (((uint64)pagetable) >> 12))

// SATP_SV39 = 8L << 60  (satp.MODE = Sv39)
// satp = [MODE=8 | ASID=0 | PPN=页表物理页号]
```

**执行此函数后，CPU 启用 Sv39 虚拟地址翻译**。由于使用恒等映射，所有内核代码的地址访问不受影响——虚拟地址 0x80000000 翻译后还是物理地址 0x80000000。

### 25.3.11 procinit()：进程表初始化

```c
// kernel/proc.c
void procinit(void) {
    struct proc *p;

    initlock(&pid_lock, "nextpid");
    nextpid = 1;

    // 初始化每个进程的锁和状态
    for(p = proc; p < &proc[NPROC]; p++) {
        initlock(&p->lock, "proc");
        p->state = UNUSED;
        p->kstack = KSTACK((int)(p - proc));
    }
}
```

`procinit` 初始化进程表 `proc[]` 数组（最多 `NPROC=64` 个进程）。每个进程结构体（`struct proc`）被设置为 `UNUSED` 状态。

注意 `p->kstack = KSTACK(...)`——为每个进程预分配了内核栈地址。`KSTACK` 宏将进程编号映射到一个固定的虚拟地址，这些内核栈在后续的页表中被映射。

### 25.3.12 trapinit() 与 trapinithart()：陷阱处理初始化

```c
// kernel/trap.c
void trapinit(void) {
    initlock(&tickslock, "time");
    // 初始化系统调用函数指针表
    // syscall.c 中的 syscall() 函数会查这个表
}

void trapinithart(void) {
    // 设置 S-mode 的陷阱向量
    // stvec 存放 kernelvec 的地址
    // 当 S-mode 发生中断/异常时，CPU 跳转到 kernelvec
    w_stvec((uint64)kernelvec);
}
```

`kernelvec` 是汇编编写的陷阱入口（`kernel/kernelvec.S`），它保存所有寄存器，然后调用 C 函数 `kerneltrap()` 或 `usertrap()`。

**为什么分开两个函数？**

- `trapinit()`：只执行一次（设置系统调用表等全局数据结构）
- `trapinithart()`：每个 hart 都要执行（每个 CPU 核心都需要设置自己的 `stvec`）

### 25.3.13 plicinit() 与 plicinithart()：中断控制器初始化

```c
// kernel/plic.c
void plicinit(void) {
    // 设置 virtio 磁盘中断的优先级
    *(uint32*)(PLIC + IRQ_VIRTIO*4) = 1;

    // 设置 UART 中断的优先级
    *(uint32*)(PLIC + IRQ_UART*4) = 1;
}

void plicinithart(void) {
    int hart = cpuid();

    // 设置当前 hart 的 S-mode 中断阈值为 0（接受所有中断）
    *(uint32*)PLIC_SENABLE(hart) = (1 << IRQ_VIRTIO) | (1 << IRQ_UART);
    *(uint32*)PLIC_SPRIORITY(hart) = 0;

    // 在 S-mode 的 sstatus 中启用外部中断
    w_sie(r_sie() | SIE_SEIE);
}
```

**PLIC**（Platform-Level Interrupt Controller，平台级中断控制器）：
- 负责将外部设备中断（磁盘、UART）路由到某个 CPU hart
- 每个中断有优先级，优先级越高越先被处理
- `PLIC_SENABLE`：设置哪些中断源被使能
- `PLIC_SPRIORITY`：中断阈值，只有优先级高于此值的中断才会被传递

### 25.3.14 binit()：缓冲区缓存初始化

```c
// kernel/bio.c
void binit(void) {
    struct buf *b;

    initlock(&bcache.lock, "bcache");

    // 创建 LRU 双向链表
    // bcach.head 是哨兵节点
    bcache.head.prev = &bcache.head;
    bcache.head.next = &bcache.head;

    // 将所有缓冲区加入链表
    for(b = bcache.buf; b < bcache.buf + NBUF; b++) {
        b->next = bcache.head.next;
        b->prev = &bcache.head;
        bcache.head.next->prev = b;
        bcache.head.next = b;
    }
}
```

缓冲区缓存是文件系统的基础——所有磁盘读写都通过它。`binit` 初始化一个 **LRU（最近最少使用）双向链表**，包含 `NBUF` 个缓冲区。

```
LRU 链表结构：

head ←→ buf[0] ←→ buf[1] ←→ ... ←→ buf[NBUF-1] ←→ head
 │                                                            │
 └────────────────────────────────────────────────────────────┘
                    循环双向链表

- 最近使用的缓冲区移到 head 后面
- 需要回收时，从 head 前面取（最久未使用的）
```

### 25.3.15 iinit()：inode 缓存初始化

```c
// kernel/fs.c
void iinit(void) {
    initlock(&itable.lock, "itable");

    // 初始化所有 inode 为 UNUSED
    for(int i = 0; i < NINODE; i++) {
        inodesleeplock(&itable.inode[i].lock, "inode");
    }
}
```

`iinit` 初始化 inode 缓存表（`itable`）。inode 缓存保存最近使用的 inode 在内存中的副本，避免每次访问都读取磁盘。

### 25.3.16 fileinit()：文件表初始化

```c
// kernel/file.c
void fileinit(void) {
    initlock(&ftable.lock, "file");
}
```

非常简单——只需初始化文件表的锁。文件表 `ftable` 是一个全局的 `struct file` 数组，所有进程共享。每个打开的文件对应一个 `struct file` 条目。

### 25.3.17 virtio_disk_init()：磁盘驱动初始化

```c
// kernel/virtio_disk.c
void virtio_disk_init(void) {
    uint32 status = 0;

    // 1. 复位设备
    *R(VIRTIO_MMIO_STATUS) = status;

    // 2. 设置 ACKNOWLEDGE 状态位
    status |= VIRTIO_CONFIG_S_ACKNOWLEDGE;
    *R(VIRTIO_MMIO_STATUS) = status;

    // 3. 设置 DRIVER 状态位
    status |= VIRTIO_CONFIG_S_DRIVER;
    *R(VIRTIO_MMIO_STATUS) = status;

    // 4. 协商特性（features）
    uint64 features = *R(VIRTIO_MMIO_DEVICE_FEATURES);
    features &= ~(1 << VIRTIO_BLK_F_RO);      // 不是只读
    features &= ~(1 << VIRTIO_BLK_F_SCSI);     // 不使用 SCSI
    features &= ~(1 << VIRTIO_BLK_F_CONFIG_WCE);
    features &= ~(1 << VIRTIO_BLK_F_MQ);       // 不使用多队列
    features &= ~(1 << VIRTIO_F_ANY_LAYOUT);
    features &= ~(1 << VIRTIO_RING_F_EVENT_IDX);
    features &= ~(1 << VIRTIO_RING_F_INDIRECT_DESC);
    *R(VIRTIO_MMIO_DRIVER_FEATURES) = features;

    // 5. 设置 FEATURES_OK
    status |= VIRTIO_CONFIG_S_FEATURES_OK;
    *R(VIRTIO_MMIO_STATUS) = status;

    // 6. 设置 QUEUE_READY 等
    // ...（初始化 virtqueue）

    // 7. 最后设置 DRIVER_OK
    status |= VIRTIO_CONFIG_S_DRIVER_OK;
    *R(VIRTIO_MMIO_STATUS) = status;
}
```

virtio 是 RISC-V 平台常用的半虚拟化 I/O 设备标准。`virtio_disk_init` 按照 virtio 规范初始化磁盘设备：复位 → 确认 → 协商特性 → 配置队列 → 就绪。

### 25.3.18 userinit()：创建第一个进程

```c
// kernel/proc.c
void userinit(void) {
    struct proc *p;

    p = allocproc();          // 分配一个新进程

    initproc = p;

    // 设置内核页表
    p->pagetable = proc_pagetable(p);

    // 设置用户态代码（initcode）
    // 这是一个硬编码的二进制程序，直接写入进程的用户内存
    p->sz = PGSIZE;

    // 从 initcode[] 数组（编译时嵌入）复制用户态代码
    uvminit(p->pagetable, initcode, sizeof(initcode));

    // 设置用户态寄存器
    p->trapframe->epc = 0;        // 用户程序入口：虚拟地址 0
    p->trapframe->sp = PGSIZE;    // 用户栈指针：一页大小

    safestrcpy(p->name, "initcode", sizeof(p->name));
    p->cwd = namei("/");         // 当前工作目录设为根目录

    p->state = RUNNABLE;         // 标记为可运行

    release(&p->lock);
}
```

**`userinit` 是整个启动流程的高潮**——它创建了第一个用户进程（PID 1）。这个进程将启动 shell，最终用户看到 `xv6$` 提示符。

---

## 25.4 第一个进程

### 25.4.1 initcode：最小的用户程序

`user/initcode.S` 是第一个用户程序的汇编代码，它在编译时被嵌入到内核中：

```asm
# user/initcode.S
# initcode 会通过 exec 调用加载 /init 程序
# 这是第一个用户程序的"种子代码"

.globl start
start:
        # exec("/init", argv)
        la a0, init      # a0 = "/init" 字符串的地址
        la a1, argv      # a1 = argv 数组的地址
        li a7, SYS_exec  # 系统调用号：exec
        ecall            # 触发系统调用，切换到内核态

# 如果 exec 失败，无限循环
for:
        j for

init:
        .string "/init\0"

.p2align 2
argv:
        .long init       # argv[0] = "/init"
        .long 0          # argv[1] = NULL
```

这个程序极其简单——只做一件事：调用 `exec("/init", argv)`。

### 25.4.2 从 initcode 到 shell 的完整路径

```
步骤 1: userinit() 创建进程
│  - 将 initcode 的机器码复制到用户地址空间 0x0
│  - 设置 trapframe->epc = 0（用户态 PC）
│  - 设置 trapframe->sp = PGSIZE（用户栈）
│  - p->state = RUNNABLE
│
▼
步骤 2: scheduler() 选中该进程
│  - 该进程是唯一的 RUNNABLE 进程
│  - 调用 swtch() 切换到该进程
│
▼
步骤 3: usertrapret() → 返回用户态
│  - 设置 stvec = uservec（用户 trap 入口）
│  - 设置 sstatus（允许用户态中断）
│  - 从 trapframe 恢复寄存器
│  - sret → 跳转到 epc = 0（initcode 开始执行）
│
▼
步骤 4: initcode.S 执行
│  - ecall（系统调用：exec）
│  - CPU 从 U-mode 切换到 S-mode
│  - 进入 usertrap() → syscall() → sys_exec()
│
▼
步骤 5: sys_exec("/init")
│  - 加载 /init ELF 文件（从磁盘）
│  - 设置新的页表（用户进程页表）
│  - 设置新的 trapframe->epc = /init 的入口地址
│  - 返回用户态，执行 /init
│
▼
步骤 6: init.c 执行
│  - 打开控制台设备（fd 0, 1, 2）
│  - fork() 创建子进程
│  - 子进程 exec("sh") → 启动 shell
│  - 父进程 wait() 等待子进程
│
▼
步骤 7: 用户看到 shell 提示符
   xv6$ _
```

### 25.4.3 init.c：用户态 init 进程

```c
// user/init.c
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/fcntl.h"

char *argv[] = { "sh", 0 };

int main(void) {
    int pid, wpid;

    // 打开控制台设备作为 stdin（fd 0）
    if(open("console", O_RDWR) < 0){
        mknod("console", CONSOLE, 0);
        open("console", O_RDWR);
    }

    // 将 stdin 复制到 stdout（fd 1）和 stderr（fd 2）
    dup(0);  // fd 1 = stdout
    dup(0);  // fd 2 = stderr

    // 循环：不断创建 shell 子进程
    for(;;){
        printf("init: starting sh\n");
        pid = fork();
        if(pid < 0){
            printf("init: fork failed\n");
            exit(1);
        }

        if(pid == 0){
            // 子进程：执行 shell
            exec("sh", argv);
            printf("init: exec sh failed\n");
            exit(1);
        }

        // 父进程（init）：等待 shell 退出
        // 如果 shell 退出了，重新启动一个
        for(;;){
            wpid = wait((int *) 0);
            if(wpid == pid){
                // shell 退出了，重新启动
                break;
            }
        }
    }
}
```

**init.c 的三个核心职责**：

1. **确保文件描述符 0、1、2 存在**：这是 Unix 的约定——stdin (0)、stdout (1)、stderr (2)。init 打开控制台设备，然后 `dup` 两次。

2. **启动 shell**：`fork()` + `exec("sh")` 创建 shell 子进程。

3. **无限循环 + wait**：如果 shell 退出（用户输入 `exit`），init 会重新启动一个 shell。这就是为什么在 xv6 中输入 `exit` 后 shell 会关闭，但 init 会立即启动一个新的。

### 25.4.4 exec 系统调用的简要流程

当 initcode 调用 `exec("/init")` 时，内核执行以下操作：

```
sys_exec()
│
├─ 1. 从用户空间读取路径名参数
│     argstr(0, path, MAXPATH)
│
├─ 2. 查找 inode
│     namei(path) → 返回 /init 的 inode
│
├─ 3. 锁定 inode，读取 ELF 头部
│     ilock(ip)
│     readi(ip, (uint64)&elf, 0, sizeof(elf))
│
├─ 4. 验证 ELF 魔数
│     if(elf.magic != ELF_MAGIC) → 错误
│
├─ 5. 分配新的页表
│     pagetable = proc_pagetable(p)
│
├─ 6. 加载每个 ELF 段到内存
│     for each phdr in program headers:
│       uvmalloc(pagetable, sz, phdr.vaddr + phdr.memsz)
│       loadseg(pagetable, phdr.vaddr, ip, phdr.off, phdr.filesz)
│
├─ 7. 设置用户栈
│     sz = uvmalloc(pagetable, sz, sz + 2*PGSIZE)
│     uvmclear(pagetable, sz - 2*PGSIZE)  // guard page
│
├─ 8. 设置参数（argv）
│     sp = sz;  // 栈顶
│     // 将 argv 和字符串复制到用户栈
│
├─ 9. 替换旧页表
│     oldpagetable = p->pagetable
│     p->pagetable = pagetable
│     p->sz = sz
│     p->trapframe->epc = elf.entry  // 程序入口
│     p->trapframe->sp = sp          // 新栈指针
│
└─ 10. 释放旧页表
      proc_freepagetable(oldpagetable, oldsz)
```

exec 是 Unix 中"进程重生"的核心——它用新程序替换当前进程的地址空间，但进程的 PID、文件描述符等保持不变。

---

## 25.5 内存布局全景

### 25.5.1 物理内存布局

```
物理地址
0x0000_0000  ┌──────────────────────┐
             │                      │
             │  设备映射区域          │
             │  (UART, PLIC,        │
             │   VirtIO 等)         │
             │                      │
0x8000_0000  ├──────────────────────┤ ← KERNBASE
             │                      │
             │  内核 ELF             │
             │  .text (代码段)       │  ← 只读可执行 (R/X)
             │  .rodata (只读数据)   │
             │  .data (数据段)       │  ← 可读写 (R/W)
             │  .bss (未初始化数据)  │
             │                      │
etext →      ├──────────────────────┤
             │                      │
             │  内核分配的物理页      │  ← kalloc() 分配
             │  (进程页表、内核栈、   │
             │   缓冲区缓存等)       │
             │                      │
PHYSTOP →    ├──────────────────────┤
             │                      │
             │  (未使用的内存区域)    │
             │                      │
             └──────────────────────┘
```

### 25.5.2 虚拟地址空间布局

```
用户态虚拟地址空间（低地址）：
0x0000_0000_0000_0000  ┌──────────────────┐
                       │  用户代码和数据    │  ← ELF 加载到这里
                       │  (text, data,     │
                       │   heap)           │
                       │  ...              │
                       ├──────────────────┤ ← sz (堆顶)
                       │                  │
                       │  用户栈           │ ← 一页，向低地址增长
                       ├──────────────────┤
                       │  Guard Page       │ ← 无映射，栈溢出检测
                       │  (unmapped)       │
                       ├──────────────────┤
                       │                  │
                       │  (未映射区域)     │
                       │                  │
0x0000_003F_FFFF_F000  ├──────────────────┤
                       │  Trampoline      │ ← trap 进出内核的跳板页
0x0000_003F_FFFF_FFFF  └──────────────────┘

内核态虚拟地址空间（高地址）：
0xFFFF_0000_0000_0000  ┌──────────────────┐
                       │                  │
                       │  (未映射区域)     │
                       │                  │
0xFFFF_FFFF_8000_0000  ├──────────────────┤ ← KERNBASE (虚拟)
                       │  内核代码段       │  ← 映射到 0x8000_0000
                       │  (text)          │
etext →                ├──────────────────┤
                       │  内核数据段+内存  │  ← 可读写
                       │  ...             │
PHYSTOP →              ├──────────────────┤
                       │                  │
0xFFFF_FFFF_C000_0000  ├──────────────────┤
                       │  Trampoline      │ ← 同一个物理页
0xFFFF_FFFF_FFFF_FFFF  └──────────────────┘
```

### 25.5.3 trapframe 结构

当用户进程陷入内核时，内核需要保存用户态的寄存器状态。xv6 使用 `trapframe` 结构：

```c
// kernel/proc.h
struct trapframe {
    /*   0 */ uint64 kernel_satp;    // 内核页表
    /*   8 */ uint64 kernel_sp;      // 内核栈指针
    /*  16 */ uint64 kernel_trap;    // usertrap() 的地址
    /*  24 */ uint64 kernel_hartid;  // hart ID
    /*  32 */ uint64 ra;             // 以下为用户态寄存器
    /*  40 */ uint64 sp;
    /*  48 */ uint64 gp;
    /*  56 */ uint64 tp;
    /*  64 */ uint64 t0;
    /*  72 */ uint64 t1;
    /*  80 */ uint64 t2;
    /*  88 */ uint64 s0;
    /*  96 */ uint64 s1;
    /* 104 */ uint64 a0;
    /* 112 */ uint64 a1;
    /* 120 */ uint64 a2;
    /* 128 */ uint64 a3;
    /* 136 */ uint64 a4;
    /* 144 */ uint64 a5;
    /* 152 */ uint64 a6;
    /* 160 */ uint64 a7;
    /* 168 */ uint64 s2;
    /* 176 */ uint64 s3;
    /* 184 */ uint64 s4;
    /* 192 */ uint64 s5;
    /* 200 */ uint64 s6;
    /* 208 */ uint64 s7;
    /* 216 */ uint64 s8;
    /* 224 */ uint64 s9;
    /* 232 */ uint64 s10;
    /* 240 */ uint64 s11;
    /* 248 */ uint64 t3;
    /* 256 */ uint64 t4;
    /* 264 */ uint64 t5;
    /* 272 */ uint64 t6;
};
```

`trapframe` 的前四个字段（`kernel_satp`, `kernel_sp`, `kernel_trap`, `kernel_hartid`）是内核用来快速恢复到内核态的，不是用户寄存器。

### 25.5.4 struct proc：进程控制块

```c
// kernel/proc.h
enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

struct proc {
    struct spinlock lock;

    // p->lock 必须持有以下字段的状态
    enum procstate state;        // 进程状态
    void *chan;                   // 如果是 SLEEPING，等待的原因
    int killed;                  // 是否被杀死
    int xstate;                  // 退出状态（给 wait 使用）
    int pid;                     // 进程 ID

    // wait_lock 必须持有的字段
    struct proc *parent;         // 父进程

    // 以下字段由进程自己持有，不需要锁
    uint64 kstack;               // 内核栈虚拟地址
    uint64 sz;                   // 进程内存大小（字节）
    pagetable_t pagetable;       // 用户页表
    struct trapframe *trapframe; // 用户态寄存器保存区
    struct context context;      // 内核态上下文（swtch 使用）
    struct file *ofile[NOFILE];  // 打开的文件表
    struct inode *cwd;           // 当前工作目录
    char name[16];               // 进程名（调试用）
};
```

---

## 25.6 trap 处理概述

### 25.6.1 从用户态到内核态

当用户进程执行 `ecall`（系统调用）或发生中断时：

```
用户态 (U-mode)
│
│ ecall / 中断 / 异常
│
▼
硬件自动完成：
  1. sstatus ← 当前状态
  2. sepc ← PC
  3. scause ← trap 原因
  4. PC ← stvec (= uservec)
  5. 模式切换 U → S
│
▼
uservec (trampoline.S)
  1. 从 sscratch 获取 trapframe 地址
  2. 保存用户态寄存器到 trapframe
  3. 从 trapframe 恢复内核信息
     (kernel_satp, kernel_sp, kernel_trap, kernel_hartid)
  4. 加载内核页表
  5. 跳转到 usertrap()
│
▼
usertrap() (trap.c)
  1. 设置 stvec = kernelvec (处理内核态 trap)
  2. 根据 scause 分发：
     - 系统调用 → syscall()
     - 设备中断 → devintr()
     - 其他异常 → 处理错误
  3. 可能调度其他进程
  4. 调用 usertrapret() 返回用户态
│
▼
usertrapret() (trap.c)
  1. 设置 stvec = uservec
  2. 设置 trapframe 中的内核信息
  3. 设置 sstatus (U-mode, 开中断)
  4. 设置 sepc = trapframe->epc
  5. 跳转到 trampoline 中的 userret
│
▼
userret (trampoline.S)
  1. 加载用户页表
  2. 恢复用户态寄存器
  3. sret → 返回用户态
  4. PC = sepc，模式切换 S → U
```

### 25.6.2 从内核态到用户态（调度）

```
scheduler() (proc.c)
│
├─ 遍历进程表，找到 RUNNABLE 的进程 p
├─ p->state = RUNNING
├─ swtch(&c->context, &p->context)
│
▼
swtch (swtch.S)
  保存当前内核上下文（callee-saved 寄存器）
  恢复目标进程的内核上下文
  返回到目标进程之前调用 swtch 的位置
│
▼
返回到 sched() → usertrapret() → userret → 用户态
```

---

## 25.7 完整启动流程时序

```
时间线 ─────────────────────────────────────────────────────────→

QEMU 启动
│
│  1. 加载 kernel ELF 到 0x80000000
│  2. 所有 hart 的 PC 设为 0x80000000
│
├── entry.S ─────────────────────────────────────────────────────
│   │  3. 每个 hart 计算自己的栈指针
│   │  4. hart 0 调用 start()
│   │  5. 其他 hart 进入 spin 循环
│   │
├── start.c (M-mode) ───────────────────────────────────────────
│   │  6. 禁用 M-mode 中断
│   │  7. 设置 mepc = main
│   │  8. 设置 mtvec = timervec
│   │  9. 禁用分页 (satp = 0)
│   │  10. 委托中断/异常给 S-mode
│   │  11. 配置 PMP
│   │  12. 配置时钟中断
│   │  13. mret → 切换到 S-mode，跳转到 main()
│   │
├── main.c (S-mode) ────────────────────────────────────────────
│   │  14. consoleinit() ──── 控制台/UART 初始化
│   │  15. printfinit() ───── printf 锁初始化
│   │  16. printf("xv6 kernel is booting\n")
│   │  17. kinit() ────────── 物理页分配器初始化
│   │  18. kvminit() ──────── 创建内核页表
│   │  19. kvminithart() ──── 启用 Sv39 分页
│   │  20. procinit() ─────── 进程表初始化
│   │  21. trapinit() ─────── 系统调用表初始化
│   │  22. trapinithart() ─── 设置 stvec = kernelvec
│   │  23. plicinit() ─────── PLIC 中断控制器初始化
│   │  24. plicinithart() ─── PLIC 当前 hart 配置
│   │  25. binit() ────────── 缓冲区缓存初始化
│   │  26. iinit() ────────── inode 缓存初始化
│   │  27. fileinit() ─────── 文件表初始化
│   │  28. virtio_disk_init() ── 磁盘驱动初始化
│   │  29. userinit() ─────── 创建第一个进程（PID 1）
│   │  30. started = 1 ────── 其他 hart 开始初始化
│   │  31. scheduler() ────── 进入调度循环（永不返回）
│   │
├── 调度到 init 进程 ────────────────────────────────────────────
│   │  32. swtch() → 切换到 init 进程的上下文
│   │  33. usertrapret() → 准备返回用户态
│   │  34. sret → 跳转到 initcode (PC=0)
│   │
├── initcode.S (U-mode) ───────────────────────────────────────
│   │  35. ecall (exec 系统调用)
│   │
├── 内核处理 exec ──────────────────────────────────────────────
│   │  36. 加载 /init ELF 到内存
│   │  37. 设置新的页表和入口地址
│   │
├── init.c (U-mode) ───────────────────────────────────────────
│   │  38. 打开控制台 (fd 0, 1, 2)
│   │  39. fork() 创建子进程
│   │  40. 子进程 exec("sh")
│   │
├── sh.c (U-mode) ─────────────────────────────────────────────
│   │  41. Shell 启动，显示提示符
│   │
▼
  xv6$ _     ← 系统启动完成！
```

---

## 面试高频考点

### Q1：xv6 启动过程中，什么时候从 M-mode 切换到 S-mode？

**答**：在 `start.c` 的 `start()` 函数中，通过 `mret` 指令切换。具体做法是：将 `main` 函数的地址写入 `mepc` 寄存器，然后执行 `mret`。`mret` 指令会将 `mepc` 的值加载到 PC，同时将处理器模式从 M-mode 切换到 S-mode。

### Q2：为什么 xv6 要先禁用分页，再在 `kvminithart()` 中启用？

**答**：因为页表本身需要内存来存储，而在分页启用之前，只能使用物理地址。xv6 的策略是：
1. `entry.S` 和 `start.c` 使用物理地址（禁用分页）
2. `kvminit()` 使用物理地址创建页表（恒等映射）
3. `kvminithart()` 启用分页，但由于是恒等映射，所有地址访问不受影响

如果在页表创建之前就启用分页，CPU 会尝试做地址翻译，但页表还不存在，会导致 fault。

### Q3：`stvec` 寄存器的作用是什么？xv6 中有几个不同的陷阱向量？

**答**：`stvec` 存放陷阱处理程序的入口地址。当 S-mode 发生中断或异常时，PC 自动跳转到 `stvec` 指向的地址。xv6 中有三个不同的陷阱向量：

- `uservec`（trampoline.S）：用户态发生 trap 时进入
- `kernelvec`（kernelvec.S）：内核态发生 trap 时进入
- `timervec`（kernelvec.S）：M-mode 时钟中断处理

### Q4：xv6 的 `main()` 为什么将初始化分成两个函数（如 `trapinit` 和 `trapinithart`）？

**答**：因为 xv6 是多核系统。有些初始化只需要执行一次（全局数据结构），而有些初始化每个 hart 都需要执行（per-CPU 数据）。例如：
- `trapinit()`：设置系统调用表（全局），只执行一次
- `trapinithart()`：设置 `stvec`（per-CPU），每个 hart 都执行

### Q5：第一个用户进程（init 进程）的代码是如何加载到内存的？

**答**：init 进程的代码（`initcode.S`）在编译时被嵌入到内核二进制中（通过 `incbin` 汇编指令）。`userinit()` 函数在创建进程时，将 `initcode[]` 数组的内容复制到进程的用户地址空间（虚拟地址 0 处）。然后通过 `exec` 系统调用加载真正的 `/init` 程序。

### Q6：xv6 的内核页表为什么使用恒等映射？

**答**：恒等映射（identity mapping）意味着虚拟地址 = 物理地址。这简化了启动流程：
1. 在分页启用之前，CPU 使用物理地址访问内核代码和数据
2. 启用分页后，同样的地址仍然有效，无需修改代码中的任何地址
3. 这避免了在切换到分页模式时需要跳转到新地址的复杂性

### Q7：`userinit()` 中为什么 `trapframe->epc = 0`？

**答**：因为 initcode 的机器码被复制到了用户地址空间的虚拟地址 0 处（`uvminit(p->pagetable, initcode, sizeof(initcode))`）。当进程从内核态返回用户态时，`sret` 指令将 `sepc` 的值加载到 PC。内核在 `usertrapret()` 中将 `sepc` 设为 `trapframe->epc`，所以用户程序从虚拟地址 0 开始执行。

### Q8：`sret` 和 `mret` 的区别是什么？

**答**：
- `sret`：从 S-mode 返回到之前的模式（通常是 U-mode）。将 `sepc` 加载到 PC，恢复 `sstatus`。
- `mret`：从 M-mode 返回到之前的模式（通常是 S-mode）。将 `mepc` 加载到 PC，恢复 `mstatus`。

两者都会切换特权级，是 RISC-V 中唯一的合法特权级切换指令。

### Q9：xv6 中 PLIC 的作用是什么？

**答**：PLIC（Platform-Level Interrupt Controller）负责将外部设备中断路由到特定的 CPU hart。xv6 使用 PLIC 管理两类外部中断：
- UART 中断（键盘输入）
- VirtIO 磁盘中断（磁盘读写完成）

PLIC 的关键操作：设置中断优先级、使能中断源、设置当前 hart 的中断阈值。

### Q10：为什么 xv6 的启动顺序中 `kinit()` 必须在 `kvminit()` 之前？

**答**：`kinit()` 初始化物理页分配器，将空闲物理页加入链表。`kvminit()` 需要调用 `kalloc()` 来分配物理页存储页表。如果 `kinit()` 还没有执行，`kalloc()` 无法分配内存，页表就无法创建。这是一个严格的依赖关系。

---

## 扩展阅读

### 推荐资源

1. **MIT 6.S081 课程**：https://pdos.csail.mit.edu/6.828/2021/
   - 官方课程网站，包含所有 lecture notes 和 labs

2. **xv6 源码与注释**：https://github.com/mit-pdos/xv6-riscv
   - 官方 GitHub 仓库

3. **xv6 book**：https://pdos.csail.mit.edu/6.828/2021/xv6/book-riscv-rev3.pdf
   - MIT 官方的 xv6 代码注释书，详细解释每个模块

4. **RISC-V 特权架构规范**：https://riscv.org/technical/specifications/
   - RISC-V 特权级规范的官方文档

5. **RISC-V Sv39 页表详解**：
   - 理解 Sv39 的最佳方式是阅读 RISC-V Privileged Specification 的 Virtual Memory 章节

### 相关章节

- **Chapter 1: 硬件基础与系统启动** — 启动流程的硬件视角
- **Chapter 3: 进程基础** — 进程结构与生命周期
- **Chapter 4: 上下文切换** — swtch() 的详细实现
- **Chapter 7: 分页** — 虚拟内存与页表基础
- **Chapter 21: xv6 文件系统** — 文件系统各层的实现

### 实验建议

MIT 6.S081 的以下 lab 与本章内容直接相关：

1. **Lab: Xv6 and Unix utilities** — 熟悉 xv6 的用户态编程
2. **Lab: System calls** — 实现新的系统调用，理解 trap 流程
3. **Lab: Page tables** — 修改内核页表，理解 Sv39
4. **Lab: Traps** — 实现信号处理，理解 trap 机制

---

> **本章小结**
>
> xv6 的启动流程是一个从硬件到软件、从底层到高层的精心设计的过程：
>
> 1. **QEMU** 加载内核到物理内存，CPU 从 M-mode 开始执行
> 2. **entry.S** 设置栈，跳转到 `start()`
> 3. **start()** 配置 M-mode 硬件，`mret` 切换到 S-mode 的 `main()`
> 4. **main()** 按顺序初始化所有内核子系统，最后调用 `userinit()` 创建第一个进程
> 5. **init 进程** 通过 `exec` 加载 `/init`，启动 shell
>
> 理解这个启动流程，就理解了操作系统从"无"到"有"的完整过程。
