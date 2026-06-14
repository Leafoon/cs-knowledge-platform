---
title: "Chapter 26: xv6 陷阱与系统调用"
description: "深入理解 xv6 陷阱处理机制，掌握系统调用从 ecall 到返回的完整路径"
updated: "2026-06-11"
---

# Chapter 26: xv6 陷阱与系统调用

> **本章目标**：
> - 理解陷阱（Trap）的三种分类：系统调用、异常、中断
> - 掌握 RISC-V 陷阱处理的硬件机制与特权级切换
> - 逐行剖析 trampoline.S 中的 uservec 和 userret 汇编代码
> - 深入理解 trap.c 中 usertrap() 和 usertrapret() 的完整实现
> - 掌握系统调用从 ecall 到返回用户态的完整路径
> - 理解 trapframe 结构与 trampoline 页的共享映射设计

---

## 26.1 陷阱机制概述

### 26.1.1 什么是陷阱？

在前面的章节中，我们学习了进程、虚拟内存、上下文切换等核心概念。但有一个关键问题始终没有深入讨论：**用户程序如何与操作系统交互？** 当用户程序需要读取文件、创建进程、分配内存时，它无法直接操作硬件——这些操作必须由内核完成。用户程序需要一种机制来"陷入"内核，请求内核服务，然后"返回"用户态继续执行。

这个机制就是**陷阱（Trap）**。陷阱是 CPU 从正常的指令执行流中被强制打断，转而执行一段特殊的处理代码的过程。在 RISC-V 架构中，陷阱是一个统一的概念，涵盖了三种不同但机制相似的事件：

**系统调用（System Call）**：这是用户程序主动发起的陷阱。当用户程序调用 `read()`、`write()`、`fork()` 等系统调用时，它执行一条特殊的指令（RISC-V 中是 `ecall`），这条指令会触发陷阱，CPU 从用户态（User Mode）切换到内核态（Supervisor Mode），跳转到内核的陷阱处理程序。内核完成相应的服务后，再返回用户态。这是用户程序与操作系统之间的"受控入口"——用户程序不能直接调用内核函数，只能通过系统调用这个"窗口"来请求服务。

**异常（Exception）**：这是 CPU 在执行指令时遇到的意外情况。常见的异常包括：
- **页错误（Page Fault）**：访问的虚拟地址没有映射到物理内存
- **非法指令（Illegal Instruction）**：执行了未定义或不允许的指令
- **断点（Breakpoint）**：执行了 `ebreak` 指令，用于调试
- **除零错误（Division by Zero）**：整数除法时除数为零

异常是被动发生的——程序本身并不想触发陷阱，而是因为某些错误或特殊情况导致 CPU 不得不中断当前执行。

**中断（Interrupt）**：这是外部设备发来的信号。常见的中断包括：
- **定时器中断（Timer Interrupt）**：定时器硬件发出，用于时间片调度
- **UART 中断**：串口设备发出，表示有键盘输入数据
- **磁盘中断**：磁盘控制器发出，表示磁盘 I/O 操作完成
- **Virtio 中断**：virtio 设备发出，用于 QEMU 虚拟设备通信

中断是异步的——它与当前正在执行的指令无关，随时可能发生。

```
陷阱的三种来源：

┌─────────────────────────────────────────────────────────┐
│                      CPU 执行指令                        │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │ 系统调用  │    │  异 常    │    │  中 断    │           │
│  │ ecall    │    │ 非法指令  │    │ 定时器    │           │
│  │ ebreak   │    │ 页错误    │    │ UART     │           │
│  │ (主动)    │    │ 除零      │    │ 磁盘     │           │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘           │
│       │               │               │                 │
│       ▼               ▼               ▼                 │
│  ┌─────────────────────────────────────────┐            │
│  │          陷阱处理程序 (Trap Handler)      │            │
│  │  保存状态 → 分发处理 → 恢复状态 → 返回   │            │
│  └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### 26.1.2 RISC-V 陷阱处理的设计目标

RISC-V 的陷阱处理机制设计遵循几个核心原则：

**第一，透明性**。陷阱对于用户程序应该是"透明"的——用户程序不需要知道陷阱发生的细节，系统调用看起来就像一个普通的函数调用。用户程序只需要执行 `ecall`，然后等待返回结果。所有的状态保存、特权级切换、分发处理都由硬件和内核自动完成。

**第二，安全性**。陷阱必须保证内核的安全性。用户程序不能绕过陷阱机制直接进入内核态，不能修改内核的代码或数据，不能访问其他进程的内存。特权级切换是硬件强制的，用户态的代码无法将自己提升到内核态。

**第三，效率**。陷阱处理的开销应该尽可能小。因为系统调用是频繁发生的操作（每次 I/O、每次进程创建都会触发），如果陷阱处理开销太大，系统性能会严重下降。RISC-V 的硬件设计尽量减少陷阱时需要软件完成的工作。

**第四，灵活性**。内核应该能够灵活地处理不同类型的陷阱，并且能够轻松扩展新的系统调用。

### 26.1.3 特权级切换

RISC-V 定义了三种特权级（在 xv6 使用的模式下）：

| 特权级 | 编码 | 名称 | 用途 |
|--------|------|------|------|
| 0 | U-mode | 用户模式 | 用户程序运行 |
| 1 | S-mode | 监督者模式 | 操作系统内核运行 |
| 3 | M-mode | 机器模式 | 固件/引导程序运行 |

xv6 主要使用 U-mode 和 S-mode 两个特权级。用户程序在 U-mode 下运行，受到诸多限制：
- 不能执行特权指令（如修改页表、关闭中断）
- 不能直接访问内核地址空间
- 不能直接访问硬件设备
- 只能通过 `ecall` 指令发起系统调用

当 `ecall` 执行时，硬件自动完成以下操作：
1. 将当前特权级从 U-mode 切换到 S-mode
2. 将当前 PC 保存到 `sepc` 寄存器
3. 将陷阱原因写入 `scause` 寄存器
4. 将 `stvec` 寄存器的值加载到 PC，跳转到陷阱处理程序
5. 关闭中断（清除 `sstatus.SIE` 位）

这个过程是硬件自动完成的，不需要软件干预。这保证了特权级切换的原子性和安全性——用户程序没有机会在切换过程中插入恶意代码。

### 26.1.4 关键 CSR 寄存器

RISC-V 使用**控制状态寄存器（Control and Status Registers, CSR）** 来管理陷阱处理。以下是 xv6 陷阱处理中最关键的几个 CSR：

**sepc（Supervisor Exception Program Counter）**：
- 存放触发陷阱时的 PC 值
- 当陷阱处理完成执行 `sret` 指令时，CPU 会将 `sepc` 的值加载到 PC，从而回到用户程序被打断的地方继续执行
- 在系统调用的情况下，`sepc` 指向 `ecall` 指令本身，因此 `sret` 后会重新执行 `ecall`——但 xv6 的 usertrapret() 会将 `sepc` 加 4，跳过 `ecall`，直接执行下一条指令

**scause（Supervisor Cause）**：
- 存放陷阱的原因编码
- 最高位（bit 63）区分中断和异常：1 表示中断，0 表示异常
- 低位编码表示具体原因：
  - 8：用户态系统调用（ecall from U-mode）
  - 12：指令页错误
  - 13：加载页错误
  - 15：存储页错误
  - 1：操作异常（如断点）
  - 5：加载访问错误
  - 7：存储访问错误

**stvec（Supervisor Trap Vector）**：
- 存放陷阱处理程序的入口地址
- 当陷阱发生时，硬件将 PC 设置为 `stvec` 的值
- 在 xv6 中，`stvec` 指向 trampoline 页中的 `uservec` 代码
- 低两位用于模式选择：0 表示 Direct 模式（所有陷阱跳转到同一地址），1 表示 Vectored 模式（不同陷阱类型跳转到不同地址）。xv6 使用 Direct 模式

**sstatus（Supervisor Status）**：
- 包含多个控制位，最重要的有：
  - `SPP`（Supervisor Previous Privilege）：陷阱发生前的特权级，0 表示 U-mode，1 表示 S-mode。`sret` 指令根据 SPP 恢复特权级
  - `SIE`（Supervisor Interrupt Enable）：S-mode 中断使能位。陷阱发生时硬件会自动清除此位，关闭中断
  - `SPIE`（Supervisor Previous Interrupt Enable）：陷阱发生前的中断使能状态。`sret` 指令会将 SPIE 恢复到 SIE

**sscratch（Supervisor Scratch）**：
- 一个通用的临时寄存器，用于陷阱处理程序和用户程序之间的通信
- 在 xv6 中，`sscratch` 保存当前进程的 trapframe 的物理地址
- 这是陷阱处理中第一个被使用的寄存器——uservec 代码通过 `sscratch` 找到 trapframe，从而保存用户寄存器

```
RISC-V 陷阱处理相关 CSR 寄存器：

┌────────────────────────────────────────────────────────┐
│ stvec        │ 陷阱处理程序入口地址                      │
│              │ [ MODE | BASE (高 62 位) ]               │
├────────────────────────────────────────────────────────┤
│ sepc         │ 陷阱发生时的 PC 值                        │
│              │ sret 时恢复到此地址                       │
├────────────────────────────────────────────────────────┤
│ scause       │ 陷阱原因编码                              │
│              │ [ Interrupt (1 bit) | Exception Code ]   │
├────────────────────────────────────────────────────────┤
│ stval        │ 附加信息（如页错误的虚拟地址）              │
├────────────────────────────────────────────────────────┤
│ sstatus      │ 处理器状态                                │
│              │ [ SPP | SPIE | SIE | ... ]              │
├────────────────────────────────────────────────────────┤
│ sscratch     │ 临时寄存器，xv6 中存放 trapframe 地址     │
└────────────────────────────────────────────────────────┘
```

---

## 26.2 陷阱处理流程

这是本章的核心内容。我们将完整追踪一个系统调用从用户态触发 `ecall` 到返回用户态的每一步。这个流程涉及汇编代码（trampoline.S）和 C 代码（trap.c）的紧密配合。

### 26.2.1 总体流程概览

```
用户态 (U-mode)                     内核态 (S-mode)
─────────────────                   ─────────────────

用户程序                            usertrap()
    │                               (trap.c)
    │ ecall                             │
    │                                   │ 检查 scause
    ▼                                   │
┌─────────┐                             │ 分发处理
│ 硬件自动 │                             │ (系统调用/中断/异常)
│ 保存状态 │                             │
│ 切换模式 │                             ▼
└────┬────┘                         syscall()
     │                              (syscall.c)
     ▼                                  │
uservec()                               │ 查表分发
(trampoline.S)                          │ 执行具体系统调用
    │ 保存 31 个用户寄存器               │
    │ 到 trapframe                      ▼
    │ 跳转到 usertrap()             usertrapret()
    │                               (trap.c)
    │                                   │ 准备返回用户态
    │                                   │ 设置 stvec = uservec
    │                                   │ 设置 sepc = sepc + 4
    │                                   │ 切换页表
    │                                   │ 跳转到 userret()
    ▼                                   │
┌─────────┐                             ▼
│ 硬件自动 │                         userret()
│ 恢复状态 │                         (trampoline.S)
│ 切换模式 │                             │ 恢复 31 个用户寄存器
└────┬────┘                             │ sret
     │                                  │
     ▼                                  │
用户程序继续执行  ◄──────────────────────┘
```

### 26.2.2 步骤一：ecall 指令触发陷阱

一切从用户程序调用系统调用开始。以 `write(1, buf, n)` 为例：

```c
// 用户程序
char buf[] = "hello";
write(1, buf, 5);  // 这个函数调用最终会执行 ecall
```

`write` 是一个库函数，它最终会调用 `ecall` 指令。在 xv6 中，这些包装函数由 `usys.pl` 脚本生成（后面会详细讨论）。

当 `ecall` 指令执行时，RISC-V 硬件**自动**完成以下操作（不需要任何软件参与）：

1. **保存当前 PC 到 `sepc`**：记录用户程序执行到哪里了
2. **设置 `scause`**：写入 8（表示 ecall from U-mode）
3. **将特权级从 U-mode 切换到 S-mode**：设置 `sstatus.SPP = 0`（记录之前是 U-mode）
4. **关闭 S-mode 中断**：清除 `sstatus.SIE`
5. **保存 SIE 到 SPIE**：`sstatus.SPIE = sstatus.SIE`
6. **将 `stvec` 的值加载到 PC**：CPU 跳转到 `stvec` 指向的地址

在 xv6 中，`stvec` 被设置为 trampoline 页中 `uservec` 函数的地址。因此，CPU 开始执行 `uservec`。

> **关键点**：硬件只做最少的工作——保存 PC、设置几个寄存器、切换特权级、跳转到处理程序。其他所有工作（保存用户寄存器、分发处理、恢复寄存器）都由软件完成。

### 26.2.3 步骤二：uservec — 保存用户寄存器

`uservec` 是 trampoline 页中的汇编代码，位于 `kernel/trampoline.S`。它的核心任务是**保存所有用户寄存器到 trapframe**。

为什么需要保存用户寄存器？因为内核代码需要使用 CPU 寄存器来执行自己的计算。如果不保存用户寄存器，内核使用寄存器时就会覆盖用户程序的值，返回用户态后程序就无法正确继续执行。

这里有一个关键难题：**在 `uservec` 执行的第一条指令时，我们还没有保存任何用户寄存器，但我们又需要使用寄存器来访问 trapframe**。怎么办？

解决方案是使用 `sscratch` 寄存器。在进入用户态之前（上一次 `userret` 返回时），内核已经将当前进程的 trapframe 物理地址写入了 `sscratch`。因此，`uservec` 可以通过 `csrrw` 指令交换 `a0` 和 `sscratch` 的值——`a0` 获得了 trapframe 地址，而 `sscratch` 保存了用户程序的 `a0` 值。

以下是 `trampoline.S` 中 `uservec` 的完整源代码与逐行注释：

```asm
# ─────────────────────────────────────────────────────
# kernel/trampoline.S — uservec
# ─────────────────────────────────────────────────────
# 当用户态发生陷阱时，硬件将 PC 设置为 stvec，
# 即本代码的起始地址（uservec）。
# 此时 CPU 已经在 S-mode，但所有通用寄存器
# 仍然保存着用户态的值。
# sscratch = 当前进程的 trapframe 物理地址
# ─────────────────────────────────────────────────────

        .section trampoline
        .globl uservec
uservec:
        #
        # 交换 a0 和 sscratch
        # 执行后：a0 = trapframe 地址，sscratch = 用户的 a0
        #
        csrrw a0, sscratch, a0

        #
        # 现在 a0 指向 trapframe，可以开始保存用户寄存器。
        # 但 trapframe 在内核地址空间中，而当前页表是用户页表。
        # 如何访问 trapframe 呢？
        #
        # 答案是：trampoline 页和 trapframe 页都被映射在
        # 每个用户页表的顶部。trampoline 页映射在 TRAMPOLINE，
        # trapframe 映射在 TRAPFRAME。两者都在用户页表中可用。
        # （这是因为 ecall 不会切换页表，页表切换由软件完成）
        #

        #
        # 保存用户寄存器到 trapframe
        # trapframe 结构定义在 kernel/proc.h
        #
        sd ra, 40(a0)       # 保存 ra（返回地址寄存器）
        sd sp, 48(a0)       # 保存 sp（栈指针）
        sd gp, 56(a0)       # 保存 gp（全局指针）
        sd tp, 64(a0)       # 保存 tp（线程指针）
        sd t0, 72(a0)       # 保存 t0
        sd t1, 80(a0)       # 保存 t1
        sd t2, 88(a0)       # 保存 t2
        sd s0, 96(a0)       # 保存 s0/fp（帧指针）
        sd s1, 104(a0)      # 保存 s1
        sd a1, 120(a0)      # 保存 a1（注意：a0 在 sscratch 中）
        sd a2, 128(a0)      # 保存 a2
        sd a3, 136(a0)      # 保存 a3
        sd a4, 144(a0)      # 保存 a4
        sd a5, 152(a0)      # 保存 a5
        sd a6, 160(a0)      # 保存 a6
        sd a7, 168(a0)      # 保存 a7（系统调用号）
        sd s2, 176(a0)      # 保存 s2
        sd s3, 184(a0)      # 保存 s3
        sd s4, 192(a0)      # 保存 s4
        sd s5, 200(a0)      # 保存 s5
        sd s6, 208(a0)      # 保存 s6
        sd s7, 216(a0)      # 保存 s7
        sd s8, 224(a0)      # 保存 s8
        sd s9, 232(a0)      # 保存 s9
        sd s10, 240(a0)     # 保存 s10
        sd s11, 248(a0)     # 保存 s11
        sd t3, 256(a0)      # 保存 t3
        sd t4, 264(a0)      # 保存 t4
        sd t5, 272(a0)      # 保存 t5
        sd t6, 280(a0)      # 保存 t6

        #
        # 从 sscratch 中恢复用户的 a0，并保存到 trapframe
        # 同时将 sscratch 设为 0，表示当前在内核态
        #
        # 为什么设为 0？这是一个安全检查。
        # 如果 usertrap 发现 sscratch != 0，说明
        # 陷阱来自内核态，这是异常情况。
        #
        csrr t0, sscratch   # t0 = 用户的 a0
        sd t0, 112(a0)      # 保存到 trapframe->a0

        #
        # 保存用户态的 sepc（陷阱发生时的 PC）
        # 注意：此时 sepc 是 ecall 指令的地址
        #
        csrr t0, sepc
        sd t0, 0(a0)        # 保存到 trapframe->epc

        #
        # 加载内核的页表
        # trapframe 中保存了当前进程的内核页表地址
        # （在 proc_pagetable() 创建进程时设置）
        #
        ld t0, 16(a0)       # t0 = trapframe->kernel_satp
        ld sp, 8(a0)        # sp = trapframe->kernel_sp（内核栈）
        ld t1, 24(a0)       # t1 = trapframe->kernel_hartid
        ld t2, 32(a0)       # t2 = trapframe->kernel_trap（usertrap 地址）

        #
        # 切换到内核页表
        # 写入 satp 寄存器，启用新的地址翻译
        # sfence.vma 清除 TLB 缓存
        #
        csrw satp, t0
        sfence.vma zero, zero

        #
        # 跳转到 usertrap()（C 函数）
        # t2 中保存了 usertrap 的地址
        #
        jr t2
```

让我们仔细分析这段代码中的几个关键设计：

**为什么 `a0` 要通过 `sscratch` 交换？** 因为在 `uservec` 的第一条指令时，所有 31 个通用寄存器都保存着用户程序的值。我们需要一个"支点"来开始保存过程。`sscratch` 就是这个支点——它在用户态时保存着 trapframe 地址，通过 `csrrw`（原子交换）一条指令就同时获得了 trapframe 地址（放入 `a0`）和保存了用户的 `a0`（放入 `sscratch`）。

**为什么 trapframe 可以在用户页表中访问？** 这是一个精巧的设计。`ecall` 指令不会切换页表——页表切换需要软件写入 `satp` 寄存器。因此，在 `uservec` 执行时，当前页表仍然是用户页表。为了让 `uservec` 能访问 trapframe 和 trampoline 代码，xv6 将这两个页面映射在每个用户页表的顶部地址：
- `TRAMPOLINE`（虚拟地址）：映射到 trampoline 物理页，包含 uservec 和 userret 代码
- `TRAPFRAME`（虚拟地址）：映射到当前进程的 trapframe 物理页

### 26.2.4 步骤三：usertrap() — 陷阱分发处理

`uservec` 完成寄存器保存和页表切换后，跳转到 `usertrap()` 函数（位于 `kernel/trap.c`）。这是陷阱处理的 C 语言核心。

```c
// kernel/trap.c
//
// 处理来自用户态的陷阱。
// 由 uservec（trampoline.S）调用。
// sstatus 寄存器已经被硬件设置。
//
void
usertrap(void)
{
    // 检查：这个函数只能处理来自用户态的陷阱
    // 如果 sstatus.SPP != 0，说明陷阱来自内核态，应该用 kerneltrap
    if((r_sstatus() & SSTATUS_SPP) != 0)
        panic("usertrap: not from user mode");

    // 将 stvec 设置为 kernelvec
    // 因为在内核态执行期间如果发生陷阱（如设备中断），
    // 应该由 kernelvec 处理，而不是 uservec
    w_stvec((uint64)kernelvec);

    // 保存 usertrap 返回后需要执行的下一条指令地址
    // sepc 是 ecall 指令的地址，加 4 得到 ecall 的下一条指令
    struct proc *p = myproc();
    p->trapframe->epc = r_sepc();

    // 读取 scause 寄存器，判断陷阱类型
    uint64 scause = r_scause();

    if(scause == 8){
        // scause == 8：系统调用（ecall from U-mode）
        // 检查进程是否已被杀死
        if(p->killed)
            exit(-1);

        // sepc 指向 ecall 指令，加 4 跳过它
        // （返回用户态后不应重新执行 ecall）
        p->trapframe->epc += 4;

        // 开启中断（系统调用处理期间允许中断）
        intr_on();

        // 调用 syscall() 分发具体的系统调用
        syscall();
    } else if((which_dev = devintr()) != 0){
        // devintr() 处理设备中断（定时器、UART、virtio 等）
        // 返回值非零表示成功处理了某个设备中断
        // ok
    } else {
        // 未知的陷阱原因，打印错误信息并杀死进程
        printf("usertrap(): unexpected scause 0x%lx pid=%d\n",
               scause, p->pid);
        printf("            sepc=%ld stval=%ld\n", r_sepc(), r_stval());
        p->killed = 1;
    }

    // 如果进程已被杀死，退出
    if(p->killed)
        exit(-1);

    // 如果发生的是定时器中断，让出 CPU
    // 这实现了时间片调度
    if(which_dev == 2)
        yield();

    // 准备返回用户态
    usertrapret();
}
```

`usertrap()` 的逻辑可以概括为：

1. **安全检查**：确认陷阱来自用户态
2. **切换 stvec**：将 stvec 改为 kernelvec，因为接下来在内核态执行期间可能发生中断
3. **保存 sepc**：将 ecall 的下一条指令地址保存到 trapframe
4. **分发处理**：根据 scause 的值，分别处理系统调用、设备中断、或报告未知陷阱
5. **调度**：如果是定时器中断，调用 yield() 让出 CPU
6. **返回**：调用 usertrapret() 准备返回用户态

### 26.2.5 步骤四：usertrapret() — 准备返回用户态

`usertrapret()` 在 `usertrap()` 的末尾被调用，负责为返回用户态做准备：

```c
// kernel/trap.c
//
// 设置返回用户态的陷阱处理路径。
// 在 usertrap() 的末尾调用，
// 也用于 fork() 时设置子进程的首次执行。
//
void
usertrapret(void)
{
    struct proc *p = myproc();

    // 关闭中断，避免在切换过程中被打断
    // （此时 stvec 已经是 kernelvec，但我们即将改回 uservec）
    intr_off();

    // 将 stvec 设置为 uservec 的地址
    // 下次从用户态发生陷阱时，硬件会跳转到 uservec
    uint64 trampoline_uservec = TRAMPOLINE +
        (uservec - trampoline);
    w_stvec(trampoline_uservec);

    // 设置 trapframe 中的字段，供 uservec 使用
    p->trapframe->kernel_satp = r_satp();         // 内核页表
    p->trapframe->kernel_sp = p->kstack + PGSIZE;  // 内核栈顶
    p->trapframe->kernel_trap = (uint64)usertrap;  // usertrap 地址
    p->trapframe->kernel_hartid = r_tp();          // 当前 CPU 核心号

    // 设置 sstatus 寄存器，为 sret 做准备
    uint64 x = r_sstatus();
    x &= ~SSTATUS_SPP;    // 清除 SPP，表示返回到 U-mode
    x |= SSTATUS_SPIE;    // 设置 SPIE，返回后启用中断
    w_sstatus(x);

    // 设置 sepc 为用户程序的下一条指令地址
    // trapframe->epc 在 usertrap 中已设置好
    w_sepc(p->trapframe->epc);

    // 计算 userret 在 trampoline 页中的虚拟地址
    // 切换页表后，需要用 TRAMPOLINE 基址来计算
    uint64 satp = MAKE_SATP(p->pagetable);
    uint64 trampoline_userret = TRAMPOLINE +
        (userret - trampoline);

    // 跳转到 userret，传入 trapframe 地址和用户页表
    // userret 会切换页表、恢复寄存器、执行 sret
    ((void (*)(uint64, uint64))trampoline_userret)(TRAPFRAME, satp);
}
```

`usertrapret()` 的关键操作：

1. **设置 stvec 为 uservec**：下次用户态陷阱会跳转到 trampoline
2. **填充 trapframe 的内核信息**：内核页表、内核栈、usertrap 地址、CPU 核心号
3. **设置 sstatus**：SPP=0（返回 U-mode），SPIE=1（返回后启用中断）
4. **设置 sepc**：指向用户程序应该继续执行的位置
5. **切换到用户页表**：通过调用 userret 来完成

### 26.2.6 步骤五：userret — 恢复用户寄存器并返回

`userret` 是 trampoline.S 中的汇编代码，它完成陷阱处理的最后一步：

```asm
# ─────────────────────────────────────────────────────
# kernel/trampoline.S — userret
# ─────────────────────────────────────────────────────
# 由 usertrapret() 调用
# a0 = TRAPFRAME（trapframe 的虚拟地址）
# a1 = 用户页表的 satp 值
# ─────────────────────────────────────────────────────

        .globl userret
userret:
        #
        # 切换到用户页表
        # sfence.vma 清除 TLB 缓存
        #
        csrw satp, a1
        sfence.vma zero, zero

        #
        # 将用户的 a0 值写入 sscratch
        # 这样下次 uservec 就能通过 sscratch 恢复 a0
        #
        ld t0, 112(a0)      # t0 = trapframe->a0（用户的 a0）
        csrw sscratch, t0   # sscratch = 用户的 a0

        #
        # 从 trapframe 恢复用户寄存器
        #
        ld ra, 40(a0)
        ld sp, 48(a0)
        ld gp, 56(a0)
        ld tp, 64(a0)
        ld t0, 72(a0)
        ld t1, 80(a0)
        ld t2, 88(a0)
        ld s0, 96(a0)
        ld s1, 104(a0)
        ld a1, 120(a0)
        ld a2, 128(a0)
        ld a3, 136(a0)
        ld a4, 144(a0)
        ld a5, 152(a0)
        ld a6, 160(a0)
        ld a7, 168(a0)
        ld s2, 176(a0)
        ld s3, 184(a0)
        ld s4, 192(a0)
        ld s5, 200(a0)
        ld s6, 208(a0)
        ld s7, 216(a0)
        ld s8, 224(a0)
        ld s9, 232(a0)
        ld s10, 240(a0)
        ld s11, 248(a0)
        ld t3, 256(a0)
        ld t4, 264(a0)
        ld t5, 272(a0)
        ld t6, 280(a0)

        #
        # 最后恢复 a0
        #
        ld a0, 112(a0)

        #
        # sret 指令（硬件自动完成）：
        # 1. 将 sepc 的值加载到 PC
        # 2. 将特权级从 S-mode 切换回 U-mode
        # 3. 将 SPIE 恢复到 SIE
        # 4. 设置 SPP = 0
        #
        sret
```

`sret` 执行后，CPU 回到用户态，从 `sepc` 指向的地址继续执行。对于系统调用来说，这个地址就是 `ecall` 的下一条指令——系统调用的返回值已经在 `a0` 寄存器中了。

### 26.2.7 完整流程时序图

```
时间轴 ──────────────────────────────────────────────────────►

用户态 (U-mode)        硬件            内核态 (S-mode)
──────────────        ────            ──────────────

write(1, buf, 5)
  │
  │ ecall
  │──────────────────►│
  │                   │ 保存 PC → sepc
  │                   │ 设置 scause = 8
  │                   │ SPP ← 0 (之前是 U-mode)
  │                   │ 关闭中断
  │                   │ PC ← stvec (uservec)
  │                   │──────────────────────────►│
  │                   │                           uservec:
  │                   │                             csrrw a0, sscratch
  │                   │                             保存 31 个寄存器
  │                   │                             保存 sepc
  │                   │                             切换到内核页表
  │                   │                             jr usertrap
  │                   │                           usertrap():
  │                   │                             stvec ← kernelvec
  │                   │                             sepc += 4
  │                   │                             scause == 8 → syscall()
  │                   │                           syscall():
  │                   │                             查找 syscall 表
  │                   │                             调用 sys_write()
  │                   │                             返回值写入 a0
  │                   │                           usertrapret():
  │                   │                             stvec ← uservec
  │                   │                             sstatus 设置
  │                   │                             sepc ← trapframe->epc
  │                   │                             调用 userret()
  │                   │                           userret():
  │                   │                             切换到用户页表
  │                   │                             恢复 31 个寄存器
  │                   │                             sret
  │                   │◄──────────────────────────│
  │                   │ 恢复 PC ← sepc
  │                   │ 特权级 → U-mode
  │                   │ 恢复 SIE ← SPIE
  │◄──────────────────│
  │
  │ 继续执行
  │ (a0 = write 返回值)
```

---

## 26.3 系统调用实现

### 26.3.1 用户态系统调用包装

在 xv6 中，每个系统调用在用户态都有一个对应的包装函数。这些函数不是手写的，而是由 `user/usys.pl` Perl 脚本自动生成的。

`usys.pl` 的内容非常简洁：

```perl
# user/usys.pl
#!/usr/bin/perl -w

print "# generated by usys.pl - do not edit\n";

print "#include \"kernel/syscall.h\"\n";

sub entry {
    print ".global $1\n";
    print "${1}:\n";
    print " li a7, SYS_${1}\n";   # 将系统调用号加载到 a7
    print " ecall\n";              # 触发陷阱
    print " ret\n";                # 返回（返回值在 a0 中）
}

entry("fork");
entry("exit");
entry("wait");
entry("pipe");
entry("read");
entry("write");
entry("close");
entry("kill");
entry("exec");
entry("fstat");
entry("chdir");
entry("dup");
entry("getpid");
entry("sbrk");
entry("sleep");
entry("uptime");
```

这个脚本为每个系统调用生成一段汇编代码。以 `write` 为例，生成的 `usys.S` 中对应代码是：

```asm
# user/usys.S（由 usys.pl 生成）

.global write
write:
 li a7, SYS_write    # a7 = 系统调用号（SYS_write = 16）
 ecall               # 触发陷阱，陷入内核
 ret                 # 返回（返回值已在 a0 中）
```

这个函数非常简单：将系统调用号放入 `a7` 寄存器，执行 `ecall`，然后返回。用户程序调用 `write(1, buf, 5)` 时，编译器会将参数放入 `a0`、`a1`、`a2` 寄存器（遵循 RISC-V 调用约定），然后调用 `write` 函数，该函数执行 `ecall`。

### 26.3.2 系统调用号与分发表

每个系统调用都有一个唯一的编号，定义在 `kernel/syscall.h` 中：

```c
// kernel/syscall.h
#define SYS_fork    1
#define SYS_exit    2
#define SYS_wait    3
#define SYS_pipe    4
#define SYS_read    5
#define SYS_write   6
#define SYS_close   7
#define SYS_kill    8
#define SYS_exec    9
#define SYS_fstat   10
#define SYS_chdir   11
#define SYS_dup     12
#define SYS_getpid  13
#define SYS_sbrk    14
#define SYS_sleep   15
#define SYS_uptime  16
```

内核中的 `syscall()` 函数根据 `a7` 中的系统调用号查找对应的处理函数：

```c
// kernel/syscall.c

// 系统调用函数指针数组
static uint64 (*syscalls[])(void) = {
[SYS_fork]    sys_fork,
[SYS_exit]    sys_exit,
[SYS_wait]    sys_wait,
[SYS_pipe]    sys_pipe,
[SYS_read]    sys_read,
[SYS_write]   sys_write,
[SYS_close]   sys_close,
[SYS_kill]    sys_kill,
[SYS_exec]    sys_exec,
[SYS_fstat]   sys_fstat,
[SYS_chdir]   sys_chdir,
[SYS_dup]     sys_dup,
[SYS_getpid]  sys_getpid,
[SYS_sbrk]    sys_sbrk,
[SYS_sleep]   sys_sleep,
[SYS_uptime]  sys_uptime,
};

void
syscall(void)
{
    struct proc *p = myproc();
    int num = p->trapframe->a7;   // 从 trapframe 中读取系统调用号

    // 查找并调用对应的系统调用处理函数
    if(num > 0 && num < NELEM(syscalls) && syscalls[num]) {
        p->trapframe->a0 = syscalls[num]();
        // 返回值存入 trapframe->a0
        // userret 恢复寄存器后，a0 就是系统调用的返回值
    } else {
        printf("%d %s: unknown sys call %d\n",
               p->pid, p->name, num);
        p->trapframe->a0 = -1;  // 返回 -1 表示错误
    }
}
```

分发过程非常直接：`a7` 寄存器保存了系统调用号（在 trapframe 中），用它作为索引查找函数指针数组，调用对应的处理函数，将返回值存入 `trapframe->a0`。

### 26.3.3 参数获取机制

系统调用的参数通过 `a0`-`a5` 六个寄存器传递（RISC-V 调用约定）。这些寄存器的值保存在 trapframe 中，内核通过以下辅助函数获取参数：

```c
// kernel/syscall.c

// 从 trapframe 中获取第 n 个整数参数
int
argint(int n, int *ip)
{
    *ip = *(int*)(&myproc()->trapframe->a0 + n);
    return 0;
}

// 从 trapframe 中获取第 n 个指针参数（64 位地址）
int
argaddr(int n, uint64 *ip)
{
    *ip = *(uint64*)(&myproc()->trapframe->a0 + n);
    return 0;
}

// 从 trapframe 中获取第 n 个字符串参数
int
argstr(int n, char *buf, int max)
{
    uint64 addr;
    argaddr(n, &addr);           // 先获取指针
    return fetchstr(addr, buf, max);  // 从用户空间读取字符串
}

// 从用户地址空间读取字符串
int
fetchstr(uint64 addr, char *buf, int max)
{
    struct proc *p = myproc();
    // 从用户页表中读取，确保不会越界
    if(copyinstr(p->pagetable, buf, addr, max) < 0)
        return -1;
    return strlen(buf);
}
```

> **注意**：`argint`、`argaddr`、`argstr` 中的 `n` 是参数的序号（0 表示第一个参数），不是字节偏移。它们通过 `&myproc()->trapframe->a0 + n` 来定位参数——因为 trapframe 中 `a0`、`a1`、`a2`... 是连续存放的。

对于需要从用户空间复制数据的系统调用，还需要使用 `copyin()` 和 `copyout()`：

```c
// kernel/vm.c

// 从用户空间复制数据到内核空间
int
copyin(pagetable_t pagetable, char *dst, uint64 srcva, uint64 len)
{
    // 验证 srcva 是合法的用户地址
    // 逐页复制数据
    // ...
}

// 从内核空间复制数据到用户空间
int
copyout(pagetable_t pagetable, uint64 dstva, char *src, uint64 len)
{
    // 验证 dstva 是合法的用户地址
    // 逐页复制数据
    // ...
}
```

### 26.3.4 完整的 sys_write 实现

让我们追踪 `write(1, buf, 5)` 的完整实现路径：

```c
// kernel/sysfile.c
uint64
sys_write(void)
{
    uint64 fd;       // 文件描述符
    uint64 p;        // 用户缓冲区地址
    int n;           // 要写的字节数

    // 从 trapframe 中获取三个参数
    argfd(0, &fd);      // 第 0 个参数：fd = 1（标准输出）
    argaddr(1, &p);     // 第 1 个参数：p = buf 的地址
    argint(2, &n);      // 第 2 个参数：n = 5

    // 调用 filewrite 完成实际写入
    return filewrite(f, p, n);
}
```

`filewrite()` 会根据文件描述符的类型进行分发：

```c
// kernel/file.c
int
filewrite(struct file *f, uint64 addr, int n)
{
    // ...
    if(f->type == FD_INODE){
        // 写入文件：通过 writei 写入 inode
        begin_op();
        ip = f->ip;
        r = writei(ip, 1, addr, f->off, n);
        f->off += r;
        iunlockput(ip);
        end_op();
        return r;
    } else if(f->type == FD_DEVICE){
        // 写入设备：根据 major 号分发到对应的设备驱动
        if(f->major < 0 || f->major >= NDEV || !devsw[f->major].write)
            return -1;
        return devsw[f->major].write(1, addr, n);
    }
    // ...
}
```

对于 `write(1, buf, 5)`（文件描述符 1 是标准输出），最终会调用 `consolewrite()`：

```c
// kernel/console.c
int
consolewrite(int user_src, uint64 src, int n)
{
    int i;
    for(i = 0; i < n; i++){
        char c;
        if(either_copyin(&c, user_src+i, 1) == -1)
            break;
        uartputc(c);  // 通过 UART 发送字符到控制台
    }
    return i;
}
```

整个 `write(1, buf, 5)` 的调用链：

```
用户态：write(1, buf, 5)
    │
    ▼ ecall
内核态：
    uservec → usertrap → syscall()
        │
        ▼ 查表
    sys_write()
        │
        ▼
    filewrite(f, p, n)
        │
        ▼ f->type == FD_DEVICE, major == CONSOLE
    consolewrite(1, addr, n)
        │
        ▼
    uartputc(c)  × n 次
        │
        ▼
    UART 硬件发送字符
```

### 26.3.5 系统调用的返回值

系统调用的返回值通过 `a0` 寄存器传回用户态。具体机制：

1. `sys_write()` 返回写入的字节数（如 5）
2. `syscall()` 将返回值存入 `p->trapframe->a0`
3. `usertrapret()` 不会修改 trapframe 中的 a0
4. `userret()` 从 trapframe 恢复 a0 寄存器
5. 用户态的 `write` 函数执行 `ret`，返回值在 a0 中

如果系统调用出错，返回值通常是 -1，用户程序可以通过 `errno` 获取具体的错误码。

---

## 26.4 trapframe 结构

### 26.4.1 trapframe 的定义

`trapframe` 是一个保存用户态寄存器状态的数据结构，定义在 `kernel/proc.h` 中：

```c
// kernel/proc.h

struct trapframe {
    /*   0 */ uint64 kernel_satp;    // 内核页表地址
    /*   8 */ uint64 kernel_sp;      // 内核栈指针（指向栈顶）
    /*  16 */ uint64 kernel_trap;    // usertrap() 的地址
    /*  24 */ uint64 kernel_hartid;  // CPU 核心号（tp 寄存器）
    /*  32 */ uint64 epc;            // 用户态 PC（sepc 的值）
    /*  40 */ uint64 ra;             // 返回地址寄存器
    /*  48 */ uint64 sp;             // 栈指针
    /*  56 */ uint64 gp;             // 全局指针
    /*  64 */ uint64 tp;             // 线程指针
    /*  72 */ uint64 t0;             // 临时寄存器 t0
    /*  80 */ uint64 t1;             // 临时寄存器 t1
    /*  88 */ uint64 t2;             // 临时寄存器 t2
    /*  96 */ uint64 s0;             // 保存寄存器 s0/帧指针
    /* 104 */ uint64 s1;             // 保存寄存器 s1
    /* 112 */ uint64 a0;             // 参数/返回值寄存器 a0
    /* 120 */ uint64 a1;             // 参数寄存器 a1
    /* 128 */ uint64 a2;             // 参数寄存器 a2
    /* 136 */ uint64 a3;             // 参数寄存器 a3
    /* 144 */ uint64 a4;             // 参数寄存器 a4
    /* 152 */ uint64 a5;             // 参数寄存器 a5
    /* 160 */ uint64 a6;             // 参数寄存器 a6
    /* 168 */ uint64 a7;             // 系统调用号寄存器
    /* 176 */ uint64 s2;             // 保存寄存器 s2
    /* 184 */ uint64 s3;             // 保存寄存器 s3
    /* 192 */ uint64 s4;             // 保存寄存器 s4
    /* 200 */ uint64 s5;             // 保存寄存器 s5
    /* 208 */ uint64 s6;             // 保存寄存器 s6
    /* 216 */ uint64 s7;             // 保存寄存器 s7
    /* 224 */ uint64 s8;             // 保存寄存器 s8
    /* 232 */ uint64 s9;             // 保存寄存器 s9
    /* 240 */ uint64 s10;            // 保存寄存器 s10
    /* 248 */ uint64 s11;            // 保存寄存器 s11
    /* 256 */ uint64 t3;             // 临时寄存器 t3
    /* 264 */ uint64 t4;             // 临时寄存器 t4
    /* 272 */ uint64 t5;             // 临时寄存器 t5
    /* 280 */ uint64 t6;             // 临时寄存器 t6
};
```

### 26.4.2 字段分析

trapframe 可以分为两个部分：

**前四个字段（内核信息）**：这四个字段不是用户寄存器，而是内核在每次返回用户态时写入的"配置信息"。`uservec` 在保存用户寄存器之前，会先读取这四个字段来完成内核环境的初始化：

| 字段 | 偏移 | 用途 |
|------|------|------|
| `kernel_satp` | 0 | 内核页表的 satp 值，用于切换到内核页表 |
| `kernel_sp` | 8 | 当前进程的内核栈顶，陷阱处理在内核栈上执行 |
| `kernel_trap` | 16 | `usertrap()` 函数的地址，uservec 通过它跳转到 C 代码 |
| `kernel_hartid` | 24 | CPU 核心号，用于多核环境下确定当前核心 |

**后续 32 个字段（用户寄存器）**：保存 RISC-V 的 31 个通用寄存器（x0 硬连线为 0，不需要保存）和 `epc`（用户态 PC）。这些寄存器在 `uservec` 中被保存，在 `userret` 中被恢复。

### 26.4.3 每进程一个 trapframe 页

每个进程都有自己的 trapframe。在 `kernel/proc.c` 的 `allocproc()` 中：

```c
// kernel/proc.c

static struct proc*
allocproc(void)
{
    // ...
    // 分配 trapframe 页
    p->trapframe = (struct trapframe *)kalloc();
    if(p->trapframe == 0){
        freeproc(p);
        release(&p->lock);
        return 0;
    }

    // 初始化 trapframe 中的内核信息
    // （用户寄存器在每次陷阱返回时动态设置）
    // ...

    return p;
}
```

`kalloc()` 分配一个 4096 字节的物理页，正好容纳一个 `trapframe` 结构体。这个页的物理地址被记录在进程的 `p->trapframe` 中，同时被映射到用户页表的 `TRAPFRAME` 虚拟地址处。

### 26.4.4 trampoline 页的共享映射

trampoline 页是一个特殊的物理页，它包含了 `uservec` 和 `userret` 两个函数的代码。这个页有一个独特的设计：**它被映射到每个进程的用户页表和内核页表中的相同虚拟地址（TRAMPOLINE）**。

```
                    物理内存
                    ┌──────────────────────┐
                    │                      │
                    │  trampoline 物理页   │
                    │  (uservec + userret) │
                    │                      │
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
     进程 A 页表         进程 B 页表         内核页表
     ┌─────────┐         ┌─────────┐         ┌─────────┐
     │ ...     │         │ ...     │         │ ...     │
     │TRAMPOLINE│        │TRAMPOLINE│        │TRAMPOLINE│
     │ → 物理页 │        │ → 物理页 │        │ → 物理页 │
     └─────────┘         └─────────┘         └─────────┘

     同一个虚拟地址映射到同一个物理页！
```

为什么这样设计？因为陷阱处理需要在用户页表和内核页表之间切换。在切换的那一刻，代码正在执行 trampoline 页中的指令。如果 trampoline 页不在新页表中映射，页表切换后 CPU 就找不到下一条指令了——这会导致崩溃。通过在所有页表中映射同一物理页到同一虚拟地址，无论使用哪个页表，trampoline 代码始终可以访问。

### 26.4.5 sscratch 寄存器的角色

`sscratch` 在整个陷阱处理过程中扮演着"桥梁"角色：

```
时间点              sscratch 的值
─────────────────────────────────────────────
用户态执行时         trapframe 物理地址
  │
  │ ecall
  ▼
uservec 第一条指令   与 a0 交换 → 保存用户的 a0
  │
  │ 保存完所有寄存器
  ▼
trapframe 中 a0 已保存，sscratch 设为 0
  │
  │ 进入内核态执行
  ▼
userret 中           设回用户的 a0 值
  │
  │ sret
  ▼
用户态执行时         用户的 a0 值（下一次 ecall 前会重新设置）
```

xv6 在每次 `userret` 返回用户态前，将 `sscratch` 设置为 trapframe 的物理地址。这样，下次发生陷阱时，`uservec` 的第一条指令 `csrrw a0, sscratch, a0` 就能获得 trapframe 地址。

### 26.4.6 内存布局图

```
用户页表（高地址部分）：

虚拟地址
──────────────────────────────────────────────
0xFFFFFFFFFFFFFFFF ───┐
                      │
TRAMPOLINE            ├──► trampoline 物理页
(顶部一页)            │    (uservec + userret)
                      │
0xFFFFFFFFFFFFFFFF    │
- PGSIZE ─────────────┤
                      │
TRAPFRAME             ├──► trapframe 物理页
                      │    (p->trapframe)
                      │
0xFFFFFFFFFFFFFFFF    │
- 2*PGSIZE ───────────┤
                      │
                      │  ... 用户地址空间 ...
                      │  (栈、堆、代码、数据)
                      │
0x0 ──────────────────┘


内核页表：

虚拟地址
──────────────────────────────────────────────
0xFFFFFFFFFFFFFFFF ───┐
                      │
TRAMPOLINE            ├──► 同一个 trampoline 物理页
(顶部一页)            │
                      │
0xFFFFFFFFFFFFFFFF    │
- PGSIZE ─────────────┤
                      │
                      │  ... 内核地址空间 ...
                      │  (PLIC, UART, VIRTIO, 内核代码/数据)
                      │
0x80000000 ───────────┤
                      │
                      │  ... 未映射 ...
                      │
0x0 ──────────────────┘
```

---

## 26.5 内核态陷阱处理

### 26.5.1 kernelvec

当内核态发生陷阱时（例如在内核代码执行期间收到设备中断），`stvec` 指向 `kernelvec`。`kernelvec` 与 `uservec` 不同——它不需要切换页表（已经在内核页表中），也不需要通过 `sscratch` 找 trapframe。它只需要保存当前正在使用的寄存器到内核栈上。

```asm
# kernel/kernelvec.S

        .globl kernelvec
kernelvec:
        #
        # 内核态陷阱处理
        # 此时已经在内核栈上运行
        # 保存所有寄存器到当前的栈帧中
        #

        # 为寄存器保存分配栈空间（304 字节 = 38 个 8 字节槽）
        addi sp, sp, -256

        # 保存寄存器到栈上
        sd ra, 0(sp)
        sd sp, 8(sp)    # 虽然是当前 sp，但为了完整性也保存
        sd gp, 16(sp)
        sd tp, 24(sp)
        sd t0, 32(sp)
        sd t1, 40(sp)
        sd t2, 48(sp)
        sd s0, 56(sp)
        sd s1, 64(sp)
        sd a0, 72(sp)
        sd a1, 80(sp)
        sd a2, 88(sp)
        sd a3, 96(sp)
        sd a4, 104(sp)
        sd a5, 112(sp)
        sd a6, 120(sp)
        sd a7, 128(sp)
        sd s2, 136(sp)
        sd s3, 144(sp)
        sd s4, 152(sp)
        sd s5, 160(sp)
        sd s6, 168(sp)
        sd s7, 176(sp)
        sd s8, 184(sp)
        sd s9, 192(sp)
        sd s10, 200(sp)
        sd s11, 208(sp)
        sd t3, 216(sp)
        sd t4, 224(sp)
        sd t5, 232(sp)
        sd t6, 240(sp)

        # 跳转到 kerneltrap()
        call kerneltrap

        # 恢复寄存器
        ld ra, 0(sp)
        ld sp, 8(sp)
        ld gp, 16(sp)
        ld tp, 24(sp)
        ld t0, 32(sp)
        ld t1, 40(sp)
        ld t2, 48(sp)
        ld s0, 56(sp)
        ld s1, 64(sp)
        ld a0, 72(sp)
        ld a1, 80(sp)
        ld a2, 88(sp)
        ld a3, 96(sp)
        ld a4, 104(sp)
        ld a5, 112(sp)
        ld a6, 120(sp)
        ld a7, 128(sp)
        ld s2, 136(sp)
        ld s3, 144(sp)
        ld s4, 152(sp)
        ld s5, 160(sp)
        ld s6, 168(sp)
        ld s7, 176(sp)
        ld s8, 184(sp)
        ld s9, 192(sp)
        ld s10, 200(sp)
        ld s11, 208(sp)
        ld t3, 216(sp)
        ld t4, 224(sp)
        ld t5, 232(sp)
        ld t6, 240(sp)

        addi sp, sp, 256

        # sret 返回到 sepc 指向的内核指令
        sret
```

与 `uservec` 的关键区别：
- **不需要切换页表**：已经在内核页表中
- **不需要 sscratch**：直接使用当前内核栈
- **寄存器保存到栈上**：而非 trapframe（因为内核有自己的栈）
- **使用 `call` 而非 `jr`**：调用 kerneltrap 后会返回

### 26.5.2 kerneltrap

`kerneltrap()` 处理来自内核态的陷阱：

```c
// kernel/trap.c

void
kerneltrap()
{
    int which_dev = 0;
    uint64 sepc = r_sepc();
    uint64 sstatus = r_sstatus();
    uint64 scause = r_scause();

    // 内核态陷阱不应该来自用户态
    if((sstatus & SSTATUS_SPP) == 0)
        panic("kerneltrap: not from s-mode");

    // 内核态中断应该被关闭
    if(intr_get() != 0)
        panic("kerneltrap: interrupts enabled");

    // 处理设备中断
    if((which_dev = devintr()) == 0){
        // 未知陷阱
        printf("scause=0x%lx sepc=0x%lx stval=0x%lx\n",
               scause, r_sepc(), r_stval());
        panic("kerneltrap");
    }

    // 如果是定时器中断，让出 CPU
    if(which_dev == 2 && myproc() != 0)
        yield();

    // 恢复 sepc 和 sstatus
    w_sepc(sepc);
    w_sstatus(sstatus);
}
```

`kerneltrap` 与 `usertrap` 的主要区别：
- 不需要切换 stvec（已经是 kernelvec）
- 不需要调用 `usertrapret`（kernelvec 的 `sret` 直接返回内核代码）
- 必须保存和恢复 `sepc` 和 `sstatus`（因为内核态陷阱可能嵌套）

### 26.5.3 定时器中断处理

定时器中断是 xv6 调度机制的核心。当定时器中断发生时，`devintr()` 会识别并处理它：

```c
// kernel/trap.c

int
devintr()
{
    uint64 scause = r_scause();

    if(scause == 0x8000000000000009L){
        // 外部中断（PLIC）
        int irq = plic_claim();
        if(irq == UART0_IRQ){
            uartintr();         // 处理 UART 中断
        } else if(irq == VIRTIO0_IRQ){
            virtio_disk_intr(); // 处理磁盘中断
        }
        if(irq)
            plic_complete(irq);
        return 1;
    } else if(scause == 0x8000000000000005L){
        // 定时器中断
        // 如果当前是 CPU 0，递增 ticks
        if(cpuid() == 0){
            acquire(&tickslock);
            ticks++;
            wakeup(&ticks);
            release(&tickslock);
        }
        // 设置下一次定时器中断
        w_stimecmp(r_stime() + 1000000);
        return 2;  // 返回 2 表示定时器中断
    } else {
        return 0;  // 未知中断
    }
}
```

定时器中断的处理非常简单：
1. 递增全局时钟计数器 `ticks`（在 CPU 0 上）
2. 唤醒所有在 `ticks` 上等待的进程（如 `sleep()`）
3. 设置下一次定时器中断的时间
4. 返回 2，`usertrap()` 或 `kerneltrap()` 据此调用 `yield()` 让出 CPU

---

## 面试高频考点

### Q1：xv6 中系统调用的完整路径是什么？

**答**：以 `write(1, buf, 5)` 为例：

1. 用户程序调用 `write()`，进入 `user/usys.S` 中的 `write` 包装函数
2. 包装函数将 `SYS_write`（6）放入 `a7`，执行 `ecall`
3. 硬件自动：保存 PC 到 `sepc`，设置 `scause = 8`，切换到 S-mode，跳转到 `stvec`（uservec）
4. `uservec`（trampoline.S）：交换 `a0` 和 `sscratch`，保存 31 个用户寄存器到 trapframe，保存 `sepc`，切换到内核页表，跳转到 `usertrap`
5. `usertrap`（trap.c）：设置 `stvec = kernelvec`，`sepc += 4`，检测 `scause == 8`，调用 `syscall()`
6. `syscall`（syscall.c）：从 `trapframe->a7` 读取系统调用号，查表调用 `sys_write()`
7. `sys_write`：通过 `argfd`/`argaddr`/`argint` 获取参数，调用 `filewrite()`
8. `filewrite()` → `consolewrite()` → `uartputc()`：发送字符到 UART
9. 返回值存入 `trapframe->a0`
10. `usertrapret()`：设置 `stvec = uservec`，设置 `sstatus`，设置 `sepc`，调用 `userret()`
11. `userret()`：切换到用户页表，恢复 31 个寄存器，`sscratch = 用户 a0`，执行 `sret`
12. 硬件自动：恢复 PC 从 `sepc`，切换到 U-mode，恢复 SIE
13. 用户程序从 `ecall` 的下一条指令继续执行，返回值在 `a0` 中

### Q2：为什么 trampoline 页需要映射在所有页表中？

**答**：因为 `ecall` 指令不会自动切换页表——页表切换需要软件写入 `satp` 寄存器。当 `uservec` 开始执行时，当前页表仍然是用户页表。`uservec` 需要执行代码（trampoline 页）来完成寄存器保存，然后才切换到内核页表。如果 trampoline 页只映射在内核页表中，用户页表中没有映射，那么 `ecall` 跳转到 `stvec` 地址时就会触发页错误。

因此，trampoline 页必须在每个用户页表中都有映射。同时，它在内核页表中也映射（在相同虚拟地址），因为 `userret` 在切换到用户页表之前需要执行代码。

此外，trampoline 页映射在所有页表的相同虚拟地址（`TRAMPOLINE`），这样切换页表后，PC 仍然指向正确的代码——不需要在切换页表后重新计算 PC。

### Q3：sscratch 寄存器的作用是什么？为什么需要它？

**答**：`sscratch` 用于在用户态和内核态之间传递信息。在 xv6 中，它保存当前进程的 trapframe 物理地址。

为什么需要它？因为当 `uservec` 开始执行时，所有 31 个通用寄存器都保存着用户程序的值。要保存这些寄存器，我们需要一个指向 trapframe 的指针——但这个指针不能存储在任何通用寄存器中（因为它们都需要被保存）。`sscratch` 是一个 CSR，不占用通用寄存器，它就是这个"额外的存储空间"。

具体机制是通过 `csrrw a0, sscratch, a0` 原子交换：一条指令同时完成"获取 trapframe 地址到 a0"和"保存用户的 a0 到 sscratch"。

### Q4：系统调用的参数是如何从用户态传递到内核态的？

**答**：系统调用的参数通过 RISC-V 的调用约定传递——前 6 个参数分别放在 `a0`-`a5` 寄存器中。系统调用号放在 `a7` 寄存器中。

当 `ecall` 触发陷阱时，这些寄存器的值被 `uservec` 保存到 trapframe 中。内核的 `syscall()` 函数从 `trapframe->a7` 获取系统调用号，各个 `sys_*` 函数通过 `argint(n, &val)`、`argaddr(n, &addr)`、`argstr(n, buf, max)` 从 trapframe 中读取参数。

返回值通过 `a0` 寄存器传回：`syscall()` 将系统调用的返回值存入 `trapframe->a0`，`userret` 恢复寄存器后，用户程序就可以从 `a0` 中读到返回值。

### Q5：usertrap 和 kerneltrap 有什么区别？

**答**：

| 方面 | usertrap | kerneltrap |
|------|----------|------------|
| 陷阱来源 | 用户态（U-mode） | 内核态（S-mode） |
| 入口 | uservec（trampoline.S） | kernelvec（kernelvec.S） |
| 页表 | 需要从用户页表切换到内核页表 | 已经在内核页表 |
| 寄存器保存 | 保存到 trapframe | 保存到内核栈 |
| stvec 切换 | 切换到 kernelvec | 不需要 |
| 返回方式 | usertrapret → userret → sret | kernelvec 直接 sret |
| sepc 处理 | 加 4 跳过 ecall | 保存/恢复（不修改） |

### Q6：为什么 usertrap 要将 stvec 改为 kernelvec？

**答**：在 `usertrap` 执行期间，CPU 处于内核态。如果此时发生中断（如定时器中断），硬件会跳转到 `stvec` 指向的地址。如果 `stvec` 仍然指向 `uservec`，CPU 会错误地执行 `uservec`——`uservec` 会尝试从 `sscratch` 获取 trapframe 地址并保存用户寄存器，但此时已经在内核态，这些操作毫无意义甚至会破坏内核状态。

因此，进入 `usertrap` 后必须将 `stvec` 改为 `kernelvec`，确保内核态中断由 `kernelvec` 正确处理。

### Q7：为什么 sret 之前要修改 sepc 加 4？

**答**：`sepc` 保存的是 `ecall` 指令的地址。`sret` 指令会将 `sepc` 恢复到 PC。如果不修改 `sepc`，返回用户态后会重新执行 `ecall`——这会导致无限循环的系统调用。

因此，在处理系统调用时，`usertrap` 中执行 `p->trapframe->epc += 4`，将返回地址指向 `ecall` 的下一条指令。这样 `sret` 后用户程序就从 `ecall` 之后继续执行。

### Q8：trapframe 和 context 有什么区别？

**答**：

| 方面 | trapframe | context |
|------|-----------|---------|
| 定义位置 | kernel/proc.h：`struct trapframe` | kernel/proc.h：`struct context` |
| 保存内容 | 31 个用户寄存器 + 4 个内核信息字段 | callee-saved 寄存器（ra, sp, s0-s11） |
| 大小 | ~288 字节 | ~104 字节 |
| 用途 | 陷阱处理：用户态 ↔ 内核态切换 | 进程切换：内核线程之间切换 |
| 保存时机 | trap 发生时（uservec） | 调度切换时（swtch） |
| 保存位置 | 每进程的 trapframe 页 | 每进程的内核栈底部 |
| 保存的寄存器 | 全部 31 个通用寄存器 | 只有 callee-saved 寄存器 |

`context` 只保存 callee-saved 寄存器，因为 `swtch` 是一个普通的函数调用——caller-saved 寄存器由编译器在调用 `swtch` 之前自动保存到栈上。而 `trapframe` 必须保存全部 31 个寄存器，因为陷阱是异步发生的，用户程序可能在任何状态下被打断。

### Q9：xv6 如何保证系统调用的安全性？

**答**：xv6 通过多层机制保证安全性：

1. **硬件特权级**：`ecall` 是唯一能从 U-mode 进入 S-mode 的途径。用户程序无法直接执行特权指令。
2. **参数验证**：`argint`/`argaddr`/`argstr` 从 trapframe 中读取参数，而非直接使用用户寄存器。`copyin`/`copyout` 在复制数据时验证用户地址的合法性。
3. **页表隔离**：用户页表和内核页表不同，用户程序无法访问内核地址空间。
4. **sscratch 归零**：进入内核态后 `sscratch` 设为 0，如果内核态再发生 `ecall`（不应该发生），可以通过检查 `sscratch` 发现异常。
5. **系统调用号验证**：`syscall()` 检查系统调用号是否在有效范围内，无效的调用号返回 -1。

### Q10：定时器中断如何实现进程调度？

**答**：定时器中断的处理路径：

1. 硬件定时器每隔一定时间触发中断
2. 如果在用户态：`uservec` → `usertrap()` → `devintr()` 返回 2
3. 如果在内核态：`kernelvec` → `kerneltrap()` → `devintr()` 返回 2
4. `devintr()` 处理定时器中断：递增 `ticks`，唤醒等待进程，设置下一次中断
5. `usertrap` 或 `kerneltrap` 检查 `which_dev == 2`，调用 `yield()`
6. `yield()` 获取进程锁，设置进程状态为 `RUNNABLE`，调用 `sched()`
7. `sched()` 调用 `swtch()` 切换到调度器线程
8. 调度器选择下一个 `RUNNABLE` 进程，切换到该进程

这个机制实现了**抢占式调度**：即使进程不主动让出 CPU，定时器中断也会强制暂停它，确保所有进程公平地分享 CPU 时间。

---

## 扩展阅读

- **MIT 6.S081 课程**：https://pdos.csail.mit.edu/6.828/2021/ — 包含 xv6 的实验和讲义
- **xv6 book（RISC-V 版）**：Chapter 4 "Traps and device drivers" — 官方教材的陷阱处理章节
- **RISC-V 特权级规范**：https://riscv.org/technical/specifications/ — "Volume II: Privileged Architecture"
- **RISC-V 手册（Patterson & Hennessy）**：附录中的陷阱处理部分
- **xv6 源码**：`kernel/trampoline.S`、`kernel/trap.c`、`kernel/syscall.c`、`kernel/proc.h`
- **MIT 6.S081 Lecture 6: Traps**：Robert Morris 教授的陷阱处理讲座
- **xv6 可视化工具**：https://github.com/mit-pdos/xv6-riscv — 可以实际运行和调试 xv6

---

> **本章小结**：陷阱是操作系统中最精密的机制之一——它在硬件和软件的配合下，实现了用户态与内核态之间的安全切换。理解 `ecall` → `uservec` → `usertrap` → `syscall` → `usertrapret` → `userret` → `sret` 的完整路径，是理解操作系统如何提供系统服务的关键。trapframe 保存了用户程序的全部状态，trampoline 页的共享映射解决了页表切换时的指令连续性问题，而 `sscratch` 寄存器巧妙地解决了"保存寄存器前需要一个寄存器来指向保存位置"的鸡生蛋问题。这些设计体现了系统编程中"精确控制每一个字节、每一个时钟周期"的工程精神。
