---
title: "Chapter 28: xv6 进程调度与同步"
description: "深入理解 xv6 进程结构与状态转换，掌握 scheduler/swtch 实现，理解 sleep/wakeup 机制"
updated: "2026-06-11"
---

# Chapter 28: xv6 进程调度与同步

> **本章目标**：
> - 深入理解 xv6 的 `struct proc` 数据结构及其各个字段的含义
> - 掌握进程状态 UNUSED/USED/SLEEPING/RUNNABLE/RUNNING/ZOMBIE 的转换规则
> - 理解 scheduler() 调度器的无限循环与每 CPU 设计
> - 深入分析 swtch() 的汇编实现与上下文切换机制
> - 掌握 sleep/wakeup 的锁协议与 lost wakeup 问题
> - 理解 kill/wait 的实现与进程资源回收

---

## 28.1 进程结构

### 28.1.1 struct proc 详解

在 xv6 中，每个进程由一个 `struct proc` 结构体表示，它是进程控制块（PCB）的具体实现。所有进程的 PCB 存储在一个固定大小的数组 `proc[NPROC]` 中，最多支持 64 个并发进程。

```c
// kernel/proc.h

struct proc {
  struct spinlock lock;

  // p->lock must be held when using these:
  enum procstate state;        // Process state
  void *chan;                   // If non-zero, sleeping on chan
  int killed;                  // If non-zero, have been killed
  int xstate;                  // Exit status to be returned to parent's wait
  int pid;                     // Process ID

  // wait_lock must be held when using this:
  struct proc *parent;         // Parent process

  // these are private to the process, so p->lock need not be held.
  uint64 kstack;               // Virtual address of kernel stack
  uint64 sz;                   // Size of process memory (bytes)
  pagetable_t pagetable;       // User page table
  struct trapframe *trapframe; // data by trampoline.S
  struct context context;      // swtch() here to run process
  struct file *ofile[NOFILE];  // Open files
  struct inode *cwd;           // Current directory
  char name[16];               // Process name (debugging)
};
```

<div data-component="Xv6ProcStructViewer"></div>

让我们逐一分析每个关键字段：

#### 状态字段 `state`

`state` 是一个枚举类型，定义了进程可能处于的所有状态：

```c
// kernel/proc.h
enum procstate {
  UNUSED,      // 未使用：PCB 槽位空闲
  USED,        // 已使用：正在被分配（过渡状态）
  SLEEPING,    // 睡眠中：等待某个事件
  RUNNABLE,    // 可运行：等待 CPU 调度
  RUNNING,     // 运行中：正在 CPU 上执行
  ZOMBIE       // 僵尸：已退出但父进程尚未回收
};
```

#### 上下文 `context`

`context` 保存了进程的内核寄存器状态，用于上下文切换：

```c
// kernel/proc.h
struct context {
  uint64 ra;    // 返回地址 (Return Address)
  uint64 sp;    // 栈指针 (Stack Pointer)
  uint64 s0;    // callee-saved 寄存器 s0-s11
  uint64 s1;
  uint64 s2;
  uint64 s3;
  uint64 s4;
  uint64 s5;
  uint64 s6;
  uint64 s7;
  uint64 s8;
  uint64 s9;
  uint64 s10;
  uint64 s11;
};
```

为什么只需要保存 14 个寄存器？因为在 RISC-V 调用约定中：
- **caller-saved 寄存器**（t0-t6, a0-a7）：由调用者负责保存，编译器在调用函数前已经将它们压栈
- **callee-saved 寄存器**（s0-s11, ra, sp）：由被调用者负责保存，swtch() 需要显式保存

#### 陷阱帧 `trapframe`

`trapframe` 指向进程的陷阱帧页面，保存了从用户态陷入内核态时的用户寄存器状态：

```c
// kernel/trapframe.h
struct trapframe {
  /*   0 */ uint64 kernel_satp;   // 内核页表
  /*   8 */ uint64 kernel_sp;     // 内核栈指针
  /*  16 */ uint64 kernel_trap;   // usertrap() 函数地址
  /*  24 */ uint64 epc;           // 用户程序计数器
  /*  32 */ uint64 kernel_hartid; // CPU ID
  /*  40 */ uint64 ra;
  /*  48 */ uint64 sp;
  /*  56 */ uint64 gp;
  /*  64 */ uint64 tp;
  /*  72 */ uint64 t0;
  /*  ... */                      // 所有 32 个通用寄存器
  /* 248 */ uint64 t6;
};
```

#### 页表 `pagetable`

`pagetable` 是进程的用户页表，每个进程拥有独立的地址空间。内核页表是全局共享的。

#### 内核栈 `kstack`

`kstack` 是进程的内核栈虚拟地址。当进程陷入内核态时，使用这个栈来执行内核代码。每个进程有独立的内核栈，大小为 4096 字节。

#### 睡眠通道 `chan`

`chan` 是一个任意值的指针，进程在 `sleep()` 时指定等待的通道。`wakeup()` 通过匹配 `chan` 来唤醒对应的进程。这个机制类似于条件变量。

### 28.1.2 进程表与 ptable.lock

所有进程的 PCB 存储在全局数组 `proc[NPROC]` 中：

```c
// kernel/proc.c
struct proc proc[NPROC];  // NPROC = 64
```

进程表的访问由 `ptable.lock`（一个自旋锁）保护：

```c
// kernel/proc.c
struct {
  struct spinlock lock;
  struct proc proc[NPROC];
} ptable;
```

`ptable.lock` 是 xv6 中最关键的锁之一，它保护：
- 进程状态的读写（`p->state`）
- 进程链表的遍历
- `sleep()` 和 `wakeup()` 的操作

<div data-component="Xv6ProcessStateTransition"></div>

### 28.1.3 进程状态转换

进程的六种状态之间的转换关系如下：

```
                    alloc()                usertrapret()
  UNUSED ──────────────► USED ──────────────────► RUNNABLE
     ▲                    │                          │
     │                    │ exit() / kill()          │ scheduler()
     │                    ▼                          ▼
     │               SLEEPING ◄──────────────── RUNNING
     │                 │    sleep()    yield()     ▲
     │                 │ wakeup()                  │
     │                 ▼                           │
     │             RUNNABLE ──────────────────────┘
     │                                    swtch()
     │                ZOMBIE
     │                  │
     │                  │ wait() by parent
     │                  ▼
     └──────────── UNUSED (freeproc)
```

各个转换的触发条件：

| 转换 | 触发函数 | 说明 |
|------|---------|------|
| UNUSED → USED | `allocproc()` | 分配新的 PCB 槽位 |
| USED → RUNNABLE | `usertrapret()` / `forkret()` | 进程初始化完成，可以被调度 |
| RUNNABLE → RUNNING | `scheduler()` | 调度器选中该进程，调用 swtch() |
| RUNNING → RUNNABLE | `yield()` | 时间片用完，主动让出 CPU |
| RUNNING → SLEEPING | `sleep()` | 等待 I/O 或其他事件 |
| SLEEPING → RUNNABLE | `wakeup()` | 等待的事件发生 |
| RUNNING → ZOMBIE | `exit()` | 进程主动退出 |
| RUNNING → ZOMBIE | `kill()` + trap | 被其他进程杀死 |
| ZOMBIE → UNUSED | `wait()` → `freeproc()` | 父进程回收资源 |

---

## 28.2 调度器

### 28.2.1 scheduler() 的设计

xv6 的调度器是一个简单的 **轮转调度（Round-Robin）** 算法。每个 CPU 核心运行自己的调度器循环，遍历进程表寻找 RUNNABLE 进程。

```c
// kernel/proc.c

void scheduler(void)
{
  struct proc *p;
  struct cpu *c = mycpu();

  c->proc = 0;
  for(;;){
    // 避免死锁：中断必须开启
    intr_on();

    int found = 0;
    for(p = proc; p < &proc[NPROC]; p++){
      acquire(&p->lock);
      if(p->state == RUNNABLE) {
        // 找到一个可运行进程，切换到它
        p->state = RUNNING;
        c->proc = p;
        swtch(&c->context, &p->context);

        // 进程完成/让出后，从这里恢复
        c->proc = 0;
        found = 1;
      }
      release(&p->lock);
    }
    if(found == 0){
      // 没有可运行进程，等待中断
      intr_on();
      asm volatile("wfi");
    }
  }
}
```

### 28.2.2 调度器的关键特性

**1. 每 CPU 独立调度器**

每个 CPU 核心有自己的调度器实例。`mycpu()` 返回当前 CPU 的 `struct cpu` 结构体，其中包含：
- `c->context`：调度器的上下文（当进程让出 CPU 时恢复到这里）
- `c->proc`：当前正在运行的进程（如果没有则为 0）

```
CPU 0                    CPU 1
┌──────────────┐        ┌──────────────┐
│ scheduler()  │        │ scheduler()  │
│ 遍历 proc[]  │        │ 遍历 proc[]  │
│ 找 RUNNABLE  │        │ 找 RUNNABLE  │
│ swtch → proc │        │ swtch → proc │
└──────────────┘        └──────────────┘
    c->context              c->context
```

**2. 无限循环**

调度器是一个 `for(;;)` 无限循环。当所有进程都不是 RUNNABLE 时，如果没有 shell 进程在运行，xv6 会 panic。正常情况下至少有一个 shell 进程处于 RUNNABLE 状态。

**3. 持锁粒度**

调度器在检查每个进程时获取该进程的 `p->lock`，而不是获取全局的 `ptable.lock`。这是 xv6-riscv 相比 xv6-x86 的改进——更细粒度的锁减少了锁竞争。

**4. 中断开启**

循环开始时调用 `intr_on()` 确保中断是开启的。如果中断被关闭，定时器中断无法触发，进程永远无法被抢占，导致系统挂起。

### 28.2.3 yield() 与调度

当定时器中断发生时，当前进程通过以下路径被让出：

```c
// kernel/trap.c
void usertrap(void) {
  // ...
  if(which_dev == 2)  // 定时器中断
    yield();
  // ...
}

// kernel/proc.c
void yield(void) {
  struct proc *p = myproc();
  acquire(&p->lock);
  p->state = RUNNABLE;
  sched();
  release(&p->lock);
}
```

`yield()` 的工作流程：
1. 获取当前进程的锁
2. 将状态设为 RUNNABLE（表示可以让出 CPU）
3. 调用 `sched()` 执行实际的上下文切换
4. 当进程重新获得 CPU 时，释放锁

### 28.2.4 sched() 的安全检查

`sched()` 是实际调用 `swtch()` 之前的守门员，它进行一系列安全检查：

```c
// kernel/proc.c
void sched(void) {
  struct proc *p = myproc();

  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  if(p->state == RUNNING)
    panic("sched running");
  if(intr_get())
    panic("sched interruptible");
  // ...
  swtch(&p->context, &mycpu()->context);
}
```

安全检查包括：
- **必须持有 p->lock**：防止竞态条件
- **中断必须关闭**：切换期间不能被中断
- **不能是 RUNNING 状态**：必须先设为其他状态
- **锁嵌套层数必须为 1**：只持有 p->lock，不能持有其他锁

---

## 28.3 上下文切换

### 28.3.1 swtch() 的汇编实现

上下文切换的核心是 `swtch()` 函数，它在 RISC-V 上用汇编实现：

```assembly
# kernel/swtch.S

# swtch(struct context *old, struct context *new)
#
# Save current registers in old. Load from new.

.globl swtch
swtch:
        # 保存旧进程的 callee-saved 寄存器
        sd ra, 0(a0)       # old->ra = ra
        sd sp, 8(a0)       # old->sp = sp
        sd s0, 16(a0)      # old->s0 = s0
        sd s1, 24(a0)      # old->s1 = s1
        sd s2, 32(a0)      # old->s2 = s2
        sd s3, 40(a0)      # old->s3 = s3
        sd s4, 48(a0)      # old->s4 = s4
        sd s5, 56(a0)      # old->s5 = s5
        sd s6, 64(a0)      # old->s6 = s6
        sd s7, 72(a0)      # old->s7 = s7
        sd s8, 80(a0)      # old->s8 = s8
        sd s9, 88(a0)      # old->s9 = s9
        sd s10, 96(a0)     # old->s10 = s10
        sd s11, 104(a0)    # old->s11 = s11

        # 加载新进程的 callee-saved 寄存器
        ld ra, 0(a1)       # ra = new->ra
        ld sp, 8(a1)       # sp = new->sp
        ld s0, 16(a1)      # s0 = new->s0
        ld s1, 24(a1)      # s1 = new->s1
        ld s2, 32(a1)      # s2 = new->s2
        ld s3, 40(a1)      # s3 = new->s3
        ld s4, 48(a1)      # s4 = new->s4
        ld s5, 56(a1)      # s5 = new->s5
        ld s6, 64(a1)      # s6 = new->s6
        ld s7, 72(a1)      # s7 = new->s7
        ld s8, 80(a1)      # s8 = new->s8
        ld s9, 88(a1)      # s9 = new->s9
        ld s10, 96(a1)     # s10 = new->s10
        ld s11, 104(a1)    # s11 = new->s11

        ret                 # 跳转到 new->ra
```

<div data-component="Xv6ContextSwitch"></div>

### 28.3.2 swtch() 的调用场景

`swtch()` 在两种场景下被调用，形成了一个对称的切换模式：

**场景 1：进程 → 调度器**（进程让出 CPU）

```
yield() → sched() → swtch(&p->context, &c->context)
```

此时 `a0 = &p->context`（保存进程寄存器），`a1 = &c->context`（加载调度器寄存器）。`ret` 跳转到 `c->context.ra`，即调度器中 `swtch()` 返回后的位置。

**场景 2：调度器 → 进程**（调度器选中进程）

```
scheduler() → swtch(&c->context, &p->context)
```

此时 `a0 = &c->context`（保存调度器寄存器），`a1 = &p->context`（加载进程寄存器）。`ret` 跳转到 `p->context.ra`，即进程上次在 `sched()` 中 `swtch()` 返回后的位置。

### 28.3.3 上下文切换的完整流程

让我们追踪一次完整的时间片切换：

```
初始状态：进程 A 在 CPU 0 上运行

1. 定时器中断触发
   ┌─────────────────────────────────────┐
   │ 硬件：sepc ← pc, 切换到 S 模式      │
   │ 跳转到 stvec (trampoline 代码)       │
   └─────────────────────────────────────┘
                    │
                    ▼
2. usertrap() 处理定时器中断
   ┌─────────────────────────────────────┐
   │ usertrap() → yield()                │
   │ yield(): p->state = RUNNABLE        │
   │ yield(): sched()                    │
   └─────────────────────────────────────┘
                    │
                    ▼
3. sched() → swtch(&A->context, &c->context)
   ┌─────────────────────────────────────┐
   │ 保存 A 的 ra,sp,s0-s11 到 A->context│
   │ 加载 c 的 ra,sp,s0-s11              │
   │ ret → 跳转到 c->context.ra          │
   └─────────────────────────────────────┘
                    │
                    ▼
4. scheduler() 恢复执行
   ┌─────────────────────────────────────┐
   │ c->proc = 0                         │
   │ release(&A->lock)                   │
   │ 继续遍历 proc[] 找下一个 RUNNABLE    │
   └─────────────────────────────────────┘
                    │
                    ▼
5. 找到进程 B (RUNNABLE)
   ┌─────────────────────────────────────┐
   │ B->state = RUNNING                  │
   │ c->proc = B                         │
   │ swtch(&c->context, &B->context)     │
   │ 保存 c 的寄存器，加载 B 的寄存器     │
   │ ret → 跳转到 B->context.ra          │
   └─────────────────────────────────────┘
                    │
                    ▼
6. 进程 B 的 sched() 返回
   ┌─────────────────────────────────────┐
   │ yield() 中 swtch() 返回             │
   │ release(&B->lock)                   │
   │ 返回到 usertrap() 或 kerneltrap()   │
   │ usertrapret() → 返回用户态           │
   └─────────────────────────────────────┘
```

### 28.3.4 为什么只保存 callee-saved 寄存器

一个常见的问题是：为什么 `swtch()` 不保存所有寄存器？

答案在于 RISC-V 的调用约定（Calling Convention）：

- **caller-saved 寄存器**（t0-t6, a0-a7, ra 在某些情况下）：编译器在调用另一个函数之前，已经将这些寄存器保存到栈上（如果后续还需要使用的话）。因此当 `swtch()` 被调用时，这些寄存器的值已经被编译器保存在栈上了。

- **callee-saved 寄存器**（s0-s11, sp）：这些寄存器由被调用的函数负责保存和恢复。`swtch()` 作为"被调用者"，必须显式保存它们。

由于 `swtch()` 在调用约定中被视为一个普通函数调用，编译器已经处理了 caller-saved 寄存器，所以 `swtch()` 只需要保存 callee-saved 寄存器。

---

## 28.4 sleep 与 wakeup

### 28.4.1 sleep() 的实现

`sleep()` 让当前进程在某个通道上等待，直到被 `wakeup()` 唤醒：

```c
// kernel/proc.c

void sleep(void *chan, struct spinlock *lk)
{
  struct proc *p = myproc();
  
  acquire(&p->lock);          // 1. 获取进程锁
  release(lk);                // 2. 释放调用者持有的锁

  // 修改状态并记录睡眠通道
  p->chan = chan;              // 3. 设置睡眠通道
  p->state = SLEEPING;        // 4. 设为睡眠状态
  
  sched();                    // 5. 让出 CPU

  // 被唤醒后从这里继续
  p->chan = 0;                // 6. 清除睡眠通道

  release(&p->lock);          // 7. 释放进程锁
  acquire(lk);                // 8. 重新获取调用者的锁
}
```

`sleep()` 的参数设计非常精巧：
- `chan`：睡眠通道，是一个任意值的指针（通常是一个全局变量的地址）
- `lk`：调用者当前持有的锁

### 28.4.2 wakeup() 的实现

`wakeup()` 唤醒所有在指定通道上睡眠的进程：

```c
// kernel/proc.c

void wakeup(void *chan)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    if(p != myproc()) {
      acquire(&p->lock);
      if(p->state == SLEEPING && p->chan == chan) {
        p->state = RUNNABLE;  // 唤醒：设为可运行
      }
      release(&p->lock);
    }
  }
}
```

`wakeup()` 遍历整个进程表，找到所有在指定通道上睡眠的进程，将它们的状态改为 RUNNABLE。

<div data-component="SleepWakeupProtocol"></div>

### 28.4.3 Lost Wakeup 问题

**Lost wakeup** 是并发编程中最经典的 bug 之一。如果 sleep/wakeup 的实现不使用锁协议，就会出现这个问题。

考虑一个不正确的实现：

```c
// ❌ 错误的实现：会导致 lost wakeup
void bad_sleep(void *chan) {
  // 没有获取 p->lock
  myproc()->chan = chan;
  myproc()->state = SLEEPING;
  sched();  // 让出 CPU
  myproc()->chan = 0;
}
```

问题场景：

```
时间线：
──────────────────────────────────────────────────────
进程 A (消费者)              进程 B (生产者)
──────────────────────────────────────────────────────
检查条件：buffer 为空
                             往 buffer 写入数据
                             wakeup(chan)
                             → 没有进程在 chan 上睡眠
                             → wakeup 无效！
设置 chan = &buffer
设置 state = SLEEPING
sched() → 让出 CPU
→ 进程 A 永远睡眠！
→ Lost Wakeup!
──────────────────────────────────────────────────────
```

问题的根源：**检查条件和进入睡眠不是原子操作**。在检查条件（buffer 为空）和实际进入睡眠之间，进程 B 可能已经写入数据并发送了 wakeup。

### 28.4.4 锁协议如何防止 Lost Wakeup

正确的 sleep/wakeup 使用锁协议来保证原子性：

```c
// 正确的使用方式
struct spinlock buffer_lock;
int buffer_ready = 0;

// 消费者
acquire(&buffer_lock);
while(!buffer_ready) {
  sleep(&buffer, &buffer_lock);  // 释放 buffer_lock 并睡眠
}
// 被唤醒后，buffer_lock 已被重新获取
buffer_ready = 0;
// 使用 buffer...
release(&buffer_lock);

// 生产者
acquire(&buffer_lock);
// 往 buffer 写入数据
buffer_ready = 1;
wakeup(&buffer);
release(&buffer_lock);
```

锁协议的关键在于 `sleep(chan, lock)` 的参数设计：

```
sleep() 内部操作顺序：

1. acquire(&p->lock)     ← 获取进程锁
2. release(lock)         ← 释放调用者的锁（在此之后，生产者可以获取锁）
3. p->chan = chan
4. p->state = SLEEPING
5. sched()               ← 让出 CPU

关键：步骤 1 和 2 的顺序确保了在释放调用者锁之前，
     进程已经被标记为 SLEEPING。
     即使生产者在步骤 2 之后立即 wakeup()，
     也能看到进程已经在 chan 上睡眠。
```

时间线对比：

```
使用锁协议（正确）：
──────────────────────────────────────────────────────
进程 A (消费者)              进程 B (生产者)
──────────────────────────────────────────────────────
acquire(&buffer_lock)
while(!buffer_ready) {
  sleep() 内部：
  acquire(&p->lock)         acquire(&buffer_lock)
  release(&buffer_lock)     ← 阻塞，等待 A 释放
  p->state = SLEEPING
  p->chan = &buffer
  sched() → 让出 CPU
                            ← 现在可以获取锁了
                            buffer_ready = 1
                            wakeup(&buffer)
                            → 看到 A 在 chan 上睡眠
                            → A 变为 RUNNABLE ✓
──────────────────────────────────────────────────────
```

### 28.4.5 为什么 while 而不是 if

使用 `sleep()` 时，条件检查必须用 `while` 循环而不是 `if`：

```c
while(!condition) {     // ✅ 正确：while
  sleep(chan, lock);
}

if(!condition) {        // ❌ 错误：if
  sleep(chan, lock);
}
```

原因：**虚假唤醒（Spurious Wakeup）**。多个进程可能在同一个通道上睡眠，`wakeup()` 会唤醒所有进程。当进程被唤醒时，条件可能已经被其他进程消费了，所以需要重新检查条件。

---

## 28.5 锁的使用

### 28.5.1 ptable.lock 的作用

在 xv6 的原始设计中（xv6-x86 版本），`ptable.lock` 是一个全局锁，保护整个进程表的所有操作：

```c
// xv6-x86 风格
struct {
  struct spinlock lock;
  struct proc proc[NPROC];
} ptable;
```

在 xv6-riscv 中，设计改为每个进程一把锁：

```c
// xv6-riscv 风格
struct proc {
  struct spinlock lock;  // 每进程锁
  // ...
};

struct proc proc[NPROC];  // 无全局 ptable.lock
```

这种改进减少了锁竞争：当 CPU 0 在修改进程 1 的状态时，CPU 1 可以同时修改进程 2 的状态。

### 28.5.2 持锁不能 sleep 的规则

这是 xv6 中最重要的锁规则之一：

> **规则**：持有任何自旋锁（spinlock）时，不能调用 `sleep()`。

违反这个规则会导致死锁。原因：

```c
// ❌ 错误：持锁 sleep
acquire(&some_lock);
sleep(chan, &some_lock);  // 死锁！
```

`sleeP()` 内部会调用 `sched()` 让出 CPU。如果在持有自旋锁的情况下让出 CPU，其他等待这把锁的 CPU 会永远自旋下去——它们在等待锁被释放，但持有锁的进程已经睡眠了。

正确的做法是先释放锁，再 sleep：

```c
// ✅ 正确
acquire(&some_lock);
// ... 使用共享数据 ...
release(&some_lock);
sleep(chan, &some_lock);  // sleep 会在内部释放并重新获取 some_lock
```

### 28.5.3 锁的获取顺序

为了避免死锁，xv6 遵循严格的锁获取顺序：

```
锁层级（从高到低）：

1. ide.lock          (磁盘 I/O)
2. bcache.lock       (缓冲区缓存)
3. ptable.lock       (进程表)  或  p->lock (单进程)
4. inode->lock       (索引节点)
5. file->lock        (文件)
```

**规则**：获取低层级的锁之前，必须先释放高层级的锁。

```c
// ❌ 错误：违反锁顺序
acquire(&inode->lock);     // 层级 4
acquire(&ptable.lock);     // 层级 3 — 更低层级
// 可能导致死锁！

// ✅ 正确
acquire(&ptable.lock);     // 层级 3
acquire(&inode->lock);     // 层级 4 — 更高层级
// 或者先释放 ptable.lock，再获取 inode->lock
```

### 28.5.4 常见的锁使用模式

**模式 1：sleep 前释放锁**

```c
// 磁盘 I/O 等待
acquire(&ide.lock);
// ... 启动磁盘操作 ...
sleep(&ide_buf, &ide.lock);  // 释放 ide.lock 并等待
// ... 磁盘操作完成 ...
release(&ide.lock);
```

**模式 2：wakeup 前获取锁**

```c
// 磁盘中断处理
acquire(&ide.lock);
// ... 标记操作完成 ...
wakeup(&ide_buf);  // 唤醒等待的进程
release(&ide.lock);
```

**模式 3：条件检查 + sleep 的原子性**

```c
acquire(&lock);
while(!condition) {
  sleep(chan, &lock);  // 原子地释放锁并进入睡眠
}
// condition 为真，继续执行
release(&lock);
```

### 28.5.5 锁与中断的交互

自旋锁和中断之间也有重要的交互规则：

```
CPU 0 上的进程              CPU 0 上的中断处理
──────────────────────────────────────────────
acquire(&lock)
  关闭中断
  自旋等待...
                            定时器中断！
                            acquire(&lock)
                            → 自旋等待锁...
                            → 但中断已关闭！
                            → 永远等不到锁！
                            → 死锁！
──────────────────────────────────────────────
```

xv6 的解决方案：`acquire()` 在自旋等待之前关闭中断。

```c
void acquire(struct spinlock *lk) {
  push_off();      // 关闭中断（增加中断嵌套计数）
  // ... 自旋获取锁 ...
}

void release(struct spinlock *lk) {
  // ... 释放锁 ...
  pop_off();       // 恢复中断状态
}
```

`push_off()` 和 `pop_off()` 使用引用计数，支持嵌套：

```c
void push_off(void) {
  int old = intr_get();
  intr_off();
  if(mycpu()->noff == 0)
    mycpu()->intena = old;
  mycpu()->noff += 1;
}

void pop_off(void) {
  struct cpu *c = mycpu();
  if(intr_get())
    panic("pop_off - interruptible");
  if(c->noff < 1)
    panic("pop_off");
  c->noff -= 1;
  if(c->noff == 0 && c->intena)
    intr_on();
}
```

---

## 28.6 kill 与 wait

### 28.6.1 kill() 的实现

`kill()` 系统调用用于终止另一个进程：

```c
// kernel/proc.c

int kill(int pid)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++){
    acquire(&p->lock);
    if(p->pid == pid){
      p->killed = 1;  // 设置 killed 标志
      if(p->state == SLEEPING){
        // 如果进程在睡眠，唤醒它
        // 让它尽快检查 killed 标志并退出
        p->state = RUNNABLE;
      }
      release(&p->lock);
      return 0;
    }
    release(&p->lock);
  }
  return -1;
}
```

`kill()` 的关键设计：

1. **不直接终止进程**：只设置 `p->killed = 1` 标志
2. **唤醒睡眠进程**：如果目标进程在 SLEEPING 状态，将其唤醒为 RUNNABLE
3. **延迟处理**：实际的退出操作在进程下次获得 CPU 时由 trap 代码处理

### 28.6.2 trap 中检查 killed 标志

进程在以下位置检查 `p->killed` 标志：

```c
// kernel/trap.c — usertrap()
void usertrap(void) {
  // ...
  if(p->killed) {
    exit(-1);  // 如果被杀，立即退出
  }
  // ...
}

// kernel/trap.c — kerneltrap() (通过 devintr)
// 在等待 I/O 时也会检查
```

```c
// kernel/proc.c — sleep()
void sleep(void *chan, struct spinlock *lk) {
  // ...
  sched();
  
  // 被唤醒后检查 killed
  p->chan = 0;
  // ... (p->killed 的检查在调用者中进行)
}
```

在多个系统调用和 I/O 等待点，xv6 都会检查 `p->killed`：

```c
// 例如在 read() 系统调用中
int fileread(struct file *f, uint64 addr, int n) {
  // ...
  while(copyout(p->pagetable, addr, buf, m) < 0) {
    // I/O 等待
    if(p->killed) {
      release(&icache.lock);
      return -1;
    }
  }
  // ...
}
```

### 28.6.3 wait() 的实现

`wait()` 系统调用让父进程等待子进程退出并回收资源：

```c
// kernel/proc.c

int wait(uint64 addr)
{
  struct proc *pp;
  int havekids, pid;
  struct proc *p = myproc();

  acquire(&wait_lock);

  for(;;){
    havekids = 0;
    for(pp = proc; pp < &proc[NPROC]; pp++){
      if(pp->parent == p){
        havekids = 1;
        acquire(&pp->lock);

        if(pp->state == ZOMBIE){
          // 找到一个僵尸子进程
          pid = pp->pid;
          if(addr != 0 && copyout(p->pagetable, addr,
             (char *)&pp->xstate, sizeof(pp->xstate)) < 0){
            release(&pp->lock);
            release(&wait_lock);
            return -1;
          }
          freeproc(pp);        // 回收子进程资源
          release(&pp->lock);
          release(&wait_lock);
          return pid;
        }
        release(&pp->lock);
      }
    }

    // 没有子进程，或者子进程都还在运行
    if(!havekids || killed(p)){
      release(&wait_lock);
      return -1;
    }

    // 等待子进程退出
    sleep(p, &wait_lock);
  }
}
```

`wait()` 的关键行为：

1. **遍历子进程**：查找状态为 ZOMBIE 的子进程
2. **复制退出状态**：将子进程的退出状态复制到用户空间
3. **回收资源**：调用 `freeproc()` 释放子进程的所有资源
4. **阻塞等待**：如果没有子进程退出，sleep 等待

### 28.6.4 exit() 的实现

`exit()` 系统调用终止当前进程：

```c
// kernel/proc.c

void exit(int status)
{
  struct proc *p = myproc();

  if(p == init)
    panic("init exiting");

  // 关闭所有打开的文件
  for(int fd = 0; fd < NOFILE; fd++){
    if(p->ofile[fd]){
      struct file *f = p->ofile[fd];
      fileclose(f);
      p->ofile[fd] = 0;
    }
  }

  begin_op();
  iput(p->cwd);
  end_op();
  p->cwd = 0;

  acquire(&wait_lock);

  // 将子进程转交给 init 进程
  reparent(p);

  // 唤醒父进程
  wakeup(p->parent);
  
  acquire(&p->lock);

  p->xstate = status;
  p->state = ZOMBIE;  // 变为僵尸状态

  release(&wait_lock);

  // 跳入调度器，永不返回
  sched();
  panic("zombie exit");
}
```

`exit()` 的关键步骤：

1. **关闭文件描述符**：释放所有打开的文件
2. **释放当前目录**：`iput(p->cwd)` 减少目录 inode 的引用计数
3. **转交子进程**：将所有子进程的 parent 指向 init 进程
4. **唤醒父进程**：`wakeup(p->parent)` 通知父进程
5. **变为 ZOMBIE**：设置退出状态，变为僵尸状态
6. **调用 sched()**：让出 CPU，永不返回

### 28.6.5 freeproc() 资源回收

当父进程通过 `wait()` 回收子进程时，调用 `freeproc()` 释放资源：

```c
// kernel/proc.c

static void freeproc(struct proc *p)
{
  if(p->trapframe)
    kfree((void*)p->trapframe);
  p->trapframe = 0;
  if(p->pagetable)
    proc_freepagetable(p->pagetable, p->sz);
  p->pagetable = 0;
  p->sz = 0;
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;
}
```

释放的资源包括：
- **trapframe 页面**：释放给物理内存分配器
- **用户页表**：释放所有用户页面和页表页面
- **进程元数据**：清零所有字段，状态设为 UNUSED

### 28.6.6 进程生命周期总结

```
                    allocproc()
  UNUSED ─────────────────────► USED
     ▲                            │
     │                         forkret()
     │                            │
     │                            ▼
     │                       RUNNABLE ◄─────────┐
     │                            │              │
     │                       scheduler()         │
     │                            │              │
     │                            ▼              │
     │                       RUNNING ────────────┘
     │                         │    yield() + sched()
     │                         │
     │                    ┌────┴────┐
     │                    │         │
     │               exit()    kill() + trap
     │                    │         │
     │                    ▼         ▼
     │                  ZOMBIE ◄────┘
     │                    │
     │               wait() by parent
     │                    │
     │               freeproc()
     └────────────────────┘
```

---

## 28.7 面试高频考点

### Q1: xv6 的进程状态有哪些？转换条件是什么？

**答**：xv6 有 6 种进程状态：
- **UNUSED**：PCB 槽位空闲
- **USED**：正在分配（过渡状态）
- **SLEEPING**：等待事件（I/O、信号等）
- **RUNNABLE**：可运行，等待 CPU
- **RUNNING**：正在 CPU 上执行
- **ZOMBIE**：已退出，等待父进程回收

关键转换：RUNNING → RUNNABLE（yield/时间片）、RUNNING → SLEEPING（sleep）、SLEEPING → RUNNABLE（wakeup）、RUNNING → ZOMBIE（exit）

### Q2: 为什么 swtch() 只保存 14 个寄存器？

**答**：因为 RISC-V 调用约定将寄存器分为 caller-saved 和 callee-saved。caller-saved（t0-t6, a0-a7）由编译器在函数调用前自动保存到栈上。swtch() 作为普通函数调用，只需要保存 callee-saved 寄存器（ra, sp, s0-s11），共 14 个。

### Q3: 什么是 Lost Wakeup？如何防止？

**答**：Lost Wakeup 是指 wakeup() 信号丢失，导致进程永远睡眠。发生在 sleep 和 wakeup 之间的竞态条件：进程 A 检查条件后、进入睡眠前，进程 B 已经修改条件并发送了 wakeup。

防止方法：使用锁协议——sleep(chan, lock) 保证"释放锁"和"进入睡眠"是原子操作。在 sleep 内部，先获取 p->lock，再释放调用者的锁，这样 wakeup() 在获取 p->lock 时能看到进程已经处于 SLEEPING 状态。

### Q4: 为什么 sleep() 使用 while 循环检查条件？

**答**：因为存在虚假唤醒（Spurious Wakeup）。wakeup() 会唤醒所有在同一通道上睡眠的进程。当进程被唤醒时，条件可能已经被其他进程消费了。使用 while 循环保证每次唤醒后重新检查条件。

### Q5: kill() 为什么不直接终止进程？

**答**：直接终止进程会留下资源泄漏（打开的文件、分配的内存等）。xv6 的设计是：kill() 只设置 p->killed 标志，进程在下次检查点（trap 处理、系统调用返回）时自行清理并退出。这种设计更安全，确保资源被正确释放。

### Q6: 为什么 exit() 将子进程转交给 init？

**答**：避免僵尸进程积累。如果父进程先于子进程退出，子进程会变成"孤儿僵尸"——没有父进程调用 wait() 回收它们。xv6 将所有孤儿进程的 parent 设为 init 进程（pid=1），init 会周期性调用 wait() 回收这些进程。

### Q7: xv6 调度器的优缺点？

**答**：
- **优点**：实现简单，易于理解；每 CPU 调度器减少锁竞争
- **缺点**：O(N) 遍历进程表，效率低；不支持优先级调度；不支持实时任务；没有考虑进程的 I/O 密集型 vs CPU 密集型特性

### Q8: p->lock 和 ptable.lock 的区别？

**答**：在 xv6-riscv 中，每个进程有自己的 `p->lock`，用于保护单个进程的状态。在 xv6-x86 中使用全局的 `ptable.lock`。per-process lock 减少了锁粒度，提高了并发性——多个 CPU 可以同时修改不同进程的状态。

---

## 本章小结

本章深入分析了 xv6 的进程调度与同步机制：

1. **进程结构**：`struct proc` 包含进程的所有状态信息，存储在 `proc[NPROC]` 数组中
2. **进程状态**：6 种状态（UNUSED/USED/SLEEPING/RUNNABLE/RUNNING/ZOMBIE）及其转换规则
3. **调度器**：scheduler() 无限循环遍历进程表，每 CPU 独立运行
4. **上下文切换**：swtch() 汇编实现只保存 14 个 callee-saved 寄存器
5. **sleep/wakeup**：基于通道的进程同步机制，使用锁协议防止 lost wakeup
6. **锁的使用**：持锁不能 sleep，锁顺序避免死锁
7. **kill/wait**：延迟终止设计，wait 回收 ZOMBIE 子进程资源

---

## 思考题

1. 如果将 xv6 的 round-robin 调度器改为优先级调度，需要修改哪些代码？需要添加哪些数据结构？

2. 在 swtch() 中，如果忘记保存某个 callee-saved 寄存器（比如 s11），会导致什么问题？

3. 假设两个进程同时调用 `wakeup(&chan)`，会发生什么？xv6 如何保证正确性？

4. 如果 xv6 的 `sleep()` 不调用 `sched()` 而是忙等待（busy-wait），会有什么问题？

5. 如何修改 xv6 的 `wait()` 使其支持 `WNOHANG` 选项（非阻塞等待）？

6. 分析 xv6 的调度器在以下场景下的行为：
   - 只有一个 CPU 密集型进程在运行
   - 一个 CPU 密集型进程和一个 I/O 密集型进程同时运行
   - 所有进程都在睡眠等待 I/O

---

## 扩展阅读

### 推荐资源

1. **MIT 6.S081 教材**：*xv6: a simple, Unix-like teaching operating system*
   - Chapter 5: Scheduling — 进程调度与上下文切换
   - Chapter 6: Locking — 锁与同步原语

2. **xv6 源码**：https://github.com/mit-pdos/xv6-riscv
   - `kernel/proc.c` — 进程管理与调度
   - `kernel/swtch.S` — 上下文切换汇编
   - `kernel/trap.c` — 中断与系统调用处理

3. **OSTEP（Operating Systems: Three Easy Pieces）**
   - Chapter 7: Scheduling: Introduction — 调度基本概念
   - Chapter 25: Condition Variables — 条件变量与 sleep/wakeup
   - Chapter 28: Locks — 锁的实现

4. **RISC-V 特权架构手册**
   - 调用约定（Calling Convention）
   - 中断与异常处理

5. **经典论文**：
   - *The Linux Scheduler: a Decade of Wasted Cores* — 现代调度器设计
   - *Sleeping Locks vs. Spinning Locks* — 锁设计权衡

---

> **下一章预告**：我们将学习 xv6 的文件系统实现，理解 inode、目录和路径解析的工作原理。
