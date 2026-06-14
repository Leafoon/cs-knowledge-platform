---
title: "Chapter 13: 并发编程基础"
description: "深入理解并发的挑战与竞态条件，掌握临界区问题的三个要求，理解 Peterson 算法、硬件原子指令与自旋锁"
updated: "2026-06-10"
---

# Chapter 13: 并发编程基础

> **本章目标**：
> - 深入理解为什么并发编程困难——竞态条件与不确定性
> - 掌握临界区问题的三个要求：互斥、进步、有限等待
> - 理解从软件方案（Peterson）到硬件原子指令的演进
> - 掌握自旋锁的实现与优化
> - 能够识别并发 Bug 并设计正确的同步方案

---

## 13.1 并发的挑战

### 13.1.1 为什么并发很难？

并发编程的核心挑战是**不确定性（Non-determinism）**：程序的结果取决于线程执行的**相对顺序**，而这个顺序每次运行都可能不同。

在顺序编程中，代码的执行顺序是确定的——从上到下，从左到右。程序员可以精确地预测程序在任意时刻的状态。但在并发编程中，多个线程同时执行，它们的指令可能以任意顺序交错执行。这种交错的可能性随着线程数和指令数的增加而**指数级增长**。

让我们通过一个具体的例子来理解这个问题。考虑两个线程同时执行一个简单的计数器递增操作：

```c
// 共享变量——两个线程都可以读写
int counter = 0;

// 线程 1 和线程 2 同时执行以下代码：
void increment(void) {
    counter++;  // 看似简单的操作
}
```

`counter++` 在 C 语言中看起来是一个原子操作，但在机器指令层面，它被分解为三步：

```
1. LOAD  counter → 寄存器    // 从内存读取 counter 的当前值到寄存器
2. ADD   寄存器, 1           // 将寄存器中的值加 1
3. STORE 寄存器 → counter    // 将寄存器中的新值写回内存
```

这三步之间，操作系统可以随时暂停当前线程，切换到另一个线程。如果两个线程交错执行这三步，就会出现问题：

```
时间    线程 1                      线程 2                      counter 的值
──────────────────────────────────────────────────────────────────────────
t1      LOAD counter → R1 (R1=0)
t2      ADD R1, 1 (R1=1)
t3                                  LOAD counter → R2 (R2=0)    ← 读到旧值！
t4      STORE R1 → counter          ADD R2, 1 (R2=1)
t5                                  STORE R2 → counter          ← 覆盖了线程 1 的结果！

最终结果：counter = 1（期望值：2）
```

在 t3 时刻，线程 2 读取 counter 的值时，线程 1 还没有将新值写回内存（t4 才写回），所以线程 2 读到了旧值（0）。然后线程 2 也计算出 1 并写回。最终 counter = 1，而不是期望的 2。

这种交错在 200000 次递增中可能发生很多次。假设发生了 16544 次，最终结果就是 200000 - 16544 = 183456。**每次运行程序，结果都不同**——这就是并发编程最难调试的地方。

<div data-component="RaceConditionDemo"></div>

### 13.1.2 竞态条件示例

让我们用一个更复杂的例子来展示竞态条件的严重后果。考虑一个简单的银行转账系统：

```c
// 共享变量——两个账户的余额
int account_A = 1000;
int account_B = 1000;

// 线程 1：从 A 转 100 到 B
void transfer_A_to_B(void) {
    if (account_A >= 100) {    // 1. 检查余额是否充足
        account_A -= 100;      // 2. 从 A 扣款
        account_B += 100;      // 3. 向 B 入账
    }
}

// 线程 2：从 B 转 100 到 A
void transfer_B_to_A(void) {
    if (account_B >= 100) {    // 4. 检查余额是否充足
        account_B -= 100;      // 5. 从 B 扣款
        account_A += 100;      // 6. 向 A 入账
    }
}
```

如果两个线程以正常的顺序执行（线程 1 完成后再执行线程 2），结果是正确的：

```
时刻  线程 1                    线程 2                    account_A  account_B
───────────────────────────────────────────────────────────────────────────────
t1    检查 A >= 100 (是)                                          1000       1000
t2    A -= 100                                                     900        1000
t3    B += 100                                                     900        1100
t4                               检查 B >= 100 (是)               900        1100
t5                               B -= 100                        900        1000
t6                               A += 100                       1000        1000

结果：A=1000, B=1000（正确！总金额守恒 = 2000）
```

但如果两个线程交错执行，特别是在线程 1 检查余额之后、扣款之前，线程 2 也检查了余额：

```
时刻  线程 1                    线程 2                    account_A  account_B
───────────────────────────────────────────────────────────────────────────────
t1    读取 A (=1000)                                              1000       1000
t2                               读取 B (=1000)                  1000       1000
t3    检查 A >= 100 (是)                                          1000       1000
t4                               检查 B >= 100 (是)              1000       1000
t5    A = A - 100 (=900)                                          900        1000
t6                               B = B - 100 (=900)              900        900
t7    B = B + 100 (=1000)                                         900        1000
t8                               A = A + 100 (=1100)            1100        1000

结果：A=1100, B=1000 → 总金额 2100！凭空产生了 100！
```

这就是竞态条件的后果：**程序的结果取决于线程执行的相对顺序，变得不可预测**。在银行系统中，这意味着钱可能凭空产生或消失。在操作系统中，这可能导致数据损坏、安全漏洞或系统崩溃。

### 13.1.3 原子性问题

**原子操作（Atomic Operation）** 是不可分割的操作——要么完全执行，要么完全不执行，不会被中断或与其他操作交错。在硬件层面，单条机器指令通常是原子的（由硬件保证）。但高级语言中的操作（如 `counter++`）通常不是原子的。

```c
// 原子操作示例（硬件保证）
// x86 的 LOCK 前缀
lock inc [counter]  // 原子递增——整个操作在一条指令中完成

// 非原子操作示例
counter++;  // 三条指令，不是原子的——可以被中断
```

原子性的缺失是竞态条件的根本原因。如果 `counter++` 是原子的，两个线程不可能交错执行它的内部步骤，竞态条件就不会发生。这就是为什么现代处理器提供了硬件原子指令（如 TSL、CAS、LL/SC）——它们让程序员能够以原子的方式操作共享数据。

### 13.1.4 不可控的调度

在没有同步机制的情况下，操作系统可以在**任意时刻**暂停一个线程并切换到另一个线程。线程可能在执行到 `counter++` 的第一条和第二条指令之间被切换，导致另一个线程看到中间状态。

```
线程 1 执行 counter++：
  LOAD counter → R1    ← 执行到这里被定时器中断切换！
  ADD R1, 1            ← 还没执行！
  STORE R1 → counter   ← 还没执行！

线程 2 执行 counter++：
  LOAD counter → R2    ← 读到旧值（线程 1 还没写回）
  ADD R2, 1
  STORE R2 → counter   ← 写入新值

线程 1 恢复执行：
  ADD R1, 1            ← 使用旧值（线程 1 之前读取的）
  STORE R1 → counter   ← 覆盖线程 2 的结果！
```

这就是为什么我们需要同步机制——它们确保关键操作不会被其他线程打断。

---

## 13.2 临界区问题

### 13.2.1 临界区定义

**临界区（Critical Section）** 是访问共享资源的代码段。同一时刻最多只有一个线程可以执行临界区代码。临界区问题是指设计一个协议，使得多个线程能够安全地访问共享资源。

```c
// 临界区的一般结构
void thread_func(void) {
    // === 进入区（Entry Section）===
    // 检查是否可以进入临界区，如果可以则进入
    enter_critical_section();
    
    // === 临界区（Critical Section）===
    // 访问共享资源——同一时刻只有一个线程在这里
    shared_variable++;
    
    // === 退出区（Exit Section）===
    // 释放临界区，允许其他线程进入
    exit_critical_section();
    
    // === 剩余区（Remainder Section）===
    // 不访问共享资源的代码——可以与其他线程并发执行
    do_other_work();
}
```

临界区的概念非常重要，因为它是所有同步机制的基础。无论是锁、信号量还是条件变量，本质上都是在管理临界区的进入和退出。

### 13.2.2 临界区问题的三个要求

一个好的临界区解决方案必须满足三个要求。这三个要求构成了并发编程的基石，任何同步机制都必须满足它们。

**要求一：互斥（Mutual Exclusion）**

同一时刻最多一个线程在临界区内。这是最基本的要求——违反互斥就会导致竞态条件。互斥保证了临界区内的代码是"原子"执行的——从其他线程的角度看，临界区内的操作要么全部完成，要么全部未开始。

```
正确：线程 A 在临界区时，线程 B 必须等待
线程 A: [====临界区====]
线程 B:                 [等待]  [====临界区====]

错误：两个线程同时在临界区（违反互斥）
线程 A: [====临界区====]
线程 B: [====临界区====]  ← 两个线程同时访问共享资源！
```

**要求二：进步（Progress）**

如果临界区空闲（没有线程在执行临界区代码），等待进入的线程能在**有限时间**内进入。不能出现"临界区空闲但所有线程都在等待"的死锁情况。

进步要求排除了两种不合理的解决方案：
1. "所有人投票决定谁进入"——如果投票过程可能永远不结束，就违反了进步
2. "轮流进入但跳过不想进入的线程"——如果一个线程不想进入，其他线程也不能被卡住

```
正确：临界区空闲时，等待的线程可以进入
线程 A: [====临界区====]  退出
线程 B:                 [等待]  → 临界区空闲，B 进入

错误：临界区空闲但 B 无法进入（违反进步——死锁）
线程 A: [====临界区====]  退出
线程 B:                 [永远等待]  ← 没有人唤醒 B！
```

**要求三：有限等待（Bounded Waiting）**

每个线程等待进入临界区的时间有**上界**。不能出现"某个线程永远无法进入临界区"的饥饿情况。有限等待保证了公平性——每个线程最终都能进入临界区。

```
正确：每个线程最多等待 N 次就能进入
线程 A: [临界区] [临界区] [临界区] ...
线程 B: [等待] [等待] [临界区]  ← 最终进入

错误：B 永远无法进入（违反有限等待——饥饿）
线程 A: [临界区] [临界区] [临界区] ...
线程 B: [等待] [等待] [等待] ...  ← A 总是先抢到，B 永远等！
```

### 13.2.3 单处理器解决方案：禁用中断

在单处理器系统中，最简单的解决方案是**禁用中断**：

```c
void enter_critical(void) {
    disable_interrupts();  // 禁用所有中断（包括定时器中断）
}

void exit_critical(void) {
    enable_interrupts();   // 恢复中断
}
```

**原理**：线程切换是由定时器中断触发的。禁用中断后，当前线程不会被切换，直到它主动释放临界区。这保证了临界区内的代码是"原子"执行的。

**优点**：
- 实现极其简单——只需要两条指令
- 开销极小——禁用/启用中断只需要几个时钟周期（~1ns）

**缺点**：
1. **不适用于多处理器**：禁用中断只影响当前 CPU，其他 CPU 上的线程仍然可以进入临界区。在多处理器系统中，这不能保证互斥。
2. **特权指令**：禁用中断是特权指令（只能在内核态执行），用户态程序不能使用。
3. **影响系统响应性**：禁用中断期间，系统无法响应任何事件——包括 I/O 完成、定时器、用户输入。如果禁用时间过长，用户会感觉到系统"卡死"。
4. **长时间禁用危险**：如果临界区代码有 Bug（如死循环），整个系统会挂起，因为无法响应定时器中断来切换到其他进程。
5. **信任问题**：将禁用中断的能力交给用户程序是危险的——恶意程序可以永久禁用中断，使系统崩溃。

由于这些限制，禁用中断通常只在内核内部使用（如 xv6 的自旋锁实现中），而且只在临界区非常短的情况下使用。

---

## 13.3 软件解决方案

### 13.3.1 Peterson 算法（两进程）

Peterson 算法是第一个正确的两进程临界区解决方案，由 Gary Peterson 于 1981 年提出。它只使用普通的共享变量，不需要特殊的硬件支持。这个算法是并发编程历史上的里程碑——它证明了仅用软件就能解决临界区问题。

```c
// Peterson 算法——两个进程的临界区解决方案
int flag[2] = {0, 0};  // flag[i] = 1 表示进程 i 想进入临界区
int turn = 0;           // 轮到谁——用于打破平局

void enter_critical(int id) {
    int other = 1 - id;         // 对方的编号（0 或 1）
    flag[id] = 1;               // 告诉对方"我想进入临界区"
    turn = other;               // 谦让——"让对方先"
    while (flag[other] && turn == other)
        ;  // 等待：对方也想进入 且 轮到对方
}

void exit_critical(int id) {
    flag[id] = 0;  // 告诉对方"我离开了"
}
```

**Peterson 算法的直觉理解**：

想象两个人（进程 0 和进程 1）都想进入一个只有一个门的房间（临界区）。他们的协议是：
1. 举起手（`flag[id] = 1`）表示"我想进入"
2. 说"你先请"（`turn = other`）——谦让对方
3. 如果对方也举手了（`flag[other] == 1`）且自己说了"你先请"（`turn == other`），就等待
4. 否则进入房间

关键在于 `turn` 变量——它打破了两个进程同时说"你先请"的僵局。如果两个进程同时执行 `turn = other`，最终 `turn` 只会有一个值，那个值对应的进程优先进入。

**正确性分析**：

**互斥**：假设两个进程同时在临界区内。那么 `flag[0]=1` 且 `flag[1]=1`。如果 `turn=0`，进程 1 的 while 条件为真（`flag[0]=1 && turn=0`），进程 1 无法进入——矛盾。如果 `turn=1`，进程 0 的 while 条件为真——矛盾。因此互斥成立。

**进步**：如果进程 0 想进入但被阻塞，那么 `flag[1]=1 && turn=1`。进程 1 要么在临界区（很快退出），要么在 while 循环中。如果进程 1 在 while 循环中，它会执行 `turn = 0`（在 enter_critical 开始时），进程 0 的 while 条件变为假，进程 0 就能进入。因此进步成立。

**有限等待**：进程 0 最多等待进程 1 执行一次临界区就能进入。进程 1 退出时设置 `flag[1]=0`，进程 0 的 while 条件立即变为假。因此有限等待成立。

<div data-component="PetersonAlgorithmVisualizer"></div>

### 13.3.2 Dekker 算法

Dekker 算法是最早的正确两进程临界区解决方案（1965 年），比 Peterson 算法早了 16 年。它的思想与 Peterson 类似，但实现更复杂——使用了额外的标志来避免活锁。

### 13.3.3 面包店算法（Bakery Algorithm，多进程）

面包店算法是 Peterson 算法的多进程推广，由 Lamport 于 1974 年提出。它模拟面包店的取号排队机制：每个进程取一个号码，号码小的优先。

```c
// 面包店算法——N 个进程的临界区解决方案
int choosing[N] = {0};  // choosing[i] = 1 表示进程 i 正在取号
int number[N] = {0};    // number[i] = 进程 i 的号码（0 表示不感兴趣）

void enter_critical(int id) {
    // 步骤 1：取号
    choosing[id] = 1;
    number[id] = 1 + max(number[0], number[1], ..., number[N-1]);
    choosing[id] = 0;
    
    // 步骤 2：等待所有号码更小的进程
    for (int j = 0; j < N; j++) {
        // 等待进程 j 取号完成
        while (choosing[j]) ;
        // 等待进程 j 的号码更大（或号码相同但 id 更大）
        while (number[j] != 0 && 
               (number[j] < number[id] || 
               (number[j] == number[id] && j < id))) ;
    }
}

void exit_critical(int id) {
    number[id] = 0;  // 清除号码，表示离开
}
```

面包店算法的直觉：就像在面包店取号——号码小的先被服务。如果两个进程取到相同的号码，id 小的优先。

### 13.3.4 软件方案的局限性

Peterson 算法和面包店算法在理论上是正确的，但在实际系统中很少使用：

1. **内存乱序（Memory Reordering）**：现代处理器可能重排指令执行顺序，导致 `flag` 和 `turn` 的更新顺序不符合预期。需要使用**内存屏障（Memory Barrier）** 来保证顺序，这增加了复杂性和开销。
2. **忙等待（Busy Waiting）**：while 循环浪费 CPU——进程在等待时不断检查条件，消耗 CPU 但不做有用的工作。
3. **扩展性差**：面包店算法的时间复杂度为 O(N)，进程数增加时性能急剧下降。
4. **复杂性**：正确性证明困难，容易出错。实际系统中更倾向于使用经过充分测试的硬件原子指令。

---

## 13.4 硬件原子指令

### 13.4.1 测试并设置（Test-and-Set / TSL）

**TSL** 是一条原子指令，将内存中的值读取到寄存器，同时将新值写入内存。整个操作是不可分割的——硬件保证在指令执行期间，其他 CPU 不能访问同一内存地址。

```c
// TSL 的语义（硬件保证原子性）
int tsl(int *addr) {
    int old = *addr;    // 读取旧值
    *addr = 1;          // 写入新值（总是 1）
    return old;         // 返回旧值
    // 整个操作在一条指令中完成，不会被中断
    // 其他 CPU 在此期间无法访问 *addr
}
```

x86 汇编实现：
```asm
; 方式 1：使用 LOCK 前缀 + XCHG 指令
TSL:  
    MOV EAX, 1          ; 将 1 加载到寄存器
    XCHG EAX, [addr]    ; 原子地交换 EAX 和 [addr]
    ; EAX 现在包含 [addr] 的旧值，[addr] 被设为 1
    RET

; 方式 2：使用 BTS 指令
TSL:
    MOV EAX, 1
    LOCK BTS [addr], 0  ; 原子地测试并设置第 0 位
    RET
```

RISC-V 汇编实现（使用 LL/SC）：
```asm
TSL:
retry:
    lr.w    t0, (a0)        ; Load-Reserved：加载值并标记地址
    li      t1, 1           
    sc.w    t1, t1, (a0)    ; Store-Conditional：条件存储
    bnez    t1, retry        ; 如果 SC 失败（其他 CPU 修改了地址），重试
    mv      a0, t0           ; 返回旧值
    ret
```

### 13.4.2 比较并交换（Compare-and-Swap / CAS）

**CAS** 是更强大的原子指令：如果内存中的当前值等于期望值，则替换为新值；否则不做任何操作。无论是否替换，都返回内存中的旧值。

```c
// CAS 的语义
int cas(int *addr, int expected, int new_value) {
    int old = *addr;           // 读取当前值
    if (old == expected) {     // 如果等于期望值
        *addr = new_value;     // 替换为新值
    }
    return old;                // 返回旧值（无论是否替换）
    // 整个操作是原子的
}
```

x86 汇编实现：
```asm
CAS:
    MOV EAX, expected      ; 将期望值加载到 EAX
    LOCK CMPXCHG [addr], new_value
    ; 如果 [addr] == EAX，则 [addr] = new_value，ZF=1
    ; 否则 EAX = [addr]，ZF=0
    RET
```

CAS 比 TSL 更强大，因为它是**条件更新**——只在值未被其他线程修改时才更新。这使得 CAS 成为无锁数据结构的基础。

**CAS 的典型使用模式——CAS 循环**：

```c
// 使用 CAS 实现原子递增
void atomic_increment(int *addr) {
    int old_val, new_val;
    do {
        old_val = *addr;           // 读取当前值
        new_val = old_val + 1;     // 计算新值
    } while (cas(addr, old_val, new_val) != old_val);
    // 如果 CAS 失败（其他线程修改了值），重试
    // CAS 返回的值不等于 old_val，说明值被修改了
}
```

这种"读取-计算-比较交换"的模式是无锁编程的核心范式。

### 13.4.3 获取并增加（Fetch-and-Add）

**Fetch-and-Add** 原子地将一个值加到内存位置，并返回旧值。

```c
// Fetch-and-Add 的语义
int fetch_and_add(int *addr, int increment) {
    int old = *addr;        // 读取旧值
    *addr += increment;     // 原子地增加
    return old;             // 返回旧值
}
```

x86 汇编实现：
```asm
FETCH_AND_ADD:
    MOV EAX, increment
    LOCK XADD [addr], EAX  ; 原子地将 EAX 加到 [addr]，EAX = 旧值
    RET
```

Fetch-and-Add 可以用于实现**公平的 Ticket Lock**——每个等待线程通过 Fetch-and-Add 获取一个递增的号码，按号码顺序获取锁。

### 13.4.4 Load-Linked / Store-Conditional (LL/SC)

**LL/SC** 是 RISC-V 和 ARM 使用的原子操作对。与 TSL/CAS 不同，LL/SC 更灵活——可以实现任意的原子操作。

```c
// LL：从内存地址加载值，并标记该地址（设置 reservation）
int ll(int *addr) {
    return *addr;  // 加载值，并在硬件中标记 addr
}

// SC：条件存储——如果 addr 自 LL 以来未被其他 CPU 修改，则存储成功
int sc(int *addr, int new_value) {
    if (addr_not_modified_since_ll) {
        *addr = new_value;
        return 1;  // 成功
    }
    return 0;  // 失败——其他 CPU 修改了 addr
}
```

**LL/SC 的工作原理**：
1. `LL` 从地址加载值，并在 CPU 的缓存中标记该地址（设置 reservation）
2. 如果其他 CPU 修改了该地址（即使写入相同的值），reservation 被清除
3. `SC` 检查 reservation 是否仍然有效：如果有效则存储成功，否则失败

**LL/SC 的优势**：
1. **灵活性**：可以实现任意的原子操作（CAS、fetch-and-add、原子递增等）
2. **避免 ABA 问题**：CAS 可能误判（值从 A 变到 B 再变回 A，CAS 以为没变），LL/SC 不会——因为任何修改都会清除 reservation
3. **硬件友好**：不需要比较逻辑，实现更简单

```c
// 用 LL/SC 实现 CAS
int cas(int *addr, int expected, int new_value) {
    int old = ll(addr);          // 加载并标记
    if (old == expected) {       // 如果等于期望值
        if (sc(addr, new_value)) // 尝试条件存储
            return old;          // 成功
    }
    return old;                  // 失败
}
```

<div data-component="AtomicInstructionComparison"></div>

### 13.4.5 原子指令的实现：缓存一致性协议

原子指令的原子性由**缓存一致性协议**（如 MESI 协议）保证。当一个 CPU 执行原子指令时：

1. 它获取对应缓存行的**独占（Exclusive）** 访问权
2. 在独占期间，其他 CPU 不能访问同一缓存行（被阻塞或等待）
3. 原子操作完成后，释放独占权，其他 CPU 可以重新访问

这就是为什么原子指令比普通指令慢——它们需要缓存一致性协议的额外协调，可能需要使其他 CPU 的缓存行失效。

### 13.4.6 原子指令的性能对比

不同原子指令的性能差异很大。让我们通过具体数据来理解：

```
指令          无竞争延迟    有竞争延迟（4 CPU）    适用场景
普通读写       ~1 ns        ~1 ns                 非共享数据
TSL           ~10 ns       ~50 ns                 简单锁
CAS           ~10 ns       ~50 ns                 无锁数据结构
Fetch-and-Add ~10 ns       ~30 ns                 计数器、Ticket Lock
LL/SC         ~10 ns       ~40 ns                 复杂原子操作
```

**为什么 CAS 比 TSL 慢？** CAS 需要比较逻辑（检查当前值是否等于期望值），而 TSL 只是无条件地设置为 1。但在有竞争时，CAS 的重试循环可能导致更多缓存一致性流量。

**为什么 Fetch-and-Add 比 CAS 快？** Fetch-and-Add 不需要重试——它总是成功。CAS 需要重试循环（读取-比较-交换），在高竞争时可能多次失败。

### 13.4.7 手算练习：CAS 重试次数

**题目**：3 个线程同时使用 CAS 递增一个共享计数器。初始值为 0。每个线程执行 100 次递增。计算期望的 CAS 重试次数。

**解答**：

每次 CAS 操作有 3 种可能：
- 成功（当前值等于期望值）：概率 1/3（3 个线程中只有一个能成功）
- 失败（其他线程修改了值）：概率 2/3

每次操作的期望重试次数 = 失败概率 / 成功概率 = (2/3) / (1/3) = 2 次

总操作次数 = 3 线程 × 100 次 = 300 次
期望重试次数 = 300 × 2 = 600 次
总 CAS 调用次数 = 300 + 600 = 900 次

**实际测量**：在 4 核 CPU 上，使用 `perf stat` 测量 CAS 操作的缓存未命中次数。高竞争时，每次 CAS 失败都可能导致一次缓存行失效。

---

## 13.5 自旋锁（Spinlock）

### 13.5.1 基于 TSL 的自旋锁实现

自旋锁是最简单的锁实现：当锁被占用时，线程在一个循环中**自旋等待**，直到锁被释放。

```c
// 自旋锁数据结构
typedef struct {
    int locked;  // 0 = 未锁定, 1 = 已锁定
} spinlock_t;

// 初始化锁
void spinlock_init(spinlock_t *lock) {
    lock->locked = 0;
}

// 获取锁——如果锁被占用则自旋等待
void spinlock_acquire(spinlock_t *lock) {
    while (test_and_set(&lock->locked) == 1)
        ;  // 忙等待：不断尝试获取锁
    // test_and_set 返回 1（旧值=1，锁被占用）→ 继续循环
    // test_and_set 返回 0（旧值=0，锁空闲）→ 获取成功，退出循环
}

// 释放锁
void spinlock_release(spinlock_t *lock) {
    lock->locked = 0;  // 普通写入即可
    // 不需要原子操作——只有持有锁的线程才会释放锁
}
```

**工作原理详解**：
1. `test_and_set(&lock->locked)` 原子地将 `locked` 设为 1，并返回旧值
2. 如果旧值为 0（锁之前未被占用），说明获取成功——当前线程现在持有了锁
3. 如果旧值为 1（锁之前已被占用），说明获取失败——继续循环重试
4. 循环会一直执行，直到锁被其他线程释放（`locked` 被设为 0）

### 13.5.2 自旋等待的问题

自旋锁的主要问题是**忙等待（Busy Waiting）**：线程在 while 循环中消耗 CPU，但什么有用的工作都不做。

```
线程 A 持有锁，在临界区内工作：
  [====临界区执行====]

线程 B 在 while 循环中自旋等待：
  [while][while][while][while][while]... → 浪费 CPU！
  每次循环都执行一条指令，但没有做任何有用的工作
```

**自旋锁的适用场景**：
- **临界区很短**（几条指令，~ns 级别）：自旋时间很短，浪费的 CPU 很少
- **多处理器系统**：自旋的线程在其他 CPU 上运行，不影响持有锁的 CPU
- **内核代码**：内核中不能随意睡眠（如在中断处理程序中），只能使用自旋锁

**自旋锁不适用的场景**：
- **临界区很长**（如文件 I/O、网络请求）：自旋浪费太多 CPU
- **单处理器系统**：自旋的线程占用了唯一 CPU，持有锁的线程无法运行——导致死锁
- **用户态程序**：应该使用互斥锁（futex），让等待线程睡眠

### 13.5.3 自旋锁优化

**Test-and-Test-and-Set（TTAS）**：先用普通读取检查锁状态，只有锁可能空闲时才用 TSL。

```c
void spinlock_acquire_ttas(spinlock_t *lock) {
    while (1) {
        // 第一步：普通读取（不修改缓存行）
        while (lock->locked == 1)
            ;  // 自旋在本地缓存上，不产生总线流量
        
        // 第二步：锁可能空闲，尝试获取
        if (test_and_set(&lock->locked) == 0)
            break;  // 获取成功
        // 如果获取失败（其他线程先抢到了），回到第一步
    }
}
```

TTAS 的优势：普通读取不会使其他 CPU 的缓存行失效，减少了缓存一致性流量。只有在锁可能空闲时才执行 TSL（会产生总线流量）。

**指数退避（Exponential Backoff）**：自旋等待时逐渐增加等待时间。

```c
void spinlock_acquire_backoff(spinlock_t *lock) {
    int backoff = 1;  // 初始退避时间
    while (test_and_set(&lock->locked) == 1) {
        delay(backoff);  // 等待一段时间
        backoff *= 2;    // 退避时间翻倍
        if (backoff > MAX_BACKOFF)
            backoff = MAX_BACKOFF;
    }
}
```

退避减少了同时竞争锁的线程数——等待时间越长的线程尝试频率越低，减少了总线争用。

**队列自旋锁（MCS Lock）**：每个等待线程自旋在自己的本地变量上，而不是共享的锁变量。

```c
// MCS Lock 节点
typedef struct mcs_node {
    struct mcs_node *next;  // 指向下一个等待者
    int locked;             // 1 = 等待, 0 = 获得锁
} mcs_node_t;

// 获取 MCS Lock
void mcs_lock(mcs_node_t **tail, mcs_node_t *node) {
    node->next = NULL;
    mcs_node_t *prev = fetch_and_store(tail, node);  // 原子地将自己加入队尾
    
    if (prev != NULL) {
        // 队列不为空，需要等待前驱释放锁
        node->locked = 1;           // 标记自己在等待
        prev->next = node;          // 前驱指向自己
        while (node->locked == 1)
            ;  // 在本地变量上自旋！不产生总线流量
    }
    // 如果 prev == NULL，说明队列之前为空，直接获得锁
}

// 释放 MCS Lock
void mcs_unlock(mcs_node_t **tail, mcs_node_t *node) {
    if (node->next == NULL) {
        // 没有等待者
        if (compare_and_swap(tail, node, NULL) == node)
            return;  // 队列已空
        // 有新等待者正在加入，等待它完成
        while (node->next == NULL) ;
    }
    node->next->locked = 0;  // 唤醒下一个等待者
}
```

MCS Lock 的优势：每个等待线程自旋在自己的 `node->locked` 上，不会产生缓存一致性流量。这是 Linux 内核中使用的队列自旋锁的基础。

---

## 13.5.4 自旋锁性能分析

自旋锁的性能取决于多个因素：临界区长度、竞争程度、处理器数量。让我们通过具体数据来分析。

**临界区长度的影响**：

```
临界区长度    自旋开销（2 线程）    互斥锁开销    推荐选择
10 ns         ~20 ns               ~1000 ns      自旋锁
100 ns        ~200 ns              ~1000 ns      自旋锁
1 μs          ~2 μs                ~1000 ns      取决于场景
10 μs         ~20 μs               ~1000 ns      互斥锁
100 μs        ~200 μs              ~1000 ns      互斥锁
1 ms          ~2 ms                ~1000 ns      互斥锁
```

当临界区长度 < 1μs 时，自旋锁的总开销（自旋时间 + 获取时间）通常小于互斥锁（上下文切换时间）。当临界区长度 > 10μs 时，互斥锁更优——等待线程睡眠不浪费 CPU。

**竞争程度的影响**：

```
线程数    自旋锁吞吐量    互斥锁吞吐量
2         高              中
4         中              中
8         低              中
16        很低            中
```

自旋锁的吞吐量随线程数增加而急剧下降——因为所有等待线程都在竞争同一条缓存行。互斥锁的吞吐量相对稳定——因为等待线程睡眠，不产生缓存一致性流量。

---

## 13.5.5 xv6 自旋锁实现

xv6 的自旋锁实现非常简洁，但包含了所有关键细节。让我们逐行分析：

```c
// kernel/spinlock.h
struct spinlock {
    uint locked;       // 锁状态：0=未锁定, 1=已锁定
    struct cpu *cpu;   // 持有锁的 CPU（调试用）
    char *name;        // 锁名称（调试用）
};
```

xv6 的自旋锁结构只有三个字段。`cpu` 和 `name` 字段纯粹用于调试——在 panic 时显示"哪个 CPU 持有哪把锁"，帮助开发者定位死锁。

```c
// kernel/spinlock.c
void acquire(struct spinlock *lk) {
    push_off();  // 禁用中断
    
    if (holding(lk))
        panic("acquire");  // 防止重复获取

    while (__sync_lock_test_and_set(&lk->locked, 1) != 0)
        ;  // 自旋等待

    __sync_synchronize();  // 内存屏障
    lk->cpu = mycpu();
}
```

**`push_off()` 的作用**：禁用当前 CPU 的中断。这是为了防止死锁——如果持有自旋锁时发生定时器中断，调度器可能切换到另一个进程，而那个进程可能尝试获取同一把锁，导致死锁。

**`__sync_lock_test_and_set`**：GCC 内置的 TSL 原子操作。它原子地将 `lk->locked` 设为 1，并返回旧值。

**`__sync_synchronize`**：全内存屏障。确保临界区内的读写操作在获取锁之后执行——防止处理器重排指令。

```c
void release(struct spinlock *lk) {
    if (!holding(lk))
        panic("release");

    lk->cpu = 0;
    __sync_synchronize();
    __sync_lock_release(&lk->locked);
    pop_off();  // 恢复中断
}
```

**`__sync_lock_release`**：GCC 内置的原子释放操作。它原子地将 `lk->locked` 设为 0。

**`pop_off()` 的作用**：恢复中断状态。与 `push_off()` 配对使用。

---

## 13.5.6 手算练习：自旋锁等待时间

**题目**：4 个线程同时竞争一把自旋锁。临界区长度为 5μs，上下文切换时间为 2μs。使用普通自旋锁和 MCS Lock，分别计算平均等待时间。

**解答**：

普通自旋锁：
- 线程 1 获取锁，执行 5μs
- 线程 2,3,4 自旋等待 5μs
- 线程 1 释放锁，线程 2,3,4 同时竞争
- 线程 2 获取锁，执行 5μs
- 线程 3,4 自旋等待 5μs
- 平均等待时间 = (0 + 5 + 10 + 15) / 4 = 7.5μs

MCS Lock（FIFO 顺序）：
- 线程 1 获取锁，执行 5μs
- 线程 2 在自己的 node->locked 上自旋 5μs
- 线程 3 在自己的 node->locked 上自旋 10μs
- 线程 4 在自己的 node->locked 上自旋 15μs
- 平均等待时间 = (0 + 5 + 10 + 15) / 4 = 7.5μs

两种锁的平均等待时间相同，但 MCS Lock 的优势在于：
1. 每个线程自旋在自己的变量上，不产生缓存一致性流量
2. 严格 FIFO 顺序，无饥饿

---

## 13.6 面试高频考点

**Q1：什么是竞态条件？**

多个线程并发访问共享数据，程序的结果取决于线程执行的相对顺序。典型例子：两个线程同时执行 `counter++`，结果可能少于预期。竞态条件的根本原因是操作的非原子性——`counter++` 由三条指令组成，线程可能在任意两条指令之间被切换。

**Q2：临界区问题的三个要求是什么？**

互斥（同一时刻最多一个线程在临界区）、进步（临界区空闲时等待线程能进入）、有限等待（每个线程等待时间有上界）。三个要求缺一不可——缺少互斥导致竞态条件，缺少进步导致死锁，缺少有限等待导致饥饿。

**Q3：Peterson 算法的原理？**

使用两个共享变量 `flag[2]` 和 `turn`。`flag[i]=1` 表示进程 i 想进入，`turn` 记录轮到谁。通过"让对方先"的策略实现互斥和进步。关键在于 `turn` 变量打破了两个进程同时谦让的僵局。

**Q4：TSL 和 CAS 的区别？**

TSL 无条件地读取旧值并写入新值（总是写入 1）。CAS 只在当前值等于期望值时才写入新值（条件更新）。CAS 更强大，是无锁数据结构的基础——它允许"乐观并发"：先尝试修改，如果失败则重试。

**Q5：自旋锁的适用场景？**

临界区很短（几条指令）、多处理器系统（自旋在其他 CPU 上）、内核代码（不能睡眠）。不适用于临界区长、单处理器系统、用户态程序。

**Q6：LL/SC 相比 CAS 的优势？**

LL/SC 更灵活（可实现任意原子操作）、避免 ABA 问题（任何修改都会清除 reservation）、硬件实现更简单（不需要比较逻辑）。CAS 需要比较逻辑，LL/SC 只需要 reservation 机制。

**Q7：什么是 ABA 问题？如何解决？**

ABA 问题是 CAS 的一个陷阱：线程 1 读取值 A，线程 2 将值改为 B 再改回 A，线程 1 的 CAS 成功（因为值仍然是 A），但语义上值已经被修改过了。

```
时刻  线程 1                    线程 2                    值
──────────────────────────────────────────────────────────────
t1    读取 A
t2                               修改为 B
t3                               修改回 A
t4    CAS(A, A+1) 成功！         ← 语义错误！值被修改过
```

解决方案：
1. **版本号**：在值旁边附加一个版本号，CAS 比较值+版本号
2. **LL/SC**：任何修改都会清除 reservation，不会出现 ABA 问题
3. **Hazard Pointer**：标记正在访问的指针，防止被回收

**Q8：如何选择合适的同步原语？**

```
场景                          推荐同步原语
────────────────────────────────────────────────────────
保护简单计数器                 原子操作（fetch_and_add）
保护短临界区（< 1μs）          自旋锁
保护长临界区（> 10μs）         互斥锁（futex）
读多写少                       读写锁或 RCU
条件同步（等待特定条件）        条件变量
资源计数（N 个资源）            信号量
无锁数据结构                   CAS 循环
```

---

## 13.8 常见并发 Bug 与调试

### 13.8.1 数据竞争（Data Race）

数据竞争是并发编程中最常见的 Bug：两个线程同时访问同一内存位置，至少一个是写操作，且没有同步保护。

```c
// 有数据竞争的代码
int counter = 0;

void *thread_func(void *arg) {
    for (int i = 0; i < 1000000; i++) {
        counter++;  // 数据竞争！
    }
    return NULL;
}

// 使用 ThreadSanitizer 检测数据竞争
// 编译：gcc -fsanitize=thread -g program.c -lpthread
// 运行：./a.out
// 输出：WARNING: ThreadSanitizer: data race on counter
```

### 13.8.2 死锁（Deadlock）

死锁是多个线程互相等待对方持有的资源，导致所有线程都无法继续执行。

```c
// 死锁示例：两个线程以不同顺序获取两把锁
pthread_mutex_t lock1, lock2;

void *thread1(void *arg) {
    pthread_mutex_lock(&lock1);    // 获取 lock1
    sleep(1);                       // 等待
    pthread_mutex_lock(&lock2);    // 尝试获取 lock2 → 死锁！
    // ...
}

void *thread2(void *arg) {
    pthread_mutex_lock(&lock2);    // 获取 lock2
    sleep(1);                       // 等待
    pthread_mutex_lock(&lock1);    // 尝试获取 lock1 → 死锁！
    // ...
}
```

**死锁的四个必要条件**（Coffman 条件）：
1. **互斥**：资源不能被共享
2. **持有并等待**：线程持有资源的同时等待其他资源
3. **非抢占**：资源不能被强制收回
4. **循环等待**：存在线程的循环等待链

**预防死锁的方法**：
- **锁排序**：所有线程以相同顺序获取锁（破坏循环等待）
- **超时**：获取锁时设置超时，超时后释放已持有的锁（破坏持有并等待）
- **一次性获取**：一次性获取所有需要的资源（破坏持有并等待）

### 13.8.3 活锁（Livelock）

活锁是多个线程不断改变状态以响应对方，但没有任何线程取得进展。

```c
// 活锁示例：两个线程互相谦让
void *thread1(void *arg) {
    while (1) {
        if (flag2) {
            // 对方想进入，我谦让
            flag1 = 0;
            sleep(1);
            flag1 = 1;
        } else {
            // 对方不想进入，我进入临界区
            break;
        }
    }
}

void *thread2(void *arg) {
    while (1) {
        if (flag1) {
            // 对方想进入，我谦让
            flag2 = 0;
            sleep(1);
            flag2 = 1;
        } else {
            // 对方不想进入，我进入临界区
            break;
        }
    }
}
// 两个线程不断谦让，谁也无法进入临界区
```

---

## 13.9 扩展阅读

- **OSTEP** Chapter 28: "Locks" — 锁的实现，自旋锁详解
- **OSTEP** Chapter 29: "Lock-based Concurrent Data Structures" — 基于锁的数据结构
- **Operating System Concepts** Chapter 5.5: "Critical-Section Problem" — 临界区问题
- [Linux spinlock implementation](https://elixir.bootlin.com/linux/latest/source/include/linux/spinlock.h) — Linux 自旋锁源码
- [Peterson's Algorithm](https://en.wikipedia.org/wiki/Peterson%27s_algorithm) — 维基百科详解
- [ABA Problem](https://en.wikipedia.org/wiki/ABA_problem) — ABA 问题详解

### 13.9.1 推荐实践

1. **编译时检测数据竞争**：使用 ThreadSanitizer（`-fsanitize=thread`）检测数据竞争
2. **运行时检测死锁**：使用 Helgrind（`valgrind --tool=helgrind`）检测死锁
3. **静态分析**：使用 Clang Thread Safety Analysis（`-Wthread-safety`）在编译时检测锁使用错误
4. **代码审查**：重点关注锁的获取顺序、临界区边界、异常处理路径
- [Peterson's Algorithm](https://en.wikipedia.org/wiki/Peterson%27s_algorithm) — 维基百科详解
