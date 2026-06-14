---
title: "Chapter 14: 锁与互斥"
description: "深入理解锁的抽象与实现，掌握自旋锁、睡眠锁、读写锁的设计与权衡，理解 futex 机制与锁的性能优化"
updated: "2026-06-10"
---

# Chapter 14: 锁与互斥

> **本章目标**：
> - 深入理解锁的抽象接口与评价标准
> - 掌握自旋锁、睡眠锁、Ticket Lock 的实现与适用场景
> - 理解 futex 的快速路径与慢速路径
> - 掌握读写锁的设计与读者/写者问题
> - 了解锁的性能优化技术与 xv6 锁实现

---

## 14.1 锁（Lock）抽象

### 14.1.1 锁的接口

锁是最基本的同步原语，用于保护共享资源不被多个线程同时访问。锁提供两个基本操作：

```c
lock_t mutex;          // 声明一个锁变量
lock_init(&mutex);     // 初始化锁（设置为"未锁定"状态）

lock(&mutex);          // 获取锁——如果锁已被其他线程占用，则等待
// === 临界区开始 ===
// 在这里安全地访问共享资源
// 其他线程无法进入这段代码（因为它们也在尝试获取同一把锁）
shared_data++;
// === 临界区结束 ===
unlock(&mutex);        // 释放锁——允许其他等待的线程获取锁
```

锁的语义非常精确：
- `lock()` 成功返回后，当前线程"持有"锁——它是唯一可以在临界区内执行的线程
- 同一时刻只有一个线程可以持有锁——其他线程调用 `lock()` 会阻塞等待
- `unlock()` 释放锁，允许其他等待的线程获取——通常会唤醒一个等待线程
- 持有锁的线程可以安全地访问受保护的共享资源——互斥性保证了安全

### 14.1.2 锁的评价标准

评价一个锁实现的好坏，需要从三个维度考虑：

**正确性**（最重要）：
- **互斥**：同一时刻最多一个线程持有锁——这是锁的基本保证
- **进步**：锁空闲时等待线程能获取——不能死锁
- **有限等待**：每个线程等待时间有上界——不能饥饿

**公平性**：所有等待线程最终都能获取锁，没有线程被无限期推迟。公平性好的锁保证 FIFO 顺序——先等待的线程先获取锁。

**性能**：
- **无竞争时的开销**：锁空闲时 `lock()`/`unlock()` 的耗时——应该尽可能小（理想情况 ~10ns）
- **有竞争时的开销**：多个线程同时竞争时的总耗时——不应该随线程数急剧增加
- **可扩展性**：线程数增加时性能的变化——好的锁应该在多核系统上扩展良好

---

## 14.2 简单锁实现

### 14.2.1 禁用中断实现（单处理器）

在单处理器系统中，最简单的锁实现是禁用中断：

```c
void lock(lock_t *lock) {
    disable_interrupts();  // 禁用所有中断（包括定时器中断）
}

void unlock(lock_t *lock) {
    enable_interrupts();   // 恢复中断
}
```

**原理**：线程切换是由定时器中断触发的。禁用中断后，当前线程不会被切换，直到它主动释放锁。这保证了临界区内的代码是"原子"执行的。

**优点**：极其简单（两条指令），开销极小（~1ns）。

**缺点**：
- 只适用于单处理器——禁用中断只影响当前 CPU
- 不适用于用户态——禁用中断是特权指令
- 长时间禁用危险——如果临界区有 Bug，系统会挂起
- 影响响应性——禁用期间无法响应 I/O、定时器等事件

### 14.2.2 自旋锁实现（多处理器）

```c
void lock(lock_t *lock) {
    while (test_and_set(&lock->locked) == 1)
        ;  // 忙等待：不断尝试获取锁
}

void unlock(lock_t *lock) {
    lock->locked = 0;  // 普通写入即可
}
```

**优点**：实现简单，无上下文切换开销，适合临界区很短的场景。

**缺点**：忙等待浪费 CPU——等待线程不断执行 while 循环，消耗 CPU 但不做有用的工作。在单处理器系统上会导致死锁（自旋线程占用了唯一 CPU，持有锁的线程无法运行）。

### 14.2.3 yield() 自旋锁

```c
void lock(lock_t *lock) {
    while (test_and_set(&lock->locked) == 1) {
        yield();  // 让出 CPU，切换到其他线程
    }
}

void unlock(lock_t *lock) {
    lock->locked = 0;
}
```

**改进**：`yield()` 让出 CPU 给其他线程，减少了 CPU 浪费。

**缺点**：上下文切换开销（~1-5μs）——每次 yield 都需要保存/恢复寄存器。如果锁持有时间很短，上下文切换的开销可能比自旋等待更大。

### 14.2.4 Ticket Lock（公平自旋锁）

普通自旋锁不保证公平性——某个线程可能连续多次获取锁，其他线程饥饿。Ticket Lock 通过"取号排队"保证 FIFO 公平性。

```c
typedef struct {
    int next_ticket;   // 下一个可分配的号码（由 fetch_and_add 原子递增）
    int now_serving;   // 当前服务的号码
} ticketlock_t;

void ticketlock_acquire(ticketlock_t *lock) {
    // 原子地获取一个号码
    int my_ticket = fetch_and_add(&lock->next_ticket, 1);
    // 等待轮到自己
    while (lock->now_serving != my_ticket)
        ;  // 自旋等待叫号
}

void ticketlock_release(ticketlock_t *lock) {
    lock->now_serving++;  // 叫下一个号
}
```

**工作原理**：
1. 每个想获取锁的线程通过 `fetch_and_add` 获取一个递增的号码
2. 线程自旋等待，直到 `now_serving` 等于自己的号码
3. 释放锁时，`now_serving++`，叫下一个号

**优势**：严格 FIFO 顺序——先到的线程先获取锁，无饥饿。

**缺点**：所有等待线程自旋在同一个变量（`now_serving`）上——每次 `now_serving++` 都会使所有等待线程的缓存行失效，产生大量缓存一致性流量。

---

## 14.3 睡眠锁（Sleeping Lock）

### 14.3.1 设计动机

自旋锁在等待时浪费 CPU。对于临界区较长的场景（如文件 I/O、网络请求、磁盘操作），等待线程应该**睡眠**——让出 CPU 给其他线程做有用的工作。

```
自旋锁（临界区长时浪费 CPU）：
线程 A: [========临界区（如磁盘读取）========]
线程 B: [自旋][自旋][自旋][自旋][自旋][自旋] → 浪费 CPU！

睡眠锁（让出 CPU）：
线程 A: [========临界区（如磁盘读取）========]
线程 B: [睡眠（让出 CPU）]                    → 不浪费 CPU
                                              [唤醒][临界区]
```

### 14.3.2 队列 + 睡眠/唤醒机制

```c
typedef struct {
    int locked;
    queue_t waiters;  // 等待队列——存储等待获取锁的线程
} sleeplock_t;

void sleeplock_acquire(sleeplock_t *lock) {
    while (lock->locked) {
        // 锁被占用——加入等待队列并睡眠
        enqueue(&lock->waiters, current_thread);
        sleep();  // 让出 CPU，当前线程进入睡眠状态
        // 被唤醒后回到 while 循环，重新检查锁状态
    }
    lock->locked = 1;  // 获取锁
}

void sleeplock_release(sleeplock_t *lock) {
    lock->locked = 0;  // 释放锁
    if (!empty(&lock->waiters)) {
        Thread *t = dequeue(&lock->waiters);
        wakeup(t);  // 唤醒一个等待线程
    }
}
```

**优点**：等待时不浪费 CPU——等待线程睡眠，让出 CPU 给其他线程。

**缺点**：上下文切换开销（~1-5μs）——每次睡眠/唤醒都需要保存/恢复寄存器。如果临界区很短（如几条指令），上下文切换的开销可能比自旋等待更大。

### 14.3.3 Linux mutex 实现

Linux 的 `mutex` 使用**乐观自旋（Optimistic Spinning）** 策略，结合了自旋锁和睡眠锁的优点。它的核心思想是：如果锁持有者正在 CPU 上运行，它很可能很快就会释放锁——此时自旋等待比睡眠更高效（避免了上下文切换的开销）。

```c
// 简化的 Linux mutex 实现
void mutex_lock(struct mutex *lock) {
    // 快速路径：无竞争——原子地尝试获取
    if (atomic_cmpxchg(&lock->count, 1, 0) == 1)
        return;  // 成功，直接返回（~10ns）
    
    // 中速路径：乐观自旋
    // 检查锁持有者是否正在运行——如果在运行，它可能很快释放
    if (lock->owner is running on another CPU) {
        while (lock->count != 1)
            cpu_relax();  // 自旋等待（pause 指令，降低功耗）
        if (atomic_cmpxchg(&lock->count, 1, 0) == 1)
            return;  // 成功获取
    }
    
    // 慢速路径：睡眠等待
    spin_lock(&lock->wait_lock);
    list_add(&waiter, &lock->wait_list);  // 加入等待队列
    set_current_state(TASK_INTERRUPTIBLE);
    spin_unlock(&lock->wait_lock);
    schedule();  // 让出 CPU，进入睡眠
    // 被唤醒后重新尝试获取锁
}
```

**三路径设计的优势**：
- 快速路径（~10ns）：无竞争时，一次 CAS 就获取锁
- 中速路径（~100ns）：锁持有者在运行时，自旋等待比睡眠更快
- 慢速路径（~1μs）：锁持有者不在运行时，睡眠等待不浪费 CPU

### 14.3.4 futex（Fast Userspace Mutex）

futex 是 Linux 提供的底层同步原语，是实现高性能互斥锁的基础。它的核心思想是：**无竞争时在用户态完成，有竞争时才陷入内核**。

```
futex 的两层结构：

用户态（快速路径）——不需要系统调用：
  int locked;  // 共享变量，0=未锁定, 1=已锁定

  lock():   
    if (atomic_cmpxchg(&locked, 0, 1) == 0) 
        return;  // 无竞争，直接获取（~10ns）
    futex_wait(&locked, 1);  // 有竞争，陷入内核

  unlock(): 
    locked = 0;              // 释放锁
    if (has_waiters) 
        futex_wake(&locked); // 唤醒等待者

内核态（慢速路径）——需要系统调用：
  futex_wait(&locked, 1):  // 如果 locked==1，将当前线程加入等待队列并睡眠
  futex_wake(&locked):     // 唤醒一个等待线程
```

**快速路径**（无竞争）：
```
线程 A 调用 lock()
  → atomic_cmpxchg(&locked, 0, 1) 成功
  → 返回，不需要系统调用
  → 耗时 ~10ns（与自旋锁相当）
```

**慢速路径**（有竞争）：
```
线程 B 调用 lock()
  → atomic_cmpxchg 失败（locked 已经是 1）
  → 调用 futex_wait(&locked, 1) 系统调用
  → 内核将 B 加入等待队列，B 睡眠
  → 耗时 ~1μs（系统调用 + 上下文切换）
```

futex 的优势：无竞争时性能接近自旋锁（~10ns），有竞争时性能接近睡眠锁（不浪费 CPU）。这是现代互斥锁实现的标准方式。

### 14.3.5 futex 的使用示例

让我们通过一个完整的示例来理解 futex 的使用：

```c
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

int locked = 0;  // 共享的锁变量

// futex_wait：如果 *addr == expected，则睡眠
int futex_wait(int *addr, int expected) {
    return syscall(SYS_futex, addr, FUTEX_WAIT, expected, NULL, NULL, 0);
}

// futex_wake：唤醒最多 val 个等待线程
int futex_wake(int *addr, int val) {
    return syscall(SYS_futex, addr, FUTEX_WAKE, val, NULL, NULL, 0);
}

// 基于 futex 的互斥锁
void lock() {
    // 快速路径：尝试获取锁
    if (__sync_lock_test_and_set(&locked, 1) == 0)
        return;  // 成功
    
    // 慢速路径：锁被占用，等待
    while (__sync_lock_test_and_set(&locked, 1) != 0) {
        futex_wait(&locked, 1);  // 如果 locked==1，则睡眠
    }
}

void unlock() {
    __sync_lock_release(&locked);  // 释放锁
    futex_wake(&locked, 1);        // 唤醒一个等待线程
}
```

### 14.3.6 futex 的性能数据

```
场景              延迟            说明
无竞争            ~10 ns          一次 CAS 操作
有竞争（2 线程）   ~100 ns         CAS 失败 + futex_wait 系统调用
有竞争（16 线程）  ~500 ns         多次 CAS 失败 + 唤醒竞争
高竞争（64 线程）  ~2 μs           严重的缓存一致性流量
```

<div data-component="FutexMechanism"></div>

---

## 14.4 xv6 自旋锁

### 14.4.1 struct spinlock 数据结构

```c
// kernel/spinlock.h
struct spinlock {
    uint locked;       // 锁状态：0=未锁定, 1=已锁定
    struct cpu *cpu;   // 持有锁的 CPU（用于调试——防止重复获取）
    char *name;        // 锁名称（用于调试——panic 时显示）
};
```

xv6 的自旋锁结构非常简洁——核心只有一个 `locked` 字段。`cpu` 和 `name` 字段纯粹用于调试，不影响锁的功能。

### 14.4.2 acquire() 与 release()

```c
// kernel/spinlock.c
void acquire(struct spinlock *lk) {
    push_off();  // 禁用中断（关键！）
    
    // 调试检查：防止重复获取同一把锁
    if (holding(lk))
        panic("acquire");  // 如果已经持有这把锁，说明有 Bug

    // 自旋等待——使用 __sync_lock_test_and_set（GCC 内置的 TSL）
    while (__sync_lock_test_and_set(&lk->locked, 1) != 0)
        ;  // 忙等待

    // 内存屏障——确保后续的读写操作在获取锁之后执行
    __sync_synchronize();

    lk->cpu = mycpu();  // 记录持有锁的 CPU（用于调试）
}

void release(struct spinlock *lk) {
    // 调试检查：确保释放的是自己持有的锁
    if (!holding(lk))
        panic("release");

    lk->cpu = 0;  // 清除持有者记录
    
    // 内存屏障——确保临界区的操作在释放锁之前完成
    __sync_synchronize();

    // 释放锁——使用 __sync_lock_release（GCC 内置）
    __sync_lock_release(&lk->locked);

    pop_off();  // 恢复中断状态
}
```

### 14.4.3 禁用中断防止死锁

xv6 的自旋锁在获取时**禁用中断**。这是为了防止以下死锁场景：

```
场景（不禁用中断）：
  CPU 0 持有锁 A
  定时器中断触发 → 调度器选择新进程 B
  进程 B 尝试获取锁 A → 自旋等待
  但 CPU 0 上的进程 A 需要运行才能释放锁 A
  而 CPU 0 现在在运行进程 B（调度器选择了 B）
  → 死锁！进程 B 等待锁 A，但锁 A 的持有者（进程 A）无法运行

解决方案：
  acquire() 调用 push_off() 禁用中断
  → 定时器中断不会触发
  → 调度器不会切换进程
  → 持有锁的进程会一直运行直到释放锁
```

### 14.4.4 xv6 锁的使用规则

xv6 内核中有一套严格的锁使用规则，违反这些规则会导致死锁或数据损坏：

**规则 1：持有自旋锁时不能睡眠**
```c
// 错误：持有自旋锁时调用 sleep()
acquire(&lock);
sleep(channel, &lock);  // 死锁！sleep 会释放 lock，但其他代码可能依赖 lock
```

**规则 2：持有自旋锁时不能调用可能睡眠的函数**
```c
// 错误：持有自旋锁时调用 kalloc()（可能睡眠）
acquire(&lock);
char *page = kalloc();  // 如果内存不足，kalloc 可能睡眠
```

**规则 3：获取多把锁时必须按固定顺序**
```c
// 正确：所有代码都按 lock1 → lock2 的顺序获取
acquire(&lock1);
acquire(&lock2);
// ...

// 错误：不同代码以不同顺序获取锁
// 线程 1: acquire(&lock1); acquire(&lock2);
// 线程 2: acquire(&lock2); acquire(&lock1);  // 死锁！
```

**规则 4：中断处理程序中不能获取可能被进程持有的锁**
```c
// 错误：中断处理程序获取进程可能持有的锁
void timer_interrupt() {
    acquire(&process_lock);  // 如果进程持有此锁，死锁！
}
```

### 14.4.5 Linux 自旋锁变体

Linux 内核提供了多种自旋锁变体，适用于不同的场景：

```c
// 普通自旋锁
spin_lock(&lock);

// 禁用中断的自旋锁（用于中断处理程序可能访问的数据）
spin_lock_irqsave(&lock, flags);

// 禁用下半部的自旋锁（用于软中断可能访问的数据）
spin_lock_bh(&lock);

// 读写自旋锁
read_lock(&lock);    // 读者获取
write_lock(&lock);   // 写者获取
```

---

## 14.5 读写锁（Reader-Writer Lock）

### 14.5.1 设计动机

很多数据结构是**读多写少**的——大部分操作是读取（如查询、遍历），偶尔有写入（如更新、插入）。普通互斥锁不允许并发读取——即使多个线程只是读取数据，也必须串行执行，浪费了并行性。

**读写锁**允许：
- **多个读者同时读取**——因为读取不会修改数据，可以安全并发
- **写者独占访问**——因为写入可能影响其他读取者的视图
- **读者和写者不能同时访问**——防止读取到写入了一半的数据

```
普通互斥锁：
读者1: [读取]      读者2: [等待]  读者3: [等待]  写者: [等待]  读者4: [等待]
                        [读取]                 [等待]          [等待]
                                                  [读取]        [等待]
                                                                    [写入]

读写锁：
读者1: [读取]  读者2: [读取]  读者3: [读取]  写者: [等待]  读者4: [等待]
                                    ↑ 读者并发！        [写入]
                                                                    [读取]
```

### 14.5.2 读者优先读写锁

```c
typedef struct {
    int readers;              // 当前正在读取的读者数量
    pthread_mutex_t mutex;    // 保护 readers 变量的互斥锁
    pthread_mutex_t write_lock; // 写锁——写者独占
} rwlock_t;

// 读者获取锁
void reader_lock(rwlock_t *lock) {
    pthread_mutex_lock(&lock->mutex);
    lock->readers++;
    if (lock->readers == 1) {
        // 第一个读者到达——锁住写锁，阻止写者
        pthread_mutex_lock(&lock->write_lock);
    }
    pthread_mutex_unlock(&lock->mutex);
    // 读者可以并发读取——不需要持有任何锁
}

// 读者释放锁
void reader_unlock(rwlock_t *lock) {
    pthread_mutex_lock(&lock->mutex);
    lock->readers--;
    if (lock->readers == 0) {
        // 最后一个读者离开——释放写锁，允许写者
        pthread_mutex_unlock(&lock->write_lock);
    }
    pthread_mutex_unlock(&lock->mutex);
}

// 写者获取锁
void writer_lock(rwlock_t *lock) {
    pthread_mutex_lock(&lock->write_lock);
    // 写者独占——没有读者或其他写者
}

// 写者释放锁
void writer_unlock(rwlock_t *lock) {
    pthread_mutex_unlock(&lock->write_lock);
}
```

**问题**：写者可能饥饿——如果有持续不断的读者到达，`readers` 永远不为 0，写锁永远无法获取。

### 14.5.3 写者优先读写锁

写者优先的读写锁在写者到达后阻止新读者加入，避免写者饥饿。

```c
typedef struct {
    int readers;
    int writers_waiting;         // 等待的写者数量
    pthread_mutex_t mutex;
    pthread_cond_t readers_done; // 读者完成的条件变量
    pthread_cond_t writers_done; // 写者完成的条件变量
} rwlock_writer_priority_t;

void reader_lock(rwlock_writer_priority_t *lock) {
    pthread_mutex_lock(&lock->mutex);
    // 如果有写者在等待，新读者必须等待
    while (lock->writers_waiting > 0) {
        pthread_cond_wait(&lock->writers_done, &lock->mutex);
    }
    lock->readers++;
    pthread_mutex_unlock(&lock->mutex);
}

void reader_unlock(rwlock_writer_priority_t *lock) {
    pthread_mutex_lock(&lock->mutex);
    lock->readers--;
    if (lock->readers == 0) {
        pthread_cond_broadcast(&lock->readers_done);  // 通知写者
    }
    pthread_mutex_unlock(&lock->mutex);
}

void writer_lock(rwlock_writer_priority_t *lock) {
    pthread_mutex_lock(&lock->mutex);
    lock->writers_waiting++;
    while (lock->readers > 0) {
        pthread_cond_wait(&lock->readers_done, &lock->mutex);
    }
    lock->writers_waiting--;
    pthread_mutex_unlock(&lock->mutex);
}

void writer_unlock(rwlock_writer_priority_t *lock) {
    pthread_mutex_lock(&lock->mutex);
    pthread_cond_broadcast(&lock->writers_done);  // 通知等待的读者
    pthread_mutex_unlock(&lock->mutex);
}
```

### 14.5.3 写者优先读写锁

写者优先的读写锁在写者到达后阻止新读者加入，避免写者饥饿。

### 14.5.4 Linux RCU（Read-Copy-Update）

RCU 是 Linux 内核中用于读多写少场景的高性能同步机制。它的核心思想是：**读端完全无锁，写端使用"复制-修改-替换"策略**。

```c
// 读端（完全没有锁！可以并发读取）
rcu_read_lock();                     // 标记进入 RCU 读侧临界区
struct data *p = rcu_dereference(global_ptr);  // 读取数据指针
// 使用 p——不需要任何锁
rcu_read_unlock();                   // 标记离开 RCU 读侧临界区

// 写端（需要互斥锁）
mutex_lock(&write_lock);
struct data *new = kmalloc(sizeof(*new));  // 分配新内存
*new = *old;                               // 复制旧数据
new->field = new_value;                    // 修改新数据
rcu_assign_pointer(global_ptr, new);       // 原子替换指针
mutex_unlock(&write_lock);
synchronize_rcu();                         // 等待所有读者完成
kfree(old);                                // 释放旧数据
```

**RCU 的工作原理**：
1. 读端通过 `rcu_read_lock()`/`rcu_read_unlock()` 标记临界区
2. 写端不直接修改数据，而是创建副本，修改后原子替换指针
3. 旧数据在所有读者完成后（`synchronize_rcu()` 返回）才释放
4. 这保证了读者永远不会看到写入了一半的数据

RCU 的优势：读端完全没有锁开销（~0ns），适合读多写少的场景（如路由表、进程列表、内核配置）。

**RCU 的核心概念——宽限期（Grace Period）**：

宽限期是指从所有读者都离开 RCU 读侧临界区的时间段。`synchronize_rcu()` 会阻塞直到一个宽限期结束。

```
时间线：
读者 1: [rcu_read_lock() ... rcu_read_unlock()]
读者 2:    [rcu_read_lock() ... rcu_read_unlock()]
读者 3:       [rcu_read_lock() ... rcu_read_unlock()]
写者:   [替换指针] [synchronize_rcu() 阻塞...] [kfree(old)]
                                ↑
                          宽限期：所有读者都完成
```

**RCU 的实现原理**（简化版）：

```c
// 读端
void rcu_read_lock(void) {
    preempt_disable();  // 禁用抢占——当前线程不会被切换
}

void rcu_read_unlock(void) {
    preempt_enable();   // 恢复抢占
}

// 写端
void synchronize_rcu(void) {
    // 等待所有 CPU 都经历一次上下文切换
    // 这保证了所有在 synchronize_rcu 之前进入 RCU 读临界区的读者都已完成
    for_each_online_cpu(cpu) {
        wait_for_context_switch(cpu);
    }
}
```

**RCU 的使用场景**：

```
场景                      推荐机制        原因
路由表（读多写少）         RCU             读端无锁，写端偶尔更新
进程列表（读多写少）       RCU             读端频繁（ps, top），写端偶尔
内核配置（几乎不写）       RCU             读端极频繁，写端极罕见
哈希表（读写均衡）         读写锁          RCU 写端开销太大
链表（频繁插入删除）       互斥锁          RCU 不适合频繁写入
```

### 14.5.5 Linux seqlock（顺序锁）

seqlock 是 Linux 内核中另一种读写同步机制，允许读者在不获取锁的情况下读取数据，但需要检测写入冲突。

```c
// kernel/include/linux/seqlock.h
typedef struct {
    unsigned sequence;  // 序列号——每次写入时递增
    spinlock_t lock;    // 写锁
} seqlock_t;

// 读者
unsigned int read_seqbegin(const seqlock_t *sl) {
    unsigned int ret;
    ret = sl->sequence;
    smp_rmb();  // 读内存屏障——确保读取 sequence 在读取数据之前
    return ret;
}

int read_seqretry(const seqlock_t *sl, unsigned int start) {
    smp_rmb();  // 读内存屏障
    return sl->sequence != start;  // 如果序列号变了，说明有写入
}

// 使用示例
unsigned int seq;
do {
    seq = read_seqbegin(&seqlock);
    // 读取数据（无锁！）
    value1 = shared_data.field1;
    value2 = shared_data.field2;
} while (read_seqretry(&seqlock, seq));
// 如果期间有写入，重试
```

**seqlock vs 读写锁**：

```
机制        读者延迟    写者延迟    读者并发    适用场景
读写锁      ~50 ns     ~100 ns    是          读多写少
seqlock     ~1 ns      ~100 ns    是          读极多写极少
互斥锁      ~100 ns    ~100 ns    否          通用
```

seqlock 的读者完全无锁（~1ns），但可能需要多次重试。如果写入频繁，读者会频繁重试，性能反而更差。

<div data-component="ReaderWriterVisualizer"></div>

---

## 14.5.5 读写锁的性能分析

读写锁的性能取决于读写比例。让我们通过具体数据来分析：

```
读写比例        互斥锁吞吐量    读写锁吞吐量    性能提升
99% 读 1% 写    1x              50x             50 倍
90% 读 10% 写   1x              10x             10 倍
50% 读 50% 写   1x              1x              无提升
10% 读 90% 写   1x              0.9x            略有下降
```

**关键洞察**：读写锁只有在读多写少的场景下才有显著优势。如果读写比例接近，读写锁的额外开销（维护读者计数、检查写锁）可能导致性能略有下降。

**RCU 的性能优势**：

```
同步机制        读端延迟        写端延迟        适用场景
互斥锁          ~100 ns        ~100 ns        通用
读写锁          ~50 ns         ~100 ns        读多写少
RCU             ~1 ns          ~1000 ns       极端读多写少
```

RCU 的读端几乎没有开销（只需要禁用抢占），但写端需要等待所有读者完成（`synchronize_rcu()`），延迟较高。RCU 适合读多写少的场景，如内核路由表、进程列表。

---

## 14.6 锁的性能优化

### 14.6.1 细粒度锁（Fine-Grained Locking）

使用多个锁保护不同的数据结构，减少锁竞争。

```c
// 粗粒度锁（整个哈希表一把锁）——所有操作串行
pthread_mutex_t table_lock;
hash_insert(&table_lock, key, value);  // 所有线程竞争同一把锁

// 细粒度锁（每个桶一把锁）——不同桶的操作可以并行
pthread_mutex_t bucket_locks[N];
hash_insert(&bucket_locks[hash(key) % N], key, value);  // 只竞争同一个桶的锁
```

**性能对比**（16 线程，100 万个操作）：

```
锁策略          吞吐量          锁竞争次数
粗粒度锁        1x              100 万次
细粒度锁（16 锁）10x             6.25 万次
细粒度锁（256 锁）50x            3906 次
```

### 14.6.2 锁分拆（Lock Splitting）

将一个大锁拆分为多个小锁，每个保护一部分数据。

```c
// 锁分拆示例：将一个大链表拆分为多个小链表
// 分拆前：一把锁保护整个链表
pthread_mutex_t list_lock;
Node *head;

// 分拆后：多把锁保护不同的子链表
pthread_mutex_t locks[NUM_SEGMENTS];
Node *heads[NUM_SEGMENTS];

void insert(int key, int value) {
    int segment = key % NUM_SEGMENTS;
    pthread_mutex_lock(&locks[segment]);
    // 只锁住对应的子链表
    insert_into_segment(&heads[segment], key, value);
    pthread_mutex_unlock(&locks[segment]);
}
```

### 14.6.3 顺序锁（Seqlock）

顺序锁允许读者在不获取锁的情况下读取数据，通过序列号检测写入冲突。

```c
// 顺序锁结构
typedef struct {
    unsigned sequence;  // 序列号——每次写入时递增
    pthread_mutex_t write_lock;  // 写锁
} seqlock_t;

// 读者（无锁！）
unsigned seq;
do {
    seq = read_seqbegin(&lock);  // 读取序列号
    // 读取数据（无锁——可能读到不一致的数据）
    value1 = shared_data.field1;
    value2 = shared_data.field2;
} while (read_seqretry(&lock, seq));  // 如果期间有写入，重试

// 写者
write_seqlock(&lock);  // 获取写锁，序列号递增
// 修改数据
shared_data.field1 = new_value1;
shared_data.field2 = new_value2;
write_sequnlock(&lock);  // 释放写锁，序列号递增
```

**顺序锁的优势**：读者完全无锁，适合读多写少的场景。**劣势**：读者可能需要多次重试，写入频繁时性能差。

### 14.6.4 无锁数据结构（Lock-Free）

使用 CAS 等原子操作实现无需锁的数据结构。适合高并发场景。

```c
// 无锁栈（Treiber Stack）
typedef struct node {
    int value;
    struct node *next;
} Node;

Node *top = NULL;  // 栈顶

void push(int value) {
    Node *new_node = malloc(sizeof(Node));
    new_node->value = value;
    do {
        new_node->next = top;  // 读取当前栈顶
    } while (cas(&top, new_node->next, new_node) != new_node->next);
    // CAS 循环：如果栈顶未被修改，则将新节点设为栈顶
}

int pop() {
    Node *old_top;
    do {
        old_top = top;  // 读取当前栈顶
        if (old_top == NULL) return -1;  // 栈空
    } while (cas(&top, old_top, old_top->next) != old_top);
    // CAS 循环：如果栈顶未被修改，则将下一个节点设为栈顶
    int value = old_top->value;
    free(old_top);
    return value;
}
```

---

## 14.7 手算练习：锁性能对比

**题目**：比较自旋锁、互斥锁、读写锁在不同场景下的性能。假设：
- 上下文切换时间 = 2μs
- 自旋锁获取时间 = 10ns
- 互斥锁获取时间 = 50ns（无竞争时）
- 读写锁获取时间 = 30ns（读者）、50ns（写者）
- 临界区长度 = 1μs
- 读写比例 = 90% 读 10% 写

**场景 1：2 线程，无竞争**

```
锁类型          获取时间    临界区时间    总时间
自旋锁          10 ns      1 μs         1.01 μs
互斥锁          50 ns      1 μs         1.05 μs
读写锁（读者）   30 ns      1 μs         1.03 μs
```

无竞争时，所有锁的性能相近。自旋锁略快。

**场景 2：4 线程，有竞争**

```
锁类型          平均等待时间    总时间
自旋锁          3 μs           4.01 μs
互斥锁          3 μs + 2 μs    6.05 μs
读写锁（读者）   0 μs（并发）    1.03 μs
```

有竞争时，读写锁在读多写少场景下性能最优。自旋锁和互斥锁的等待时间相近，但互斥锁多了上下文切换开销。

---

## 14.8 面试高频考点

**Q1：自旋锁和互斥锁的区别？**

自旋锁在等待时忙等（不释放 CPU），互斥锁在等待时睡眠（释放 CPU）。自旋锁适合临界区短、多处理器场景；互斥锁适合临界区长、单处理器场景。

**Q2：futex 的工作原理？**

futex 在用户态使用原子操作实现快速路径（无竞争时 ~10ns），只在有竞争时才陷入内核（慢速路径 ~1μs）。这结合了自旋锁的低开销和睡眠锁的低 CPU 浪费。

**Q3：读写锁的适用场景？**

读多写少的场景。读写锁允许多个读者并发，提高并行性。写者需要独占访问。

**Q4：Ticket Lock 和普通自旋锁的区别？**

Ticket Lock 通过"取号排队"保证 FIFO 公平性，无饥饿。普通自旋锁不保证公平性。

**Q5：RCU 和读写锁的区别？**

读写锁的读者需要获取锁（虽然可以并发），有锁开销。RCU 的读者完全无锁（~0ns），但写端需要等待所有读者完成（宽限期）。RCU 适合读极多写极少的场景，读写锁适合读多写少的场景。

**Q6：如何选择合适的锁？**

```
场景                          推荐锁
────────────────────────────────────────────────
临界区很短（< 1μs）           自旋锁
临界区很长（> 10μs）          互斥锁（futex）
读多写少                      读写锁
读极多写极少                  RCU 或 seqlock
需要公平性                    Ticket Lock
内核中断上下文                自旋锁（禁用中断）
用户态程序                    互斥锁（pthread_mutex）
```

---

## 14.7 扩展阅读

- **OSTEP** Chapter 28: "Locks" — 锁的实现
- **Operating System Concepts** Chapter 5.5: "Critical-Section Problem"
- [Linux futex](https://man7.org/linux/man-pages/man2/futex.2.html)
- [Linux RCU](https://www.kernel.org/doc/html/latest/RCU/)
- [xv6-riscv kernel/spinlock.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/spinlock.c)

### 14.7.1 推荐实践

1. **锁的粒度**：使用尽可能细粒度的锁——只保护需要保护的数据
2. **锁的顺序**：所有代码以相同顺序获取多把锁——避免死锁
3. **锁的持有时间**：尽量缩短临界区——减少锁竞争
4. **锁的选择**：根据场景选择合适的锁类型——不要用自旋锁保护长临界区
5. **锁的调试**：使用 lockdep（Linux 内核）或 ThreadSanitizer（用户态）检测锁错误

### 14.7.2 锁的常见错误

**错误 1：忘记释放锁**
```c
void buggy_function() {
    pthread_mutex_lock(&mutex);
    if (error_condition) {
        return;  // 错误！没有释放锁
    }
    // ...
    pthread_mutex_unlock(&mutex);
}

// 正确：使用 goto 或 RAII 模式
void correct_function() {
    pthread_mutex_lock(&mutex);
    if (error_condition) {
        goto out;
    }
    // ...
out:
    pthread_mutex_unlock(&mutex);
}
```

**错误 2：重复获取同一把锁**
```c
void buggy_function() {
    pthread_mutex_lock(&mutex);
    // ...
    helper_function();  // helper_function 也获取 mutex
    // ...
    pthread_mutex_unlock(&mutex);
}

void helper_function() {
    pthread_mutex_lock(&mutex);  // 死锁！mutex 已被持有
    // ...
    pthread_mutex_unlock(&mutex);
}
```

**错误 3：在信号处理程序中获取锁**
```c
pthread_mutex_t mutex;

void signal_handler(int sig) {
    pthread_mutex_lock(&mutex);  // 可能死锁！
    // 如果信号在 mutex 被持有时到达，会死锁
    pthread_mutex_unlock(&mutex);
}
```
