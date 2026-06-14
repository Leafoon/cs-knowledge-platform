---
title: "Chapter 15: 信号量与条件变量"
description: "深入理解 Dijkstra 信号量的设计与实现，掌握 POSIX 信号量 API，理解条件变量的 Mesa 语义与 Hoare 语义，掌握 xv6 sleep/wakeup 机制"
updated: "2026-06-10"
---

# Chapter 15: 信号量与条件变量

> **本章目标**：
> - 深入理解 Dijkstra 信号量的设计思想与实现
> - 掌握信号量的三种使用模式：互斥锁、条件同步、资源计数
> - 理解条件变量的 Mesa 语义与 Hoare 语义
> - 掌握 xv6 的 sleep/wakeup 机制与 Lost Wakeup 问题
> - 能够使用信号量和条件变量解决同步问题

---

## 15.1 信号量（Semaphore）

### 15.1.1 Dijkstra 的信号量设计

信号量由 Edsger Dijkstra 于 1965 年提出，是最经典的同步原语之一。Dijkstra 当时在设计 THE 操作系统时，需要一种优雅的方式来解决进程同步问题——他发明了信号量，这个概念至今仍是并发编程的基础。

信号量本质上是一个**整数变量**，支持两个**原子操作**：

- **P（wait/down/sem_wait）**：荷兰语"Proberen"（尝试）——如果信号量 > 0，减 1 并继续执行；否则等待（阻塞）
- **V（signal/up/sem_post）**：荷兰语"Verhogen"（增加）——信号量加 1，如果有等待的线程则唤醒一个

```c
// 信号量的定义
typedef struct {
    int value;       // 信号量的值——表示可用资源的数量
    queue_t waiters; // 等待队列——存储被阻塞的线程
    spinlock_t lock; // 保护 value 和 waiters 的自旋锁
} semaphore_t;

// P 操作（等待/获取资源）
void P(semaphore_t *s) {
    acquire(&s->lock);      // 获取内部锁
    s->value--;             // 信号量减 1
    if (s->value < 0) {
        // 信号量为负——没有可用资源，需要等待
        enqueue(&s->waiters, current_thread);  // 加入等待队列
        release(&s->lock);  // 释放内部锁
        sleep();            // 让出 CPU，进入睡眠
        // 被唤醒后从这里继续执行
    } else {
        release(&s->lock);  // 释放内部锁
    }
}

// V 操作（释放/通知）
void V(semaphore_t *s) {
    acquire(&s->lock);      // 获取内部锁
    s->value++;             // 信号量加 1
    if (s->value <= 0) {
        // 有等待的线程——唤醒一个
        Thread *t = dequeue(&s->waiters);
        wakeup(t);
    }
    release(&s->lock);      // 释放内部锁
}
```

### 15.1.2 信号量的语义

信号量的值有不同的含义，理解这些含义对于正确使用信号量至关重要：

**值 > 0**：表示可用资源的数量。例如，值=3 表示有 3 个资源可用。P 操作获取一个资源（值减 1），V 操作释放一个资源（值加 1）。

**值 = 0**：表示没有可用资源，也没有等待线程。P 操作会阻塞（值变为 -1，线程加入等待队列）。

**值 < 0**：绝对值表示等待线程的数量。例如，值=-3 表示有 3 个线程在等待。V 操作会唤醒一个等待线程（值从 -3 变为 -2）。

```
信号量值的变化示例（初始值=2，表示 2 个资源可用）：

操作        值    含义
初始        2     2 个资源可用
P()         1     获取了 1 个资源，还剩 1 个
P()         0     获取了 1 个资源，没有剩余
P()        -1     资源耗尽，1 个线程等待
P()        -2     资源耗尽，2 个线程等待
V()        -1     释放了 1 个资源，唤醒 1 个等待线程
V()         0     释放了 1 个资源，唤醒 1 个等待线程
V()         1     释放了 1 个资源，还有 1 个可用
```

### 15.1.3 二值信号量 vs 计数信号量

**二值信号量（Binary Semaphore）**：初始值为 1，只有两个状态（0 和 1）。当初始值为 1 时，信号量的行为与互斥锁完全相同——P 操作获取锁，V 操作释放锁。

```c
semaphore_t mutex;
sem_init(&mutex, 1);  // 初始值 = 1

// 使用二值信号量作为互斥锁
P(&mutex);      // 获取锁（值从 1 变为 0）
// 临界区——只有一个线程能在这里
V(&mutex);      // 释放锁（值从 0 变为 1）
```

**计数信号量（Counting Semaphore）**：初始值为 N，表示有 N 个可用资源。允许多个线程同时访问资源。

```c
semaphore_t pool;
sem_init(&pool, 5);  // 初始值 = 5，表示有 5 个资源

// 使用计数信号量管理资源池
P(&pool);      // 获取一个资源（如果池空则等待）
// 使用资源——最多 5 个线程可以同时使用
V(&pool);      // 归还资源（唤醒等待的线程）
```

### 15.1.4 信号量 vs 互斥锁

信号量和互斥锁看起来相似，但有重要的区别：

| 维度 | 信号量 | 互斥锁 |
|------|--------|--------|
| **初始值** | 可以 > 1（计数信号量） | 只能 = 1 |
| **V()/unlock() 由谁调用** | 任何线程都可以调用 V() | 只能由持有锁的线程调用 unlock() |
| **语义** | 资源计数——"有多少个资源可用" | 所有权——"谁持有锁" |
| **用途** | 互斥 + 条件同步 + 资源计数 | 仅互斥 |
| **是否有"持有者"概念** | 没有——任何线程都可以 V() | 有——只有持有者可以 unlock() |

关键区别：信号量没有"所有权"的概念——任何线程都可以调用 V()，而互斥锁的 unlock() 只能由持有锁的线程调用。这使得信号量更适合**条件同步**（一个线程通知另一个线程），而互斥锁更适合**互斥保护**。

---

## 15.2 信号量使用模式

### 15.2.1 互斥锁（初值 = 1）

```c
semaphore_t mutex;
sem_init(&mutex, 1);  // 初始值 = 1

void thread_func(void) {
    P(&mutex);      // 获取锁——值从 1 变为 0
    // 临界区：访问共享资源
    shared_data++;
    V(&mutex);      // 释放锁——值从 0 变为 1
}
```

### 15.2.2 条件同步（初值 = 0）

信号量可以用于**条件同步**：线程 A 等待线程 B 完成某个操作。初始值为 0 意味着"还没有信号"。

```c
semaphore_t done;
sem_init(&done, 0);  // 初始值 = 0——没有信号

// 线程 A：等待 B 完成
void thread_A(void) {
    // ... 做一些工作 ...
    P(&done);  // 等待 B 完成——如果 B 还没完成则阻塞
    // B 已经完成，继续执行
    printf("B is done!\n");
}

// 线程 B：完成工作后通知 A
void thread_B(void) {
    // ... 完成一些工作（可能需要很长时间）...
    V(&done);  // 通知 A——将 done 从 0 变为 1，唤醒 A
}
```

**为什么初始值为 0？** 因为"完成"这个条件初始时为假。如果线程 A 先执行 P(&done)，它会阻塞（值从 0 变为 -1）。当线程 B 执行 V(&done) 时，值从 -1 变为 0，唤醒线程 A。

### 15.2.3 资源计数（初值 = N）

```c
semaphore_t pool;
sem_init(&pool, N);  // 初始值 = N——N 个资源可用

void use_resource(void) {
    P(&pool);      // 获取一个资源——如果池空则等待
    // 使用资源——最多 N 个线程可以同时使用
    V(&pool);      // 归还资源——唤醒等待的线程
}
```

---

## 15.2.4 信号量实现细节

让我们深入理解信号量的实现细节。信号量的核心是维护一个整数值和一个等待队列：

```c
// 完整的信号量实现
typedef struct {
    int value;           // 信号量的值
    queue_t waiters;     // 等待队列
    pthread_mutex_t lock; // 保护 value 和 waiters 的互斥锁
} semaphore_t;

void sem_init(semaphore_t *s, int initial_value) {
    s->value = initial_value;
    queue_init(&s->waiters);
    pthread_mutex_init(&s->lock, NULL);
}

void sem_wait(semaphore_t *s) {
    pthread_mutex_lock(&s->lock);
    s->value--;
    if (s->value < 0) {
        // 信号量为负——需要等待
        enqueue(&s->waiters, current_thread);
        pthread_mutex_unlock(&s->lock);
        // 释放内部锁后睡眠——避免死锁
        thread_sleep();  // 让出 CPU
    } else {
        pthread_mutex_unlock(&s->lock);
    }
}

void sem_post(semaphore_t *s) {
    pthread_mutex_lock(&s->lock);
    s->value++;
    if (s->value <= 0) {
        // 有等待线程——唤醒一个
        Thread *t = dequeue(&s->waiters);
        thread_wakeup(t);
    }
    pthread_mutex_unlock(&s->lock);
}
```

**关键实现细节**：
1. **内部锁**：信号量的 `value` 和 `waiters` 需要互斥保护
2. **睡眠前释放锁**：`sem_wait` 在睡眠前释放内部锁，避免死锁
3. **唤醒后重新获取锁**：被唤醒的线程需要重新获取内部锁

### 15.2.5 信号量的性能分析

```
操作        无竞争延迟    有竞争延迟（4 线程）
sem_wait    ~50 ns       ~200 ns
sem_post    ~50 ns       ~100 ns
```

信号量的性能介于自旋锁和互斥锁之间——无竞争时接近自旋锁，有竞争时需要上下文切换。

---

## 15.3 条件变量（Condition Variable）

### 15.3.1 设计动机

信号量的问题是：P 操作只能等待信号量变为正数，无法等待**任意条件**。例如，"等待队列非空"、"等待缓冲区不满"、"等待计数器达到某个值"这样的条件，用信号量表达起来很困难。

条件变量允许线程等待**任意条件**为真。它是 Pthreads 标准中最重要的同步原语之一。

```c
// 条件变量接口
pthread_cond_wait(&cond, &mutex);      // 释放 mutex，等待 cond，被唤醒后重新获取 mutex
pthread_cond_signal(&cond);            // 唤醒一个等待线程
pthread_cond_broadcast(&cond);         // 唤醒所有等待线程
```

### 15.3.2 条件变量 + 锁的配合使用

条件变量**必须**与互斥锁配合使用。这是因为"检查条件"和"等待条件"之间需要原子性——如果在检查和等待之间条件变为真，唤醒就会丢失。

```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int ready = 0;  // 共享条件变量

// 线程 A：等待条件为真
void thread_A(void) {
    pthread_mutex_lock(&mutex);
    while (!ready) {  // 必须用 while 循环！不能用 if
        // ready 为假——需要等待
        pthread_cond_wait(&cond, &mutex);
        // 被唤醒后，重新检查条件（while 循环）
    }
    // ready 为真——继续执行
    printf("Ready!\n");
    pthread_mutex_unlock(&mutex);
}

// 线程 B：设置条件为真并通知
void thread_B(void) {
    pthread_mutex_lock(&mutex);
    ready = 1;  // 设置条件为真
    pthread_cond_signal(&cond);  // 通知等待的线程
    pthread_mutex_unlock(&mutex);
}
```

**`pthread_cond_wait` 的原子性**：这个函数在"释放 mutex"和"将线程加入等待队列"之间是原子的——不会被其他线程打断。这确保了 `signal` 不会在这两个操作之间执行，避免了 Lost Wakeup。

### 15.3.3 Mesa 语义 vs Hoare 语义

条件变量有两种语义，它们的区别在于 `signal()` 后被唤醒线程何时运行。

**Hoare 语义**（1974 年，由 C.A.R. Hoare 提出）：
- `signal()` 后，被唤醒的线程**立即**运行
- signal 线程暂停，等待被唤醒线程检查完条件后再继续
- 被唤醒线程**一定**能看到条件为真（因为 signal 线程还没来得及修改）

```
线程 A: while (!ready) wait();
线程 B: ready = 1; signal();

Hoare 语义执行顺序：
B: ready = 1 → signal → 切换到 A → A 检查 ready(真) → A 继续
→ A 被唤醒后一定能看到 ready=1（因为 B 还没机会修改它）
```

**Mesa 语义**（1980 年，由 Xerox Mesa 语言引入）：
- `signal()` 后，被唤醒的线程只是从等待队列移到就绪队列
- signal 线程继续运行，被唤醒线程稍后被调度时才运行
- 被唤醒线程**不一定**能看到条件为真（其他线程可能抢先修改了条件）

```
线程 A: while (!ready) wait();
线程 B: ready = 1; signal → B 继续运行 → A 稍后运行 → A 检查 ready

但可能：B: ready = 1; signal → B 继续 → B: ready = 0 → A 运行 → ready=0！
```

**为什么必须用 while 循环**：Mesa 语义下，被唤醒的线程在实际运行前，条件可能已经变为假（其他线程可能抢先修改了条件）。while 循环保证每次醒来都重新检查条件——如果条件仍为假，继续等待。

```c
// 错误（Hoare 语义下正确，Mesa 语义下错误）
if (!ready) {
    pthread_cond_wait(&cond, &mutex);
}
// 问题：被唤醒后不重新检查 ready，可能使用错误的条件

// 正确（Mesa 语义）
while (!ready) {
    pthread_cond_wait(&cond, &mutex);
}
// 每次醒来都重新检查 ready——保证条件为真才继续
```

**为什么 Mesa 语义更常用**：Mesa 语义的实现更简单、更高效。signal 不需要立即切换到被唤醒的线程，可以延迟到调度器决定时再切换。这减少了上下文切换的次数。现代系统（包括 Pthreads、Java、Go）都使用 Mesa 语义。

**Mesa 语义与 Hoare 语义的代码对比**：

```c
// Hoare 语义下的生产者-消费者
// 生产者
mutex_lock(&mutex);
if (count == MAX) {
    cond_wait(&not_full, &mutex);  // 被唤醒后一定有 count < MAX
}
buffer[in++] = item;
cond_signal(&not_empty);
mutex_unlock(&mutex);

// Mesa 语义下的生产者-消费者
// 生产者
mutex_lock(&mutex);
while (count == MAX) {  // 必须用 while！被唤醒后需要重新检查
    cond_wait(&not_full, &mutex);
}
buffer[in++] = item;
cond_signal(&not_empty);
mutex_unlock(&mutex);
```

### 15.3.4 虚假唤醒（Spurious Wakeup）

即使没有线程调用 `signal()`，`wait()` 也可能返回。这称为**虚假唤醒**。虚假唤醒是 POSIX 标准允许的行为，可能是由于：
- 实现细节（如信号处理中断了 wait）
- 内核内部的唤醒机制
- 多核系统上的缓存一致性协议

因此，**必须在 while 循环中检查条件**，而不是 if 语句。这是 Pthreads 编程中最常见的错误之一。

### 15.3.5 signal vs broadcast

**`pthread_cond_signal`**：唤醒一个等待线程。适合只有一个等待者需要被唤醒的场景（如生产者-消费者中，唤醒一个消费者）。

**`pthread_cond_broadcast`**：唤醒所有等待线程。适合所有等待者都需要检查条件的场景（如条件可能对多个线程都有意义）。

```c
// 使用 signal（只有一个消费者需要唤醒）
pthread_cond_signal(&not_empty);

// 使用 broadcast（所有消费者都需要检查）
pthread_cond_broadcast(&not_empty);
```

选择 signal 还是 broadcast 取决于语义：如果只有一个等待者需要被唤醒，用 signal（更高效）；如果不确定哪个等待者需要被唤醒，用 broadcast（更安全）。

---

## 15.4 xv6 sleep/wakeup

### 15.4.1 sleep() 与 wakeup() 原语

xv6 使用 `sleep()` 和 `wakeup()` 作为基本的同步原语。它们比信号量更简单，但也更容易出错——是理解操作系统同步机制的绝佳切入点。

```c
// kernel/proc.c
void sleep(void *chan, struct spinlock *lk) {
    struct proc *p = myproc();
    
    acquire(&ptable.lock);  // 获取进程表锁
    release(lk);            // 释放调用者持有的锁
    
    p->chan = chan;          // 设置睡眠通道——"我在等待什么"
    p->state = SLEEPING;    // 设置进程状态为睡眠
    sched();                // 让出 CPU——调度器选择其他进程运行
    
    // 被唤醒后从这里继续执行
    p->chan = 0;            // 清除睡眠通道
    release(&ptable.lock);  // 释放进程表锁
    acquire(lk);            // 重新获取调用者的锁
}

void wakeup(void *chan) {
    acquire(&ptable.lock);
    // 遍历所有进程，唤醒在同一个通道上睡眠的进程
    for (struct proc *p = proc; p < &proc[NPROC]; p++) {
        if (p->state == SLEEPING && p->chan == chan) {
            p->state = RUNNABLE;  // 唤醒——设置为可运行
        }
    }
    release(&ptable.lock);
}
```

**睡眠通道（Sleep Channel）**：`chan` 是一个任意的地址，用于标识"在等待什么"。`wakeup()` 唤醒所有在同一个通道上睡眠的进程。通道本身没有特殊含义——它只是一个标识符，通常使用相关数据结构的地址（如 `&pi->nread` 表示"等待管道有数据可读"）。

### 15.4.2 Lost Wakeup 问题

**Lost Wakeup** 是 `sleep()`/`wakeup()` 的经典 Bug：`wakeup()` 在 `sleep()` 之前执行，导致唤醒丢失——没有人会再来唤醒睡眠的进程。

```c
// 有问题的代码——不使用锁保护条件检查
// 生产者：
while (count == MAX) {
    sleep(channel);  // 条件不满足，睡眠
}

// 消费者：
count--;
wakeup(channel);  // 唤醒生产者
```

问题场景：

```
时刻  生产者                     消费者
──────────────────────────────────────────
t1    检查 count == MAX (是)
t2                               count-- (现在 count < MAX)
t3                               wakeup(channel) —— 没有人在睡眠！唤醒丢失！
t4    sleep(channel) → 睡眠 —— 没有人会来唤醒我！
→ 生产者永远睡眠！这就是 Lost Wakeup。

**Lost Wakeup 的根本原因**：检查条件和调用 sleep 之间不是原子的。在检查 `count == MAX` 之后、调用 `sleep()` 之前，消费者可能已经修改了 count 并调用了 wakeup。

**解决方案详解**：

```c
// 正确的代码——使用锁保护条件检查
// 生产者：
acquire(&lock);                    // 获取锁
while (count == MAX) {             // 检查条件
    sleep(channel, &lock);         // 原子地释放锁 + 睡眠
    // sleep(chan, lk) 内部：
    //   1. 获取 ptable.lock
    //   2. 释放 lk（lock）
    //   3. 设置进程状态为 SLEEPING
    //   4. 调用 sched() 让出 CPU
    //   5. 被唤醒后重新获取 lk（lock）
}
// count < MAX，可以生产
buffer[in++] = item;
count++;
wakeup(channel);                   // 唤醒消费者
release(&lock);                    // 释放锁

// 消费者：
acquire(&lock);                    // 获取锁
while (count == 0) {               // 检查条件
    sleep(channel, &lock);         // 原子地释放锁 + 睡眠
}
// count > 0，可以消费
item = buffer[--out];
count--;
wakeup(channel);                   // 唤醒生产者
release(&lock);                    // 释放锁
```

**为什么这个解决方案正确？**

1. **原子性保证**：`sleep(chan, lk)` 在持有 `ptable.lock` 的情况下释放 `lk` 并设置进程状态。这确保了 `wakeup()` 不会在 `sleep()` 设置状态之后、实际调度之前执行。

2. **while 循环**：使用 while 而不是 if——被唤醒后需要重新检查条件（Mesa 语义）。

3. **锁的传递**：`sleep()` 接收调用者的锁 `lk`，在内部释放它，被唤醒后重新获取。这确保了条件检查和睡眠之间的原子性。

### 15.4.3 xv6 sleep/wakeup 的使用示例

让我们通过 xv6 中的实际使用来理解 sleep/wakeup 的应用：

```c
// xv6 pipe 的读端
int piperead(struct pipe *pi, char *addr, int n) {
    int i;
    acquire(&pi->lock);                          // 获取管道锁
    while (pi->nread == pi->nwrite && pi->writeopen) {
        // 缓冲区空（nread == nwrite）且写端打开
        // 需要等待数据
        sleep(&pi->nread, &pi->lock);            // 睡眠在 nread 通道上
    }
    // 有数据可读
    for (i = 0; i < n && pi->nread < pi->nwrite; i++) {
        addr[i] = pi->data[pi->nread++ % PIPESIZE];
    }
    wakeup(&pi->nwrite);                         // 唤醒写端
    release(&pi->lock);
    return i;
}

// xv6 pipe 的写端
int pipewrite(struct pipe *pi, char *addr, int n) {
    int i;
    acquire(&pi->lock);
    for (i = 0; i < n; i++) {
        while (pi->nwrite == pi->nread + PIPESIZE) {
            // 缓冲区满，需要等待空位
            if (!pi->readopen) {
                release(&pi->lock);
                return -1;  // 读端已关闭，返回错误
            }
            sleep(&pi->nwrite, &pi->lock);       // 睡眠在 nwrite 通道上
        }
        pi->data[pi->nwrite++ % PIPESIZE] = addr[i];
    }
    wakeup(&pi->nread);                          // 唤醒读端
    release(&pi->lock);
    return n;
}
```

**关键设计模式**：
1. **睡眠通道**：使用数据结构的地址作为通道（`&pi->nread`、`&pi->nwrite`）
2. **while 循环**：被唤醒后重新检查条件
3. **持锁检查**：在持有锁的情况下检查条件和调用 sleep
4. **对称唤醒**：读端唤醒写端，写端唤醒读端

### 15.4.4 sleep/wakeup vs 信号量

```
特性              sleep/wakeup    信号量
实现复杂度        简单            中等
类型安全          低（void*）     高（semaphore_t）
语义清晰度        低              高
计数功能          无              有（初始值 N）
使用难度          高（容易出错）  低
适用场景          内核内部        通用
```

sleep/wakeup 更简单但更容易出错——没有类型检查，没有计数功能，需要手动管理睡眠通道。信号量更安全、更清晰，但实现更复杂。
```

**解决方案**：在持有锁的情况下检查条件和调用 `sleep()`：

```c
// 正确的代码——使用锁保护条件检查
// 生产者：
acquire(&lock);
while (count == MAX) {
    sleep(channel, &lock);  // 原子地释放锁 + 睡眠
    // 被唤醒后重新获取锁
}
// ... 生产 ...
release(&lock);

// 消费者：
acquire(&lock);
count--;
wakeup(channel);
release(&lock);
```

`sleep(chan, lk)` 的设计确保了原子性：它在持有 `ptable.lock` 的情况下释放 `lk` 并设置进程状态为 SLEEPING。这确保了 `wakeup()` 不会在 `sleep()` 设置状态之后、实际调度之前执行——因为 `wakeup()` 也需要获取 `ptable.lock`。

<div data-component="SleepWakeupMechanism"></div>

---

## 15.5 面试高频考点

**Q1：信号量和互斥锁的区别？**

互斥锁只能用于互斥（初值=1），信号量还可以用于条件同步（初值=0）和资源计数（初值=N）。互斥锁的 unlock() 必须由持有锁的线程调用，信号量的 V() 可以由任何线程调用。

**Q2：为什么条件变量必须用 while 循环？**

Mesa 语义下，被唤醒的线程在实际运行前条件可能已经变为假（其他线程可能抢先修改了条件）。虚假唤醒也可能发生。while 循环保证每次醒来都重新检查条件。

**Q3：什么是 Lost Wakeup？如何解决？**

wakeup() 在 sleep() 之前执行，导致唤醒丢失。解决方案：在持有锁的情况下检查条件和调用 sleep()，确保 wakeup() 不会在检查和睡眠之间执行。

**Q4：Mesa 语义和 Hoare 语义的区别？**

Hoare 语义：signal() 后被唤醒线程立即运行，一定能看到条件为真。Mesa 语义：signal() 后被唤醒线程只是就绪，signal 线程继续运行，条件可能已变。现代系统使用 Mesa 语义。

**Q5：signal 和 broadcast 的区别？什么时候用哪个？**

signal 唤醒一个等待线程，broadcast 唤醒所有等待线程。使用 signal 的场景：只有一个等待者需要被唤醒（如生产者-消费者中唤醒一个消费者）。使用 broadcast 的场景：所有等待者都需要检查条件（如条件可能对多个线程都有意义）。

```c
// 使用 signal 的场景：只有一个消费者需要唤醒
pthread_cond_signal(&not_empty);  // 唤醒一个等待的消费者

// 使用 broadcast 的场景：所有等待者都需要检查
pthread_cond_broadcast(&cond);  // 唤醒所有等待线程
```

**Q6：信号量和条件变量的区别？**

信号量是一个整数 + P/V 操作，适合资源计数和条件同步。条件变量 + 锁适合等待任意条件。信号量的 V() 可以由任何线程调用，条件变量的 signal() 通常由修改条件的线程调用。

```
特性              信号量          条件变量
有计数功能        是              否
可以无锁调用 V    是              否（需要持锁）
适合资源计数      是              否
适合等待任意条件  否              是
实现复杂度        中等            中等
```

---

## 15.6 经典同步问题：信号量解法

### 15.6.1 信号量实现互斥锁

```c
semaphore_t mutex;
sem_init(&mutex, 1);  // 初始值 = 1

void critical_section() {
    P(&mutex);      // 获取锁
    // 临界区
    V(&mutex);      // 释放锁
}
```

### 15.6.2 信号量实现条件同步

```c
semaphore_t done;
sem_init(&done, 0);  // 初始值 = 0

// 线程 A：等待线程 B 完成
void thread_A() {
    // 做一些工作
    P(&done);  // 等待 B 完成
    // B 已完成，继续
}

// 线程 B：完成工作后通知 A
void thread_B() {
    // 完成一些工作
    V(&done);  // 通知 A
}
```

### 15.6.3 信号量实现资源池

```c
#define POOL_SIZE 5
semaphore_t pool;
sem_init(&pool, POOL_SIZE);  // 5 个资源

void use_resource() {
    P(&pool);      // 获取资源（如果池空则等待）
    // 使用资源
    V(&pool);      // 归还资源
}
```

---

## 15.7 条件变量的高级用法

### 15.7.1 条件变量实现信号量

```c
// 用条件变量实现信号量
typedef struct {
    int value;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} cv_semaphore_t;

void cv_sem_init(cv_semaphore_t *s, int initial_value) {
    s->value = initial_value;
    pthread_mutex_init(&s->mutex, NULL);
    pthread_cond_init(&s->cond, NULL);
}

void cv_sem_wait(cv_semaphore_t *s) {
    pthread_mutex_lock(&s->mutex);
    while (s->value <= 0) {  // 必须用 while
        pthread_cond_wait(&s->cond, &s->mutex);
    }
    s->value--;
    pthread_mutex_unlock(&s->mutex);
}

void cv_sem_post(cv_semaphore_t *s) {
    pthread_mutex_lock(&s->mutex);
    s->value++;
    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->mutex);
}
```

### 15.7.2 条件变量实现屏障（Barrier）

```c
// 屏障：所有线程到达后才能继续
typedef struct {
    int count;           // 需要到达的线程数
    int arrived;         // 已到达的线程数
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} barrier_t;

void barrier_init(barrier_t *b, int count) {
    b->count = count;
    b->arrived = 0;
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
}

void barrier_wait(barrier_t *b) {
    pthread_mutex_lock(&b->mutex);
    b->arrived++;
    if (b->arrived == b->count) {
        // 最后一个到达的线程唤醒所有等待者
        b->arrived = 0;  // 重置以便重复使用
        pthread_cond_broadcast(&b->cond);
    } else {
        // 等待所有线程到达
        pthread_cond_wait(&b->cond, &b->mutex);
    }
    pthread_mutex_unlock(&b->mutex);
}
```

---

## 15.8 信号量与条件变量的性能对比

```
操作                延迟            适用场景
sem_wait            ~50 ns          资源计数
sem_post            ~50 ns          资源计数
pthread_mutex_lock  ~30 ns          互斥
pthread_cond_wait   ~100 ns         条件等待
pthread_cond_signal ~50 ns          条件通知
```

信号量和条件变量的性能相近，但适用场景不同：
- **信号量**：适合资源计数（如线程池、连接池）
- **条件变量**：适合等待任意条件（如队列非空、缓冲区不满）

### 15.8.1 选择同步原语的指南

```
场景                          推荐原语        原因
保护临界区                    互斥锁          简单、高效
等待特定条件                  条件变量        灵活、支持任意条件
管理 N 个资源                 信号量          内置计数功能
一个线程通知另一个线程        信号量（初值0）  简单、无需锁
多个线程需要同步              屏障            所有线程到达后继续
```

### 15.8.2 常见错误

**错误 1：使用 if 而不是 while**
```c
// 错误
if (!ready) {
    pthread_cond_wait(&cond, &mutex);
}

// 正确
while (!ready) {
    pthread_cond_wait(&cond, &mutex);
}
```

**错误 2：在持有锁时调用 signal**
```c
// 可能导致"惊群"效应
pthread_mutex_lock(&mutex);
ready = 1;
pthread_cond_broadcast(&cond);  // 唤醒所有等待者
pthread_mutex_unlock(&mutex);
// 所有等待者同时竞争锁

// 更好的做法：先释放锁再 signal
pthread_mutex_lock(&mutex);
ready = 1;
pthread_mutex_unlock(&mutex);
pthread_cond_signal(&cond);  // 只唤醒一个等待者
```

**错误 3：使用 signal 而不是 broadcast**
```c
// 错误：只有一个等待者被唤醒，但条件可能对多个线程有意义
pthread_cond_signal(&cond);

// 正确：所有等待者都需要检查条件
pthread_cond_broadcast(&cond);
```

---

## 15.9 扩展阅读

- **OSTEP** Chapter 30: "Condition Variables" — 条件变量详解
- **OSTEP** Chapter 31: "Semaphores" — 信号量详解
- **Operating System Concepts** Chapter 5.6-5.9 — 同步工具
- **xv6-riscv** kernel/proc.c — sleep()/wakeup() 实现
- [POSIX Threads Programming](https://hpc-tutorials.llnl.gov/posix/) — Pthreads 教程
- [The Little Book of Semaphores](https://greenteapress.com/wp/semaphores/) — 信号量练习题

### 15.9.1 推荐实践

1. **优先使用互斥锁 + 条件变量**：这是最通用的同步模式
2. **信号量用于资源计数**：当需要管理 N 个资源时使用信号量
3. **始终使用 while 循环**：条件变量的 wait 必须在 while 循环中调用
4. **避免惊群效应**：使用 signal 而不是 broadcast（除非所有等待者都需要唤醒）
5. **保持临界区短**：减少锁竞争，提高并发性

### 15.9.2 信号量与条件变量的实现原理

**信号量的实现**（基于互斥锁 + 条件变量）：

```c
typedef struct {
    int value;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} semaphore_t;

void sem_wait(semaphore_t *s) {
    pthread_mutex_lock(&s->mutex);
    while (s->value <= 0) {
        pthread_cond_wait(&s->cond, &s->mutex);
    }
    s->value--;
    pthread_mutex_unlock(&s->mutex);
}

void sem_post(semaphore_t *s) {
    pthread_mutex_lock(&s->mutex);
    s->value++;
    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->mutex);
}
```

**条件变量的实现**（基于信号量）：

```c
typedef struct {
    pthread_mutex_t *mutex;
    semaphore_t sem;
    int waiters;
} condition_t;

void cond_wait(condition_t *cond, pthread_mutex_t *mutex) {
    cond->mutex = mutex;
    cond->waiters++;
    pthread_mutex_unlock(mutex);
    sem_wait(&cond->sem);
    pthread_mutex_lock(mutex);
}

void cond_signal(condition_t *cond) {
    if (cond->waiters > 0) {
        cond->waiters--;
        sem_post(&cond->sem);
    }
}
```

**关键洞察**：信号量和条件变量可以互相实现——它们在功能上是等价的。选择哪个取决于语义和使用场景。

### 15.9.3 信号量的历史

信号量由 Edsger Dijkstra 于 1965 年提出，用于解决 THE 操作系统中的同步问题。Dijkstra 使用 P（Proberen，尝试）和 V（Verhogen，增加）两个操作来管理共享资源。

Dijkstra 的贡献不仅限于信号量——他还提出了：
- **最短路径算法**（Dijkstra 算法）
- **银行家算法**（死锁避免）
- **结构化编程**（goto 有害论）
- **并发编程的形式化方法**

信号量是并发编程历史上最重要的发明之一——它提供了一种简单而强大的方式来解决同步问题，至今仍是操作系统和并发编程的基础。

### 15.9.4 信号量的变体

**计数信号量**：初始值为 N，管理 N 个资源
```c
semaphore_t pool;
sem_init(&pool, 5);  // 5 个资源
```

**二值信号量**：初始值为 1，等价于互斥锁
```c
semaphore_t mutex;
sem_init(&mutex, 1);  // 互斥锁
```

**命名信号量**：通过名称共享，用于进程间同步
```c
sem_t *sem = sem_open("/mysem", O_CREAT, 0644, 1);
sem_wait(sem);
// 临界区
sem_post(sem);
sem_close(sem);
```

**匿名信号量**：通过内存共享，用于线程间同步
```c
semaphore_t sem;
sem_init(&sem, 0, 1);  // pshared=0（线程间），value=1
```

### 15.9.5 条件变量的实现原理

条件变量的实现依赖于互斥锁和等待队列：

```c
typedef struct {
    queue_t waiters;     // 等待队列
    pthread_mutex_t lock; // 保护 waiters
} cond_t;

void cond_wait(cond_t *cond, pthread_mutex_t *mutex) {
    pthread_mutex_lock(&cond->lock);
    enqueue(&cond->waiters, current_thread);
    pthread_mutex_unlock(&cond->lock);
    
    pthread_mutex_unlock(mutex);  // 释放外部锁
    thread_sleep();               // 睡眠
    pthread_mutex_lock(mutex);    // 重新获取外部锁
}

void cond_signal(cond_t *cond) {
    pthread_mutex_lock(&cond->lock);
    if (!empty(&cond->waiters)) {
        Thread *t = dequeue(&cond->waiters);
        thread_wakeup(t);
    }
    pthread_mutex_unlock(&cond->lock);
}
```

**关键实现细节**：
1. **原子性**：`cond_wait` 在释放外部锁和睡眠之间是原子的
2. **重新获取锁**：被唤醒后需要重新获取外部锁
3. **while 循环**：使用 while 而不是 if 检查条件（Mesa 语义）

### 15.9.6 信号量的实现原理

信号量的实现依赖于互斥锁、条件变量和等待队列：

```c
typedef struct {
    int value;           // 信号量的值
    queue_t waiters;     // 等待队列
    pthread_mutex_t lock; // 保护 value 和 waiters
} semaphore_t;

void sem_wait(semaphore_t *s) {
    pthread_mutex_lock(&s->lock);
    s->value--;
    if (s->value < 0) {
        enqueue(&s->waiters, current_thread);
        pthread_mutex_unlock(&s->lock);
        thread_sleep();
    } else {
        pthread_mutex_unlock(&s->lock);
    }
}

void sem_post(semaphore_t *s) {
    pthread_mutex_lock(&s->lock);
    s->value++;
    if (s->value <= 0) {
        Thread *t = dequeue(&s->waiters);
        thread_wakeup(t);
    }
    pthread_mutex_unlock(&s->lock);
}
```

**关键实现细节**：
1. **内部锁**：保护 value 和 waiters
2. **睡眠前释放锁**：避免死锁
3. **唤醒后重新获取锁**：被唤醒的线程需要重新获取内部锁
4. **计数功能**：value 可以是负数（绝对值表示等待线程数）
