---
title: "Chapter 16: 经典同步问题"
description: "深入理解生产者-消费者、读者-写者、哲学家就餐、睡理发师等经典同步问题的完整解决方案与正确性分析"
updated: "2026-06-10"
---

# Chapter 16: 经典同步问题

> **本章目标**：
> - 掌握生产者-消费者问题的信号量和条件变量解决方案
> - 理解读者-写者问题的三种变体与解决方案
> - 掌握哲学家就餐问题的多种解决方案（避免死锁）
> - 了解睡理发师问题
> - 能够分析同步方案的正确性

---

## 16.1 生产者-消费者问题

### 16.1.1 问题描述

**生产者-消费者问题**（也称有界缓冲区问题）是最经典的同步问题之一，由 Dijkstra 在 1965 年首次提出。它出现在许多实际系统中：管道（pipe）、消息队列、线程池、缓冲区 I/O、网络数据包处理等。

问题的设定：
- **生产者**：生成数据项，放入缓冲区。例如，从网络读取数据包、生成计算结果
- **消费者**：从缓冲区取出数据项，进行处理。例如，解析数据包、存储计算结果
- **缓冲区**：固定大小的环形缓冲区（有限容量）。例如，10 个槽位

约束条件（必须同时满足）：
1. **缓冲区满时，生产者必须等待**——不能覆盖未消费的数据
2. **缓冲区空时，消费者必须等待**——不能读取无效数据
3. **同一时刻只有一个线程可以访问缓冲区**——防止竞态条件

```
生产者-消费者模型：

生产者 → [item1][item2][item3][item4][item5] → 消费者
         ↑          有限缓冲区              ↑
    放入缓冲区                          从缓冲区取出
    （可能等待满）                    （可能等待空）
```

### 16.1.2 信号量解决方案

```c
#define BUFFER_SIZE 10

int buffer[BUFFER_SIZE];  // 环形缓冲区
int in = 0;   // 生产者写入位置——下一个数据项放入的位置
int out = 0;  // 消费者读取位置——下一个数据项取出的位置

semaphore_t mutex;  // 互斥锁（初值=1）——保护缓冲区访问
semaphore_t empty;  // 空槽位计数（初值=BUFFER_SIZE）——缓冲区满时生产者等待
semaphore_t full;   // 满槽位计数（初值=0）——缓冲区空时消费者等待

void init(void) {
    sem_init(&mutex, 1);             // 互斥锁——初始为 1
    sem_init(&empty, BUFFER_SIZE);   // 空槽位——初始为缓冲区大小
    sem_init(&full, 0);              // 满槽位——初始为 0（没有数据）
}

// 生产者线程
void producer(void) {
    while (1) {
        int item = produce_item();  // 生成一个数据项
        
        P(&empty);     // 等待空槽位——如果缓冲区满则阻塞
        P(&mutex);     // 获取互斥锁——独占访问缓冲区
        
        buffer[in] = item;                    // 将数据项放入缓冲区
        in = (in + 1) % BUFFER_SIZE;          // 更新写入位置（环形）
        
        V(&mutex);     // 释放互斥锁
        V(&full);      // 增加满槽位计数——通知消费者有新数据
    }
}

// 消费者线程
void consumer(void) {
    while (1) {
        P(&full);      // 等待满槽位——如果缓冲区空则阻塞
        P(&mutex);     // 获取互斥锁——独占访问缓冲区
        
        int item = buffer[out];               // 从缓冲区取出数据项
        out = (out + 1) % BUFFER_SIZE;        // 更新读取位置（环形）
        
        V(&mutex);     // 释放互斥锁
        V(&empty);     // 增加空槽位计数——通知生产者有空位
        
        consume_item(item);  // 处理数据项
    }
}
```

**信号量的作用详解**：
- `mutex`（初值=1）：保证同一时刻只有一个线程访问缓冲区——这是互斥保护
- `empty`（初值=BUFFER_SIZE）：记录空槽位数量——缓冲区满时（empty=0）生产者阻塞
- `full`（初值=0）：记录满槽位数量——缓冲区空时（full=0）消费者阻塞

**P 操作的顺序非常重要**：必须先 `P(&empty)` 再 `P(&mutex)`。如果反过来：

```
错误顺序：P(&mutex) → P(&empty)

如果缓冲区满（empty=0）：
  生产者：P(&mutex) 成功，持有 mutex
  生产者：P(&empty) 阻塞——等待消费者消费数据
  消费者：P(&mutex) 阻塞——等待生产者释放 mutex
  → 死锁！生产者等待消费者，消费者等待生产者
```

<div data-component="ProducerConsumerSimulator"></div>

### 16.1.3 条件变量解决方案

```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;   // 缓冲区不满的条件
pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;  // 缓冲区不空的条件

int buffer[BUFFER_SIZE];
int in = 0, out = 0, count = 0;  // count = 当前数据项数量

// 生产者线程
void producer(void) {
    while (1) {
        int item = produce_item();
        
        pthread_mutex_lock(&mutex);
        while (count == BUFFER_SIZE) {  // 缓冲区满——必须用 while！
            pthread_cond_wait(&not_full, &mutex);
        }
        
        buffer[in] = item;
        in = (in + 1) % BUFFER_SIZE;
        count++;  // 数据项数量增加
        
        pthread_cond_signal(&not_empty);  // 通知消费者：有新数据
        pthread_mutex_unlock(&mutex);
    }
}

// 消费者线程
void consumer(void) {
    while (1) {
        pthread_mutex_lock(&mutex);
        while (count == 0) {  // 缓冲区空——必须用 while！
            pthread_cond_wait(&not_empty, &mutex);
        }
        
        int item = buffer[out];
        out = (out + 1) % BUFFER_SIZE;
        count--;  // 数据项数量减少
        
        pthread_cond_signal(&not_full);  // 通知生产者：有空位
        pthread_mutex_unlock(&mutex);
        
        consume_item(item);
    }
}
```

### 16.1.4 xv6 pipe 实现分析

xv6 的 pipe（管道）是生产者-消费者问题的实际应用。管道是 Unix 系统中最基本的进程间通信机制——一个进程写入数据，另一个进程读取数据。

```c
// kernel/pipe.c
struct pipe {
    struct spinlock lock;     // 保护管道的自旋锁
    char data[PIPESIZE];     // 环形缓冲区（512 字节）
    uint nread;              // 已读取的总字节数
    uint nwrite;             // 已写入的总字节数
    int readopen;            // 读端是否打开
    int writeopen;           // 写端是否打开
};

// 读取管道
int piperead(struct pipe *pi, char *addr, int n) {
    acquire(&pi->lock);
    while (pi->nread == pi->nwrite && pi->writeopen) {
        // 缓冲区空（nread == nwrite）且写端打开——等待数据
        sleep(&pi->nread, &pi->lock);
    }
    // 读取数据
    while (i < n && pi->nread < pi->nwrite) {
        addr[i++] = pi->data[pi->nread++ % PIPESIZE];
    }
    wakeup(&pi->nwrite);  // 唤醒写端——有空位了
    release(&pi->lock);
    return i;
}

// 写入管道
int pipewrite(struct pipe *pi, char *addr, int n) {
    acquire(&pi->lock);
    for (i = 0; i < n; i++) {
        while (pi->nwrite == pi->nread + PIPESIZE) {
            // 缓冲区满——等待空位
            if (!pi->readopen) {
                release(&pi->lock);
                return -1;  // 读端已关闭
            }
            sleep(&pi->nwrite, &pi->lock);
        }
        pi->data[pi->nwrite++ % PIPESIZE] = addr[i];
    }
    wakeup(&pi->nread);  // 唤醒读端——有数据了
    release(&pi->lock);
    return n;
}
```

xv6 pipe 的特点：
- 使用 `sleep()`/`wakeup()` 而不是信号量——更简单但更容易出错
- 使用 `nread` 和 `nwrite` 的差值判断缓冲区空/满——`(nwrite - nread) == PIPESIZE` 表示满
- 睡眠通道分别是 `&pi->nread` 和 `&pi->nwrite`——用数据结构的地址作为标识符

---

## 16.2 读者-写者问题

### 16.2.1 问题描述

**读者-写者问题**由 Courtois、Heymans 和 Parnas 于 1971 年提出。多个线程访问共享数据结构：
- **读者**：只读取数据，不修改——可以并发执行
- **写者**：修改数据——必须独占访问

约束条件：
- 多个读者可以同时读取——因为读取不会修改数据
- 写者必须独占访问——不能与读者或其他写者同时
- 不同变体对读者/写者的优先级有不同要求

### 16.2.2 第一类读者-写者问题：读者优先

只要有读者在读取，新读者可以立即加入——写者必须等待所有读者完成。

```c
semaphore_t rw_mutex;   // 读写互斥（初值=1）——写者独占
semaphore_t mutex;      // 保护 reader_count 的互斥锁（初值=1）
int reader_count = 0;   // 当前正在读取的读者数量

// 读者
void reader(void) {
    P(&mutex);
    reader_count++;
    if (reader_count == 1) {
        // 第一个读者到达——锁住写锁，阻止写者
        P(&rw_mutex);
    }
    V(&mutex);
    
    // 读取数据——多个读者可以并发执行
    read_data();
    
    P(&mutex);
    reader_count--;
    if (reader_count == 0) {
        // 最后一个读者离开——释放写锁，允许写者
        V(&rw_mutex);
    }
    V(&mutex);
}

// 写者
void writer(void) {
    P(&rw_mutex);      // 获取写锁——独占访问
    
    // 写入数据——没有读者或其他写者
    write_data();
    
    V(&rw_mutex);      // 释放写锁
}
```

**问题**：写者可能饥饿——如果有持续不断的读者到达，`readers` 永远不为 0，写锁永远无法获取。

### 16.2.3 第二类读者-写者问题：写者优先

写者到达后，阻止新读者加入——写者优先执行。

```c
semaphore_t rw_mutex;    // 读写互斥
semaphore_t read_mutex;  // 阻止新读者
int reader_count = 0;

// 读者
void reader(void) {
    P(&read_mutex);     // 等待——如果有写者在等待则阻塞
    P(&mutex);
    reader_count++;
    if (reader_count == 1) {
        P(&rw_mutex);   // 第一个读者锁住写锁
    }
    V(&mutex);
    V(&read_mutex);     // 允许其他读者进入
    
    read_data();
    
    P(&mutex);
    reader_count--;
    if (reader_count == 0) {
        V(&rw_mutex);   // 最后一个读者释放写锁
    }
    V(&mutex);
}

// 写者
void writer(void) {
    P(&read_mutex);     // 阻止新读者——确保写者能获取 rw_mutex
    P(&rw_mutex);       // 等待当前读者完成
    
    write_data();
    
    V(&rw_mutex);
    V(&read_mutex);     // 允许新读者进入
}
```

<div data-component="ReaderWriterVisualizer"></div>

### 16.2.4 Linux RCU（Read-Copy-Update）

RCU 是 Linux 内核中用于读多写少场景的高性能同步机制。读端完全无锁，写端使用"复制-修改-替换"策略。

```c
// 读端（完全没有锁！可以并发读取）
rcu_read_lock();                     // 标记进入 RCU 读侧临界区
struct data *p = rcu_dereference(global_ptr);  // 读取数据指针
// 使用 p——不需要任何锁，可以安全读取
rcu_read_unlock();                   // 标记离开 RCU 读侧临界区

// 写端（需要互斥锁保护）
mutex_lock(&write_lock);
struct data *new = kmalloc(sizeof(*new));  // 分配新内存
*new = *old;                               // 复制旧数据
new->field = new_value;                    // 修改新数据
rcu_assign_pointer(global_ptr, new);       // 原子替换指针
mutex_unlock(&write_lock);
synchronize_rcu();                         // 等待所有读者完成
kfree(old);                                // 释放旧数据
```

---

## 16.3 哲学家就餐问题

### 16.3.1 问题描述

5 个哲学家围坐在圆桌旁，每人左右各有一根叉子（共 5 根）。哲学家交替思考和进餐。进餐时需要同时获取左右两根叉子。

```
        哲学家 0
    叉4         叉0
哲学家4           哲学家1
    叉3         叉1
        哲学家3
            叉2
        哲学家2
```

这个问题的核心挑战是**避免死锁**：如果所有哲学家同时拿起左叉子，所有人都在等待右叉子，但右叉子被邻居持有——死锁！

<div data-component="DiningPhilosophersAnimation"></div>

### 16.3.2 错误方案：简单获取两个叉子

```c
semaphore_t fork[5];
// 所有 fork 初值 = 1

void philosopher(int i) {
    while (1) {
        think();                       // 思考
        P(&fork[i]);                   // 获取左叉子
        P(&fork[(i+1) % 5]);           // 获取右叉子
        eat();                         // 进餐（需要两根叉子）
        V(&fork[(i+1) % 5]);           // 释放右叉子
        V(&fork[i]);                   // 释放左叉子
    }
}
```

**死锁场景**：

```
时刻  哲学家0        哲学家1        哲学家2        哲学家3        哲学家4
──────────────────────────────────────────────────────────────────────────
t1    P(叉0) ✓       P(叉1) ✓       P(叉2) ✓       P(叉3) ✓       P(叉4) ✓
t2    P(叉1) 等待    P(叉2) 等待    P(叉3) 等待    P(叉4) 等待    P(叉0) 等待
→ 所有人都持有左叉子，等待右叉子，但右叉子被邻居持有 → 死锁！
```

### 16.3.3 解决方案一：非对称方案

让一个哲学家先拿右叉子，其他先拿左叉子——打破循环等待（死锁的四个必要条件之一）。

```c
void philosopher(int i) {
    while (1) {
        think();
        if (i == 0) {
            // 哲学家 0 先拿右叉子（打破对称性）
            P(&fork[(i+1) % 5]);   // 先拿右叉子（叉1）
            P(&fork[i]);           // 再拿左叉子（叉0）
        } else {
            // 其他哲学家先拿左叉子
            P(&fork[i]);           // 先拿左叉子
            P(&fork[(i+1) % 5]);   // 再拿右叉子
        }
        eat();
        V(&fork[(i+1) % 5]);
        V(&fork[i]);
    }
}
```

**为什么有效**：哲学家 0 先拿右叉子（叉1），哲学家 1 先拿左叉子（叉1）。它们竞争同一根叉子，只有一个人能拿到。这打破了循环等待——不再是"每个人都等待右边的人"。

### 16.3.4 解决方案二：最多 N-1 个哲学家同时就餐

限制同时就餐的哲学家数量为 N-1（4 个），保证至少一个哲学家能获取两根叉子。

```c
semaphore_t room;
sem_init(&room, 4);  // 最多 4 个哲学家同时尝试进餐

void philosopher(int i) {
    while (1) {
        think();
        P(&room);               // 进入餐厅（最多 4 人）
        P(&fork[i]);            // 获取左叉子
        P(&fork[(i+1) % 5]);    // 获取右叉子
        eat();
        V(&fork[(i+1) % 5]);    // 释放右叉子
        V(&fork[i]);            // 释放左叉子
        V(&room);               // 离开餐厅
    }
}
```

**为什么有效**：5 根叉子，最多 4 个哲学家。根据鸽巢原理，至少有一个哲学家能同时拿到两根叉子——因为 4 个人最多占用 4 根叉子，至少有 1 根叉子空闲。

### 16.3.5 解决方案三：使用互斥锁

用一个全局互斥锁保护整个进餐过程——简单但并行性差。

```c
pthread_mutex_t table_lock = PTHREAD_MUTEX_INITIALIZER;

void philosopher(int i) {
    while (1) {
        think();
        pthread_mutex_lock(&table_lock);
        P(&fork[i]);
        P(&fork[(i+1) % 5]);
        eat();
        V(&fork[(i+1) % 5]);
        V(&fork[i]);
        pthread_mutex_unlock(&table_lock);
    }
}
```

**缺点**：并行性差——同一时刻只有一个哲学家可以尝试进餐。5 个哲学家完全串行化。

### 16.3.6 解决方案四：Chandy-Misra 方案

Chandy-Misra 方案是一个更高级的解决方案，允许哲学家并发请求叉子，通过消息传递避免死锁。

```c
// 每根叉子的状态：clean 或 dirty
// 初始状态：所有叉子都是 dirty
// 规则：
//   1. 如果叉子是 dirty，必须给请求的邻居
//   2. 如果叉子是 clean，可以拒绝请求
//   3. 哲学家吃完后，叉子变为 dirty
//   4. 拿到叉子后，叉子变为 clean

void philosopher(int i) {
    while (1) {
        think();
        request_forks(i);  // 请求左右叉子
        eat();
        release_forks(i);  // 释放叉子，标记为 dirty
    }
}
```

**为什么有效**：dirty 叉子必须让出，clean 叉子可以保留。这确保了至少有一个哲学家能拿到两根 clean 叉子。

### 16.3.7 哲学家就餐问题的性能对比

```
方案              并行性    实现复杂度    死锁风险    饥饿风险
非对称方案        高        低           无          无
限制就餐人数      高        低           无          无
互斥锁            低        低           无          无
Chandy-Misra      高        高           无          无
```

---

## 16.4 睡理发师问题

### 16.4.1 问题描述

理发店有一个理发师和 N 个等待椅子。理发师在没有顾客时睡觉。顾客到达时，如果理发师在睡觉则唤醒他，如果有空椅子则等待，否则离开。

```c
semaphore_t customers;  // 等待顾客数（初值=0）——理发师在此信号量上等待
semaphore_t barbers;    // 等待理发师数（初值=0）——顾客在此信号量上等待
semaphore_t mutex;      // 互斥锁（初值=1）——保护 waiting 变量
int waiting = 0;        // 当前等待的顾客数量

// 理发师线程
void barber(void) {
    while (1) {
        P(&customers);      // 等待顾客——如果没有顾客则睡觉
        P(&mutex);
        waiting--;           // 一个顾客开始理发
        V(&mutex);
        V(&barbers);         // 通知顾客：理发师就绪
        cut_hair();          // 理发
    }
}

// 顾客线程
void customer(void) {
    P(&mutex);
    if (waiting < N) {
        waiting++;           // 有空椅子，坐下等待
        V(&customers);       // 通知理发师：有顾客
        V(&mutex);
        P(&barbers);         // 等待理发师就绪
        get_haircut();       // 理发
    } else {
        V(&mutex);           // 没有空椅子，离开
    }
}
```

**睡理发师问题的分析**：

1. **理发师睡眠**：当没有顾客时（`customers` 为 0），理发师在 `P(&customers)` 处睡眠
2. **顾客唤醒理发师**：第一个顾客到达时调用 `V(&customers)`，唤醒理发师
3. **顾客等待理发师**：如果理发师正在理发，新顾客在 `P(&barbers)` 处等待
4. **理发师唤醒顾客**：理发师完成理发后调用 `V(&barbers)`，唤醒等待的顾客

**关键设计**：
- `customers` 信号量：理发师等待顾客
- `barbers` 信号量：顾客等待理发师
- `mutex` 互斥锁：保护 `waiting` 变量
- `waiting` 变量：当前等待的顾客数量

// 顾客线程
void customer(void) {
    P(&mutex);
    if (waiting < N) {
        waiting++;           // 有空椅子，坐下等待
        V(&customers);       // 通知理发师：有顾客
        V(&mutex);
        P(&barbers);         // 等待理发师就绪
        get_haircut();       // 理发
    } else {
        V(&mutex);           // 没有空椅子，离开
    }
}
```

---

## 16.5 面试高频考点

**Q1：生产者-消费者问题中 P 操作的顺序为什么重要？**

必须先 P(&empty) 再 P(&mutex)。如果反过来，缓冲区满时生产者持有 mutex 等待 empty，消费者需要 mutex 才能消费 → 死锁。

**Q2：读者-写者问题中读者优先和写者优先的区别？**

读者优先：只要有读者就允许读取，写者可能饥饿。写者优先：写者到达后阻止新读者，避免写者饥饿。

**Q3：哲学家就餐问题的死锁原因？**

循环等待——每个哲学家都持有左叉子等待右叉子，形成环形依赖。解决方案：打破循环（非对称方案、限制就餐人数）。

**Q4：信号量和条件变量的区别？**

信号量是一个整数 + P/V 操作，适合资源计数和条件同步。条件变量 + 锁适合等待任意条件。信号量的 V() 可以由任何线程调用，条件变量的 signal() 通常由修改条件的线程调用。

**Q5：如何证明一个同步方案的正确性？**

需要证明三个性质：互斥（临界区互斥访问）、进步（无死锁）、有限等待（无饥饿）。通常使用不变式（invariant）和反证法。

---

## 16.6 吸烟者问题

### 16.6.1 问题描述

三个吸烟者，每人有无限的一种材料（烟草、纸、胶水）。一个代理者随机放两种材料在桌上。拥有第三种材料的吸烟者拿走材料，卷烟，吸烟。代理者等吸烟者完成后再放新材料。

```c
// 材料：0=烟草，1=纸，2=胶水
semaphore_t agent;          // 代理者信号量（初值=1）
semaphore_t smoker[3];      // 三个吸烟者的信号量（初值=0）
int material[2];            // 桌上的两种材料

void agent_thread() {
    while (1) {
        P(&agent);
        // 随机选择两种材料
        int m1 = rand() % 3;
        int m2 = (m1 + 1 + rand() % 2) % 3;
        material[0] = m1;
        material[1] = m2;
        // 唤醒拥有第三种材料的吸烟者
        int smoker_id = 3 - m1 - m2;  // 计算第三种材料
        V(&smoker[smoker_id]);
    }
}

void smoker_thread(int id) {
    while (1) {
        P(&smoker[id]);
        // 拿走材料，卷烟，吸烟
        printf("Smoker %d is smoking\n", id);
        V(&agent);  // 通知代理者
    }
}
```

### 16.6.2 吸烟者问题的分析

吸烟者问题展示了信号量的另一种使用模式：**代理者-工人模式**。代理者负责分配任务（放材料），工人负责执行任务（卷烟吸烟）。

**关键设计**：
- `agent` 信号量：控制代理者的节奏——一次只放一种材料组合
- `smoker[i]` 信号量：控制每个吸烟者的节奏——只有拿到材料才吸烟
- 材料计算：`smoker_id = 3 - m1 - m2`——巧妙地计算出缺少哪种材料

---

## 16.7 经典同步问题的对比

```
问题              核心挑战        推荐解决方案
生产者-消费者      有界缓冲区      信号量（empty, full, mutex）
读者-写者          读写并发        读写锁或 RCU
哲学家就餐        死锁避免        非对称方案或限制就餐人数
睡理发师          线程协调        信号量（customers, barbers）
吸烟者问题        资源分配        信号量（agent, smoker[]）
```

### 16.7.1 选择同步原语的指南

```
场景                          推荐原语        原因
保护临界区                    互斥锁          简单、高效
等待特定条件                  条件变量        灵活、支持任意条件
管理 N 个资源                 信号量          内置计数功能
一个线程通知另一个线程        信号量（初值0）  简单、无需锁
多个线程需要同步              屏障            所有线程到达后继续
读多写少                      读写锁          并发读取
```

### 16.7.2 同步问题的解题框架

1. **识别共享资源**：确定哪些资源需要保护
2. **识别约束条件**：确定互斥、同步、资源数量等约束
3. **选择同步原语**：根据约束选择合适的原语
4. **设计解决方案**：使用信号量、条件变量或锁
5. **验证正确性**：检查互斥、进步、有限等待

---

## 16.8 面试高频考点

**Q1：生产者-消费者问题中 P 操作的顺序为什么重要？**

必须先 P(&empty) 再 P(&mutex)。如果反过来，缓冲区满时生产者持有 mutex 等待 empty，消费者需要 mutex 才能消费 → 死锁。

**Q2：读者-写者问题中读者优先和写者优先的区别？**

读者优先：只要有读者就允许读取，写者可能饥饿。写者优先：写者到达后阻止新读者，避免写者饥饿。

**Q3：哲学家就餐问题的死锁原因？**

循环等待——每个哲学家都持有左叉子等待右叉子，形成环形依赖。解决方案：打破循环（非对称方案、限制就餐人数）。

**Q4：信号量和条件变量的区别？**

信号量是一个整数 + P/V 操作，适合资源计数和条件同步。条件变量 + 锁适合等待任意条件。信号量的 V() 可以由任何线程调用，条件变量的 signal() 通常由修改条件的线程调用。

**Q5：如何证明一个同步方案的正确性？**

需要证明三个性质：互斥（临界区互斥访问）、进步（无死锁）、有限等待（无饥饿）。通常使用不变式（invariant）和反证法。

**Q6：生产者-消费者问题的信号量实现中，为什么需要三个信号量？**

`mutex` 保护缓冲区访问（互斥），`empty` 记录空槽位数量（条件同步），`full` 记录满槽位数量（条件同步）。三个信号量分别解决三个不同的问题：互斥、生产者等待、消费者等待。

---

## 16.9 手算练习

### 练习 1：生产者-消费者问题

**题目**：3 个生产者和 2 个消费者共享一个大小为 5 的缓冲区。使用信号量实现。计算在以下操作序列下的信号量值变化：

```
操作序列：
1. 生产者 1 生产
2. 生产者 2 生产
3. 消费者 1 消费
4. 生产者 3 生产
5. 生产者 1 生产
6. 消费者 2 消费
```

**解答**：

初始状态：empty=5, full=0, mutex=1

```
操作              empty  full  mutex  说明
初始              5      0     1      缓冲区空
生产者 1 生产     4      1     1      empty--, full++
生产者 2 生产     3      2     1      empty--, full++
消费者 1 消费     4      1     1      empty++, full--
生产者 3 生产     3      2     1      empty--, full++
生产者 1 生产     2      3     1      empty--, full++
消费者 2 消费     3      2     1      empty++, full--
```

### 练习 2：读者-写者问题

**题目**：3 个读者和 1 个写者访问共享数据。使用读者优先读写锁。计算在以下操作序列下的 reader_count 和 write_lock 状态：

```
操作序列：
1. 读者 1 到达
2. 读者 2 到达
3. 写者 1 到达
4. 读者 3 到达
5. 读者 1 离开
6. 读者 2 离开
```

**解答**：

```
操作              reader_count  write_lock  说明
初始              0             未锁定
读者 1 到达       1             锁定        第一个读者锁住写锁
读者 2 到达       2             锁定        读者并发
写者 1 到达       2             锁定        写者等待
读者 3 到达       3             锁定        读者优先，新读者直接加入
读者 1 离开       2             锁定        还有读者
读者 2 离开       1             锁定        还有读者
读者 3 离开       0             未锁定      最后一个读者释放写锁
写者 1 获取       0             锁定        写者终于可以写入
```

### 练习 3：哲学家就餐问题

**题目**：5 个哲学家使用非对称方案（哲学家 0 先拿右叉子）。计算在以下操作序列下的叉子状态：

```
操作序列：
1. 哲学家 0 尝试进餐
2. 哲学家 1 尝试进餐
3. 哲学家 2 尝试进餐
4. 哲学家 0 完成进餐
5. 哲学家 3 尝试进餐
```

**解答**：

```
操作              叉0  叉1  叉2  叉3  叉4  说明
初始              空闲 空闲 空闲 空闲 空闲
哲学家 0 尝试     空闲 拿起 空闲 空闲 空闲  先拿右叉子（叉1）
哲学家 1 尝试     空闲 拿起 拿起 空闲 空闲  先拿左叉子（叉1）——等待！
哲学家 2 尝试     空闲 拿起 拿起 拿起 空闲  先拿左叉子（叉2）
哲学家 0 完成     空闲 空闲 拿起 拿起 空闲  释放叉0 和叉1
哲学家 1 获取     空闲 拿起 拿起 拿起 空闲  现在可以拿叉1 了
哲学家 3 尝试     空闲 拿起 拿起 拿起 拿起  先拿左叉子（叉3）
```

---

## 16.10 同步问题的性能分析

### 16.10.1 生产者-消费者性能

```
缓冲区大小    生产者数    消费者数    吞吐量（ops/s）
1            1           1           1000
10           1           1           5000
100          1           1           10000
10           3           3           15000
100          3           3           30000
```

**关键洞察**：缓冲区大小对性能有显著影响。太小的缓冲区导致频繁的生产者-消费者切换，太大的缓冲区浪费内存。

### 16.10.2 读者-写者性能

```
读写比例        互斥锁吞吐量    读写锁吞吐量    性能提升
99% 读 1% 写    1x              50x             50 倍
90% 读 10% 写   1x              10x             10 倍
50% 读 50% 写   1x              1x              无提升
```

**关键洞察**：读写锁只有在读多写少的场景下才有显著优势。

### 16.10.3 哲学家就餐性能

```
方案              并行度    吞吐量
非对称方案        5         最高
限制就餐人数      4         高
互斥锁            1         低
```

---

## 16.11 同步问题的常见错误

### 16.11.1 死锁

```c
// 死锁示例：两个线程以不同顺序获取两把锁
pthread_mutex_t lock1, lock2;

void *thread1(void *arg) {
    pthread_mutex_lock(&lock1);
    sleep(1);
    pthread_mutex_lock(&lock2);  // 死锁！
    // ...
}

void *thread2(void *arg) {
    pthread_mutex_lock(&lock2);
    sleep(1);
    pthread_mutex_lock(&lock1);  // 死锁！
    // ...
}
```

**解决方案**：所有线程以相同顺序获取锁。

### 16.11.2 饥饿

```c
// 饥饿示例：读者优先读写锁中写者饥饿
// 如果不断有读者到达，写者永远无法获取锁
void reader() {
    reader_lock();
    // 读取
    reader_unlock();
}

void writer() {
    writer_lock();  // 可能永远等待！
    // 写入
    writer_unlock();
}
```

**解决方案**：使用写者优先读写锁或公平读写锁。

### 16.11.3 活锁

```c
// 活锁示例：两个线程互相谦让
void *thread1(void *arg) {
    while (1) {
        if (flag2) {
            flag1 = 0;
            sleep(1);
            flag1 = 1;
        } else {
            break;
        }
    }
}

void *thread2(void *arg) {
    while (1) {
        if (flag1) {
            flag2 = 0;
            sleep(1);
            flag2 = 1;
        } else {
            break;
        }
    }
}
```

**解决方案**：引入随机性或使用退避策略。

---

## 16.12 同步问题的调试技巧

### 16.12.1 死锁检测

```bash
# 使用 GDB 检测死锁
$ gdb ./program
(gdb) run
# 程序挂起后
(gdb) thread apply all bt  # 查看所有线程的调用栈
# 如果多个线程都在等待锁，可能是死锁
```

### 16.12.2 数据竞争检测

```bash
# 使用 ThreadSanitizer
$ gcc -fsanitize=thread -g program.c -lpthread
$ ./a.out
# 输出：WARNING: ThreadSanitizer: data race
```

### 16.12.3 死锁预防

```bash
# 使用 lockdep（Linux 内核）
# 在内核配置中启用 LOCKDEP
# lockdep 会自动检测锁顺序违反
```

---

## 16.13 同步问题的实际应用

### 16.13.1 生产者-消费者在操作系统中的应用

1. **管道（Pipe）**：xv6 pipe 是生产者-消费者的典型实现
2. **消息队列**：进程间通信的消息缓冲区
3. **线程池**：任务队列 + 工作线程
4. **缓冲区 I/O**：磁盘读写缓冲区

### 16.13.2 读者-写者在数据库中的应用

1. **数据库锁**：行锁、表锁、页锁
2. **缓存系统**：Redis、Memcached 的读写锁
3. **配置管理**：读多写少的配置文件

### 16.13.3 哲学家就餐在分布式系统中的应用

1. **资源分配**：多个进程竞争多个资源
2. **死锁避免**：银行家算法、资源排序
3. **分布式锁**：Redis 分布式锁、Zookeeper 锁

---

## 16.14 推荐实践

1. **优先使用高级同步原语**：互斥锁、条件变量、信号量，而不是手动实现
2. **保持临界区短**：减少锁竞争，提高并发性
3. **避免嵌套锁**：如果必须使用，确保固定顺序
4. **使用调试工具**：ThreadSanitizer、Helgrind、lockdep
5. **代码审查**：重点关注锁的获取顺序、临界区边界、异常处理

---

## 16.15 同步问题的面试准备

### 16.15.1 常见面试题

1. **手写生产者-消费者**：使用信号量或条件变量实现
2. **手写读者-写者**：实现读者优先或写者优先版本
3. **手写哲学家就餐**：实现避免死锁的方案
4. **分析同步方案的正确性**：证明互斥、进步、有限等待
5. **选择同步原语**：根据场景选择合适的原语

### 16.15.2 面试技巧

1. **画图**：画出线程/进程的执行时间线
2. **逐步分析**：一步一步分析信号量/锁的状态变化
3. **考虑边界情况**：缓冲区满/空、所有哲学家同时进餐
4. **验证正确性**：检查互斥、进步、有限等待
5. **讨论性能**：分析锁竞争、上下文切换开销

---

## 16.16 推荐阅读

### 16.16.1 经典教材

- **OSTEP** Chapter 30-31：条件变量和信号量详解
- **Operating System Concepts** Chapter 5.7-5.9：经典同步问题
- **The Little Book of Semaphores**：信号量练习题集

### 16.16.2 在线资源

- [xv6-riscv kernel/pipe.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/pipe.c)：pipe 实现
- [POSIX Threads Programming](https://hpc-tutorials.llnl.gov/posix/)：Pthreads 教程
- [The Little Book of Semaphores](https://greenteapress.com/wp/semaphores/)：信号量练习题

### 16.16.3 实践项目

1. **实现生产者-消费者**：使用信号量和条件变量分别实现
2. **实现读者-写者**：实现读者优先和写者优先版本
3. **实现哲学家就餐**：实现多种解决方案
4. **实现线程池**：使用生产者-消费者模式
5. **实现屏障**：使用条件变量实现
