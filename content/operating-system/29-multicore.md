---
title: "Chapter 29: 多核与并发扩展性"
description: "深入理解 SMP/NUMA 架构与缓存一致性协议，掌握 Amdahl 定律与锁竞争分析"
updated: "2026-06-11"
---

# Chapter 29: 多核与并发扩展性

> **本章目标**：
> - 深入理解 SMP 与 NUMA 多核架构的区别与适用场景
> - 掌握 MESI/MOESI 缓存一致性协议的工作原理
> - 理解 False Sharing 问题及其解决方案
> - 掌握 Amdahl 定律，分析并行加速的理论上限
> - 理解锁竞争与可扩展性瓶颈
> - 掌握细粒度锁、锁分拆、锁分层等优化技术
> - 理解无锁数据结构的设计思想与 ABA 问题
> - 深入理解 RCU 机制及其在 Linux 内核中的应用

---

## 29.1 多核架构

### 29.1.1 SMP 对称多处理

SMP（Symmetric Multi-Processing，对称多处理）是最基本的多核架构。在 SMP 系统中，所有处理器核心地位对等，共享同一物理内存和 I/O 总线。

```
┌─────────────────────────────────────────────────────┐
│                   共享内存 (DRAM)                      │
│         ┌─────────────────────────────┐              │
│         │   物理地址空间 (统一编址)      │              │
│         └─────────────────────────────┘              │
└────────────┬──────────────┬──────────────┬───────────┘
             │              │              │
        ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
        │  CPU 0  │    │  CPU 1  │    │  CPU 2  │
        │ L1:32KB │    │ L1:32KB │    │ L1:32KB │
        │ L2:256KB│    │ L2:256KB│    │ L2:256KB│
        └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │
        ┌────┴──────────────┴──────────────┴────┐
        │          共享 L3 Cache (8MB)            │
        └───────────────────────────────────────┘
```

**SMP 的关键特征**：

1. **对等性**：每个 CPU 地位相等，可以执行操作系统代码、处理中断、访问任意内存
2. **共享内存**：所有 CPU 访问同一物理地址空间，任何 CPU 都可以读写任何内存位置
3. **统一总线**：CPU 通过共享总线或交叉开关访问内存
4. **硬件透明**：程序员看到的是单一地址空间，硬件负责一致性

**SMP 的优势**：
- 编程模型简单——共享地址空间，线程可以直接通信
- 负载均衡容易——任何 CPU 都可以执行任何任务
- 操作系统实现相对简单——调度器可以在任意 CPU 上运行任意进程

**SMP 的局限**：
- 总线带宽成为瓶颈——CPU 数量增加时，总线竞争加剧
- 内存访问延迟均匀但不最优——所有 CPU 访问内存的延迟相同
- 扩展性有限——通常不超过 8-16 个 CPU

```c
// SMP 系统中的典型编程模型
// 所有线程共享同一地址空间
int shared_counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *worker(void *arg) {
    for (int i = 0; i < 1000000; i++) {
        pthread_mutex_lock(&mutex);
        shared_counter++;     // 任何 CPU 上的线程都可以访问
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}
```

### 29.1.2 NUMA 非统一内存访问

当 CPU 数量继续增加时，SMP 的总线瓶颈促使我们采用 NUMA（Non-Uniform Memory Access，非统一内存访问）架构。NUMA 将系统分为多个"节点"，每个节点有自己的 CPU 和本地内存。

```
┌──────────────────────┐     ┌──────────────────────┐
│       Node 0         │     │       Node 1         │
│  ┌──────┐  ┌──────┐  │     │  ┌──────┐  ┌──────┐  │
│  │CPU 0 │  │CPU 1 │  │     │  │CPU 2 │  │CPU 3 │  │
│  │L1/L2 │  │L1/L2 │  │     │  │L1/L2 │  │L1/L2 │  │
│  └──┬───┘  └──┬───┘  │     │  └──┬───┘  └──┬───┘  │
│     │         │      │     │     │         │      │
│  ┌──┴─────────┴──┐   │     │  ┌──┴─────────┴──┐   │
│  │  本地内存 32GB  │   │     │  │  本地内存 32GB  │   │
│  └───────────────┘   │     │  └───────────────┘   │
└──────────┬───────────┘     └──────────┬───────────┘
           │                            │
           └──────── QPI/UPI ───────────┘
                 (互联总线)
```

**NUMA 的关键特征**：

| 特性 | 说明 |
|------|------|
| 本地访问 | CPU 访问本节点内存，延迟低（~100ns） |
| 远程访问 | CPU 通过互联总线访问其他节点内存，延迟高（~150-300ns） |
| 节点内带宽高 | 本节点 CPU 与内存之间带宽充足 |
| 节点间带宽低 | 互联总线带宽有限，成为瓶颈 |

**NUMA 感知的内存分配**：

```c
// Linux NUMA 感知内存分配
#include <numa.h>
#include <numaif.h>

// 在当前 CPU 所在节点分配内存（本地分配）
void *local_buf = numa_alloc_onnode(size, numa_node_of_cpu(sched_getcpu()));

// 在指定节点分配内存
void *node1_buf = numa_alloc_onnode(size, 1);

// 获取当前 CPU 所在的 NUMA 节点
int node = numa_node_of_cpu(sched_getcpu());

// 设置内存分配策略
// MPOL_DEFAULT: 在当前节点分配
// MPOL_BIND: 只在指定节点分配
// MPOL_INTERLEAVE: 在多个节点间交替分配
struct bitmask *nodes = numa_allocate_nodemask();
numa_bitmask_setbit(nodes, 0);
numa_bitmask_setbit(nodes, 1);
mbind(buf, size, MPOL_INTERLEAVE, nodes->maskp, nodes->size + 1, 0);
```

**NUMA 架构的设计权衡**：

```
                    扩展性
                      ▲
                      │
         NUMA ████████│████████████
                      │
           SMP ███████│████
                      │
          UMA ████    │
                      │
                      └──────────────► CPU 数量
                      4    16    64   128
```

- **UMA（统一内存访问）**：所有 CPU 访问内存延迟相同，适合小规模系统
- **SMP**：共享总线，中等规模，编程简单
- **NUMA**：大规模扩展，但需要 NUMA 感知的编程

### 29.1.3 缓存一致性协议 MESI

在多核系统中，每个 CPU 核心都有自己的私有缓存。当多个 CPU 缓存同一内存地址时，必须保证它们看到的值是一致的——这就是**缓存一致性**（Cache Coherence）问题。

**为什么需要缓存一致性？**

```
时间线：
  CPU 0                    CPU 1
    │                        │
    ▼                        ▼
读 x=0 (缓存未命中)       读 x=0 (缓存未命中)
缓存 x=0                  缓存 x=0
    │                        │
    ▼                        ▼
写 x=1 (只写入缓存)       读 x=? (应该读到什么?)
    │                        │
    x=1 (CPU0缓存)          x=0 (CPU1缓存) ← 不一致！
```

**MESI 协议**是最经典的缓存一致性协议，每个缓存行有 4 个状态：

| 状态 | 含义 | 共享 | 脏数据 |
|------|------|------|--------|
| **M** (Modified) | 本 CPU 独占修改，与内存不一致 | 否 | 是 |
| **E** (Exclusive) | 本 CPU 独占，与内存一致 | 否 | 否 |
| **S** (Shared) | 多个 CPU 共享，与内存一致 | 是 | 否 |
| **I** (Invalid) | 缓存行无效 | - | - |

**MESI 状态转换图**：

```
                    本地读命中
              ┌──────────────────────┐
              │                      ▼
    ┌─────────┴──┐    本地读    ┌─────────┐
    │            │◄────────────│         │
    │  Invalid   │             │ Shared  │
    │    (I)     │             │   (S)   │
    │            │────────────►│         │
    └─────────┬──┘    总线读   └────┬────┘
              │                     │
              │ 本地写              │ 本地写
              ▼                     ▼ (其他CPU无副本时)
    ┌─────────┐              ┌──────────┐
    │         │    本地写     │          │
    │Modified │◄────────────│Exclusive │
    │   (M)   │              │    (E)   │
    │         │─────────────►│          │
    └─────────┘   总线读     └──────────┘
         │ (回写到内存后)
         ▼
    ┌─────────┐
    │Shared   │
    │   (S)   │
    └─────────┘
```

**MESI 协议的关键操作**：

```c
// 当 CPU 0 写入缓存行时，需要发送总线事务：

// 1. BusRd（总线读）：其他 CPU 读取该缓存行
//    - 如果状态是 M：回写到内存，转为 S
//    - 如果状态是 E：转为 S
//    - 如果状态是 S：保持 S
//    - 如果状态是 I：从内存读取，转为 S 或 E

// 2. BusRdX（总线读独占）：其他 CPU 要写入该缓存行
//    - 所有其他 CPU 的该缓存行转为 I（无效化）
//    - 请求者获得 M 状态

// 3. BusUpgr（总线升级）：CPU 要从 S 升级到 M
//    - 所有其他 CPU 的该缓存行转为 I
//    - 不需要从内存读取（因为数据一致）
```

**MESI 协议的实现细节**：

```c
// 伪代码：MESI 状态机
typedef enum { INVALID, SHARED, EXCLUSIVE, MODIFIED } mesi_state_t;

typedef struct {
    mesi_state_t state;
    uint64_t tag;
    uint8_t data[CACHE_LINE_SIZE];
} cache_line_t;

// CPU 读操作
void cache_read(cache_line_t *line, uint64_t addr) {
    if (line->state == INVALID) {
        // 缓存未命中，需要从内存或其他 CPU 获取
        bus_read(addr);  // 发送总线读请求
        // 检查其他 CPU 是否有该缓存行
        if (other_cpus_have_copy(addr)) {
            line->state = SHARED;
        } else {
            line->state = EXCLUSIVE;
        }
        line->data = memory_read(addr);
    }
    // 如果是 S 或 E 或 M 状态，直接读取（命中）
    return line->data;
}

// CPU 写操作
void cache_write(cache_line_t *line, uint64_t addr, uint8_t *data) {
    if (line->state == INVALID) {
        // 需要先获取独占权
        bus_read_exclusive(addr);  // BusRdX
        invalidate_other_copies(addr);
        line->state = MODIFIED;
    } else if (line->state == SHARED) {
        // 需要升级到独占
        bus_upgrade(addr);  // BusUpgr
        invalidate_other_copies(addr);
        line->state = MODIFIED;
    } else if (line->state == EXCLUSIVE) {
        // 直接升级到修改
        line->state = MODIFIED;
    }
    // M 状态下直接写入
    line->data = data;
}
```

### 29.1.4 MOESI 协议

MOESI 是 MESI 的扩展，增加了 **O**（Owned）状态，允许脏数据在 CPU 之间共享。

| 状态 | 含义 | 共享 | 脏数据 |
|------|------|------|--------|
| **M** (Modified) | 本 CPU 独占修改，与内存不一致 | 否 | 是 |
| **O** (Owned) | 本 CPU 持有最新数据，其他 CPU 可共享，与内存不一致 | 是 | 是 |
| **E** (Exclusive) | 本 CPU 独占，与内存一致 | 否 | 否 |
| **S** (Shared) | 多个 CPU 共享，与内存一致 | 是 | 否 |
| **I** (Invalid) | 缓存行无效 | - | - |

**MOESI 的优势**：当 CPU 0 有脏数据（M 状态），CPU 1 要读取时，CPU 0 可以直接将数据发送给 CPU 1，CPU 0 转为 O 状态，CPU 1 转为 S 状态。不需要先写回内存，减少了内存访问。

### 29.1.5 False Sharing 问题

False Sharing（伪共享）是多核编程中最隐蔽的性能杀手之一。当两个 CPU 访问不同的变量，但这些变量恰好在同一个缓存行中时，缓存一致性协议会导致不必要的缓存行失效。

**问题演示**：

```c
// 两个线程分别修改不同的计数器
struct {
    int counter_a;  // CPU 0 修改
    int counter_b;  // CPU 1 修改
} shared_data;

// 问题：counter_a 和 counter_b 在同一个缓存行中（64字节）
// CPU 0 修改 counter_a 时，会使 CPU 1 的缓存行失效
// CPU 1 修改 counter_b 时，会使 CPU 0 的缓存行失效
// 导致两个 CPU 不断互相"踢"对方的缓存行
```

**False Sharing 的缓存行竞争**：

```
缓存行 (64 字节):
┌─────────────────────────────────────────────────────────┐
│ counter_a (4B) │ counter_b (4B) │      padding (56B)    │
└─────────────────────────────────────────────────────────┘
     CPU 0 写        CPU 1 写

时间线：
  CPU 0                          CPU 1
    │                              │
    ▼                              │
写 counter_a → 缓存行状态 M        │
                ──────────────────► 缓存行状态 I (CPU1)
    │                              ▼
    │                    读 counter_b → 缓存未命中!
    │                    发送 BusRd → CPU 0 回写
    │                    缓存行状态 S
    │                              │
    ▼                              │
读 counter_a → 缓存未命中!         │
发送 BusRd → CPU 1 回写            │
    │                              │
    ... 无限循环 ...
```

**False Sharing 的性能影响**：

```c
// 测试 False Sharing 的影响
#include <pthread.h>
#include <time.h>

#define ITERATIONS 100000000

// 方案 1：有 False Sharing
struct {
    long counter_a __attribute__((aligned(64)));  // 注意：即使对齐，如果在同一结构体中
    long counter_b __attribute__((aligned(64)));  // 可能仍在同一缓存行
} shared_fs;

// 方案 2：消除 False Sharing（使用 padding）
struct {
    long counter_a;
    char padding[56];  // 填充到 64 字节
    long counter_b;
    char padding2[56];
} shared_no_fs;

// 或者使用对齐方式
struct {
    long counter_a __attribute__((aligned(64)));
    long counter_b __attribute__((aligned(64)));
} shared_aligned;
```

**False Sharing 的检测方法**：

```bash
# 使用 perf 工具检测 False Sharing
perf c2c record -a -- sleep 5
perf c2c report

# 输出示例：
#   Total records     : 1234567
#   False sharing     : 89.2%
#   True sharing      : 10.8%
#
#   Shared Data Cache Line Table
#   ─────────────────────────────
#   Line 0x7f1234567890  (counter_a, counter_b)
#     Hitm: 95%  (95% of accesses caused cache line bouncing)
```

### 29.1.6 解决 False Sharing 的 Padding 技术

```c
// 方法 1：手动 padding
struct padded_counter {
    long value;
    char padding[64 - sizeof(long)];  // 填充到缓存行大小
} __attribute__((aligned(64)));

struct {
    struct padded_counter counter_a;
    struct padded_counter counter_b;
} data;

// 方法 2：使用编译器属性
struct {
    long counter_a __attribute__((aligned(64)));
    long counter_b __attribute__((aligned(64)));
} data_aligned;

// 方法 3：C11 alignas
#include <stdalign.h>
struct {
    alignas(64) long counter_a;
    alignas(64) long counter_b;
} data_c11;

// 方法 4：Linux 内核的 ____cacheline_aligned
struct {
    long counter_a ____cacheline_aligned;
    long counter_b ____cacheline_aligned;
} data_kernel;
```

**实际性能对比**：

```c
// 基准测试
void benchmark_false_sharing() {
    struct timespec start, end;
    
    // 测试有 False Sharing 的版本
    clock_gettime(CLOCK_MONOTONIC, &start);
    // 两个线程分别修改 counter_a 和 counter_b
    pthread_create(&t1, NULL, worker_a, &shared_fs);
    pthread_create(&t2, NULL, worker_b, &shared_fs);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("False Sharing: %ld ns\n", elapsed_ns(start, end));
    
    // 测试无 False Sharing 的版本
    clock_gettime(CLOCK_MONOTONIC, &start);
    pthread_create(&t1, NULL, worker_a, &shared_aligned);
    pthread_create(&t2, NULL, worker_b, &shared_aligned);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("No False Sharing: %ld ns\n", elapsed_ns(start, end));
}

// 典型结果：
// False Sharing:  2,500,000,000 ns (2.5 秒)
// No False Sharing:   150,000,000 ns (0.15 秒)
// 性能差距: ~16 倍
```

---

## 29.2 并发扩展性挑战

### 29.2.1 Amdahl 定律

Amdahl 定律是并行计算中最基本的理论，它揭示了并行加速的理论上限。即使拥有无限多的处理器，程序的加速比也受限于串行部分的比例。

**Amdahl 定律公式**：

```
加速比 = 1 / (S + P/N)

其中：
  S = 程序中必须串行执行的比例 (0 ≤ S ≤ 1)
  P = 程序中可以并行执行的比例 (P = 1 - S)
  N = 处理器数量
```

**直观理解**：

```
程序执行时间：
┌─────────────────────────────────────────────┐
│ 串行部分 S │         并行部分 P              │
│  (不可并行) │  (可以分配到 N 个 CPU)          │
└─────────────────────────────────────────────┘

N=1 时：[████████████████████████████]  总时间 = S + P
N=2 时：[██████████████████]          总时间 = S + P/2
N=4 时：[██████████████]              总时间 = S + P/4
N=∞ 时：[████████████]                总时间 = S (无法再减少)
```

**数学推导**：

```
设程序总执行时间为 T_serial = T_s + T_p
其中 T_s 是串行部分时间，T_p 是并行部分时间

并行化后：T_parallel = T_s + T_p / N

加速比 Speedup = T_serial / T_parallel
               = (T_s + T_p) / (T_s + T_p / N)

令 S = T_s / (T_s + T_p) 为串行比例
令 P = T_p / (T_s + T_p) = 1 - S 为并行比例

Speedup = 1 / (S + P/N)
        = 1 / (S + (1-S)/N)
        = N / (N*S + 1 - S)
        = N / (1 + (N-1)*S)
```

**Amdahl 定律的实际影响**：

| 串行比例 S | N=4 | N=8 | N=16 | N=32 | N=64 | N=∞ |
|-----------|-----|-----|------|------|------|-----|
| 1% | 3.9 | 7.5 | 14.0 | 25.4 | 40.6 | 100 |
| 5% | 3.6 | 6.0 | 8.8 | 11.1 | 12.5 | 20 |
| 10% | 3.1 | 4.7 | 6.4 | 7.6 | 8.2 | 10 |
| 20% | 2.5 | 3.3 | 4.2 | 4.7 | 4.9 | 5 |
| 50% | 1.6 | 1.8 | 1.9 | 1.9 | 2.0 | 2 |

**关键洞察**：

1. **串行比例是瓶颈**：即使只有 5% 的串行代码，加速比也限制在 20 倍
2. **收益递减**：随着 CPU 数量增加，加速比增长越来越慢
3. **理论上限**：当 N→∞ 时，Speedup = 1/S
4. **实际中 S 更大**：锁竞争、同步开销、内存分配等都会增加串行比例

```c
// Amdahl 定律计算器
double amdahl_speedup(double serial_fraction, int n) {
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / n);
}

// 示例：10% 串行代码
printf("N=4:  %.2f\n", amdahl_speedup(0.10, 4));   // 3.08
printf("N=8:  %.2f\n", amdahl_speedup(0.10, 8));   // 4.71
printf("N=16: %.2f\n", amdahl_speedup(0.10, 16));  // 6.40
printf("N=∞:  %.2f\n", amdahl_speedup(0.10, 1000000));  // 10.00
```

### 29.2.2 Gustafson 定律

Amdahl 定律假设问题规模固定，但实际中我们通常会随着 CPU 增加而处理更大规模的问题。Gustafson 定律提供了另一种视角：

```
加速比 = S + P * N = N - S * (N - 1)

其中：
  S = 串行比例
  P = 并行比例
  N = 处理器数量
```

**Gustafson 定律的意义**：
- 问题规模可以随 CPU 数量增加
- 串行部分的绝对时间保持不变
- 加速比可以线性增长

### 29.2.3 锁竞争 Lock Contention

锁竞争是多核系统中最常见的可扩展性瓶颈。当多个 CPU 争抢同一把锁时，大量时间被浪费在等待上。

**锁竞争的层次**：

```
锁竞争严重程度：
┌─────────────────────────────────────────────┐
│  无竞争 (No Contention)                      │
│  - 只有一个线程获取锁                         │
│  - 开销最小（~10ns）                          │
├─────────────────────────────────────────────┤
│  低竞争 (Low Contention)                     │
│  - 偶尔有多个线程同时获取锁                   │
│  - 等待时间短                                │
├─────────────────────────────────────────────┤
│  中等竞争 (Medium Contention)                │
│  - 经常有多个线程等待锁                       │
│  - 等待时间中等                              │
├─────────────────────────────────────────────┤
│  高竞争 (High Contention)                    │
│  - 大量线程同时争抢锁                         │
│  - 等待时间长，CPU 空转                       │
└─────────────────────────────────────────────┘
```

**锁竞争的性能影响**：

```c
// 测试锁竞争
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 8
#define ITERATIONS 10000000

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
long shared_counter = 0;

void *worker(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        pthread_mutex_lock(&mutex);
        shared_counter++;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    printf("Total time: %ld ms\n", elapsed_ms(start, end));
    printf("Effective throughput: %f ops/ms\n", 
           (double)(NUM_THREADS * ITERATIONS) / elapsed_ms(start, end));
    
    return 0;
}

// 结果示例（8 核 CPU）：
// 线程数  总时间    有效吞吐量
// 1       100ms     100,000 ops/ms
// 2       180ms     111,111 ops/ms  (↑11%)
// 4       350ms     114,286 ops/ms  (↑3%)
// 8       700ms     114,286 ops/ms  (→0%)
// 结论：吞吐量随线程数增加而饱和
```

**锁竞争的开销分析**：

```
锁操作的真实开销：
┌─────────────────────────────────────────────┐
│ 1. 原子操作（CAS）       ~10-20ns           │
│ 2. 缓存行传输            ~50-100ns          │
│ 3. 上下文切换（如果睡眠） ~1000-5000ns       │
│ 4. 总线仲裁              ~20-50ns           │
└─────────────────────────────────────────────┘

实际锁操作耗时：
- 无竞争：~20ns
- 低竞争：~50ns
- 中等竞争：~200ns
- 高竞争：~1000ns+
```

### 29.2.4 可扩展性瓶颈分析

**常见的可扩展性瓶颈**：

```c
// 1. 全局锁
pthread_mutex_t global_lock;  // 所有操作都竞争同一把锁

// 2. 共享数据结构
struct {
    pthread_mutex_t lock;
    int data[1000];
} shared_array;  // 所有线程访问同一数组

// 3. 内存分配器
void *ptr = malloc(size);  // malloc 内部有全局锁

// 4. 日志系统
log_message("...");  // 日志文件有全局锁
```

**可扩展性度量**：

```c
// 可扩展性度量函数
typedef struct {
    int num_threads;
    double throughput;  // ops/ms
    double efficiency;  // 相对于单线程的效率
} scalability_result_t;

scalability_result_t measure_scalability(int num_threads) {
    scalability_result_t result;
    result.num_threads = num_threads;
    
    // 运行基准测试
    result.throughput = run_benchmark(num_threads);
    
    // 计算效率 = 实际吞吐量 / 理想吞吐量
    double ideal_throughput = single_thread_throughput * num_threads;
    result.efficiency = result.throughput / ideal_throughput;
    
    return result;
}

// 理想的可扩展性：
// 效率 = 100% (无论多少线程，吞吐量线性增长)

// 实际的可扩展性：
// 效率随线程数增加而下降
// 线程数    效率
// 1         100%
// 2         95%
// 4         85%
// 8         70%
// 16        50%
```

---

## 29.3 细粒度锁

### 29.3.1 锁分拆 Lock Splitting

锁分拆是将一个保护多个独立资源的锁，拆分成多个锁，每个锁保护一个资源。这样不同资源的操作可以并行进行。

**锁分拆前**：

```c
// 问题：一个锁保护所有账户
pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;
struct account {
    int id;
    double balance;
} accounts[1000];

void transfer(int from_id, int to_id, double amount) {
    pthread_mutex_lock(&global_lock);  // 所有转账操作竞争同一把锁
    accounts[from_id].balance -= amount;
    accounts[to_id].balance += amount;
    pthread_mutex_unlock(&global_lock);
}
```

**锁分拆后**：

```c
// 解决方案：每个账户一把锁
pthread_mutex_t account_locks[1000];
struct account {
    int id;
    double balance;
} accounts[1000];

void transfer(int from_id, int to_id, double amount) {
    // 注意：需要按顺序获取锁，避免死锁
    int first = (from_id < to_id) ? from_id : to_id;
    int second = (from_id < to_id) ? to_id : from_id;
    
    pthread_mutex_lock(&account_locks[first]);
    pthread_mutex_lock(&account_locks[second]);
    
    accounts[from_id].balance -= amount;
    accounts[to_id].balance += amount;
    
    pthread_mutex_unlock(&account_locks[second]);
    pthread_mutex_unlock(&account_locks[first]);
}
```

**锁分拆的适用场景**：
- 多个独立资源被同一把锁保护
- 不同线程访问不同的资源
- 资源之间没有依赖关系

### 29.3.2 锁分层 Lock Striping

锁分层是锁分拆的变体，将资源分成多个"条带"（stripe），每个条带一把锁。适用于哈希表等数据结构。

**锁分层的实现**：

```c
#define NUM_STRIPES 16
#define STRIPE_MASK (NUM_STRIPES - 1)

typedef struct {
    pthread_mutex_t locks[NUM_STRIPES];
    hash_entry_t *buckets[1024];
} striped_hashmap_t;

void hashmap_init(striped_hashmap_t *map) {
    for (int i = 0; i < NUM_STRIPES; i++) {
        pthread_mutex_init(&map->locks[i], NULL);
    }
}

// 获取条带锁
int get_stripe(uint64_t hash) {
    return hash & STRIPE_MASK;
}

void hashmap_put(striped_hashmap_t *map, uint64_t key, void *value) {
    uint64_t hash = hash_function(key);
    int stripe = get_stripe(hash);
    int bucket = hash & 1023;
    
    pthread_mutex_lock(&map->locks[stripe]);  // 只锁对应的条带
    // 插入操作
    hash_entry_t *entry = malloc(sizeof(hash_entry_t));
    entry->key = key;
    entry->value = value;
    entry->next = map->buckets[bucket];
    map->buckets[bucket] = entry;
    pthread_mutex_unlock(&map->locks[stripe]);
}

void *hashmap_get(striped_hashmap_t *map, uint64_t key) {
    uint64_t hash = hash_function(key);
    int stripe = get_stripe(hash);
    int bucket = hash & 1023;
    
    pthread_mutex_lock(&map->locks[stripe]);
    hash_entry_t *entry = map->buckets[bucket];
    while (entry) {
        if (entry->key == key) {
            void *value = entry->value;
            pthread_mutex_unlock(&map->locks[stripe]);
            return value;
        }
        entry = entry->next;
    }
    pthread_mutex_unlock(&map->locks[stripe]);
    return NULL;
}
```

**锁分层的优势**：
- 并发度 = 条带数量
- 内存开销 = 条带数量 × 锁大小
- 适用于哈希表、数组等数据结构

### 29.3.3 每 CPU 数据结构

每 CPU 数据结构（Per-CPU Data Structure）是消除锁竞争的终极方案：每个 CPU 核心有自己的私有数据，完全不需要锁。

```c
// 每 CPU 计数器
#define MAX_CPUS 64

typedef struct {
    long value;
    char padding[64 - sizeof(long)];  // 避免 False Sharing
} __attribute__((aligned(64))) per_cpu_counter_t;

typedef struct {
    per_cpu_counter_t counters[MAX_CPUS];
} distributed_counter_t;

void init_counter(distributed_counter_t *counter) {
    for (int i = 0; i < MAX_CPUS; i++) {
        counter->counters[i].value = 0;
    }
}

// 无锁递增
void increment(distributed_counter_t *counter) {
    int cpu = sched_getcpu();  // 获取当前 CPU 编号
    counter->counters[cpu].value++;  // 无需锁！
}

// 获取总和（需要遍历所有 CPU）
long get_total(distributed_counter_t *counter) {
    long total = 0;
    for (int i = 0; i < MAX_CPUS; i++) {
        total += counter->counters[i].value;
    }
    return total;
}
```

**每 CPU 数据结构的优势**：
- 完全无锁——每个 CPU 只访问自己的数据
- 极高并发度——CPU 数量个并发
- 缓存友好——数据在本地缓存中

**每 CPU 数据结构的劣势**：
- 内存开销大——每个 CPU 一份数据
- 查询需要遍历——获取全局视图需要遍历所有 CPU
- CPU 迁移问题——线程可能在不同 CPU 之间迁移

### 29.3.4 Linux per-CPU 变量

Linux 内核广泛使用 per-CPU 变量来避免锁竞争：

```c
// Linux 内核 per-CPU 变量
#include <linux/percpu.h>

// 定义 per-CPU 变量
DEFINE_PER_CPU(long, counter);

// 使用 per-CPU 变量
void increment_counter(void) {
    // 禁用抢占，确保当前 CPU 不会改变
    preempt_disable();
    
    this_cpu_inc(counter);  // 递增当前 CPU 的计数器
    
    preempt_enable();
}

// 获取总和
long get_counter_total(void) {
    long total = 0;
    int cpu;
    
    for_each_possible_cpu(cpu) {
        total += per_cpu(counter, cpu);
    }
    return total;
}

// per-CPU 变量的内存布局
// ┌─────────────┐
// │   CPU 0     │  counter @ 0x1000
// │   CPU 1     │  counter @ 0x1000 + PER_CPU_OFFSET(1)
// │   CPU 2     │  counter @ 0x1000 + PER_CPU_OFFSET(2)
// │   ...       │
// └─────────────┘
// 每个 CPU 访问的是不同的内存地址！
```

---

## 29.4 无锁数据结构

### 29.4.1 无锁队列 CAS 实现

无锁数据结构使用原子操作（如 CAS）代替锁，实现线程安全的并发访问。

**CAS（Compare-And-Swap）操作**：

```c
// CAS 原语
bool CAS(int *addr, int expected, int new_value) {
    // 原子操作：如果 *addr == expected，则 *addr = new_value，返回 true
    // 否则不修改，返回 false
    return __sync_bool_compare_and_swap(addr, expected, new_value);
}

// 或使用 C11 原子操作
bool cas_c11(_Atomic int *addr, int expected, int new_value) {
    return atomic_compare_exchange_strong(addr, &expected, new_value);
}
```

**无锁队列（Michael-Scott 队列）**：

```c
#include <stdatomic.h>
#include <stdlib.h>

typedef struct node {
    void *data;
    _Atomic(struct node *) next;
} node_t;

typedef struct {
    _Atomic(node_t *) head;
    _Atomic(node_t *) tail;
} lock_free_queue_t;

void queue_init(lock_free_queue_t *q) {
    node_t *dummy = malloc(sizeof(node_t));
    dummy->data = NULL;
    atomic_store(&dummy->next, NULL);
    atomic_store(&q->head, dummy);
    atomic_store(&q->tail, dummy);
}

void enqueue(lock_free_queue_t *q, void *data) {
    node_t *new_node = malloc(sizeof(node_t));
    new_node->data = data;
    atomic_store(&new_node->next, NULL);
    
    while (1) {
        node_t *tail = atomic_load(&q->tail);
        node_t *next = atomic_load(&tail->next);
        
        // 检查 tail 是否仍然是尾节点
        if (tail == atomic_load(&q->tail)) {
            if (next == NULL) {
                // 尝试将新节点链接到尾节点
                if (atomic_compare_exchange_strong(&tail->next, &next, new_node)) {
                    break;  // 成功
                }
            } else {
                // tail 没有指向真正的尾节点，帮助移动 tail
                atomic_compare_exchange_strong(&q->tail, &tail, next);
            }
        }
    }
    
    // 尝试移动 tail 指针
    atomic_compare_exchange_strong(&q->tail, &atomic_load(&q->tail), new_node);
}

void *dequeue(lock_free_queue_t *q) {
    while (1) {
        node_t *head = atomic_load(&q->head);
        node_t *tail = atomic_load(&q->tail);
        node_t *next = atomic_load(&head->next);
        
        if (head == atomic_load(&q->head)) {
            if (head == tail) {
                if (next == NULL) {
                    return NULL;  // 队列为空
                }
                // tail 落后了，帮助移动
                atomic_compare_exchange_strong(&q->tail, &tail, next);
            } else {
                void *data = next->data;
                if (atomic_compare_exchange_strong(&q->head, &head, next)) {
                    free(head);  // 释放旧的头节点
                    return data;
                }
            }
        }
    }
}
```

### 29.4.2 无锁栈

无锁栈使用 CAS 操作实现后进先出（LIFO）的数据结构：

```c
typedef struct stack_node {
    void *data;
    struct stack_node *next;
} stack_node_t;

typedef struct {
    _Atomic(stack_node_t *) top;
} lock_free_stack_t;

void stack_init(lock_free_stack_t *s) {
    atomic_store(&s->top, NULL);
}

void push(lock_free_stack_t *s, void *data) {
    stack_node_t *new_node = malloc(sizeof(stack_node_t));
    new_node->data = data;
    
    while (1) {
        stack_node_t *old_top = atomic_load(&s->top);
        new_node->next = old_top;
        
        if (atomic_compare_exchange_strong(&s->top, &old_top, new_node)) {
            break;  // 成功
        }
        // CAS 失败，重试
    }
}

void *pop(lock_free_stack_t *s) {
    while (1) {
        stack_node_t *old_top = atomic_load(&s->top);
        
        if (old_top == NULL) {
            return NULL;  // 栈为空
        }
        
        stack_node_t *new_top = old_top->next;
        
        if (atomic_compare_exchange_strong(&s->top, &old_top, new_top)) {
            void *data = old_top->data;
            // 注意：不能立即 free(old_top)，可能有其他线程正在读取
            // 需要使用 Hazard Pointer 或 RCU 来安全回收
            return data;
        }
        // CAS 失败，重试
    }
}
```

### 29.4.3 ABA 问题

ABA 问题是无锁数据结构中最经典的陷阱。当一个值从 A 变为 B 再变回 A 时，CAS 操作会误认为值没有变化。

**ABA 问题的演示**：

```
初始状态：栈顶 -> A -> B -> C

线程 1 (pop)：
  读取 top = A
  读取 next = B
  准备 CAS(top, A, B)
  // 被抢占...

线程 2 (pop)：
  读取 top = A
  CAS(top, A, B) → 成功
  free(A)  // A 被释放

线程 2 (push)：
  分配新节点（恰好复用了 A 的地址）
  push(A')  // A' 的地址与 A 相同
  栈顶 -> A' -> C

线程 1 恢复：
  CAS(top, A, B)
  // A 的地址相同，CAS 成功！
  // 但此时 top 应该是 A'，不是 B
  // 栈被破坏：top -> B，但 A' 和 C 丢失了
```

**ABA 问题的解决方案**：

**方案 1：Tagged Pointer（带标记的指针）**

```c
// 使用版本号避免 ABA 问题
typedef struct {
    stack_node_t *ptr;
    uint64_t tag;  // 每次修改递增
} tagged_ptr_t;

typedef struct {
    _Atomic(tagged_ptr_t) top;
} aba_safe_stack_t;

void push(aba_safe_stack_t *s, void *data) {
    stack_node_t *new_node = malloc(sizeof(stack_node_t));
    new_node->data = data;
    
    tagged_ptr_t old_top, new_top;
    while (1) {
        old_top = atomic_load(&s->top);
        new_node->next = old_top.ptr;
        new_top.ptr = new_node;
        new_top.tag = old_top.tag + 1;  // 递增版本号
        
        if (atomic_compare_exchange_strong(&s->top, &old_top, new_top)) {
            break;
        }
    }
}

void *pop(aba_safe_stack_t *s) {
    tagged_ptr_t old_top, new_top;
    while (1) {
        old_top = atomic_load(&s->top);
        
        if (old_top.ptr == NULL) {
            return NULL;
        }
        
        new_top.ptr = old_top.ptr->next;
        new_top.tag = old_top.tag + 1;  // 递增版本号
        
        if (atomic_compare_exchange_strong(&s->top, &old_top, new_top)) {
            void *data = old_top.ptr->data;
            return data;
        }
    }
}
```

**方案 2：Hazard Pointer**

```c
// Hazard Pointer：保护正在访问的指针
#define MAX_THREADS 64

typedef struct {
    _Atomic(void *) hazard[MAX_THREADS];
} hazard_pointer_t;

// 声明危险指针
void declare_hazard(hazard_pointer_t *hp, int thread_id, void *ptr) {
    atomic_store(&hp->hazard[thread_id], ptr);
}

// 清除危险指针
void clear_hazard(hazard_pointer_t *hp, int thread_id) {
    atomic_store(&hp->hazard[thread_id], NULL);
}

// 检查指针是否被其他线程保护
bool is_hazardous(hazard_pointer_t *hp, void *ptr) {
    for (int i = 0; i < MAX_THREADS; i++) {
        if (atomic_load(&hp->hazard[i]) == ptr) {
            return true;
        }
    }
    return false;
}
```

### 29.4.4 内存屏障 Memory Barriers

内存屏障用于控制指令的执行顺序，防止 CPU 和编译器对内存操作进行重排序。

**为什么需要内存屏障？**

```c
// 问题：CPU 可能重排序内存操作
int flag = 0;
int data = 0;

// 线程 1
data = 42;       // ① 写数据
flag = 1;        // ② 设置标志

// 线程 2
while (!flag);   // ③ 等待标志
print(data);     // ④ 读取数据，可能输出 0！

// 原因：CPU 可能将 ② 重排序到 ① 之前
// 或者线程 2 的 ④ 重排序到 ③ 之前
```

**内存屏障的类型**：

```c
// 编译器屏障：防止编译器重排序
__asm__ __volatile__("" ::: "memory");

// 完整内存屏障：防止 CPU 重排序
__sync_synchronize();

// C11 原子操作
atomic_thread_fence(memory_order_seq_cst);  // 顺序一致性
atomic_thread_fence(memory_order_acquire);   // 获取屏障
atomic_thread_fence(memory_order_release);   // 释放屏障
```

**内存屏障的使用示例**：

```c
#include <stdatomic.h>

int data = 0;
atomic_int flag = ATOMIC_VAR_INIT(0);

// 线程 1：写入数据
void writer(void) {
    data = 42;
    atomic_store_explicit(&flag, 1, memory_order_release);  // 释放屏障
}

// 线程 2：读取数据
void reader(void) {
    while (!atomic_load_explicit(&flag, memory_order_acquire));  // 获取屏障
    printf("data = %d\n", data);  // 保证输出 42
}
```

**x86 与 ARM 的内存模型差异**：

```
x86 (TSO - Total Store Order)：
- 加载-加载：不重排序
- 加载-存储：不重排序
- 存储-存储：不重排序
- 存储-加载：可能重排序！（Store Buffer）

ARM (弱内存模型)：
- 所有类型的内存操作都可能重排序
- 需要显式内存屏障

实际影响：
- x86 程序在 ARM 上可能出错
- 需要使用内存屏障保证可移植性
```

---

## 29.5 RCU (Read-Copy-Update)

### 29.5.1 RCU 设计思想

RCU（Read-Copy-Update）是 Linux 内核中最重要的同步机制之一，专为读多写少的场景设计。RCU 的核心思想是：

1. **读者无锁**：读者不需要获取任何锁，直接访问数据
2. **写者复制**：写者先复制一份数据，在副本上修改
3. **原子替换**：写者使用原子操作将指针指向新副本
4. **延迟回收**：旧数据在所有读者都离开临界区后才被释放

```
RCU 的基本流程：

初始状态：
  ptr -> [数据版本 1] (所有读者访问这个版本)

写者操作：
  1. 分配新内存
  2. 复制数据到新内存
  3. 修改新内存中的数据
  4. 原子替换指针：ptr -> [数据版本 2]
  5. 等待所有旧读者离开（Grace Period）
  6. 释放旧内存 [数据版本 1]

读者操作：
  1. 进入 RCU 临界区
  2. 读取指针：data = rcu_dereference(ptr)
  3. 访问数据
  4. 离开 RCU 临界区
```

### 29.5.2 RCU API

```c
// 读者 API
rcu_read_lock();          // 进入 RCU 临界区（实际上只是禁用抢占）
rcu_dereference(ptr);     // 安全地读取 RCU 保护的指针
rcu_read_unlock();        // 离开 RCU 临界区

// 写者 API
rcu_assign_pointer(ptr, new_ptr);  // 原子地更新 RCU 保护的指针
synchronize_rcu();        // 等待所有读者离开（阻塞）
call_rcu(&head, callback); // 异步等待，然后调用回调（非阻塞）

// 典型使用模式
struct my_data {
    int value;
    char name[32];
};

struct my_data *global_data;

// 读者
void reader(void) {
    struct my_data *data;
    
    rcu_read_lock();
    data = rcu_dereference(global_data);
    if (data) {
        printf("value: %d, name: %s\n", data->value, data->name);
    }
    rcu_read_unlock();
}

// 写者
void writer(int new_value, const char *new_name) {
    struct my_data *old_data, *new_data;
    
    // 1. 分配新内存
    new_data = kmalloc(sizeof(struct my_data), GFP_KERNEL);
    
    // 2. 复制旧数据
    old_data = rcu_dereference_protected(global_data, lockdep_is_held(&my_lock));
    if (old_data) {
        memcpy(new_data, old_data, sizeof(struct my_data));
    }
    
    // 3. 修改新数据
    new_data->value = new_value;
    strncpy(new_data->name, new_name, sizeof(new_data->name));
    
    // 4. 原子替换
    rcu_assign_pointer(global_data, new_data);
    
    // 5. 等待旧读者离开，然后释放旧数据
    synchronize_rcu();
    kfree(old_data);
}
```

### 29.5.3 Grace Period

Grace Period 是 RCU 的核心概念。它是指从指针被替换到所有可能访问旧数据的读者都离开临界区的时间段。

```
Grace Period 的时间线：

时间 →
────────────────────────────────────────────────────

读者 1: ┌──────────────────────────┐
        │ rcu_read_lock()          │ rcu_read_unlock()
        └──────────────────────────┘

读者 2:         ┌──────────────────────────────┐
                │ rcu_read_lock()              │ rcu_read_unlock()
                └──────────────────────────────┘

写者:           │ synchronize_rcu()
                │ (阻塞等待)
                ▼
                        ← Grace Period →
                │                           │
                │                           │ 返回
                │                           │
                                        kfree(old_data)

Grace Period 结束的条件：
- 所有在 synchronize_rcu() 调用前进入 RCU 临界区的读者都已离开
```

**Grace Period 的实现原理**：

```c
// Linux 内核中的 RCU 实现（简化版）
// 使用抢占计数和上下文切换来检测 Grace Period

// 每个 CPU 维护一个状态
struct rcu_data {
    int completed;  // 已完成的 Grace Period 编号
    int passed_quiesc;  // 是否经历了静默状态
};

// 静默状态（Quiescent State）：
// - 用户态执行
// - 上下文切换
// - 空闲循环
// 在这些状态下，CPU 不可能持有 RCU 读锁

// Grace Period 检测：
// 1. 等待所有 CPU 都经历至少一次静默状态
// 2. 如果所有 CPU 都经历了，Grace Period 结束
```

### 29.5.4 同步 RCU vs 异步 RCU

```c
// 同步 RCU：阻塞等待
void sync_example(void) {
    struct my_data *old = global_data;
    
    rcu_assign_pointer(global_data, new_data);
    
    synchronize_rcu();  // 阻塞，直到所有读者离开
    // synchronize_rcu() 返回后，可以安全释放 old
    kfree(old);
}

// 异步 RCU：注册回调，立即返回
void async_callback(struct rcu_head *head) {
    struct my_data *data = container_of(head, struct my_data, rcu);
    kfree(data);
}

void async_example(void) {
    struct my_data *old = global_data;
    
    rcu_assign_pointer(global_data, new_data);
    
    // 异步：注册回调，立即返回
    call_rcu(&old->rcu, async_callback);
    // 旧数据将在 Grace Period 后被释放
}
```

**同步 vs 异步的选择**：

| 特性 | synchronize_rcu | call_rcu |
|------|----------------|----------|
| 阻塞 | 是 | 否 |
| 延迟 | 高（等待 Grace Period） | 低（立即返回） |
| 内存开销 | 低 | 高（需要保存回调状态） |
| 适用场景 | 写操作不频繁 | 写操作频繁 |

### 29.5.5 RCU 在 Linux 内核中的应用

**场景 1：路由表查找**

```c
// Linux 网络路由表使用 RCU 保护
struct fib_table {
    struct hlist_node tb_hlist;
    int tb_id;
    // ...
};

// 读者：路由查找（高频操作）
int fib_lookup(struct net *net, const struct flowi4 *flp, struct fib_result *res) {
    struct fib_table *tb;
    
    rcu_read_lock();
    
    // 遍历路由表
    hlist_for_each_entry_rcu(tb, &net->ipv4.fib_table_hash[hash], tb_hlist) {
        if (tb->tb_id == flp->flowi4_table_id) {
            // 查找路由
            if (fib_table_lookup(tb, flp, res) == 0) {
                rcu_read_unlock();
                return 0;
            }
        }
    }
    
    rcu_read_unlock();
    return -ENETUNREACH;
}

// 写者：更新路由表（低频操作）
void fib_update(struct net *net, struct fib_table *new_tb) {
    struct fib_table *old_tb;
    
    // 找到旧的路由表
    old_tb = find_fib_table(net, new_tb->tb_id);
    
    // 原子替换
    hlist_replace_rcu(&old_tb->tb_hlist, &new_tb->tb_hlist);
    
    // 等待所有读者离开
    synchronize_rcu();
    
    // 释放旧路由表
    kfree(old_tb);
}
```

**场景 2：模块热插拔**

```c
// 设备驱动使用 RCU 保护设备列表
struct device {
    struct list_head list;
    // ...
};

// 读者：遍历设备列表
void device_foreach(void (*callback)(struct device *)) {
    struct device *dev;
    
    rcu_read_lock();
    list_for_each_entry_rcu(dev, &device_list, list) {
        callback(dev);
    }
    rcu_read_unlock();
}

// 写者：添加设备
void device_add(struct device *new_dev) {
    spin_lock(&device_lock);
    list_add_rcu(&new_dev->list, &device_list);
    spin_unlock(&device_lock);
}

// 写者：删除设备
void device_remove(struct device *dev) {
    spin_lock(&device_lock);
    list_del_rcu(&dev->list);
    spin_unlock(&device_lock);
    
    // 等待所有读者离开设备
    synchronize_rcu();
    
    kfree(dev);
}
```

### 29.5.6 读多写少场景的 RCU 优势

**性能对比**：

```c
// 基准测试：RCU vs 读写锁
#define NUM_READERS 16
#define NUM_WRITERS 2
#define ITERATIONS 10000000

// 方案 1：读写锁
struct {
    pthread_rwlock_t lock;
    int data;
} rwlock_data;

void *rwlock_reader(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        pthread_rwlock_rdlock(&rwlock_data.lock);
        int val = rwlock_data.data;
        pthread_rwlock_unlock(&rwlock_data.lock);
    }
    return NULL;
}

// 方案 2：RCU
struct {
    _Atomic(int *) data;
} rcu_data;

void *rcu_reader(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        rcu_read_lock();
        int *p = atomic_load(&rcu_data.data);
        int val = *p;
        rcu_read_unlock();
    }
    return NULL;
}

// 性能对比（16 读者 + 2 写者）：
// 方案         读吞吐量      写延迟
// 读写锁       500K ops/ms   100μs
// RCU          5000K ops/ms  1000μs
// 
// RCU 读吞吐量提升 10 倍！
// 但写延迟增加（需要等待 Grace Period）
```

**RCU 的适用场景**：

```
适用 RCU 的场景：
✓ 读多写少（读写比 > 10:1）
✓ 读者性能是瓶颈
✓ 可以容忍写延迟
✓ 数据结构是链表、树等指针结构

不适用 RCU 的场景：
✗ 写操作频繁
✗ 数据结构是数组等连续存储
✗ 需要严格的写顺序保证
✗ 内存受限（需要保存多个版本）
```

---

## 面试高频考点

### 1. SMP vs NUMA

**Q: SMP 和 NUMA 有什么区别？各自的优缺点？**

**A**: SMP 所有 CPU 访问内存延迟相同，编程简单但扩展性差（通常 ≤ 16 CPU）。NUMA 本地访问快、远程访问慢，扩展性好但需要 NUMA 感知编程。

### 2. MESI 协议

**Q: 解释 MESI 协议的四个状态及转换条件。**

**A**:
- M (Modified): 独占修改，与内存不一致
- E (Exclusive): 独占，与内存一致
- S (Shared): 共享，与内存一致
- I (Invalid): 无效

转换：读命中保持状态，写命中转为 M，读缺失可能转为 S/E，写缺失需要 BusRdX。

### 3. False Sharing

**Q: 什么是 False Sharing？如何解决？**

**A**: 两个 CPU 访问不同变量但这些变量在同一缓存行，导致缓存行不断在 CPU 间传输。解决方法：使用 padding 将变量分配到不同缓存行，或使用 `__attribute__((aligned(64)))`。

### 4. Amdahl 定律

**Q: 如果程序 10% 是串行的，使用 8 个 CPU 能加速多少？**

**A**: Speedup = 1 / (0.1 + 0.9/8) = 1 / 0.2125 ≈ 4.71 倍。理论上限 = 1/0.1 = 10 倍。

### 5. 锁分拆 vs 锁分层

**Q: 锁分拆和锁分层有什么区别？**

**A**: 锁分拆是将一个保护多个独立资源的锁拆分为多个锁。锁分层是将资源分成多个条带，每个条带一把锁，适用于哈希表等数据结构。

### 6. 无锁数据结构与 ABA 问题

**Q: 什么是 ABA 问题？如何解决？**

**A**: CAS 操作无法区分值从 A→B→A 的变化。解决方案：1) Tagged Pointer（版本号）；2) Hazard Pointer（保护指针）；3) 使用 GC 或 RCU。

### 7. RCU 机制

**Q: RCU 的核心思想是什么？适用于什么场景？**

**A**: RCU 读者无锁，写者复制后原子替换，延迟回收。适用于读多写少（读写比 > 10:1）的场景，如路由表、模块列表。

### 8. Grace Period

**Q: 什么是 RCU 的 Grace Period？如何判断它结束？**

**A**: Grace Period 是从指针替换到所有旧读者离开临界区的时间段。当所有 CPU 都经历至少一次静默状态（Quiescent State，如上下文切换、用户态执行）时，Grace Period 结束。

### 9. per-CPU 数据结构

**Q: per-CPU 数据结构的优势和劣势？**

**A**: 优势：完全无锁，极高并发，缓存友好。劣势：内存开销大（每个 CPU 一份），查询需要遍历所有 CPU，CPU 迁移可能导致数据不一致。

### 10. 内存屏障

**Q: 为什么需要内存屏障？x86 和 ARM 的内存模型有什么区别？**

**A**: CPU 和编译器可能重排序内存操作，导致多线程程序出错。x86 是 TSO（Total Store Order）模型，只有 Store-Load 可能重排序。ARM 是弱内存模型，所有类型的内存操作都可能重排序。

---

## 扩展阅读

1. **《Is Parallel Programming Still Hard, And, If So, What Can You Do About It?》** - Perfbook, Paul E. McKenney
2. **《A Primer on Memory Consistency and Cache Coherence》** - Daniel J. Sorin, Mark D. Hill, David A. Wood
3. **《The Art of Multiprocessor Programming》** - Maurice Herlihy, Nir Shavit
4. **《Linux Kernel Development》** - Robert Love (Chapter 10: Kernel Synchronization Methods)
5. **《Systems Performance》** - Brendan Gregg (Chapter 6: CPUs)

---

> **下一章预告**：我们将探讨分布式系统的基础知识，包括 CAP 定理、一致性模型和分布式共识算法。
