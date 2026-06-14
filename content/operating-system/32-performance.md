---
title: "Chapter 32: 性能分析与优化"
description: "深入理解性能指标，掌握 perf/strace/flame graph 等分析工具，理解瓶颈定位与优化技术"
updated: "2026-06-11"
---

# Chapter 32: 性能分析与优化

> **本章目标**：
> - 理解吞吐量、延迟、利用率、可扩展性等核心性能指标
> - 掌握 perf、strace、ltrace、火焰图等性能分析工具的使用方法
> - 学会定位 CPU、内存、I/O、锁竞争等常见性能瓶颈
> - 掌握算法优化、缓存局部性、并发优化、编译器优化等技术
> - 理解 Linux 内核参数调优的方法与实践

---

## 32.1 性能指标

### 32.1.1 性能分析的基本框架

性能分析的核心是回答一个问题：**系统在做什么，为什么慢？**

性能分析遵循一个系统化的方法论：

```
性能分析的基本流程：

  观察 (Observe) → 度量 (Measure) → 分析 (Analyze) → 优化 (Optimize) → 验证 (Verify)
       ↑                                                                    │
       └────────────────────────────────────────────────────────────────────┘
                              持续迭代

关键原则：
1. 先度量，后优化 —— 不要凭直觉猜测瓶颈
2. 关注最大瓶颈 —— 优化 Amdahl 定律中占比最大的部分
3. 一次改一个变量 —— 确保每次优化的效果可度量
4. 建立基线 —— 优化前后对比，量化改进幅度
```

### 32.1.2 吞吐量（Throughput）

**吞吐量** 是单位时间内系统完成的工作量。

```
吞吐量的常见度量：

┌─────────────────────────────────────────────────────────┐
│  维度          │  度量单位                  │  典型场景   │
├─────────────────────────────────────────────────────────┤
│  CPU           │  指令/秒 (IPC × 频率)      │  计算密集   │
│  网络          │  包/秒 (PPS), 字节/秒 (BPS) │  网络服务   │
│  磁盘 I/O     │  IOPS, MB/s                │  数据库     │
│  Web 服务      │  请求/秒 (RPS/QPS)         │  HTTP 服务  │
│  数据库        │  查询/秒 (QPS), 事务/秒(TPS)│ OLTP 系统  │
└─────────────────────────────────────────────────────────┘
```

吞吐量的计算公式：

```
吞吐量 = 完成的工作总量 / 总耗时时间

示例：Web 服务器
  - 10 秒内处理了 5000 个请求
  - 吞吐量 = 5000 / 10 = 500 QPS

示例：磁盘
  - 每秒完成 200 次 4KB 随机读
  - 吞吐量 = 200 IOPS（I/O Operations Per Second）
  - 带宽吞吐 = 200 × 4KB = 800 KB/s
```

**吞吐量的局限性**：高吞吐量不代表用户体验好。一个批处理系统可能有很高的吞吐量，但单个请求的延迟可能很高。

### 32.1.3 延迟（Latency）

**延迟** 是完成单个操作所需的时间。

```
延迟的组成：

一次请求的延迟分解：

  总延迟 = 网络传输 + 排队等待 + 处理时间 + I/O 等待

  ┌──────────────────────────────────────────────────┐
  │  阶段           │  典型耗时        │  优化方向     │
  ├──────────────────────────────────────────────────┤
  │  网络 RTT       │  1-100ms         │  CDN/就近部署 │
  │  排队等待       │  0-1000ms        │  扩容/限流    │
  │  CPU 计算       │  0.1-10ms        │  算法优化     │
  │  内存访问       │  100ns           │  缓存优化     │
  │  SSD 随机读     │  0.1ms           │  索引优化     │
  │  HDD 随机读     │  10ms            │  换 SSD       │
  └──────────────────────────────────────────────────┘
```

延迟的统计分布比平均值更有意义：

```
延迟百分位（Percentile）：

  P50 (中位数): 50% 的请求在此延迟内完成
  P90:          90% 的请求在此延迟内完成
  P99:          99% 的请求在此延迟内完成
  P999:         99.9% 的请求在此延迟内完成

示例：
  P50  = 10ms   ← 一半用户感受
  P90  = 50ms   ← 尾部延迟开始显现
  P99  = 200ms  ← 1% 用户体验差
  P999 = 2s     ← 最差的 0.1% 用户

为什么 P99 比平均值重要？
  - 平均延迟 20ms 看起来不错
  - 但如果 P99 = 2s，意味着每 100 个请求有 1 个要等 2 秒
  - 对于高并发系统，每秒数万请求，尾部延迟影响大量用户
```

### 32.1.4 利用率（Utilization）

**利用率** 是资源被使用的时间或容量占比。

```
利用率的计算：

  CPU 利用率 = CPU 忙碌时间 / 总时间 × 100%
  内存利用率 = 已用内存 / 总内存 × 100%
  磁盘利用率 = 磁盘忙碌时间 / 总时间 × 100%（iostat 的 %util）

利用率与性能的关系：

  0%          50%         70%    80%    90%   100%
  ├──────────┼──────────┼──────┼──────┼─────┤
  │  空闲     │  正常     │ 注意 │ 警告 │ 危险│

  关键阈值：
  - CPU 利用率 > 70%: 开始关注，可能有排队
  - CPU 利用率 > 90%: 严重瓶颈，任务大量排队
  - 磁盘利用率 > 70%: 延迟开始显著增加
  - 内存利用率 > 85%: 可能触发 swap，影响性能
```

**利用率的陷阱**：高利用率不一定意味着瓶颈。如果系统在高利用率下仍能满足延迟目标，则无需优化。但利用率持续接近 100% 时，排队延迟会急剧增加（排队理论）。

```
排队理论基础 —— Little's Law：

  L = λ × W

  L: 系统中平均请求数
  λ: 请求到达速率（吞吐量）
  W: 平均等待时间（延迟）

  推论：如果利用率 U 接近 1，
  W ≈ W₀ / (1 - U)
  其中 W₀ 是零负载时的服务时间

  当 U = 0.9 时，延迟是零负载的 10 倍
  当 U = 0.99 时，延迟是零负载的 100 倍！
```

### 32.1.5 可扩展性（Scalability）

**可扩展性** 描述系统通过增加资源来提升性能的能力。

```
扩展方式：

  垂直扩展（Scale Up）         水平扩展（Scale Out）
  ┌──────────┐                ┌───┐ ┌───┐ ┌───┐
  │ 更强的   │                │ S │ │ S │ │ S │
  │ CPU/内存 │                │ 1 │ │ 2 │ │ 3 │
  │ 单机     │                └───┘ └───┘ └───┘
  └──────────┘                 多台机器并行

扩展效率的度量：

  加速比 S(n) = T(1) / T(n)
  其中 T(1) 是单处理器执行时间，T(n) 是 n 个处理器的执行时间

  理想情况：S(n) = n（线性扩展）
  实际情况：S(n) < n，因为存在串行部分和通信开销
```

**Amdahl 定律**：

```
  S(n) = 1 / (f + (1-f)/n)

  f: 程序中不可并行的串行部分比例
  n: 处理器数量

  示例：
  - 如果 f = 5%（5% 是串行的）
  - 即使 n → ∞，最大加速比 = 1/0.05 = 20 倍

  这意味着：
  - 100 台机器最多获得 20 倍加速（不是 100 倍）
  - 串行部分是可扩展性的根本瓶颈
```

**Gustafson 定律**（修正视角）：

```
  S(n) = n - f × (n - 1)

  视角不同：问题规模随处理器数量增大
  实际中，更多处理器意味着处理更大规模的问题
  因此并行扩展仍然有意义
```

### 32.1.6 性能指标之间的权衡

```
性能指标的权衡关系：

  吞吐量 ←──→ 延迟
    │            │
    │  提高吞吐量可能增加延迟（批处理、排队）
    │  降低延迟可能减少吞吐量（更多上下文切换）
    │
  利用率 ←──→ 响应时间
    │
    │  高利用率 → 高排队延迟
    │  低利用率 → 资源浪费

实际案例：
  - 批处理系统：优先吞吐量，延迟可接受
  - 实时系统：优先延迟，吞吐量次要
  - Web 服务：两者都重要，P99 延迟是关键指标
```

---

## 32.2 性能分析工具

### 32.2.1 系统监控工具：top / htop

**top** 是最基本的实时系统监控工具。

```bash
# 启动 top
top

# top 输出的关键信息：
# %Cpu(s):  5.3 us,  2.1 sy,  0.0 ni, 92.1 id,  0.3 wa,  0.0 hi,  0.2 si
#            ↑        ↑        ↑       ↑         ↑        ↑        ↑
#          用户态   系统态   nice调整  空闲    I/O等待  硬中断   软中断

# 关键字段含义：
#   us (user):      用户空间进程消耗 CPU 百分比
#   sy (system):    内核空间消耗 CPU 百分比
#   ni (nice):      调整过优先级的进程消耗
#   id (idle):      空闲百分比
#   wa (iowait):    等待 I/O 完成的时间百分比
#   hi (hardware):  硬件中断处理时间
#   si (software):  软件中断处理时间
```

**htop** 是 top 的增强版本，提供更友好的界面：

```
htop 的优势：
  - 彩色显示，直观易读
  - 支持鼠标操作
  - 树形视图显示进程层级关系
  - 支持水平/垂直滚动
  - 可以直接发送信号给进程
  - 搜索/过滤进程
```

### 32.2.2 虚拟内存统计：vmstat

**vmstat** 报告虚拟内存、CPU、I/O 的综合统计信息。

```bash
# 每 2 秒采样一次，共采样 5 次
vmstat 2 5

# 输出示例：
# procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs  us sy id wa st
#  2  0      0 512000 128000 2048000    0    0     5    20  500 1000  10  5 84  1  0

# 关键字段：
#   r:    运行队列中的进程数（> CPU 核数表示 CPU 饱和）
#   b:    不可中断睡眠的进程数（通常等待 I/O）
#   swpd: 使用的 swap 空间（KB）— 非零表示内存不足
#   free: 空闲内存（KB）
#   si/so: swap in/out 速率 — 频繁换入换出说明内存不足
#   bi/bo: 块设备读入/写出（KB/s）
#   in:   每秒中断数
#   cs:   每秒上下文切换数 — 过高可能表示锁竞争
#   wa:   I/O 等待百分比 — 高值说明 I/O 瓶颈
```

### 32.2.3 I/O 统计：iostat

**iostat** 提供详细的磁盘 I/O 统计。

```bash
# 每 2 秒输出一次扩展统计
iostat -xz 2

# 输出示例：
# Device  r/s    w/s   rMB/s  wMB/s  rrqm/s  wrqm/s  %rrqm  %wrqm  r_await  w_await  aqu-sz  rareq-sz  wareq-sz  svctm  %util
# sda     150.0  50.0  2.3    1.5    10.0    5.0     6.3    9.1    0.5      1.2      0.15    16.0     32.0      0.8    16.0

# 关键指标：
#   r/s, w/s:       每秒读/写请求数（IOPS）
#   rMB/s, wMB/s:   读/写吞吐量
#   r_await:        读请求平均延迟（ms）
#   w_await:        写请求平均延迟（ms）
#   aqu-sz:         平均队列长度
#   %util:          设备利用率 — > 70% 表示设备饱和
```

### 32.2.4 Linux perf 工具

**perf** 是 Linux 内核自带的性能分析工具，基于硬件性能计数器（PMC）和内核事件。

```
perf 工具家族：

  perf stat      — 统计事件计数（快速概览）
  perf record    — 采样记录性能数据
  perf report    — 分析采样数据
  perf top       — 实时显示热点函数
  perf trace     — 跟踪系统调用（类似 strace）
  perf bench      — 运行基准测试
  perf list       — 列出可用事件
```

#### perf stat：事件计数

```bash
# 统计程序的硬件事件
perf stat ./my_program

# 输出示例：
#  Performance counter stats for './my_program':
#
#          1,234.56 msec  task-clock                #    0.987 CPUs utilized
#               150      context-switches           #  121.512 /sec
#                12      cpu-migrations             #    9.721 /sec
#             5,678      page-faults                #  4.600 K/sec
#    45,678,901,234      cycles                     #    3.700 GHz
#    23,456,789,012      instructions               #    0.51  insn per cycle
#     4,567,890,123      branches                   #    3.700 G/sec
#        56,789,012      branch-misses              #    1.24% of all branches
#     6,789,012,345      cache-references           #    5.500 G/sec
#       123,456,789      cache-misses               #    1.82% of all cache refs

# 关键指标解读：
#   insn per cycle (IPC): 每周期指令数 — 理想值 2-4，低值说明流水线停顿
#   branch-misses: 分支预测失败率 — > 5% 需要关注
#   cache-misses: 缓存未命中率 — > 10% 需要关注
#   context-switches: 上下文切换 — 高值可能有锁竞争
```

#### perf record / report：采样分析

```bash
# 采样记录（默认基于 CPU 周期）
perf record -g ./my_program          # -g 记录调用图（call graph）
perf record -F 99 -g ./my_program    # -F 99 每秒 99 次采样

# 分析报告
perf report                          # 交互式报告
perf report --stdio                  # 文本输出
perf report --sort comm,dso          # 按进程和库排序

# perf report 输出示例：
# Overhead  Command      Shared Object        Symbol
#   35.20%  my_program   my_program           [.] compute_heavy
#   18.45%  my_program   libc-2.31.so         [.] __memcpy_avx2
#   12.30%  my_program   my_program           [.] process_data
#    8.75%  my_program   my_program           [.] hash_lookup
#    5.60%  my_program   libc-2.31.so         [.] malloc
```

#### perf top：实时热点

```bash
# 实时显示 CPU 热点函数
perf top

# 类似 top，但显示的是函数级 CPU 使用率
# 按 Enter 可以查看函数的反汇编
```

### 32.2.5 strace：系统调用跟踪

**strace** 跟踪进程的系统调用和信号。

```bash
# 跟踪新进程
strace ./my_program

# 跟踪已运行的进程
strace -p <PID>

# 统计系统调用耗时
strace -c ./my_program

# 输出示例（-c 模式）：
# % time     seconds  usecs/call     calls    errors syscall
# ------ ----------- ----------- --------- --------- ----------------
#  45.23    0.123456         123      1004           read
#  23.45    0.067890          67      1012           write
#  15.67    0.045678        4567        10           mmap
#   8.90    0.023456         234       100           futex
#   6.75    0.018901         189       100           openat

# 常用选项：
#   -e trace=file    只跟踪文件相关调用
#   -e trace=network 只跟踪网络相关调用
#   -e trace=process 只跟踪进程相关调用
#   -e trace=memory  只跟踪内存相关调用
#   -T              显示每个调用的耗时
#   -f              跟踪子进程
#   -tt             显示微秒级时间戳
```

### 32.2.6 ltrace：库函数跟踪

**ltrace** 跟踪动态库函数调用。

```bash
# 跟踪库函数调用
ltrace ./my_program

# 输出示例：
# malloc(4096)                            = 0x55a1234
# printf("Hello %s\n", "world")           = 12
# memcpy(0x55a1234, 0x55a5678, 1024)     = 0x55a1234
# free(0x55a1234)                         = <void>
# strlen("test string")                   = 11

# 对比 strace：
#   strace 跟踪系统调用（内核接口）
#   ltrace 跟踪库函数调用（用户空间）
#   ltrace 不需要内核支持，但只对动态链接的程序有效
```

### 32.2.7 火焰图（Flame Graphs）

火焰图是 Brendan Gregg 发明的性能数据可视化方法，用于快速识别 CPU 热点。

```
火焰图的结构：

  ┌─────────────────────────────────────────────────────┐
  │                    main (100%)                       │  ← 顶层：入口函数
  ├─────────────────────┬───────────────────────────────┤
  │    process (60%)    │       handle_io (40%)          │  ← 中间：调用链
  ├──────────┬──────────┼────────────┬──────────────────┤
  │ compute  │ hash_   │  read()    │   write()         │  ← 下层：具体函数
  │ (35%)    │ lookup  │  (25%)     │   (15%)           │
  │          │ (25%)   │            │                    │
  └──────────┴─────────┴────────────┴──────────────────┘

  - X 轴：字母顺序排列，宽度表示采样占比（不是时间线！）
  - Y 轴：调用栈深度（从下到上）
  - 颜色：随机暖色调，无特殊含义
  - 宽度越大 = 该函数在采样中出现次数越多 = 越热
```

使用 perf 生成火焰图：

```bash
# 步骤 1：采集性能数据
perf record -F 99 -g --call-graph dwarf ./my_program

# 步骤 2：生成折叠栈
perf script | stackcollapse-perf.pl > out.folded

# 步骤 3：生成火焰图 SVG
flamegraph.pl out.folded > flamegraph.svg

# 也可以用 Brendan Gregg 的 FlameGraph 工具：
git clone https://github.com/brendangregg/FlameGraph.git
perf record -F 99 -g ./my_program
perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flame.svg
```

火焰图的类型：

```
火焰图类型：

  1. CPU 火焰图（On-CPU）
     - 显示消耗 CPU 时间的函数
     - perf record -F 99 -g -p <PID>

  2. Off-CPU 火焰图
     - 显示等待（睡眠）的函数
     - 用于分析 I/O 等待、锁竞争

  3. 内存火焰图
     - 显示内存分配的调用栈
     - perf record -e 'kmem:kmalloc' -g

  4. 差分火焰图（Differential）
     - 对比两次采样的差异
     - 红色 = 增加，蓝色 = 减少
     - flamegraph.pl --diff
```

---

## 32.3 性能瓶颈定位

### 32.3.1 USE 方法论

Brendan Gregg 提出的 **USE（Utilization, Saturation, Errors）方法论** 是系统化定位瓶颈的框架：

```
USE 方法论 —— 对每种资源检查三个维度：

  ┌────────────┬──────────────┬──────────────────┬──────────────────┐
  │  资源       │  利用率 (U)   │  饱和度 (S)       │  错误 (E)        │
  ├────────────┼──────────────┼──────────────────┼──────────────────┤
  │  CPU       │  mpstat %usr │  vmstat r > 核数  │  perf stat 异常  │
  │            │  + %sys      │  运行队列长度     │                  │
  ├────────────┼──────────────┼──────────────────┼──────────────────┤
  │  内存      │  free -m     │  vmstat si/so     │  dmesg OOM       │
  │            │  已用/总量    │  swap 活动        │                  │
  ├────────────┼──────────────┼──────────────────┼──────────────────┤
  │  磁盘 I/O │  iostat %util│  iostat aqu-sz    │  smartctl 错误   │
  │            │              │  队列深度         │                  │
  ├────────────┼──────────────┼──────────────────┼──────────────────┤
  │  网络      │  sar -n DEV  │  ifconfig overruns│  ifconfig errors │
  │            │  带宽使用率   │  接口溢出         │  网卡错误计数     │
  └────────────┴──────────────┴──────────────────┴──────────────────┘
```

### 32.3.2 CPU 瓶颈

**症状**：高 CPU 使用率，运行队列长，响应时间增加。

```bash
# 1. 确认 CPU 是否瓶颈
mpstat -P ALL 2        # 查看每个 CPU 的使用情况
pidstat 1              # 查看每个进程的 CPU 使用

# 2. 定位热点函数
perf record -F 999 -g -p <PID> -- sleep 30
perf report

# 3. 常见 CPU 瓶颈原因
```

```
CPU 瓶颈的常见原因：

  ┌─────────────────────────────────────────────────────────────┐
  │  原因                    │  诊断方法                         │
  ├─────────────────────────────────────────────────────────────┤
  │  算法复杂度过高           │  perf top 热点函数                │
  │  死循环 / 自旋等待       │  top 显示单核 100%                │
  │  频繁的锁竞争            │  perf stat 上下文切换高           │
  │  过多的系统调用           │  strace -c 统计                  │
  │  分支预测失败率高         │  perf stat branch-misses         │
  │  缓存命中率低            │  perf stat cache-misses          │
  │  内核态占比高 (%sys)      │  中断风暴、频繁上下文切换         │
  └─────────────────────────────────────────────────────────────┘
```

### 32.3.3 内存瓶颈

**症状**：频繁的页故障、swap 活动、OOM Killer 触发。

```bash
# 1. 检查内存使用
free -h                # 查看内存和 swap 使用
vmstat 1               # 观察 si/so（swap in/out）
sar -B 1               # 页故障统计

# 2. 检查页故障
perf stat -e page-faults,minor-faults,major-faults ./my_program

# 3. 定位内存热点
valgrind --tool=massif ./my_program    # 堆内存分析
perf record -e 'kmem:kmalloc' -g -p <PID>
```

```
内存瓶颈的层次：

  ┌─────────────────────────────────────────────────────────┐
  │  层次            │  现象              │  影响            │
  ├─────────────────────────────────────────────────────────┤
  │  L1/L2 缓存未命中│  cache-misses 高   │  3-10x 延迟      │
  │  L3 缓存未命中   │  LLC-load-misses   │  10-50x 延迟     │
  │  TLB 未命中      │  dTLB-load-misses  │  100x 延迟       │
  │  页故障 (minor)  │  page-faults       │  μs 级延迟       │
  │  页故障 (major)  │  需要磁盘 I/O      │  ms 级延迟       │
  │  Swap 使用       │  vmstat si/so > 0  │  严重影响性能    │
  │  OOM Kill        │  dmesg | grep oom  │  进程被杀死      │
  └─────────────────────────────────────────────────────────┘
```

### 32.3.4 I/O 瓶颈

**症状**：高 iowait、磁盘利用率高、请求延迟增加。

```bash
# 1. 确认 I/O 瓶颈
iostat -xz 2           # 查看磁盘利用率和延迟
iotop                  # 实时查看进程 I/O

# 2. 定位 I/O 密集的进程
pidstat -d 1           # 进程级 I/O 统计

# 3. 跟踪 I/O 模式
strace -e trace=read,write -p <PID>   # 跟踪读写调用
perf record -e block:block_rq_issue -g -p <PID>
```

```
I/O 优化策略：

  问题                   → 解决方案

  频繁小 I/O             → 合并为批量 I/O（readahead）
  同步 I/O 阻塞          → 异步 I/O（io_uring / aio）
  随机读过多              → 增大缓存 / 调整读取模式
  fsync 过于频繁          → 批量提交 / 调整刷盘策略
  文件系统碎片            → 定期 defrag / 选择合适 FS
  日志同步开销            → 异步日志 / 批量刷写
```

### 32.3.5 锁竞争

**症状**：高上下文切换率、CPU 利用率不高但延迟高。

```bash
# 1. 检查上下文切换
vmstat 1               # 查看 cs（上下文切换数）
pidstat -w 1           # 进程级上下文切换

# 2. 使用 perf 分析锁竞争
perf lock record -p <PID> sleep 10
perf lock report

# 3. Off-CPU 火焰图分析等待时间
perf record -e 'sched:sched_switch' -g -p <PID> -- sleep 10
```

```
锁竞争的优化方向：

  ┌─────────────────────────────────────────────────────────┐
  │  问题                    │  优化策略                     │
  ├─────────────────────────────────────────────────────────┤
  │  全局锁粒度过粗           │  拆分为细粒度锁               │
  │  读写比例悬殊             │  读写锁 (rwlock) / RCU        │
  │  临界区过大               │  缩小临界区范围               │
  │  锁嵌套导致死锁风险       │  统一加锁顺序                 │
  │  自旋锁在长等待浪费 CPU   │  改用互斥锁 (mutex)           │
  │  无锁数据结构适用         │  CAS 操作 / 无锁队列          │
  └─────────────────────────────────────────────────────────┘
```

### 32.3.6 快速诊断清单

```
60 秒快速诊断清单（Brendan Gregg）：

  1. uptime          → 查看负载均值（1/5/15 分钟）
  2. dmesg -T | tail → 检查内核错误（OOM、硬件错误）
  3. vmstat 1        → 整体 CPU/内存/IO 概览
  4. mpstat -P ALL 1 → 每个 CPU 的使用情况
  5. pidstat 1       → 每个进程的 CPU 使用
  6. iostat -xz 1    → 磁盘 I/O 统计
  7. free -h         → 内存使用情况
  8. sar -n DEV 1    → 网络接口吞吐量
  9. sar -n TCP,ETCP 1 → TCP 状态统计
  10. top            → 进程排名概览

  通过这 10 个命令，可以在 60 秒内对系统状态做出初步判断。
```

---

## 32.4 优化技术

### 32.4.1 算法优化

算法优化是效果最显著的优化手段。

```
算法复杂度与性能的关系：

  数据规模 n = 1,000,000

  O(n²)     = 10^12 操作  → 不可行（数小时）
  O(n log n) = 2×10^7 操作 → 可接受（毫秒级）
  O(n)       = 10^6 操作   → 很快（亚毫秒）
  O(log n)   = 20 操作     → 极快（纳秒级）

  算法优化优先级：
  1. 减少算法复杂度（O(n²) → O(n log n)）
  2. 优化常数因子（减少实际操作数）
  3. 使用更高效的数据结构（数组 vs 链表、哈希表 vs 树）
```

常见优化示例：

```c
// 优化前：O(n²) 的查找
for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        if (target[i] == source[j]) {
            result[count++] = target[i];
            break;
        }
    }
}

// 优化后：O(n + m) 的查找（使用哈希表）
HashSet set = new HashSet(source);
for (int i = 0; i < n; i++) {
    if (set.contains(target[i])) {
        result[count++] = target[i];
    }
}
```

### 32.4.2 缓存局部性与预取

CPU 缓存的速度比主存快 100 倍，利用缓存局部性是关键优化。

```
缓存层次与访问延迟：

  ┌──────────────┬───────────┬──────────────────┐
  │  存储层次     │  大小      │  访问延迟         │
  ├──────────────┼───────────┼──────────────────┤
  │  L1 Cache    │  32-64 KB  │  1-2 ns (4 cycle)│
  │  L2 Cache    │  256 KB    │  3-5 ns          │
  │  L3 Cache    │  8-32 MB   │  10-20 ns        │
  │  主存 DRAM   │  16-256 GB │  50-100 ns       │
  │  SSD         │  512 GB    │  100 μs          │
  │  HDD         │  2 TB      │  10 ms           │
  └──────────────┴───────────┴──────────────────┘
```

**空间局部性优化**：

```c
// 差：列优先遍历（跳跃访问，缓存不友好）
for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
        sum += matrix[i][j];  // 每次跳过整行

// 好：行优先遍历（连续访问，缓存友好）
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        sum += matrix[i][j];  // 连续访问相邻元素

// 性能差异可达 10-50 倍！
```

**时间局部性优化**：

```c
// 差：每次重新计算
double get_position(int frame) {
    return sin(frame * M_PI / 180) * radius;  // 三角函数很慢
}

// 好：预计算并缓存
double sin_table[360];
void init_table() {
    for (int i = 0; i < 360; i++)
        sin_table[i] = sin(i * M_PI / 180);
}
double get_position(int frame) {
    return sin_table[frame % 360] * radius;  // 表查找，O(1)
}
```

**预取（Prefetching）**：

```c
// 手动预取（在数据使用前提示 CPU 加载到缓存）
for (int i = 0; i < n; i++) {
    __builtin_prefetch(&array[i + 16], 0, 3);  // 预取 16 个元素之后的数据
    process(array[i]);
}

// 编译器自动预取（-O2 及以上）
// 硬件预取器会自动检测顺序访问模式并预取
```

### 32.4.3 并发优化与减少锁竞争

```
锁优化策略：

  策略 1：减小锁粒度
  ┌─────────────────────────────────────────┐
  │  粗粒度锁（全局锁）                      │
  │  ┌───────────────────────────────────┐  │
  │  │  临界区 A  │  临界区 B  │  临界区 C│  │
  │  └───────────────────────────────────┘  │
  │              ↓ 拆分为 ↓                  │
  │  细粒度锁（分区锁）                      │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
  │  │ 锁 A    │ │ 锁 B    │ │ 锁 C    │   │
  │  └─────────┘ └─────────┘ └─────────┘   │
  └─────────────────────────────────────────┘

  策略 2：读写锁分离
  rwlock_t rwlock;
  read_lock(&rwlock);    // 多个读者可以并发
  // 读操作
  read_unlock(&rwlock);

  write_lock(&rwlock);   // 写者独占
  // 写操作
  write_unlock(&rwlock);

  策略 3：无锁数据结构
  // 使用 CAS（Compare-And-Swap）原子操作
  do {
      old_val = *ptr;
      new_val = compute(old_val);
  } while (!__sync_bool_compare_and_swap(ptr, old_val, new_val));

  策略 4：RCU（Read-Copy-Update）
  // 读者无锁，写者先复制再更新指针
  // 适用于读多写少的场景（如路由表、配置）
```

### 32.4.4 I/O 优化

```
I/O 优化技术：

  1. 批量处理（Batching）
     - 差：每条记录单独 write()
     - 好：缓冲多条记录，一次 writev() 或 sendmsg()
     - 效果：减少系统调用次数，减少用户/内核切换

  2. 异步 I/O
     - 同步 I/O：线程阻塞等待 I/O 完成
     - 异步 I/O：发起 I/O 后继续执行，完成时通知
     - Linux: io_uring（最新，最高性能）
             aio（传统 POSIX AIO）
             epoll（网络 I/O 多路复用）

  3. 零拷贝（Zero Copy）
     传统路径（4 次拷贝）：
     磁盘 → 内核缓冲区 → 用户缓冲区 → Socket 缓冲区 → 网卡
     零拷贝路径（2 次拷贝）：
     磁盘 → 内核缓冲区 → 网卡（sendfile/splice）
     效果：减少 CPU 拷贝开销，降低延迟

  4. 内存映射（mmap）
     - 将文件映射到进程地址空间
     - 避免 read/write 系统调用
     - 由内核按需加载页（page fault 驱动）

  5. 直接 I/O（O_DIRECT）
     - 绕过页缓存，直接从用户缓冲区到设备
     - 适用于数据库等自带缓存的应用
```

### 32.4.5 编译器优化

```
GCC/Clang 优化级别：

  -O0    无优化（默认）—— 编译最快，调试方便
  -O1    基本优化 —— 去除无用代码、简单优化
  -O2    推荐优化 —— 大多数优化开启，编译时间适中
  -O3    激进优化 —— 向量化、循环展开、内联
  -Os    大小优化 —— 优化代码大小（嵌入式场景）
  -Ofast 最激进 —— O3 + 违反 IEEE 浮点标准的优化

关键优化技术：

  1. 内联展开（Inlining）
     - 将小函数体直接展开到调用点
     - 消除函数调用开销
     - GCC: -finline-functions, -O2 以上自动启用

  2. 循环优化
     - 循环展开（Loop Unrolling）：减少分支开销
     - 循环向量化（Loop Vectorization）：使用 SIMD 指令
     - 循环融合（Loop Fusion）：合并相邻循环

  3. 分支优化
     - __builtin_expect() 提示编译器分支概率
     - if (__builtin_expect(ptr != NULL, 1)) { ... }
     - likely()/unlikely() 宏

  4. PGO（Profile-Guided Optimization）
     步骤：
     a. 编译插桩版本：gcc -fprofile-generate -O2 prog.c
     b. 运行程序收集数据：./prog
     c. 使用数据优化编译：gcc -fprofile-use -O2 prog.c
     效果：10-30% 性能提升（热点代码优化更好）
```

### 32.4.6 内存分配优化

```
内存分配优化策略：

  1. 对象池（Object Pool）
     - 预分配大量对象，避免频繁 malloc/free
     - 适用于创建/销毁频繁的小对象

  2. 内存池（Memory Pool / Slab Allocator）
     - 一次性分配大块内存，自行管理
     - Linux 内核的 slab 分配器就是这个原理

  3. 栈分配（Stack Allocation）
     - 小对象使用 alloca() 或 VLA（变长数组）
     - 自动释放，无碎片问题
     - 注意：不要分配过大的栈对象

  4. 减少内存碎片
     - 使用固定大小的内存块
     - 定期整理内存（compact）
     - 选择合适的分配器（jemalloc / tcmalloc）
```

---

## 32.5 Linux 性能调优

### 32.5.1 调度器参数

```bash
# 查看调度器
cat /proc/sys/kernel/sched_rr_timeslice_ms    # RR 调度时间片

# CFS 调度器参数
/proc/sys/kernel/sched_min_granularity_ns     # 最小调度粒度
/proc/sys/kernel/sched_latency_ns             # 调度延迟目标
/proc/sys/kernel/sched_wakeup_granularity_ns  # 唤醒粒度
/proc/sys/kernel/sched_migration_cost_ns      # 迁移代价

# 实时调度参数
chrt -f 99 ./my_program    # 设置 FIFO 实时调度，优先级 99
chrt -r 50 ./my_program    # 设置 RR 实时调度，优先级 50

# CPU 亲和性绑定
taskset -c 0,1 ./my_program          # 绑定到 CPU 0 和 1
taskset -p -c 2,3 <PID>              # 修改运行中进程的亲和性
```

```
调度器调优建议：

  场景                  → 参数调整

  低延迟应用            → 减小 sched_min_granularity_ns
  (HFT、游戏)           → 减小 sched_latency_ns
                         → 使用实时调度 (SCHED_FIFO)

  高吞吐批处理          → 增大 sched_min_granularity_ns
  (数据分析、编译)       → 增大 sched_latency_ns

  NUMA 优化             → 绑定进程到特定 CPU
                         → 使用 numactl --membind=0 ./prog
```

### 32.5.2 内存管理参数：vm.swappiness

```bash
# 查看 swappiness
cat /proc/sys/vm/swappiness

# 设置 swappiness
sysctl -w vm.swappiness=10    # 临时设置
echo 10 > /proc/sys/vm/swappiness

# 永久设置：编辑 /etc/sysctl.conf
# vm.swappiness = 10
```

```
swappiness 的含义：

  值      含义
  ────────────────────────────────────────
  0       尽可能不使用 swap（Linux 3.5+）
          内存不足时才 swap
  1-10    极少使用 swap
  10      服务器推荐值
  60      默认值（桌面系统）
  100     积极使用 swap

  数据库服务器推荐：vm.swappiness = 1（几乎不 swap）
  通用服务器推荐：vm.swappiness = 10
  内存充足的桌面：vm.swappiness = 60（默认）
```

其他重要的内存参数：

```bash
# 脏页管理
vm.dirty_ratio = 20            # 脏页占内存比例超过 20% 时阻塞写入
vm.dirty_background_ratio = 10 # 脏页超过 10% 时后台回写
vm.dirty_expire_centisecs = 3000  # 脏页过期时间（30 秒）
vm.dirty_writeback_centisecs = 500 # 回写检查间隔（5 秒）

# OOM 控制
vm.overcommit_memory = 0       # 0=启发式, 1=总是允许, 2=严格检查
vm.overcommit_ratio = 50       # overcommit_memory=2 时的允许比例

# 大页（Huge Pages）
# 预分配大页减少 TLB 未命中
echo 1024 > /proc/sys/vm/nr_hugepages
# 程序使用大页：mmap with MAP_HUGETLB
```

### 32.5.3 网络栈调优

```bash
# 网络缓冲区大小
net.core.rmem_max = 16777216           # 接收缓冲区最大值
net.core.wmem_max = 16777216           # 发送缓冲区最大值
net.core.rmem_default = 262144         # 接收缓冲区默认值
net.core.wmem_default = 262144         # 发送缓冲区默认值

# TCP 缓冲区（min, default, max）
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# 连接队列
net.core.somaxconn = 65535             # 监听队列最大长度
net.ipv4.tcp_max_syn_backlog = 65535   # SYN 队列长度
net.core.netdev_max_backlog = 65535    # 网卡接收队列长度

# 连接复用
net.ipv4.tcp_tw_reuse = 1             # 重用 TIME_WAIT 连接
net.ipv4.tcp_fin_timeout = 15         # FIN_WAIT_2 超时时间
net.ipv4.tcp_keepalive_time = 600     # Keepalive 探测间隔

# TCP 拥塞控制
net.ipv4.tcp_congestion_control = bbr  # 使用 BBR 拥塞控制算法
net.core.default_qdisc = fq           # 配合 BBR 的队列调度

# 查看当前拥塞控制算法
sysctl net.ipv4.tcp_congestion_control
# 可选: cubic (默认), bbr, reno, vegas
```

### 32.5.4 文件系统挂载选项

```bash
# ext4 常用优化选项
mount -o noatime,nodiratime,data=writeback /dev/sda1 /data
#   noatime      — 不更新文件访问时间（减少写操作）
#   nodiratime   — 不更新目录访问时间
#   data=writeback — 元数据日志模式（性能最好，断电可能丢数据）

# XFS 推荐选项
mount -o noatime,logbufs=8,logbsize=256k /dev/sdb1 /data

# SSD 优化
mount -o noatime,discard /dev/nvme0n1p1 /data
#   discard — 启用 TRIM，帮助 SSD 管理闪存块

# I/O 调度器选择
cat /sys/block/sda/queue/scheduler
# 可选: mq-deadline (通用), bfq (桌面), none (NVMe SSD)

# SSD 推荐 none 或 mq-deadline
echo none > /sys/block/nvme0n1/queue/scheduler

# HDD 推荐 mq-deadline 或 bfq
echo mq-deadline > /sys/block/sda/queue/scheduler
```

### 32.5.5 其他系统参数

```bash
# 文件描述符限制
ulimit -n                # 查看当前限制
ulimit -n 65535          # 设置进程限制
# 永久设置：/etc/security/limits.conf
# *  soft  nofile  65535
# *  hard  nofile  65535

# 透明大页（THP）
echo never > /sys/kernel/mm/transparent_hugepage/enabled
# 数据库通常建议禁用 THP（Redis、MySQL、PostgreSQL）
# 因为 THP 的合并操作会导致延迟抖动

# NUMA 平衡
echo 0 > /proc/sys/kernel/numa_balancing
# 对延迟敏感的应用，可以关闭自动 NUMA 平衡
# 使用 numactl 手动控制内存分配策略

# CPU C-State（节能状态）
# 高性能场景禁用深度 C-State
# 通过 BIOS 设置或内核参数：
# intel_idle.max_cstate=0 processor.max_cstate=0
```

---

## 32.6 面试高频考点

### 考点 1：性能分析方法论

```
Q: 如何系统地分析一个系统性能问题？

A: 使用分层法，自顶向下：
   1. 先看整体：uptime（负载均值）、top（CPU/内存概览）
   2. 确定瓶颈类型：CPU？内存？I/O？网络？
   3. 定位具体原因：perf top / strace / iostat
   4. 找到热点代码：perf record + 火焰图
   5. 优化并验证：对比优化前后的指标

   关键原则：
   - 不要猜测，用数据说话
   - 先量化，后优化
   - 一次改一个变量
```

### 考点 2：perf 工具的使用

```
Q: perf stat 和 perf record 的区别？

A:
  perf stat:   事件计数，统计整个程序的事件总数
               输出：instructions, cache-misses 等总计数
               用途：快速概览，判断瓶颈类型

  perf record: 事件采样，记录每个采样点的调用栈
               输出：采样文件 perf.data
               配合 perf report 或火焰图使用
               用途：定位热点函数和调用路径

  类比：perf stat 像是体检报告的汇总数据
        perf record 像是详细的检查记录
```

### 考点 3：缓存优化

```
Q: 如何利用 CPU 缓存优化程序性能？

A: 三个层面：
   1. 时间局部性 —— 热数据反复使用，减少缓存失效
   2. 空间局部性 —— 顺序访问，充分利用缓存行（64B）
   3. 数据布局 —— 结构体数组(AoS) vs 数组结构体(SoA)
      AoS: [{x,y,z}, {x,y,z}, ...]  // 遍历单字段时缓存不友好
      SoA: [{x,x,...}, {y,y,...}, {z,z,...}]  // 遍历单字段时缓存友好

   典型优化案例：
   - 矩阵乘法：分块(Tiling) 提高 L1/L2 命中率
   - 链表遍历 → 数组遍历（链表节点分散在内存中）
   - 减小数据结构大小，提高缓存行利用率
```

### 考点 4：I/O 优化

```
Q: 什么是零拷贝？为什么能提高性能？

A: 传统文件传输路径：
   磁盘 → 内核缓冲区 → 用户缓冲区 → Socket缓冲区 → 网卡
   4 次拷贝 + 4 次上下文切换

   零拷贝路径（sendfile）：
   磁盘 → 内核缓冲区 → 网卡
   2 次拷贝（DMA） + 2 次上下文切换

   性能提升：
   - 减少 CPU 拷贝开销（大数据量下显著）
   - 减少用户/内核态切换
   - 减少内存带宽消耗
   - 应用：Nginx sendfile、Kafka zero-copy、Redis RDB
```

### 考点 5：锁竞争优化

```
Q: 如何诊断和优化锁竞争？

A: 诊断：
   1. vmstat 看 cs（上下文切换数）异常高
   2. perf stat 看 context-switches
   3. perf lock record + report 分析锁等待
   4. Off-CPU 火焰图看等待时间分布

   优化：
   1. 减小锁粒度：全局锁 → 分段锁 → 每对象锁
   2. 读写锁分离：pthread_rwlock_t
   3. 无锁编程：CAS、原子操作、无锁队列
   4. RCU：读者无锁，写者延迟释放
   5. 消除共享：线程本地存储(TLS)、分区数据结构
```

### 考点 6：Amdahl 定律

```
Q: 程序有 20% 是串行的，用 8 个处理器能加速多少？

A: S(8) = 1 / (0.2 + 0.8/8) = 1 / (0.2 + 0.1) = 1 / 0.3 ≈ 3.33

   理想加速比是 8 倍，实际只能达到 3.33 倍。
   即使用无限多处理器，最大加速比 = 1/0.2 = 5 倍。

   启示：并行优化的前提是减少串行部分的占比。
```

### 考点 7：swappiness 的影响

```
Q: 数据库服务器为什么建议设置低 swappiness？

A:
  - 数据库的工作集（热数据）应尽可能留在内存中
  - swappiness=60（默认）时，内核可能过早将匿名页换出到 swap
  - 换出后再换入会产生磁盘 I/O，增加延迟
  - 数据库延迟敏感，几毫秒的 swap I/O 可能导致请求超时
  - 设置 swappiness=1 或 10，让内核优先回收文件缓存页
  - 同时确保物理内存充足，避免 OOM
```

---

## 32.7 扩展阅读

### 推荐书籍

1. **《Systems Performance》** - Brendan Gregg
   - 性能分析领域的圣经
   - 涵盖 Linux/Unix 系统的全面性能分析方法
   - USE 方法论和 60 秒诊断清单的来源

2. **《BPF Performance Tools》** - Brendan Gregg
   - 基于 eBPF 的现代性能分析方法
   - 覆盖 bcc 和 bpftrace 工具

3. **《Computer Architecture: A Quantitative Approach》** - Hennessy & Patterson
   - 计算机体系结构的经典教材
   - 深入理解缓存、流水线、并行等对性能的影响

4. **《The Art of Multiprocessor Programming》** - Herlihy & Shavit
   - 并发编程与无锁数据结构
   - 理解锁竞争优化的理论基础

### 在线资源

1. **Brendan Gregg 的博客**：https://www.brendangregg.com/
   - Linux 性能分析工具图谱
   - 火焰图的发明者和最佳实践
   - 大量实战案例

2. **Linux Performance**：https://www.brendangregg.com/linuxperf.html
   - Linux 性能分析工具全景图
   - 从 USE 方法论到具体工具的映射

3. **perf 官方文档**：https://perf.wiki.kernel.org/
   - perf 工具的完整参考

4. **io_uring 文档**：https://kernel.dk/io_uring.pdf
   - Linux 最新的高性能异步 I/O 接口

### 关键命令速查

```
性能分析命令速查表：

  # 系统概览
  uptime                          # 负载均值
  top / htop                      # 实时进程监控
  vmstat 1                        # CPU/内存/IO 综合
  mpstat -P ALL 1                 # 每核 CPU 统计

  # CPU 分析
  perf stat ./prog                # 事件计数
  perf record -F 999 -g ./prog   # 采样记录
  perf report                     # 分析报告
  perf top                        # 实时热点

  # 内存分析
  free -h                         # 内存使用
  vmstat 1                        # swap 活动
  valgrind --tool=massif ./prog   # 堆分析

  # I/O 分析
  iostat -xz 2                    # 磁盘统计
  iotop                           # 进程 I/O
  pidstat -d 1                    # 进程 I/O 统计

  # 网络分析
  sar -n DEV 1                    # 网络接口统计
  ss -s                           # 连接统计
  tcpdump -i eth0                 # 抓包分析

  # 跟踪工具
  strace -p <PID>                 # 系统调用跟踪
  ltrace ./prog                   # 库函数跟踪
  perf trace ./prog               # perf 版 strace
```

---

> **本章总结**：
> - 性能分析遵循「先度量后优化」的原则，使用 USE 方法论系统化排查
> - 吞吐量、延迟、利用率、可扩展性是核心性能指标，需要根据场景权衡
> - perf 是 Linux 最强大的性能分析工具家族，火焰图是最直观的可视化方法
> - CPU、内存、I/O、锁竞争是四大类常见瓶颈，各有对应的诊断和优化手段
> - 算法优化效果最大，缓存局部性是微观层面最关键的优化，编译器优化提供免费的性能提升
> - Linux 内核参数（swappiness、调度器、网络栈、文件系统）对特定场景有显著影响
