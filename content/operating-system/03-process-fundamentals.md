# Chapter 3: 进程抽象基础

> **本章目标**：深入理解进程（Process）这一操作系统的核心抽象。通过分析进程的定义、组成、状态模型、创建与终止机制，掌握操作系统如何将单个 CPU 虚拟化为多个"虚拟 CPU"，使多个程序看似同时运行。

---

## 3.1 进程的概念

### 3.1.1 程序 vs 进程：静态 vs 动态

**程序（Program）** 和 **进程（Process）** 是两个截然不同的概念，理解它们的区别是掌握操作系统的第一步。

**程序（Program）**：
- **静态实体**：存储在磁盘上的可执行文件（如 `/bin/ls`、`a.out`）。
- **被动**：程序本身不执行任何操作，只是一系列指令和数据的集合。
- **持久性**：程序文件在磁盘上长期存在，删除后才消失。
- **组成部分**：
  - **代码段（Text Segment）**：机器指令。
  - **数据段（Data Segment）**：全局变量、静态变量。
  - **符号表（Symbol Table）**：调试信息（可选）。

**进程（Process）**：
- **动态实体**：程序的**执行实例**。
- **主动**：进程在 CPU 上执行，消耗资源（CPU 时间、内存、文件）。
- **临时性**：进程运行结束后消失，资源被回收。
- **组成部分**：
  - **程序代码**：从可执行文件加载。
  - **程序数据**：全局变量、静态变量。
  - **堆（Heap）**：动态分配的内存（`malloc()`）。
  - **栈（Stack）**：函数调用栈、局部变量。
  - **进程控制块（PCB）**：内核维护的进程元数据。
  - **打开的文件**：文件描述符表。
  - **CPU 寄存器**：程序计数器、栈指针、通用寄存器等。

**类比**：
- **程序** = **食谱**（Recipe）：书架上的一本菜谱。
- **进程** = **烹饪过程**（Cooking Process）：按照菜谱正在制作菜肴的厨师，使用食材（数据）、工具（CPU、内存）。

**示例**：
```c
// program.c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

1. **编译**：`gcc program.c -o program` → 生成程序文件 `program`（ELF 可执行文件）。
2. **执行**：`./program` → 操作系统创建进程，加载程序到内存，分配资源，执行指令。

**一个程序可以对应多个进程**：
```bash
$ ./program &     # 进程 1
[1] 1234
$ ./program &     # 进程 2
[2] 1235
$ ./program &     # 进程 3
[3] 1236
```

三个进程运行同一程序（`program`），但拥有独立的内存空间、CPU 状态、文件描述符。

<div data-component="ProgramVsProcessComparison"></div>

---

### 3.1.2 进程的定义：执行中的程序实例

操作系统教材中对进程的经典定义：

> **进程（Process）** 是程序的一次**执行实例**，是操作系统进行**资源分配**和**调度**的基本单位。

**进程的三大本质特征**：

**1. 动态性（Dynamic）**
- 进程是程序的**动态执行过程**，有生命周期（创建 → 运行 → 终止）。
- 进程状态随时间变化（就绪、运行、阻塞）。

**2. 并发性（Concurrency）**
- 多个进程在宏观上"同时"运行（通过时间片轮转）。
- 即使单核 CPU，也能通过快速切换实现伪并发。

**3. 独立性（Independence）**
- 每个进程拥有独立的地址空间、资源（文件、内存）。
- 一个进程崩溃不影响其他进程（隔离性）。

**进程的资源**：

一个进程包含以下资源：

| 资源类型       | 描述                                      | 示例                          |
|----------------|-------------------------------------------|-------------------------------|
| **内存空间**   | 代码段、数据段、堆、栈                     | 虚拟地址空间（如 4GB / 256TB）|
| **CPU 状态**   | 寄存器（PC、SP、通用寄存器）              | RIP、RSP、RAX、RBX 等         |
| **文件资源**   | 打开的文件描述符                           | stdin、stdout、打开的文件     |
| **I/O 设备**   | 分配的设备（终端、网络接口）              | TTY、Socket                   |
| **进程 ID**    | 唯一标识符                                 | PID（如 1234）                |
| **权限信息**   | 用户 ID、组 ID、权限位                     | UID、GID、capabilities        |

<div data-component="ProcessResourceVisualization"></div>

---

### 3.1.3 进程的组成：代码、数据、堆、栈、PCB

进程在内存中的典型布局（以 Linux x86-64 为例）：

```
高地址
+---------------------------+  0x00007FFFFFFFFFFF (128TB)
|    内核空间（Kernel）      |
|    (不可访问)              |
+---------------------------+  0x00007FFF00000000
|    栈（Stack）↓           |  ← RSP（栈指针）
|    局部变量、函数调用栈    |
|    ...                    |
|    [栈向下增长]            |
+---------------------------+
|    [空闲区域]              |
|    (未映射)                |
+---------------------------+
|    共享库（Shared Libs）   |
|    libc.so, ld.so         |
+---------------------------+
|    [空闲区域]              |
+---------------------------+
|    堆（Heap）↑            |  ← brk（堆顶指针）
|    动态分配内存 (malloc)   |
|    ...                    |
|    [堆向上增长]            |
+---------------------------+
|    BSS 段（未初始化数据）  |
|    静态变量（初始化为 0）  |
+---------------------------+
|    数据段（Data Segment）  |
|    已初始化的全局变量      |
+---------------------------+
|    代码段（Text Segment）  |
|    程序指令（只读）        |
+---------------------------+  0x0000000000400000
低地址
```

**各部分详解**：

**1. 代码段（Text Segment）**
- **内容**：程序的机器指令（编译后的二进制代码）。
- **权限**：只读、可执行（`r-x`）。
- **共享**：多个进程可以共享同一程序的代码段（节省内存）。
- **示例**：
```c
int add(int a, int b) {
    return a + b;  // 编译为机器指令，存储在代码段
}
```

**2. 数据段（Data Segment）**
- **内容**：已初始化的全局变量和静态变量。
- **权限**：可读写（`rw-`）。
- **示例**：
```c
int global_var = 42;       // 存储在数据段
static int static_var = 100;
```

**3. BSS 段（Block Started by Symbol）**
- **内容**：未初始化的全局变量和静态变量（初始化为 0）。
- **优化**：编译器不在可执行文件中存储 BSS 段内容，加载时由内核分配并清零。
- **示例**：
```c
int uninitialized_array[1000];  // 存储在 BSS 段
```

**4. 堆（Heap）**
- **内容**：动态分配的内存（`malloc()`、`new`）。
- **增长方向**：向上增长（地址增大）。
- **管理**：通过 `brk()` / `sbrk()` / `mmap()` 系统调用扩展。
- **示例**：
```c
int *p = malloc(sizeof(int) * 100);  // 在堆上分配 400 字节
```

**5. 栈（Stack）**
- **内容**：函数调用栈帧、局部变量、函数参数、返回地址。
- **增长方向**：向下增长（地址减小）。
- **大小限制**：通常 8MB（可通过 `ulimit -s` 查看/修改）。
- **示例**：
```c
void func() {
    int local_var = 10;  // 存储在栈上
    int array[100];      // 存储在栈上（400 字节）
}
```

**栈溢出（Stack Overflow）**：
```c
void recursive() {
    int huge_array[1000000];  // 栈空间不足
    recursive();              // 无限递归
}
// 运行时错误：Segmentation fault (core dumped)
```

**6. 共享库（Shared Libraries）**
- **内容**：动态链接库（如 `libc.so.6`、`libpthread.so.0`）。
- **映射**：通过 `mmap()` 映射到进程地址空间。
- **共享**：多个进程共享同一份代码（节省内存）。

**查看进程内存布局**：
```bash
$ cat /proc/self/maps
00400000-00401000 r-xp 00000000 08:01 123456  /bin/cat  # 代码段
00601000-00602000 rw-p 00001000 08:01 123456  /bin/cat  # 数据段
7f8e2a000000-7f8e2a021000 rw-p 00000000 00:00 0         # 堆
7f8e2a400000-7f8e2a5c1000 r-xp 00000000 08:01 234567    # libc.so.6 代码段
7ffef7c00000-7ffef7c21000 rw-p 00000000 00:00 0         # 栈
```

<div data-component="ProcessMemoryLayout"></div>

---

### 3.1.4 进程抽象的价值：并发与隔离

进程抽象为操作系统带来两大核心价值：**并发**和**隔离**。

**1. 并发（Concurrency）**

**问题**：单核 CPU 一次只能执行一条指令，如何让多个程序"同时"运行？

**解决方案**：**时间片轮转**（Time Slicing）
- 操作系统将 CPU 时间划分为短时间片（如 10ms）。
- 每个进程轮流占用 CPU 一个时间片。
- 宏观上，用户感知多个程序同时运行。

**示例**：
```
时间轴：
0-10ms:   进程 A 运行
10-20ms:  进程 B 运行
20-30ms:  进程 C 运行
30-40ms:  进程 A 运行
...
```

**用户视角**：
- 进程 A、B、C "同时"在播放音乐、下载文件、编辑文档。
- 实际上 CPU 在三者之间快速切换（每秒 100 次）。

**并发的好处**：
- **提高 CPU 利用率**：当一个进程等待 I/O 时，CPU 可以执行其他进程。
- **改善响应性**：用户界面保持流畅（如浏览器在下载时仍可滚动页面）。
- **多任务处理**：后台程序（如邮件客户端）在用户编辑文档时继续运行。

**2. 隔离（Isolation）**

**问题**：多个程序共享内存，如何防止相互干扰？

**解决方案**：**虚拟地址空间**（Virtual Address Space）
- 每个进程拥有独立的虚拟地址空间（如 0x0 ~ 0x7FFFFFFFFFFF）。
- 进程 A 的地址 `0x1000` 和进程 B 的地址 `0x1000` 映射到不同的物理内存。
- MMU（内存管理单元）通过页表实现地址翻译，防止进程访问其他进程的内存。

**示例**：
```c
// 进程 A
int *p = (int *)0x1000;
*p = 42;  // 写入进程 A 的地址 0x1000

// 进程 B
int *p = (int *)0x1000;
*p = 100;  // 写入进程 B 的地址 0x1000（不同的物理内存）

// 进程 A 和 B 互不影响
```

**隔离的好处**：
- **安全性**：恶意程序无法读取其他进程的敏感数据（如密码）。
- **稳定性**：一个进程崩溃（如段错误）不影响其他进程。
- **简化编程**：程序员无需担心内存冲突，可以自由使用地址空间。

**隔离的实现机制**：
1. **虚拟内存**：MMU + 页表实现地址翻译。
2. **特权级保护**：用户态进程无法执行特权指令或访问内核内存。
3. **系统调用**：通过受控接口访问内核服务，内核验证参数合法性。

**示例：进程无法访问其他进程内存**：
```c
#include <stdio.h>

int main() {
    int *p = (int *)0xDEADBEEF;  // 随意地址
    printf("%d\n", *p);          // Segmentation fault (core dumped)
    return 0;
}
```

**总结**：
- **并发**：让多个程序看似同时运行，提高资源利用率。
- **隔离**：保护进程互不干扰，确保安全与稳定性。
- **进程抽象**是操作系统最伟大的发明之一，奠定了现代多任务操作系统的基础。

<div data-component="ConcurrencyIsolationDemo"></div>

---

## 3.2 进程控制块（PCB）

### 3.2.1 PCB 的数据结构设计

**进程控制块（Process Control Block，PCB）** 是内核为每个进程维护的**核心数据结构**，存储进程的所有元数据。PCB 是操作系统管理进程的关键，所有进程相关的操作（创建、调度、切换、终止）都依赖 PCB。

**PCB 的设计原则**：
1. **完整性**：包含进程运行所需的所有信息（状态、寄存器、内存、文件等）。
2. **高效性**：频繁访问的字段（如状态、优先级）放在前面，减少缓存未命中。
3. **可扩展性**：支持新功能（如命名空间、cgroup）而不破坏兼容性。

**PCB 的核心字段**（通用操作系统）：

| 字段类型         | 字段名称                  | 描述                                           |
|------------------|---------------------------|------------------------------------------------|
| **进程标识**     | PID                       | 进程 ID（唯一标识符）                           |
|                  | PPID                      | 父进程 ID                                       |
|                  | 进程组 ID                 | 用于作业控制（如管道）                          |
| **进程状态**     | 状态                      | 运行、就绪、阻塞、僵尸等                        |
|                  | 优先级                    | 调度优先级                                      |
|                  | 调度策略                  | FIFO、RR、CFS 等                                |
| **CPU 状态**     | 程序计数器（PC）          | 下一条指令的地址                                |
|                  | 栈指针（SP）              | 栈顶地址                                        |
|                  | 通用寄存器                | RAX、RBX、RCX、RDX 等                           |
|                  | 状态寄存器（EFLAGS）      | 标志位（零标志、进位标志等）                    |
| **内存管理**     | 页表基址                  | 页表起始地址（CR3 寄存器值）                    |
|                  | 虚拟内存区域（VMA）       | 代码段、数据段、堆、栈的地址范围                |
|                  | 内存限制                  | 最大虚拟内存、堆大小限制                        |
| **文件管理**     | 打开文件表                | 文件描述符数组（fd 0-1023）                     |
|                  | 当前工作目录              | `pwd` 的路径                                    |
|                  | 根目录                    | chroot 的根路径                                 |
| **权限与安全**   | 用户 ID（UID）            | 真实 UID、有效 UID、保存的 UID                  |
|                  | 组 ID（GID）              | 真实 GID、有效 GID、保存的 GID                  |
|                  | Capabilities              | Linux 细粒度权限（如 CAP_NET_ADMIN）            |
| **信号处理**     | 信号挂起位图              | 记录已发送但未处理的信号                        |
|                  | 信号处理器表              | 每个信号的处理函数地址                          |
|                  | 信号屏蔽位图              | 被阻塞的信号                                    |
| **资源统计**     | CPU 时间                  | 用户态时间、内核态时间                          |
|                  | 内存使用                  | RSS（常驻内存）、VSZ（虚拟内存）                |
|                  | I/O 统计                  | 读写字节数、系统调用次数                        |
| **进程关系**     | 父进程指针                | 指向父进程 PCB                                  |
|                  | 子进程链表                | 所有子进程的链表                                |
|                  | 兄弟进程链表              | 同一父进程的其他子进程                          |

<div data-component="PCBStructureExplorer"></div>

---

### 3.2.2 关键字段：PID、状态、寄存器、内存指针、打开文件

**1. 进程 ID（PID）**

**定义**：进程的唯一标识符，系统中每个进程的 PID 不同。

**特性**：
- **取值范围**：1 ~ 32768（可配置，最大 4194304）。
- **PID 1**：`init` 或 `systemd`（所有进程的祖先）。
- **PID 复用**：进程终止后，PID 可能被新进程复用（但有延迟，避免混淆）。

**获取 PID**：
```c
#include <unistd.h>
#include <stdio.h>

int main() {
    printf("My PID: %d\n", getpid());
    printf("Parent PID: %d\n", getppid());
    return 0;
}
```

**内核实现**（Linux）：
```c
// include/linux/sched.h
struct task_struct {
    pid_t pid;          // 进程 ID
    pid_t tgid;         // 线程组 ID（主线程的 PID）
    // ...
};
```

**2. 进程状态**

**定义**：进程当前的运行状态（运行、就绪、阻塞等）。

**Linux 进程状态**（`ps aux` 输出的 `STAT` 列）：
| 状态   | 符号  | 描述                              |
|--------|-------|-----------------------------------|
| 运行   | `R`   | Running（正在运行或可运行）       |
| 睡眠   | `S`   | Sleeping（可中断睡眠，等待事件）  |
| 深度睡眠 | `D` | Uninterruptible sleep（不可中断，等待 I/O） |
| 僵尸   | `Z`   | Zombie（已终止，等待父进程回收）  |
| 停止   | `T`   | Stopped（被信号暂停）             |
| 死亡   | `X`   | Dead（正在退出，不可见）          |

**状态转换**（后续章节详细讲解）：
```
新建 → 就绪 → 运行 → 阻塞 → 就绪 → ... → 终止
```

**3. CPU 寄存器**

**定义**：保存进程的 CPU 状态，用于上下文切换。

**关键寄存器**（x86-64）：
- **RIP（Instruction Pointer）**：下一条指令的地址。
- **RSP（Stack Pointer）**：栈顶地址。
- **RBP（Base Pointer）**：栈帧基址。
- **通用寄存器**：RAX、RBX、RCX、RDX、RSI、RDI、R8-R15。
- **RFLAGS**：标志寄存器（零标志 ZF、符号标志 SF、进位标志 CF 等）。

**上下文切换时的保存/恢复**：
```c
// 伪代码
void context_switch(PCB *old, PCB *new) {
    // 保存旧进程的寄存器
    old->regs.rip = read_rip();
    old->regs.rsp = read_rsp();
    old->regs.rax = read_rax();
    // ...
    
    // 加载新进程的寄存器
    write_rip(new->regs.rip);
    write_rsp(new->regs.rsp);
    write_rax(new->regs.rax);
    // ...
}
```

**4. 内存指针**

**定义**：指向进程的页表、内存区域描述符等。

**关键指针**：
- **页表基址（CR3）**：指向顶层页表（PML4）的物理地址。
- **内存描述符（mm_struct）**：管理虚拟内存区域（VMA）。
  - `code_start`/`code_end`：代码段范围。
  - `data_start`/`data_end`：数据段范围。
  - `brk`：堆顶地址。
  - `start_stack`：栈起始地址。

**Linux 实现**（简化）：
```c
struct mm_struct {
    unsigned long start_code, end_code;   // 代码段
    unsigned long start_data, end_data;   // 数据段
    unsigned long start_brk, brk;         // 堆
    unsigned long start_stack;            // 栈
    pgd_t *pgd;                           // 页表基址
    // ...
};
```

**5. 打开文件表**

**定义**：记录进程打开的所有文件（文件描述符表）。

**文件描述符（File Descriptor）**：
- **0**：stdin（标准输入）
- **1**：stdout（标准输出）
- **2**：stderr（标准错误）
- **3+**：其他打开的文件

**数据结构**：
```c
struct files_struct {
    struct file *fd_array[NR_OPEN_DEFAULT];  // 文件指针数组
    int max_fds;                             // 最大文件描述符数
    // ...
};
```

**示例**：
```c
int fd = open("file.txt", O_RDONLY);  // 返回 3
// 进程 PCB 的 fd_array[3] 指向内核文件对象
```

<div data-component="PCBFieldsVisualization"></div>

---

### 3.2.3 xv6 的 struct proc 详解

xv6 的 PCB 数据结构非常简洁，是学习操作系统的绝佳材料。

**xv6 struct proc 定义**（`kernel/proc.h`）：
```c
// Per-process state
struct proc {
    struct spinlock lock;

    // p->lock must be held when using these:
    enum procstate state;        // Process state
    void *chan;                  // If non-zero, sleeping on chan
    int killed;                  // If non-zero, have been killed
    int xstate;                  // Exit status to be returned to parent's wait
    int pid;                     // Process ID

    // wait_lock must be held when using this:
    struct proc *parent;         // Parent process

    // these are private to the process, so p->lock need not be held.
    uint64 kstack;               // Virtual address of kernel stack
    uint64 sz;                   // Size of process memory (bytes)
    pagetable_t pagetable;       // User page table
    struct trapframe *trapframe; // data page for trampoline.S
    struct context context;      // swtch() here to run process
    struct file *ofile[NOFILE];  // Open files
    struct inode *cwd;           // Current directory
    char name[16];               // Process name (debugging)
};
```

**字段详解**：

**1. 状态管理**：
```c
enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };
```

**2. 身份信息**：
- `pid`：进程 ID。
- `parent`：父进程指针（用于 `wait()` 和资源回收）。

**3. 内存管理**：
- `sz`：进程虚拟内存大小（字节数）。
- `pagetable`：用户页表基址。
- `kstack`：内核栈地址（进程在内核态运行时使用）。

**4. CPU 状态**：
- `trapframe`：保存用户态寄存器（系统调用、中断时使用）。
- `context`：保存内核态寄存器（进程切换时使用）。

**trapframe vs context**：
- **trapframe**：用户态 → 内核态切换时保存（系统调用、中断）。
- **context**：进程切换时保存（调度器切换）。

```c
// kernel/proc.h
struct trapframe {
    uint64 kernel_satp;   // kernel page table
    uint64 kernel_sp;     // top of process's kernel stack
    uint64 kernel_trap;   // usertrap()
    uint64 epc;           // saved user program counter
    uint64 kernel_hartid; // saved kernel tp
    uint64 ra;
    uint64 sp;
    uint64 gp;
    uint64 tp;
    uint64 t0;
    // ... 32 个寄存器
};

struct context {
    uint64 ra;  // 返回地址
    uint64 sp;  // 栈指针
    // callee-saved registers
    uint64 s0;
    uint64 s1;
    // ... s2-s11
};
```

**5. 文件管理**：
- `ofile[NOFILE]`：打开文件数组（最多 16 个文件）。
- `cwd`：当前工作目录的 inode。

**6. 调试信息**：
- `name[16]`：进程名称（如 "sh"、"cat"）。

**xv6 进程表**：
```c
struct proc proc[NPROC];  // 进程数组（最多 64 个进程）
```

<div data-component="Xv6ProcStructViewer"></div>

---

### 3.2.4 Linux 的 task_struct 核心字段

Linux 的 PCB（`task_struct`）是世界上最复杂的数据结构之一，包含 **超过 200 个字段**，管理进程的方方面面。

**task_struct 定义**（`include/linux/sched.h`，简化版）：
```c
struct task_struct {
    volatile long state;          // 进程状态
    void *stack;                  // 内核栈指针
    unsigned int flags;           // 进程标志
    
    // 调度相关
    int prio, static_prio, normal_prio;  // 优先级
    const struct sched_class *sched_class;  // 调度器类
    struct sched_entity se;       // CFS 调度实体
    unsigned int policy;          // 调度策略（SCHED_NORMAL、SCHED_FIFO 等）
    
    // 进程标识
    pid_t pid;
    pid_t tgid;                   // 线程组 ID
    
    // 进程关系
    struct task_struct __rcu *parent;       // 父进程
    struct list_head children;              // 子进程链表
    struct list_head sibling;               // 兄弟进程链表
    struct task_struct *group_leader;       // 线程组领导者
    
    // 内存管理
    struct mm_struct *mm;         // 内存描述符
    struct mm_struct *active_mm;  // 活动内存描述符
    
    // 文件系统
    struct fs_struct *fs;         // 文件系统信息（根目录、当前目录）
    struct files_struct *files;   // 打开文件表
    
    // 信号处理
    struct signal_struct *signal;
    struct sighand_struct *sighand;
    sigset_t blocked, real_blocked;
    
    // CPU 状态（x86）
    struct thread_struct thread;  // CPU 寄存器
    
    // 权限
    const struct cred *cred;      // UID、GID、Capabilities
    
    // 资源限制
    struct rlimit rlim[RLIM_NLIMITS];
    
    // 命名空间
    struct nsproxy *nsproxy;      // PID、网络、文件系统命名空间
    
    // cgroup
    struct css_set *cgroups;      // cgroup 控制组
    
    // 性能统计
    u64 utime, stime;             // 用户态/内核态 CPU 时间
    unsigned long nvcsw, nivcsw;  // 自愿/非自愿上下文切换次数
    
    // 其他
    char comm[TASK_COMM_LEN];     // 进程名称（15 字符）
    // ... 超过 100 个字段
};
```

**核心字段分类**：

**1. 调度相关**（约 30 个字段）：
- `policy`：调度策略（`SCHED_NORMAL`、`SCHED_FIFO`、`SCHED_RR`、`SCHED_DEADLINE`）。
- `prio`：动态优先级（调度器使用）。
- `static_prio`：静态优先级（`nice` 值）。
- `se`：CFS 调度实体（虚拟运行时间 `vruntime`）。

**2. 内存管理**（约 20 个字段）：
- `mm`：进程的内存描述符（`mm_struct`）。
  - 页表基址、VMA 链表、堆栈范围、内存统计等。
- `active_mm`：当前活动的内存描述符（内核线程借用用户进程的 `mm`）。

**3. 文件系统**：
- `fs`：文件系统信息（根目录、当前工作目录、umask）。
- `files`：打开文件表（文件描述符数组）。

**4. 信号处理**（约 15 个字段）：
- `signal`：共享信号处理器（线程组共享）。
- `sighand`：信号处理函数表。
- `blocked`：被阻塞的信号集。

**5. 权限与安全**：
- `cred`：凭证（UID、GID、Capabilities、Seccomp 等）。

**6. 命名空间与容器化**：
- `nsproxy`：PID、网络、IPC、UTS、挂载命名空间。
- Docker 容器通过命名空间隔离进程。

**7. cgroup（资源限制）**：
- `cgroups`：CPU 配额、内存限制、I/O 带宽限制。

**task_struct 的内存占用**：
- **大小**：约 **1.7 KB**（Linux 5.x）。
- **位置**：内核栈底部（通过 `current` 宏快速访问）。

**获取当前进程的 task_struct**：
```c
struct task_struct *current = get_current();
printk("Current process: PID=%d, name=%s\n", current->pid, current->comm);
```

<div data-component="LinuxTaskStructExplorer"></div>

---

## 3.3 进程状态模型

### 3.3.1 五状态模型：新建、就绪、运行、阻塞、终止

**进程状态（Process State）** 描述进程在生命周期中的不同阶段。经典的**五状态模型**是理解进程调度的基础。

**五状态模型**：

```
        创建
         ↓
    +----------+
    |   新建   |  (New)
    +----------+
         ↓ 就绪
    +----------+
    |   就绪   |  (Ready)  ← ────────┐
    +----------+                      │
         ↓ 调度                       │ 时间片用完
    +----------+                      │ 或被抢占
    |   运行   |  (Running) ──────────┘
    +----------+
         ↓ 等待事件
    +----------+
    |   阻塞   |  (Blocked/Waiting)
    +----------+
         ↓ 事件发生
    +----------+
    |   终止   |  (Terminated/Exit)
    +----------+
```

**状态详解**：

**1. 新建（New）**
- **定义**：进程正在被创建，PCB 已分配但尚未完全初始化。
- **操作**：
  - 分配 PID。
  - 分配内存（代码段、数据段、堆、栈）。
  - 初始化 PCB（状态、寄存器、文件描述符等）。
- **转换**：初始化完成 → **就绪**。

**2. 就绪（Ready）**
- **定义**：进程已准备好运行，等待 CPU 调度。
- **特征**：
  - 进程拥有运行所需的所有资源（内存、文件），唯独缺少 CPU。
  - 多个就绪进程在**就绪队列**中排队。
- **转换**：
  - 调度器选中 → **运行**。

**3. 运行（Running）**
- **定义**：进程正在 CPU 上执行指令。
- **特征**：
  - 单核 CPU 同一时刻只有一个进程处于运行状态。
  - 多核 CPU 可以有多个进程同时运行（每个核一个）。
- **转换**：
  - 时间片用完 → **就绪**（被抢占）。
  - 等待 I/O（如 `read()`） → **阻塞**。
  - 进程终止（`exit()`） → **终止**。

**4. 阻塞（Blocked/Waiting）**
- **定义**：进程等待某个事件发生（如 I/O 完成、信号到达、子进程终止）。
- **特征**：
  - 即使 CPU 空闲，阻塞进程也不能运行。
  - 多个阻塞进程在**等待队列**中（按等待事件分类）。
- **等待事件示例**：
  - **I/O 操作**：`read()` 等待磁盘读取。
  - **信号**：`pause()` 等待信号。
  - **子进程终止**：`wait()` 等待子进程退出。
  - **锁**：等待互斥锁释放。
- **转换**：
  - 事件发生（如 I/O 完成） → **就绪**。

**5. 终止（Terminated/Exit）**
- **定义**：进程已结束执行，等待父进程回收资源。
- **操作**：
  - 释放内存、关闭文件。
  - 保留 PCB（父进程需要读取退出码）。
- **僵尸进程**：
  - 终止但未被回收的进程（`ps aux` 显示为 `Z` 状态）。
  - 父进程调用 `wait()` 后，内核删除 PCB。

**状态转换图**：

```
          +-------+
          |  新建 |
          +-------+
              ↓ 就绪
          +-------+
     ┌────| 就绪  |←────┐
     │    +-------+     │ 时间片用完/抢占
     │调度    │         │
     ↓        ↓ 运行    │
   +-------+ +-------+  │
   | 运行  |→| 阻塞  |──┘ 事件发生
   +-------+ +-------+
       ↓ 终止    ↓ 事件发生
   +-------+     ↓ 就绪
   | 终止  |←────┘
   +-------+
```

<div data-component="ProcessStateTransition"></div>

---

### 3.3.2 七状态模型：增加挂起就绪、挂起阻塞

**七状态模型** 引入**挂起（Suspend）** 概念，允许操作系统将进程暂时移出内存（swap out），释放内存给其他进程。

**挂起的动机**：
- **内存不足**：物理内存紧张，将低优先级进程移到磁盘（交换分区）。
- **调试需求**：暂停进程以便调试（`SIGSTOP` 信号）。
- **系统维护**：暂停后台服务。

**七状态模型**：

```
新建 → 就绪 ⇄ 挂起就绪
        ↓ ↑
       运行
        ↓ ↑
       阻塞 ⇄ 挂起阻塞
        ↓
       终止
```

**新增状态**：

**6. 挂起就绪（Suspended Ready）**
- **定义**：就绪进程被移出内存（swap out），存储在磁盘交换区。
- **特征**：
  - 不占用物理内存。
  - 一旦被换入（swap in），立即进入**就绪**状态。
- **转换**：
  - 内存充足 → **就绪**。
  - 系统恢复 → **就绪**。

**7. 挂起阻塞（Suspended Blocked）**
- **定义**：阻塞进程被移出内存。
- **特征**：
  - 等待事件发生 + 不占用物理内存。
- **转换**：
  - 事件发生 → **挂起就绪**（仍在磁盘，但可运行）。
  - 换入内存 → **阻塞**（等待事件）。

**状态转换表**：

| 当前状态       | 事件                 | 新状态           |
|----------------|----------------------|------------------|
| 就绪           | 内存不足，swap out   | 挂起就绪         |
| 挂起就绪       | swap in              | 就绪             |
| 运行           | 等待 I/O             | 阻塞             |
| 阻塞           | swap out             | 挂起阻塞         |
| 挂起阻塞       | 事件发生             | 挂起就绪         |
| 挂起阻塞       | swap in              | 阻塞             |

**Linux 的挂起实现**：
- **OOM Killer**：内存不足时杀死低优先级进程（而非挂起）。
- **Swap 机制**：将不活跃页换出到磁盘（按页粒度，非整个进程）。
- **SIGSTOP/SIGCONT**：信号控制进程暂停/恢复（非挂起到磁盘）。

<div data-component="SevenStateModel"></div>

---

### 3.3.3 状态转换条件与触发事件

**状态转换的触发机制**：

| 转换            | 触发事件                                  | 示例代码                              |
|-----------------|-------------------------------------------|---------------------------------------|
| 新建 → 就绪     | 进程初始化完成                             | `fork()` 返回后子进程进入就绪         |
| 就绪 → 运行     | 调度器选中进程                             | 时间片轮转、优先级调度                |
| 运行 → 就绪     | 时间片用完或被高优先级进程抢占             | 10ms 时间片耗尽                       |
| 运行 → 阻塞     | 进程主动等待事件                           | `read(fd, buf, n)` 等待磁盘 I/O       |
| 阻塞 → 就绪     | 等待的事件发生                             | 磁盘 I/O 完成，中断唤醒进程           |
| 运行 → 终止     | 进程终止                                   | `exit(0)` 或 `return 0` from main     |

**示例：read() 系统调用的状态转换**

```c
// 用户程序
char buf[1024];
int n = read(fd, buf, 1024);  // 阻塞等待数据
printf("Read %d bytes\n", n);
```

**内核视角的状态转换**：

```
1. 进程处于 [运行] 状态，执行 read() 系统调用
2. 内核检查数据是否就绪
   - 如果数据在缓冲区：立即返回，保持 [运行] 状态
   - 如果数据未就绪（需从磁盘读取）：
     a. 进程状态改为 [阻塞]
     b. 进程加入磁盘 I/O 等待队列
     c. 调度器选择其他进程运行
3. 磁盘 I/O 完成，触发中断
   - 中断处理程序将进程从等待队列移除
   - 进程状态改为 [就绪]
   - 进程加入就绪队列
4. 调度器选中进程
   - 进程状态改为 [运行]
   - 从 read() 系统调用返回，继续执行
```

**内核代码示例**（简化版）：

```c
// kernel/fs/read.c
ssize_t sys_read(int fd, void *buf, size_t count) {
    struct file *file = get_file(fd);
    
    if (data_ready(file)) {
        // 数据就绪，直接返回
        return copy_data_to_user(buf, file, count);
    } else {
        // 数据未就绪，阻塞等待
        current->state = TASK_INTERRUPTIBLE;  // 设置为阻塞状态
        add_wait_queue(&file->wait_queue, &wait);  // 加入等待队列
        schedule();  // 放弃 CPU，调度其他进程
        
        // 被唤醒后继续执行
        remove_wait_queue(&file->wait_queue, &wait);
        return copy_data_to_user(buf, file, count);
    }
}
```

<div data-component="StateTransitionTriggers"></div>

---

### 3.3.4 僵尸进程（Zombie）与孤儿进程（Orphan）

**僵尸进程（Zombie Process）**

**定义**：已终止但父进程尚未调用 `wait()` 回收的进程。

**特征**：
- **状态**：`Z`（Zombie）。
- **资源**：已释放内存、文件，仅保留 PCB（包含退出码）。
- **危害**：占用 PID 资源（系统 PID 有限），大量僵尸进程耗尽 PID。

**产生原因**：
父进程未调用 `wait()` / `waitpid()` 回收子进程。

**示例**：
```c
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程：立即退出
        printf("Child exiting\n");
        exit(0);
    } else {
        // 父进程：不调用 wait()，直接睡眠
        printf("Parent sleeping (not calling wait)\n");
        sleep(60);  // 子进程变成僵尸
    }
    
    return 0;
}
```

**查看僵尸进程**：
```bash
$ ps aux | grep Z
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
user      1234  0.0  0.0      0     0 ?        Z    10:00   0:00 [child] <defunct>
```

**修复僵尸进程**：
```c
// 方法 1：父进程调用 wait()
int status;
wait(&status);  // 回收子进程

// 方法 2：信号处理器自动回收
#include <signal.h>

void sigchld_handler(int sig) {
    while (waitpid(-1, NULL, WNOHANG) > 0);  // 非阻塞回收所有子进程
}

signal(SIGCHLD, sigchld_handler);
```

**孤儿进程（Orphan Process）**

**定义**：父进程先于子进程终止，子进程被 `init` 进程（PID 1）收养。

**特征**：
- **父进程**：变为 `init`（PID 1）或 `systemd`。
- **无害**：`init` 进程会自动回收所有孤儿进程，不会产生僵尸。

**示例**：
```c
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程：睡眠 10 秒
        sleep(10);
        printf("Child (PID=%d) running, parent PID=%d\n", getpid(), getppid());
    } else {
        // 父进程：立即退出
        printf("Parent (PID=%d) exiting\n", getpid());
        exit(0);
    }
    
    return 0;
}
```

**运行结果**：
```bash
Parent (PID=1234) exiting
# 10 秒后
Child (PID=1235) running, parent PID=1  # 父进程变为 init (PID 1)
```

**init 进程的职责**：
- 周期性调用 `wait()` 回收所有孤儿进程。
- 防止系统积累僵尸进程。

<div data-component="ZombieOrphanDemo"></div>

---

## 3.4 进程创建

### 3.4.1 fork() 系统调用详解

**fork()** 是 Unix/Linux 创建进程的**唯一方式**（Windows 使用 `CreateProcess()`）。

**fork() 的语义**：
```c
#include <unistd.h>

pid_t fork(void);
```

**返回值**：
- **父进程**：返回子进程的 PID（正数）。
- **子进程**：返回 0。
- **失败**：返回 -1（如系统进程数达到上限）。

**fork() 的行为**：
1. **创建子进程**：内核分配新 PCB、PID、内存空间。
2. **复制父进程**：子进程获得父进程的几乎完整副本（代码、数据、堆、栈、文件描述符等）。
3. **差异**：
   - PID 不同。
   - 父进程指针不同（子进程的 `parent` 指向父进程）。
   - `fork()` 返回值不同。

**示例**：
```c
#include <stdio.h>
#include <unistd.h>

int main() {
    int x = 100;
    
    printf("Before fork: x=%d\n", x);
    
    pid_t pid = fork();
    
    if (pid < 0) {
        perror("fork failed");
        return 1;
    } else if (pid == 0) {
        // 子进程
        x = 200;
        printf("Child: PID=%d, x=%d\n", getpid(), x);
    } else {
        // 父进程
        x = 300;
        printf("Parent: PID=%d, Child PID=%d, x=%d\n", getpid(), pid, x);
    }
    
    printf("After fork: PID=%d, x=%d\n", getpid(), x);
    
    return 0;
}
```

**输出**：
```
Before fork: x=100
Parent: PID=1234, Child PID=1235, x=300
After fork: PID=1234, x=300
Child: PID=1235, x=200
After fork: PID=1235, x=200
```

**关键点**：
- `Before fork` 只打印一次（fork 之前）。
- `After fork` 打印两次（父进程和子进程各一次）。
- 父子进程的 `x` 互不影响（独立地址空间）。

<div data-component="ForkBehaviorDemo"></div>

---

### 3.4.2 父子进程的关系：内存复制 vs 共享

**fork() 后父子进程的内存关系**：

**早期实现（直接复制）**：
- 子进程完整复制父进程的所有内存页（代码、数据、堆、栈）。
- **缺点**：
  - 慢（复制大量内存）。
  - 浪费（子进程可能立即 `exec()` 替换内存）。

**现代实现（写时复制 COW）**：
- **共享只读页**：代码段等只读页直接共享，不复制。
- **写时复制**：可写页（数据段、堆、栈）初始共享，首次写入时才复制。

**示例**：
```c
int main() {
    int *p = malloc(4096);  // 分配堆内存
    *p = 100;
    
    fork();
    
    // fork 后，父子进程共享堆内存页（但页表项标记为只读）
    // 首次写入时触发缺页异常（Page Fault），内核复制页
    
    *p = 200;  // 触发写时复制，父子进程各有独立的页
}
```

**写时复制的优势**：
- **性能**：fork() 几乎不复制内存，速度快（~1ms）。
- **节省内存**：只复制实际修改的页。

**共享与独立资源总结**：

| 资源类型       | 共享/独立   | 说明                                      |
|----------------|-------------|-------------------------------------------|
| **PID**        | 独立        | 子进程有新 PID                             |
| **代码段**     | 共享        | 只读，父子进程共享同一物理页               |
| **数据段**     | COW 共享    | 写时复制                                  |
| **堆**         | COW 共享    | 写时复制                                  |
| **栈**         | COW 共享    | 写时复制                                  |
| **文件描述符** | 共享        | 父子进程共享同一文件表项（引用计数 +1）    |
| **打开文件偏移**| 共享       | 父子进程共享文件偏移（一个 `lseek()` 影响另一个）|
| **信号处理器** | 继承        | 子进程继承父进程的信号处理器               |
| **环境变量**   | 继承        | 子进程继承父进程的 `environ`               |

**文件描述符共享示例**：
```c
int main() {
    int fd = open("file.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    
    fork();
    
    // 父子进程都可以写入同一文件
    write(fd, "Hello\n", 6);  // 两个进程都写入，顺序不确定
    
    close(fd);  // 引用计数 -1，两个进程都关闭后文件才真正关闭
    return 0;
}
```

<div data-component="MemorySharingVisualization"></div>

---

### 3.4.3 写时复制（Copy-on-Write）优化

**写时复制（COW）** 是操作系统最重要的性能优化之一，广泛用于 `fork()`、内存管理、文件系统等场景。

**COW 的核心思想**：
- **延迟复制**：不立即复制内存，标记页为只读。
- **按需复制**：首次写入时触发缺页异常，此时才真正复制页。

**COW 的实现流程**：

**1. fork() 时**：
- 子进程的页表指向与父进程相同的物理页。
- 将所有可写页的页表项（PTE）标记为**只读**（清除 `PTE_W` 位）。
- 页表项增加**写时复制标志**（内核自定义位）。

**2. 首次写入时**：
- CPU 尝试写入只读页，触发**缺页异常**（Page Fault）。
- 内核捕获异常，检查页表项：
  - 如果是 COW 页（有写时复制标志），执行复制。
  - 分配新物理页。
  - 复制原页内容到新页。
  - 更新页表，指向新页，恢复可写权限。
  - 返回用户态，重新执行写指令（此时成功）。

**3. 引用计数**：
- 每个物理页维护引用计数（多少个页表项指向它）。
- 写时复制时，原页引用计数 -1，新页引用计数 = 1。
- 引用计数为 0 时，释放物理页。

**伪代码**：
```c
// fork() 时
void fork_cow() {
    for (each page in parent) {
        child.pagetable[vpn] = parent.pagetable[vpn];  // 共享物理页
        parent.pagetable[vpn].writable = false;        // 标记为只读
        child.pagetable[vpn].writable = false;
        page_refcount[pfn]++;                          // 引用计数 +1
    }
}

// 缺页异常处理
void page_fault_handler(addr) {
    pte = lookup_pte(addr);
    
    if (pte.cow_flag) {  // 写时复制页
        new_page = alloc_page();                // 分配新页
        copy_page(pte.pfn, new_page);           // 复制内容
        pte.pfn = new_page;                     // 更新页表
        pte.writable = true;                    // 恢复可写
        pte.cow_flag = false;                   // 清除 COW 标志
        page_refcount[old_pfn]--;               // 旧页引用计数 -1
        if (page_refcount[old_pfn] == 0)
            free_page(old_pfn);                 // 释放旧页
    }
}
```

**COW 的性能优势**：

**测试代码**：
```c
#include <sys/time.h>
#include <unistd.h>

int main() {
    char *p = malloc(1024 * 1024 * 1024);  // 1GB 内存
    memset(p, 0, 1024 * 1024 * 1024);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    fork();
    
    gettimeofday(&end, NULL);
    long ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000;
    
    printf("fork() time: %ld ms\n", ms);
    return 0;
}
```

**结果**：
- **无 COW**（直接复制 1GB）：~1000ms。
- **有 COW**：~1ms（仅复制页表）。

**COW 的其他应用**：
1. **exec() 优化**：fork() 后立即 exec() 的场景（如 shell 执行命令），子进程不需要复制内存。
2. **文件系统**：Btrfs、ZFS 使用 COW 实现快照（snapshot）。
3. **虚拟化**：KVM 使用 COW 共享虚拟机内存。

<div data-component="CopyOnWriteAnimation"></div>

---

### 3.4.4 fork() 的返回值语义

**fork() 返回值的巧妙设计**：

**为什么父进程返回子进程 PID，子进程返回 0？**

**设计理由**：
1. **父进程需要管理子进程**：
   - 父进程需要知道子进程 PID 以便调用 `wait()`、`kill()` 等系统调用。
   - 示例：
   ```c
   pid_t pid = fork();
   if (pid > 0) {
       // 父进程等待子进程
       waitpid(pid, &status, 0);
   }
   ```

2. **子进程无需知道自己的 PID**（可通过 `getpid()` 获取）：
   - 子进程主要需要区分"我是父进程还是子进程"。
   - 返回 0 是最简单的区分标志。

3. **统一的错误处理**：
   - 返回 -1 表示失败（父进程检测）。

**返回值的完整逻辑**：
```c
pid_t pid = fork();

if (pid < 0) {
    // 错误：fork() 失败
    perror("fork");
    exit(1);
} else if (pid == 0) {
    // 子进程
    printf("I am child, PID=%d\n", getpid());
} else {
    // 父进程
    printf("I am parent, child PID=%d\n", pid);
}
```

**常见错误**：
```c
// 错误 1：混淆父子进程
if (fork()) {
    // 这是父进程（pid > 0），不是子进程！
    printf("Child running\n");  // 错误！
}

// 正确写法
if (fork() == 0) {
    printf("Child running\n");
}
```

**经典问题：fork() 炸弹**

```c
int main() {
    while (1)
        fork();  // 无限创建进程，耗尽系统资源
}
```

**运行后果**：
- 进程数指数增长：1 → 2 → 4 → 8 → 16 → ...
- 系统资源耗尽，无法创建新进程（包括登录 shell）。
- **防御**：`ulimit -u` 限制用户最大进程数。

<div data-component="ForkReturnValueDemo"></div>

---

### 3.4.5 xv6 fork() 实现源码剖析

xv6 的 `fork()` 实现是学习操作系统的经典材料，代码简洁但完整。

**完整实现**（`kernel/proc.c`）：

```c
// Create a new process, copying the parent.
// Sets up child kernel stack to return as if from fork() system call.
int fork(void)
{
  int i, pid;
  struct proc *np;
  struct proc *p = myproc();

  // Allocate process.
  if((np = allocproc()) == 0){
    return -1;
  }

  // Copy user memory from parent to child.
  if(uvmcopy(p->pagetable, np->pagetable, p->sz) < 0){
    freeproc(np);
    release(&np->lock);
    return -1;
  }
  np->sz = p->sz;

  // copy saved user registers.
  *(np->trapframe) = *(p->trapframe);

  // Cause fork to return 0 in the child.
  np->trapframe->a0 = 0;

  // increment reference counts on open file descriptors.
  for(i = 0; i < NOFILE; i++)
    if(p->ofile[i])
      np->ofile[i] = filedup(p->ofile[i]);
  np->cwd = idup(p->cwd);

  safestrcpy(np->name, p->name, sizeof(p->name));

  pid = np->pid;

  release(&np->lock);

  acquire(&wait_lock);
  np->parent = p;
  release(&wait_lock);

  acquire(&np->lock);
  np->state = RUNNABLE;
  release(&np->lock);

  return pid;
}
```

**逐步解析**：

**1. 分配新进程（allocproc）**：
```c
static struct proc* allocproc(void)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == UNUSED) {
      goto found;
    } else {
      release(&p->lock);
    }
  }
  return 0;

found:
  p->pid = allocpid();
  p->state = USED;

  // Allocate a trapframe page.
  if((p->trapframe = (struct trapframe *)kalloc()) == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // An empty user page table.
  p->pagetable = proc_pagetable(p);
  if(p->pagetable == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&p->context, 0, sizeof(p->context));
  p->context.ra = (uint64)forkret;
  p->context.sp = p->kstack + PGSIZE;

  return p;
}
```

**关键操作**：
- 在进程表中找到空闲槽位。
- 分配 PID（`allocpid()`）。
- 分配 `trapframe`（保存用户态寄存器）。
- 创建空页表（`proc_pagetable()`）。
- 设置上下文：返回地址 = `forkret`，栈指针 = 内核栈顶。

**2. 复制内存（uvmcopy）**：
```c
int uvmcopy(pagetable_t old, pagetable_t new, uint64 sz)
{
  pte_t *pte;
  uint64 pa, i;
  uint flags;
  char *mem;

  for(i = 0; i < sz; i += PGSIZE){
    if((pte = walk(old, i, 0)) == 0)
      panic("uvmcopy: pte should exist");
    if((*pte & PTE_V) == 0)
      panic("uvmcopy: page not present");
    pa = PTE2PA(*pte);
    flags = PTE_FLAGS(*pte);
    if((mem = kalloc()) == 0)
      goto err;
    memmove(mem, (char*)pa, PGSIZE);  // 复制页内容
    if(mappages(new, i, PGSIZE, (uint64)mem, flags) != 0){
      kfree(mem);
      goto err;
    }
  }
  return 0;

 err:
  uvmunmap(new, 0, i / PGSIZE, 1);
  return -1;
}
```

**关键操作**：
- 遍历父进程的所有页。
- 为每一页分配新物理页（`kalloc()`）。
- 复制页内容（`memmove()`）。
- 映射到子进程页表（`mappages()`）。

**注意**：xv6 不使用写时复制（为简化实现）。

**3. 复制 trapframe**：
```c
*(np->trapframe) = *(p->trapframe);
```

**作用**：
- 子进程继承父进程的所有寄存器（PC、SP、通用寄存器等）。
- 子进程从与父进程相同的位置继续执行。

**4. 设置子进程返回值**：
```c
np->trapframe->a0 = 0;
```

**作用**：
- `a0` 寄存器用于系统调用返回值（RISC-V）。
- 子进程的 `fork()` 返回 0。

**5. 复制文件描述符**：
```c
for(i = 0; i < NOFILE; i++)
  if(p->ofile[i])
    np->ofile[i] = filedup(p->ofile[i]);
```

**作用**：
- 子进程继承父进程打开的所有文件。
- `filedup()` 增加文件对象的引用计数。

**6. 设置进程关系**：
```c
np->parent = p;
```

**作用**：
- 建立父子关系（用于 `wait()` 和资源回收）。

**7. 设置为可运行**：
```c
np->state = RUNNABLE;
```

**作用**：
- 子进程进入就绪队列，等待调度。

**8. 返回子进程 PID**：
```c
return pid;  // 父进程返回子进程 PID
```

<div data-component="Xv6ForkFlowchart"></div>

---

## 3.5 进程执行与终止

### 3.5.1 exec() 系列系统调用：execl()、execv()、execve()

**exec() 系列**用于将当前进程的内存映像替换为新程序。

**exec 系列函数**：
```c
#include <unistd.h>

int execl(const char *path, const char *arg0, ..., NULL);
int execle(const char *path, const char *arg0, ..., NULL, char *const envp[]);
int execlp(const char *file, const char *arg0, ..., NULL);
int execv(const char *path, char *const argv[]);
int execve(const char *path, char *const argv[], char *const envp[]);
int execvp(const char *file, char *const argv[]);
```

**命名规则**：
- `l`（list）：参数以变长参数列表传递（`arg0, arg1, ..., NULL`）。
- `v`（vector）：参数以数组传递（`argv[]`）。
- `e`（environment）：显式传递环境变量数组（`envp[]`）。
- `p`（path）：在 `PATH` 环境变量中搜索可执行文件。

**execl() 示例**：
```c
#include <unistd.h>

int main() {
    execl("/bin/ls", "ls", "-l", "/tmp", NULL);
    
    // 如果 execl 成功，下面的代码不会执行
    perror("execl failed");
    return 1;
}
```

**execv() 示例**：
```c
char *argv[] = {"ls", "-l", "/tmp", NULL};
execv("/bin/ls", argv);
```

**execlp() 示例**（在 PATH 中搜索）：
```c
execlp("ls", "ls", "-l", "/tmp", NULL);
// 等价于 execl("/bin/ls", ...)
```

**execve() 示例**（传递环境变量）：
```c
char *argv[] = {"env", NULL};
char *envp[] = {"PATH=/bin:/usr/bin", "HOME=/home/user", NULL};
execve("/usr/bin/env", argv, envp);
```

**exec() 的底层实现**：
所有 `exec` 变体最终调用 `execve()` 系统调用：
```c
int execve(const char *filename, char *const argv[], char *const envp[]);
```

**glibc 封装**（`execl()` 的实现）：
```c
int execl(const char *path, const char *arg0, ...) {
    va_list ap;
    char *argv[MAX_ARGS];
    int i = 0;
    
    argv[i++] = (char *)arg0;
    va_start(ap, arg0);
    while ((argv[i++] = va_arg(ap, char *)) != NULL);
    va_end(ap);
    
    return execve(path, argv, environ);  // 调用 execve 系统调用
}
```

<div data-component="ExecVariantsComparison"></div>

---

### 3.5.2 exec() 的内存替换过程

**exec() 的关键操作**：

**1. 加载新程序**：
- 打开可执行文件（ELF 格式）。
- 解析 ELF 头部：
  - 代码段（Text Segment）地址和大小。
  - 数据段（Data Segment）地址和大小。
  - 入口点（Entry Point，`main()` 函数地址）。

**2. 释放旧内存**：
- 释放旧进程的代码段、数据段、堆、栈。
- **保留**：文件描述符（除非设置 `FD_CLOEXEC`）、PID、父进程、权限等。

**3. 创建新内存布局**：
- 分配代码段：只读、可执行。
- 分配数据段：可读写。
- 分配堆：初始为空（`brk` 指向数据段末尾）。
- 分配栈：压入 `argv`、`envp`、辅助向量（auxiliary vector）。

**4. 设置寄存器**：
- `RIP`（指令指针）= 入口点地址（`_start` 或 `main`）。
- `RSP`（栈指针）= 栈顶地址。
- 清空通用寄存器。

**5. 跳转到新程序**：
- CPU 开始执行新程序的 `_start` 函数（C 运行时入口）。

**exec() 的内存变化**：

**执行 exec() 前（假设进程正在运行 `/bin/sh`）**：
```
+---------------------------+
|    内核空间               |
+---------------------------+
|    栈（sh）               |
|    sh 的局部变量          |
+---------------------------+
|    堆（sh）               |
|    sh 动态分配的内存      |
+---------------------------+
|    数据段（sh）           |
|    sh 的全局变量          |
+---------------------------+
|    代码段（sh）           |
|    sh 的机器指令          |
+---------------------------+
```

**执行 `execl("/bin/ls", "ls", "-l", NULL)` 后**：
```
+---------------------------+
|    内核空间               |
+---------------------------+
|    栈（ls）               |
|    argv = ["ls", "-l"]    |
+---------------------------+
|    堆（ls）               |
|    （初始为空）           |
+---------------------------+
|    数据段（ls）           |
|    ls 的全局变量          |
+---------------------------+
|    代码段（ls）           |
|    ls 的机器指令          |
+---------------------------+
```

**关键点**：
- sh 的所有内存被释放。
- ls 的代码从磁盘加载到内存。
- 进程 PID、打开的文件描述符保持不变。

**exec() 的原子性**：
- 如果 `exec()` 失败（如文件不存在），原进程的内存保持不变。
- `exec()` 要么完全成功（替换内存），要么完全失败（返回 -1）。

<div data-component="ExecMemoryReplacement"></div>

---

### 3.5.3 wait() 与 waitpid()：父进程等待子进程

**wait() 系列系统调用**用于父进程等待子进程终止并回收资源。

**函数原型**：
```c
#include <sys/wait.h>

pid_t wait(int *status);
pid_t waitpid(pid_t pid, int *status, int options);
```

**wait() 的行为**：
- **阻塞**：父进程阻塞，直到任意子进程终止。
- **回收**：回收子进程资源（释放 PCB、PID）。
- **返回**：返回终止的子进程 PID。
- **status**：通过指针返回子进程退出状态（退出码、信号等）。

**示例**：
```c
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程
        printf("Child: PID=%d\n", getpid());
        sleep(2);
        return 42;  // 退出码 42
    } else {
        // 父进程
        int status;
        pid_t child_pid = wait(&status);
        
        printf("Child %d terminated\n", child_pid);
        
        if (WIFEXITED(status)) {
            printf("Exit code: %d\n", WEXITSTATUS(status));
        }
    }
    
    return 0;
}
```

**输出**：
```
Child: PID=1235
Child 1235 terminated
Exit code: 42
```

**waitpid() 的高级功能**：

**1. 等待特定子进程**：
```c
waitpid(1235, &status, 0);  // 等待 PID 1235 的子进程
```

**2. 非阻塞等待**：
```c
pid_t pid = waitpid(-1, &status, WNOHANG);  // 不阻塞
if (pid == 0) {
    printf("No child terminated yet\n");
} else if (pid > 0) {
    printf("Child %d terminated\n", pid);
}
```

**3. 等待任意子进程**：
```c
waitpid(-1, &status, 0);  // 等价于 wait(&status)
```

**4. 等待进程组**：
```c
waitpid(-pgid, &status, 0);  // 等待进程组 pgid 的任意子进程
```

**status 宏**：

| 宏                    | 作用                                  |
|-----------------------|---------------------------------------|
| `WIFEXITED(status)`   | 子进程正常退出（调用 `exit()` 或 `return`）？|
| `WEXITSTATUS(status)` | 获取退出码（0-255）                   |
| `WIFSIGNALED(status)` | 子进程被信号终止？                     |
| `WTERMSIG(status)`    | 获取终止信号编号                       |
| `WIFSTOPPED(status)`  | 子进程被停止（`SIGSTOP`）？            |
| `WSTOPSIG(status)`    | 获取停止信号编号                       |

**示例**：
```c
int status;
waitpid(pid, &status, 0);

if (WIFEXITED(status)) {
    printf("Normal exit, code=%d\n", WEXITSTATUS(status));
} else if (WIFSIGNALED(status)) {
    printf("Killed by signal %d\n", WTERMSIG(status));
} else if (WIFSTOPPED(status)) {
    printf("Stopped by signal %d\n", WSTOPSIG(status));
}
```

<div data-component="WaitBehaviorDemo"></div>

---

### 3.5.4 exit() 与进程清理

**exit() 的作用**：
- 终止进程。
- 执行清理工作（刷新 stdio 缓冲区、调用 `atexit()` 注册的函数）。
- 设置退出码（0-255）。

**exit() vs _exit()**：
```c
#include <stdlib.h>
void exit(int status);  // 库函数

#include <unistd.h>
void _exit(int status);  // 系统调用
```

**区别**：

| 特性           | exit()                              | _exit()                           |
|----------------|-------------------------------------|-----------------------------------|
| **类型**       | C 库函数                            | 系统调用                          |
| **清理工作**   | 执行（刷新缓冲区、调用 `atexit()`）| 不执行                            |
| **速度**       | 慢                                  | 快                                |
| **使用场景**   | 正常退出                             | fork() 后子进程立即退出            |

**示例**：
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void cleanup() {
    printf("Cleanup called\n");
}

int main() {
    atexit(cleanup);  // 注册清理函数
    
    printf("Message 1");  // 无换行符，留在缓冲区
    
    exit(0);  // 刷新缓冲区，调用 cleanup()
    // 输出：Message 1Cleanup called
}
```

**使用 _exit() 的错误示例**：
```c
int main() {
    printf("Message 1");  // 无换行符
    _exit(0);  // 缓冲区未刷新
    // 输出：（空）
}
```

**atexit() 的使用**：
```c
#include <stdlib.h>

int atexit(void (*function)(void));
```

**示例**：
```c
void cleanup1() { printf("Cleanup 1\n"); }
void cleanup2() { printf("Cleanup 2\n"); }

int main() {
    atexit(cleanup1);
    atexit(cleanup2);
    
    printf("Main\n");
    return 0;
}

// 输出：
// Main
// Cleanup 2  （后注册的先执行，LIFO）
// Cleanup 1
```

**进程终止的完整流程**：

**1. 调用 exit()**：
```c
exit(42);
```

**2. 执行清理工作**：
- 调用 `atexit()` 注册的函数（LIFO 顺序）。
- 刷新所有打开的 stdio 流（`fflush()`）。
- 关闭所有打开的流（`fclose()`）。

**3. 调用 _exit() 系统调用**：
```c
_exit(42);
```

**4. 内核处理**：
- 关闭所有文件描述符。
- 释放内存（代码段、数据段、堆、栈）。
- 设置进程状态为 `ZOMBIE`。
- 向父进程发送 `SIGCHLD` 信号。
- 保留 PCB（包含退出码），等待父进程 `wait()`。

**5. 父进程回收**：
- 调用 `wait(&status)` 读取退出码。
- 内核释放子进程 PCB、PID。

<div data-component="ExitCleanupFlow"></div>

---

### 3.5.5 进程终止的资源回收

**进程终止时的资源回收**分为两个阶段：

**阶段 1：进程自身清理**（exit() 调用时）：

1. **关闭文件描述符**：
   - 关闭所有打开的文件（减少文件引用计数）。
   - 关闭套接字、管道等。

2. **释放内存**：
   - 释放代码段、数据段、BSS 段、堆、栈的物理内存。
   - 释放页表。

3. **释放其他资源**：
   - 释放信号量、共享内存（减少引用计数）。
   - 释放 IPC 资源。

4. **设置为僵尸状态**：
   - 进程状态 → `ZOMBIE`。
   - 保留 PCB（包含 PID、退出码、CPU 使用时间等）。

5. **通知父进程**：
   - 向父进程发送 `SIGCHLD` 信号。
   - 如果父进程已终止，子进程被 `init` 进程收养。

**阶段 2：父进程回收**（wait() 调用时）：

1. **读取退出状态**：
   - 父进程通过 `wait(&status)` 读取子进程退出码。

2. **释放 PCB**：
   - 内核删除子进程 PCB。
   - 释放 PID（可被新进程复用）。

3. **更新统计信息**：
   - 累加子进程的 CPU 时间到父进程。

**孤儿进程的回收**：

如果父进程先于子进程终止：
- 子进程被 `init` 进程（PID 1）收养。
- `init` 进程周期性调用 `wait()`，回收所有孤儿进程。

**init 进程的简化实现**：
```c
// init 进程的主循环
void init_main() {
    while (1) {
        // 非阻塞回收所有子进程
        while (waitpid(-1, NULL, WNOHANG) > 0);
        
        // 执行其他任务（启动服务等）
        sleep(1);
    }
}
```

**资源泄漏的防止**：

**错误示例**（产生僵尸进程）：
```c
int main() {
    for (int i = 0; i < 1000; i++) {
        if (fork() == 0) {
            exit(0);  // 子进程立即退出
        }
        // 父进程不调用 wait()
    }
    sleep(60);  // 产生 1000 个僵尸进程
}
```

**正确示例**（使用信号处理器回收）：
```c
#include <signal.h>
#include <sys/wait.h>

void sigchld_handler(int sig) {
    // 非阻塞回收所有终止的子进程
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

int main() {
    signal(SIGCHLD, sigchld_handler);
    
    for (int i = 0; i < 1000; i++) {
        if (fork() == 0) {
            exit(0);
        }
    }
    
    sleep(60);  // 子进程被自动回收，无僵尸进程
}
```

<div data-component="ResourceCleanupTimeline"></div>

---

## 本章小结

本章深入探讨了**进程抽象**这一操作系统的核心概念。我们学习了：

### 核心概念
1. **程序 vs 进程**：程序是静态的可执行文件，进程是动态的执行实例。
2. **进程的组成**：代码段、数据段、堆、栈、PCB、文件描述符等。
3. **进程抽象的价值**：**并发**（时间片轮转）和**隔离**（虚拟地址空间）。

### 进程控制块（PCB）
1. **PCB 的作用**：存储进程的所有元数据（状态、寄存器、内存、文件等）。
2. **xv6 struct proc**：简洁的 PCB 实现，包含 18 个核心字段。
3. **Linux task_struct**：复杂的 PCB 实现，超过 200 个字段，管理调度、内存、文件、信号、权限、命名空间、cgroup 等。

### 进程状态模型
1. **五状态模型**：新建、就绪、运行、阻塞、终止。
2. **七状态模型**：增加挂起就绪、挂起阻塞（swap out/in）。
3. **状态转换**：由时间片用完、I/O 等待、事件发生等事件触发。
4. **僵尸进程 vs 孤儿进程**：僵尸进程需父进程 `wait()` 回收，孤儿进程被 `init` 收养。

### 进程创建
1. **fork() 系统调用**：创建子进程，父子进程通过返回值区分。
2. **内存关系**：现代系统使用**写时复制（COW）** 优化性能（避免立即复制内存）。
3. **xv6 fork() 实现**：分配 PCB、复制内存（uvmcopy）、复制文件描述符、设置返回值。

### 进程执行与终止
1. **exec() 系列**：替换进程内存映像，加载新程序。
2. **wait()/waitpid()**：父进程等待子进程终止，回收资源。
3. **exit()/_exit()**：终止进程，执行清理工作（exit() 刷新缓冲区，_exit() 直接退出）。
4. **资源回收**：分两阶段（进程自身清理 + 父进程回收 PCB）。

### 关键要点
- 进程是操作系统**资源分配**和**调度**的基本单位。
- **PCB** 是操作系统管理进程的核心数据结构。
- **fork()** 是 Unix/Linux 创建进程的唯一方式，**COW** 是关键优化。
- **进程状态模型**是理解调度的基础。
- 理解进程生命周期（创建 → 运行 → 阻塞 → 终止 → 回收）对掌握操作系统至关重要。

**下一章预告**：我们将学习**上下文切换与进程调度基础**，深入理解操作系统如何在多个进程之间快速切换，以及调度算法的设计与实现。
