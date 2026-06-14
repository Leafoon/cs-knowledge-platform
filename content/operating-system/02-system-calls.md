# Chapter 2: 系统调用与操作系统接口

> **本章目标**：深入理解系统调用的本质、机制与实现。通过分析 xv6 和 Linux 的系统调用实现，掌握用户态与内核态交互的底层细节，理解系统调用的性能开销、参数传递、错误处理等关键问题。

---

## 2.1 系统调用概述

### 2.1.1 什么是系统调用？用户态与内核态的桥梁

系统调用（System Call，简称 syscall）是操作系统提供给用户程序的**唯一受控接口**，允许用户程序请求内核提供的服务。它是连接用户态（User Mode）与内核态（Kernel Mode）的桥梁，是实现进程隔离与安全保护的关键机制。

**系统调用的本质**：
- **特权操作的入口**：普通用户程序运行在 Ring 3（用户态），不能直接访问硬件、修改内核数据结构或执行特权指令。系统调用通过陷入内核态（Ring 0），使得内核代表用户进程执行需要特权的操作。
- **抽象与封装**：系统调用隐藏了底层硬件细节和内核实现，提供统一的 API。例如，`read()` 系统调用屏蔽了不同存储设备（HDD、SSD、网络）的差异，提供一致的文件读取接口。
- **安全边界**：内核在处理系统调用时会检查参数合法性、权限、资源配额等，防止恶意或错误的用户程序破坏系统稳定性。

**为什么需要系统调用？**

考虑一个简单的程序：
```c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

这段代码最终需要将字符串输出到终端。然而：
1. **硬件访问受限**：用户程序不能直接操作 VGA 显存、串口设备或显卡。
2. **I/O 需要内核协调**：多个进程可能同时写终端，需要内核同步输出，避免混乱。
3. **文件描述符管理**：`stdout` 对应的文件描述符由内核维护，用户程序只能通过系统调用间接访问。

**系统调用的执行流程**（高层视图）：
```
用户程序：printf("Hello\n")
    ↓
C 库函数：write(STDOUT_FILENO, "Hello\n", 6)
    ↓ (触发软中断/syscall 指令)
CPU 模式切换：User Mode → Kernel Mode
    ↓
内核系统调用处理程序：sys_write()
    ↓ (检查参数、权限、缓冲区)
设备驱动：终端驱动输出字符
    ↓
CPU 模式切换：Kernel Mode → User Mode
    ↓
用户程序继续执行
```

<div data-component="SystemCallConceptDiagram"></div>

---

### 2.1.2 系统调用 vs 库函数调用

许多初学者混淆系统调用与库函数。它们的根本区别在于**是否涉及 CPU 特权级切换**。

| 对比维度           | 系统调用（System Call）                                | 库函数调用（Library Function）                           |
|--------------------|--------------------------------------------------------|----------------------------------------------------------|
| **定义**           | 内核提供的服务接口，涉及用户态→内核态切换              | 用户空间的函数，可能封装系统调用或纯计算                 |
| **执行环境**       | 内核态（Ring 0）                                        | 用户态（Ring 3）                                         |
| **触发方式**       | 软中断（INT 0x80）或专用指令（SYSCALL/SYSENTER）        | 普通函数调用（CALL 指令）                                 |
| **性能开销**       | 高（上下文保存、模式切换、TLB 刷新）                    | 低（仅函数调用栈帧开销）                                  |
| **示例**           | `write()`, `fork()`, `mmap()`                          | `printf()`, `malloc()`, `strlen()`                       |
| **可移植性**       | POSIX 标准定义，跨 Unix-like 系统兼容                   | 依赖 C 库实现（glibc、musl、newlib）                      |
| **错误处理**       | 返回 -1 并设置 `errno`                                  | 视具体函数而定，可能调用系统调用或纯计算                  |

**示例对比**：

```c
// 1. 纯库函数调用（无系统调用）
size_t len = strlen("hello");  // strlen 仅在用户态遍历字符串

// 2. 库函数封装系统调用
FILE *fp = fopen("file.txt", "r");  // fopen 内部调用 open() 系统调用
// fopen() 是 C 库函数，但它会触发 open() 系统调用

// 3. 直接系统调用
int fd = open("file.txt", O_RDONLY);  // 直接调用 open() 系统调用
```

**关键点**：
- `strlen()` 是纯用户态函数，不涉及内核。
- `fopen()` 是库函数，但内部会调用 `open()` 系统调用。
- `open()` 是系统调用，必须陷入内核执行。

<div data-component="SyscallVsLibraryComparison"></div>

---

### 2.1.3 POSIX 标准与操作系统兼容性

POSIX（Portable Operating System Interface）是 IEEE 定义的一系列标准，旨在保证不同 Unix-like 操作系统之间的源代码兼容性。

**POSIX 定义的系统调用类别**：
1. **进程控制**：`fork()`, `exec()`, `wait()`, `exit()`, `kill()`
2. **文件操作**：`open()`, `read()`, `write()`, `close()`, `lseek()`, `stat()`
3. **目录操作**：`mkdir()`, `rmdir()`, `readdir()`, `chdir()`
4. **进程通信**：`pipe()`, `socket()`, `shmget()`, `msgget()`, `semget()`
5. **信号处理**：`signal()`, `sigaction()`, `kill()`, `alarm()`
6. **时间管理**：`time()`, `gettimeofday()`, `clock_gettime()`

**POSIX 兼容性的意义**：
- **可移植性**：遵循 POSIX 的程序可以在 Linux、macOS、FreeBSD、Solaris 等系统上无需修改地编译运行。
- **标准化接口**：开发者学习一套 API 即可，不需要为每个系统学习不同的系统调用。
- **生态繁荣**：大量开源软件（如 GNU coreutils、Apache、PostgreSQL）依赖 POSIX，促进了 Unix-like 系统的生态发展。

**非 POSIX 扩展**：
虽然 POSIX 定义了核心接口，但各操作系统也有自己的扩展：
- **Linux**：`epoll()`, `inotify()`, `perf_event_open()`, `memfd_create()`
- **macOS**：`kqueue()`, `mach_*()` 系列（Mach 微内核接口）
- **FreeBSD**：`kqueue()`, `capsicum()`（capability-based 安全）

**示例：跨平台的文件复制程序**：
```c
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>

int copy_file(const char *src, const char *dst) {
    int fd_src = open(src, O_RDONLY);
    if (fd_src < 0) {
        perror("open source");
        return -1;
    }
    
    int fd_dst = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_dst < 0) {
        perror("open destination");
        close(fd_src);
        return -1;
    }
    
    char buf[4096];
    ssize_t n;
    while ((n = read(fd_src, buf, sizeof(buf))) > 0) {
        if (write(fd_dst, buf, n) != n) {
            perror("write");
            close(fd_src);
            close(fd_dst);
            return -1;
        }
    }
    
    close(fd_src);
    close(fd_dst);
    return 0;
}
```

这段代码可以在任何 POSIX 兼容系统上编译运行，因为 `open()`, `read()`, `write()`, `close()` 都是 POSIX 标准系统调用。

---

### 2.1.4 系统调用的性能开销分析

系统调用的性能开销远高于普通函数调用，主要包括以下几个方面：

**1. 模式切换开销（Mode Switch Overhead）**

- **保存用户态上下文**：将用户态寄存器（通用寄存器、栈指针、指令指针）保存到内核栈。
- **切换到内核栈**：加载内核栈指针（从 TSS 或内核数据结构获取）。
- **特权级切换**：CPU 从 Ring 3 切换到 Ring 0，需要检查权限、刷新流水线。
- **恢复内核态上下文**：加载内核代码段、数据段等。
- **返回时的逆过程**：恢复用户态寄存器、栈指针、切换回 Ring 3。

**2. 缓存失效（Cache Invalidation）**

- **TLB 刷新**：进入内核后，TLB 中的用户态地址翻译条目可能失效（取决于 ASID 机制）。
- **CPU 缓存污染**：内核代码和数据会占用 CPU L1/L2/L3 缓存，导致用户程序的缓存数据被驱逐。
- **指令缓存未命中**：内核代码路径与用户代码不同，导致指令缓存（I-Cache）未命中。

**3. 参数验证与复制**

- **参数合法性检查**：内核必须验证用户传递的指针是否有效、是否越界、是否有权限访问。
- **数据复制**：对于 `read()`/`write()` 等系统调用，需要在用户缓冲区与内核缓冲区之间复制数据（`copy_from_user()` / `copy_to_user()`）。

**4. 间接开销**

- **调度延迟**：系统调用可能触发进程调度（如 `sleep()`、`wait()`），导致上下文切换。
- **中断延迟**：系统调用期间可能被硬件中断打断，增加总执行时间。

**性能测量**：

使用 `lmbench` 微基准测试工具可以测量系统调用开销：
```bash
# 测量 getpid() 系统调用的延迟
lat_syscall -N 1000000 null
lat_syscall -N 1000000 getpid
```

典型结果：
- **空系统调用（null syscall）**：~100-200 纳秒（现代 x86-64 CPU）
- **getpid()**：~150-300 纳秒
- **read() 1 字节**：~500-1000 纳秒（包含缓冲区复制）

**优化策略**：
1. **批处理**：一次系统调用处理多个操作（如 `readv()`/`writev()`）。
2. **vDSO（Virtual Dynamic Shared Object）**：将部分系统调用（如 `gettimeofday()`、`clock_gettime()`）映射到用户空间，避免模式切换。
3. **减少系统调用频率**：使用用户态缓冲（如 C 库的 `stdio` 缓冲）。

<div data-component="SyscallOverheadBreakdown"></div>

---

## 2.2 系统调用机制

### 2.2.1 软件中断（INT 0x80、SYSCALL 指令）

系统调用的触发依赖于 CPU 提供的**陷入内核态的机制**。不同架构和时代使用不同的指令：

**1. 软件中断（Software Interrupt）**

早期 x86 Linux 使用 `INT 0x80` 指令：
```asm
; 用户态代码
mov eax, 1          ; 系统调用号：1 = exit
mov ebx, 0          ; 参数：退出码 = 0
int 0x80            ; 触发软件中断，陷入内核
```

**INT 0x80 的执行过程**：
1. CPU 查找中断描述符表（IDT），找到 0x80 号中断的处理程序入口（`system_call`）。
2. 保存 `EFLAGS`、`CS`、`EIP` 到内核栈。
3. 切换到 Ring 0，跳转到 `system_call` 内核函数。
4. 内核根据 `EAX` 中的系统调用号分派到具体处理函数（如 `sys_exit`）。

**缺点**：
- **性能低**：INT 指令会检查权限、查找 IDT、保存大量状态，开销高达 100+ 时钟周期。
- **通用中断机制**：INT 指令设计用于处理异常和硬件中断，对系统调用来说过于重量级。

**2. 快速系统调用指令（SYSCALL/SYSENTER）**

为了优化系统调用性能，现代 CPU 引入专用指令：
- **Intel x86**：`SYSENTER`/`SYSEXIT`
- **AMD x86-64**：`SYSCALL`/`SYSRET`

**SYSCALL 指令的优势**：
- **硬件优化**：直接从特定 MSR（Model-Specific Register）加载内核代码段、栈指针，无需查表。
- **快速切换**：延迟降低到 ~50 时钟周期。
- **简化流程**：CPU 自动保存 `RCX`（返回地址）和 `R11`（`RFLAGS`），减少内核开销。

**x86-64 SYSCALL 执行过程**：
```asm
; 用户态代码（glibc 封装）
mov rax, 60         ; 系统调用号：60 = exit
mov rdi, 0          ; 第一个参数：退出码 = 0
syscall             ; 触发系统调用
```

1. CPU 从 `MSR_LSTAR` 寄存器读取内核入口地址（`entry_SYSCALL_64`）。
2. 保存用户态 `RIP` 到 `RCX`，`RFLAGS` 到 `R11`。
3. 从 `MSR_STAR` 加载内核代码段和栈段选择子。
4. 跳转到 `entry_SYSCALL_64`（内核入口函数）。

**3. ARM 架构：SVC 指令**

ARM 使用 `SVC`（Supervisor Call，旧称 `SWI`）指令：
```asm
; ARM64 系统调用示例
mov x8, #64         ; 系统调用号：64 = write
mov x0, #1          ; 参数1：fd = 1 (stdout)
adr x1, msg         ; 参数2：缓冲区地址
mov x2, #13         ; 参数3：长度
svc #0              ; 触发系统调用
```

<div data-component="SyscallInstructionComparison"></div>

---

### 2.2.2 系统调用号与参数传递

**系统调用号（Syscall Number）**

内核为每个系统调用分配唯一编号。用户程序通过寄存器传递系统调用号，内核据此分派到对应处理函数。

**Linux x86-64 系统调用号示例**（参考 `arch/x86/entry/syscalls/syscall_64.tbl`）：
```
0   common  read                    sys_read
1   common  write                   sys_write
2   common  open                    sys_open
3   common  close                   sys_close
...
39  common  getpid                  sys_getpid
57  common  fork                    sys_fork
59  common  execve                  sys_execve
60  common  exit                    sys_exit
...
```

**参数传递约定**

不同架构使用不同寄存器传递参数：

**x86-64（Linux）**：
| 参数顺序     | 寄存器   |
|--------------|----------|
| 系统调用号   | `RAX`    |
| 参数 1       | `RDI`    |
| 参数 2       | `RSI`    |
| 参数 3       | `RDX`    |
| 参数 4       | `R10`    |
| 参数 5       | `R8`     |
| 参数 6       | `R9`     |
| 返回值       | `RAX`    |

**示例：write() 系统调用**：
```c
ssize_t write(int fd, const void *buf, size_t count);
```

对应汇编（glibc 封装）：
```asm
; write(1, "hello", 5)
mov rax, 1          ; 系统调用号：1 = write
mov rdi, 1          ; 参数1：fd = 1
lea rsi, [rel msg]  ; 参数2：buf = "hello"
mov rdx, 5          ; 参数3：count = 5
syscall
```

**x86-32（旧版）**：
参数通过栈传递，`EAX` 保存系统调用号：
```asm
push 5              ; 参数3：count
push offset msg     ; 参数2：buf
push 1              ; 参数1：fd
mov eax, 4          ; 系统调用号：4 = write
int 0x80
```

**ARM64**：
| 参数顺序     | 寄存器   |
|--------------|----------|
| 系统调用号   | `X8`     |
| 参数 1-6     | `X0-X5`  |
| 返回值       | `X0`     |

<div data-component="SyscallParameterPassing"></div>

---

### 2.2.3 系统调用表（Syscall Table）

内核维护一个**系统调用表**（syscall table），将系统调用号映射到内核函数指针。

**Linux 系统调用表定义**（简化版）：
```c
// arch/x86/entry/syscall_64.c
const sys_call_ptr_t sys_call_table[] = {
    [0] = sys_read,
    [1] = sys_write,
    [2] = sys_open,
    [3] = sys_close,
    [4] = sys_stat,
    // ... 共 300+ 个系统调用
    [60] = sys_exit,
};
```

**系统调用分派逻辑**（`entry_SYSCALL_64` 简化版）：
```c
asmlinkage long entry_SYSCALL_64(void) {
    unsigned long nr = regs->rax;  // 获取系统调用号
    
    if (nr >= NR_syscalls) {
        regs->rax = -ENOSYS;  // 非法系统调用号
        return -ENOSYS;
    }
    
    // 调用系统调用处理函数
    regs->rax = sys_call_table[nr](
        regs->rdi,  // 参数1
        regs->rsi,  // 参数2
        regs->rdx,  // 参数3
        regs->r10,  // 参数4
        regs->r8,   // 参数5
        regs->r9    // 参数6
    );
    
    return regs->rax;
}
```

**系统调用表的版本兼容性**：
- **系统调用号固定**：一旦分配，系统调用号不能改变，以保证二进制兼容性。
- **废弃系统调用**：旧系统调用（如 `select()` → `pselect()`）保留在表中，返回错误或重定向到新版本。
- **新系统调用**：添加到表尾，旧程序不受影响。

<div data-component="SyscallTableViewer"></div>

---

### 2.2.4 返回值与错误处理（errno）

**系统调用的返回值约定**：
- **成功**：返回非负值（通常是文件描述符、读取字节数、进程 PID 等）。
- **失败**：返回 `-1`，并设置全局变量 `errno` 表示错误类型。

**errno 的定义**（glibc）：
```c
#include <errno.h>

extern int errno;  // 全局错误码变量
```

**常见错误码**（定义在 `<errno.h>` 或 `<asm-generic/errno-base.h>`）：
| 错误码        | 值  | 含义                                      |
|---------------|-----|-------------------------------------------|
| `EPERM`       | 1   | 操作不允许（权限不足）                     |
| `ENOENT`      | 2   | 文件或目录不存在                           |
| `EINTR`       | 4   | 系统调用被信号中断                         |
| `EIO`         | 5   | I/O 错误                                   |
| `EBADF`       | 9   | 非法文件描述符                             |
| `ENOMEM`      | 12  | 内存不足                                   |
| `EACCES`      | 13  | 权限被拒绝                                 |
| `EFAULT`      | 14  | 非法地址（指针越界）                       |
| `EEXIST`      | 17  | 文件已存在                                 |
| `EINVAL`      | 22  | 非法参数                                   |
| `EMFILE`      | 24  | 打开文件数过多                             |
| `ENOSYS`      | 38  | 系统调用未实现                             |

**错误处理最佳实践**：
```c
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

int main() {
    int fd = open("/nonexistent", O_RDONLY);
    if (fd < 0) {
        // 方法1：使用 perror() 打印错误信息
        perror("open");
        // 输出：open: No such file or directory
        
        // 方法2：使用 strerror() 获取错误描述
        fprintf(stderr, "Error: %s\n", strerror(errno));
        
        // 方法3：根据 errno 分类处理
        switch (errno) {
            case ENOENT:
                fprintf(stderr, "File not found\n");
                break;
            case EACCES:
                fprintf(stderr, "Permission denied\n");
                break;
            default:
                fprintf(stderr, "Unknown error: %d\n", errno);
        }
        
        return 1;
    }
    
    close(fd);
    return 0;
}
```

**errno 的线程安全性**：
在多线程程序中，`errno` 是**线程局部变量**（TLS，Thread-Local Storage），每个线程有独立的 `errno` 副本，避免竞态条件。

**内核如何设置 errno**：
内核系统调用返回负数错误码（如 `-ENOENT`），glibc 封装函数检测到负返回值后：
1. 将绝对值存入 `errno`（`errno = -retval`）。
2. 返回 `-1` 给用户程序。

示例（glibc 封装的 `open()` 简化版）：
```c
int open(const char *pathname, int flags, ...) {
    long ret = syscall(SYS_open, pathname, flags, ...);
    if (ret < 0) {
        errno = -ret;  // 转换为正数错误码
        return -1;
    }
    return (int)ret;  // 返回文件描述符
}
```

<div data-component="ErrorHandlingFlow"></div>

---

## 2.3 系统调用分类

### 2.3.1 进程控制：fork()、exec()、wait()、exit()

进程控制系统调用是操作系统最核心的接口，负责进程的创建、执行、等待和终止。

**1. fork() - 创建子进程**

```c
#include <unistd.h>

pid_t fork(void);
// 返回值：父进程返回子进程 PID，子进程返回 0，失败返回 -1
```

**fork() 的语义**：
- 创建当前进程的**完整副本**（子进程）。
- 子进程继承父进程的：代码、数据、堆、栈、打开的文件描述符、环境变量等。
- 子进程拥有独立的地址空间（通过写时复制优化）。
- 父子进程通过返回值区分：父进程获得子进程 PID，子进程获得 0。

**示例**：
```c
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    
    if (pid < 0) {
        perror("fork failed");
        return 1;
    } else if (pid == 0) {
        // 子进程代码
        printf("Child: PID=%d, Parent PID=%d\n", getpid(), getppid());
    } else {
        // 父进程代码
        printf("Parent: PID=%d, Child PID=%d\n", getpid(), pid);
        wait(NULL);  // 等待子进程结束
    }
    
    return 0;
}
```

**输出示例**：
```
Parent: PID=1234, Child PID=1235
Child: PID=1235, Parent PID=1234
```

**fork() 的经典陷阱**：
```c
int i;
for (i = 0; i < 3; i++) {
    fork();
    printf("i=%d\n", i);
}
// 问题：会创建多少个进程？打印多少次？
// 答案：8 个进程，打印 14 次（2^0 + 2^1 + 2^2 + ... ）
```

**2. exec() 系列 - 执行新程序**

`exec()` 系列系统调用用当前进程的新程序**替换当前进程的内存映像**。

**常用 exec 变体**：
```c
#include <unistd.h>

int execl(const char *path, const char *arg0, ..., NULL);
int execv(const char *path, char *const argv[]);
int execle(const char *path, const char *arg0, ..., NULL, char *const envp[]);
int execve(const char *path, char *const argv[], char *const envp[]);
int execlp(const char *file, const char *arg0, ..., NULL);
int execvp(const char *file, char *const argv[]);
```

**命名规则**：
- `l`（list）：参数以列表形式传递（`arg0, arg1, ..., NULL`）。
- `v`（vector）：参数以数组形式传递（`argv[]`）。
- `e`（environment）：显式传递环境变量数组 `envp[]`。
- `p`（path）：在 `PATH` 环境变量中搜索可执行文件。

**示例**：
```c
#include <unistd.h>
#include <stdio.h>

int main() {
    printf("Before exec\n");
    
    // 执行 /bin/ls 程序
    execl("/bin/ls", "ls", "-l", "/tmp", NULL);
    
    // 如果 exec 成功，下面的代码不会执行
    perror("exec failed");
    return 1;
}
```

**exec() 的关键特性**：
- **替换内存映像**：代码段、数据段、堆、栈全部被新程序覆盖。
- **保留文件描述符**：除非设置了 `FD_CLOEXEC` 标志。
- **保留进程 PID**：进程 ID、父进程 ID 不变。
- **不返回**：成功时不返回，失败时返回 -1。

**3. wait() / waitpid() - 等待子进程**

```c
#include <sys/wait.h>

pid_t wait(int *status);
pid_t waitpid(pid_t pid, int *status, int options);
```

**wait() 的作用**：
- 阻塞父进程，直到任意子进程终止。
- 回收子进程资源（避免僵尸进程）。
- 获取子进程退出状态（通过 `status` 指针）。

**示例**：
```c
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程
        printf("Child running\n");
        sleep(2);
        exit(42);  // 退出码 42
    } else {
        // 父进程
        int status;
        pid_t child_pid = wait(&status);
        
        if (WIFEXITED(status)) {
            printf("Child %d exited with code %d\n", 
                   child_pid, WEXITSTATUS(status));
        }
    }
    
    return 0;
}
```

**状态宏**：
- `WIFEXITED(status)`：子进程正常退出？
- `WEXITSTATUS(status)`：获取退出码（0-255）。
- `WIFSIGNALED(status)`：子进程被信号终止？
- `WTERMSIG(status)`：获取终止信号编号。

**4. exit() - 终止进程**

```c
#include <stdlib.h>
void exit(int status);

#include <unistd.h>
void _exit(int status);
```

**exit() vs _exit()**：
- `exit()`：库函数，执行清理工作（刷新 stdio 缓冲区、调用 `atexit()` 注册的函数），然后调用 `_exit()`。
- `_exit()`：系统调用，直接终止进程，不执行清理。

**示例**：
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    printf("Message 1");  // 无换行符，留在缓冲区
    _exit(0);             // 直接退出，缓冲区未刷新
    // 输出为空！
}
```

修正版本：
```c
printf("Message 1");
exit(0);  // 或 printf("Message 1\n");
// 输出：Message 1
```

<div data-component="ForkExecWaitDemo"></div>

---

### 2.3.2 文件操作：open()、read()、write()、close()

文件 I/O 系统调用是 Unix"一切皆文件"哲学的核心。

**1. open() - 打开文件**

```c
#include <fcntl.h>

int open(const char *pathname, int flags);
int open(const char *pathname, int flags, mode_t mode);
```

**flags 参数**（必选一个访问模式）：
- `O_RDONLY`：只读
- `O_WRONLY`：只写
- `O_RDWR`：读写

**可选标志**（按位或组合）：
- `O_CREAT`：文件不存在时创建（需提供 `mode` 参数）
- `O_TRUNC`：打开时清空文件内容
- `O_APPEND`：追加模式（写操作总是在文件末尾）
- `O_EXCL`：与 `O_CREAT` 配合，文件已存在则失败
- `O_NONBLOCK`：非阻塞模式
- `O_SYNC`：同步 I/O（每次写操作等待磁盘完成）

**mode 参数**（文件权限，八进制）：
```c
0644  // rw-r--r--（用户读写，组/其他只读）
0755  // rwxr-xr-x（用户读写执行，组/其他读执行）
```

**示例**：
```c
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
    // 创建新文件，权限 rw-r--r--
    int fd = open("test.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    write(fd, "Hello\n", 6);
    close(fd);
    
    return 0;
}
```

**2. read() - 读取文件**

```c
#include <unistd.h>

ssize_t read(int fd, void *buf, size_t count);
// 返回值：读取的字节数，0 表示 EOF，-1 表示错误
```

**关键点**：
- `read()` 可能读取少于 `count` 字节（如管道、网络套接字）。
- 需要循环读取直到满足需求或遇到 EOF。

**示例**：
```c
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
    int fd = open("test.txt", O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    char buf[1024];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        write(STDOUT_FILENO, buf, n);  // 输出到终端
    }
    
    if (n < 0) {
        perror("read");
    }
    
    close(fd);
    return 0;
}
```

**3. write() - 写入文件**

```c
#include <unistd.h>

ssize_t write(int fd, const void *buf, size_t count);
// 返回值：写入的字节数，-1 表示错误
```

**关键点**：
- `write()` 也可能写入少于 `count` 字节（磁盘满、信号中断）。
- 需要循环写入直到所有数据完成。

**健壮的写入示例**：
```c
ssize_t write_all(int fd, const void *buf, size_t count) {
    size_t written = 0;
    while (written < count) {
        ssize_t n = write(fd, (char *)buf + written, count - written);
        if (n < 0) {
            if (errno == EINTR) continue;  // 被信号中断，重试
            return -1;
        }
        written += n;
    }
    return written;
}
```

**4. close() - 关闭文件**

```c
#include <unistd.h>

int close(int fd);
```

**作用**：
- 释放文件描述符。
- 刷新内核缓冲区（对于写操作）。
- 如果是最后一个引用，释放内核文件对象。

**常见错误**：
```c
close(fd);
if (close(fd) < 0) {  // 错误！重复关闭
    perror("close");
}
```

修正：
```c
if (close(fd) < 0) {
    perror("close");
}
fd = -1;  // 标记为无效
```

<div data-component="FileIOWorkflow"></div>

---

### 2.3.3 内存管理：brk()、mmap()、munmap()

**1. brk() / sbrk() - 调整堆大小**

```c
#include <unistd.h>

int brk(void *addr);
void *sbrk(intptr_t increment);
```

**brk() 的作用**：
- 设置进程的堆顶地址（program break）。
- 增加堆大小以分配更多内存。

**sbrk() 的作用**：
- `sbrk(0)`：返回当前堆顶地址。
- `sbrk(n)`：将堆顶增加 `n` 字节，返回旧堆顶地址。

**示例**（简单的 malloc 实现）：
```c
#include <unistd.h>
#include <stddef.h>

void *my_malloc(size_t size) {
    void *p = sbrk(0);      // 获取当前堆顶
    if (sbrk(size) == (void *)-1) {
        return NULL;        // 分配失败
    }
    return p;
}
```

**现代实践**：
- `brk()`/`sbrk()` 已过时，现代 `malloc()` 主要使用 `mmap()`。
- 原因：`brk()` 只能线性增长堆，无法释放中间内存。

**2. mmap() - 内存映射**

```c
#include <sys/mman.h>

void *mmap(void *addr, size_t length, int prot, int flags,
           int fd, off_t offset);
int munmap(void *addr, size_t length);
```

**mmap() 的两大用途**：

**A. 匿名映射（分配内存）**：
```c
// 分配 4MB 内存
void *ptr = mmap(NULL, 4*1024*1024, 
                 PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS,
                 -1, 0);
if (ptr == MAP_FAILED) {
    perror("mmap");
    return NULL;
}

// 使用内存
memset(ptr, 0, 4*1024*1024);

// 释放内存
munmap(ptr, 4*1024*1024);
```

**B. 文件映射（内存映射 I/O）**：
```c
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main() {
    int fd = open("data.txt", O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);  // 获取文件大小
    
    // 将文件映射到内存
    char *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // 映射后可以关闭文件描述符
    
    // 像访问数组一样读取文件
    for (int i = 0; i < sb.st_size; i++) {
        putchar(data[i]);
    }
    
    munmap(data, sb.st_size);
    return 0;
}
```

**mmap() 的优势**：
- **高效 I/O**：避免 `read()`/`write()` 的数据复制（内核缓冲区 ↔ 用户缓冲区）。
- **共享内存**：多个进程映射同一文件可实现进程间通信。
- **延迟分配**：映射时不立即分配物理内存，访问时才分配（按需分页）。

**prot 参数**（内存保护）：
- `PROT_READ`：可读
- `PROT_WRITE`：可写
- `PROT_EXEC`：可执行
- `PROT_NONE`：不可访问

**flags 参数**：
- `MAP_PRIVATE`：写时复制（修改不影响原文件）
- `MAP_SHARED`：共享映射（修改写回文件）
- `MAP_ANONYMOUS`：匿名映射（不关联文件）
- `MAP_FIXED`：强制使用指定地址（危险）

<div data-component="MmapVisualization"></div>

---

### 2.3.4 进程通信：pipe()、socket()、shmget()

**1. pipe() - 管道**

```c
#include <unistd.h>

int pipe(int pipefd[2]);
// pipefd[0]：读端，pipefd[1]：写端
```

**管道的特性**：
- **单向通信**：数据只能从写端流向读端。
- **亲缘进程**：通常用于父子进程通信。
- **同步机制**：读端阻塞直到有数据，写端在管道满时阻塞。

**示例**：
```c
#include <unistd.h>
#include <stdio.h>
#include <string.h>

int main() {
    int pipefd[2];
    pipe(pipefd);
    
    if (fork() == 0) {
        // 子进程：写入管道
        close(pipefd[0]);  // 关闭读端
        write(pipefd[1], "Hello from child", 17);
        close(pipefd[1]);
    } else {
        // 父进程：读取管道
        close(pipefd[1]);  // 关闭写端
        char buf[128];
        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        write(STDOUT_FILENO, buf, n);
        close(pipefd[0]);
    }
    
    return 0;
}
```

**2. socket() - 网络通信**

```c
#include <sys/socket.h>

int socket(int domain, int type, int protocol);
```

**domain（协议族）**：
- `AF_INET`：IPv4
- `AF_INET6`：IPv6
- `AF_UNIX`：本地域套接字（IPC）

**type（套接字类型）**：
- `SOCK_STREAM`：TCP（可靠、有序、双向字节流）
- `SOCK_DGRAM`：UDP（不可靠、无连接数据报）

**简单 TCP 服务器示例**：
```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>

int main() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(8080),
        .sin_addr.s_addr = INADDR_ANY
    };
    
    bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));
    listen(sockfd, 5);
    
    int clientfd = accept(sockfd, NULL, NULL);
    write(clientfd, "Hello, client!\n", 15);
    close(clientfd);
    close(sockfd);
    
    return 0;
}
```

**3. shmget() - 共享内存**

```c
#include <sys/ipc.h>
#include <sys/shm.h>

int shmget(key_t key, size_t size, int shmflg);
void *shmat(int shmid, const void *shmaddr, int shmflg);
int shmdt(const void *shmaddr);
```

**共享内存示例**：
```c
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <string.h>

int main() {
    // 创建共享内存段
    int shmid = shmget(1234, 4096, IPC_CREAT | 0666);
    
    // 附加到进程地址空间
    char *shm = shmat(shmid, NULL, 0);
    
    // 写入数据
    strcpy(shm, "Shared data");
    
    // 分离共享内存
    shmdt(shm);
    
    return 0;
}
```

<div data-component="IPCMechanismsComparison"></div>

---

### 2.3.5 信号处理：signal()、kill()、sigaction()

**信号（Signal）** 是 Unix 系统的异步通知机制，用于通知进程发生了某个事件。

**常见信号**：
| 信号       | 值  | 默认行为   | 描述                        |
|------------|-----|------------|---------------------------|
| `SIGHUP`   | 1   | 终止       | 终端挂起                   |
| `SIGINT`   | 2   | 终止       | 中断（Ctrl+C）             |
| `SIGQUIT`  | 3   | 终止+核心转储 | 退出（Ctrl+\）             |
| `SIGKILL`  | 9   | 终止       | 强制终止（不可捕获）       |
| `SIGSEGV`  | 11  | 终止+核心转储 | 段错误                     |
| `SIGPIPE`  | 13  | 终止       | 管道破裂                   |
| `SIGALRM`  | 14  | 终止       | alarm() 定时器超时         |
| `SIGTERM`  | 15  | 终止       | 终止信号（可捕获）         |
| `SIGCHLD`  | 17  | 忽略       | 子进程状态改变             |
| `SIGSTOP`  | 19  | 停止       | 停止进程（不可捕获）       |
| `SIGCONT`  | 18  | 继续       | 继续已停止的进程           |

**1. signal() - 注册信号处理器**

```c
#include <signal.h>

typedef void (*sighandler_t)(int);
sighandler_t signal(int signum, sighandler_t handler);
```

**示例**：
```c
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

void handler(int sig) {
    printf("Caught signal %d\n", sig);
}

int main() {
    signal(SIGINT, handler);  // 注册 Ctrl+C 处理器
    
    while (1) {
        printf("Running...\n");
        sleep(1);
    }
    
    return 0;
}
```

**2. kill() - 发送信号**

```c
#include <signal.h>

int kill(pid_t pid, int sig);
```

**示例**：
```c
#include <signal.h>
#include <unistd.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程：等待信号
        while (1) sleep(1);
    } else {
        // 父进程：2 秒后终止子进程
        sleep(2);
        kill(pid, SIGTERM);
    }
    
    return 0;
}
```

**3. sigaction() - 高级信号处理**

`sigaction()` 提供更强大的信号控制（推荐使用，`signal()` 行为不一致）：

```c
#include <signal.h>

int sigaction(int signum, const struct sigaction *act,
              struct sigaction *oldact);
```

**示例**：
```c
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

void handler(int sig, siginfo_t *info, void *context) {
    printf("Signal %d from PID %d\n", sig, info->si_pid);
}

int main() {
    struct sigaction sa = {
        .sa_sigaction = handler,
        .sa_flags = SA_SIGINFO  // 使用扩展处理器
    };
    sigemptyset(&sa.sa_mask);
    
    sigaction(SIGINT, &sa, NULL);
    
    while (1) {
        sleep(1);
    }
    
    return 0;
}
```

<div data-component="SignalHandlingDemo"></div>

---

## 2.4 xv6 系统调用实现

### 2.4.1 用户空间：usys.S 汇编代码分析

xv6 的系统调用从用户空间到内核空间的路径非常清晰。用户程序不直接调用汇编指令，而是通过 C 函数封装。

**usys.S 的作用**：
为每个系统调用生成一个封装函数，执行 `ecall` 指令（RISC-V 的系统调用指令）。

**usys.S 源码**（简化版）：
```asm
# user/usys.S
# 为每个系统调用生成封装函数

.global fork
fork:
    li a7, SYS_fork   # 系统调用号 1
    ecall             # 陷入内核
    ret

.global exit
exit:
    li a7, SYS_exit   # 系统调用号 2
    ecall
    ret

.global wait
wait:
    li a7, SYS_wait   # 系统调用号 3
    ecall
    ret

# ... 其他系统调用类似
```

**指令解析**：
- `li a7, SYS_fork`：将系统调用号加载到寄存器 `a7`（RISC-V 调用约定）。
- `ecall`：触发环境调用异常，CPU 从 User Mode 切换到 Supervisor Mode，跳转到内核的陷阱处理程序 `uservec`。
- `ret`：从系统调用返回后，返回到用户程序。

**参数传递（RISC-V）**：
| 参数位置     | 寄存器   |
|--------------|----------|
| 系统调用号   | `a7`     |
| 参数 1       | `a0`     |
| 参数 2       | `a1`     |
| 参数 3       | `a2`     |
| 参数 4       | `a3`     |
| 参数 5       | `a4`     |
| 参数 6       | `a5`     |
| 返回值       | `a0`     |

**用户程序调用示例**：
```c
// user/sh.c
int main() {
    int pid = fork();  // 调用 user/usys.S 中的 fork()
    if (pid == 0) {
        exec("/bin/ls", argv);
    }
    wait(0);
}
```

编译后的汇编（简化）：
```asm
# 调用 fork()
call fork        # 跳转到 usys.S 的 fork 标签
                 # fork 内部执行 ecall
mv s0, a0        # 保存返回值（PID）
```

<div data-component="Xv6UsysAnalysis"></div>

---

### 2.4.2 内核空间：syscall.c 分发逻辑

当 `ecall` 触发后，CPU 跳转到内核的陷阱处理程序。xv6 的系统调用分发逻辑在 `kernel/syscall.c`。

**系统调用分发流程**：

1. **usertrap() 捕获异常**（`kernel/trap.c`）：
```c
void usertrap(void) {
    struct proc *p = myproc();
    
    // 保存用户态寄存器到 trapframe
    p->trapframe->epc = r_sepc();  // 保存用户程序计数器
    
    if (r_scause() == 8) {  // 8 = 环境调用（系统调用）
        syscall();  // 调用系统调用分发函数
    }
    
    // 返回用户态
    usertrapret();
}
```

2. **syscall() 查表分发**（`kernel/syscall.c`）：
```c
// 系统调用函数指针表
static uint64 (*syscalls[])(void) = {
    [SYS_fork]    sys_fork,
    [SYS_exit]    sys_exit,
    [SYS_wait]    sys_wait,
    [SYS_read]    sys_read,
    [SYS_write]   sys_write,
    [SYS_open]    sys_open,
    [SYS_close]   sys_close,
    // ... 共 21 个系统调用
};

void syscall(void) {
    int num;
    struct proc *p = myproc();
    
    num = p->trapframe->a7;  // 从 a7 寄存器获取系统调用号
    
    if (num > 0 && num < NELEM(syscalls) && syscalls[num]) {
        p->trapframe->a0 = syscalls[num]();  // 调用对应处理函数，返回值存入 a0
    } else {
        printf("%d %s: unknown sys call %d\n",
               p->pid, p->name, num);
        p->trapframe->a0 = -1;  // 非法系统调用，返回 -1
    }
}
```

**关键点**：
- 系统调用号从 `trapframe->a7` 获取（`ecall` 前用户代码设置的）。
- 返回值写入 `trapframe->a0`，用户程序恢复后从 `a0` 读取。
- 如果系统调用号非法，返回 `-1`。

<div data-component="Xv6SyscallDispatch"></div>

---

### 2.4.3 参数获取：argint()、argaddr()、argstr()

内核需要从用户态寄存器中提取系统调用参数。xv6 提供三个辅助函数：

**1. argint() - 获取整数参数**

```c
// kernel/syscall.c
int argint(int n, int *ip) {
    *ip = argraw(n);
    return 0;
}

static uint64 argraw(int n) {
    struct proc *p = myproc();
    switch (n) {
        case 0: return p->trapframe->a0;
        case 1: return p->trapframe->a1;
        case 2: return p->trapframe->a2;
        case 3: return p->trapframe->a3;
        case 4: return p->trapframe->a4;
        case 5: return p->trapframe->a5;
    }
    panic("argraw");
}
```

**使用示例**（`sys_exit`）：
```c
uint64 sys_exit(void) {
    int n;
    argint(0, &n);  // 获取第 1 个参数（退出码）
    exit(n);
    return 0;  // 不会到达
}
```

**2. argaddr() - 获取地址参数**

```c
int argaddr(int n, uint64 *ip) {
    *ip = argraw(n);
    return 0;
}
```

**3. argstr() - 获取字符串参数**

字符串参数以指针形式传递，内核需要从用户空间复制到内核空间：

```c
// 从用户空间复制字符串到内核缓冲区
int argstr(int n, char *buf, int max) {
    uint64 addr;
    argaddr(n, &addr);  // 获取字符串指针
    return fetchstr(addr, buf, max);  // 从用户空间复制字符串
}

// 从用户地址空间复制字符串
int fetchstr(uint64 addr, char *buf, int max) {
    struct proc *p = myproc();
    int err = copyinstr(p->pagetable, buf, addr, max);
    return (err < 0) ? err : strlen(buf);
}
```

**安全检查**：
`copyinstr()` 会验证：
- 用户地址是否有效（不在内核地址空间）。
- 用户地址是否有读权限。
- 字符串是否以 `\0` 结尾（防止缓冲区溢出）。

**使用示例**（`sys_open`）：
```c
uint64 sys_open(void) {
    char path[MAXPATH];
    int fd, omode;
    
    argstr(0, path, MAXPATH);  // 获取文件路径
    argint(1, &omode);         // 获取打开模式
    
    if ((fd = open(path, omode)) < 0) {
        return -1;
    }
    
    return fd;
}
```

<div data-component="Xv6ArgFetchDemo"></div>

---

### 2.4.4 完整示例：sys_fork() 实现剖析

`fork()` 是最复杂的系统调用之一，涉及进程复制、内存复制、文件描述符继承等。

**用户空间调用**：
```c
// user/sh.c
int pid = fork();
```

**内核空间实现**（`kernel/proc.c`）：
```c
uint64 sys_fork(void) {
    return fork();  // 调用内核 fork() 函数
}

int fork(void) {
    int i, pid;
    struct proc *np;  // 新进程（子进程）
    struct proc *p = myproc();  // 当前进程（父进程）
    
    // 1. 分配新进程结构
    if ((np = allocproc()) == 0) {
        return -1;
    }
    
    // 2. 复制用户内存（使用 uvmcopy）
    if (uvmcopy(p->pagetable, np->pagetable, p->sz) < 0) {
        freeproc(np);
        return -1;
    }
    np->sz = p->sz;
    
    // 3. 复制 trapframe（用户态寄存器）
    *(np->trapframe) = *(p->trapframe);
    
    // 4. 设置子进程返回值为 0
    np->trapframe->a0 = 0;
    
    // 5. 复制打开的文件描述符
    for (i = 0; i < NOFILE; i++) {
        if (p->ofile[i])
            np->ofile[i] = filedup(p->ofile[i]);
    }
    np->cwd = idup(p->cwd);  // 复制当前工作目录
    
    // 6. 复制进程名称
    safestrcpy(np->name, p->name, sizeof(p->name));
    
    pid = np->pid;
    
    np->parent = p;  // 设置父进程指针
    
    // 7. 设置子进程为 RUNNABLE 状态
    acquire(&np->lock);
    np->state = RUNNABLE;
    release(&np->lock);
    
    // 8. 父进程返回子进程 PID
    return pid;
}
```

**关键步骤详解**：

**1. allocproc() - 分配进程结构**：
- 在进程表（`proc[NPROC]`）中找到空闲槽位。
- 分配内核栈。
- 初始化 `trapframe`（保存用户态寄存器）。
- 设置上下文切换的返回地址（`ra = forkret`）。

**2. uvmcopy() - 复制页表**：
xv6 早期版本直接复制所有页，现代版本使用**写时复制（COW）**：
```c
int uvmcopy(pagetable_t old, pagetable_t new, uint64 sz) {
    pte_t *pte;
    uint64 pa, i;
    uint flags;
    
    for (i = 0; i < sz; i += PGSIZE) {
        if ((pte = walk(old, i, 0)) == 0)
            panic("uvmcopy: pte should exist");
        if ((*pte & PTE_V) == 0)
            panic("uvmcopy: page not present");
        
        pa = PTE2PA(*pte);
        flags = PTE_FLAGS(*pte);
        
        // 分配新页并复制内容
        char *mem = kalloc();
        if (mem == 0) goto err;
        memmove(mem, (char *)pa, PGSIZE);
        
        // 映射到子进程页表
        if (mappages(new, i, PGSIZE, (uint64)mem, flags) != 0) {
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

**3. 文件描述符复制**：
`filedup()` 增加文件对象的引用计数，父子进程共享同一个文件表项：
```c
struct file *filedup(struct file *f) {
    acquire(&ftable.lock);
    f->ref++;  // 引用计数 +1
    release(&ftable.lock);
    return f;
}
```

**4. 返回值设置**：
- 父进程：`return pid`（子进程 PID）。
- 子进程：`np->trapframe->a0 = 0`（子进程从系统调用返回时 `a0 = 0`）。

<div data-component="Xv6ForkVisualization"></div>

---

## 2.5 Linux 系统调用深入

### 2.5.1 系统调用约定：寄存器传参（x86-64）

Linux x86-64 遵循 System V ABI 调用约定：

**系统调用寄存器约定**：
| 寄存器   | 用途                     |
|----------|-------------------------|
| `RAX`    | 系统调用号 + 返回值      |
| `RDI`    | 第 1 个参数              |
| `RSI`    | 第 2 个参数              |
| `RDX`    | 第 3 个参数              |
| `R10`    | 第 4 个参数（注意：不是 RCX！）|
| `R8`     | 第 5 个参数              |
| `R9`     | 第 6 个参数              |
| `RCX`    | 保存用户态 RIP（返回地址）|
| `R11`    | 保存用户态 RFLAGS        |

**为什么 R10 而非 RCX？**
`SYSCALL` 指令会自动将返回地址保存到 `RCX`，因此第 4 个参数使用 `R10`。

**示例：write(1, "hello", 5)**

**C 代码**：
```c
write(1, "hello", 5);
```

**对应汇编**（glibc 封装）：
```asm
; syscall号
mov rax, 1          ; SYS_write = 1

; 参数
mov rdi, 1          ; fd = 1 (stdout)
lea rsi, [rel msg]  ; buf = "hello"
mov rdx, 5          ; count = 5

; 系统调用
syscall

; 返回值在 RAX
```

**内核入口（简化版）**：
```asm
; arch/x86/entry/entry_64.S
ENTRY(entry_SYSCALL_64)
    swapgs                      ; 切换 GS 寄存器（指向内核数据）
    movq %rsp, PER_CPU_VAR(rsp_scratch)  ; 保存用户栈指针
    movq PER_CPU_VAR(cpu_current_top_of_stack), %rsp  ; 加载内核栈
    
    pushq $__USER_DS            ; 保存用户数据段
    pushq PER_CPU_VAR(rsp_scratch)  ; 保存用户栈指针
    pushq %r11                  ; 保存 RFLAGS
    pushq $__USER_CS            ; 保存用户代码段
    pushq %rcx                  ; 保存返回地址
    
    pushq %rax                  ; 保存系统调用号
    
    ; ... 保存其他寄存器到栈
    
    call do_syscall_64          ; 调用 C 函数处理
    
    ; ... 恢复寄存器并返回
    sysretq                     ; 返回用户态
END(entry_SYSCALL_64)
```

<div data-component="LinuxSyscallRegisters"></div>

---

### 2.5.2 vDSO（Virtual Dynamic Shared Object）优化

**vDSO 的设计动机**：
某些系统调用（如 `gettimeofday()`、`clock_gettime()`）频繁被调用，但不涉及特权操作，只需读取内核维护的时间戳。传统系统调用的模式切换开销（~100ns）成为性能瓶颈。

**vDSO 的原理**：
内核将部分系统调用的实现代码映射到**每个进程的用户地址空间**，使得这些调用在用户态直接执行，无需陷入内核。

**vDSO 提供的函数**（Linux x86-64）：
- `__vdso_gettimeofday()`
- `__vdso_clock_gettime()`
- `__vdso_time()`
- `__vdso_getcpu()`

**实现机制**：
1. 内核在启动时创建 vDSO 共享库（`linux-vdso.so.1`）。
2. 加载器（`ld.so`）将 vDSO 映射到进程地址空间。
3. glibc 优先调用 vDSO 版本，如果不存在再调用系统调用。

**查看 vDSO 映射**：
```bash
$ cat /proc/self/maps | grep vdso
7ffef7ffa000-7ffef7ffc000 r-xp 00000000 00:00 0  [vdso]
```

**性能对比**：
```c
#include <time.h>
#include <sys/time.h>
#include <stdio.h>

int main() {
    struct timespec ts;
    for (int i = 0; i < 10000000; i++) {
        clock_gettime(CLOCK_REALTIME, &ts);  // vDSO 版本：~10-20ns
        // 传统系统调用版本：~100-200ns
    }
    return 0;
}
```

**vDSO 的限制**：
- 只能用于只读操作（不能修改内核状态）。
- 依赖内核版本（不同内核 vDSO 接口可能不同）。

<div data-component="VDSOPerformanceComparison"></div>

---

### 2.5.3 系统调用追踪：strace 工具使用

**strace** 是调试利器，可以追踪进程的所有系统调用。

**基本用法**：
```bash
# 追踪 ls 命令的系统调用
$ strace ls /tmp

# 输出示例
execve("/usr/bin/ls", ["ls", "/tmp"], 0x7ffe... /* 64 vars */) = 0
brk(NULL)                               = 0x55d5a8e9f000
mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f5c3e2a9000
openat(AT_FDCWD, "/tmp", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
getdents64(3, /* 5 entries */, 32768)  = 160
write(1, "file1\nfile2\n", 13)         = 13
close(3)                                = 0
exit_group(0)                           = ?
```

**常用选项**：
```bash
# 统计系统调用次数和时间
$ strace -c ls /tmp
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 42.31    0.000055          14         4           mmap
 19.23    0.000025          25         1           openat
 15.38    0.000020          20         1           getdents64
  7.69    0.000010          10         1           write
...

# 只追踪特定系统调用
$ strace -e open,read,write cat file.txt

# 追踪正在运行的进程
$ strace -p <PID>

# 追踪子进程
$ strace -f ./parent_process

# 输出到文件
$ strace -o trace.log ./program

# 显示时间戳
$ strace -t ls

# 显示相对时间
$ strace -r ls
```

**调试示例**：
```bash
# 程序打开文件失败，查看原因
$ strace ./myprogram 2>&1 | grep open
openat(AT_FDCWD, "/nonexistent", O_RDONLY) = -1 ENOENT (No such file or directory)
```

<div data-component="StraceOutputAnalyzer"></div>

---

### 2.5.4 新系统调用的添加流程

为 Linux 添加新系统调用需要修改内核源码。以下是完整流程（以 x86-64 为例）：

**步骤 1：定义系统调用号**

编辑 `arch/x86/entry/syscalls/syscall_64.tbl`：
```
# 在文件末尾添加
548    common  my_syscall              sys_my_syscall
```

**步骤 2：声明系统调用**

编辑 `include/linux/syscalls.h`：
```c
asmlinkage long sys_my_syscall(int arg1, const char __user *arg2);
```

**步骤 3：实现系统调用**

创建 `kernel/my_syscall.c`：
```c
#include <linux/syscalls.h>
#include <linux/kernel.h>
#include <linux/uaccess.h>

SYSCALL_DEFINE2(my_syscall, int, arg1, const char __user *, arg2)
{
    char buf[256];
    
    // 从用户空间复制字符串
    if (copy_from_user(buf, arg2, sizeof(buf)) != 0) {
        return -EFAULT;
    }
    
    // 执行操作
    printk(KERN_INFO "my_syscall: arg1=%d, arg2=%s\n", arg1, buf);
    
    return 0;
}
```

**步骤 4：修改 Makefile**

编辑 `kernel/Makefile`：
```makefile
obj-y += my_syscall.o
```

**步骤 5：编译内核**

```bash
$ make -j$(nproc)
$ sudo make modules_install
$ sudo make install
```

**步骤 6：测试系统调用**

```c
#include <unistd.h>
#include <sys/syscall.h>
#include <stdio.h>

#define SYS_my_syscall 548

int main() {
    long ret = syscall(SYS_my_syscall, 42, "Hello");
    printf("Return value: %ld\n", ret);
    return 0;
}
```

编译并运行：
```bash
$ gcc test.c -o test
$ ./test
Return value: 0

$ dmesg | tail -1
my_syscall: arg1=42, arg2=Hello
```

<div data-component="SyscallAdditionWorkflow"></div>

---

## 本章小结

本章深入探讨了操作系统最核心的接口——系统调用。我们学习了：

### 核心概念
1. **系统调用的本质**：用户态与内核态的唯一受控接口，实现特权操作与安全隔离。
2. **POSIX 标准**：保证 Unix-like 系统的源代码兼容性，促进生态繁荣。
3. **性能开销**：模式切换、缓存失效、参数验证是主要开销来源（~100-200ns）。

### 机制与实现
1. **触发方式**：从早期 `INT 0x80` 到现代 `SYSCALL/SYSENTER` 的演进。
2. **参数传递**：通过寄存器传递参数（x86-64 使用 RAX、RDI、RSI、RDX、R10、R8、R9）。
3. **系统调用表**：内核通过系统调用号索引函数指针表，分派到具体处理函数。
4. **错误处理**：通过返回 -1 + 设置 `errno` 的约定。

### xv6 实现剖析
1. **usys.S**：用户空间封装，执行 `ecall` 指令。
2. **syscall.c**：内核分发逻辑，查表调用处理函数。
3. **参数获取**：`argint()`、`argaddr()`、`argstr()` 从寄存器提取参数。
4. **fork() 实现**：进程复制、内存复制（uvmcopy）、文件描述符继承、返回值设置。

### Linux 高级特性
1. **寄存器约定**：System V ABI 定义的参数传递规则。
2. **vDSO 优化**：将部分系统调用映射到用户空间，避免模式切换（性能提升 5-10倍）。
3. **strace 工具**：追踪系统调用，调试利器。
4. **添加系统调用**：修改系统调用表、声明、实现、编译内核的完整流程。

### 关键要点
- 系统调用是操作系统提供的**唯一受控入口**，保证安全与隔离。
- 理解系统调用机制是掌握操作系统内核的基础。
- 不同架构（x86、ARM、RISC-V）的系统调用机制有差异，但核心思想一致。
- 性能优化（vDSO、批处理）在高性能系统中至关重要。

**下一章预告**：我们将深入探讨**进程抽象**，学习进程的创建、状态管理、上下文切换等核心机制。
