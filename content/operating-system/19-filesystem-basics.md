---
title: "Chapter 19: 文件系统基础"
description: "深入理解文件系统的核心抽象：文件、目录、文件描述符，掌握 POSIX 文件 API 与文件共享机制"
updated: "2026-06-11"
---

# Chapter 19: 文件系统基础

> **本章目标**：
> - 理解持久化存储的必要性与存储设备特性
> - 掌握文件的核心抽象：文件、目录、文件描述符
> - 深入理解 POSIX 文件 API 的语义与使用
> - 理解文件共享机制与文件描述符的继承

---

## 19.1 持久化存储概述

### 19.1.1 为什么需要持久化？

在前面的章节中，我们学习了进程、内存管理、调度等机制——它们都是**易失性**的。当进程终止或系统断电时，内存中的所有数据都会丢失。但用户需要数据能够**超越进程的生命周期**——文档需要保存、数据库需要持久存储、配置需要在重启后仍然存在。

**持久化存储**解决了这个问题：将数据写入非易失性存储设备（如硬盘、SSD），即使系统断电，数据也不会丢失。文件系统是操作系统提供持久化存储的核心抽象——它将底层的块设备（一系列固定大小的块）组织成用户友好的层次结构（文件和目录）。

```
用户视角：
  /home/user/document.txt    ← 文件名（人类可读）
  "Hello, World!"            ← 文件内容（字节序列）

底层视角：
  磁盘块 1024-1028           ← 物理存储位置
  0x48 0x65 0x6C 0x6C 0x6F   ← 原始字节

文件系统的角色：
  将用户视角映射到底层视角
  提供创建、读取、写入、删除等操作
  管理磁盘空间的分配与回收
  保护数据的完整性和安全性
```

### 19.1.2 存储设备

理解存储设备的特性对于理解文件系统的设计至关重要。不同的存储设备有不同的性能特征，文件系统的设计需要适应这些特征。

**机械硬盘（HDD）**：

机械硬盘使用旋转的磁盘和移动的磁头来读写数据。它的核心参数是：
- **寻道时间（Seek Time）**：磁头移动到目标磁道的时间，通常 3-10ms
- **旋转延迟（Rotational Latency）**：等待目标扇区旋转到磁头下的时间，7200 RPM 硬盘平均 4.17ms
- **传输时间（Transfer Time）**：实际读写数据的时间，通常 100-200 MB/s

HDD 的关键特性是**顺序访问远快于随机访问**。顺序读取可以达到 200 MB/s，而随机读取（每次寻道）可能只有 0.5-2 MB/s。这就是为什么文件系统设计中"减少寻道"是如此重要的优化目标。

**固态硬盘（SSD）**：

SSD 使用闪存芯片存储数据，没有机械部件。它的核心优势是：
- **随机读取快**：~50-100 μs（比 HDD 快 100 倍）
- **顺序读取快**：~500-3500 MB/s（NVMe）
- **无寻道时间**：任何位置的访问时间相近

SSD 的特殊限制：
- **写入前必须擦除**：闪存不能直接覆盖写入，必须先擦除整个块（通常 256KB-1MB）
- **写放大（Write Amplification）**：实际写入量大于逻辑写入量
- **磨损均衡（Wear Leveling）**：闪存有写入寿命限制（通常 1000-100000 次），需要均匀使用

**NVMe（Non-Volatile Memory Express）**：

NVMe 是专为 SSD 设计的高速接口协议，通过 PCIe 总线直接连接 CPU，绕过了传统 SATA 接口的瓶颈。NVMe SSD 的顺序读取速度可达 7000 MB/s，随机读取延迟可低至 10 μs。

### 19.1.3 文件系统的作用

文件系统是操作系统中负责**持久化存储**的核心子系统。它的主要职责包括：

**组织（Organization）**：将磁盘上的原始块组织成有层次结构的文件和目录。用户不需要知道数据存储在磁盘的哪个块上，只需要通过路径名访问。

**命名（Naming）**：为文件提供人类可读的名称。文件名是用户与文件系统交互的主要接口——用户通过路径名（如 `/home/user/document.txt`）访问文件，而不是通过磁盘块号。

**保护（Protection）**：控制谁可以访问哪些文件。Unix 文件系统使用权限位（读/写/执行 × 用户/组/其他）来实现访问控制。

**共享（Sharing）**：允许多个进程或用户共享同一文件。文件系统通过链接（硬链接和符号链接）和文件描述符的继承来支持共享。

**可靠性（Reliability）**：确保数据在系统崩溃或断电后不会丢失或损坏。日志文件系统（如 ext4）通过写前日志来保证崩溃一致性。

---

## 19.2 文件抽象

### 19.2.1 文件的定义：字节序列 vs 记录序列

在 Unix/Linux 系统中，文件被定义为**字节序列（Byte Sequence）**。这意味着文件没有固定的结构——它只是一系列字节，由应用程序来解释这些字节的含义。

```c
// 文件就是字节序列
// 文件 "data.txt" 的内容：Hello
// 底层存储：0x48 0x65 0x6C 0x6C 0x6F

int fd = open("data.txt", O_RDONLY);
char buf[5];
read(fd, buf, 5);  // 读取 5 个字节
// buf = {'H', 'e', 'l', 'l', 'o'}
```

这种设计的优势是**通用性**——同一个文件可以被不同程序以不同方式解释。例如，一个 JPEG 图片文件就是一系列字节，图片查看器知道如何解释这些字节来显示图像，而文本编辑器会把它当作乱码。

某些操作系统（如早期的 IBM 大型机系统）使用**记录序列（Record Sequence）** 来定义文件——文件由固定长度的记录组成。这种设计简化了某些操作（如按记录号随机访问），但灵活性不如字节序列模型。

### 19.2.2 文件类型

Unix/Linux 系统支持多种文件类型，每种类型有不同的语义：

**普通文件（Regular File）**：包含用户数据的文件。可以是文本文件（ASCII 编码）或二进制文件（如可执行程序、图片、视频）。

```bash
$ ls -la /etc/passwd
-rw-r--r-- 1 root root 2847 Jun 10 10:00 /etc/passwd
# 第一个字符 '-' 表示普通文件
```

**目录（Directory）**：包含其他文件和目录的容器。目录是一种特殊的文件，其内容是一系列目录项（文件名 → inode 号的映射）。

```bash
$ ls -la /home
drwxr-xr-x 3 root root 4096 Jun 10 10:00 .
# 第一个字符 'd' 表示目录
```

**设备文件（Device File）**：代表硬件设备的特殊文件。分为块设备（如硬盘）和字符设备（如终端）。

```bash
$ ls -la /dev/sda
brw-rw---- 1 root disk 8, 0 Jun 10 10:00 /dev/sda
# 第一个字符 'b' 表示块设备

$ ls -la /dev/tty
crw-rw-rw- 1 root tty 5, 0 Jun 10 10:00 /dev/tty
# 第一个字符 'c' 表示字符设备
```

**符号链接（Symbolic Link）**：指向另一个文件的快捷方式。符号链接可以跨越文件系统，可以指向目录。

```bash
$ ls -la /usr/bin/python3
lrwxrwxrwx 1 root root 9 Jun 10 10:00 /usr/bin/python3 -> python3.11
# 第一个字符 'l' 表示符号链接
```

**硬链接（Hard Link）**：指向同一 inode 的多个文件名。硬链接不能跨越文件系统，不能指向目录。

```bash
$ ln file.txt hardlink.txt
$ ls -li file.txt hardlink.txt
1234567 -rw-r--r-- 2 user group 100 Jun 10 10:00 file.txt
1234567 -rw-r--r-- 2 user group 100 Jun 10 10:00 hardlink.txt
# 相同的 inode 号（1234567），链接数为 2
```

### 19.2.3 文件属性

每个文件都有一组属性（元数据），存储在 inode 中：

```
文件属性：
┌─────────────────────────────────────────────────┐
│ inode 号：1234567                               │
│ 文件类型：普通文件                              │
│ 权限：rw-r--r-- (0644)                         │
│ 所有者：user (uid=1000)                        │
│ 组：group (gid=1000)                           │
│ 大小：1024 字节                                │
│ 链接数：2                                      │
│ 最后访问时间（atime）：2026-06-11 10:00:00     │
│ 最后修改时间（mtime）：2026-06-10 15:30:00     │
│ inode 变更时间（ctime）：2026-06-10 15:30:00   │
│ 数据块指针：[1024, 1025, 1026, ...]            │
└─────────────────────────────────────────────────┘
```

**三个时间戳的区别**：
- **atime（Access Time）**：最后一次读取文件的时间
- **mtime（Modification Time）**：最后一次修改文件内容的时间
- **ctime（Change Time）**：最后一次修改文件元数据（权限、所有者等）的时间

```bash
# 查看文件属性
$ stat /etc/passwd
  File: /etc/passwd
  Size: 2847       Blocks: 8          IO Block: 4096   regular file
Access: 2026-06-11 10:00:00.000000000 +0800  ← atime
Modify: 2026-06-10 15:30:00.000000000 +0800  ← mtime
Change: 2026-06-10 15:30:00.000000000 +0800  ← ctime
```

### 19.2.4 文件操作

POSIX 标准定义了以下核心文件操作：

**创建（Create）**：`creat()` 或 `open()` 带 `O_CREAT` 标志。

```c
// 创建文件
int fd = open("newfile.txt", O_CREAT | O_WRONLY, 0644);
```

**打开（Open）**：`open()` 打开文件，返回文件描述符。

```c
// 打开文件
int fd = open("data.txt", O_RDONLY);
if (fd < 0) {
    perror("open");
    exit(1);
}
```

**读取（Read）**：`read()` 从文件读取数据。

```c
char buf[1024];
ssize_t n = read(fd, buf, sizeof(buf));
// n > 0: 成功读取 n 字节
// n = 0: 文件结束（EOF）
// n < 0: 错误
```

**写入（Write）**：`write()` 向文件写入数据。

```c
const char *data = "Hello, World!";
ssize_t n = write(fd, data, strlen(data));
// n >= 0: 成功写入 n 字节
// n < 0: 错误
```

**关闭（Close）**：`close()` 关闭文件描述符，释放资源。

```c
close(fd);
```

**定位（Seek）**：`lseek()` 移动文件偏移量。

```c
// 移动到文件开头
off_t pos = lseek(fd, 0, SEEK_SET);

// 移动到文件末尾
off_t size = lseek(fd, 0, SEEK_END);

// 从当前位置向前移动 10 字节
lseek(fd, -10, SEEK_CUR);
```

**获取文件信息（Stat）**：`stat()` 获取文件的元数据。

```c
struct stat st;
stat("data.txt", &st);
printf("Size: %ld bytes\n", st.st_size);
printf("Permissions: %o\n", st.st_mode & 0777);
printf("Inode: %ld\n", st.st_ino);
```

<div data-component="FileSystemAbstraction"></div>

---

## 19.3 目录抽象

### 19.3.1 目录的作用

目录是文件系统中组织文件的核心机制。目录将文件组织成**层次结构**（树形结构），使得用户可以通过路径名来定位文件。

目录本身是一种特殊的文件——它的内容是一系列**目录项（Directory Entry）**，每个目录项包含文件名和对应的 inode 号。

```
目录 /home/user/ 的内容：
┌──────────────┬────────────┐
│ 文件名       │ inode 号    │
├──────────────┼────────────┤
│ .            │ 12345      │  ← 当前目录
│ ..           │ 12340      │  ← 父目录
│ document.txt │ 12346      │
│ photos/      │ 12347      │
│ .bashrc      │ 12348      │
└──────────────┴────────────┘
```

### 19.3.2 单级目录 vs 多级目录

**单级目录**：所有文件在同一目录下。简单但不实用——文件名冲突、无法组织。

```
单级目录：
/file1.txt
/file2.txt
/program.c
/photo.jpg
→ 所有文件在同一层，难以管理
```

**多级目录（树形结构）**：文件可以组织在子目录中，形成层次结构。

```
多级目录：
/
├── home/
│   └── user/
│       ├── document.txt
│       ├── photos/
│       │   ├── vacation.jpg
│       │   └── family.jpg
│       └── projects/
│           └── code.c
├── etc/
│   └── config.txt
└── tmp/
    └── temp.txt
```

### 19.3.3 路径：绝对路径 vs 相对路径

**绝对路径**：从根目录 `/` 开始的完整路径。

```c
// 绝对路径
open("/home/user/document.txt", O_RDONLY);
```

**相对路径**：从当前工作目录开始的路径。

```c
// 假设当前工作目录是 /home/user
open("document.txt", O_RDONLY);        // 等价于 /home/user/document.txt
open("photos/vacation.jpg", O_RDONLY); // 等价于 /home/user/photos/vacation.jpg
open("../other/file.txt", O_RDONLY);   // 等价于 /home/other/file.txt
```

**路径解析过程**：操作系统将路径名解析为 inode 的过程。

```
解析 "/home/user/document.txt"：
1. 从根目录 inode（通常是 inode 2）开始
2. 在根目录中查找 "home" → 找到 inode 12340
3. 在 /home 目录中查找 "user" → 找到 inode 12345
4. 在 /home/user 目录中查找 "document.txt" → 找到 inode 12346
5. 返回 inode 12346
```

<div data-component="DirectoryTreeVisualizer"></div>

### 19.3.4 当前工作目录（cwd）

每个进程都有一个**当前工作目录（Current Working Directory）**，用于解析相对路径。

```c
// 获取当前工作目录
char cwd[1024];
getcwd(cwd, sizeof(cwd));
printf("Current directory: %s\n", cwd);

// 改变当前工作目录
chdir("/home/user");
```

`fork()` 创建子进程时，子进程继承父进程的当前工作目录。

---

## 19.4 文件系统接口

### 19.4.1 POSIX 文件 API

POSIX 标准定义了一套完整的文件操作 API。让我们详细分析每个函数的语义。

**open() — 打开文件**

```c
int open(const char *pathname, int flags, mode_t mode);
```

`flags` 参数控制打开模式：
- `O_RDONLY`：只读
- `O_WRONLY`：只写
- `O_RDWR`：读写
- `O_CREAT`：文件不存在时创建
- `O_TRUNC`：打开时清空文件
- `O_APPEND`：写入时追加到末尾

`mode` 参数指定新文件的权限（仅在 `O_CREAT` 时有效）：
- `0644`：所有者读写，组和其他只读
- `0755`：所有者读写执行，组和其他读执行

**read() — 读取数据**

```c
ssize_t read(int fd, void *buf, size_t count);
```

`read()` 从文件描述符 `fd` 读取最多 `count` 字节到缓冲区 `buf`。返回实际读取的字节数：
- 返回值 > 0：成功读取了那么多字节
- 返回值 = 0：到达文件末尾（EOF）
- 返回值 < 0：发生错误（检查 `errno`）

**write() — 写入数据**

```c
ssize_t write(int fd, const void *buf, size_t count);
```

`write()` 将缓冲区 `buf` 中的 `count` 字节写入文件描述符 `fd`。返回实际写入的字节数。

**close() — 关闭文件**

```c
int close(int fd);
```

`close()` 关闭文件描述符，释放内核中的相关资源。关闭后，文件描述符 `fd` 可以被重新使用。

### 19.4.2 文件描述符（File Descriptor）

文件描述符是操作系统内核为每个进程维护的**打开文件表**的索引。当进程调用 `open()` 时，内核分配一个最小的可用文件描述符号。

```
进程的文件描述符表：
┌──────┬────────────────────────────┐
│ fd 0 │ stdin  (标准输入)           │ ← 终端
│ fd 1 │ stdout (标准输出)           │ ← 终端
│ fd 2 │ stderr (标准错误)           │ ← 终端
│ fd 3 │ data.txt                   │ ← 文件
│ fd 4 │ socket                     │ ← 网络连接
│ fd 5 │ pipe                       │ ← 管道
└──────┴────────────────────────────┘
```

**标准文件描述符**：每个进程默认打开三个文件描述符：
- `fd 0`（stdin）：标准输入，默认连接到终端
- `fd 1`（stdout）：标准输出，默认连接到终端
- `fd 2`（stderr）：标准错误，默认连接到终端

<div data-component="FileDescriptorTable"></div>

### 19.4.3 文件偏移量（File Offset）

每个打开的文件都有一个**文件偏移量（File Offset）**，表示下一次读写操作的位置。`read()` 和 `write()` 操作会自动推进文件偏移量。

```c
int fd = open("data.txt", O_RDWR);

// 偏移量初始为 0
char buf[5];
read(fd, buf, 5);  // 读取 0-4 字节，偏移量变为 5
read(fd, buf, 5);  // 读取 5-9 字节，偏移量变为 10

// 使用 lseek() 手动移动偏移量
lseek(fd, 0, SEEK_SET);  // 移回文件开头
read(fd, buf, 5);         // 再次读取 0-4 字节
```

<div data-component="FileOffsetSimulator"></div>

### 19.4.4 打开文件表

操作系统维护三个层次的打开文件表：

**进程级：文件描述符表**：每个进程有自己的文件描述符表，将文件描述符号映射到系统级打开文件表的条目。

**系统级：打开文件表**：所有进程共享的全局表，存储打开文件的状态（偏移量、打开模式、引用计数）。

**inode 表**：存储文件的元数据（大小、权限、数据块位置等）。

```
文件描述符表（进程 A）     打开文件表（系统级）     inode 表
┌──────┬─────────┐      ┌─────────────────┐   ┌──────────────┐
│ fd 0 │ ──────→ │      │ 偏移量: 100     │   │ inode 12346  │
│ fd 1 │ ──────→ │      │ 模式: O_RDWR    │   │ 大小: 1024   │
│ fd 3 │ ──────→ │      │ 引用计数: 2     │──→│ 权限: 0644   │
└──────┘         │      │ inode 指针 ─────│   │ 数据块: [...]│
                 │      └─────────────────┘   └──────────────┘
文件描述符表（进程 B）
┌──────┬─────────┐
│ fd 0 │ ──────→ │  ← 共享同一个打开文件表条目
│ fd 3 │ ──────→ │
└──────┘
```

---

## 19.5 文件共享

### 19.5.1 多进程共享文件

多个进程可以通过以下方式共享文件：

**共享同一打开文件**：多个进程打开同一个文件时，它们共享同一个打开文件表条目（如果使用相同的文件描述符）。这意味着一个进程的写入会影响另一个进程的偏移量。

```c
// 父进程
int fd = open("shared.txt", O_RDWR);
write(fd, "Hello", 5);

if (fork() == 0) {
    // 子进程继承 fd
    char buf[10];
    read(fd, buf, 5);  // 读取 "World"（因为偏移量已被父进程推进）
}
```

### 19.5.2 fork() 后的文件描述符继承

`fork()` 创建子进程时，子进程**继承**父进程的所有文件描述符。子进程的文件描述符表是父进程的副本，但它们指向相同的打开文件表条目。

```c
int fd = open("data.txt", O_RDWR);
write(fd, "Hello", 5);  // 偏移量 = 5

if (fork() == 0) {
    // 子进程：fd 仍然指向同一个打开文件
    write(fd, "World", 5);  // 偏移量 = 10
    exit(0);
}

wait(NULL);
// 父进程：偏移量 = 10（被子进程修改了！）
lseek(fd, 0, SEEK_SET);
char buf[11];
read(fd, buf, 10);
buf[10] = '\0';
printf("%s\n", buf);  // 输出 "HelloWorld"
```

### 19.5.3 dup() 与 dup2()：文件描述符复制

```c
int dup(int oldfd);      // 复制 oldfd，返回新的文件描述符
int dup2(int oldfd, int newfd);  // 复制 oldfd 到 newfd
```

`dup()` 创建一个指向同一个打开文件的新文件描述符。

```c
int fd = open("data.txt", O_WRONLY);
int fd2 = dup(fd);  // fd2 指向同一个打开文件

write(fd, "Hello", 5);
write(fd2, "World", 5);  // 写入同一个文件，偏移量共享
```

`dup2()` 常用于重定向：

```c
// 将标准输出重定向到文件
int fd = open("output.txt", O_WRONLY | O_CREAT, 0644);
dup2(fd, STDOUT_FILENO);  // fd 1 现在指向 output.txt
close(fd);
printf("This goes to output.txt\n");
```

### 19.5.4 文件锁

当多个进程同时写入同一文件时，需要同步机制来防止数据损坏。

**建议锁（Advisory Lock）**：进程自愿遵守锁协议。如果进程不检查锁，它仍然可以写入文件。

```c
// 使用 flock() 获取建议锁
int fd = open("shared.txt", O_RDWR);
flock(fd, LOCK_EX);    // 获取排他锁
// 写入数据
flock(fd, LOCK_UN);    // 释放锁
```

**强制锁（Mandatory Lock）**：内核强制执行锁。如果进程尝试违反锁，`read()` 和 `write()` 会阻塞或失败。Linux 默认不启用强制锁。

---

## 19.6 面试高频考点

**Q1：文件描述符是什么？**

文件描述符是操作系统内核为每个进程维护的打开文件表的索引。当进程调用 `open()` 时，内核分配一个最小的可用文件描述符号。默认有三个：0（stdin）、1（stdout）、2（stderr）。

**Q2：硬链接和符号链接的区别？**

硬链接指向同一 inode，不能跨文件系统，不能指向目录。符号链接是独立的文件，存储目标路径，可以跨文件系统，可以指向目录。

**Q3：fork() 后文件描述符的状态？**

子进程继承父进程的所有文件描述符，指向相同的打开文件表条目。父子进程共享文件偏移量——一个进程的写入会影响另一个进程的偏移量。

**Q4：mtime 和 ctime 的区别？**

mtime 是最后一次修改文件**内容**的时间。ctime 是最后一次修改文件**元数据**（权限、所有者等）的时间。修改内容会同时更新 mtime 和 ctime。

**Q5：dup2() 的用途？**

`dup2(oldfd, newfd)` 将 oldfd 复制到 newfd。常用于 I/O 重定向——将标准输出重定向到文件，或在 shell 中实现管道。

---

## 19.7 文件操作的内核实现

### 19.7.1 open() 的内核实现

当用户调用 `open()` 时，内核执行以下步骤：

```
用户调用 open("data.txt", O_RDONLY)
    │
    ▼
系统调用入口（陷入内核）
    │
    ▼
1. 路径解析：将 "data.txt" 解析为 inode
    ├── 查找当前目录的 inode
    ├── 在目录中查找 "data.txt" 的 inode 号
    └── 从 inode 表中读取 inode
    │
    ▼
2. 权限检查：检查进程是否有权限打开文件
    ├── 检查文件权限位（rwx）
    └── 检查进程的 UID/GID
    │
    ▼
3. 分配文件描述符：在进程的文件描述符表中找最小可用 fd
    │
    ▼
4. 创建打开文件表条目：分配 struct file，设置偏移量、模式
    │
    ▼
5. 返回文件描述符号给用户
```

### 19.7.2 read() 的内核实现

```
用户调用 read(fd, buf, count)
    │
    ▼
系统调用入口
    │
    ▼
1. 查找文件描述符：从进程的文件描述符表中获取 struct file
    │
    ▼
2. 检查权限：检查文件是否以读模式打开
    │
    ▼
3. 计算磁盘位置：根据文件偏移量和 inode 中的数据块指针
    ├── 计算逻辑块号 = 偏移量 / 块大小
    ├── 通过 inode 的直接/间接指针找到物理块号
    └── 计算块内偏移
    │
    ▼
4. 读取数据：从磁盘读取数据到内核缓冲区
    ├── 检查缓冲区缓存（Buffer Cache）
    ├── 如果缓存命中：直接返回
    └── 如果缓存未命中：从磁盘读取
    │
    ▼
5. 复制到用户空间：将数据从内核缓冲区复制到用户缓冲区
    │
    ▼
6. 更新文件偏移量：偏移量 += 实际读取字节数
    │
    ▼
7. 返回实际读取的字节数
```

### 19.7.3 write() 的内核实现

```
用户调用 write(fd, buf, count)
    │
    ▼
系统调用入口
    │
    ▼
1. 查找文件描述符
2. 检查权限（写模式）
3. 计算磁盘位置
4. 将数据从用户空间复制到内核缓冲区
5. 将缓冲区标记为"脏"（需要写回磁盘）
6. 更新文件大小（如果扩展了文件）
7. 更新文件偏移量
8. 返回实际写入的字节数
```

**注意**：`write()` 通常不立即写入磁盘——数据先写入缓冲区缓存，稍后由内核的回写线程（writeback thread）批量写入磁盘。这提高了性能但增加了崩溃风险（数据可能丢失）。使用 `fsync()` 可以强制将数据写入磁盘。

---

## 19.8 文件系统的性能特征

### 19.8.1 顺序访问 vs 随机访问

```
访问模式        HDD            SSD            NVMe
顺序读取        ~200 MB/s      ~500 MB/s      ~3500 MB/s
随机读取(4KB)   ~0.5 MB/s      ~50 MB/s       ~500 MB/s
顺序写入        ~180 MB/s      ~450 MB/s      ~3000 MB/s
随机写入(4KB)   ~0.4 MB/s      ~40 MB/s       ~400 MB/s
```

**关键洞察**：HDD 的顺序访问比随机访问快 400 倍！这就是为什么文件系统设计中"减少磁盘寻道"是如此重要的优化目标。SSD 的差距小得多（约 10 倍），但仍然显著。

### 19.8.2 缓存的作用

文件系统使用**缓冲区缓存（Buffer Cache）** 来减少磁盘访问次数。频繁访问的数据块被缓存在内存中，后续访问直接从缓存读取。

```
缓存命中：~100 ns（内存访问）
缓存未命中：~10 ms（HDD 磁盘访问）
命中率 99% 时的平均延迟：0.99 × 100ns + 0.01 × 10ms = 200ns
→ 缓存将访问延迟降低了 50 倍！
```

---

## 19.9 手算练习

### 练习 1：文件描述符分配

**题目**：进程依次执行以下操作，列出每次操作后的文件描述符表。

```c
int fd1 = open("a.txt", O_RDONLY);     // 操作 1
int fd2 = open("b.txt", O_WRONLY);     // 操作 2
close(fd1);                            // 操作 3
int fd3 = open("c.txt", O_RDWR);       // 操作 4
int fd4 = dup(fd3);                    // 操作 5
```

**解答**：

```
初始状态：
fd 0: stdin
fd 1: stdout
fd 2: stderr

操作 1 后（fd1 = 3）：
fd 0: stdin
fd 1: stdout
fd 2: stderr
fd 3: a.txt (只读)

操作 2 后（fd2 = 4）：
fd 0: stdin
fd 1: stdout
fd 2: stderr
fd 3: a.txt (只读)
fd 4: b.txt (只写)

操作 3 后（close fd1）：
fd 0: stdin
fd 1: stdout
fd 2: stderr
fd 3: (空闲)
fd 4: b.txt (只写)

操作 4 后（fd3 = 3）：
fd 0: stdin
fd 1: stdout
fd 2: stderr
fd 3: c.txt (读写)
fd 4: b.txt (只写)

操作 5 后（fd4 = 5）：
fd 0: stdin
fd 1: stdout
fd 2: stderr
fd 3: c.txt (读写)
fd 4: b.txt (只写)
fd 5: c.txt (读写)  ← 与 fd3 指向同一个打开文件
```

### 练习 2：文件偏移量追踪

**题目**：追踪以下操作序列中的文件偏移量变化。

```c
int fd = open("data.txt", O_RDWR);
// 文件内容：ABCDEFGHIJ（10 字节）
read(fd, buf, 3);      // 操作 1
lseek(fd, 5, SEEK_SET); // 操作 2
read(fd, buf, 2);      // 操作 3
write(fd, "XY", 2);    // 操作 4
lseek(fd, -4, SEEK_CUR); // 操作 5
read(fd, buf, 3);      // 操作 6
```

**解答**：

```
初始偏移量：0

操作 1：read(fd, buf, 3) → 读取 ABC，偏移量 = 3
操作 2：lseek(fd, 5, SEEK_SET) → 偏移量 = 5
操作 3：read(fd, buf, 2) → 读取 FG，偏移量 = 7
操作 4：write(fd, "XY", 2) → 写入 XY（覆盖 HI），偏移量 = 9
操作 5：lseek(fd, -4, SEEK_CUR) → 偏移量 = 9 - 4 = 5
操作 6：read(fd, buf, 3) → 读取 FXY，偏移量 = 8

文件内容变为：ABCDEFGXYJ
```

---

## 19.10 常见错误

### 19.10.1 忘记检查返回值

```c
// 错误：不检查 open() 的返回值
int fd = open("data.txt", O_RDONLY);
read(fd, buf, 100);  // 如果 open 失败，fd = -1，read 会失败

// 正确：检查返回值
int fd = open("data.txt", O_RDONLY);
if (fd < 0) {
    perror("open");
    exit(1);
}
```

### 19.10.2 忘记关闭文件描述符

```c
// 错误：循环中不断打开文件但不关闭
for (int i = 0; i < 100000; i++) {
    int fd = open("data.txt", O_RDONLY);
    read(fd, buf, 100);
    // 忘记 close(fd) → 文件描述符泄漏！
}

// 正确：及时关闭
for (int i = 0; i < 100000; i++) {
    int fd = open("data.txt", O_RDONLY);
    read(fd, buf, 100);
    close(fd);
}
```

### 19.10.3 不检查 read() 的返回值

```c
// 错误：假设 read() 一次读取所有数据
char buf[1000];
read(fd, buf, 1000);
// read() 可能返回 < 1000（如遇到 EOF 或信号中断）

// 正确：循环读取直到获取所有数据
ssize_t total = 0;
while (total < 1000) {
    ssize_t n = read(fd, buf + total, 1000 - total);
    if (n <= 0) break;
    total += n;
}
```

---

## 19.11 推荐实践

1. **始终检查返回值**：`open()`、`read()`、`write()` 都可能失败
2. **及时关闭文件描述符**：避免资源泄漏
3. **使用缓冲 I/O**：`fread()`/`fwrite()` 比 `read()`/`write()` 更高效
4. **使用 fsync() 确保数据持久化**：重要数据写入后调用 `fsync()`
5. **避免不必要的文件操作**：缓存文件大小、权限等信息

---

## 19.12 扩展阅读

- **OSTEP** Chapter 39: "Files and Directories" — 文件与目录
- **OSTEP** Chapter 40: "File System Implementation" — 文件系统实现
- **Operating System Concepts** Chapter 11: "File System Interface" — 文件系统接口
- [Linux man pages: open(2)](https://man7.org/linux/man-pages/man2/open.2.html)
- [Linux man pages: read(2)](https://man7.org/linux/man-pages/man2/read.2.html)
- [Linux man pages: write(2)](https://man7.org/linux/man-pages/man2/write.2.html)
- [Linux man pages: dup(2)](https://man7.org/linux/man-pages/man2/dup.2.html)

### 19.12.1 推荐实践

1. **始终检查返回值**：`open()`、`read()`、`write()` 都可能失败
2. **及时关闭文件描述符**：避免资源泄漏
3. **使用缓冲 I/O**：`fread()`/`fwrite()` 比 `read()`/`write()` 更高效
4. **使用 fsync() 确保数据持久化**：重要数据写入后调用 `fsync()`
5. **避免不必要的文件操作**：缓存文件大小、权限等信息

### 19.12.2 文件系统的面试准备

**常见面试题**：
1. 文件描述符是什么？与文件指针的区别？
2. 硬链接和符号链接的区别？
3. fork() 后文件描述符的状态？
4. dup2() 的用途？
5. 文件系统的层次结构？

**面试技巧**：
1. 画图：画出文件描述符表、打开文件表、inode 表的关系
2. 追踪：追踪 open()/read()/write() 的内核执行路径
3. 对比：比较不同文件系统的优缺点
4. 应用：举出实际系统中的文件系统设计案例

### 19.12.3 文件系统的历史

文件系统的设计随着存储技术的发展而不断演进：

**1960-1970 年代**：早期文件系统使用简单的连续分配或链接分配。Unix 文件系统引入了 inode 概念。

**1980-1990 年代**：Berkeley FFS（Fast File System）引入了块组（block group）和位图（bitmap），显著提高了性能。

**2000 年代**：日志文件系统（ext3、XFS）通过写前日志提高了崩溃一致性。Btrfs 和 ZFS 引入了写时复制和快照功能。

**2010 年代至今**：针对 SSD 优化的文件系统（F2FS）和分布式文件系统（Ceph、GlusterFS）。

### 19.12.4 文件系统的设计挑战

1. **性能**：如何最小化磁盘访问次数？
2. **可靠性**：如何在崩溃后恢复数据？
3. **安全性**：如何控制文件访问权限？
4. **可扩展性**：如何支持大文件和大目录？
5. **兼容性**：如何支持不同的存储设备？

### 19.12.5 文件系统的总结

文件系统是操作系统中最重要的子系统之一。它将底层的块设备抽象为用户友好的文件和目录，提供了持久化存储、数据保护、文件共享等核心功能。

**关键要点**：
1. 文件是字节序列，由应用程序解释其含义
2. 文件描述符是打开文件的句柄，由内核管理
3. inode 存储文件的元数据，与文件名分离
4. 目录是特殊的文件，存储文件名到 inode 的映射
5. 文件系统通过缓冲区缓存提高性能

### 19.12.6 文件系统的扩展阅读

- **OSTEP** Chapter 39: "Files and Directories" — 文件与目录
- **OSTEP** Chapter 40: "File System Implementation" — 文件系统实现
- **Operating System Concepts** Chapter 11: "File System Interface" — 文件系统接口
- [Linux man pages: open(2)](https://man7.org/linux/man-pages/man2/open.2.html)
- [Linux man pages: read(2)](https://man7.org/linux/man-pages/man2/read.2.html)
- [Linux man pages: write(2)](https://man7.org/linux/man-pages/man2/write.2.html)
- [Linux man pages: dup(2)](https://man7.org/linux/man-pages/man2/dup.2.html)
- [Linux man pages: stat(2)](https://man7.org/linux/man-pages/man2/stat.2.html)
- [Linux man pages: lseek(2)](https://man7.org/linux/man-pages/man2/lseek.2.html)

### 19.12.7 推荐实践项目

1. **实现简单的文件系统**：在内存中实现一个支持创建、读取、写入、删除文件的简单文件系统
2. **实现文件描述符表**：模拟内核的文件描述符管理
3. **实现路径解析**：将路径名解析为 inode 的算法
4. **实现缓冲区缓存**：LRU 缓存策略的实现
5. **使用 strace 追踪文件操作**：观察 open()/read()/write() 的系统调用

### 19.12.8 文件系统的性能优化

**1. 减少磁盘访问**：使用缓冲区缓存、预读（readahead）、延迟写入（writeback）

**2. 减少寻道**：将相关数据放在相邻的磁盘块上（块组、分配策略）

**3. 批量操作**：将多个小写入合并为一个大写入（日志、写合并）

**4. 异步操作**：使用异步 I/O（aio、io_uring）避免阻塞

**5. 并行访问**：使用多线程同时访问不同的文件或文件区域

### 19.12.9 文件系统的常见问题

**问题 1：文件描述符泄漏**

```c
// 症状：程序运行一段时间后无法打开文件
// 原因：打开文件后忘记关闭
// 诊断：ls -l /proc/<pid>/fd | wc -l
```

**问题 2：文件偏移量混乱**

```c
// 症状：读取到错误的数据
// 原因：多个线程共享文件描述符，但没有同步
// 解决：使用 pread()/pwrite() 指定偏移量
```

**问题 3：数据丢失**

```c
// 症状：写入的数据在崩溃后丢失
// 原因：write() 只写入缓存，没有写入磁盘
// 解决：重要数据写入后调用 fsync()
```

### 19.12.10 文件系统的总结

文件系统是操作系统中最重要的子系统之一。它将底层的块设备抽象为用户友好的文件和目录，提供了持久化存储、数据保护、文件共享等核心功能。

**关键要点**：
1. 文件是字节序列，由应用程序解释其含义
2. 文件描述符是打开文件的句柄，由内核管理
3. inode 存储文件的元数据，与文件名分离
4. 目录是特殊的文件，存储文件名到 inode 的映射
5. 文件系统通过缓冲区缓存提高性能
