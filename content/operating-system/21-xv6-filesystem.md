---
title: "Chapter 21: xv6 文件系统剖析"
description: "深入理解 xv6 文件系统的七层架构，掌握缓冲区缓存、inode 层、日志层、路径解析的实现细节"
updated: "2026-06-11"
---

# Chapter 21: xv6 文件系统剖析

> **本章目标**：
> - 理解 xv6 文件系统的七层架构设计
> - 掌握缓冲区缓存的实现与 LRU 替换策略
> - 深入理解 inode 层的操作：ialloc、iget、bmap、readi、writei
> - 理解目录层与路径名层的实现
> - 掌握文件描述符层与系统调用的实现

---

## 21.1 xv6 文件系统概述

### 21.1.1 设计目标

xv6 的文件系统设计遵循以下原则：
- **简单性**：代码量小，易于理解和教学
- **正确性**：正确实现 Unix 文件系统的基本语义
- **效率**：在简单性的前提下尽可能高效

xv6 文件系统基于经典的 Unix 文件系统设计，使用 inode、目录、路径名等概念。它的实现分为七个层次，每层向上层提供抽象接口，向下层调用底层服务。

### 21.1.2 文件系统层次

```
用户空间
    │
    ▼
┌─────────────────────────────────────────┐
│ 第 7 层：文件描述符层                    │
│ struct file, fd table                   │
│ sys_open, sys_read, sys_write, sys_close│
├─────────────────────────────────────────┤
│ 第 6 层：路径名层                        │
│ namei(), nameiparent()                  │
│ 路径解析："/home/user/file" → inode     │
├─────────────────────────────────────────┤
│ 第 5 层：目录层                          │
│ struct dirent                           │
│ dirlookup(), dirlink()                  │
├─────────────────────────────────────────┤
│ 第 4 层：inode 层                        │
│ struct inode, struct dinode             │
│ ialloc(), iget(), ilock()              │
│ bmap(), readi(), writei()              │
├─────────────────────────────────────────┤
│ 第 3 层：日志层                          │
│ struct log                              │
│ begin_op(), end_op(), log_write()       │
├─────────────────────────────────────────┤
│ 第 2 层：缓冲区缓存层                    │
│ struct buf                              │
│ bread(), bwrite(), brelse()            │
├─────────────────────────────────────────┤
│ 第 1 层：磁盘层                          │
│ virtio_disk_intr(), virtio_disk_rw()   │
└─────────────────────────────────────────┘
    │
    ▼
物理磁盘
```

<div data-component="Xv6FSArchitecture"></div>

---

## 21.2 磁盘布局

### 21.2.1 块大小与磁盘布局

xv6 使用 1024 字节的块大小（BSIZE=1024）。磁盘布局如下：

```
块号    内容
0       引导块（boot block）— 未使用
1       超级块（superblock）
2-N     日志块（log blocks）
N+1-M   inode 块（inode blocks）
M+1-M+K 位图块（bitmap blocks）
M+K+1-End 数据块（data blocks）
```

### 21.2.2 超级块（block 1）

超级块存储文件系统的元数据：

```c
// kernel/fs.h
struct superblock {
    uint magic;      // 魔数：0x10203040
    uint size;       // 文件系统总块数
    uint nblocks;    // 数据块数
    uint ninodes;    // inode 数
    uint nlog;       // 日志块数
    uint logstart;   // 日志起始块号
    uint inodestart; // inode 起始块号
    uint bmapstart;  // 位图起始块号
};
```

### 21.2.3 日志块

日志块用于写前日志（WAL），保证崩溃一致性。日志块的数量在文件系统创建时确定（mkfs）。

### 21.2.4 inode 块

inode 块存储所有 inode 的数据。每个 inode 64 字节，一个块可以存储 16 个 inode。

### 21.2.5 位图块

位图块追踪哪些数据块是空闲的。每个位对应一个数据块：0 = 空闲，1 = 已分配。

### 21.2.6 数据块

数据块存储文件的实际数据和目录的内容。

<div data-component="Xv6DiskLayout"></div>

---

## 21.3 缓冲区缓存（Buffer Cache）

### 21.3.1 struct buf 数据结构

缓冲区缓存是文件系统与磁盘之间的缓存层。每个缓存块对应一个磁盘块：

```c
// kernel/buf.h
struct buf {
    int valid;     // 数据是否有效（已从磁盘读取）
    int disk;      // 是否正在磁盘 I/O
    uint dev;      // 设备号
    uint blockno;  // 磁盘块号
    struct sleeplock lock;  // 睡眠锁
    uint refcnt;   // 引用计数
    struct buf *prev;  // LRU 链表
    struct buf *next;
    uchar data[BSIZE];  // 块数据（1024 字节）
};
```

### 21.3.2 bread() 与 bwrite()

```c
// kernel/bio.c
struct buf* bread(uint dev, uint blockno) {
    struct buf *b;
    
    // 查找缓存
    b = bget(dev, blockno);
    
    if (!b->valid) {
        // 缓存未命中，从磁盘读取
        virtio_disk_rw(b, 0);  // 0 = 读
        b->valid = 1;
    }
    return b;
}

void bwrite(struct buf *b) {
    if (!holding(&b->lock))
        panic("bwrite");
    virtio_disk_rw(b, 1);  // 1 = 写
}
```

### 21.3.3 LRU 替换策略

xv6 使用 LRU（最近最少使用）策略替换缓存块：

```c
// kernel/bio.c
static struct buf head;  // LRU 链表头

// bget() 查找缓存块
struct buf* bget(uint dev, uint blockno) {
    struct buf *b;
    
    // 检查是否在缓存中
    for (b = head.next; b != &head; b = b->next) {
        if (b->dev == dev && b->blockno == blockno) {
            b->refcnt++;
            acquire(&bcache.lock);
            return b;
        }
    }
    
    // 不在缓存中，找一个 LRU 块替换
    for (b = head.prev; b != &head; b = b->prev) {
        if (b->refcnt == 0) {
            b->dev = dev;
            b->blockno = blockno;
            b->valid = 0;
            b->refcnt = 1;
            return b;
        }
    }
    
    panic("bget: no buffers");
}
```

<div data-component="BufferCacheLRU"></div>

### 21.3.4 缓冲区锁

每个缓冲区有一个睡眠锁，保证同一时刻只有一个线程可以访问该缓冲区。

```c
// 获取缓冲区锁
struct buf *b = bread(dev, blockno);
acquiresleep(&b->lock);
// 修改数据
b->data[0] = 42;
// 写回磁盘
bwrite(b);
// 释放锁
releasesleep(&b->lock);
// 释放缓冲区
brelse(b);
```

### 21.3.5 缓冲区缓存的作用

缓冲区缓存有三个重要作用：

1. **同步访问磁盘块**：通过锁保证同一时刻只有一个线程可以修改同一块
2. **缓存常用块**：减少磁盘访问次数，提高性能
3. **协调块的修改**：多个操作可以修改同一块的不同部分，缓存保证一致性

---

## 21.4 inode 层

### 21.4.1 struct dinode（磁盘 inode）

磁盘上的 inode 结构：

```c
// kernel/fs.h
struct dinode {
    short type;           // 文件类型（0=空闲, 1=目录, 2=文件, 3=设备）
    short major;          // 主设备号（设备文件）
    short minor;          // 次设备号（设备文件）
    short nlink;          // 硬链接数
    uint size;            // 文件大小（字节）
    uint addrs[NDIRECT+2]; // 数据块指针（12 直接 + 1 单间接 + 1 双间接）
};
```

### 21.4.2 struct inode（内存 inode）

内存中的 inode 结构，包含磁盘 inode 的数据加上管理信息：

```c
// kernel/file.h
struct inode {
    uint dev;           // 设备号
    uint inum;          // inode 号
    int ref;            // 引用计数
    struct sleeplock lock;  // 睡眠锁
    int valid;          // 内存中的数据是否有效
    
    // 以下字段从磁盘 inode 复制
    short type;
    short major;
    short minor;
    short nlink;
    uint size;
    uint addrs[NDIRECT+2];
};
```

### 21.4.3 ialloc() 与 iget()

```c
// 分配新的 inode
struct inode* ialloc(uint dev, short type) {
    int inum;
    struct buf *bp;
    struct dinode *dip;
    
    for (inum = 1; inum < sb.ninodes; inum++) {
        bp = bread(dev, IBLOCK(inum, sb));
        dip = (struct dinode *)bp->data + inum % IPB;
        
        if (dip->type == 0) {  // 找到空闲 inode
            memset(dip, 0, sizeof(*dip));
            dip->type = type;
            log_write(bp);  // 写入日志
            brelse(bp);
            return iget(dev, inum);
        }
        brelse(bp);
    }
    panic("ialloc: no inodes");
}

// 获取 inode（从缓存或磁盘）
struct inode* iget(uint dev, uint inum) {
    struct inode *ip, *empty;
    
    // 检查缓存
    for (ip = &icache.inode[0]; ip < &icache.inode[NINODE]; ip++) {
        if (ip->ref > 0 && ip->dev == dev && ip->inum == inum) {
            ip->ref++;
            return ip;
        }
    }
    
    // 不在缓存中，找一个空闲槽位
    empty = 0;
    for (ip = &icache.inode[0]; ip < &icache.inode[NINODE]; ip++) {
        if (ip->ref == 0) {
            empty = ip;
            break;
        }
    }
    
    if (!empty)
        panic("iget: no inodes");
    
    ip = empty;
    ip->dev = dev;
    ip->inum = inum;
    ip->ref = 1;
    ip->valid = 0;  // 需要从磁盘读取
    return ip;
}
```

### 21.4.4 ilock() 与 iunlock()

```c
// 锁定 inode 并从磁盘读取数据
void ilock(struct inode *ip) {
    if (ip == 0 || ip->ref < 1)
        panic("ilock");
    
    acquiresleep(&ip->lock);
    
    if (ip->valid == 0) {
        // 从磁盘读取 inode 数据
        struct buf *bp = bread(ip->dev, IBLOCK(ip->inum, sb));
        struct dinode *dip = (struct dinode *)bp->data + ip->inum % IPB;
        ip->type = dip->type;
        ip->major = dip->major;
        ip->minor = dip->minor;
        ip->nlink = dip->nlink;
        ip->size = dip->size;
        memmove(ip->addrs, dip->addrs, sizeof(ip->addrs));
        brelse(bp);
        ip->valid = 1;
        if (ip->type == 0)
            panic("ilock: no type");
    }
}

void iunlock(struct inode *ip) {
    if (ip == 0 || !holding(&ip->lock) || ip->ref < 1)
        panic("iunlock");
    releasesleep(&ip->lock);
}
```

### 21.4.5 bmap()：逻辑块号 → 物理块号

```c
// 将逻辑块号映射到物理块号
static uint bmap(struct inode *ip, uint bn) {
    uint addr, *a;
    struct buf *bp;
    
    if (bn < NDIRECT) {
        // 直接指针
        if ((addr = ip->addrs[bn]) == 0) {
            addr = balloc(ip->dev);
            ip->addrs[bn] = addr;
        }
        return addr;
    }
    bn -= NDIRECT;
    
    if (bn < NINDIRECT) {
        // 单间接指针
        if ((addr = ip->addrs[NDIRECT]) == 0) {
            addr = balloc(ip->dev);
            ip->addrs[NDIRECT] = addr;
        }
        bp = bread(ip->dev, addr);
        a = (uint *)bp->data;
        if ((addr = a[bn]) == 0) {
            addr = balloc(ip->dev);
            a[bn] = addr;
            log_write(bp);
        }
        brelse(bp);
        return addr;
    }
    bn -= NINDIRECT;
    
    // 双间接指针（类似处理）
    // ...
    
    panic("bmap: out of range");
}
```

<div data-component="Xv6BmapVisualizer"></div>

### 21.4.6 readi() 与 writei()

```c
// 从 inode 读取数据
int readi(struct inode *ip, char *dst, uint off, uint n) {
    uint tot, m;
    struct buf *bp;
    
    if (ip->type == T_DEV) {
        // 设备文件
        if (ip->major < 0 || ip->major >= NDEV || !devsw[ip->major].read)
            return -1;
        return devsw[ip->major].read(ip, dst, n);
    }
    
    if (off > ip->size || off + n < off)
        return -1;
    if (off + n > ip->size)
        n = ip->size - off;
    
    for (tot = 0; tot < n; tot += m, off += m, dst += m) {
        bp = bread(ip->dev, bmap(ip, off / BSIZE));
        m = min(n - tot, BSIZE - off % BSIZE);
        memmove(dst, bp->data + off % BSIZE, m);
        brelse(bp);
    }
    return n;
}

// 向 inode 写入数据
int writei(struct inode *ip, char *src, uint off, uint n) {
    uint tot, m;
    struct buf *bp;
    
    if (ip->type == T_DEV) {
        if (ip->major < 0 || ip->major >= NDEV || !devsw[ip->major].write)
            return -1;
        return devsw[ip->major].write(ip, src, n);
    }
    
    if (off > ip->size || off + n < off)
        return -1;
    if (off + n > MAXFILE * BSIZE)
        return -1;
    
    for (tot = 0; tot < n; tot += m, off += m, src += m) {
        bp = bread(ip->dev, bmap(ip, off / BSIZE));
        m = min(n - tot, BSIZE - off % BSIZE);
        memmove(bp->data + off % BSIZE, src, m);
        log_write(bp);
        brelse(bp);
    }
    
    if (n > 0 && off > ip->size) {
        ip->size = off;
        iupdate(ip);
    }
    return n;
}
```

---

## 21.5 目录层

### 21.5.1 struct dirent 目录项

```c
// kernel/fs.h
struct dirent {
    ushort inum;    // inode 号
    char name[DIRSIZ];  // 文件名（14 字节）
};
```

### 21.5.2 dirlookup()：目录查找

```c
// 在目录中查找文件名
struct inode* dirlookup(struct inode *dp, char *name, uint *poff) {
    uint off, inum;
    struct dirent de;
    
    if (dp->type != T_DIR)
        panic("dirlookup not DIR");
    
    for (off = 0; off < dp->size; off += sizeof(de)) {
        if (readi(dp, (char *)&de, off, sizeof(de)) != sizeof(de))
            panic("dirlookup read");
        if (de.inum == 0)
            continue;
        if (namecmp(name, de.name) == 0) {
            // 找到匹配的目录项
            if (poff)
                *poff = off;
            return iget(dp->dev, de.inum);
        }
    }
    
    return 0;  // 未找到
}
```

### 21.5.3 dirlink()：添加目录项

```c
// 在目录中添加新的目录项
int dirlink(struct inode *dp, char *name, uint inum) {
    int off;
    struct dirent de;
    struct inode *ip;
    
    // 检查是否已存在同名文件
    if ((ip = dirlookup(dp, name, 0)) != 0) {
        iput(ip);
        return -1;
    }
    
    // 查找空闲目录项
    for (off = 0; off < dp->size; off += sizeof(de)) {
        if (readi(dp, (char *)&de, off, sizeof(de)) != sizeof(de))
            panic("dirlink read");
        if (de.inum == 0)
            break;
    }
    
    // 写入新目录项
    de.inum = inum;
    safestrcpy(de.name, name, DIRSIZ);
    if (writei(dp, (char *)&de, off, sizeof(de)) != sizeof(de))
        panic("dirlink write");
    
    return 0;
}
```

---

## 21.6 路径名层

### 21.6.1 namei() 与 nameiparent()

```c
// 解析路径名，返回 inode
struct inode* name(char *path) {
    char *s;
    struct inode *ip;
    
    if (*path == '/')
        ip = iget(ROOTDEV, ROOTINO);  // 从根目录开始
    else
        ip = idup(myproc()->cwd);     // 从当前目录开始
    
    while ((s = skipelem(path, name)) != 0) {
        ilock(ip);
        if (ip->type != T_DIR) {
            iunlockput(ip);
            return 0;
        }
        if (namecmp(name, ".") == 0) {
            // 当前目录
            iunlock(ip);
        } else if (namecmp(name, "..") == 0) {
            // 父目录
            struct inode *parent = dirlookup(ip, "..", 0);
            iunlockput(ip);
            ip = parent;
        } else {
            // 普通文件/目录
            struct inode *next = dirlookup(ip, name, 0);
            iunlockput(ip);
            if (next == 0)
                return 0;
            ip = next;
        }
        path = s;
    }
    return ip;
}

// 解析路径名，返回父目录的 inode 和文件名
struct inode* nameiparent(char *path, char *name) {
    // 类似 namei()，但在最后一级停止
    // ...
}
```

---

## 21.7 文件描述符层

### 21.7.1 struct file 数据结构

```c
// kernel/file.h
struct file {
    enum { FD_NONE, FD_PIPE, FD_INODE, FD_DEVICE } type;
    int ref;           // 引用计数
    char readable;     // 是否可读
    char writable;     // 是否可写
    struct pipe *pipe; // 管道（FD_PIPE）
    struct inode *ip;  // inode（FD_INODE, FD_DEVICE）
    uint off;          // 文件偏移量（FD_INODE）
    short major;       // 设备号（FD_DEVICE）
};
```

### 21.7.2 sys_open() 实现

```c
uint64 sys_open(void) {
    char path[MAXPATH];
    int fd, omode;
    struct file *f;
    struct inode *ip;
    
    if (argstr(0, path, MAXPATH) < 0 || argint(1, &omode) < 0)
        return -1;
    
    begin_op();
    
    if (omode & O_CREATE) {
        ip = create(path, T_FILE, 0, 0);
    } else {
        if ((ip = namei(path)) == 0) {
            end_op();
            return -1;
        }
        ilock(ip);
    }
    
    // 分配 file 结构
    if ((f = filealloc()) == 0 || (fd = fdalloc(f)) < 0) {
        if (f)
            fileclose(f);
        iunlockput(ip);
        end_op();
        return -1;
    }
    
    // 初始化 file 结构
    if (ip->type == T_DEVICE) {
        f->type = FD_DEVICE;
        f->major = ip->major;
    } else {
        f->type = FD_INODE;
        f->off = 0;
    }
    f->ip = ip;
    f->readable = !(omode & O_WRONLY);
    f->writable = (omode & O_WRONLY) || (omode & O_RDWR);
    
    iunlock(ip);
    end_op();
    
    return fd;
}
```

### 21.7.3 sys_read() 与 sys_write()

```c
uint64 sys_read(void) {
    struct file *f;
    int n;
    uint64 p;
    
    argaddr(1, &p);
    argint(2, &n);
    if (argfd(0, 0, &f) < 0)
        return -1;
    return fileread(f, p, n);
}

uint64 sys_write(void) {
    struct file *f;
    int n;
    uint64 p;
    
    argaddr(1, &p);
    argint(2, &n);
    if (argfd(0, 0, &f) < 0)
        return -1;
    return filewrite(f, p, n);
}
```

### 21.7.4 sys_close()

```c
uint64 sys_close(void) {
    int fd;
    struct file *f;
    
    if (argfd(0, &fd, &f) < 0)
        return -1;
    myproc()->ofile[fd] = 0;
    fileclose(f);
    return 0;
}
```

---

## 21.8 面试高频考点

**Q1：xv6 文件系统的七层架构是什么？**

从下到上：磁盘层、缓冲区缓存层、日志层、inode 层、目录层、路径名层、文件描述符层。每层向上层提供抽象接口，向下层调用底层服务。

**Q2：缓冲区缓存的作用是什么？**

三个作用：同步访问磁盘块（通过锁）、缓存常用块（减少磁盘访问）、协调块的修改（保证一致性）。

**Q3：bmap() 函数的作用是什么？**

将逻辑块号映射到物理块号。支持直接指针（12 个）、单间接指针、双间接指针。

**Q4：namei() 如何解析路径名？**

从根目录或当前目录开始，逐级查找目录项。每级调用 dirlookup() 查找文件名对应的 inode。

---

## 21.9 读取文件的完整过程

让我们追踪一个完整的文件读取过程，从用户调用 `read()` 到数据返回：

```
用户调用 read(fd, buf, n)
    │
    ▼
sys_read() → argfd() 获取 struct file
    │
    ▼
fileread(f, p, n)
    │
    ├── FD_PIPE: piperead()
    ├── FD_DEVICE: devsw[f->major].read()
    └── FD_INODE: readi(f->ip, p, f->off, n)
                    │
                    ▼
              readi() 循环读取
                    │
                    ├── bmap(ip, off/BSIZE) → 物理块号
                    │       │
                    │       ├── 直接指针：ip->addrs[bn]
                    │       └── 间接指针：读取间接块，查找指针
                    │
                    ├── bread(dev, blockno) → 缓冲区
                    │       │
                    │       ├── 缓存命中：直接返回
                    │       └── 缓存未命中：从磁盘读取
                    │
                    ├── memmove(dst, bp->data + offset, m) → 复制数据
                    │
                    └── f->off += 实际读取字节数
```

<div data-component="Xv6ReadFileFlow"></div>

---

## 21.10 创建文件的完整过程

```
用户调用 open("newfile.txt", O_CREATE | O_WRONLY)
    │
    ▼
sys_open()
    │
    ├── begin_op()  // 开始日志事务
    │
    ├── create(path, T_FILE, 0, 0)
    │       │
    │       ├── nameiparent(path, name) → 父目录 inode
    │       │
    │       ├── ialloc(dev, T_FILE) → 分配新 inode
    │       │       │
    │       │       ├── 扫描 inode 表找空闲 inode
    │       │       ├── 初始化 dinode（type=T_FILE, size=0）
    │       │       └── log_write() 写入日志
    │       │
    │       └── dirlink(dp, name, inum) → 添加目录项
    │               │
    │               ├── 查找空闲目录项
    │               └── writei() 写入目录项
    │
    ├── filealloc() → 分配 struct file
    ├── fdalloc(f) → 分配文件描述符
    │
    └── end_op()  // 提交日志事务
```

<div data-component="Xv6CreateFileFlow"></div>

---

## 21.11 删除文件的完整过程

```
用户调用 unlink("file.txt")
    │
    ▼
sys_unlink()
    │
    ├── begin_op()
    │
    ├── nameiparent(path, name) → 父目录 inode
    │
    ├── dirlookup(dp, name, &off) → 文件 inode
    │
    ├── ilock(ip)
    │
    ├── 清空目录项：memset(&de, 0, sizeof(de))
    │   writei(dp, (char*)&de, off, sizeof(de))
    │
    ├── ip->nlink--
    │
    ├── 如果 nlink == 0 且无进程打开：
    │       │
    │       ├── itrunc(ip) → 释放所有数据块
    │       │       │
    │       │       ├── 释放直接指针指向的块
    │       │       ├── 释放间接块和数据块
    │       │       └── 释放双间接块
    │       │
    │       └── ip->type = 0 → 释放 inode
    │
    ├── iunlockput(ip)
    │
    └── end_op()
```

---

## 21.12 手算练习

### 练习 1：计算 xv6 最大文件大小

**题目**：xv6 块大小 1024 字节，指针大小 4 字节，inode 有 12 个直接指针、1 个单间接、1 个双间接。计算最大文件大小。

**解答**：

```
每个块的指针数 = 1024 / 4 = 256

直接指针：12 × 1024 = 12KB
单间接：256 × 1024 = 256KB
双间接：256 × 256 × 1024 = 64MB

最大文件大小 = 12KB + 256KB + 64MB ≈ 64MB
```

### 练习 2：追踪 read() 系统调用

**题目**：进程调用 `read(fd, buf, 5000)`，文件大小 10000 字节，偏移量 0。追踪执行过程。

**解答**：

```
1. sys_read() → 获取 struct file
2. fileread() → f->type == FD_INODE
3. readi(ip, buf, 0, 5000)
4. 循环读取：
   - 块 0：bmap(ip, 0) → 物理块号 100
     bread(dev, 100) → 读取 1024 字节
     memmove(dst, bp->data, 1024)
     off = 1024, 已读 = 1024
   - 块 1：bmap(ip, 1) → 物理块号 101
     bread(dev, 101) → 读取 1024 字节
     off = 2048, 已读 = 2048
   - 块 2：bmap(ip, 2) → 物理块号 102
     bread(dev, 102) → 读取 1024 字节
     off = 3072, 已读 = 3072
   - 块 3：bmap(ip, 3) → 物理块号 103
     bread(dev, 103) → 读取 1024 字节
     off = 4096, 已读 = 4096
   - 块 4：bmap(ip, 4) → 物理块号 104
     bread(dev, 104) → 读取 904 字节（5000 - 4096）
     off = 5000, 已读 = 5000
5. 返回 5000
```

---

## 21.13 扩展阅读

- **xv6-riscv** Chapter "File system" — xv6 文件系统官方文档
- [xv6-riscv kernel/fs.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/fs.c) — 文件系统源码
- [xv6-riscv kernel/bio.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/bio.c) — 缓冲区缓存源码
- [xv6-riscv kernel/log.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/log.c) — 日志系统源码
- [xv6-riscv kernel/file.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/file.c) — 文件描述符层源码

### 21.13.1 推荐实践

1. **阅读 xv6 源码**：从 `sys_open()` 开始，追踪文件创建的完整路径
2. **使用 GDB 调试**：在 xv6 中设置断点，单步执行文件操作
3. **修改 xv6**：尝试添加新的文件系统功能（如符号链接）
4. **性能分析**：测量不同文件大小的读写性能

### 21.13.2 xv6 文件系统的性能特征

```
操作              磁盘访问次数    延迟
读取 1 字节       1-2 次          ~10-20ms
读取 4KB          1-4 次          ~10-40ms
写入 1 字节       3-5 次          ~30-50ms
创建文件          5-10 次         ~50-100ms
删除文件          3-5 次          ~30-50ms
```

**关键洞察**：
1. 写入比读取慢——因为需要更新日志、inode、位图等多个块
2. 创建文件比读取慢——因为需要分配 inode、更新目录、更新位图
3. 缓冲区缓存可以显著减少磁盘访问次数

### 21.13.3 xv6 文件系统的局限性

1. **不支持符号链接**：xv6 只支持硬链接
2. **不支持大文件**：最大文件大小约 64MB
3. **不支持长文件名**：文件名最长 14 字节
4. **简单的日志系统**：没有组提交、异步提交等优化
5. **单磁盘**：不支持 RAID 或分布式存储

### 21.13.4 xv6 文件系统的扩展阅读

- **xv6-riscv** Chapter "File system" — xv6 文件系统官方文档
- [xv6-riscv kernel/fs.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/fs.c) — 文件系统源码
- [xv6-riscv kernel/bio.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/bio.c) — 缓冲区缓存源码
- [xv6-riscv kernel/log.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/log.c) — 日志系统源码
- [xv6-riscv kernel/file.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/file.c) — 文件描述符层源码
- [MIT 6.S081 Lecture 10: File system](https://pdos.csail.mit.edu/6.S081/2020/lec/l-file.pdf) — 文件系统讲义

### 21.13.5 xv6 文件系统的面试准备

**常见面试题**：
1. 描述 xv6 文件系统的七层架构
2. 缓冲区缓存的作用是什么？
3. bmap() 函数如何工作？
4. namei() 如何解析路径名？
5. read() 系统调用的完整执行路径？

**面试技巧**：
1. 画图：画出文件系统的层次结构
2. 追踪：追踪 read()/write() 的完整执行路径
3. 代码：解释关键函数的实现（bmap, readi, namei）
4. 对比：比较 xv6 与 Linux 文件系统的异同

### 21.13.6 xv6 文件系统的总结

xv6 文件系统是一个经典的教学文件系统，它用简洁的代码展示了文件系统的核心概念和实现技术。通过学习 xv6 文件系统，可以深入理解：

1. **分层架构**：每层提供抽象接口，降低复杂度
2. **缓冲区缓存**：减少磁盘访问，提高性能
3. **inode 结构**：文件元数据的组织方式
4. **日志系统**：保证崩溃一致性
5. **路径解析**：从路径名到 inode 的映射过程

这些概念是理解所有现代文件系统的基础。

### 21.13.7 xv6 文件系统的扩展阅读推荐

- **xv6-riscv** Chapter "File system" — xv6 文件系统官方文档
- [xv6-riscv kernel/fs.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/fs.c) — 文件系统源码
- [xv6-riscv kernel/bio.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/bio.c) — 缓冲区缓存源码
- [xv6-riscv kernel/log.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/log.c) — 日志系统源码
- [xv6-riscv kernel/file.c](https://github.com/mit-pdos/xv6-riscv/blob/riscv/kernel/file.c) — 文件描述符层源码
- [MIT 6.S081 Lecture 10: File system](https://pdos.csail.mit.edu/6.S081/2020/lec/l-file.pdf) — 文件系统讲义
- [OSTEP Chapter 40: File System Implementation](https://pages.cs.wisc.edu/~remzi/OSTEP/file-implementation.pdf) — 文件系统实现

### 21.13.8 推荐实践项目

1. **为 xv6 添加符号链接**：实现 symlink() 系统调用
2. **为 xv6 添加大文件支持**：实现三间接指针
3. **优化 xv6 缓冲区缓存**：实现更高效的替换策略
4. **为 xv6 添加文件锁**：实现 advisory 文件锁
5. **性能测试**：测量不同文件大小的读写性能

### 21.13.9 xv6 文件系统的常见问题

**问题 1：缓冲区缓存未命中**

```c
// 症状：文件读取速度慢
// 原因：缓冲区缓存太小，频繁从磁盘读取
// 解决：增加缓存大小或优化访问模式
```

**问题 2：日志空间不足**

```c
// 症状：文件操作失败，panic: log_write: too big
// 原因：单个事务修改的块数超过日志空间
// 解决：将大操作拆分为多个小事务
```

**问题 3：inode 用尽**

```c
// 症状：无法创建新文件
// 原因：inode 表已满
// 解决：删除不需要的文件或增加 inode 数量
```

### 21.13.10 xv6 文件系统的总结

xv6 文件系统是一个经典的教学文件系统，它用简洁的代码展示了文件系统的核心概念和实现技术。通过学习 xv6 文件系统，可以深入理解：

1. **分层架构**：每层提供抽象接口，降低复杂度
2. **缓冲区缓存**：减少磁盘访问，提高性能
3. **inode 结构**：文件元数据的组织方式
4. **日志系统**：保证崩溃一致性
5. **路径解析**：从路径名到 inode 的映射过程

这些概念是理解所有现代文件系统的基础。
