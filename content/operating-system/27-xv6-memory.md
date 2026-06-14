---
title: "Chapter 27: xv6 内存管理"
description: "深入理解 xv6 物理内存分配器、内核页表与用户页表的布局，掌握 fork/exec 的内存管理"
updated: "2026-06-11"
---

# Chapter 27: xv6 内存管理

> **本章目标**：
> - 理解 xv6 物理内存分配器的空闲链表实现
> - 掌握内核页表的直接映射布局与初始化过程
> - 理解用户页表的地址空间布局
> - 深入分析 fork 的内存复制与写时复制优化
> - 掌握 exec 的 ELF 加载与页表切换

---

## 27.1 物理内存管理

### 27.1.1 设计概述

xv6 运行在 RISC-V 架构上，物理内存的管理是操作系统最基础的功能之一。xv6 使用一个简单的 **空闲链表（Free List）** 分配器来管理物理页面，每次分配和释放的单位是 4096 字节（一个页面）。

物理内存的布局如下：

```
物理地址空间 (以 128MB RAM 为例)
┌──────────────────────┐ 0x80000000 (PHYSTOP = 0x88000000)
│                      │
│   可用物理页面        │
│   (空闲链表管理)      │
│                      │
├──────────────────────┤ 内核代码结束 (由 linker 脚本确定)
│   内核代码 + 数据     │
├──────────────────────┤ 0x80000000 (KERNBASE)
│   CLINT (中断控制器)  │
├──────────────────────┤ 0x02000000
│   PLIC (中断控制器)   │
├──────────────────────┤ 0x0C000000
│   VIRTIO (磁盘)      │
├──────────────────────┤ 0x10000000
│   UART (串口)        │
├──────────────────────┤ 0x10000000
│   ...                │
└──────────────────────┘ 0x00000000
```

xv6 的物理内存分配器只做一件事：将空闲页面串成一个链表，`kalloc` 从链表头部取出一个页面，`kfree` 将页面归还到链表头部。

### 27.1.2 数据结构

物理页面通过 `struct run` 组织成链表。每个空闲页面的起始地址处存储一个指向下一个空闲页面的指针：

```c
// kernel/kalloc.c

struct run {
  struct run *next;
};

struct {
  struct spinlock lock;
  struct run *freelist;
} kmem;
```

关键设计点：

1. **零额外空间**：空闲链表不需要额外的元数据存储，链表指针直接存储在空闲页面内部
2. **每个节点恰好是一个页面**：节点大小 = 4096 字节 = `PGSIZE`
3. **自引用**：一个空闲页面的前 8 字节（在 64 位系统上）存放指向下一个空闲页面的指针
4. **互斥保护**：`kmem.lock` 保护 freelist 的并发访问

```
空闲链表结构：

kmem.freelist
    │
    ▼
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌───────
│  next ─────┼───>│  next ─────┼───>│  next ─────┼───>│  NULL
│  (剩余空间) │    │  (剩余空间) │    │  (剩余空间) │    │
└────────────┘    └────────────┘    └────────────┘    └───────
  物理页面 P1        物理页面 P2        物理页面 P3

每个页面的前 8 字节存储 next 指针
页面大小 = 4096 字节
```

### 27.1.3 kinit：初始化物理内存分配器

系统启动时，`kinit` 函数负责将所有可用的物理页面加入空闲链表：

```c
// kernel/kalloc.c

void kinit()
{
  initlock(&kmem.lock, "kmem");
  freerange(end, (void*)PHYSTOP);
}

void freerange(void *pa_start, void *pa_end)
{
  char *p;
  p = (char*)PGROUNDUP((uint64)pa_start);
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
    kfree(p);
}
```

初始化过程：

1. **初始化自旋锁**：`initlock` 初始化 `kmem.lock`
2. **计算起始地址**：`end` 是内核代码的结束地址（由 linker 脚本 `kernel.ld` 定义），`PGROUNDUP` 将其向上对齐到页面边界
3. **遍历所有页面**：从 `end` 到 `PHYSTOP`（物理内存上限），每个页面调用 `kfree` 加入空闲链表

```c
// kernel/kernel.ld (linker 脚本片段)
PROVIDE(end = .);  // end 标记内核代码的结束位置
```

```
初始化过程示意：

end (内核结束)        PHYSTOP (0x88000000)
  │                        │
  ▼                        ▼
  ┌────┬────┬────┬────┬────┬────┬────┐
  │kern│free│free│free│free│free│free│
  └────┴────┴────┴────┴────┴────┴────┘
       │    ▲    │    │    │    │    │
       │    └────┘    │    │    │    │  kfree 逐页加入
       └─────────────┘    │    │    │
                          │    │    │
                           ... ...

最终状态：
kmem.freelist → page1 → page2 → page3 → ... → NULL
```

### 27.1.4 kfree：释放物理页面

`kfree` 将一个物理页面归还到空闲链表头部：

```c
// kernel/kalloc.c

void kfree(void *pa)
{
  struct run *r;

  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP)
    panic("kfree");

  // 用垃圾数据填充页面，帮助检测 use-after-free
  memset(pa, 1, PGSIZE);

  r = (struct run*)pa;

  acquire(&kmem.lock);
  r->next = kmem.freelist;
  kmem.freelist = r;
  release(&kmem.lock);
}
```

执行步骤：

1. **合法性检查**：验证地址页面对齐、在有效范围内
2. **填充垃圾数据**：`memset(pa, 1, PGSIZE)` 将页面全部填为 1，这是为了帮助发现 use-after-free 错误——如果释放后仍有代码读取该页面，会得到错误的值 0x01010101...
3. **插入链表头部**：将页面地址强制转换为 `struct run*`，然后插入链表头部（O(1) 操作）

```c
// 释放前：页面包含旧数据
// pa → [旧数据......................]

// 释放后：前 8 字节存储 next 指针
// pa → [next → 原链表头] [0x010101...]
//       ↑ struct run      ↑ memset 填充
```

### 27.1.5 kalloc：分配物理页面

`kalloc` 从空闲链表头部取出一个页面：

```c
// kernel/kalloc.c

void* kalloc(void)
{
  struct run *r;

  acquire(&kmem.lock);
  r = kmem.freelist;
  if(r)
    kmem.freelist = r->next;
  release(&kmem.lock);

  if(r)
    memset((char*)r, 5, PGSIZE); // 填充垃圾数据
  return (void*)r;
}
```

执行步骤：

1. **获取锁**：`acquire(&kmem.lock)` 保护 freelist
2. **取出头部**：`r = kmem.freelist` 取出链表头部
3. **更新链表**：`kmem.freelist = r->next` 将链表头指向下一个节点
4. **释放锁**：`release(&kmem.lock)`
5. **填充垃圾数据**：`memset((char*)r, 5, PGSIZE)` 填充为 0x05，帮助发现未初始化页面的读取
6. **返回页面地址**：返回 `void*` 类型的物理地址

```
kalloc 执行过程：

执行前：
kmem.freelist → [P1] → [P2] → [P3] → NULL

执行后：
kmem.freelist ─────→ [P2] → [P3] → NULL
                    (P1 已被取出)
返回值：P1 的物理地址
```

### 27.1.6 并发安全：kmem.lock

物理内存分配器可能被多个 CPU 核心同时调用（例如多个进程同时执行 `fork` 需要分配页面），因此需要自旋锁保护：

```c
struct {
  struct spinlock lock;   // 自旋锁保护 freelist
  struct run *freelist;   // 全局空闲链表
} kmem;
```

锁的粒度分析：

| 操作 | 持锁时间 | 说明 |
|------|---------|------|
| `kalloc` | 很短 | 仅读取和更新链表头 |
| `kfree` | 很短 | 仅修改链表头 |
| `freerange` | 不持锁 | 循环调用 `kfree`，每次单独持锁 |

```
多核并发场景：

CPU 0: kalloc()                CPU 1: kalloc()
  │                              │
  ▼                              ▼
acquire(lock) ──等待──┐    acquire(lock)
  │                   │      │
  r = freelist        │      ▼  (等待锁释放)
  freelist = r->next  │      ...
  release(lock) ──────┘      │
  │                          ▼
  │                    acquire(lock) ← 获取锁
  ▼                      │
使用页面 P1              r = freelist
                         freelist = r->next
                         release(lock)
                         │
                         ▼
                       使用页面 P2
```

> **设计思考**：xv6 使用单一的全局锁保护整个空闲链表，这在多核系统上会成为性能瓶颈。更高效的实现可以为每个 CPU 核心维护独立的空闲链表，减少锁竞争。

---

## 27.2 内核页表

### 27.2.1 直接映射（Direct Mapping）

xv6 的内核页表采用 **直接映射**（Direct Mapping），也称为恒等映射：内核虚拟地址等于物理地址。这意味着对于内核访问的大部分地址，虚拟地址到物理地址的转换是透明的。

```c
// kernel/vm.c

// 创建内核页表
pagetable_t kernel_pagetable;

void kvminit()
{
  kernel_pagetable = kvmmake();
}
```

```
直接映射示意：

虚拟地址                物理地址
0x80000000  ───────────>  0x80000000  (KERNBASE)
0x80001000  ───────────>  0x80001000
0x80002000  ───────────>  0x80002000
...              1:1      ...
0x88000000  ───────────>  0x88000000  (PHYSTOP)

内核通过虚拟地址访问物理内存时，
地址值不变，无需地址转换
```

### 27.2.2 内核地址空间布局

xv6 的内核地址空间布局如下：

```
内核虚拟地址空间 (64 位，仅使用低 39 位)
┌────────────────────────────────────┐ MAXVA (2^39)
│                                    │
│  (未使用区域)                       │
│                                    │
├────────────────────────────────────┤ 0xFFFFFFFF80000000 + PHYSTOP
│  Trampoline                        │ (映射到物理地址 trampoline)
│  (页面跳板，用于内核/用户态切换)     │
├────────────────────────────────────┤ 0xFFFFFFFF80000000 + 内核代码结束
│  内核栈 (每个 CPU 一个)             │
│  2 个页面，带 guard page           │
├────────────────────────────────────┤
│  物理内存 (直接映射)                │
│  [KERNBASE, KERNBASE + PHYSTOP)    │
│  KERNBASE = 0x80000000             │
│  PHYSTOP = KERNBASE + 128MB        │
├────────────────────────────────────┤ 0x80000000
│  PLIC (Platform Level Interrupt)   │
│  0x0C000000 - 0x0C002000           │
├────────────────────────────────────┤ 0x0C000000
│  VIRTIO (虚拟磁盘)                 │
│  0x10000000 - 0x10000100           │
├────────────────────────────────────┤ 0x10000000
│  UART0 (串口)                      │
│  0x10000000 - 0x10000008           │
├────────────────────────────────────┤ 0x10000000
│  CLINT (Core Local Interrupt)      │
│  0x02000000 - 0x02010000           │
├────────────────────────────────────┤ 0x02000000
│  ...                               │
└────────────────────────────────────┘ 0x00000000
```

### 27.2.3 kvmmake：构建内核页表

```c
// kernel/vm.c

pagetable_t kvmmake(void)
{
  pagetable_t kpgtbl;

  kpgtbl = (pagetable_t) kalloc();
  memset(kpgtbl, 0, PGSIZE);

  // UART 寄存器
  kvmmap(kpgtbl, UART0, UART0, PGSIZE, PTE_R | PTE_W);

  // VIRTIO 磁盘
  kvmmap(kpgtbl, VIRTIO0, VIRTIO0, PGSIZE, PTE_R | PTE_W);

  // PLIC 中断控制器
  kvmmap(kpgtbl, PLIC, PLIC, 0x400000, PTE_R | PTE_W);

  // 内核代码和数据（直接映射）
  kvmmap(kpgtbl, KERNBASE, KERNBASE,
         (uint64)etext - KERNBASE, PTE_R | PTE_X);

  // 物理内存（直接映射，内核代码之后的部分）
  kvmmap(kpgtbl, (uint64)etext, (uint64)etext,
         PHYSTOP - (uint64)etext, PTE_R | PTE_W);

  // trampoline 页面（用户态/内核态切换）
  kvmmap(kpgtbl, TRAMPOLINE, (uint64)trampoline,
         PGSIZE, PTE_R | PTE_X);

  // 每个 CPU 的内核栈
  proc_mapstacks(kpgtbl);

  return kpgtbl;
}
```

### 27.2.4 kvmmap 与 walk 实现

`kvmmap` 在内核页表中建立一个映射：

```c
// kernel/vm.c

void kvmmap(pagetable_t kpgtbl, uint64 va, uint64 pa, uint64 sz, int perm)
{
  if(mappages(kpgtbl, va, sz, pa, perm) != 0)
    panic("kvmmap");
}
```

`mappages` 逐页面建立映射：

```c
// kernel/vm.c

int mappages(pagetable_t pagetable, uint64 va, uint64 size, uint64 pa, int perm)
{
  uint64 a, last;
  pte_t *pte;

  a = PGROUNDDOWN(va);
  last = PGROUNDDOWN(va + size - 1);
  for(;;){
    if((pte = walk(pagetable, a, 1)) == 0)
      return -1;
    if(*pte & PTE_V)
      panic("mappages: remap");
    *pte = PA2PTE(pa) | perm | PTE_V;
    a += PGSIZE;
    pa += PGSIZE;
    if(a == last)
      break;
  }
  return 0;
}
```

`walk` 是页表遍历的核心函数，在三级页表中查找或创建 PTE：

```c
// kernel/vm.c

pte_t* walk(pagetable_t pagetable, uint64 va, int alloc)
{
  if(va >= MAXVA)
    panic("walk");

  for(int level = 2; level > 0; level--){
    pte_t *pte = &pagetable[PX(level, va)];
    if(*pte & PTE_V){
      pagetable = (pagetable_t)PTE2PA(*pte);
    } else {
      if(!alloc || (pagetable = (pde_t*)kalloc()) == 0)
        return 0;
      memset(pagetable, 0, PGSIZE);
      *pte = PA2PTE(pagetable) | PTE_V;
    }
  }
  return &pagetable[PX(0, va)];
}
```

```
walk 函数遍历三级页表：

虚拟地址 va = [L2 索引 | L1 索引 | L0 索引 | 页内偏移]
                9 位      9 位      9 位      12 位

Level 2 (根页表):
  pagetable[PX(2, va)] → 获取 L2 PTE
  ┌─────────────────────────┐
  │ PTE_V set?              │
  │ YES: 取出下一级页表地址   │─────┐
  │ NO:  分配新页表 (alloc=1)│     │
  └─────────────────────────┘     │
                                  ▼
Level 1 (中间页表):
  pagetable[PX(1, va)] → 获取 L1 PTE
  ┌─────────────────────────┐
  │ PTE_V set?              │
  │ YES: 取出下一级页表地址   │─────┐
  │ NO:  分配新页表 (alloc=1)│     │
  └─────────────────────────┘     │
                                  ▼
Level 0 (叶页表):
  返回 &pagetable[PX(0, va)] → 指向最终的 PTE
```

### 27.2.5 kvminithart：激活内核页表

```c
// kernel/vm.c

void kvminithart()
{
  w_satp(MAKE_SATP(kernel_pagetable));
  sfence_vma();
}
```

`w_satp` 将内核页表的根地址写入 `satp` 寄存器，`sfence_vma` 刷新 TLB。

> **面试考点**：为什么内核使用直接映射？
> - 简化实现：内核可以直接使用物理地址，无需复杂的地址转换
> - 性能：减少页表遍历开销
> - 但直接映射也限制了内核不能轻松处理物理内存不连续的情况

---

## 27.3 用户页表

### 27.3.1 每进程独立页表

每个 xv6 进程都有自己的独立页表，实现了地址空间隔离：

```c
// kernel/proc.h

struct proc {
  ...
  pagetable_t pagetable;     // 用户页表
  ...
};
```

用户页表与内核页表的关键区别：

| 特性 | 内核页表 | 用户页表 |
|------|---------|---------|
| 数量 | 全局唯一 | 每进程一个 |
| 内核映射 | 完整映射 | 仅 trampoline + trapframe |
| 地址空间 | 内核虚拟地址 | 用户虚拟地址 |
| 创建时机 | 系统启动 | 进程创建时 |

### 27.3.2 用户地址空间布局

```
用户虚拟地址空间 (MAXVA = 2^39 = 512GB)
┌────────────────────────────────────┐ MAXVA - PGSIZE
│  Trampoline                        │ (用于 trap 处理)
│  (映射到物理 trampoline 页面)       │ 权限: R-X
├────────────────────────────────────┤ TRAPFRAME (0x3fffffffe000)
│  Trapframe                         │ (保存用户寄存器)
│  (每个进程独立的物理页面)           │ 权限: RW-
├────────────────────────────────────┤
│                                    │
│  (未使用的高地址空间)               │
│                                    │
├────────────────────────────────────┤ 用户栈顶
│  Guard Page (不可访问)             │
├────────────────────────────────────┤
│  用户栈                            │
│  (向下增长)                        │ 权限: RW-
├────────────────────────────────────┤
│  ...                               │
│  堆 (向上增长)                     │ 权限: RW-
│  ...                               │
├────────────────────────────────────┤ p->sz
│  (未使用的地址空间)                 │
├────────────────────────────────────┤
│  数据段 (.data, .bss)             │ 权限: RW-
├────────────────────────────────────┤
│  代码段 (.text)                    │ 权限: R-X
└────────────────────────────────────┘ 0x00000000
```

### 27.3.3 proc_pagetable：创建用户页表

```c
// kernel/vm.c

pagetable_t proc_pagetable(struct proc *p)
{
  pagetable_t pagetable;

  // 创建一个空的页表
  pagetable = (pagetable_t) kalloc();
  if(pagetable == 0)
    return 0;
  memset(pagetable, 0, PGSIZE);

  // 映射 trampoline 代码（虚拟地址最高处）
  if(mappages(pagetable, TRAMPOLINE, PGSIZE,
              (uint64)trampoline, PTE_R | PTE_X) < 0){
    kfree((void*)pagetable);
    return 0;
  }

  // 映射 trapframe（紧接 trampoline 下方）
  if(mappages(pagetable, TRAPFRAME, PGSIZE,
              (uint64)(p->trapframe), PTE_R | PTE_W) < 0){
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    kfree((void*)pagetable);
    return 0;
  }

  return pagetable;
}
```

### 27.3.4 uvmalloc：分配用户内存

`uvmalloc` 在用户页表中分配新的物理页面：

```c
// kernel/vm.c

uint64 uvmalloc(pagetable_t pagetable, uint64 oldsz, uint64 newsz, int xperm)
{
  char *mem;
  uint64 a;

  if(newsz < oldsz)
    return oldsz;

  oldsz = PGROUNDUP(oldsz);
  for(a = oldsz; a < newsz; a += PGSIZE){
    mem = kalloc();
    if(mem == 0){
      uvmdealloc(pagetable, a, oldsz);
      return 0;
    }
    memset(mem, 0, PGSIZE);
    if(mappages(pagetable, a, PGSIZE, (uint64)mem, PTE_R|PTE_U|xperm) != 0){
      kfree(mem);
      uvmdealloc(pagetable, a, oldsz);
      return 0;
    }
  }
  return newsz;
}
```

执行流程：

```
uvmalloc(pagetable, 0x10000, 0x13000, 0)

oldsz = PGROUNDUP(0x10000) = 0x10000

迭代 1: a = 0x10000
  mem = kalloc() → 分配物理页面 PA1
  memset(mem, 0, PGSIZE)
  mappages(0x10000 → PA1, PTE_R|PTE_U)

迭代 2: a = 0x11000
  mem = kalloc() → 分配物理页面 PA2
  memset(mem, 0, PGSIZE)
  mappages(0x11000 → PA2, PTE_R|PTE_U)

迭代 3: a = 0x12000
  mem = kalloc() → 分配物理页面 PA3
  memset(mem, 0, PGSIZE)
  mappages(0x12000 → PA3, PTE_R|PTE_U)

返回 0x13000
```

### 27.3.5 uvmdealloc：释放用户内存

```c
// kernel/vm.c

uint64 uvmdealloc(pagetable_t pagetable, uint64 oldsz, uint64 newsz)
{
  if(newsz >= oldsz)
    return oldsz;

  if(PGROUNDUP(newsz) < PGROUNDUP(oldsz)){
    int npages = (PGROUNDUP(oldsz) - PGROUNDUP(newsz)) / PGSIZE;
    uvmunmap(pagetable, PGROUNDUP(newsz), npages, 1);
  }

  return newsz;
}
```

`uvmunmap` 解除映射并可选地释放物理页面：

```c
// kernel/vm.c

void uvmunmap(pagetable_t pagetable, uint64 va, uint64 npages, int do_free)
{
  uint64 a;
  pte_t *pte;

  if((va % PGSIZE) != 0)
    panic("uvmunmap: not aligned");

  for(a = va; a < va + npages * PGSIZE; a += PGSIZE){
    if((pte = walk(pagetable, a, 0)) == 0)
      panic("uvmunmap: walk");
    if((*pte & PTE_V) == 0)
      panic("uvmunmap: not mapped");
    if(PTE_FLAGS(*pte) == PTE_V)
      panic("uvmunmap: not a leaf");
    if(do_free){
      uint64 pa = PTE2PA(*pte);
      kfree((void*)pa);
    }
    *pte = 0;
  }
}
```

### 27.3.6 copyout 与 copyin

内核需要在用户地址空间和内核地址空间之间复制数据。由于用户页表和内核页表不同，不能直接使用 `memcpy`：

```c
// kernel/vm.c

// 从用户空间复制到内核空间
int copyin(pagetable_t pagetable, char *dst, uint64 srcva, uint64 len)
{
  uint64 n, va0, pa0;

  while(len > 0){
    va0 = PGROUNDDOWN(srcva);
    pa0 = walkaddr(pagetable, va0);
    if(pa0 == 0)
      return -1;
    n = PGSIZE - (srcva - va0);
    if(n > len)
      n = len;
    memmove(dst, (void *)(pa0 + (srcva - va0)), n);

    len -= n;
    dst += n;
    srcva = va0 + PGSIZE;
  }
  return 0;
}

// 从内核空间复制到用户空间
int copyout(pagetable_t pagetable, uint64 dstva, char *src, uint64 len)
{
  uint64 n, va0, pa0;

  while(len > 0){
    va0 = PGROUNDDOWN(dstva);
    pa0 = walkaddr(pagetable, va0);
    if(pa0 == 0)
      return -1;
    n = PGSIZE - (dstva - va0);
    if(n > len)
      n = len;
    memmove((void *)(pa0 + (dstva - va0)), src, n);

    len -= n;
    src += n;
    dstva = va0 + PGSIZE;
  }
  return 0;
}
```

```
copyout 工作原理：

用户页表                      内核页表
┌──────────┐                 ┌──────────┐
│ VA 0x1000│──┐              │          │
│          │  │  物理页面     │          │
│          │  └─>┌────────┐  │          │
│          │     │  PA    │<─┼── src    │
│          │     └────────┘  │          │
└──────────┘                 └──────────┘

copyout 通过 walkaddr 找到用户虚拟地址对应的物理地址，
然后用 memmove 在物理地址层面复制数据
```

---

## 27.4 fork 与写时复制

### 27.4.1 fork 的内存管理

当 xv6 执行 `fork` 系统调用时，需要为子进程创建完整的父进程副本，包括所有用户内存。核心步骤：

1. **分配新进程**：`allocproc` 分配一个新的进程结构体
2. **复制用户内存**：`uvmcopy` 复制父进程的整个用户地址空间
3. **复制寄存器状态**：将父进程的 trapframe 复制到子进程
4. **设置返回值**：父进程返回子进程 PID，子进程返回 0

```c
// kernel/proc.c

int fork()
{
  int i, pid;
  struct proc *np;
  struct proc *p = myproc();

  // 分配新进程
  if((np = allocproc()) == 0){
    return -1;
  }

  // 复制用户内存
  if(uvmcopy(p->pagetable, np->pagetable, p->sz) < 0){
    freeproc(np);
    release(&np->lock);
    return -1;
  }
  np->sz = p->sz;

  // 复制 trapframe（包含用户寄存器）
  *(np->trapframe) = *(p->trapframe);

  // 子进程 fork 返回值设为 0
  np->trapframe->a0 = 0;

  // 复制文件描述符等其他状态
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

### 27.4.2 uvmcopy：复制页表

`uvmcopy` 是 fork 的核心，负责复制父进程的整个用户地址空间：

```c
// kernel/vm.c

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
    memmove(mem, (char*)pa, PGSIZE);
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

```
uvmcopy 过程：

父进程页表                    子进程页表
┌──────────┐                 ┌──────────┐
│ VA 0x0000│──┐              │ VA 0x0000│──┐
│          │  │              │          │  │
│          │  └─>┌────────┐  │          │  └─>┌────────┐
│          │     │ PA_old │  │          │     │ PA_new │ ← 新分配
│          │     │[数据...]│  │          │     │[数据...]│ ← memmove 复制
│          │     └────────┘  │          │     └────────┘
│ VA 0x1000│──┐              │ VA 0x1000│──┐
│          │  │              │          │  │
│          │  └─>┌────────┐  │          │  └─>┌────────┐
│          │     │ PA_old │  │          │     │ PA_new │ ← 新分配
│          │     │[数据...]│  │          │     │[数据...]│ ← memmove 复制
│          │     └────────┘  │          │     └────────┘
└──────────┘                 └──────────┘

每个页面都分配新的物理页面并复制内容
```

### 27.4.3 fork 的性能问题

原始 fork 的主要问题是 **内存复制开销**：

```
时间复杂度分析：

假设进程使用 N 个页面
  - 分配 N 个新物理页面：N 次 kalloc
  - 复制 N 个页面的内容：N × 4096 字节 memmove
  - 建立 N 个页表映射：N 次 mappages

总开销 = O(N × PGSIZE)
```

对于大进程（例如使用 100MB 内存），fork 需要复制 25600 个页面，约 100MB 数据复制，这非常耗时。

### 27.4.4 写时复制（Copy-on-Write, COW）优化

写时复制（COW）是一种延迟复制技术，核心思想：

> **fork 时只复制页表，不复制物理页面。所有页面标记为只读。当任一进程尝试写入时，触发页故障（Page Fault），此时才真正复制该页面。**

#### COW 的三个关键机制

**1. 标记只读**

fork 时，父子进程的页表项都标记为只读（清除 PTE_W 位）：

```
fork 后（COW 模式）：

父进程页表                    子进程页表
┌──────────┐                 ┌──────────┐
│ VA 0x0000│──┐              │ VA 0x0000│──┐
│ flags: R-│  │              │ flags: R-│  │
│          │  └─>┌────────┐  │          │  └─┐
│          │     │ PA_0   │<─┼──────────────┘  │ 共享同一物理页面
│          │     │[数据...]│  │          │      │
│          │     └────────┘  │          │      │
└──────────┘                 └──────────┘      │
                                               │
                              物理页面 PA_0 被父子进程共享
```

**2. 页故障复制**

当进程尝试写入只读页面时，CPU 触发页故障。内核的页故障处理程序分配新页面，复制内容，更新页表：

```c
// COW 页故障处理伪代码

void cow_page_fault(uint64 va) {
  pte_t *pte = walk(myproc()->pagetable, va, 0);
  
  // 检查是否是 COW 页面
  if(!is_cow_page(pte))
    panic("not a cow page");
  
  // 获取旧物理地址
  uint64 old_pa = PTE2PA(*pte);
  
  // 分配新物理页面
  uint64 new_pa = (uint64)kalloc();
  if(new_pa == 0) {
    // 内存不足，杀死进程
    kill(myproc()->pid);
    return;
  }
  
  // 复制页面内容
  memmove((void*)new_pa, (void*)old_pa, PGSIZE);
  
  // 更新页表：映射到新页面，恢复写权限
  *pte = PA2PTE(new_pa) | PTE_FLAGS(*pte) | PTE_W;
  
  // 减少旧页面的引用计数
  // 如果引用计数变为 0，释放旧页面
  decrement_ref(old_pa);
}
```

```
COW 页故障处理过程：

1. 父进程尝试写入 VA 0x1000
   ┌──────────┐
   │ VA 0x1000│──┐
   │ flags: R-│  │
   └──────────┘  └─>┌────────┐
                     │ PA_0   │  ← 只读，触发页故障
                     │[数据...]│
                     └────────┘

2. 内核处理页故障
   - 分配新物理页面 PA_new
   - 复制 PA_0 → PA_new
   - 更新父进程页表：
     VA 0x1000 → PA_new (flags: RW)

3. 父进程重新执行写操作
   ┌──────────┐
   │ VA 0x1000│──┐
   │ flags: RW│  │
   └──────────┘  └─>┌────────┐
                     │ PA_new │  ← 可写
                     │[数据...]│
                     └────────┘
   
   子进程页表仍指向 PA_0（只读）
```

**3. 引用计数**

需要为每个物理页面维护引用计数，以确定何时可以安全释放页面：

```c
// 引用计数数据结构

struct {
  struct spinlock lock;
  int refcount[PHYSTOP / PGSIZE];  // 每个页面一个引用计数
} kref;

void kfree_cow(uint64 pa) {
  acquire(&kref.lock);
  if(kref.refcount[pa / PGSIZE] > 1) {
    // 还有其他进程引用，只减少计数
    kref.refcount[pa / PGSIZE]--;
    release(&kref.lock);
    return;
  }
  release(&kref.lock);
  
  // 引用计数为 0，真正释放
  kfree((void*)pa);
}
```

```
引用计数管理：

初始状态（fork 前）：
PA_0: refcount = 1 (父进程独占)

fork 后（COW）：
PA_0: refcount = 2 (父子共享)

父进程写入，触发 COW：
PA_0: refcount = 1 (仅子进程)
PA_new: refcount = 1 (父进程)

子进程写入，触发 COW：
PA_0: refcount = 0 → 释放
PA_new2: refcount = 1 (子进程)
```

### 27.4.5 COW 的性能优势

| 指标 | 原始 fork | COW fork |
|------|----------|----------|
| fork 时间 | O(N × PGSIZE) | O(N) — 仅复制页表 |
| 内存使用 | 立即翻倍 | 共享直到写入 |
| 写入延迟 | 无额外开销 | 页故障 + 页面复制 |
| 适用场景 | 小进程 | 大进程，exec 前 fork |

> **面试高频题**：fork 后立即 exec 为什么适合 COW？
> 
> 答：fork 后子进程通常立即执行 exec，替换整个地址空间。如果用原始 fork，复制的所有页面都会被 exec 丢弃，造成巨大浪费。COW fork 只复制页表，exec 时直接释放，无需复制物理页面。

---

## 27.5 exec 内存管理

### 27.5.1 exec 系统调用概述

`exec` 系统调用用新的程序替换当前进程的地址空间。它执行以下步骤：

1. 读取并验证 ELF 文件头
2. 创建新的页表
3. 加载 ELF 程序的各个段（代码段、数据段）
4. 分配用户栈
5. 设置参数 argc/argv
6. 切换到新页表

```c
// kernel/exec.c

int exec(char *path, char **argv)
{
  char *s, *last;
  int i, off;
  uint64 argc, sz = 0, sp, ustack[MAXARG+1];
  struct elfhdr elf;
  struct proghdr ph;
  pagetable_t pagetable = 0, oldpagetable;
  struct proc *p = myproc();

  begin_op();

  if((ip = namei(path)) == 0){
    end_op();
    return -1;
  }
  ilock(ip);

  // 1. 读取 ELF 头
  if(readi(ip, 0, (uint64)&elf, 0, sizeof(elf)) != sizeof(elf))
    goto bad;
  if(elf.magic != ELF_MAGIC)
    goto bad;

  // 2. 创建新页表
  if((pagetable = proc_pagetable(p)) == 0)
    goto bad;

  // 3. 加载每个程序段
  for(i = 0, off = elf.phoff; i < elf.phnum; i++, off += sizeof(ph)){
    if(readi(ip, 0, (uint64)&ph, off, sizeof(ph)) != sizeof(ph))
      goto bad;
    if(ph.type != ELF_PROG_LOAD)
      continue;
    if(ph.memsz < ph.filesz)
      goto bad;
    if(ph.vaddr + ph.memsz < ph.vaddr)
      goto bad;
    if(ph.vaddr % PGSIZE != 0)
      goto bad;

    // 分配内存并加载段
    sz = uvmalloc(pagetable, sz, ph.vaddr + ph.memsz, PTE_W);
    if(sz == 0)
      goto bad;
    if(loadseg(pagetable, ph.vaddr, ip, ph.off, ph.filesz) < 0)
      goto bad;
  }
  iunlockput(ip);
  end_op();
  ip = 0;

  p = myproc();
  uint64 oldsz = p->sz;

  // 4. 分配用户栈 (2 个页面)
  sz = PGROUNDUP(sz);
  uint64 sz1;
  if((sz1 = uvmalloc(pagetable, sz, sz + 2*PGSIZE, PTE_W)) == 0)
    goto bad;
  sz = sz1;
  uvmclear(pagetable, sz - 2*PGSIZE);
  sp = sz;
  ustack[0] = 0; // argv[0] 的终止符

  // 5. 将参数复制到用户栈
  for(argc = 0; argv[argc]; argc++){
    if(argc >= MAXARG)
      goto bad;
    sp -= strlen(argv[argc]) + 1;
    sp -= sp % 16; // RISC-V 对齐
    if(sp < sz - 2*PGSIZE)
      goto bad;
    if(copyout(pagetable, sp, argv[argc], strlen(argv[argc]) + 1) < 0)
      goto bad;
    ustack[argc] = sp;
  }
  ustack[argc] = 0;

  // 将 argv 指针数组压入栈
  sp -= (argc+1) * sizeof(uint64);
  sp -= sp % 16;
  if(sp < sz - 2*PGSIZE)
    goto bad;
  if(copyout(pagetable, sp, (char*)ustack, (argc+1)*sizeof(uint64)) < 0)
    goto bad;

  // 设置寄存器
  p->trapframe->a1 = sp;  // argv
  p->trapframe->a0 = argc; // argc

  // 保存程序名称
  for(last=s=path; *s; s++)
    if(*s == '/')
      last = s+1;
  safestrcpy(p->name, last, sizeof(p->name));

  // 6. 切换页表
  oldpagetable = p->pagetable;
  p->pagetable = pagetable;
  p->sz = sz;
  p->trapframe->epc = elf.entry;  // 设置 PC 为 ELF 入口地址
  p->trapframe->sp = sp;          // 设置栈指针
  proc_freepagetable(oldpagetable, oldsz);

  return argc;

 bad:
  if(pagetable)
    proc_freepagetable(pagetable, sz);
  if(ip){
    iunlockput(ip);
    end_op();
  }
  return -1;
}
```

### 27.5.2 ELF 文件格式

ELF（Executable and Linkable Format）是 Unix 系统的标准可执行文件格式：

```
ELF 文件结构：
┌────────────────────┐
│ ELF Header         │ ← 魔数、入口地址、程序头偏移
│ (struct elfhdr)    │
├────────────────────┤
│ Program Header 0   │ ← 描述一个段：类型、偏移、虚拟地址、大小
│ Program Header 1   │
│ ...                │
│ Program Header N   │
├────────────────────┤
│ .text (代码段)     │ ← 可执行代码
├────────────────────┤
│ .rodata (只读数据) │ ← 常量字符串等
├────────────────────┤
│ .data (数据段)     │ ← 已初始化全局变量
├────────────────────┤
│ .bss               │ ← 未初始化全局变量（零填充）
├────────────────────┤
│ .symtab            │ ← 符号表（调试用）
│ .strtab            │ ← 字符串表
│ ...                │
└────────────────────┘
```

```c
// kernel/elf.h

struct elfhdr {
  uint magic;       // ELF_MAGIC = 0x464C457F ("\x7FELF")
  char elf[12];     // ELF 标识
  ushort type;      // 文件类型 (ET_EXEC = 2)
  ushort machine;   // 目标架构 (RISC-V = 243)
  uint version;     // ELF 版本
  uint64 entry;     // 程序入口地址 (main 函数的地址)
  uint64 phoff;     // 程序头表偏移
  uint64 shoff;     // 节头表偏移
  uint flags;       // 处理器特定标志
  ushort ehsize;    // ELF 头大小
  ushort phentsize; // 程序头条目大小
  ushort phnum;     // 程序头条目数量
  ushort shentsize; // 节头条目大小
  ushort shnum;     // 节头条目数量
  ushort shstrndx;  // 节名字符串表索引
};

struct proghdr {
  uint32 type;    // 段类型 (PT_LOAD = 1)
  uint32 flags;   // 段标志 (PF_R, PF_W, PF_X)
  uint64 off;     // 段在文件中的偏移
  uint64 vaddr;   // 段的虚拟地址
  uint64 paddr;   // 段的物理地址（xv6 未使用）
  uint64 filesz;  // 段在文件中的大小
  uint64 memsz;   // 段在内存中的大小（>= filesz）
  uint64 align;   // 对齐要求
};
```

### 27.5.3 loadseg：加载程序段

```c
// kernel/exec.c

static int loadseg(pagetable_t pagetable, uint64 va, struct inode *ip, uint offset, uint sz)
{
  uint i, n;
  uint64 pa;

  for(i = 0; i < sz; i += PGSIZE){
    pa = walkaddr(pagetable, va + i);
    if(pa == 0)
      panic("loadseg: address should exist");
    if(sz - i < PGSIZE)
      n = sz - i;
    else
      n = PGSIZE;
    if(readi(ip, 0, pa + (va+i)%PGSIZE, offset+i, n) != n)
      return -1;
  }
  return 0;
}
```

### 27.5.4 构建用户栈

exec 在用户栈上构建 argc 和 argv：

```
exec("echo", ["echo", "hello", "world"])

用户栈布局（从高地址向低地址生长）：

高地址
┌──────────────────────────┐ sp (栈顶)
│ argv[2] = "world\0"      │
├──────────────────────────┤
│ argv[1] = "hello\0"      │
├──────────────────────────┤
│ argv[0] = "echo\0"       │
├──────────────────────────┤
│ 16 字节对齐填充           │
├──────────────────────────┤
│ ustack[3] = 0 (NULL)     │ ← argv 终止符
│ ustack[2] = ptr to "world│" ← argv[2] 指针
│ ustack[1] = ptr to "hello│" ← argv[1] 指针
│ ustack[0] = ptr to "echo"│ ← argv[0] 指针
├──────────────────────────┤
│ Guard Page (不可访问)     │
├──────────────────────────┤
│ 未使用空间               │
└──────────────────────────┘ 低地址
```

### 27.5.5 切换页表

exec 的最后一步是切换到新页表：

```c
// 切换页表
oldpagetable = p->pagetable;
p->pagetable = pagetable;
p->sz = sz;
p->trapframe->epc = elf.entry;  // PC 指向新程序入口
p->trapframe->sp = sp;          // 栈指针指向新栈顶

// 释放旧页表（不释放 trampoline 和 trapframe）
proc_freepagetable(oldpagetable, oldsz);
```

```
exec 过程中的页表切换：

执行前：
p->pagetable → 旧页表（fork 时复制的）
  映射：旧程序的代码、数据、栈

执行后：
p->pagetable → 新页表（exec 时创建的）
  映射：新程序的代码、数据、栈、trampoline、trapframe

旧页表 → 释放所有物理页面 → 回收到空闲链表
```

---

## 面试高频考点

### Q1: xv6 的物理内存分配器使用什么数据结构？

**答**：xv6 使用 **空闲链表（Free List）** 管理物理页面。每个空闲页面的前 8 字节存储指向下一个空闲页面的指针。`kalloc` 从链表头部取出页面，`kfree` 将页面归还到链表头部。链表通过 `kmem.lock` 自旋锁保护并发访问。

### Q2: 为什么 xv6 的内核页表使用直接映射？

**答**：直接映射（Direct Mapping）使内核虚拟地址等于物理地址，简化了内核代码实现。内核可以直接使用物理地址访问内存，无需复杂的地址转换。但直接映射的缺点是无法处理物理内存不连续的情况，且内核空间利用率受限。

### Q3: fork 的写时复制（COW）是如何工作的？

**答**：
1. fork 时只复制页表，不复制物理页面
2. 父子进程的页面都标记为只读（清除 PTE_W 位）
3. 当任一进程写入时，触发页故障
4. 内核分配新物理页面，复制原页面内容
5. 更新页表映射到新页面，恢复写权限
6. 每个物理页面维护引用计数，计数为 0 时释放

### Q4: exec 如何加载程序？

**答**：
1. 读取并验证 ELF 文件头（检查魔数 0x464C457F）
2. 创建新的用户页表
3. 遍历程序头表，对每个 PT_LOAD 类型的段：
   - 调用 `uvmalloc` 分配物理页面
   - 调用 `loadseg` 将文件内容复制到物理页面
4. 分配用户栈（2 个页面）
5. 在栈上构建 argc/argv
6. 切换到新页表，设置 PC 为 ELF 入口地址

### Q5: copyout 和 copyin 为什么不能直接用 memcpy？

**答**：因为用户页表和内核页表是不同的页表。用户虚拟地址在内核页表中可能没有映射，或者映射到不同的物理地址。`copyout`/`copyin` 通过 `walkaddr` 在用户页表中查找物理地址，然后在物理地址层面进行复制。

### Q6: xv6 的物理内存分配器有什么性能问题？如何改进？

**答**：
- **问题**：全局单一锁 `kmem.lock`，多核竞争严重
- **改进方案**：
  - 每 CPU 独立空闲链表（Per-CPU Free List）
  - 批量分配（Bulk Allocation）：一次取多个页面
  - 伙伴系统（Buddy System）：支持不同大小的内存块

### Q7: trampoline 页面为什么在内核页表和用户页表中都映射？

**答**：trampoline 页面包含 trap 处理代码。当用户态切换到内核态时，需要先切换页表（写 satp 寄存器），但在切换的瞬间，PC 仍然指向用户地址空间。trampoline 在两个页表中映射到相同的虚拟地址，确保切换页表时代码执行不会中断。

---

## 扩展阅读

### 推荐资源

1. **MIT 6.S081 教材**：*xv6: a simple, Unix-like teaching operating system*
   - Chapter 3: Page Tables — 详细的页表实现
   - Chapter 4: Traps and Interrupts — trap 处理与 trampoline

2. **xv6 源码**：https://github.com/mit-pdos/xv6-riscv
   - `kernel/kalloc.c` — 物理内存分配器
   - `kernel/vm.c` — 虚拟内存管理
   - `kernel/exec.c` — exec 系统调用

3. **RISC-V 特权架构手册**：
   - Sv39 页表格式
   - satp 寄存器
   - 页表权限位

4. **操作系统概念**：
   - *Operating Systems: Three Easy Pieces* (OSTEP)
   - Chapter 19: Paging: Faster Translations (TLBs)
   - Chapter 20: Paging: Smaller Tables

### 思考题

1. 如果 xv6 支持 64 位物理地址（超过 4GB RAM），需要修改哪些代码？
2. 如何实现一个支持大页（Huge Page）的 xv6 内核页表？
3. 如果要为 xv6 添加页面换出（Page Swapping）功能，需要修改哪些模块？
4. 比较 xv6 的物理内存分配器与 Linux 的伙伴系统，各有什么优缺点？

---

> **下一章预告**：我们将学习 xv6 的 trap 处理机制，理解系统调用、中断和异常的处理流程。
