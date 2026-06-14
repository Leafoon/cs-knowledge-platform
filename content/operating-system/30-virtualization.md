---
title: "Chapter 30: 虚拟化技术"
description: "深入理解全虚拟化/半虚拟化/硬件辅助虚拟化原理，掌握 CPU/内存/I/O 虚拟化与容器技术"
updated: "2026-06-11"
---

# Chapter 30: 虚拟化技术

> **本章目标**：
> - 理解虚拟化的核心概念、动机与发展历程
> - 掌握全虚拟化、半虚拟化与硬件辅助虚拟化三种方案的原理与差异
> - 深入理解 CPU 虚拟化的关键问题：敏感指令、二进制翻译、VT-x/AMD-V 硬件辅助
> - 掌握内存虚拟化的影子页表与 EPT/NPT 二级地址翻译机制
> - 理解 I/O 虚拟化的多种方案：设备模拟、virtio、SR-IOV、设备直通
> - 掌握容器技术的核心原理：Namespace 隔离与 cgroup 资源限制
> - 对比容器与虚拟机的架构差异与适用场景

---

## 30.1 虚拟化概述

### 30.1.1 什么是虚拟化

**虚拟化（Virtualization）** 是一种将物理硬件资源抽象为多个逻辑资源的技术。通过虚拟化，一台物理机器可以同时运行多个相互隔离的操作系统实例（称为 **虚拟机 VM**），每个虚拟机都以为自己独占了整个硬件。

```
虚拟化的核心思想：

┌─────────────────────────────────────────────────┐
│              物理硬件 (Physical Hardware)         │
│    CPU  |  内存  |  磁盘  |  网卡  |  设备       │
└─────────────────────────┬───────────────────────┘
                          │
                    ┌─────┴─────┐
                    │   VMM /   │  ← 虚拟机监控器
                    │ Hypervisor│    (Virtual Machine Monitor)
                    └─────┬─────┘
              ┌───────────┼───────────┐
              │           │           │
        ┌─────┴─────┐ ┌──┴──────┐ ┌──┴──────┐
        │    VM 1   │ │  VM 2   │ │  VM 3   │
        │  Linux    │ │ Windows │ │ FreeBSD │
        │  Guest    │ │  Guest  │ │  Guest  │
        └───────────┘ └─────────┘ └─────────┘
```

虚拟化的核心目标：

1. **资源复用**：在一台物理机上运行多个工作负载，提高硬件利用率
2. **隔离性**：虚拟机之间相互隔离，一个 VM 的故障不影响其他 VM
3. **可移植性**：虚拟机可以像文件一样拷贝、迁移、快照
4. **兼容性**：在同一硬件上运行不同操作系统
5. **安全性**：通过隔离提供额外的安全边界

### 30.1.2 虚拟化的动机与发展

虚拟化的概念最早由 IBM 在 1960 年代提出，用于在大型机上实现多用户共享。现代虚拟化的复兴始于 VMware 在 1998 年推出的 x86 虚拟化产品。

**虚拟化的驱动力**：

| 驱动因素 | 说明 |
|---------|------|
| 服务器整合 | 将多个低利用率服务器合并到一台物理机 |
| 开发测试 | 为开发人员提供隔离的测试环境 |
| 云计算 | IaaS 的基础技术（AWS EC2、Azure VM） |
| 灾难恢复 | 虚拟机快照与快速迁移 |
| 安全隔离 | 在不可信环境中运行代码 |
| 遗留系统 | 在新硬件上运行旧操作系统 |

### 30.1.3 虚拟化的三种方案

#### 1. 全虚拟化（Full Virtualization）

全虚拟化通过 **二进制翻译（Binary Translation）** 或硬件辅助，在 Guest OS 完全不修改的情况下运行。Guest OS 感知不到自己被虚拟化。

```
全虚拟化架构：

┌────────────────────────────────────────┐
│              Guest OS                   │
│   (未修改，以为运行在真实硬件上)         │
├────────────────────────────────────────┤
│              VMM                        │
│   ┌──────────────┐  ┌───────────────┐  │
│   │ 二进制翻译器  │  │ 设备模拟器    │  │
│   │ (Binary       │  │ (QEMU)        │  │
│   │  Translator)  │  │               │  │
│   └──────────────┘  └───────────────┘  │
├────────────────────────────────────────┤
│              真实硬件                    │
└────────────────────────────────────────┘
```

**原理**：VMM 运行在最高特权级（Ring 0），Guest OS 运行在较低特权级（Ring 1 或 Ring 3）。当 Guest OS 执行敏感指令时，会触发异常，VMM 捕获并模拟这些指令的行为。

**优点**：
- Guest OS 无需修改，兼容性好
- 可运行任何操作系统

**缺点**：
- 二进制翻译带来显著性能开销（某些指令序列需要翻译）
- 敏感指令的捕获和模拟代价高昂

**代表产品**：早期 VMware Workstation、VirtualBox

#### 2. 半虚拟化（Paravirtualization）

半虚拟化要求 **修改 Guest OS 的源代码**，将敏感指令替换为对 VMM 的 **超级调用（Hypercall）**。Guest OS 知道自己被虚拟化。

```
半虚拟化架构：

┌────────────────────────────────────────┐
│          Guest OS (已修改)              │
│   敏感指令 → Hypercall                  │
│   设备驱动 → 前端驱动 (Frontend)        │
├────────────────────────────────────────┤
│              VMM                        │
│   接收 Hypercall，执行特权操作           │
│   后端驱动 (Backend) 处理 I/O           │
├────────────────────────────────────────┤
│              真实硬件                    │
└────────────────────────────────────────┘
```

**原理**：Guest OS 被修改后，不再执行敏感指令，而是通过 Hypercall 直接请求 VMM 完成特权操作。由于不需要捕获异常和二进制翻译，性能接近原生。

**优点**：
- 性能接近原生（消除了二进制翻译开销）
- 上下文切换开销更小

**缺点**：
- 需要修改 Guest OS 内核源码
- 不支持闭源操作系统（如 Windows）

**代表产品**：Xen（早期）

#### 3. 硬件辅助虚拟化（Hardware-Assisted Virtualization）

硬件辅助虚拟化由 CPU 厂商直接在硬件层面提供虚拟化支持，引入新的 CPU 模式和指令集。

```
硬件辅助虚拟化架构：

┌────────────────────────────────────────┐
│              Guest OS (未修改)           │
│   运行在 VMX Non-Root 模式              │
├────────────────────────────────────────┤
│              VMM (Hypervisor)           │
│   运行在 VMX Root 模式                  │
│   通过 VMCS 管理 VM Entry/Exit          │
├────────────────────────────────────────┤
│   硬件虚拟化扩展 (VT-x / AMD-V)        │
│   VMX Root / Non-Root 模式              │
│   VMCS (Virtual Machine Control Str.)   │
├────────────────────────────────────────┤
│              真实硬件                    │
└────────────────────────────────────────┘
```

**原理**：CPU 引入两种运行模式 —— VMX Root 模式（VMM 运行）和 VMX Non-Root 模式（Guest 运行）。当 Guest 执行敏感指令时，CPU 硬件自动执行 VM Exit 切换到 VMM；VMM 处理后通过 VM Entry 返回 Guest。

**优点**：
- Guest OS 无需修改
- 性能接近原生（硬件直接处理，无需软件模拟）
- 简化了 VMM 的设计

**缺点**：
- 需要 CPU 支持虚拟化扩展
- VM Entry/VM Exit 仍有开销（约几百个 CPU 周期）

**代表产品**：KVM、现代 VMware ESXi、Hyper-V、Xen HVM

### 30.1.4 VMM/Hypervisor 的分类

#### Type 1 Hypervisor（裸金属型）

Type 1 Hypervisor 直接运行在物理硬件上，不依赖宿主操作系统。

```
Type 1 架构 (Bare-Metal):

┌──────────┐ ┌──────────┐ ┌──────────┐
│   VM 1   │ │   VM 2   │ │   VM 3   │
│ (Guest)  │ │ (Guest)  │ │ (Guest)  │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
┌────┴────────────┴────────────┴───────┐
│         Type 1 Hypervisor             │
│    (直接运行在硬件上，类似微内核OS)     │
├──────────────────────────────────────┤
│           物理硬件                    │
└──────────────────────────────────────┘
```

**特点**：
- 直接管理硬件资源
- 更高的性能和更低的延迟
- 更好的安全性（攻击面更小）
- 通常用于数据中心和云计算

**代表产品**：VMware ESXi、Microsoft Hyper-V、Xen、KVM*

> *注：KVM 作为 Linux 内核模块运行，虽然依赖 Linux 内核，但从虚拟化角度看它直接管理硬件，常被归类为 Type 1。

#### Type 2 Hypervisor（宿主型）

Type 2 Hypervisor 作为应用程序运行在宿主操作系统之上。

```
Type 2 架构 (Hosted):

┌──────────┐ ┌──────────┐
│   VM 1   │ │   VM 2   │
│ (Guest)  │ │ (Guest)  │
└────┬─────┘ └────┬─────┘
     │            │
┌────┴────────────┴──────────────┐
│       Type 2 Hypervisor         │
│   (作为应用程序运行在宿主OS上)   │
├────────────────────────────────┤
│       宿主操作系统 (Host OS)     │
│       Linux / Windows / macOS   │
├────────────────────────────────┤
│           物理硬件               │
└────────────────────────────────┘
```

**特点**：
- 安装和使用简单，像普通软件一样
- 依赖宿主操作系统管理硬件
- 性能低于 Type 1（多了一层 OS 抽象）
- 通常用于桌面开发测试

**代表产品**：VMware Workstation/Fusion、VirtualBox、Parallels Desktop

#### Type 1 vs Type 2 对比

| 维度 | Type 1 (裸金属) | Type 2 (宿主型) |
|------|----------------|----------------|
| 运行位置 | 直接运行在硬件上 | 运行在宿主 OS 之上 |
| 性能 | 更高（无额外 OS 层） | 较低（宿主 OS 开销） |
| 安全性 | 更高（攻击面小） | 较低（依赖宿主 OS） |
| 管理复杂度 | 较高（需要专用管理） | 较低（像普通软件安装） |
| 硬件支持 | 需要专用驱动 | 利用宿主 OS 驱动 |
| 典型用途 | 数据中心、云计算 | 桌面开发、测试 |
| 代表产品 | ESXi, Hyper-V, Xen | VirtualBox, VMware WS |

---

## 30.2 CPU 虚拟化

CPU 虚拟化是虚拟化技术中最具挑战性的部分。x86 架构在设计时并未考虑虚拟化支持，其指令集中存在大量 **敏感指令**，使得虚拟化变得困难。

### 30.2.1 敏感指令问题

**Popek-Goldberg 虚拟化定理**（1974）指出：一个可虚拟化的计算机架构必须满足 —— 所有敏感指令都是特权指令。

- **特权指令（Privileged Instructions）**：只能在最高特权级执行的指令，非特权级执行时会触发异常
- **敏感指令（Sensitive Instructions）**：会影响系统全局状态的指令，包括：
  - **控制敏感指令**：修改系统配置（如修改页表基址寄存器 CR3）
  - **行为敏感指令**：执行结果依赖于特权级或系统配置（如读取当前特权级）

**x86 的问题**：x86 架构中有 17 条敏感但非特权的指令，例如：
- `POPF`（Pop Flags）：在用户态执行时静默失败，不触发异常
- `SGDT`/`SIDT`：存储描述符表寄存器，可在用户态执行
- `PUSHF`：将标志寄存器压栈，暴露真实的特权级信息

```
x86 特权级模型 (Ring Model):

Ring 0  ┌───────────────────┐ ← 内核态 (Kernel)
        │  最高特权级         │
        │  可执行所有指令      │
        ├───────────────────┤
Ring 1  │  (通常未使用)       │ ← VMM 可以运行在这里?
        ├───────────────────┤
Ring 2  │  (通常未使用)       │
        ├───────────────────┤
Ring 3  │  用户态 (User)      │ ← Guest OS 运行在这里?
        │  受限指令集         │
        └───────────────────┘

问题：如果 Guest OS 运行在 Ring 3，
     它的特权指令会触发异常 → VMM 必须捕获并模拟
     但敏感非特权指令不会触发异常 → 产生错误行为！
```

### 30.2.2 二进制翻译

**二进制翻译（Binary Translation）** 是早期全虚拟化的核心技术，由 VMware 在 1998 年引入。

```
二进制翻译的工作流程：

Guest 代码执行
       │
       ▼
┌──────────────────┐
│ 扫描基本块 (BBB)  │ ← 每次执行新代码页时触发
│ 检查是否包含      │
│ 敏感指令          │
└───────┬──────────┘
        │
   ┌────┴────┐
   │ 包含?   │
   └────┬────┘
    Yes │    No
    ▼       ▼
┌────────┐ ┌────────────┐
│翻译为   │ │直接执行     │
│等价的   │ │(影子代码)   │
│安全代码 │ │             │
└───┬────┘ └─────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ 翻译后的代码 (Translation Cache/TC)  │
│                                      │
│ 例如:                                │
│   POPF → MOV [VMM_FLAGS], new_flags  │
│         (VMM 会审查并可能修改 flags)   │
│                                      │
│   SGDT → MOV [mem], [fake_GDT_value] │
│         (返回 VMM 提供的假值)          │
└──────────────────────────────────────┘
```

**优化技术**：
1. **基本块翻译**：以基本块为单位翻译，缓存翻译结果
2. **直接块链接（Direct Block Chaining）**：将翻译后的基本块直接链接，避免返回调度器
3. **影子翻译**：不含敏感指令的代码段直接运行，无需翻译
4. **页面级监控**：将已翻译的代码页面标记为可执行，未翻译的页面触发缺页异常进行翻译

**性能开销来源**：
- 翻译本身的一次性开销（可被缓存缓解）
- 翻译代码的执行效率可能低于原生代码
- 系统调用密集型工作负载影响最大（每个 syscall 入口/出口都需要翻译）

### 30.2.3 Intel VT-x / AMD-V 硬件辅助虚拟化

2005-2006 年，Intel 和 AMD 分别推出了硬件虚拟化扩展：**Intel VT-x** 和 **AMD-V（SVM）**。它们从根本上解决了 x86 的虚拟化难题。

#### VMX 模式

```
CPU 运行模式：

┌─────────────────────────────────────────────────┐
│                VMX Root 模式                     │
│         (VMM/Hypervisor 运行)                    │
│                                                  │
│   ┌──────────────────────────────────────────┐   │
│   │           VMX Non-Root 模式               │   │
│   │         (Guest OS / Guest App 运行)       │   │
│   │                                           │   │
│   │   Ring 0: Guest Kernel                    │   │
│   │   Ring 3: Guest User Application          │   │
│   └──────────────────────────────────────────┘   │
│                                                  │
│   VM Entry: Root → Non-Root (进入虚拟机)          │
│   VM Exit:  Non-Root → Root (退出虚拟机)          │
└─────────────────────────────────────────────────┘
```

- **VMX Root 模式**：VMM 运行在此模式，拥有完整权限，等价于传统的 Ring 0
- **VMX Non-Root 模式**：Guest 运行在此模式，某些操作会触发 VM Exit
- **VM Entry**：从 Root 模式切换到 Non-Root 模式，加载 Guest 状态
- **VM Exit**：从 Non-Root 模式切换到 Root 模式，保存 Guest 状态，恢复 Host 状态

#### VMCS（Virtual Machine Control Structure）

VMCS 是一个 4KB 的内存数据结构，由硬件管理，用于保存虚拟机的完整状态。

```
VMCS 结构 (4096 字节):

┌──────────────────────────────────────────┐
│              VMCS Region                  │
├──────────────────────────────────────────┤
│  Guest State Area                        │
│  ┌──────────────────────────────────┐    │
│  │ Guest CR0, CR3, CR4             │    │
│  │ Guest RSP, RIP, RFLAGS          │    │
│  │ Guest CS, SS, DS, ES, FS, GS    │    │
│  │ Guest GDTR, IDTR, LDTR, TR      │    │
│  │ Guest EFER, PAT, etc.           │    │
│  └──────────────────────────────────┘    │
├──────────────────────────────────────────┤
│  Host State Area                         │
│  ┌──────────────────────────────────┐    │
│  │ Host CR0, CR3, CR4              │    │
│  │ Host RSP, RIP                   │    │
│  │ Host CS, SS, DS, ES, FS, GS     │    │
│  └──────────────────────────────────┘    │
├──────────────────────────────────────────┤
│  VM-Execution Control Fields             │
│  ┌──────────────────────────────────┐    │
│  │ Pin-Based Controls               │    │
│  │ (外部中断/ NMI 虚拟化)            │    │
│  │ Processor-Based Controls         │    │
│  │ (I/O 位图, MSR 位图, CR3 目标等) │    │
│  │ Exception Bitmap                 │    │
│  │ (哪些异常触发 VM Exit)            │    │
│  └──────────────────────────────────┘    │
├──────────────────────────────────────────┤
│  VM-Exit Information Fields              │
│  ┌──────────────────────────────────┐    │
│  │ Exit Reason                      │    │
│  │ Exit Qualification               │    │
│  │ VM-Exit Instruction Length       │    │
│  └──────────────────────────────────┘    │
├──────────────────────────────────────────┤
│  VM-Entry Control Fields                 │
│  ┌──────────────────────────────────┐    │
│  │ Entry Controls (MSR load, etc.)  │    │
│  │ VM-Entry Interruption Info       │    │
│  └──────────────────────────────────┘    │
├──────────────────────────────────────────┤
│  VM-Exit Control Fields                  │
│  ┌──────────────────────────────────┐    │
│  │ Exit Controls (MSR store, etc.)  │    │
│  └──────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

#### VM Entry / VM Exit 流程

```
VM Entry 流程:
1. VMM 执行 VMLAUNCH/VMRESUME 指令
2. CPU 检查 VMCS 合法性
3. 从 VMCS Guest State Area 加载 Guest 寄存器
4. 处理 Pending Event（如虚拟中断注入）
5. 切换到 VMX Non-Root 模式
6. 从 Guest RIP 继续执行

VM Exit 流程:
1. Guest 执行敏感操作（如 CPUID, I/O, CR访问, 外部中断等）
2. CPU 自动：
   a. 保存 Guest 状态到 VMCS Guest State Area
   b. 从 VMCS Host State Area 加载 Host 状态
   c. 记录 Exit Reason 到 VMCS
   d. 切换到 VMX Root 模式
   e. 跳转到 VMCS 中记录的 Host RIP
3. VMM 的 VM Exit Handler 处理该事件
4. VMM 执行 VMRESUME 返回 Guest
```

**典型的 VM Exit 原因**：

| Exit Reason | 说明 | VMM 处理方式 |
|-------------|------|-------------|
| CPUID | Guest 执行 CPUID 指令 | 返回虚拟化的 CPU 特征 |
| CR Access | Guest 读写控制寄存器 | 模拟或直接转发 |
| I/O Instruction | Guest 执行 IN/OUT 指令 | 模拟设备 I/O |
| EPT Violation | Guest 访问未映射的物理页 | 分配物理页并更新 EPT |
| External Interrupt | 外部设备中断 | 虚拟中断注入或直接传递 |
| HLT | Guest 执行停机指令 | 调度其他 vCPU 或 idle |
| MSR Read/Write | Guest 访问 MSR 寄存器 | 模拟或记录 |

### 30.2.4 KVM：基于 Linux 的硬件辅助虚拟化

**KVM（Kernel-based Virtual Machine）** 是 Linux 内核中的虚拟化模块，它将 Linux 内核本身转变为 Type 1 Hypervisor。

```
KVM 架构：

┌─────────────────────────────────────────────┐
│  用户态 (QEMU)                               │
│  ┌──────────────┐ ┌────────────────────────┐│
│  │ 设备模拟      │ │ VM 管理 (创建/配置)     ││
│  │ (QEMU 设备模型)│ │ (ioctl /dev/kvm)       ││
│  └──────┬───────┘ └───────────┬────────────┘│
│         │                     │              │
├─────────┼─────────────────────┼──────────────┤
│  内核态 (Linux Kernel + KVM)  │              │
│  ┌──────┴─────────────────────┴───────────┐ │
│  │              KVM 模块                   │ │
│  │  ┌──────────┐  ┌──────────┐            │ │
│  │  │ VT-x/    │  │ EPT      │            │ │
│  │  │ AMD-V    │  │ 管理     │            │ │
│  │  │ 管理     │  │          │            │ │
│  │  └──────────┘  └──────────┘            │ │
│  │  ┌──────────┐  ┌──────────┐            │ │
│  │  │ vCPU     │  │ IRQ      │            │ │
│  │  │ 调度     │  │ 虚拟化   │            │ │
│  │  └──────────┘  └──────────┘            │ │
│  └────────────────────────────────────────┘ │
│              Linux 内核调度器                 │
├─────────────────────────────────────────────┤
│              物理硬件                         │
└─────────────────────────────────────────────┘
```

KVM 的设计哲学：
- **vCPU 对应内核线程**：每个虚拟 CPU 由一个 Linux 内核线程实现
- **利用 Linux 调度器**：vCPU 的调度复用 Linux CFS 调度器
- **QEMU 作为用户态伴侣**：设备模拟、BIOS 等由 QEMU 在用户态实现
- **KVM 提供 ioctl 接口**：用户态通过 `/dev/kvm` 设备文件控制虚拟机

---

## 30.3 内存虚拟化

内存虚拟化是虚拟化的另一核心挑战。在虚拟化环境中存在两层地址空间，需要高效的地址翻译机制。

### 30.3.1 两层地址空间

```
虚拟化环境中的地址空间：

Guest 应用程序
    │  GVA (Guest Virtual Address)
    ▼
Guest OS 页表
    │  GPA (Guest Physical Address)
    ▼
VMM 管理
    │  HPA (Host Physical Address)
    ▼
物理内存

三层地址：
- GVA: Guest 虚拟地址 (应用看到的地址)
- GPA: Guest 物理地址 (Guest OS 认为的"物理"地址)
- HPA: 宿主物理地址 (真实的物理内存地址)

翻译需求：GVA → GPA → HPA (两级翻译)
```

**问题**：传统操作系统只做一级翻译（VA → PA），虚拟化环境需要两级翻译。如果每访问一个内存地址都要走两级页表查找，性能损失将非常严重。

### 30.3.2 影子页表（Shadow Page Tables）

影子页表是早期的软件解决方案（VMware 提出），VMM 为每个 Guest 进程维护一个 **影子页表**，直接将 GVA 映射到 HPA。

```
影子页表工作原理：

Guest 视角：                 VMM 维护的影子页表：
┌────────────┐              ┌────────────┐
│ GVA 0x4000 │              │ GVA 0x4000 │
│     ↓      │              │     ↓      │
│  Guest PT  │   ──映射──→  │   HPA      │
│     ↓      │              │  0x80000   │
│ GPA 0x4000 │              └────────────┘
└────────────┘              (直接 GVA → HPA)

实现过程：
1. Guest 创建自己的页表 (GVA → GPA)
2. VMM 截获 Guest 页表的修改
3. VMM 查找 GPA 对应的 HPA
4. VMM 创建影子页表 (GVA → HPA)
5. VMM 将影子页表加载到 CR3
6. Guest 应用使用影子页表进行地址翻译

当 Guest 修改页表时：
- Guest 执行 MOV CR3, <新页表地址>
- 触发 VM Exit
- VMM 分析 Guest 的新页表
- VMM 创建/更新对应的影子页表
- VMM 执行 VMRESUME
```

**影子页表的挑战**：

1. **维护开销**：Guest 每次修改页表都会触发 VM Exit，VMM 需要同步更新影子页表
2. **内存开销**：每个 Guest 进程都需要一个影子页表副本
3. **复杂度高**：VMM 需要精确模拟 Guest 的页表行为，包括 TLB 管理
4. **写保护监控**：VMM 需要将 Guest 页表所在页面设为只写保护，当 Guest 写入时触发 EPT Violation

### 30.3.3 EPT/NPT：二级地址翻译

Intel 的 **EPT（Extended Page Tables）** 和 AMD 的 **NPT（Nested Page Tables）** 从硬件层面解决了两层地址翻译的问题。

#### EPT 翻译过程

```
EPT 二级地址翻译 (GVA → GPA → HPA)：

Guest 应用发出内存请求：GVA = 0x0040_0000

第一步：Guest 页表翻译 (GVA → GPA)
┌─────────────────────────────────────────────┐
│  CPU 使用 Guest CR3 指向的 Guest 页表        │
│                                              │
│  GVA: 0x0040_0000                           │
│  ├── PML4 Index: 0                          │
│  ├── PDPT Index: 1                          │
│  ├── PD Index: 0                            │
│  ├── PT Index: 0                            │
│  └── Offset: 0x000                          │
│                                              │
│  结果: GPA = 0x2000_0000                     │
│  (Guest 页表中的 PTE 记录了 GPA)             │
└─────────────────────────────────────────────┘
                    │
                    ▼ GPA = 0x2000_0000

第二步：EPT 页表翻译 (GPA → HPA)
┌─────────────────────────────────────────────┐
│  CPU 使用 EPTP 指向的 EPT 页表              │
│                                              │
│  GPA: 0x2000_0000                           │
│  ├── EPT PML4 Index: 4                      │
│  ├── EPT PDPT Index: 0                      │
│  ├── EPT PD Index: 0                        │
│  ├── EPT PT Index: 0                        │
│  └── Offset: 0x000                          │
│                                              │
│  结果: HPA = 0x5000_0000                     │
└─────────────────────────────────────────────┘
                    │
                    ▼ HPA = 0x5000_0000

最终：物理内存访问 HPA = 0x5000_0000
```

#### EPT 的页表结构

```
EPT 页表层次 (以 4KB 页面为例)：

EPTP ──→ EPT PML4 (512 项)
              │
              ├── EPT PDPTE[0] ──→ EPT PD (512 项)
              │                        │
              │                        ├── EPT PDE[0] ──→ EPT PT (512 项)
              │                        │                    │
              │                        │                    ├── EPT PTE[0] ──→ HPA
              │                        │                    ├── EPT PTE[1] ──→ HPA
              │                        │                    └── ...
              │                        │
              │                        ├── EPT PDE[1] ──→ ...
              │                        └── ...
              │
              ├── EPT PDPTE[1] ──→ ...
              └── ...

每项格式 (64 bits):
┌────┬──────────────┬───┬───┬───┬───┬───┬───┐
│Rsvd│   HPA[51:12] │ D │ A │ X │ W │ R │   │
└────┴──────────────┴───┴───┴───┴───┴───┴───┘
  R=可读, W=可写, X=可执行, A=访问位, D=脏位
```

#### EPT 的性能分析

EPT 引入了 **TLB 压力增大** 的问题，因为一次 GVA→HPA 翻译需要走两级页表，总共可能需要 24 次内存访问（最坏情况：4 级 Guest 页表 × 4 级 EPT 页表 + 两个页表自身的加载）。

```
最坏情况的内存访问次数：

无虚拟化 (VA → PA):
  4 级页表 × 1 次/级 + 1 次数据访问 = 5 次

有 EPT (GVA → GPA → HPA):
  Guest 页表遍历: 4 次 Guest PTE 访问
  每次 Guest PTE 访问需要 EPT 翻译: 4 × 4 = 16 次
  + EPT 自身加载: 4 次
  + 1 次最终数据访问 (也需要 EPT 翻译): 4 次
  总计: 4 × 4 + 4 + 4 = 24 次

优化：TLB 缓存 GVA → HPA 直接映射
     大页 (2MB/1GB) 减少页表级数
     EPT 预取 (EPT Prefetching)
```

### 30.3.4 内存气球（Memory Ballooning）

内存气球是虚拟化环境中的内存动态管理技术，允许 VMM 在虚拟机之间动态调整内存分配。

```
内存气球工作原理：

初始状态: VM1 分配 4GB, VM2 分配 4GB, 物理机 8GB
┌──────────────┐ ┌──────────────┐
│     VM1      │ │     VM2      │
│   4GB RAM    │ │   4GB RAM    │
│              │ │              │
│  ┌────────┐  │ │  ┌────────┐  │
│  │Balloon │  │ │  │Balloon │  │
│  │ Driver │  │ │  │ Driver │  │
│  └────────┘  │ │  └────────┘  │
└──────────────┘ └──────────────┘

当 VMM 需要从 VM1 回收内存给 VM2:

Step 1: VMM 通知 VM1 的 Balloon Driver "充气"
Step 2: Balloon Driver 在 Guest 内分配内存页面
Step 3: Balloon Driver 将这些页面"归还"给 VMM
Step 4: VMM 将这些物理页面重新分配给 VM2

充气后:
┌──────────────┐ ┌──────────────┐
│     VM1      │ │     VM2      │
│  4GB(2可用)  │ │   4GB(6可用) │
│  ┌────────┐  │ │              │
│  │████████│  │ │  (额外2GB)   │
│  │充气中  │  │ │              │
│  │████████│  │ │              │
│  └────────┘  │ │              │
└──────────────┘ └──────────────┘

放气过程:
VMM 通知 Balloon Driver "放气"
→ Balloon Driver 释放之前分配的页面
→ VMM 回收这些物理页面
→ VM1 恢复可用内存
```

**应用场景**：
- 超售（Overcommit）：总虚拟内存超过物理内存
- 热迁移：在迁移过程中逐步释放源端内存
- 内存回收：在 VM 负载降低时回收多余内存

### 30.3.5 KSM（Kernel Same-Page Merging）

KSM 是 Linux 内核的一个功能，通过扫描物理内存，将内容相同的页面合并为一个共享页面（写时复制）。

```
KSM 页共享过程：

合并前：
VM1 Page A: [数据 X] ──→ HPA 0x1000
VM1 Page B: [数据 Y] ──→ HPA 0x2000
VM2 Page C: [数据 X] ──→ HPA 0x3000  ← 内容与 Page A 相同
VM2 Page D: [数据 Z] ──→ HPA 0x4000

KSM 扫描后：
VM1 Page A: [数据 X] ──→ HPA 0x1000 ← 共享 (COW)
VM1 Page B: [数据 Y] ──→ HPA 0x2000
VM2 Page C: [数据 X] ──→ HPA 0x1000 ← 共享 (COW)
VM2 Page D: [数据 Z] ──→ HPA 0x4000

节省：1 个物理页面 (0x3000 被释放)

当 VM1 写入 Page A:
触发 COW → VMM 分配新物理页面
VM1 Page A: [数据 X'] ──→ HPA 0x5000 (私有副本)
VM2 Page C: [数据 X]  ──→ HPA 0x1000 (仍为共享)
```

**KSM 的哈希策略**：
1. 使用 **两棵红黑树** 组织页面
   - `stable_tree`：已确认合并的页面
   - `unstable_tree`：待比较的候选页面
2. 先计算页面内容的哈希值
3. 哈希匹配的页面再逐字节比较
4. 内容完全一致的页面合并

**KSM 的适用场景**：
- 运行相同操作系统和应用的多个虚拟机
- VDI（虚拟桌面基础设施）
- 大量相似容器的场景

**KSM 的风险**：
- 安全风险：基于时间的侧信道攻击可能推断共享页面内容
- 性能开销：扫描和哈希计算消耗 CPU
- 写时复制开销：写共享页面时需要复制

---

## 30.4 I/O 虚拟化

I/O 设备种类繁多，I/O 虚拟化的目标是让虚拟机能够高效地使用物理 I/O 设备，同时保持设备的隔离性和可迁移性。

### 30.4.1 设备模拟（Device Emulation）

设备模拟是最直接的 I/O 虚拟化方式，VMM 在软件中完全模拟一个硬件设备。

```
设备模拟架构：

Guest OS
┌──────────────────────────┐
│   Guest Device Driver     │
│   (标准驱动，未修改)       │
└────────────┬─────────────┘
             │ I/O 请求 (IN/OUT, MMIO)
             ▼
┌──────────────────────────┐
│   设备模拟器 (QEMU)       │
│   ┌────────────────────┐ │
│   │ 模拟设备:           │ │
│   │ - e1000 网卡        │ │
│   │ - IDE/SATA 磁盘    │ │
│   │ - VGA 显卡          │ │
│   │ - PS/2 键盘鼠标     │ │
│   │ - 串口/并口         │ │
│   └────────────────────┘ │
│                          │
│   模拟设备寄存器、中断、   │
│   DMA 操作              │
└────────────┬─────────────┘
             │ 转换为宿主 I/O 操作
             ▼
┌──────────────────────────┐
│   宿主 OS 设备驱动        │
│   (真实硬件驱动)          │
└──────────────────────────┘
```

**工作流程**：
1. Guest 驱动向模拟设备的寄存器写入命令
2. CPU 执行 I/O 指令或访问 MMIO 区域 → VM Exit
3. VMM 将请求转发给设备模拟器
4. 模拟器执行设备逻辑，可能调用宿主 OS 的设备驱动
5. 完成后通过虚拟中断通知 Guest

**性能问题**：
- 每次 I/O 操作都触发 VM Exit（上下文切换开销大）
- 数据需要在 Guest 和 Host 之间拷贝
- 模拟器本身有 CPU 开销
- 不适合高性能 I/O 场景

### 30.4.2 半虚拟化 I/O（virtio）

virtio 是一种标准化的半虚拟化 I/O 框架，由 Rusty Russell 提出。它定义了一套通用的设备抽象，Guest 安装 virtio 前端驱动，VMM 实现 virtio 后端。

```
virtio 架构：

Guest OS                                Host/VMM
┌──────────────────┐                  ┌──────────────────┐
│  virtio 前端驱动  │                  │  virtio 后端     │
│  (virtio-net)    │                  │  (QEMU/vhost)   │
│  (virtio-blk)    │                  │                  │
│  (virtio-scsi)   │                  │                  │
└────────┬─────────┘                  └────────┬─────────┘
         │                                     │
         │     共享内存 (Virtqueue)              │
         └──────────────────┬──────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │      Virtqueue 结构         │
              │                            │
              │  ┌──────────┐              │
              │  │Available │ ← Guest 放入 │
              │  │  Ring    │   描述符      │
              │  ├──────────┤              │
              │  │  Used    │ ← Host 处理  │
              │  │  Ring    │   完成后归还  │
              │  ├──────────┤              │
              │  │Descriptor│ ← 描述符表   │
              │  │  Table   │   (sg list)  │
              │  └──────────┘              │
              └────────────────────────────┘
```

**Virtqueue 详细机制**：

```
描述符表 (Descriptor Table):
┌─────┬──────┬──────┬──────────────┐
│ Idx │ Addr │ Len  │   Flags      │
├─────┼──────┼──────┼──────────────┤
│  0  │ 0x.. │ 256  │ NEXT         │ → 指向 Idx 1
│  1  │ 0x.. │ 1024 │ WRITE        │ → Host 写入结果
│  2  │ 0x.. │ 64   │ (独立)       │
└─────┴──────┴──────┴──────────────┘

Available Ring (Guest → Host):
┌──────────────────────────┐
│ flags │ idx │ ring[0..N] │
│       │     │ desc_idx   │
└──────────────────────────┘

Used Ring (Host → Guest):
┌──────────────────────────┐
│ flags │ idx │ ring[0..N] │
│       │     │ {id, len}  │
└──────────────────────────┘

I/O 流程:
1. Guest 构建描述符链 (scatter-gather list)
2. Guest 将描述符头索引写入 Available Ring
3. Guest 通知 Host (写 MMIO 或 eventfd 通知)
4. Host 从 Available Ring 取出描述符
5. Host 解析描述符链，执行实际 I/O
6. Host 将结果写入描述符的 WRITE 缓冲区
7. Host 将完成信息放入 Used Ring
8. Host 通过虚拟中断通知 Guest
```

**virtio 的优化演进**：
- **vhost-net**：将 virtio 后端从 QEMU 用户态移到内核态，减少上下文切换
- **vhost-user**：将后端移到 DPDK 等用户态进程，适用于网络功能
- **virtio 1.1+**：支持 packed virtqueue（单环替代双环，减少缓存未命中）

**性能对比**：

| 方案 | IOPS | 延迟 | CPU 开销 |
|------|------|------|---------|
| 设备模拟 | 1x | 高 | 高 |
| virtio (QEMU 后端) | 3-5x | 中 | 中 |
| vhost-net | 5-10x | 低-中 | 低 |
| 设备直通 | 95% 原生 | 最低 | 最低 |

### 30.4.3 SR-IOV（Single Root I/O Virtualization）

SR-IOV 是 PCI-SIG 定义的硬件标准，允许一个物理 PCIe 设备虚拟出多个 **虚拟功能（VF, Virtual Function）**，每个 VF 可以直接分配给一个虚拟机。

```
SR-IOV 架构：

┌─────────────────────────────────────────────┐
│              物理 PCIe 设备                   │
│  ┌────────────────────────────────────────┐ │
│  │  PF (Physical Function)                │ │
│  │  - 完整的 PCIe 功能                     │ │
│  │  - 管理和配置 VF                       │ │
│  └────────────────────────────────────────┘ │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
│  │ VF 0 │ │ VF 1 │ │ VF 2 │ │ VF 3 │       │
│  │      │ │      │ │      │ │      │       │
│  │独立的 │ │独立的 │ │独立的 │ │独立的 │       │
│  │BAR   │ │BAR   │ │BAR   │ │BAR   │       │
│  │中断  │ │中断  │ │中断  │ │中断  │       │
│  │队列  │ │队列  │ │队列  │ │队列  │       │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘       │
└─────┼────────┼────────┼────────┼────────────┘
      │        │        │        │
      ▼        ▼        ▼        ▼
   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
   │VM 0 │ │VM 1 │ │VM 2 │ │VM 3 │
   │     │ │     │ │     │ │     │
   │直接  │ │直接  │ │直接  │ │直接  │
   │访问  │ │访问  │ │访问  │ │访问  │
   │VF 0 │ │VF 1 │ │VF 2 │ │VF 3 │
   └─────┘ └─────┘ └─────┘ └─────┘
```

**PF 和 VF**：
- **PF（Physical Function）**：完整的 PCIe 功能，由宿主机管理，可以配置和创建 VF
- **VF（Virtual Function）**：轻量级 PCIe 功能，只能处理数据面，直接分配给 VM

**SR-IOV 优势**：
- **接近原生性能**：VM 直接访问 VF 的硬件队列，无需 VMM 介入
- **硬件级隔离**：每个 VF 有独立的中断、DMA 地址空间
- **低 CPU 开销**：数据路径不经过 VMM

**SR-IOV 限制**：
- 需要硬件支持 SR-IOV
- 热迁移困难（设备状态绑定在硬件上）
- VF 数量有限（典型网卡支持 64-256 个 VF）

### 30.4.4 设备直通（Passthrough / VFIO）

设备直通将整个物理设备（或 SR-IOV 的 VF）直接分配给一个虚拟机，Guest 直接控制设备硬件。

```
设备直通架构：

方案 1: 整个设备直通
┌────────────────────────┐
│        VM              │
│  ┌──────────────────┐  │
│  │ 原生设备驱动      │  │
│  │ (如 ixgbe, nvidia)│  │
│  └────────┬─────────┘  │
└───────────┼────────────┘
            │ 直接 DMA/MMIO
            ▼
    ┌───────────────┐
    │  物理设备      │
    │  (独占)       │
    └───────────────┘

方案 2: SR-IOV VF 直通
┌──────────┐ ┌──────────┐
│   VM 1   │ │   VM 2   │
│ ┌──────┐ │ │ ┌──────┐ │
│ │VF 驱动│ │ │ │VF 驱动│ │
│ └──┬───┘ │ │ └──┬───┘ │
└────┼─────┘ └────┼─────┘
     │            │
   ┌─┴──┐      ┌─┴──┐
   │VF 0│      │VF 1│  (同一物理设备的不同 VF)
   └────┘      └────┘
```

**VFIO（Virtual Function I/O）** 是 Linux 内核的设备直通框架：

```
VFIO 组件：

┌─────────────────────────────────────┐
│            VM (Guest)                │
│   原生设备驱动                        │
└────────────┬────────────────────────┘
             │ DMA, MMIO
┌────────────┴────────────────────────┐
│         IOMMU (VT-d / AMD-Vi)       │
│   ┌─────────────────────────────┐   │
│   │ DMA Remapping               │   │
│   │ GPA → HPA 地址转换           │   │
│   │ 设备级隔离 (Domain)          │   │
│   └─────────────────────────────┘   │
├─────────────────────────────────────┤
│         VFIO (内核框架)              │
│   ┌─────────────────────────────┐   │
│   │ /dev/vfio/vfio  (容器)      │   │
│   │ /dev/vfio/N     (设备组)    │   │
│   │                             │   │
│   │ IOMMU Group → Domain 隔离   │   │
│   └─────────────────────────────┘   │
├─────────────────────────────────────┤
│         QEMU (设备模型)              │
│   - 配置 VFIO 设备                   │
│   - 设置 IOMMU 映射                  │
│   - 模拟 MSI-X 中断路由              │
├─────────────────────────────────────┤
│         物理设备                      │
└─────────────────────────────────────┘
```

**IOMMU 的作用**：
1. **DMA 重映射**：将设备的 DMA 地址（GPA）翻译为真实物理地址（HPA）
2. **设备隔离**：每个设备只能访问分配给它的内存区域
3. **中断重映射**：将设备中断路由到正确的 VM

---

## 30.5 容器与轻量级虚拟化

容器是一种操作系统级虚拟化技术，与传统虚拟机相比，容器更轻量、启动更快、资源开销更小。

### 30.5.1 容器 vs 虚拟机

```
虚拟机 vs 容器 架构对比：

虚拟机 (VM):                    容器 (Container):
┌────────────┐ ┌────────────┐  ┌────────────┐ ┌────────────┐
│   App A    │ │   App B    │  │   App A    │ │   App B    │
├────────────┤ ├────────────┤  ├────────────┤ ├────────────┤
│  Binaries  │ │  Binaries  │  │  Binaries  │ │  Binaries  │
│  Libraries │ │  Libraries │  │  Libraries │ │  Libraries │
├────────────┤ ├────────────┤  ├────────────┤ ├────────────┤
│ Guest OS   │ │ Guest OS   │  │  容器运行时 (containerd)   │
│ (完整内核) │ │ (完整内核) │  │  Namespace + Cgroup        │
├────────────┴─┴────────────┤  ├────────────────────────────┤
│     Hypervisor            │  │      宿主 OS 内核           │
├───────────────────────────┤  │      (共享内核)             │
│     物理硬件               │  ├────────────────────────────┤
└───────────────────────────┘  │      物理硬件               │
                               └────────────────────────────┘
```

**核心差异对比**：

| 维度 | 虚拟机 (VM) | 容器 (Container) |
|------|-----------|-----------------|
| 隔离级别 | 硬件级（完整 OS） | 操作系统级（进程级） |
| 内核 | 每个 VM 独立内核 | 共享宿主内核 |
| 启动时间 | 分钟级 | 秒级（甚至毫秒级） |
| 内存开销 | GB 级（每个 VM） | MB 级（每个容器） |
| 镜像大小 | GB 级 | MB 级 |
| 安全隔离 | 强（硬件隔离） | 中（内核共享是攻击面） |
| 性能损耗 | 5-15% | 接近 0% |
| 运行密度 | 每物理机 10-50 VM | 每物理机 100-1000 容器 |
| 适用场景 | 多租户、强隔离 | 微服务、CI/CD、开发环境 |

### 30.5.2 Linux Namespace 隔离

**Namespace** 是 Linux 内核提供的资源隔离机制，它让每个进程拥有独立的系统资源视图。容器的隔离性主要来自 Namespace。

#### PID Namespace

```
PID Namespace 隔离：

宿主机视角:
PID 1 (systemd) ── 宿主机 init
PID 500 (containerd)
PID 1000 (bash) ── 容器 init 进程
PID 1001 (nginx)
PID 1002 (worker)

容器内部视角 (PID Namespace):
PID 1 (bash) ── 容器内看到的 init 进程
PID 2 (nginx)
PID 3 (worker)

关键特性:
- 容器内 PID 1 是容器的 init 进程
- 容器内看不到宿主机和其他容器的进程
- 容器内 PID 1 负责回收僵尸进程
- 可嵌套: 容器内可创建子 PID Namespace
```

#### Network Namespace

```
Network Namespace 隔离：

每个 Network Namespace 拥有独立的:
- 网络接口 (lo, eth0, ...)
- IP 地址和路由表
- iptables 规则
- /proc/net 和 /sys/class/net

容器网络拓扑 (veth pair):

┌──────────────┐          ┌──────────────┐
│  Container   │          │  Host        │
│  Network NS  │          │  Network NS  │
│              │          │              │
│  eth0        │          │  veth-xxx    │
│  10.0.0.2    │──────────│  (bridge端口)│
│              │  veth    │              │
│              │  pair    │  docker0     │
│              │          │  10.0.0.1    │
└──────────────┘          └──────┬───────┘
                                 │
                                 │ NAT/路由
                                 ▼
                           ┌──────────┐
                           │ eth0     │
                           │ 物理网卡  │
                           └──────────┘

Docker 默认网络:
- 创建 docker0 网桥
- 每个容器一对 veth pair
- veth 一端在容器 NS，另一端在 docker0
- 通过 NAT 实现外网访问
```

#### Mount Namespace

```
Mount Namespace 隔离：

每个 Mount Namespace 拥有独立的文件系统挂载点视图

宿主机挂载点:
/                  (rootfs)
├── /boot          (ext4)
├── /home          (ext4)
├── /mnt/data      (xfs)
├── /proc          (proc)
└── /sys           (sysfs)

容器 A 挂载点 (独立的 Mount NS):
/                  (overlay2 rootfs)
├── /bin           (容器镜像层)
├── /etc           (容器镜像层)
├── /proc          (proc, 容器 PID NS)
└── /sys           (sysfs, 容器化)

关键机制:
- pivot_root / chroot: 切换根文件系统
- overlay2: 分层文件系统 (联合挂载)
- /proc 和 /sys: 根据 PID/Net NS 虚拟化

overlay2 层次结构:
┌──────────────────────────┐
│  Container Layer (R/W)   │ ← 可写层
├──────────────────────────┤
│  Image Layer 3 (R/O)     │ ← apt install ...
├──────────────────────────┤
│  Image Layer 2 (R/O)     │ ← COPY app.jar
├──────────────────────────┤
│  Image Layer 1 (R/O)     │ ← FROM ubuntu:22.04
└──────────────────────────┘
```

#### UTS Namespace

```
UTS Namespace 隔离:

UTS (UNIX Time-Sharing) Namespace 隔离 hostname 和 domain name

宿主机: hostname = "prod-server-01"
容器 A:  hostname = "web-app-01"
容器 B:  hostname = "db-master"

每个容器可以有独立的主机名，不影响宿主机
```

#### IPC Namespace

```
IPC Namespace 隔离:

IPC Namespace 隔离 System V IPC 和 POSIX 消息队列

隔离的 IPC 资源:
- 共享内存段 (shmget/shmat)
- 信号量 (semget)
- 消息队列 (msgget)
- POSIX 共享内存 (/dev/shm)

容器 A 不能访问容器 B 的 IPC 资源
```

#### User Namespace

```
User Namespace 隔离:

User Namespace 隔离 UID/GID 空间

宿主机 UID:          容器内 UID:
UID 0 (root)    ←→   UID 1000 (容器内普通用户)
UID 1001        ←→   UID 0 (容器内 root)

映射关系 (/proc/PID/uid_map):
容器 UID 0 → 宿主 UID 100000, 范围 65536

安全意义:
- 容器内 root (UID 0) 映射到宿主非特权用户
- 即使容器逃逸，在宿主上也只有普通用户权限
- 称为 "rootless container" 的基础
```

#### Namespace 系统调用

```
Linux Namespace 系统调用:

创建新 Namespace:
  int clone(int (*fn)(void*), void* stack, int flags, void* arg);
  flags:
    CLONE_NEWPID    - 新 PID Namespace
    CLONE_NEWNET    - 新 Network Namespace
    CLONE_NEWNS     - 新 Mount Namespace
    CLONE_NEWUTS    - 新 UTS Namespace
    CLONE_NEWIPC    - 新 IPC Namespace
    CLONE_NEWUSER   - 新 User Namespace
    CLONE_NEWCGROUP - 新 Cgroup Namespace

加入已有 Namespace:
  int setns(int fd, int nstype);
  (通过 /proc/PID/ns/* 获取 fd)

在新 Namespace 中执行:
  int unshare(int flags);
  (当前进程脱离，创建新 Namespace)
```

### 30.5.3 cgroup 资源限制

**cgroup（Control Groups）** 是 Linux 内核提供的资源限制机制，用于控制进程组可以使用的系统资源量。

#### cgroup v1 vs v2

```
cgroup v1 架构:
- 每个资源控制器独立的层次结构
- 一个进程可以属于不同控制器的不同 cgroup
- 配置复杂，可能出现不一致

/sys/fs/cgroup/
├── cpu/                    ← CPU 控制器层次
│   ├── docker/
│   │   └── container-1/
│   │       └── cpu.shares
│   └── systemd/
├── memory/                 ← 内存控制器层次
│   ├── docker/
│   │   └── container-1/
│   │       └── memory.limit_in_bytes
│   └── systemd/
└── blkio/                  ← 块I/O控制器层次
    └── docker/
        └── container-1/
            └── blkio.weight

cgroup v2 架构:
- 统一的层次结构
- 所有控制器共享一棵树
- 配置更简洁，语义更一致

/sys/fs/cgroup/
└── system.slice/           ← 统一层次
    └── docker-xxx.scope/
        ├── cgroup.controllers  (cpu memory io ...)
        ├── cpu.max             (CPU 配额)
        ├── memory.max          (内存限制)
        └── io.max              (I/O 限制)
```

#### cgroup 资源控制器

| 控制器 | 功能 | 典型配置文件 |
|--------|------|------------|
| cpu | CPU 时间分配 | `cpu.max` (配额), `cpu.weight` (权重) |
| cpuset | CPU 核绑定 | `cpuset.cpus`, `cpuset.mems` |
| memory | 内存限制 | `memory.max`, `memory.high`, `memory.swap.max` |
| io | 块 I/O 限制 | `io.max` (BPS/IOPS 限制) |
| pids | 进程数限制 | `pids.max` |
| freezer | 冻结/解冻 | `cgroup.freeze` |
| hugetlb | 大页限制 | `hugetlb.2MB.max` |

```
cgroup v2 CPU 配额示例:

# 限制容器最多使用 0.5 核 (50ms / 100ms 周期)
echo "50000 100000" > /sys/fs/cgroup/container/cpu.max
#                  ↑ 周期 100ms
#           ↑ 配额 50ms

# 限制容器最多使用 2 核
echo "200000 100000" > /sys/fs/cgroup/container/cpu.max

cgroup v2 内存限制示例:

# 限制容器内存上限为 512MB
echo 536870912 > /sys/fs/cgroup/container/memory.max

# 设置内存高水位 (触发回收)
echo 268435456 > /sys/fs/cgroup/container/memory.high

# 限制 swap 使用
echo 0 > /sys/fs/cgroup/container/memory.swap.max
```

### 30.5.4 Docker 容器运行时

Docker 是最流行的容器平台，它的架构包括多个组件：

```
Docker 架构:

┌─────────────────────────────────────────────┐
│  用户空间工具                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ docker   │  │ docker   │  │ docker   │  │
│  │ build    │  │ run      │  │ push     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └──────────────┴─────────────┘        │
│                    │                         │
│              ┌─────┴─────┐                  │
│              │ Docker    │ (REST API)       │
│              │ Engine    │                  │
│              │ (dockerd) │                  │
│              └─────┬─────┘                  │
│                    │                         │
│              ┌─────┴─────┐                  │
│              │containerd │ (容器运行时)      │
│              └─────┬─────┘                  │
│                    │                         │
│              ┌─────┴─────┐                  │
│              │   runc    │ (OCI 运行时)     │
│              │  (实际创建容器)               │
│              └─────┬─────┘                  │
│                    │                         │
│              ┌─────┴─────┐                  │
│              │ Linux 内核                    │
│              │ Namespace │                  │
│              │ cgroup    │                  │
│              │ seccomp   │                  │
│              └───────────┘                  │
└─────────────────────────────────────────────┘

Docker 容器创建流程:
1. docker run → Docker Engine API
2. Docker Engine → containerd (创建容器配置)
3. containerd → runc (OCI spec)
4. runc:
   a. 创建 Namespace (clone)
   b. 配置 cgroup
   c. 设置 seccomp 过滤器
   d. 挂载 rootfs (overlay2)
   e. 切换根目录 (pivot_root)
   f. 执行容器进程 (execve)
```

### 30.5.5 容器安全

容器共享宿主内核，安全性是容器技术的关键挑战。

#### seccomp（Secure Computing Mode）

```
seccomp 系统调用过滤:

seccomp 限制容器可以使用的系统调用

Docker 默认 seccomp profile:
- 允许 ~300 个系统调用
- 禁止危险系统调用:
  - reboot (重启机器)
  - mount (挂载文件系统)
  - kexec_load (加载新内核)
  - ptrace (调试其他进程)
  - userfaultfd (用户态缺页处理)

BPF 过滤器工作原理:
┌──────────────────────────┐
│ seccomp-bpf Filter       │
│                          │
│ if (syscall == reboot)   │
│   → return EPERM         │
│ if (syscall == mount)    │
│   → return EPERM         │
│ if (syscall == open)     │
│   → 检查参数 → ALLOW/DENY│
│ ...                      │
│ else                     │
│   → return ALLOW         │
└──────────────────────────┘
```

#### AppArmor

```
AppArmor 强制访问控制:

AppArmor 基于路径的 MAC (Mandatory Access Control)

# 容器 AppArmor profile 示例
profile docker-container flags=(attach_disconnected) {
  # 允许读取 /etc/**
  /etc/** r,
  
  # 允许写入 /tmp/**
  /tmp/** rw,
  
  # 禁止访问 /proc/sysrq-trigger
  deny /proc/sysrq-trigger w,
  
  # 禁止加载内核模块
  deny capability sys_module,
  
  # 禁止挂载
  deny mount,
}

AppArmor 与 seccomp 的关系:
- seccomp: 过滤系统调用号 (哪些 syscall)
- AppArmor: 过滤文件和 capability 访问 (哪些文件, 哪些权限)
- 两者互补，构成纵深防御
```

#### SELinux

```
SELinux (Security-Enhanced Linux):

SELinux 基于标签的 MAC

两种模型:
- MLS (Multi-Level Security): 军事级安全
- TE (Type Enforcement): 类型强制

容器 SELinux 标签:
container_t     ← 容器进程的类型
container_file_t ← 容器文件的类型
container_runtime_t ← 运行时进程类型

规则示例:
# 允许容器进程读取容器文件
allow container_t container_file_t:file { read open };

# 禁止容器进程访问宿主文件
# (默认策略已拒绝，无需显式规则)

安全增强:
- 即使容器逃逸，SELinux 限制逃逸后的行为
- 每个容器有不同的 MCS (Multi-Category Security) 标签
- 防止容器间非授权访问
```

#### 容器安全最佳实践

```
容器安全层次:

┌──────────────────────────────────────┐
│ Layer 5: 供应链安全                    │
│ - 镜像签名与验证                       │
│ - 漏洞扫描 (Trivy, Clair)            │
│ - 最小基础镜像 (distroless, scratch) │
├──────────────────────────────────────┤
│ Layer 4: 运行时安全                    │
│ - seccomp 系统调用过滤                │
│ - AppArmor/SELinux 强制访问控制       │
│ - 只读根文件系统                       │
├──────────────────────────────────────┤
│ Layer 3: 内核隔离                      │
│ - User Namespace (rootless)          │
│ - 能力裁剪 (drop ALL + needed)       │
│ - cgroup 资源限制                     │
├──────────────────────────────────────┤
│ Layer 2: 容器运行时                    │
│ - gVisor (用户态内核)                 │
│ - Kata Containers (轻量级 VM)        │
│ - Firecracker (microVM)              │
├──────────────────────────────────────┤
│ Layer 1: 硬件隔离                     │
│ - 虚拟机 (完整隔离)                   │
│ - SGX/TDX (硬件可信执行环境)          │
└──────────────────────────────────────┘
```

---

## 面试高频考点

### 1. 全虚拟化 vs 半虚拟化 vs 硬件辅助虚拟化

**问**：请比较三种虚拟化方案的原理和优缺点。

**答**：
- **全虚拟化**：通过二进制翻译捕获敏感指令，Guest 无需修改但性能较差（代表：早期 VMware）
- **半虚拟化**：修改 Guest 内核，用 Hypercall 替代敏感指令，性能好但需改内核（代表：Xen）
- **硬件辅助虚拟化**：CPU 提供 VMX 模式，硬件自动处理敏感指令的 VM Exit，兼具兼容性和性能（代表：KVM + VT-x）

### 2. EPT 地址翻译过程

**问**：描述 EPT 的二级地址翻译过程，以及 TLB miss 时最坏情况的内存访问次数。

**答**：EPT 先用 Guest CR3 做 GVA→GPA 翻译（4 级页表），每一步 Guest PTE 的访问都需要通过 EPT 做 GPA→HPA 翻译（又是 4 级），最坏 4×4 + 4 + 1 = 21 次内存访问。通过 TLB 缓存 GVA→HPA 直接映射和使用大页（2MB/1GB）可显著减少。

### 3. VMCS 的作用

**问**：VMCS 包含哪些内容？VM Entry 和 VM Exit 分别做什么？

**答**：VMCS 包含 Guest State、Host State、VM-Execution Controls、VM-Exit/Entry Controls。VM Entry 加载 Guest 寄存器并切换到 Non-Root 模式；VM Exit 保存 Guest 状态、加载 Host 状态、记录 Exit Reason 并切换回 Root 模式。

### 4. 影子页表 vs EPT

**问**：为什么 EPT 优于影子页表？

**答**：影子页表是纯软件方案，Guest 每次修改页表都触发 VM Exit，维护开销大且实现复杂。EPT 由硬件自动完成两级翻译，VMM 只需管理 EPT 页表，Guest 修改自己的页表不会触发 VM Exit，简化了实现且减少了 VM Exit 次数。

### 5. 容器 vs 虚拟机

**问**：容器和虚拟机的核心区别是什么？容器为什么更快？

**答**：虚拟机通过 Hypervisor 虚拟完整硬件，每个 VM 运行独立内核；容器共享宿主内核，通过 Namespace 隔离资源视图 + cgroup 限制资源用量。容器快是因为无需启动完整 OS、无需硬件虚拟化开销、直接在宿主内核上运行进程。

### 6. Namespace 的作用

**问**：列举至少 4 种 Linux Namespace 并说明各自隔离什么。

**答**：
- **PID Namespace**：隔离进程 ID 空间，容器内 PID 从 1 开始
- **Network Namespace**：隔离网络接口、IP 地址、路由表
- **Mount Namespace**：隔离文件系统挂载点
- **UTS Namespace**：隔离主机名
- **IPC Namespace**：隔离 System V IPC / POSIX 消息队列
- **User Namespace**：隔离 UID/GID 映射
- **Cgroup Namespace**：隔离 cgroup 根目录视图

### 7. KVM 的设计哲学

**问**：KVM 为什么选择将 Linux 内核作为 Hypervisor？

**答**：KVM 复用 Linux 内核的 CPU 调度器、内存管理、设备驱动等成熟子系统，将 vCPU 映射为内核线程，避免重新实现这些复杂功能。通过硬件虚拟化扩展（VT-x/AMD-V），KVM 以极小的代码量实现了高效的虚拟化。QEMU 在用户态提供设备模拟，实现了关注点分离。

### 8. Docker 容器创建流程

**问**：描述 `docker run` 的底层实现过程。

**答**：docker CLI → Docker Engine REST API → containerd → runc。runc 使用 clone() 系统调用创建新 Namespace，配置 cgroup 资源限制，设置 seccomp 过滤器，挂载 overlay2 rootfs，执行 pivot_root 切换根目录，最后 execve 执行容器入口进程。

---

## 扩展阅读

1. **论文**: Popek & Goldberg, "Formal Requirements for Virtualizable Third Generation Architectures" (1974) — 虚拟化理论基础
2. **论文**: Robin & Irvine, "Analysis of the Intel Pentium's Ability to Support a Secure Virtual Machine Monitor" (2000) — x86 虚拟化问题分析
3. **KVM 文档**: https://www.kernel.org/doc/html/latest/virt/kvm/index.html
4. **Intel SDM Vol 3**: Chapter 23-33 — VT-x 硬件虚拟化完整规范
5. **virtio 规范**: https://docs.oasis-open.org/virtio/virtio/v1.2/virtio-v1.2.html
6. **Linux Namespace**: `man 7 namespaces`
7. **Docker 安全**: https://docs.docker.com/engine/security/
8. **Firecracker**: https://firecracker-microvm.github.io/ — AWS 开源的轻量级虚拟化
