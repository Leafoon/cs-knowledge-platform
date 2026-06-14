# Chapter 24: 网络安全基础 — 密码学与认证

> **学习目标**：
> - 理解安全目标 CIA 三元组（机密性/完整性/可用性）及其在协议设计中的体现
> - 掌握对称加密算法 DES/AES 的内部结构与分组密码工作模式（ECB/CBC/CTR/GCM）
> - 理解非对称加密 RSA 的数学原理与 Diffie-Hellman 密钥交换过程
> - 掌握哈希函数（SHA-256/MD5）的构造与三大安全特性
> - 理解消息认证码 HMAC 与数字签名的原理及应用场景
> - 掌握 PKI 证书体系、CA 层级、X.509 证书结构与证书链验证
> - 了解密钥管理的核心问题：HSM 硬件安全模块与密钥生命周期

---

## 24.1 安全目标与 CIA 三元组

### 24.1.1 什么是网络安全

在讨论任何安全技术之前，我们先回答一个根本问题：**安全的目标是什么？**

想象你要寄一封信。你希望：
1. 信的内容不被别人偷看（**机密性**）
2. 信的内容不被别人篡改（**完整性**）
3. 信能准时送到收件人手中（**可用性**）

这三个需求构成了信息安全的核心框架 —— **CIA 三元组（CIA Triad）**。

### 24.1.2 CIA 三元组详解

#### **机密性（Confidentiality）**

机密性确保信息只能被授权方访问。未授权方即使截获了数据，也无法理解其内容。

**类比**：你写日记时用只有自己知道的密码加密，即使日记本被别人拿到，他们也看不懂。

**技术手段**：对称加密（AES）、非对称加密（RSA）、传输加密（TLS）

#### **完整性（Integrity）**

完整性确保信息在传输或存储过程中未被篡改。任何微小的修改都能被检测到。

**类比**：银行支票上的金额被涂改时，水印和防伪标记能暴露篡改行为。

**技术手段**：哈希函数（SHA-256）、消息认证码（HMAC）、数字签名

#### **可用性（Availability）**

可用性确保授权用户在需要时能够访问信息和资源。

**类比**：图书馆在开放时间必须能借书，不能因为管理员休假就关门。

**技术手段**：负载均衡、DDoS 防御、冗余设计、灾备

### 24.1.3 扩展安全属性

除了 CIA，实际系统还需要考虑：

| 属性 | 英文 | 含义 | 典型威胁 |
|------|------|------|----------|
| 认证性 | Authentication | 验证通信方的身份 | 冒充攻击 |
| 不可否认性 | Non-repudiation | 发送方不能否认已发送的消息 | 抵赖 |
| 可审计性 | Accountability | 行为可追溯到具体实体 | 日志篡改 |
| 访问控制 | Access Control | 按权限控制资源访问 | 越权访问 |

<div data-component="CIATriad"></div>

### 24.1.4 攻击者模型

在设计安全系统前，我们需要明确攻击者的能力。经典的 **Dolev-Yao 模型** 假设攻击者：

- 能**截获**网络上的任何消息
- 能**解密**用已知密钥加密的消息
- 能**伪造**新的消息
- **不能**破解密码学原语本身（如不能因式分解大整数）

这意味着：安全协议必须在攻击者控制整个通信信道的前提下仍然安全。

---

## 24.2 对称加密

### 24.2.1 对称加密基本概念

**对称加密（Symmetric Encryption）** 是加密和解密使用**同一把密钥**的加密方式。

```
发送方：明文 P + 密钥 K → 密文 C = E(K, P)
接收方：密文 C + 密钥 K → 明文 P = D(K, C)
```

**类比**：你和朋友共用一把锁的钥匙。你用钥匙锁上箱子（加密），朋友用同一把钥匙打开箱子（解密）。

**核心问题**：双方如何安全地共享这把密钥？这就是著名的**密钥分发问题**。

对称加密分为两大类：
- **分组密码（Block Cipher）**：将明文分成固定大小的块，逐块加密（如 DES、AES）
- **流密码（Stream Cipher）**：逐比特/逐字节加密（如 RC4、ChaCha20）

### 24.2.2 DES — 数据加密标准

**DES（Data Encryption Standard）** 是 1977 年由 NIST 发布的对称加密标准，密钥长度 56 位，分组大小 64 位。虽然已不安全，但其 Feistel 结构是理解现代密码学的基础。

#### **DES 整体结构**

```
64位明文
    ↓
初始置换 IP (Initial Permutation)
    ↓
16轮 Feistel 网络 ← 48位子密钥 × 16轮
    ↓
逆初始置换 IP⁻¹
    ↓
64位密文
```

#### **初始置换 IP**

64 位明文首先经过一个固定的**初始置换（Initial Permutation, IP）**。这个置换只是重新排列比特位，不增加安全性，当年是为了让数据按字节对齐方便硬件实现。

IP 置换表（将第 i 位映射到第 IP[i] 位）：

```
58 50 42 34 26 18 10  2
60 52 44 36 28 20 12  4
62 54 46 38 30 22 14  6
64 56 48 40 32 24 16  8
57 49 41 33 25 17  9  1
59 51 43 35 27 19 11  3
61 53 45 37 29 21 13  5
63 55 47 39 31 23 15  7
```

**解读**：原明文的第 58 位变成输出的第 1 位，第 50 位变成第 2 位，以此类推。

#### **16 轮 Feistel 网络**

IP 之后，64 位数据被分为左右两半各 32 位：`L₀` 和 `R₀`。

每轮的 Feistel 运算：

```
Lᵢ = Rᵢ₋₁
Rᵢ = Lᵢ₋₁ ⊕ F(Rᵢ₋₁, Kᵢ)
```

其中 `F` 是轮函数，`Kᵢ` 是第 i 轮的 48 位子密钥，`⊕` 是异或运算。

**类比**：想象一个流水线工厂，每一轮都是一个工位。半成品（32 位右半部分）经过加工（F 函数）后，和另一半交换位置。

#### **轮函数 F 的详细结构**

`F(R, K)` 的计算过程：

```
32位 R
    ↓
扩展置换 E → 48位（将32位扩展到48位，部分位重复）
    ↓
与48位子密钥 K 异或
    ↓
8个 S 盒代换 → 每个S盒输入6位，输出4位（共32位）
    ↓
置换 P（固定排列）
    ↓
32位输出
```

#### **S 盒 — DES 的灵魂**

S 盒（Substitution Box）是 DES 中唯一的**非线性**组件，也是安全性的核心来源。DES 有 8 个 S 盒，每个 S 盒是一个 4 行 × 16 列的查找表。

**S₁ 盒示例**（8 个 S 盒之一）：

```
行\列  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
  0   14  4 13  1  2 15 11  8  3 10  6 12  5  9  0  7
  1    0 15  7  4 14  2 13  1 10  6 12 11  9  5  3  8
  2    4  1 14  8 13  6  2 11 15 12  9  7  3 10  5  0
  3   15 12  8  2  4  9  1  7  5 11  3 14 10  0  6 13
```

**查表规则**：6 位输入 `b₁b₂b₃b₄b₅b₆` 中：
- 行号 = `b₁b₆`（首尾两位，0~3）
- 列号 = `b₂b₃b₄b₅`（中间四位，0~15）

例如输入 `101100`：行号 = `10`₂ = 2，列号 = `0110`₂ = 6，查表得 `S₁[2][6] = 2`（即 `0010`）。

**为什么 S 盒是非线性的？** 如果把 S 盒看作一个函数 `f(x)`，那么 `f(a ⊕ b) ≠ f(a) ⊕ f(b)`。这种非线性性使得攻击者无法用线性方程组来分析密码。

#### **密钥调度 — 从 64 位到 16 个 48 位子密钥**

DES 的主密钥是 64 位，但其中 8 位是奇偶校验位，实际有效密钥只有 56 位。

```
64位主密钥
    ↓
PC-1 置换（去除8位校验位，打乱顺序）→ 56位
    ↓
分为 C₀ (28位) 和 D₀ (28位)
    ↓
16轮：每轮左移1或2位 → Cᵢ, Dᵢ
    ↓
PC-2 压缩置换（56位→48位）→ 子密钥 Kᵢ
```

左移位数表：轮 1,2,9,16 左移 1 位；其余轮左移 2 位。

#### **DES 安全性分析**

| 攻击方式 | 时间复杂度 | 说明 |
|----------|-----------|------|
| 暴力穷举 | 2⁵⁶ | 56 位密钥，1998 年 EFF 用 Deep Crack 破解 |
| 差分密码分析 | 2⁴⁷ | 需要 2⁴⁷ 选择明文对 |
| 线性密码分析 | 2⁴³ | 需要 2⁴³ 已知明文 |

> **💡 关键结论**：DES 的 56 位密钥长度在今天完全不安全。任何现代应用都应使用 AES-128 或更高强度的算法。

### 24.2.3 3DES — 三重 DES

为了解决 DES 密钥过短的问题，**3DES（Triple DES）** 使用三次 DES 运算：

```
C = E(K₃, D(K₂, E(K₃, P)))
```

注意中间是**解密**操作（EDE 模式），这样当 K₁=K₂=K₃ 时退化为普通 DES，保持向后兼容。

- **两密钥 3DES**：K₁=K₃，有效密钥长度 112 位
- **三密钥 3DES**：K₁, K₂, K₃ 独立，有效密钥长度 168 位

**缺点**：速度是 DES 的三分之一（三次运算），且分组大小仍为 64 位，容易受到生日攻击。

### 24.2.4 AES — 高级加密标准

**AES（Advanced Encryption Standard）** 是 2001 年 NIST 选定的 DES 替代算法，由比利时密码学家 Joan Daemen 和 Vincent Rijmen 设计（原名 Rijndael）。

AES 的参数：

| 参数 | AES-128 | AES-192 | AES-256 |
|------|---------|---------|---------|
| 密钥长度 | 128 位 | 192 位 | 256 位 |
| 分组大小 | 128 位 | 128 位 | 128 位 |
| 轮数 | 10 | 12 | 14 |

#### **AES 的状态矩阵**

AES 将 128 位数据组织为 **4×4 字节矩阵**（State），每个元素是 8 位（1 字节）：

```
输入字节：b₀b₁b₂...b₁₅

状态矩阵：
┌────┬────┬────┬────┐
│ b₀ │ b₄ │ b₈ │ b₁₂│
├────┼────┼────┼────┤
│ b₁ │ b₅ │ b₉ │ b₁₃│
├────┼────┼────┼────┤
│ b₂ │ b₆ │ b₁₀│ b₁₄│
├────┼────┼────┼────┤
│ b₃ │ b₇ │ b₁₁│ b₁₅│
└────┴────┴────┴────┘

注意：按列填充，不是按行！
```

#### **AES 轮函数 — 四步变换**

每一轮（最后一轮略有不同）包含四个操作：

```
State → SubBytes → ShiftRows → MixColumns → AddRoundKey → State'
                              （最后一轮省略 MixColumns）
```

**① SubBytes — 字节代换**

对状态矩阵中的每个字节，通过一个固定的 **S 盒** 进行替换。AES 的 S 盒是 16×16 的查找表（256 个条目）。

S 盒的构造过程：
1. 将字节看作 GF(2⁸) 上的元素
2. 求其乘法逆元（0 映射为 0）
3. 进行一个仿射变换

```
S 盒前几行（十六进制）：
63 7c 77 7b f2 6b 6f c5 30 01 67 2b fe d7 ab 76
ca 82 c9 7d fa 59 47 f0 ad d4 a2 af 9c a4 72 c0
...
```

**② ShiftRows — 行移位**

对状态矩阵的每一行进行循环左移：
- 第 0 行：不移位
- 第 1 行：左移 1 字节
- 第 2 行：左移 2 字节
- 第 3 行：左移 3 字节

```
移位前：          移位后：
a₀₀ a₀₁ a₀₂ a₀₃    a₀₀ a₀₁ a₀₂ a₀₃   （不移）
a₁₀ a₁₁ a₁₂ a₁₃ →  a₁₁ a₁₂ a₁₃ a₁₀   （左移1）
a₂₀ a₂₁ a₂₂ a₂₃    a₂₂ a₂₃ a₂₀ a₂₁   （左移2）
a₃₀ a₃₁ a₃₂ a₃₃    a₃₃ a₃₀ a₃₁ a₃₂   （左移3）
```

**③ MixColumns — 列混合**

对状态矩阵的每一列，乘以一个固定的多项式矩阵（在 GF(2⁸) 上）：

```
┌2 3 1 1┐   ┌a₀₀ a₀₁ a₀₂ a₀₃┐
│1 2 3 1│ × │a₁₀ a₁₁ a₁₂ a₁₃│
│1 1 2 3│   │a₂₀ a₂₁ a₂₂ a₂₃│
└3 1 1 2┘   └a₃₀ a₃₁ a₃₂ a₃₃┘
```

GF(2⁸) 上的乘法示例（不可约多项式 m(x) = x⁸ + x⁴ + x³ + x + 1）：

```
计算 2 × {57}：
{57} = 0101 0111
左移一位：1010 1110
最高位为1，所以异或 {1b}（即 m(x) 去掉最高位 = 0001 1011）
结果：1010 1110 ⊕ 0001 1011 = 1011 0101 = {b5}
```

**④ AddRoundKey — 轮密钥加**

将状态矩阵与轮密钥进行逐字节异或：

```
State'[i][j] = State[i][j] ⊕ RoundKey[i][j]
```

#### **AES 密钥扩展**

以 AES-128 为例，128 位主密钥扩展为 11 个 128 位轮密钥（共 176 字节）：

```
原始密钥：W[0] W[1] W[2] W[3]  （每个 32 位字）

扩展规则：
若 i 不是 4 的倍数：W[i] = W[i-4] ⊕ W[i-1]
若是 4 的倍数：W[i] = W[i-4] ⊕ T(W[i-1])

其中 T（变换函数）：
1. RotWord：循环左移 1 字节 [a,b,c,d] → [b,c,d,a]
2. SubWord：对每个字节查 S 盒
3. Rcon：异或轮常量（Rcon[i] = [xⁱ⁻¹, 0, 0, 0]，xⁱ⁻¹ 在 GF(2⁸) 中）
```

<div data-component="AESEncryptEngine"></div>

#### **AES 轮函数硬件实现**

在硬件（FPGA/ASIC）中实现 AES 轮函数，需要关注以下关键设计：

**S 盒实现方案对比**：

| 方案 | 原理 | 面积 | 速度 | 适用场景 |
|------|------|------|------|----------|
| 查找表（LUT） | ROM 存储 256 字节 | 较大 | 最快（1 周期） | FPGA 实现 |
| 组合逻辑 | GF(2⁸) 逆元 + 仿射变换 | 小 | 慢（多级门延迟） | 面积敏感场景 |
| 复合域 | GF(2⁴)² 分解 | 中等 | 较快 | ASIC 平衡方案 |

组合逻辑 S 盒的核心路径：
```
输入 8 位
    ↓
GF(2⁸) 求逆（组合逻辑）→ 需要多级 AND/XOR 门
    ↓
仿射变换（8 个 XOR 门）
    ↓
输出 8 位
```

**MixColumns 的 GF(2⁸) 乘法硬件**：

乘以 2 只需要左移一位再条件异或 `0x1B`（当最高位为 1 时）。乘以 3 = 乘以 2 再异或原值。因此 MixColumns 可以用简单的移位器和 XOR 门实现，不需要通用乘法器。

**流水线设计**：

为了提高吞吐量，可以将 10 轮 AES 展开为流水线：

```
┌──────┐  ┌──────┐  ┌──────┐       ┌──────┐
│Round0│→│Round1│→│Round2│→ ... →│Round10│
└──────┘  └──────┘  └──────┘       └──────┘
  Stage0    Stage1    Stage2         Stage10
```

- 每个流水级之间插入寄存器
- **吞吐量**：每周期处理一个 128 位分组（10 级流水线满载后）
- **延迟**：10 个时钟周期
- **代价**：面积 ×10（每级都需要独立的组合逻辑和寄存器）

对于资源受限的场景，也可以采用**轮折叠**架构：只用一轮的硬件，循环执行 10 次，面积最小但吞吐量降低 10 倍。

### 24.2.5 分组密码工作模式

单独的分组密码只能加密固定大小的数据块。**工作模式（Mode of Operation）** 定义了如何用分组密码处理任意长度的数据。

#### **ECB — 电子密码本模式**

```
明文：P₁ P₂ P₃ ... （每个 128 位）
       ↓    ↓    ↓
加密：E(K,P₁) E(K,P₂) E(K,P₃) ...
       ↓    ↓    ↓
密文：C₁  C₂  C₃  ...
```

- 相同的明文块总是产生相同的密文块
- **严重缺陷**：泄露数据模式！著名的"ECB 企鹅"图就是例证
- **不推荐**用于任何实际应用

#### **CBC — 密码块链接模式**

```
         IV        C₁        C₂
          ↓         ↓         ↓
明文 P₁ →⊕  明文 P₂ →⊕  明文 P₃ →⊕
          ↓         ↓         ↓
        E(K,·)    E(K,·)    E(K,·)
          ↓         ↓         ↓
        密文 C₁   密文 C₂   密文 C₃
```

- 每个明文块先与前一个密文块异或，再加密
- 第一个块与**初始化向量 IV** 异或
- 相同的明文块在不同位置会产生不同的密文
- **缺点**：加密必须串行（无法并行），解密可以并行

#### **CFB — 密码反馈模式**

```
         IV
          ↓
        E(K,·) → 与 P₁ 异或 → C₁
                    ↓
                  E(K,·) → 与 P₂ 异或 → C₂
```

- 将分组密码当作流密码使用
- 可以处理小于分组大小的数据

#### **OFB — 输出反馈模式**

```
         IV
          ↓
        E(K,·) → 密钥流 Z₁ → 与 P₁ 异或 → C₁
          ↓
        E(K,·) → 密钥流 Z₂ → 与 P₂ 异或 → C₂
```

- 密钥流独立于明文和密文
- 不会出现错误传播

#### **CTR — 计数器模式**

```
Nonce||0  Nonce||1  Nonce||2
    ↓        ↓        ↓
  E(K,·)   E(K,·)   E(K,·)
    ↓        ↓        ↓
  Z₁ ⊕P₁   Z₂ ⊕P₂   Z₃ ⊕P₃
    ↓        ↓        ↓
   C₁       C₂       C₃
```

- 用计数器产生密钥流：Nonce + 递增计数器
- **完全可并行**（加密和解密都可以）
- 随机访问：可以解密任意位置的块
- 现代协议的首选模式

#### **GCM — Galois/Counter 模式**

```
CTR 模式加密 + GHASH 认证
```

- GCM = CTR 加密 + GMAC 认证
- **同时提供加密和认证**（AEAD：Authenticated Encryption with Associated Data）
- TLS 1.3 中唯一必须支持的密码套件就是 `TLS_AES_128_GCM_SHA256`
- 认证标签（Tag）通常为 128 位

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

# AES-128-GCM 加密示例
key = AESGCM.generate_key(bit_length=128)
aesgcm = AESGCM(key)
nonce = os.urandom(12)  # 96 位 nonce，每个密钥下不能重复
plaintext = b"Hello, this is a secret message!"
aad = b"additional authenticated data"  # 认证但不加密的数据

# 加密：同时得到密文和认证标签
ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
print(f"密文: {ciphertext.hex()}")
print(f"长度: {len(ciphertext)} 字节（明文32字节 + 16字节标签）")

# 解密：验证认证标签，失败则抛出异常
try:
    decrypted = aesgcm.decrypt(nonce, ciphertext, aad)
    print(f"解密: {decrypted.decode()}")
except Exception as e:
    print(f"认证失败: {e}")
```

<div data-component="BlockCipherModes"></div>

### 24.2.6 C 语言实现 XOR 流密码

最简单的对称加密是 XOR 密码 —— 密钥流与明文逐字节异或：

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// 简单的伪随机数生成器作为密钥流（仅用于教学演示）
typedef struct {
    unsigned int state;
} prng_t;

void prng_init(prng_t *rng, unsigned int seed) {
    rng->state = seed;
}

unsigned char prng_next(prng_t *rng) {
    // 线性同余生成器（不安全，仅演示用）
    rng->state = rng->state * 1103515245 + 12345;
    return (unsigned char)((rng->state >> 16) & 0xFF);
}

// XOR 流密码加密/解密（对称操作）
void xor_cipher(const unsigned char *input, unsigned char *output,
                size_t len, unsigned int key) {
    prng_t rng;
    prng_init(&rng, key);
    for (size_t i = 0; i < len; i++) {
        output[i] = input[i] ^ prng_next(&rng);
    }
}

int main(void) {
    const char *plaintext = "Hello, Network Security!";
    size_t len = strlen(plaintext);
    unsigned char *ciphertext = malloc(len);
    unsigned char *decrypted = malloc(len);
    unsigned int key = 0xDEADBEEF;

    // 加密
    xor_cipher((const unsigned char *)plaintext, ciphertext, len, key);
    printf("明文:   %s\n", plaintext);
    printf("密文(hex): ");
    for (size_t i = 0; i < len; i++) {
        printf("%02x ", ciphertext[i]);
    }
    printf("\n");

    // 解密（同样的操作）
    xor_cipher(ciphertext, decrypted, len, key);
    decrypted[len] = '\0';
    printf("解密:   %s\n", decrypted);

    free(ciphertext);
    free(decrypted);
    return 0;
}
```

---

## 24.3 非对称加密

### 24.3.1 为什么需要非对称加密

对称加密有一个根本问题：**密钥分发**。Alice 和 Bob 要安全通信，必须先共享密钥；但共享密钥本身就需要安全信道 —— 这是一个鸡生蛋的问题。

**非对称加密（Asymmetric Encryption）** 使用一对密钥：
- **公钥（Public Key）**：公开发布，任何人可用
- **私钥（Private Key）**：自己保管，绝不泄露

```
加密：用接收方的公钥加密 → 只有接收方的私钥能解密
签名：用发送方的私钥签名 → 任何人可用公钥验证
```

**类比**：公钥就像一个邮箱的投信口（任何人都能投信），私钥就像邮箱的钥匙（只有主人能打开取信）。

### 24.3.2 RSA 算法

RSA 是 1977 年由 Rivest、Shamir、Adleman 发明的，是第一个广泛使用的非对称加密算法。其安全性基于**大整数因式分解的困难性**。

#### **RSA 密钥生成**

**步骤 1：选择两个大素数 p 和 q**

```
p = 61, q = 53  （实际应用中 p, q 各为 1024 位以上的大素数）
```

**步骤 2：计算 n = p × q**

```
n = 61 × 53 = 3233
n 的长度就是 RSA 密钥长度（如 2048 位）
```

**步骤 3：计算欧拉函数 φ(n) = (p-1)(q-1)**

```
φ(3233) = (61-1)(53-1) = 60 × 52 = 3120
```

> **💡 欧拉函数**：φ(n) 表示 1 到 n 中与 n 互质的正整数个数。当 n = p×q（p, q 为素数）时，φ(n) = (p-1)(q-1)。

**步骤 4：选择公钥指数 e**

要求：1 < e < φ(n)，且 gcd(e, φ(n)) = 1（互质）

```
选择 e = 17，验证 gcd(17, 3120) = 1 ✓
```

**步骤 5：计算私钥指数 d**

要求：d × e ≡ 1 (mod φ(n))，即 d 是 e 对 φ(n) 的模逆元

```
d × 17 ≡ 1 (mod 3120)
d = 2753  （因为 17 × 2753 = 46801 = 15 × 3120 + 1）
```

**最终密钥**：
- 公钥：(e, n) = (17, 3233)
- 私钥：(d, n) = (2753, 3233)

#### **RSA 加密与解密**

```
加密：C = Mᵉ mod n
解密：M = Cᵈ mod n
```

示例：加密消息 M = 65

```
加密：C = 65¹⁷ mod 3233 = 2790
解密：M = 2790²⁷⁵³ mod 3233 = 65 ✓
```

#### **RSA 正确性的数学证明**

为什么 `M^(e×d) mod n = M`？核心是**欧拉定理**：

```
欧拉定理：若 gcd(M, n) = 1，则 M^φ(n) ≡ 1 (mod n)

因为 e × d ≡ 1 (mod φ(n))
所以 e × d = k × φ(n) + 1（某个整数 k）

M^(e×d) = M^(k×φ(n)+1) = (M^φ(n))^k × M ≡ 1^k × M = M (mod n) ✓
```

#### **模幂运算的高效实现**

直接计算 `M^e mod n` 会溢出。使用**快速幂（Square-and-Multiply）**算法：

```python
def mod_pow(base, exp, mod):
    """快速模幂运算：base^exp mod mod"""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:          # exp 是奇数
            result = (result * base) % mod
        exp = exp >> 1            # exp /= 2
        base = (base * base) % mod  # base 平方
    return result

# 验证 RSA
print(mod_pow(65, 17, 3233))   # 2790（加密）
print(mod_pow(2790, 2753, 3233))  # 65（解密）
```

时间复杂度：O(log e) 次乘法，而非 O(e) 次。

<div data-component="RSAModularExponentiation"></div>

#### **RSA 大数运算的硬件实现**

实际 RSA 硬件（如 HSM 中的加密处理器）需要处理 2048 位或更长的大数，核心瓶颈是**模乘运算**。

**Montgomery 模乘算法**：

直接计算 `a × b mod n` 需要大数除法（代价极高）。Montgomery 算法将模运算转化为更高效的移位操作：

```
标准模乘：a × b mod n  （需要除法）
Montgomery 模乘：a × b × R⁻¹ mod n  （R = 2^k，用移位代替除法）
```

Montgomery 模乘的核心步骤：
```
输入：a, b, n（n 为 k 位奇数），R = 2^k
输出：a × b × R⁻¹ mod n

1. T = a × b          （普通大数乘法）
2. m = T × n' mod R   （n' = -n⁻¹ mod R，预计算）
3. t = (T + m × n) / R  （右移 k 位，整除保证）
4. if t ≥ n: t = t - n
5. return t
```

**流水线设计**：
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  大数乘法器  │ → │  加法+移位   │ → │  条件减法    │
│  (多级流水)  │   │  (Montgomery) │   │  (归约)     │
└─────────────┘   └─────────────┘   └─────────────┘
```

每个 2048 位模乘需要约 2048/64 = 32 个 64 位乘法周期（如果用 64 位乘法器阵列）。

**CRT 加速 RSA 解密**：

RSA 解密 `M = Cᵈ mod n` 中，d 通常与 n 同长（2048 位），计算很慢。利用中国剩余定理（CRT）可以加速约 4 倍：

```
标准解密：M = Cᵈ mod n  （一次 2048 位模幂）

CRT 加速：
dp = d mod (p-1)     （预计算，1024 位）
dq = d mod (q-1)     （预计算，1024 位）
Mp = Cᵈᵖ mod p       （1024 位模幂，快 4 倍）
Mq = Cᵈᵍ mod q       （1024 位模幂，快 4 倍）
h = (dp - dq) × q⁻¹ mod p
M = Mq + q × h       （CRT 合并）
```

因为模幂运算的复杂度是 O(k³)（k 为位数），将 2048 位分为两个 1024 位操作，总计算量约为 `2 × (1024/2048)³ = 2 × 1/8 = 1/4`，加速约 4 倍。

#### **C 语言实现 RSA 基础运算**

```c
#include <stdio.h>
#include <stdint.h>

// 快速模幂运算
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        exp >>= 1;
        base = (base * base) % mod;
    }
    return result;
}

// 扩展欧几里得算法求模逆元
int64_t mod_inverse(int64_t e, int64_t phi) {
    int64_t old_r = e, r = phi;
    int64_t old_s = 1, s = 0;
    while (r != 0) {
        int64_t q = old_r / r;
        int64_t temp_r = r;
        r = old_r - q * r;
        old_r = temp_r;
        int64_t temp_s = s;
        s = old_s - q * s;
        old_s = temp_s;
    }
    // old_r 应为 1（e 和 phi 互质）
    if (old_r != 1) return -1;  // 逆元不存在
    return (old_s % phi + phi) % phi;
}

int main(void) {
    // 小素数演示（实际应用需要大素数）
    uint64_t p = 61, q = 53;
    uint64_t n = p * q;           // 3233
    uint64_t phi = (p-1) * (q-1); // 3120
    uint64_t e = 17;
    int64_t d = mod_inverse(e, phi);  // 2753

    printf("公钥 (e, n): (%lu, %lu)\n", e, n);
    printf("私钥 (d, n): (%ld, %lu)\n", d, n);

    // 加密和解密
    uint64_t msg = 65;
    uint64_t cipher = mod_pow(msg, e, n);
    uint64_t decrypted = mod_pow(cipher, (uint64_t)d, n);
    printf("明文: %lu\n", msg);
    printf("密文: %lu\n", cipher);
    printf("解密: %lu\n", decrypted);

    return 0;
}
```

### 24.3.3 Diffie-Hellman 密钥交换

RSA 可以加密，但另一个问题是：两个从未见过面的人如何建立共享密钥？**Diffie-Hellman（DH）密钥交换** 解决了这个问题。

#### **DH 协议过程**

```
公开参数：素数 p，生成元 g

Alice                              Bob
─────                              ───
选择私密 a                          选择私密 b
计算 A = gᵃ mod p                  计算 B = gᵇ mod p
         ─── A ───→
         ←── B ───
计算 s = Bᵃ mod p                  计算 s = Aᵇ mod p
   = (gᵇ)ᵃ mod p                    = (gᵃ)ᵇ mod p
   = gᵃᵇ mod p                      = gᵃᵇ mod p  ← 相同！
```

**安全性**：攻击者知道 p、g、A、B，但要计算 a 或 b 需要解**离散对数问题**（在大素数域上目前无高效算法）。

#### **DH 交换示例**

```
公开参数：p = 23, g = 5

Alice: a = 6,  A = 5⁶ mod 23 = 15625 mod 23 = 8
Bob:   b = 15, B = 5¹⁵ mod 23 = 30517578125 mod 23 = 19

Alice 计算共享密钥：s = 19⁶ mod 23 = 2
Bob 计算共享密钥：s = 8¹⁵ mod 23 = 2  ✓
```

#### **ECDH — 椭圆曲线 Diffie-Hellman**

传统 DH 使用有限域上的指数运算，ECDH 使用椭圆曲线上的点乘运算：

```
公开参数：椭圆曲线 E，基点 G，阶 n

Alice: 私钥 a，公钥 A = aG（点乘）
Bob:   私钥 b，公钥 B = bG

共享密钥：S = aB = b(ab)G = a(bG) = abG
```

ECDH 的优势：相同安全强度下，密钥长度短得多（256 位 ECC ≈ 3072 位 RSA）。

### 24.3.4 椭圆曲线密码学简介

**椭圆曲线**的一般形式（在有限域 GF(p) 上）：

```
y² = x³ + ax + b  (mod p)
```

其中 p 是大素数，a、b 是曲线参数，且 `4a³ + 27b² ≠ 0`（避免奇异曲线）。

**点加法几何意义**：
1. 取曲线上两点 P 和 Q
2. 画一条直线通过 P 和 Q
3. 直线与曲线交于第三点 R'
4. R' 关于 x 轴的对称点就是 P + Q

**点乘（标量乘法）**：`nP = P + P + ... + P`（n 次），用 Double-and-Add 算法高效计算（类似快速幂）。

```
计算 11P：
11 = 1011₂
步骤：P → 2P → 4P → 5P (=4P+P) → 10P → 11P (=10P+P)
```

---

## 24.4 哈希函数

### 24.4.1 哈希函数的基本概念

**哈希函数（Hash Function）** 将任意长度的输入映射为固定长度的输出（摘要/指纹）。

```
H: {0,1}* → {0,1}^n
任意长度输入 → 固定长度输出（如 SHA-256 输出 256 位）
```

**类比**：哈希函数就像人的指纹 —— 不同的人有不同的指纹（唯一性），不能从指纹反推出人的长相（不可逆），任何微小的改变都会产生完全不同的"指纹"。

### 24.4.2 哈希函数的三大安全特性

#### **抗原像性（Pre-image Resistance）**

给定哈希值 h，**难以**找到消息 m 使得 H(m) = h。

```
已知：H(m) = 0x3a7bd3e2...
求解：m = ?（计算上不可行）
```

**意义**：保证哈希函数是单向的，不能从摘要恢复原始数据。

#### **抗第二原像性（Second Pre-image Resistance）**

给定消息 m₁，**难以**找到另一个消息 m₂ ≠ m₁ 使得 H(m₁) = H(m₂)。

```
已知：m₁ = "Hello"，H(m₁) = 0x3a7bd3e2...
求解：m₂ = ? 使得 H(m₂) = 0x3a7bd3e2...（计算上不可行）
```

**意义**：防止对已知消息的伪造。

#### **抗碰撞性（Collision Resistance）**

**难以**找到任意两个不同的消息 m₁ ≠ m₂ 使得 H(m₁) = H(m₂)。

```
求解：m₁, m₂（m₁ ≠ m₂）使得 H(m₁) = H(m₂)
```

**注意**：碰撞必然存在（输入空间无限，输出空间有限），但必须在计算上**难以找到**。

**生日悖论**：对于 n 位哈希值，大约需要 2^(n/2) 次尝试就能找到碰撞。因此：
- MD5（128 位）：约 2⁶⁴ 次 → 已不安全
- SHA-256（256 位）：约 2¹²⁸ 次 → 安全

### 24.4.3 SHA-256 算法详解

SHA-256 是 SHA-2 家族的一员，输出 256 位（32 字节）哈希值，被广泛用于 TLS、比特币、数字签名等。

#### **SHA-256 处理流程**

```
输入消息 M
    ↓
① 消息填充（Padding）
   - 末尾添加 1 位 '1'
   - 添加 k 个 '0'，使得总长度 ≡ 448 mod 512
   - 添加 64 位原始消息长度
    ↓
② 分块（每块 512 位）
    ↓
③ 对每个块进行 64 轮压缩
    ↓
④ 输出 256 位哈希值
```

#### **SHA-256 压缩函数**

初始哈希值 H₀ 是前 8 个素数的平方根小数部分的前 32 位：

```
H₀ = 6a09e667, H₁ = bb67ae85, H₂ = 3c6ef372, H₃ = a54ff53a
H₄ = 510e527f, H₅ = 9b05688c, H₆ = 1f83d9ab, H₇ = 5be0cd19
```

每轮使用的常量 Kᵢ 是前 64 个素数的立方根小数部分的前 32 位。

对每个 512 位消息块：

```
① 消息调度：将 16 个 32 位字扩展为 64 个
   Wᵢ = σ₁(Wᵢ₋₂) + Wᵢ₋₇ + σ₀(Wᵢ₋₁₅) + Wᵢ₋₁₆   (i = 16..63)

② 初始化工作变量：a,b,c,d,e,f,g,h = 当前哈希值

③ 64 轮压缩：
   for i = 0 to 63:
       T₁ = h + Σ₁(e) + Ch(e,f,g) + Kᵢ + Wᵢ
       T₂ = Σ₀(a) + Maj(a,b,c)
       h = g
       g = f
       f = e
       e = d + T₁
       d = c
       c = b
       b = a
       a = T₁ + T₂

④ 更新哈希值：Hᵢ = Hᵢ₋₁ + (a,b,c,d,e,f,g,h)
```

其中：
```
Ch(x,y,z) = (x AND y) XOR (NOT x AND z)    — 选择函数
Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z) — 多数函数
Σ₀(x) = ROTR²(x) XOR ROTR¹³(x) XOR ROTR²²(x)
Σ₁(x) = ROTR⁶(x) XOR ROTR¹¹(x) XOR ROTR²⁵(x)
σ₀(x) = ROTR⁷(x) XOR ROTR¹⁸(x) XOR SHR³(x)
σ₁(x) = ROTR¹⁷(x) XOR ROTR¹⁹(x) XOR SHR¹⁰(x)
```

### 24.4.4 MD5 的碰撞攻击

MD5（Message Digest 5）输出 128 位哈希值，由 Ron Rivest 于 1991 年设计。

**MD5 已被彻底破解**：

| 年份 | 事件 |
|------|------|
| 2004 | 王小云团队宣布 MD5 碰撞攻击，复杂度 2⁶⁹ |
| 2007 | Stevens 等在普通 PC 上几秒内生成 MD5 碰撞 |
| 2008 | 利用 MD5 碰撞伪造了合法的 SSL 证书 |
| 2012 | Flame 恶意软件利用 MD5 碰撞伪造微软代码签名证书 |

**碰撞示例**（概念演示）：

攻击者构造两个不同的 PDF 文件 A 和 B，使得 MD5(A) = MD5(B)。然后用 A 获取合法签名，再将签名应用到 B 上 —— 因为哈希值相同，签名仍然有效。

> **💡 教训**：永远不要在安全场景中使用 MD5。SHA-256 是当前的推荐选择。

```python
import hashlib

# SHA-256 和 MD5 对比
data = b"Hello, Network Security!"

md5_hash = hashlib.md5(data).hexdigest()
sha256_hash = hashlib.sha256(data).hexdigest()

print(f"MD5    ({len(md5_hash)*4}位): {md5_hash}")
print(f"SHA-256({len(sha256_hash)*4}位): {sha256_hash}")

# 雪崩效应：改变一个比特，哈希值完全不同
data2 = b"Iello, Network Security!"  # H→I
sha256_hash2 = hashlib.sha256(data2).hexdigest()
print(f"SHA-256(改1字节): {sha256_hash2}")
print(f"变化位数: {sum(a != b for a, b in zip(sha256_hash, sha256_hash2))}")
```

### 24.4.5 C 语言实现简化哈希骨架

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// 简化的哈希函数演示（非安全实现，仅教学用）
// 使用简单的 DJB2 哈希算法
uint32_t simple_hash(const uint8_t *data, size_t len) {
    uint32_t hash = 5381;  // 初始值
    for (size_t i = 0; i < len; i++) {
        // hash * 33 + data[i]（位运算优化）
        hash = ((hash << 5) + hash) + data[i];
    }
    return hash;
}

// SHA-256 轮函数中使用的辅助函数（位操作演示）
static inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

int main(void) {
    const char *msg1 = "Hello, Network Security!";
    const char *msg2 = "Hello, Network Security?";  // 仅末尾不同

    uint32_t h1 = simple_hash((const uint8_t *)msg1, strlen(msg1));
    uint32_t h2 = simple_hash((const uint8_t *)msg2, strlen(msg2));

    printf("消息1: \"%s\"\n", msg1);
    printf("哈希1: 0x%08x\n", h1);
    printf("消息2: \"%s\"\n", msg2);
    printf("哈希2: 0x%08x\n", h2);
    printf("碰撞: %s\n", h1 == h2 ? "是" : "否");

    return 0;
}
```

---

## 24.5 消息认证码 — HMAC

### 24.5.1 为什么需要 MAC

哈希函数本身不提供认证 —— 任何人都可以计算 `H(m)`。如果 Alice 发送 `(m, H(m))` 给 Bob，攻击者可以篡改消息并重新计算哈希值。

**消息认证码（Message Authentication Code, MAC）** 引入密钥来解决这个问题：

```
发送方：MAC = F(K, m)  → 发送 (m, MAC)
接收方：验证 F(K, m) == MAC
```

只有拥有密钥的人才能生成正确的 MAC。

### 24.5.2 HMAC 的构造

**HMAC（Hash-based MAC）** 是使用哈希函数构造 MAC 的标准方法（RFC 2104）：

```
HMAC(K, m) = H((K' ⊕ opad) || H((K' ⊕ ipad) || m))
```

其中：
- K' 是密钥 K 填充到哈希函数块大小的结果
- `ipad = 0x36` 重复（块大小字节数）
- `opad = 0x5C` 重复（块大小字节数）

```
         K' ⊕ ipad     K' ⊕ opad
              ↓              ↓
消息 m → 内层哈希 → 外层哈希 → HMAC 输出
```

**为什么不直接用 H(K || m)？** 因为直接拼接会导致**长度扩展攻击** —— 攻击者可以在不知道 K 的情况下，利用 H(K || m) 计算 H(K || m || padding || m')。HMAC 的双层结构避免了这个问题。

```python
import hmac
import hashlib

# HMAC-SHA256 示例
key = b"my-secret-key-32bytes-long!!!!!!"
message = b"Important financial transaction: pay $1000"

# 生成 HMAC
mac = hmac.new(key, message, hashlib.sha256).hexdigest()
print(f"HMAC-SHA256: {mac}")

# 验证 HMAC（使用 compare_digest 防止时序攻击）
received_mac = mac
expected_mac = hmac.new(key, message, hashlib.sha256).hexdigest()
is_valid = hmac.compare_digest(received_mac, expected_mac)
print(f"验证结果: {'通过' if is_valid else '失败'}")

# 篡改消息后验证失败
tampered = b"Important financial transaction: pay $9999"
tampered_mac = hmac.new(key, tampered, hashlib.sha256).hexdigest()
print(f"篡改后验证: {hmac.compare_digest(tampered_mac, expected_mac)}")
```

<div data-component="HMACConstruction"></div>

---

## 24.6 数字签名

### 24.6.1 数字签名的概念

**数字签名（Digital Signature）** 是非对称加密的另一个核心应用，提供：
- **认证性**：确认消息确实来自声称的发送方
- **完整性**：确认消息未被篡改
- **不可否认性**：发送方不能否认已签名的消息

```
签名过程（发送方）：
  私钥 + 消息 → 签名值

验证过程（接收方）：
  公钥 + 消息 + 签名值 → 验证通过/失败
```

**类比**：手写签名附在文件上，但数字签名远比手写签名安全 —— 它与消息内容绑定，复制到其他消息上无效。

### 24.6.2 RSA 数字签名

RSA 签名直接使用私钥"解密"消息的哈希值：

```
签名：σ = H(m)ᵈ mod n  （用私钥 d 对哈希值做"解密"运算）
验证：H(m) = σᵉ mod n  （用公钥 e "加密"签名，与实际哈希比较）
```

```python
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# 生成 RSA 密钥对
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)
public_key = private_key.public_key()

# 签名
message = b"This is a legally binding document."
signature = private_key.sign(
    message,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH,
    ),
    hashes.SHA256(),
)
print(f"签名长度: {len(signature)} 字节")

# 验证
try:
    public_key.verify(
        signature,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    print("签名验证: 通过 ✓")
except Exception:
    print("签名验证: 失败 ✗")

# 篡改消息后验证失败
try:
    public_key.verify(
        signature,
        b"This is a FORGED document.",
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
except Exception:
    print("篡改后验证: 失败 ✗")
```

### 24.6.3 DSA 与 ECDSA

#### **DSA — 数字签名算法**

DSA 是 NIST 于 1991 年发布的签名标准，安全性基于**离散对数问题**。

```
参数生成：
- 选择素数 p（2048 位）、q（256 位，q | p-1）
- g = h^((p-1)/q) mod p（h 为任意生成元）

签名：
- 选择随机 k（1 < k < q）
- r = (gᵏ mod p) mod q
- s = k⁻¹(H(m) + x·r) mod q  （x 为私钥）
- 签名 = (r, s)

验证：
- w = s⁻¹ mod q
- u₁ = H(m)·w mod q
- u₂ = r·w mod q
- v = (gᵘ¹ · yᵘ² mod p) mod q  （y = gˣ mod p 为公钥）
- 接受当且仅当 v = r
```

#### **ECDSA — 椭圆曲线数字签名算法**

ECDSA 将 DSA 移植到椭圆曲线上，是目前最广泛使用的签名算法（TLS、比特币、以太坊等）。

```
签名：
- 选择随机 k
- 计算 (x₁, y₁) = kG
- r = x₁ mod n
- s = k⁻¹(H(m) + d·r) mod n  （d 为私钥）
- 签名 = (r, s)

验证：
- w = s⁻¹ mod n
- u₁ = H(m)·w mod n
- u₂ = r·w mod n
- (x₁, y₁) = u₁G + u₂Q  （Q = dG 为公钥）
- 接受当且仅当 x₁ mod n = r
```

> **⚠️ 重大警告**：ECDSA 的 k（随机数）**必须**是密码学安全的随机数，且**不能重复使用**。2010 年索尼 PS3 的 ECDSA 实现使用了固定的 k，导致私钥被直接计算出来。

---

## 24.7 PKI 体系 — 公钥基础设施

### 24.7.1 为什么需要 PKI

非对称加密有一个关键问题：**如何确认公钥真的属于你认为的那个人？**

如果 Alice 想给 Bob 发加密邮件，她需要 Bob 的公钥。但攻击者 Mallory 可能伪装成 Bob，把自己的公钥发给 Alice。这样 Alice 加密的消息就会被 Mallory 读取 —— 这就是**中间人攻击**。

**PKI（Public Key Infrastructure，公钥基础设施）** 通过**证书**和**证书颁发机构（CA）** 来解决公钥的信任问题。

**类比**：你的身份证就是一种"证书"。政府（CA）颁发身份证，验证你的身份信息。别人看到你的身份证，因为信任政府，所以信任你的身份。

### 24.7.2 CA 证书颁发机构

**CA（Certificate Authority）** 是受信任的第三方机构，负责：
1. 验证申请者的身份
2. 用 CA 的私钥签发证书
3. 维护证书吊销列表（CRL）
4. 提供在线证书状态查询（OCSP）

#### **CA 层级结构**

```
根 CA（Root CA）
├── 签发中间 CA 1（Intermediate CA）
│   ├── 签发终端实体证书（如 google.com）
│   └── 签发终端实体证书（如 facebook.com）
├── 签发中间 CA 2
│   └── ...
└── 签发中间 CA 3
    └── ...
```

**为什么需要层级？**
- 根 CA 的私钥最敏感（离线存储在物理保险柜中）
- 日常签发由中间 CA 完成
- 如果中间 CA 私钥泄露，只需吊销该中间 CA，不影响根 CA

**全球知名根 CA**：
- DigiCert、Let's Encrypt、Sectigo（原 Comodo）、GlobalSign
- 浏览器和操作系统内置了约 100-150 个根 CA 证书

### 24.7.3 X.509 证书结构

**X.509** 是最广泛使用的证书标准。每个证书包含以下字段：

```
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 0x04:e1:3a:...
    Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, O=Let's Encrypt, CN=R3
        Validity:
            Not Before: Jan  1 00:00:00 2024 GMT
            Not After : Mar 31 23:59:59 2024 GMT
        Subject: CN=www.example.com
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus: 00:b4:...
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Subject Alternative Name:
                DNS:www.example.com, DNS:example.com
            X509v3 Key Usage: Digital Signature, Key Encipherment
    Signature Algorithm: sha256WithRSAEncryption
         3a:4b:5c:...
```

| 字段 | 含义 |
|------|------|
| Version | X.509 版本（通常是 v3） |
| Serial Number | 证书序列号（CA 唯一标识） |
| Issuer | 颁发者（CA 的可分辨名称） |
| Validity | 有效期（Not Before / Not After） |
| Subject | 证书持有者信息 |
| Subject Public Key Info | 持有者的公钥 |
| Extensions | 扩展字段（SAN、Key Usage 等） |
| Signature | CA 对上述所有内容的数字签名 |

```python
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime

# 生成自签名证书
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CS2 Course"),
    x509.NameAttribute(NameOID.COMMON_NAME, "example.cs2.edu"),
])

cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(private_key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
    .add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("example.cs2.edu"),
            x509.DNSName("www.example.cs2.edu"),
        ]),
        critical=False,
    )
    .sign(private_key, hashes.SHA256())
)

# 输出证书信息
print(f"主题: {cert.subject}")
print(f"颁发者: {cert.issuer}")
print(f"序列号: {cert.serial_number}")
print(f"有效期: {cert.not_valid_before} ~ {cert.not_valid_after}")
print(f"公钥长度: {cert.public_key().key_size} 位")
print(f"签名算法: {cert.signature_algorithm_oid._name}")
```

### 24.7.4 证书链验证

当你访问 `https://www.google.com` 时，浏览器需要验证 Google 的证书链：

```
终端证书（www.google.com）
    ↑ 签发
中间 CA 证书（GTS CA 1C3）
    ↑ 签发
根 CA 证书（GlobalSign Root CA）← 浏览器内置信任
```

**验证过程**：

```
① 获取服务器证书和中间 CA 证书
② 从终端证书开始，逐级向上验证签名
③ 每一级验证：
   a. 用上级 CA 的公钥验证当前证书的签名
   b. 检查证书有效期
   c. 检查证书是否被吊销（CRL/OCSP）
   d. 检查证书用途（是否允许用于 TLS 服务器）
④ 直到到达根 CA，且根 CA 在本地信任库中
⑤ 验证域名是否匹配（CN 或 SAN 字段）
```

<div data-component="CertificateValidator"></div>

### 24.7.5 证书吊销 — CRL 与 OCSP

证书可能在有效期内需要被吊销（如私钥泄露、域名所有权变更）。

#### **CRL — 证书吊销列表**

```
CA 定期发布 CRL（Certificate Revocation List）
包含所有被吊销证书的序列号和吊销时间
客户端下载 CRL 并本地检查
```

**缺点**：
- CRL 文件可能很大（几十 MB）
- 更新有延迟（通常每天或每周发布一次）
- 客户端需要下载整个列表

#### **OCSP — 在线证书状态协议**

```
客户端发送证书序列号给 OCSP 服务器
OCSP 服务器实时返回：Good / Revoked / Unknown
```

**优点**：实时性好，不需要下载整个列表。
**缺点**：增加了一次网络往返，且 OCSP 服务器可能成为隐私泄露源（知道用户访问了哪些网站）。

#### **OCSP Stapling**

```
服务器定期查询自己的 OCSP 状态
将 OCSP 响应"钉"在 TLS 握手中一起发送
客户端无需单独查询 OCSP 服务器
```

这是目前最推荐的方案，兼顾了实时性和隐私。

---

## 24.8 密钥管理

### 24.8.1 密钥管理的重要性

> **Kerckhoffs 原则**：密码系统的安全性应该依赖于密钥的保密，而不是算法的保密。

这意味着密钥管理是整个安全体系中最关键的环节。据统计，大部分安全事件不是因为密码算法被破解，而是因为密钥管理不当。

### 24.8.2 密钥生命周期

密钥从生成到销毁经历以下阶段：

```
生成 → 分发 → 使用 → 存储 → 轮换 → 归档 → 销毁
```

| 阶段 | 关键要求 |
|------|----------|
| 生成 | 使用密码学安全的随机数生成器（CSPRNG） |
| 分发 | 通过安全信道或密钥协商协议 |
| 使用 | 限制使用次数和有效期 |
| 存储 | 加密存储，最小权限原则 |
| 轮换 | 定期更换密钥，限制单密钥加密的数据量 |
| 归档 | 安全保存旧密钥以解密历史数据 |
| 销毁 | 彻底删除，不可恢复 |

### 24.8.3 HSM — 硬件安全模块

**HSM（Hardware Security Module）** 是专门用于密钥管理和加密运算的物理设备。

**核心特性**：
- 密钥**永远不离开** HSM 的安全边界
- 防篡改设计（物理破坏检测）
- 高性能加密运算加速
- FIPS 140-2/140-3 认证

```
┌─────────────────────────────────┐
│           HSM 设备               │
│  ┌───────────┐  ┌───────────┐  │
│  │ 密钥存储   │  │ 加密引擎   │  │
│  │ (加密NVRAM)│  │ (RSA/AES) │  │
│  └───────────┘  └───────────┘  │
│  ┌───────────┐  ┌───────────┐  │
│  │ 访问控制   │  │ 审计日志   │  │
│  │ (策略引擎) │  │ (防篡改)  │  │
│  └───────────┘  └───────────┘  │
│  ┌─────────────────────────┐   │
│  │    防篡改检测电路         │   │
│  │  (开盖自毁、温度检测)     │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
         ↑ API (PKCS#11)
┌─────────────────────────────────┐
│        应用服务器                │
│  调用 HSM API 进行加密运算       │
│  密钥不出 HSM，只返回结果        │
└─────────────────────────────────┘
```

**典型应用场景**：
- CA 签发证书（根 CA 私钥存储在 HSM 中）
- 银行交易签名
- 数据库加密主密钥管理
- 云 KMS（Key Management Service）后端

### 24.8.4 密钥分发架构

<div data-component="KeyManagementSystem"></div>

**密钥分发的几种模式**：

#### **预共享密钥（PSK）**

```
双方事先通过安全信道交换密钥
适用于：小型网络、设备配对
缺点：扩展性差，N 个用户需要 N(N-1)/2 个密钥
```

#### **KDC — 密钥分发中心**

```
Kerberos 协议模型：
Alice ←→ KDC ←→ Bob
KDC 与每个用户共享一个长期密钥
按需生成会话密钥分发给通信双方
```

#### **公钥分发**

```
用非对称加密分发对称密钥
例如：TLS 握手中用 RSA/ECDH 协商会话密钥
```

#### **密钥层次结构**

```
主密钥（Master Key，存储在 HSM 中）
├── 密钥加密密钥 KEK₁
│   ├── 数据密钥 DK₁
│   └── 数据密钥 DK₂
├── 密钥加密密钥 KEK₂
│   ├── 数据密钥 DK₃
│   └── 数据密钥 DK₄
└── ...
```

- 主密钥保护 KEK
- KEK 保护数据密钥
- 数据密钥直接加密用户数据
- 轮换时只需更换数据密钥，不影响主密钥

```python
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

# 密钥派生：从密码生成强密钥
password = b"user-password-123"
salt = os.urandom(16)

kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,        # 派生 256 位密钥
    salt=salt,
    iterations=480000, # OWASP 推荐迭代次数
)
derived_key = kdf.derive(password)
print(f"派生密钥: {derived_key.hex()}")
print(f"盐值: {salt.hex()}")

# 使用派生密钥加密
aesgcm = AESGCM(derived_key)
nonce = os.urandom(12)
plaintext = b"Sensitive data encrypted with derived key"
ciphertext = aesgcm.encrypt(nonce, plaintext, None)
print(f"密文: {ciphertext.hex()}")
```

### 24.8.5 C 语言实现 AES SubBytes 位运算

```c
#include <stdio.h>
#include <stdint.h>

// AES S 盒（完整 256 字节查找表）
static const uint8_t sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,
    0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,
    0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,
    0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,
    0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,
    0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,
    0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,
    0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,
    0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,
    0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,
    0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,
    0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,
    0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,
    0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,
    0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,
    0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,
    0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
};

// AES SubBytes — S 盒查表
void sub_bytes(uint8_t state[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] = sbox[state[i][j]];
        }
    }
}

// 用位运算实现 GF(2⁸) 求逆（组合逻辑方式，教学用）
uint8_t gf256_inverse(uint8_t a) {
    if (a == 0) return 0;
    // 用扩展欧几里得算法求 GF(2⁸) 乘法逆元
    uint8_t t0, t1, t2, t3;
    // a^2
    t0 = ((a & 0x80) ? (a << 1) ^ 0x1b : a << 1);
    // a^4
    t1 = ((t0 & 0x80) ? (t0 << 1) ^ 0x1b : t0 << 1);
    // a^8
    t2 = ((t1 & 0x80) ? (t1 << 1) ^ 0x1b : t1 << 1);
    // a^16
    t3 = ((t2 & 0x80) ? (t2 << 1) ^ 0x1b : t2 << 1);
    // a^254 = a^2 * a^4 * a^8 * a^16 * a^32 * a^64 * a^128
    // 254 = 2+4+8+16+32+64+128
    uint8_t result = a;  // a^1
    // 利用费马小定理：a^255 = 1，所以 a^254 = a^(-1)
    // 通过重复平方和乘法计算 a^254
    uint8_t powers[8];
    powers[0] = a;
    for (int i = 1; i < 8; i++) {
        uint8_t prev = powers[i-1];
        powers[i] = ((prev & 0x80) ? (prev << 1) ^ 0x1b : prev << 1);
    }
    // 254 = 0b11111110
    result = powers[1]; // a^2
    for (int i = 2; i < 8; i++) {
        result ^= powers[i]; // a^2+4+8+16+32+64+128 = a^254
    }
    return result;
}

// ShiftRows 操作
void shift_rows(uint8_t state[4][4]) {
    uint8_t temp;
    // 第1行左移1位
    temp = state[1][0];
    state[1][0] = state[1][1];
    state[1][1] = state[1][2];
    state[1][2] = state[1][3];
    state[1][3] = temp;
    // 第2行左移2位
    temp = state[2][0];
    state[2][0] = state[2][2];
    state[2][2] = temp;
    temp = state[2][1];
    state[2][1] = state[2][3];
    state[2][3] = temp;
    // 第3行左移3位（等价于右移1位）
    temp = state[3][3];
    state[3][3] = state[3][2];
    state[3][2] = state[3][1];
    state[3][1] = state[3][0];
    state[3][0] = temp;
}

// 打印状态矩阵
void print_state(const uint8_t state[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%02x ", state[i][j]);
        }
        printf("\n");
    }
}

int main(void) {
    // 示例状态矩阵（列优先填充）
    // 明文: 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
    uint8_t state[4][4] = {
        {0x00, 0x44, 0x88, 0xcc},
        {0x11, 0x55, 0x99, 0xdd},
        {0x22, 0x66, 0xaa, 0xee},
        {0x33, 0x77, 0xbb, 0xff},
    };

    printf("原始状态:\n");
    print_state(state);

    // SubBytes
    sub_bytes(state);
    printf("\nSubBytes 后:\n");
    print_state(state);

    // ShiftRows
    shift_rows(state);
    printf("\nShiftRows 后:\n");
    print_state(state);

    // GF(2⁸) 求逆验证
    printf("\nGF(2^8) 求逆验证:\n");
    uint8_t test_val = 0x53;
    uint8_t inv = gf256_inverse(test_val);
    printf("0x%02x 的逆元: 0x%02x\n", test_val, inv);
    // 验证: 0x53 * 逆元 应该等于 1
    // 简单验证: 查 S 盒的逆
    printf("查表验证 (S盒[0x%02x]): 0x%02x\n", test_val, sbox[test_val]);

    return 0;
}
```

---

## 24.9 综合应用 — 安全通信的完整流程

### 24.9.1 一个安全通信的完整设计

让我们把本章学到的所有技术组合起来，设计一个完整的安全通信系统：

```
Alice 想要安全地给 Bob 发送消息，需要满足：
✓ 机密性：只有 Bob 能读
✓ 完整性：消息不被篡改
✓ 认证性：确认是 Alice 发的
✓ 不可否认性：Alice 不能否认
```

**步骤**：

```
① Alice 计算消息的哈希值
   digest = SHA-256(m)

② Alice 用自己的私钥签名
   signature = Sign(Alice_privkey, digest)

③ Alice 生成随机会话密钥
   session_key = random(256 bits)

④ Alice 用会话密钥加密（消息+签名）
   ciphertext = AES-GCM(session_key, m || signature)

⑤ Alice 用 Bob 的公钥加密会话密钥
   encrypted_key = RSA-Encrypt(Bob_pubkey, session_key)

⑥ Alice 发送 (encrypted_key, ciphertext) 给 Bob

⑦ Bob 用自己的私钥解密会话密钥
   session_key = RSA-Decrypt(Bob_privkey, encrypted_key)

⑧ Bob 用会话密钥解密得到消息和签名
   m, signature = AES-GCM-Decrypt(session_key, ciphertext)

⑨ Bob 用 Alice 的公钥验证签名
   Verify(Alice_pubkey, SHA-256(m), signature)
```

这就是 **混合加密系统** —— 结合了对称加密的高效性和非对称加密的密钥管理便利性。TLS 协议的核心思想正是如此。

### 24.9.2 Python 完整示例

```python
import os
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

class SecureChannel:
    """简化的安全通信信道（教学演示）"""

    def __init__(self):
        # 每方生成自己的 RSA 密钥对
        self.private_key = rsa.generate_private_key(65537, 2048)
        self.public_key = self.private_key.public_key()

    def send(self, message: bytes, recipient_public_key) -> dict:
        """发送方：加密+签名"""
        # ① 计算消息摘要
        digest = hashlib.sha256(message).digest()

        # ② 签名
        signature = self.private_key.sign(
            digest,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        # ③ 生成会话密钥
        session_key = AESGCM.generate_key(bit_length=256)

        # ④ AES-GCM 加密消息和签名
        aesgcm = AESGCM(session_key)
        nonce = os.urandom(12)
        plaintext = message + b"||SIG||" + signature
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # ⑤ RSA 加密会话密钥
        encrypted_key = recipient_public_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        return {
            "encrypted_key": encrypted_key,
            "nonce": nonce,
            "ciphertext": ciphertext,
        }

    def receive(self, package: dict, sender_public_key) -> bytes:
        """接收方：解密+验证"""
        # ⑦ RSA 解密会话密钥
        session_key = self.private_key.decrypt(
            package["encrypted_key"],
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # ⑧ AES-GCM 解密
        aesgcm = AESGCM(session_key)
        plaintext = aesgcm.decrypt(package["nonce"], package["ciphertext"], None)

        # 分离消息和签名
        parts = plaintext.split(b"||SIG||")
        message = parts[0]
        signature = parts[1]

        # ⑨ 验证签名
        digest = hashlib.sha256(message).digest()
        sender_public_key.verify(
            signature,
            digest,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return message


# 演示
alice = SecureChannel()
bob = SecureChannel()

# Alice 发送给 Bob
msg = b"Transfer $1000 to Bob's account"
package = alice.send(msg, bob.public_key)
print(f"加密传输: {len(package['ciphertext'])} 字节密文")

# Bob 接收并验证
received = bob.receive(package, alice.public_key)
print(f"解密消息: {received.decode()}")
print("签名验证: 通过 ✓")
```

---

## 24.10 本章小结

### 核心知识点回顾

| 主题 | 关键概念 | 安全性基础 |
|------|----------|-----------|
| 对称加密 | AES（SubBytes/ShiftRows/MixColumns/AddRoundKey） | 密钥保密 |
| 工作模式 | GCM = CTR 加密 + GHASH 认证 | Nonce 唯一 |
| RSA | C = Mᵉ mod n，φ(n) = (p-1)(q-1) | 大整数分解困难 |
| DH 密钥交换 | s = gᵃᵇ mod p | 离散对数困难 |
| ECC | y² = x³ + ax + b | 椭圆曲线离散对数困难 |
| SHA-256 | 64 轮压缩，Ch/Maj/Σ 函数 | 抗碰撞性 |
| HMAC | H((K'⊕opad) ‖ H((K'⊕ipad) ‖ m)) | 密钥 + 哈希 |
| 数字签名 | σ = H(m)ᵈ mod n | 私钥唯一性 |
| PKI | CA 层级 + X.509 证书链 | 信任锚 |
| 密钥管理 | HSM + 密钥层次结构 | 物理安全 |

### 安全设计原则

1. **Kerckhoffs 原则**：安全性依赖密钥，不依赖算法保密
2. **纵深防御**：多层安全机制，一层被破不影响全局
3. **最小权限**：每个组件只拥有完成任务所需的最小权限
4. **默认安全**：安全配置应是默认选项，而非可选项

### 常见错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 使用 ECB 模式 | 泄露数据模式 | 使用 GCM |
| MD5 用于安全场景 | 碰撞攻击 | 使用 SHA-256 |
| RSA 密钥太短 | 被因式分解 | 至少 2048 位 |
| ECDSA 重复使用 k | 私钥泄露 | 每次用 CSPRNG 生成 |
| 硬编码密钥 | 源码泄露即密钥泄露 | 使用 HSM/KMS |
| 忽略证书验证 | 中间人攻击 | 严格验证证书链 |

---

## 24.11 练习题

### 基础题

**题目 1**：RSA 密钥生成中，取 p=7, q=11，计算 n, φ(n)，并选择 e=13，求私钥 d。

**题目 2**：解释为什么 ECB 模式不适合加密图像文件，并说明 CBC 模式如何解决这个问题。

**题目 3**：SHA-256 的抗碰撞性意味着什么？为什么生日攻击将安全强度从 256 位降到 128 位？

### 进阶题

**题目 4**：设计一个协议，让 Alice 和 Bob 在不安全信道上建立共享密钥，然后用该密钥进行加密通信。要求：
- 使用 DH 密钥交换建立共享密钥
- 使用 AES-GCM 加密数据
- 使用 HMAC 验证完整性

**题目 5**：分析以下场景中的安全漏洞，并提出修复方案：
- 一个 Web 应用使用 `MD5(password)` 存储用户密码
- 一个 IoT 设备使用硬编码的 AES 密钥
- 一个 API 使用 `HMAC(key, timestamp)` 作为认证令牌

### 思考题

**题目 6**：量子计算对现有密码体系有什么影响？了解后量子密码学（Post-Quantum Cryptography）的发展方向。

**题目 7**：为什么 TLS 1.3 移除了 RSA 密钥交换（静态 RSA），只保留了 (EC)DHE 密钥交换？

**题目 8**：对比 HSM 和 TPM（Trusted Platform Module）的设计目标和应用场景。
