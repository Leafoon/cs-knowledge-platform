# Chapter 35: 摊销分析（Amortized Analysis）

> **学习目标**：掌握聚合法、记账法、势能法三种摊销分析方法，能选择合适方法构造完整证明；能用势能法对动态表扩容和二进制计数器进行严格分析；理解摊销分析与平均情况分析的本质区别；理解 Splay 树和 Fibonacci 堆的摊销复杂度来源。

---

## 章节导读

在算法分析中，我们常常遇到这样一种现象：**某些操作偶尔会非常慢，但大多数时候都很快**。

典型的例子是 Python 的 `list.append()`：绝大多数时候是 $O(1)$（只是向列表末尾写入一个元素），但每隔一段时间，当容量满时会触发**扩容**，将所有元素复制到一个更大的数组，代价为 $O(n)$。

如果我们说这个操作"最坏情况 $O(n)$"，是不是太悲观了？毕竟每次 $O(n)$ 的扩容发生之后，要经历很长一段时间的 $O(1)$ 操作才会再次发生扩容。

**摊销分析（Amortized Analysis）** 正是用来表达这种"平均下来其实挺快"的数学工具。

| 分析方法 | 关心的问题 | 适用场合 |
|---|---|---|
| **最坏情况分析** | 单次操作最坏代价 | 实时系统（不允许任何一次慢） |
| **平均情况分析** | 随机输入的期望代价 | 需要概率假设，输入分布已知 |
| **摊销分析** | **操作序列总代价 / 操作数** | 数据结构设计（保证长期高效） |

> **关键区别**：摊销分析针对**最坏情况输入序列**，不假设任何概率分布 —— 这比平均情况分析更强、更具实际保障。

本章配备 4 个交互式组件，分别直观展示代价曲线、势能骤降、三种证明方式对比和 Splay 旋转势能变化。

---

## 35.1 摊销分析的三种方法

摊销分析有三种等价但视角不同的表达方式。我们先用一个统一的例子——**动态表（Dynamic Array）的 n 次 push_back** ——来感受三者的差异，然后再分节详述。

### 35.1.1 聚合法（Aggregate Analysis）：总量除以次数

**核心思路**：计算 $n$ 次操作的总代价 $T(n)$，摊销代价 = $T(n)/n$。

**优点**：直觉最自然，一句话说完。  
**缺点**：不同操作混在一起时，分配不公平（所有操作摊到相同的代价）。

> **比喻**：你的手机每100次充电有一次要充很久（快充没电了），其余都只要几分钟。把所有充电时间加起来，除以100次，得到"平均每次充电时间" —— 这就是聚合法的精髓。

**形式化**：
$$\hat{c}_i = \frac{T(n)}{n}$$
其中 $T(n) = \sum_{i=1}^{n} c_i$ 为所有实际代价之和，$\hat{c}_i$ 是摊销代价。

**聚合法的核心**：找到整个操作序列的**总代价上界**，然后平均。

### 35.1.2 记账法（Accounting Method）：提前收"信用"

**核心思路**：为每个操作人为指定一个摊销代价 $\hat{c}_i$（可以 ≠ 实际代价 $c_i$）。规则如下：
- 若 $\hat{c}_i > c_i$：多收的部分存入"信用账户"（Credit）
- 若 $\hat{c}_i < c_i$：从账户中支取信用来支付差额
- **约束**：任何时刻账户余额 ≥ 0（不允许透支）

**保证**：若账户余额始终 ≥ 0，则 $\sum \hat{c}_i \geq \sum c_i$，即摊销代价是实际代价的合法上界。

> **比喻**：你每个月强制存 1000 块钱到"汽车维修基金"（即使这个月车没坏）。偶尔大修时，从基金里取钱。只要基金不亏空，你每月的"总支出"（实际花销+存款）就是真实维修成本的上界。

**关键技巧**：选择好"每次操作的信用分配方案"，使得高代价操作之前积累了足够的余额。

### 35.1.3 势能法（Potential Method）：定义"内在能量"

**核心思路**：定义一个"势能函数"$\Phi(D)$，衡量数据结构内部状态 $D$ 的"蓄积能量"。摊销代价定义为：

$$\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = c_i + \Delta\Phi_i$$

其中 $D_i$ 是第 $i$ 次操作后的状态。

**总摊销代价与总实际代价的关系**：
$$\sum_{i=1}^{n} \hat{c}_i = \sum_{i=1}^{n} c_i + \Phi(D_n) - \Phi(D_0)$$

只要 $\Phi(D_n) \geq \Phi(D_0)$（通常令 $\Phi(D_0) = 0$ 且 $\Phi \geq 0$ 恒成立），则总摊销代价 ≥ 总实际代价。

> **比喻**：弹簧玩具。每次压缩弹簧（廉价操作）都在积累弹性势能（$\Delta\Phi > 0$）；放开弹簧（昂贵操作）时势能骤降（$\Delta\Phi < 0$），相当于势能补贴了这次昂贵操作的代价。摊销代价 = 弹簧操作代价 + 势能变化量，始终保持"平稳"。

**设计势能函数的指导原则**：
1. $\Phi(D_0) = 0$（初始无积累）
2. $\Phi(D_i) \geq 0$（不能透支）
3. 廉价操作让 $\Phi$ 升高（积累信用）
4. 昂贵操作让 $\Phi$ 大幅下降（释放信用），使摊销代价仍然小

### 35.1.4 三种方法的等价性

三种方法本质上是同一件事的不同视角：

| 方法 | 对应概念 | 余额/势能如何流动 |
|---|---|---|
| 聚合法 | 粗粒度，直接均摊 | 不关心个体操作 |
| 记账法 | 信用（Credit） | 显式追踪账户余额 |
| 势能法 | 势能（Potential） | 隐式编码为数据结构状态函数 |

实际上，记账法中的"当前信用余额"恰好等于势能法中的 $\Phi(D_i) - \Phi(D_0)$。三者总是给出相同的摊销上界（只是表达形式不同）。

### 35.1.5 摊销 vs 平均：不要混淆！

这是最常见的误解之一。

**平均情况分析（Average-Case Analysis）**：
- 假设输入来自某个概率分布（如随机排列）
- 期望代价 = 对随机输入取平均
- 对最坏情况输入**没有保证**

**摊销分析（Amortized Analysis）**：
- 针对**最坏情况输入序列**（对手最恶意的操作顺序）
- n 次操作总代价的确定性上界，除以 n
- 不需要概率假设，**比平均情况更强**

举例：快速排序期望 $O(n \log n)$（平均情况），但最坏 $O(n^2)$。动态数组 push_back 摊销 $O(1)$（在任何操作序列下，总代价除以总操作数 ≤ $O(1)$）。

---

## 35.2 动态表（Dynamic Table）的摊销分析

### 35.2.1 问题描述：倍增扩容策略

**动态表**是支持 push_back（追加元素）的表，内部用固定大小数组实现。当数组满时，分配一个**两倍大**的新数组，将所有旧元素复制过去（这一步代价为 $O(\text{当前大小})$），然后插入新元素。

```python
class DynamicArray:
    """
    支持动态扩容的数组（类似 Python list 或 C++ vector）
    扩容因子 = 2（每次容量翻倍）
    """
    def __init__(self):
        self.data = [None]   # 初始容量为 1
        self.size = 0        # 当前元素数量
        self.capacity = 1    # 当前数组容量

    def push_back(self, val):
        # 满了就扩容
        if self.size == self.capacity:
            # 分配 2 倍容量的新数组，复制所有元素
            # 这一步代价为 O(self.size)
            new_data = [None] * (self.capacity * 2)
            for i in range(self.size):
                new_data[i] = self.data[i]
            self.data = new_data
            self.capacity *= 2
        # 插入元素，代价 O(1)
        self.data[self.size] = val
        self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError("out of range")
        return self.data[idx]

    def load_factor(self):
        return self.size / self.capacity
```

```cpp
#include <cassert>
#include <cstddef>
#include <utility>

template<typename T>
class DynamicArray {
public:
    DynamicArray() : data_(new T[1]), size_(0), cap_(1) {}
    ~DynamicArray() { delete[] data_; }

    void push_back(const T& val) {
        if (size_ == cap_) {
            // 倍增扩容：代价 O(size_)
            T* new_data = new T[cap_ * 2];
            for (std::size_t i = 0; i < size_; i++)
                new_data[i] = data_[i];   // 复制旧元素
            delete[] data_;
            data_ = new_data;
            cap_ *= 2;
        }
        data_[size_++] = val;   // O(1) 插入
    }

    std::size_t size()     const { return size_; }
    std::size_t capacity() const { return cap_; }
    double load_factor()   const { return (double)size_ / cap_; }
    T& operator[](std::size_t i) { assert(i < size_); return data_[i]; }

private:
    T* data_;
    std::size_t size_, cap_;
};
```

### 35.2.2 聚合法证明：n 次追加总代价 < 3n

**实际代价分析**：第 $i$ 次 push_back 的代价 $c_i$ 为：

$$c_i = \begin{cases} i & \text{若第 } i \text{ 次引发扩容（即 } i-1 \text{ 是 2 的幂）} \\ 1 & \text{否则（普通插入）} \end{cases}$$

**总实际代价**（n 次 push_back）：

只有在 $i = 1, 2, 4, 8, 16, \ldots$ 时才发生扩容，扩容代价分别为 $1, 2, 4, 8, \ldots$

$$T(n) = n + \underbrace{\sum_{j=0}^{\lfloor \log_2 n \rfloor} 2^j}_{\text{所有扩容复制的总代价}} < n + 2n = 3n$$

因此：

$$\text{单次摊销代价} = \frac{T(n)}{n} < \frac{3n}{n} = 3 = O(1)$$

**直观理解**：扩容发生的频率越来越低（1, 2, 4, 8, … 次后才发生），虽然每次越来越贵，但"被摊销的操作数"也越来越多。

<div data-component="DynamicTableGrowthAmortized"></div>

### 35.2.3 势能法证明：$\Phi = 2 \times \text{size} - \text{capacity}$

**势能函数**：
$$\Phi(D) = 2 \times \text{size} - \text{capacity}$$

**验证基本性质**：
- 初始状态：$\text{size}=0, \text{capacity}=1$，$\Phi_0 = 2 \times 0 - 1 = -1$  
  > 注意：我们通常令初始 $\Phi_0 = 0$，可改为 $\Phi = 2 \times \text{size} - \text{capacity} + 1$ 保证初始为 0。这里为简洁用经典形式，初始势能 $\leq 0$ 视为从 "负债" 起步（不影响上界证明，因为 $\Phi$ 在扩容后非负）。

更常见的表述是仅对满足 $\text{capacity} \leq 2 \times \text{size}$ 的状态有效（扩容后立刻满足）。**下面用标准版本**：令初始 capacity=1, size=0，Φ₀=0（约定），分析两种情形：

**情形 1：普通插入（size < capacity 时）**

实际代价 $c_i = 1$（只写入一个元素），插入后 size 增 1，capacity 不变：
$$\Delta\Phi = 2(s+1) - k - (2s - k) = 2$$
摊销代价 $\hat{c}_i = 1 + 2 = 3$

**情形 2：触发扩容（size = capacity = k 时）**

实际代价 $c_i = k + 1$（复制 k 个元素 + 插入 1 个），扩容后 size = k+1，capacity = 2k：
$$\Phi_{\text{after}} = 2(k+1) - 2k = 2$$
$$\Phi_{\text{before}} = 2k - k = k$$
$$\Delta\Phi = 2 - k$$
摊销代价 $\hat{c}_i = (k+1) + (2-k) = 3$

**结论**：无论哪种情形，摊销代价 $\hat{c}_i = 3 = O(1)$。

> **势能函数的设计直觉**：$\Phi = 2 \times \text{size} - \text{capacity}$ 在数组满时等于 $\text{capacity}$（积累了足够的"能量"），扩容后骤降到 2（刚花掉了所有积累的势能来支付复制代价）。

<div data-component="PotentialMethodVisualizer"></div>

### 35.2.4 缩容策略与装载因子

只做扩容的动态表空间利用率可能低至 50%。为了节省内存，可以在**装载因子 < 阈值**时触发缩容。

**关键设计选择**：

| 缩容阈值 | 摊销代价 | 原因 |
|---|---|---|
| $\text{size} < \text{capacity}/2$（即 50%） | $O(n)$ —— **退化！** | push + pop 交替时每次都触发扩/缩 |
| $\text{size} < \text{capacity}/4$（即 25%） | $O(1)$ —— 安全 | 缩容后留有"缓冲"，短期内不会再次扩容 |

**正确策略（25% 阈值）**：当 size < capacity/4 时，将 capacity 减半（缩至 capacity/2）。

```python
class DynamicArrayWithShrink:
    """带缩容的动态数组：扩容阈值 100%，缩容阈值 25%"""
    def __init__(self):
        self.data = [None] * 4
        self.size = 0
        self.capacity = 4        # 初始容量 ≥ 4 避免频繁缩容

    def _resize(self, new_cap):
        """通用扩/缩容实现"""
        new_data = [None] * new_cap
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data
        self.capacity = new_cap

    def push_back(self, val):
        if self.size == self.capacity:
            self._resize(self.capacity * 2)   # 扩容：×2
        self.data[self.size] = val
        self.size += 1

    def pop_back(self):
        if self.size == 0:
            raise IndexError("empty")
        val = self.data[self.size - 1]
        self.data[self.size - 1] = None   # 避免内存泄漏
        self.size -= 1
        # 缩容阈值：25%（注意保留最小容量避免归零）
        if self.size > 0 and self.size <= self.capacity // 4 and self.capacity > 4:
            self._resize(self.capacity // 2)  # 缩至一半
        return val
```

```cpp
template<typename T>
class DynamicArrayShrink {
public:
    DynamicArrayShrink() : data_(new T[4]), size_(0), cap_(4) {}
    ~DynamicArrayShrink() { delete[] data_; }

    void push_back(const T& val) {
        if (size_ == cap_) resize(cap_ * 2);
        data_[size_++] = val;
    }

    T pop_back() {
        assert(size_ > 0);
        T val = data_[--size_];
        // 缩容阈值：25%；保留最小容量 4 防止无限缩
        if (size_ > 0 && size_ <= cap_ / 4 && cap_ > 4)
            resize(cap_ / 2);
        return val;
    }

private:
    void resize(std::size_t new_cap) {
        T* nd = new T[new_cap];
        for (std::size_t i = 0; i < size_; i++) nd[i] = data_[i];
        delete[] data_;
        data_ = nd;
        cap_ = new_cap;
    }
    T* data_;
    std::size_t size_, cap_;
};
```

**摊销证明（带缩容）**：定义势能函数（满足 $\Phi \geq 0$ 的分段函数）：

$$\Phi(D) = \begin{cases} 2 \times \text{size} - \text{capacity} & \text{若 size} \geq \text{capacity}/2 \text{（半满以上）} \\ \text{capacity}/2 - \text{size} & \text{若 size} < \text{capacity}/2 \text{（半满以下）} \end{cases}$$

可以验证，对 push_back 和 pop_back 所有情形，摊销代价均为 $O(1)$（证明留作练习）。

### 35.2.5 实际应用：Python list 与 C++ vector 的增长因子

| 实现 | 增长因子 | 摊销代价 | 说明 |
|---|---|---|---|
| CPython `list` | ≈1.125（渐进） | $O(1)$ | 公式: new_cap = old_cap + old_cap/8 + 6 |
| GCC `std::vector` | 2 | $O(1)$ | 标准倍增 |
| MSVC `std::vector` | 1.5 | $O(1)$ | 更节省内存 |
| Java `ArrayList` | 1.5 | $O(1)$ | new_cap = old_cap + old_cap/2 |

> **为什么不用加固定大小 $c$（如每次 +100）？**  
> 若每次追加固定 $c$ 个槽，则第 $k$ 次扩容触发时已有 $kc$ 个元素，复制 $kc$ 代价。$n$ 次操作总扩容代价 $= c + 2c + 3c + \ldots + \frac{n}{c} \cdot c = O(n^2/c) = O(n^2)$。摊销代价 $O(n)$，远不如倍增的 $O(1)$！

---

## 35.3 二进制计数器的摊销分析

### 35.3.1 问题描述：INCREMENT 操作

一个 $k$ 位二进制计数器，存储在数组 $A[0..k-1]$ 中（$A[0]$ 是最低位），初始全为 0。每次 INCREMENT 操作使计数器加 1。

**INCREMENT 的实现**：从最低位开始翻转，直到遇到一个 0：

```python
def increment(A: list[int]) -> None:
    """
    二进制计数器 +1 操作。
    A[0] 是最低有效位（LSB），A[k-1] 是最高有效位（MSB）。
    
    时间复杂度：最坏 O(k)（当所有位都是 1 时，如 0111...1 -> 1000...0）
    例如：0011 + 1 = 0100（翻转2位），1111 + 1 = 0000（翻转4位）
    """
    i = 0
    k = len(A)
    # 从低位开始：将 1 翻转为 0（进位），直到遇到 0 停止
    while i < k and A[i] == 1:
        A[i] = 0   # 清除进位位
        i += 1
    # i < k 时：将首个 0 翻转为 1（吸收进位）
    # i == k 时：溢出（计数器归零）
    if i < k:
        A[i] = 1
```

```cpp
#include <vector>

void increment(std::vector<int>& A) {
    /*
     * 二进制计数器 +1。
     * A[0] = LSB (最低位)，A[k-1] = MSB (最高位)
     * 最坏情形：A 全为 1，翻转 k 位，代价 O(k)
     * 算法原理：模拟二进制加法的进位链
     */
    std::size_t i = 0;
    while (i < A.size() && A[i] == 1) {
        A[i] = 0;   // 进位：清零当前位
        i++;
    }
    if (i < A.size()) A[i] = 1;   // 将第一个 0 置为 1
    // 若 i == A.size()：计数器溢出，所有位变 0（正确行为，自然回卷）
}
```

**最坏情况**：计数器全为 1 时（如 `1111`），INCREMENT 需翻转 $k$ 位，代价 $O(k)$。如果每次都是最坏情况，$n$ 次操作代价 $O(nk)$。但这不可能每次都发生！

### 35.3.2 聚合法：总翻转次数 < 2n

**观察**：在 $n$ 次 INCREMENT 中，各位翻转的频率是：

- $A[0]$（最低位）：每次都翻转 → 翻转 $n$ 次
- $A[1]$（次低位）：每2次翻转1次 → 翻转 $\lfloor n/2 \rfloor$ 次
- $A[2]$：每4次翻转1次 → 翻转 $\lfloor n/4 \rfloor$ 次
- $A[j]$：翻转 $\lfloor n / 2^j \rfloor$ 次

**总翻转次数**（即总代价）：
$$T(n) = \sum_{j=0}^{k-1} \left\lfloor \frac{n}{2^j} \right\rfloor < n \sum_{j=0}^{\infty} \frac{1}{2^j} = 2n$$

因此摊销代价 $= T(n)/n < 2 = O(1)$。

> **直觉**：高位翻转的频率以几何级数衰减，总代价的主要贡献来自低位（最低位翻 $n$ 次），高位的贡献越来越小，总和收敛到 $2n$。

```python
def simulate_counter(n: int, k: int) -> dict:
    """
    模拟 n 次 INCREMENT，统计每一位的翻转次数和总代价。
    验证总翻转次数 < 2n 的聚合法结论。
    """
    A = [0] * k
    flip_count = [0] * k   # 每一位翻转的次数
    total_cost = 0

    for step in range(n):
        i = 0
        while i < k and A[i] == 1:
            A[i] = 0
            flip_count[i] += 1
            total_cost += 1
            i += 1
        if i < k:
            A[i] = 1
            flip_count[i] += 1
            total_cost += 1

    return {
        "total_flips": total_cost,
        "per_bit": flip_count,
        "amortized_per_op": total_cost / n
    }

# 验证：
result = simulate_counter(1000, 16)
print(f"总翻转次数: {result['total_flips']} (上界 2n = {2000})")
print(f"摊销代价: {result['amortized_per_op']:.4f}")
```

```cpp
#include <vector>
#include <iostream>

struct CounterStats {
    long long total_flips;
    std::vector<long long> per_bit;
    double amortized;
};

CounterStats simulate_counter(int n, int k) {
    std::vector<int> A(k, 0);
    std::vector<long long> per_bit(k, 0);
    long long total = 0;
    for (int step = 0; step < n; ++step) {
        int i = 0;
        while (i < k && A[i] == 1) {
            A[i] = 0; per_bit[i]++; total++; i++;
        }
        if (i < k) { A[i] = 1; per_bit[i]++; total++; }
    }
    return { total, per_bit, (double)total / n };
}

// 输出示例（n=1000, k=16）:
// 总翻转次数: 1994 (上界 2n = 2000)
// 摊销代价: 1.9940
```

### 35.3.3 势能法证明：$\Phi = $ 当前 1-bit 的个数

**势能函数**：$\Phi(D) = $ 计数器中当前值为 1 的位的数量（即二进制表示中 1 的个数，$b_i$）。

**性质验证**：$\Phi \geq 0$ 始终成立（1 的个数不能为负）；初始全为 0 时 $\Phi_0 = 0$。

**分析第 $i$ 次 INCREMENT**：

假设本次翻转了 $t$ 个 1 变 0（进位链长度），然后翻转 1 个 0 变 1（吸收进位）：
- 实际代价 $c_i = t + 1$（翻转 $t + 1$ 位）
- 势能变化：消去 $t$ 个 1，增加 1 个 1，故 $\Delta\Phi = 1 - t$
- 摊销代价：$\hat{c}_i = (t+1) + (1-t) = 2$

**结论**：每次 INCREMENT 的摊销代价恰好等于 **2**，与进位链长度 $t$ 无关！$n$ 次 INCREMENT 总代价 $\leq 2n + \Phi_0 = 2n$（与聚合法一致）。

> **势能法的优越性**：聚合法需要仔细推导每一位的翻转频率，而势能法一旦找到合适的 $\Phi$，推导极为简洁（只需分析单次操作）。

<div data-component="ThreeAmortizedMethodsCompare"></div>

---

## 35.4 Splay 树的摊销分析

Splay 树（伸展树）是一种自调整二叉搜索树，每次访问一个节点后，都通过一系列旋转将其"伸展"到根。Splay 树没有维护显式的平衡信息，却能保证 $m$ 次操作的摊销 $O(m \log n)$ 总代价。

### 35.4.1 Splay 操作的三种情形

Splay 将节点 $x$ 旋转到根，分三种情形（$p$ = 父节点，$g$ = 祖父节点）：

**Zig（单旋）**：$x$ 的父节点就是根，直接旋转一次。

```
    p              x
   / \            / \
  x   C    →    A   p
 / \                / \
A   B              B   C
```

**Zig-Zig（同侧双旋）**：$x$、$p$、$g$ 在同侧（同为左子或同为右子），先旋转 $p$，再旋转 $x$。

**Zig-Zag（异侧双旋）**：$x$ 和 $p$ 方向相反（一左一右），旋转两次 $x$（类似 AVL 的 LR/RL 旋转）。

```python
class SplayNode:
    """Splay 树节点：key, 左/右/父指针"""
    __slots__ = ['key', 'left', 'right', 'parent', 'size']
    def __init__(self, key: int):
        self.key = key
        self.left = self.right = self.parent = None
        self.size = 1

class SplayTree:
    def __init__(self):
        self.root: SplayNode | None = None

    def _update_size(self, x: SplayNode):
        """更新子树大小（用于势能函数的 rank 计算）"""
        x.size = 1
        if x.left:  x.size += x.left.size
        if x.right: x.size += x.right.size

    def _rotate(self, x: SplayNode):
        """将 x 向上旋转一次（AVL 标准旋转）"""
        p = x.parent
        g = p.parent

        # 判断 x 是 p 的左儿子还是右儿子
        if p.left is x:
            # 右旋：x 变 p 的位置，p 降为 x 的右孩子
            p.left = x.right
            if x.right: x.right.parent = p
            x.right = p
        else:
            # 左旋
            p.right = x.left
            if x.left: x.left.parent = p
            x.left = p

        p.parent = x
        x.parent = g

        if g:
            if g.left is p: g.left = x
            else:           g.right = x
        else:
            self.root = x   # p 原来是根，旋转后 x 成为新根

        self._update_size(p)
        self._update_size(x)

    def splay(self, x: SplayNode):
        """
        将节点 x 伸展到根。
        按照 zig / zig-zig / zig-zag 三种情形处理。
        摊销复杂度 O(log n)（势能法证明）。
        """
        while x.parent is not None:
            p = x.parent
            g = p.parent
            if g is None:
                # Zig：x 的父节点就是根，单旋一次
                self._rotate(x)
            elif (g.left is p) == (p.left is x):
                # Zig-Zig：同方向（先旋 p，再旋 x）
                self._rotate(p)
                self._rotate(x)
            else:
                # Zig-Zag：不同方向（旋 x 两次）
                self._rotate(x)
                self._rotate(x)
```

```cpp
#include <cstddef>
#include <cmath>

struct Node {
    int key;
    Node *left, *right, *parent;
    int sz;   // 子树大小，用于 rank(x) = log(sz)
    Node(int k) : key(k), left(nullptr), right(nullptr), parent(nullptr), sz(1) {}
};

class SplayTree {
    Node* root = nullptr;

    void update_sz(Node* x) {
        if (!x) return;
        x->sz = 1;
        if (x->left)  x->sz += x->left->sz;
        if (x->right) x->sz += x->right->sz;
    }

    void rotate(Node* x) {
        Node* p = x->parent;
        Node* g = p->parent;
        bool is_left = (p->left == x);

        if (is_left) { p->left = x->right; if (x->right) x->right->parent = p; x->right = p; }
        else         { p->right = x->left;  if (x->left)  x->left->parent  = p; x->left  = p; }
        p->parent = x; x->parent = g;
        if (g) { if (g->left == p) g->left = x; else g->right = x; }
        else root = x;
        update_sz(p); update_sz(x);
    }

public:
    void splay(Node* x) {
        while (x->parent) {
            Node* p = x->parent;
            Node* g = p->parent;
            if (!g)                                  rotate(x);           // Zig
            else if ((g->left==p) == (p->left==x)) { rotate(p); rotate(x); } // Zig-Zig
            else                                   { rotate(x); rotate(x); } // Zig-Zag
        }
    }
};
```

### 35.4.2 势能函数与摊销上界

**定义**：节点 $x$ 的秩（rank）为 $r(x) = \lfloor \log_2(|T_x|) \rfloor$，其中 $|T_x|$ 为以 $x$ 为根的子树大小。

**势能函数**：

$$\Phi(T) = \sum_{x \in T} r(x) = \sum_{x \in T} \lfloor \log_2(|T_x|) \rfloor$$

**引理**（访问引理 Access Lemma，Sleator & Tarjan 1985）：  
Splay 操作将节点 $x$ 伸展至根 $t$，其摊销代价满足：

$$\hat{c}(\text{splay}(x, t)) \leq 3(r(t) - r(x)) + 1$$

由于 $r(t) \leq \log_2 n$，故摊销代价 $= O(\log n)$。

**对三种情形的摊销代价分别推导**（以 Zig-Zig 为例）：

Zig-Zig 中旋转两次，实际代价 = 2。需证明：

$$2 + \Phi_{\text{after}} - \Phi_{\text{before}} \leq 3(r'(x) - r(x))$$

其中 $r'$ 表示旋转后的 rank。利用**对数的凹性**（$a + b \leq 2\log_2 n$ 当 $2^a + 2^b \leq n$）可以严格证明上述不等式。

> 完整证明细节见 CLRS 第4版 Chapter 17.4 或 Sleator & Tarjan 1985 原始论文。

<div data-component="SplayAmortizedTrace"></div>

### 35.4.3 Splay 树的总复杂度

**定理**：对有 $n$ 个节点的 Splay 树，从空树开始执行 $m$ 次任意操作（搜索、插入、删除），总时间为：

$$O((m + n) \log n)$$

即每次操作平均 $O(\log n)$（与 AVL 树和红黑树相当），但无需维护额外的平衡因子。

**Splay 树的特殊性质（动态最优猜想）**：

Splay 树的实际性能往往优于朴素的 $O(\log n)$：
- **工作集定理**：最近访问过的元素再次访问更快
- **序列定理**：按序访问 $n$ 个元素代价 $O(n)$，而不是 $O(n \log n)$
- **动态指针定理**（未完全证明）：Splay 树在任何输入序列上的代价可能与任何竞争者相当

---

## 35.5 Fibonacci 堆的摊销分析

Fibonacci 堆（第 36 章详述）是一种支持 DECREASE-KEY $O(1)$ 摊销的高级优先队列。这里仅从势能分析角度理解其复杂度来源。

### 35.5.1 势能函数定义

$$\Phi(H) = t(H) + 2 \times m(H)$$

其中：
- $t(H)$ = 根列表中的树的数量
- $m(H)$ = 被标记的节点数（`mark = true` 的节点）

**设计动机**：
- 根列表中每多一棵树，EXTRACT-MIN 时约定义 CONSOLIDATE 的代价（合并同度树）
- 标记节点的级联裁剪（CASCADING-CUT）代价由势能预支付

### 35.5.2 各操作的摊销分析

**INSERT**：新建节点加入根列表，$t(H)$ 增加 1，$m(H)$ 不变。
$$\hat{c}(\text{INSERT}) = 1 + \Delta\Phi = 1 + 1 = O(1)$$

**EXTRACT-MIN**：取出最小根，将其 $d$ 个子节点加入根列表（$\Delta t = +d - 1$），CONSOLIDATE 使树数从 $t + d$ 降至最多 $D(n) + 1$，实际代价 $\approx t(H) + d$：
$$\hat{c}(\text{EXTRACT-MIN}) = O(D(n)) = O(\log n)$$

**DECREASE-KEY（级联裁剪）**：每次裁剪一个已标记节点，$t(H)$ 增加 1，$m(H)$ 减少 1：
$$\Delta\Phi = 1 + 2 \times (-1) = -1$$
实际耗费 1 步，摊销代价 = $1 + (-1) = 0$；最后一次（未标记节点）$\Delta m = +1$，代价 1，故：
$$\hat{c}(\text{DECREASE-KEY}) = O(1)$$

**操作复杂度汇总**：

| 操作 | 实际代价 | 摊销代价 | 对比二叉堆 |
|---|---|---|---|
| INSERT | $O(1)$ | $O(1)$ | $O(\log n)$ |
| MINIMUM | $O(1)$ | $O(1)$ | $O(1)$ |
| UNION | $O(1)$ | $O(1)$ | $O(n)$ |
| EXTRACT-MIN | $O(\log n)$ 均摊 | $O(\log n)$ | $O(\log n)$ |
| DECREASE-KEY | $O(\log n)$ 最坏 | **$O(1)$** | $O(\log n)$ |
| DELETE | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |

### 35.5.3 度数界 D(n) ≤ log_φ n

CONSOLIDATE 之后，根列表中不同度数的树分别最多一棵，故最终树数 ≤ $D(n)$。

**引理**（Fibonacci 性质）：Fibonacci 堆中，度数为 $k$ 的节点至少有 $F_{k+2} \geq \phi^k$ 个后代（$\phi = \frac{1+\sqrt{5}}{2} \approx 1.618$，黄金分割）。

证明依赖 DECREASE-KEY 中 mark 机制：每个节点最多失去一个孩子（否则触发裁剪），保证了子树的"Fibonacci 下界"。

$$D(n) \leq \log_\phi n \approx 1.44 \log_2 n$$

### 35.5.4 工程中为何罕用？

尽管理论复杂度完美，Fibonacci 堆在实践中几乎未被采用：

1. **大常数因子**：指针操作多，同量级代价约为二叉堆的 5~10 倍
2. **缓存不友好**：多棵离散树分布在内存各处，L1/L2 缓存命中率低
3. **实现极其复杂**：需要正确处理 CUT、CASCADING-CUT、CONSOLIDATE 等边界
4. **Dijkstra 实际差距小**：对大多数图，边数 $E \ll V^2$，普通二叉堆（$O(E \log V)$）实际上比 Fibonacci 堆（$O(E + V \log V)$）更快

> **工程建议**：Dijkstra 用 `heapq`（Python）或 `priority_queue`（C++）；理论研究用 Fibonacci 堆。

---

## 35.6 三种分析方法综合对比与选择指南

选择合适的摊销分析方法时，参考以下原则：

| 场景 | 推荐方法 | 原因 |
|---|---|---|
| 操作序列总代价有简洁公式 | 聚合法 | 最直接，一行结论 |
| 不同操作代价差异大（如 push vs pop） | 记账法 | 为贵操作单独分配信用 |
| 数据结构状态可量化的"积累" | 势能法 | 最通用，可严格推导 |
| Splay、Fibonacci 堆等复杂结构 | 势能法 | 唯一可操作的方法 |

**通用势能函数设计步骤**：
1. 识别"廉价操作"和"昂贵操作"
2. 昂贵操作发生前，哪个量"膨胀"了？（这应该是 $\Phi$）
3. 验证 $\Phi \geq 0$ 且 $\Phi_0 = 0$
4. 计算 $\hat{c}_i = c_i + \Delta\Phi_i$，证明对所有情形 $\hat{c}_i = O(f(n))$

---

## 35.7 工程陷阱与常见错误

### 陷阱 1：势能函数可能为负

**错误做法**：定义 $\Phi(D)$ 时没有保证 $\Phi \geq 0$。若在某些状态下 $\Phi < 0$，则总摊销代价 $< $ 总实际代价，证明无效。

**正确做法**：严格验证所有可达状态满足 $\Phi(D) \geq 0$，并明确 $\Phi_0 = 0$。

### 陷阱 2：混淆"摊销 O(1)"和"每次 O(1)"

**错误理解**："动态数组的 push_back 摊销 O(1)"不等于"每次 push_back 都是 O(1)"。

**正确理解**：在最差的 $n$ 次操作序列中，总代价 $\leq cn$，每次的**平均**代价 $\leq c$。个别操作可以是 $O(n)$，只要足够罕见。

**影响**：实时系统（如自动控制系统）无法接受偶尔的 $O(n)$ 尖峰 —— 此时应考虑带有**最坏情况 O(1)** 保证的数据结构（如 "worst-case O(1)" 哈希表）。

### 错误 3：缩容阈值选 1/2 导致退化

当缩容阈值设为 50%（size < capacity/2 时就缩容）：

```
操作序列：push, push, push, pop, push, pop, push, pop, ...  ← 在阈值附近反复触发扩/缩
每次 push/pop 都触发 resize，摊销代价退化为 O(n)！
```

必须在扩容阈值（100%）和缩容阈值（25%）之间保留足够"缓冲区"。

### 陷阱 4：忽略初始状态的势能

势能法需要证明 $\sum \hat{c}_i \geq \sum c_i$，即：

$$\sum \hat{c}_i = \sum c_i + \underbrace{\Phi(D_n) - \Phi(D_0)}_{\geq 0 \text{ 需要保证}}$$

若 $\Phi_0 > 0$（初始有积累），需要额外说明或调整定义，使 $\Phi(D_0) = 0$。

---

## 35.8 小练习

**练习 1（聚合法）**：  
假设动态数组每次扩容增加固定 $c$ 个空间（而非倍增）。请用聚合法证明 $n$ 次 push_back 的总代价为 $\Theta(n^2/c)$，即摊销代价 $\Theta(n/c)$。当 $c = O(1)$ 时为 $O(n)$，远不如倍增的 $O(1)$。

**练习 2（势能法）**：  
对带缩容（25% 阈值）的动态数组，定义分段势能函数，验证 push_back 和 pop_back 的摊销代价均为 $O(1)$。

**练习 3（计数器变种）**：  
考虑支持 INCREMENT 和 RESET（将计数器归零）操作的二进制计数器。定义合适的势能函数，分析两种操作的摊销代价。

**练习 4（思考题）**：  
KMP 算法中，每次字符比较后 `j` 指针会前进或后退。用聚合法（或势能法）证明 KMP 的总比较次数为 $O(n)$（其中 $n$ 为文本长度）。提示：$j$ 能前进的次数被 $j$ 能后退的次数所限制。

---

## 章节小结

| 知识点 | 一句话要点 |
|---|---|
| 摊销分析目标 | n 次操作总代价 / n，不假设概率分布，对最坏序列成立 |
| 聚合法 | 找总代价上界 T(n)，摊销 = T(n)/n |
| 记账法 | 分配"信用"，账户余额恒 ≥ 0 |
| 势能法 | $\hat{c}_i = c_i + \Delta\Phi$，设计 Φ 使摊销代价均匀 |
| 动态表倍增 | 摊销 $O(1)$；势能 $\Phi = 2s - k$ 在扩容时骤降 |
| 二进制计数器 | $n$ 次 INCREMENT 总翻转 < 2n；势能 = 1-bit 数量 |
| Splay 树 | m 次操作 $O((m+n)\log n)$；势能 = Σ rank(x) |
| Fibonacci 堆 | DECREASE-KEY $O(1)$ 摊销；实践很少用 |
| 摊销 ≠ 平均 | 摊销是确定性界，平均需要概率假设 |

---

## 参考资料

- CLRS 第4版 Chapter 16（三种摊销方法）、Chapter 17（动态表、Splay 摊销）、Chapter 19（Fibonacci 堆）
- Sleator & Tarjan 1985：*Self-Adjusting Binary Search Trees*（Splay 原始论文）
- Fredman & Tarjan 1987：*Fibonacci Heaps and Their Uses in Improved Network Optimization Algorithms*
- MIT 6.046 Lecture 2：摊销分析与动态表
- MIT 6.854 Advanced Algorithms：Splay 树势能法详细证明
