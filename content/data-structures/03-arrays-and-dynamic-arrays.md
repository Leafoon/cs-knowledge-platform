# Chapter 3: 数组与动态数组

> **难度**：🟢 初级 ｜ **前置要求**：Chapter 0–2（基础概念、复杂度分析、算法设计）  
> **核心目标**：理解连续内存模型 → 掌握动态扩容摊销分析 → 熟练前缀和/差分/双指针三大技术

---

## 3.1 数组基础

### 3.1.1 连续内存模型与随机访问

数组是**最基础的数据结构**，本质是一段**连续的内存区域**，存放同类型的元素。

**随机访问公式**：

$$\text{addr}(A[i]) = \text{base\_addr} + i \times \text{element\_size}$$

这个公式使得访问任意下标元素都只需**一次加法 + 一次乘法** → $O(1)$ 时间。

```python
# Python 示例：模拟数组的随机访问
arr = [10, 20, 30, 40, 50]

# 设 base_addr = 1000，int = 8 bytes（64-bit）
# arr[3] 的地址 = 1000 + 3 × 8 = 1024
# 这就是为什么 arr[3] 在 O(1) 时间内就能取到值

# Python list 内部维护一个指针数组（PyObject* 数组）
# 每个槽位存的是对象的指针（8 bytes），而非对象本身
print(arr[3])  # 40，O(1)
```

```cpp
// C++ 原生数组：元素连续存储于栈或堆
int arr[] = {10, 20, 30, 40, 50};

// base_addr = &arr[0]
// arr[3] 的地址 = base_addr + 3 * sizeof(int) = base_addr + 12
std::cout << arr[3]; // 40
std::cout << *(arr + 3); // 40，完全等价（指针算术）
```

> **💡 关键洞察**：随机访问 $O(1)$ 的前提是"连续 + 同类型"。如果元素大小不同（如多态对象），则必须通过指针间接访问，退化为两次内存读取。

### 3.1.2 行主序 vs 列主序（多维数组的内存布局）

对于二维数组 $A[r][c]$（$m$ 行 $n$ 列）：

| 布局 | 公式 | 特点 |
|------|------|------|
| **行主序**（Row-major，C/C++/Python/Java） | $\text{addr}(A[i][j]) = \text{base} + (i \times n + j) \times s$ | 同行元素相邻，按行遍历缓存友好 |
| **列主序**（Column-major，Fortran/MATLAB/NumPy 可选） | $\text{addr}(A[i][j]) = \text{base} + (j \times m + i) \times s$ | 同列元素相邻，按列遍历缓存友好 |

**缓存行（Cache Line）影响**：现代 CPU 一次从内存加载 **64 字节**（通常 16 个 `int`）到缓存。

```python
# 行主序（C 顺序）遍历——缓存友好 ✅
import numpy as np
A = np.zeros((1000, 1000), dtype=np.float64, order='C')  # 行主序

# 按行遍历：A[0][0], A[0][1], ..., A[0][999], A[1][0], ...
# 连续内存访问，每次读取到缓存的 8 个 float64 都会被用到
for i in range(1000):
    for j in range(1000):
        A[i][j] = i * j   # ✅ 缓存命中率高

# 列遍历（对行主序数组）——缓存不友好 ❌
for j in range(1000):
    for i in range(1000):
        A[i][j] = i * j   # ❌ 每次跨越 1000*8 = 8000 字节，大量缓存缺失
```

```cpp
// C++ 行主序：外层 i 行，内层 j 列
// 访问 arr[i][j] 与 arr[i][j+1] 相差 4 字节（相邻）
int arr[1000][1000];
for (int i = 0; i < 1000; i++)
    for (int j = 0; j < 1000; j++)
        arr[i][j] = i + j;  // ✅ 行主序，缓存友好
```

> **面试考点** 🎯：矩阵转置/乘法算法中，内外循环的顺序对实际运行时间有 **5–10 倍** 影响，原因就是缓存局部性。

<div data-component="ArrayMemoryLayout"></div>

### 3.1.3 数组操作时间复杂度汇总

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 随机访问 `arr[i]` | $O(1)$ | 直接计算地址 |
| 头部插入 | $O(n)$ | 需后移所有元素 |
| 尾部插入（有空间） | $O(1)$ | 直接赋值 |
| 中间插入 `arr[i]` 后 | $O(n)$ | 后移 $n-i$ 个元素 |
| 头部删除 | $O(n)$ | 需前移所有元素 |
| 尾部删除 | $O(1)$ | 直接缩减长度 |
| 中间删除 `arr[i]` | $O(n)$ | 前移 $n-i-1$ 个元素 |
| 线性搜索 | $O(n)$ | 无序数组遍历 |
| 二分搜索 | $O(\log n)$ | 有序数组 |

### 3.1.4 静态数组 vs 动态数组

| 特征 | 静态数组 | 动态数组 |
|------|---------|---------|
| 大小 | 编译时固定 | 运行时可增长 |
| 内存分配 | 栈或静态区 | 堆 |
| 越界检查 | 通常无（C/C++） | 通常有 |
| 代表 | C 数组、C++ `array<T,N>` | Python `list`、C++ `vector<T>`、Java `ArrayList<T>` |

---

## 3.2 动态数组（Dynamic Array）

动态数组是数组 + **自动扩容**机制的组合，允许在不知道元素数量的情况下持续追加。

### 3.2.1 动态扩容策略：倍增法 vs 固定步长

**核心问题**：当数组满了（`size == capacity`），需要重新分配更大的内存并迁移数据。

**方案一：固定步长扩容**（每次扩充固定 $k$ 个槽位）

$$\text{总复制次数} = k + 2k + 3k + \ldots + n = \frac{n}{k} \cdot \frac{n/k + 1}{2} \cdot k \approx \frac{n^2}{2k} = O(n^2)$$

**方案二：倍增法（Doubling）**（每次容量翻倍）

$$\text{总复制次数} = 1 + 2 + 4 + \ldots + n/2 = n - 1 < n = O(n)$$

→ 平均每次 `append` 的摊销代价 = $O(n)/n = O(1)$。

```python
class DynamicArray:
    """手动实现动态数组，演示倍增扩容"""
    def __init__(self):
        self._data = [None] * 1   # 初始容量 1
        self._size = 0            # 当前元素数
        self._capacity = 1        # 当前容量
        self._copies = 0          # 记录总复制次数（用于演示）

    def append(self, val):
        if self._size == self._capacity:
            self._resize(self._capacity * 2)  # 倍增！
        self._data[self._size] = val
        self._size += 1

    def _resize(self, new_cap):
        new_data = [None] * new_cap
        for i in range(self._size):   # 复制旧数据
            new_data[i] = self._data[i]
            self._copies += 1
        self._data = new_data
        self._capacity = new_cap

    def __getitem__(self, i):
        if 0 <= i < self._size:
            return self._data[i]
        raise IndexError("index out of range")

    def __len__(self):
        return self._size

# 演示：追加 16 个元素，观察扩容时机和复制次数
da = DynamicArray()
for i in range(16):
    da.append(i)
    print(f"append {i}: size={len(da)}, cap={da._capacity}, total_copies={da._copies}")
```

```
append 0: size=1, cap=1,  total_copies=0
append 1: size=2, cap=2,  total_copies=1   ← 扩容：复制1次
append 2: size=3, cap=4,  total_copies=3   ← 扩容：复制2次（累计3）
append 3: size=4, cap=4,  total_copies=3
append 4: size=5, cap=8,  total_copies=7   ← 扩容：复制4次（累计7）
...
append 8: size=9, cap=16, total_copies=15  ← 扩容：复制8次（累计15）
```

```cpp
// C++ std::vector 使用倍增法（实现细节因编译器而异）
#include <vector>
#include <iostream>
int main() {
    std::vector<int> v;
    for (int i = 0; i < 16; i++) {
        v.push_back(i);
        std::cout << "size=" << v.size()
                  << " capacity=" << v.capacity() << "\n";
    }
    // 输出：capacity 在 1, 2, 4, 8, 16 时依次翻倍
}
```

<div data-component="DynamicArrayGrowth"></div>

### 3.2.2 摊销 O(1) 追加的严格证明

**方法一：聚合分析（Aggregate Analysis）**

设追加 $n$ 次，总代价 $T(n) =$（直接追加代价）+（复制代价）

$$T(n) = n + \underbrace{(1 + 2 + 4 + \ldots + n/2)}_{=n-1} < 2n$$

摊销代价 $= T(n)/n < 2 = O(1)$。$\blacksquare$

**方法二：势能法（Potential Method）**

定义势能函数 $\Phi = 2 \times \text{size} - \text{capacity}$（表示"为下次扩容准备的余量"）。

- 初始：$\Phi_0 = 2 \times 0 - 1 = -1$（约定 $\Phi_0 = 0$）
- 普通追加（不触发扩容）：实际代价 $c_i = 1$，$\Delta\Phi = 2$  
  摊销代价 $\hat{c}_i = c_i + \Delta\Phi = 1 + 2 = 3 = O(1)$
- 触发扩容（`size = capacity = k`）：实际代价 $c_i = 1 + k$（复制 $k$ 次），$\Delta\Phi = 2 - k$  
  摊销代价 $\hat{c}_i = (1+k) + (2-k) = 3 = O(1)$

→ 每次 `append` 的摊销代价始终为 $O(1)$。$\blacksquare$

> **💡 直觉**：势能函数 $\Phi$ 衡量"预存能量"：每次普通追加存入 $+2$ 能量，扩容时用这些能量"支付"复制代价。

### 3.2.3 Python list / C++ vector / Java ArrayList 对比

| 特性 | Python `list` | C++ `vector<T>` | Java `ArrayList<T>` |
|------|--------------|----------------|---------------------|
| 扩容因子 | ~1.125（小数组）/ ~2（大数组） | 通常 ×2（GCC）/ ×1.5（MSVC） | ×1.5 |
| 元素存储 | PyObject 指针（统一 8 bytes） | 值类型（直接存对象） | 装箱对象引用 |
| 随机访问 | $O(1)$ | $O(1)$ | $O(1)$ |
| 缓存局部性 | 中（额外一层间接） | 好（值紧密排列） | 差（对象在堆中分散） |
| 线程安全 | GIL 一定程度保护 | 不安全 | 不安全（需手动同步） |

### 3.2.4 缩容策略：阈值选 1/4 而非 1/2 的原因

**问题**：当元素被大量删除后不缩容，会浪费内存。

**错误方案**：当 `size == capacity / 2` 时缩容到 `capacity / 2`。

设容量为 $C$，在 `size = C/2` 时反复追加、删除：

$$\text{追加} \to \text{扩容到} 2C \to \text{删除} \to \text{缩容到} C \to \text{追加} \to \ldots$$

每次操作触发 $O(n)$ 的重分配 → **摊销代价退化为 $O(n)$！**

**正确方案**：当 `size ≤ capacity / 4` 时，才缩容到 `capacity / 2`。

这样在触发缩容后，`size = capacity/2 × 1/2 = capacity/4`，需要再删除 `capacity/4` 个元素才能再次触发缩容，保证了操作序列的摊销 $O(1)$。

```python
def remove_last(self):
    if self._size == 0:
        raise IndexError("empty array")
    self._data[self._size - 1] = None
    self._size -= 1
    # 只有使用率跌到 25% 才缩容（而非 50%）
    if self._size > 0 and self._size <= self._capacity // 4:
        self._resize(self._capacity // 2)
```

### 3.2.5 工程注意事项

```python
# 1. 预分配：已知大小时避免多次扩容
arr = [None] * n          # Python：预分配 n 个槽位
# C++: vector.reserve(n);  # 预分配不设 size

# 2. 内存分配大数组时，警惕 GC 压力（Python / JVM）
import sys
lst = list(range(10_000_000))
print(sys.getsizeof(lst))  # 约 80 MB for 10M pointers

# 3. NumPy 数组：连续内存 + 向量化操作，远优于 Python list
import numpy as np
arr_np = np.arange(10_000_000, dtype=np.int64)   # ~80 MB，C 层面连续
```

---

## 3.3 前缀和与差分数组

### 3.3.1 前缀和（Prefix Sum）

**问题**：给定数组 $A[0..n-1]$，需要快速回答 **区间求和查询** $\text{sum}(l, r) = A[l] + A[l+1] + \ldots + A[r]$。

若每次暴力求和：$O(n)$/次，$Q$ 次查询共 $O(nQ)$。

**前缀和预处理**：

$$\text{pre}[i] = A[0] + A[1] + \ldots + A[i-1] \quad (\text{pre}[0] = 0)$$

$$\text{sum}(l, r) = \text{pre}[r+1] - \text{pre}[l]$$

预处理 $O(n)$，每次查询 $O(1)$，$Q$ 次查询共 $O(n + Q)$。

```python
def build_prefix_sum(arr):
    n = len(arr)
    pre = [0] * (n + 1)   # pre[0] = 0 是哨兵，避免特判 l=0
    for i in range(n):
        pre[i + 1] = pre[i] + arr[i]
    return pre

def query_sum(pre, l, r):
    """查询 arr[l..r] 的和，左闭右闭"""
    return pre[r + 1] - pre[l]

# 示例
arr = [3, 1, 4, 1, 5, 9, 2, 6]
pre = build_prefix_sum(arr)
print(pre)      # [0, 3, 4, 8, 9, 14, 23, 25, 31]
print(query_sum(pre, 2, 5))  # arr[2..5] = 4+1+5+9 = 19
```

```cpp
#include <vector>
using namespace std;

vector<long long> buildPrefix(const vector<int>& arr) {
    int n = arr.size();
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++)
        pre[i + 1] = pre[i] + arr[i];
    return pre;
}

long long querySum(const vector<long long>& pre, int l, int r) {
    return pre[r + 1] - pre[l];
}
```

**⚠️ 常见错误**：

```python
# 错误：pre 数组长度不加 1，l=0 时 pre[-1] 访问越界！
pre = [0] * n
for i in range(n):
    pre[i] = (pre[i-1] if i > 0 else 0) + arr[i]
# query: pre[r] - pre[l-1]  → l=0 时 pre[-1] 是 Python 末尾元素，逻辑错误！

# 正确：pre 长度为 n+1，pre[0]=0 作为哨兵
```

### 3.3.2 二维前缀和（矩阵子矩形和）

$$\text{pre}[i][j] = \sum_{r=0}^{i-1} \sum_{c=0}^{j-1} A[r][c]$$

**容斥公式**：矩形 $(r_1, c_1) \to (r_2, c_2)$ 的和：

$$S = \text{pre}[r_2+1][c_2+1] - \text{pre}[r_1][c_2+1] - \text{pre}[r_2+1][c_1] + \text{pre}[r_1][c_1]$$

```python
def build_2d_prefix(matrix):
    m, n = len(matrix), len(matrix[0])
    pre = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pre[i][j] = (matrix[i-1][j-1]
                         + pre[i-1][j]
                         + pre[i][j-1]
                         - pre[i-1][j-1])  # 容斥减去重叠部分
    return pre

def query_rect(pre, r1, c1, r2, c2):
    return (pre[r2+1][c2+1]
            - pre[r1][c2+1]
            - pre[r2+1][c1]
            + pre[r1][c1])

# 示例
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
pre = build_2d_prefix(matrix)
print(query_rect(pre, 0, 0, 1, 1))  # 1+2+4+5 = 12
```

### 3.3.3 差分数组（Difference Array）

**问题**：给定数组 $A[0..n-1]$，需要快速执行 **区间加法操作** $A[l..r] += \delta$，最终输出完整数组。

若每次暴力更新：$O(r-l+1)$/次，$M$ 次操作共 $O(Mn)$。

**差分数组**：

$$\text{diff}[i] = A[i] - A[i-1] \quad (\text{diff}[0] = A[0])$$

区间 $[l, r]$ 加 $\delta$：

$$\text{diff}[l] \mathrel{+}= \delta, \quad \text{diff}[r+1] \mathrel{-}= \delta$$

还原：$A = \text{prefix\_sum}(\text{diff})$，即 $A[i] = \sum_{j=0}^{i} \text{diff}[j]$。

```python
def range_add(diff, l, r, delta):
    """O(1) 区间加法"""
    diff[l] += delta
    if r + 1 < len(diff):
        diff[r + 1] -= delta

def restore(diff):
    """O(n) 还原数组"""
    arr = diff[:]
    for i in range(1, len(arr)):
        arr[i] += arr[i - 1]
    return arr

# 示例
arr = [1, 3, 5, 7, 9]
diff = arr[:]          # 差分初始化
range_add(diff, 1, 3, 2)   # arr[1..3] += 2
range_add(diff, 2, 4, -1)  # arr[2..4] -= 1

result = restore(diff)
print(result)  # [1, 5, 6, 8, 8]
# 验证：
# arr[0]=1, arr[1]=3+2=5, arr[2]=5+2-1=6, arr[3]=7+2-1=8, arr[4]=9-1=8
```

```cpp
vector<int> applyDiff(vector<int>& arr,
                       vector<pair<int,int>>& queries,  // (l, r, delta)
                       vector<int>& deltas) {
    int n = arr.size();
    vector<int> diff(n + 1, 0);
    for (auto& [l, r] : queries) {
        int delta = deltas[&queries[0] - &queries[0]]; // simplified
        diff[l] += delta;
        if (r + 1 <= n) diff[r + 1] -= delta;
    }
    // Restore
    for (int i = 1; i <= n; i++) diff[i] += diff[i - 1];
    for (int i = 0; i < n; i++) arr[i] += diff[i];
    return arr;
}
```

<div data-component="PrefixSumVisualizer"></div>

### 3.3.4 差分数组与前缀和的互逆关系

|  | 操作 | 查询 |
|--|------|------|
| **前缀和** | $O(n)$ 预处理 | $O(1)$ 区间查询 |
| **差分数组** | $O(1)$ 区间更新 | $O(n)$ 还原一次 |

数学关系：差分是前缀和的逆运算。

$$\text{prefix\_sum}(\text{diff}(A)) = A$$
$$\text{diff}(\text{prefix\_sum}(A)) = A$$

### 3.3.5 变体：前缀积与前缀异或

```python
# 前缀积：product(l..r) = prefix_prod[r+1] / prefix_prod[l]
# 注意：需要处理包含 0 的情况（不能直接除法）
def build_prefix_prod(arr):
    pre = [1] * (len(arr) + 1)
    for i, v in enumerate(arr):
        pre[i + 1] = pre[i] * v
    return pre

# 前缀异或：xor(l..r) = prefix_xor[r+1] ^ prefix_xor[l]
# 利用 a^a=0, a^0=a 性质
def build_prefix_xor(arr):
    pre = [0] * (len(arr) + 1)
    for i, v in enumerate(arr):
        pre[i + 1] = pre[i] ^ v
    return pre

# 🎯 LeetCode #238 Product of Array Except Self（OJ 禁用除法）
def productExceptSelf(nums):
    n = len(nums)
    prefix = [1] * n
    suffix = [1] * n
    for i in range(1, n):
        prefix[i] = prefix[i-1] * nums[i-1]
    for i in range(n-2, -1, -1):
        suffix[i] = suffix[i+1] * nums[i+1]
    return [prefix[i] * suffix[i] for i in range(n)]
# 输入: [1,2,3,4] → 输出: [24, 12, 8, 6]
```

---

## 3.4 双指针技术

双指针（Two Pointers）是一类利用**两个索引变量**协同推进来降低时间复杂度的技术。核心思想：利用单调性（排序、单调递增窗口等）将 $O(n^2)$ 暴力枚举优化到 $O(n)$。

### 3.4.1 对撞指针

**适用场景**：有序数组中寻找满足某条件的配对。

**核心思路**：左指针 $l=0$，右指针 $r=n-1$，根据当前配对值与目标的关系移动指针。

```python
# 🎯 LeetCode #167 Two Sum II（有序数组两数之和）
def twoSum(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        s = nums[l] + nums[r]
        if s == target:
            return [l + 1, r + 1]  # 题目要求 1-indexed
        elif s < target:
            l += 1  # 和太小，移大左指针
        else:
            r -= 1  # 和太大，移小右指针
    return []
# 时间 O(n)，空间 O(1)；暴力 O(n²)

# 🎯 LeetCode #11 盛最多水的容器
def maxWater(height):
    l, r = 0, len(height) - 1
    res = 0
    while l < r:
        # 容量 = 两板中较矮的 × 距离
        res = max(res, min(height[l], height[r]) * (r - l))
        # 移动较矮的一侧（移动较高的一侧只会让结果更小）
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return res

# 🎯 LeetCode #15 三数之和（枚举第一个数，对后两个用对撞）
def threeSum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:  # 去重
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l+1]: l += 1  # 去重
                while l < r and nums[r] == nums[r-1]: r -= 1  # 去重
                l += 1; r -= 1
            elif s < 0:
                l += 1
            else:
                r -= 1
    return res
```

```cpp
// 接雨水（Trapping Rain Water）对撞指针 O(n) 解法
int trap(vector<int>& height) {
    int l = 0, r = height.size() - 1;
    int maxL = 0, maxR = 0, res = 0;
    while (l < r) {
        maxL = max(maxL, height[l]);
        maxR = max(maxR, height[r]);
        if (maxL < maxR) {
            res += maxL - height[l++];  // 左侧水深 = maxL - height[l]
        } else {
            res += maxR - height[r--];
        }
    }
    return res;
}
```

### 3.4.2 快慢指针

**适用场景**：数组原地去重、按条件过滤、数组分区。

```python
# 🎯 LeetCode #26 删除有序数组中的重复项（原地，O(1) 空间）
def removeDuplicates(nums):
    if not nums:
        return 0
    slow = 0  # 慢指针：维护"已处理区段"的末尾
    for fast in range(1, len(nums)):  # 快指针：遍历
        if nums[fast] != nums[slow]:  # 发现新值
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1  # 新数组长度

# 🎯 LeetCode #283 移动零（非零元素前移，零补尾）
def moveZeroes(nums):
    slow = 0  # 指向下一个非零放置位置
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
    # 所有非零已移到前面，其余自然为 0

# 快慢指针奇偶分区（偶数在前，奇数在后，相对顺序不变需稳定版本）
def partition_odd_even(nums):
    slow = 0  # 偶数区段末尾
    for fast in range(len(nums)):
        if nums[fast] % 2 == 0:  # 发现偶数
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
    return nums
```

```cpp
// C++ 原地去重（类似 unique 函数）
int removeDuplicates(vector<int>& nums) {
    int slow = 0;
    for (int fast = 1; fast < nums.size(); fast++) {
        if (nums[fast] != nums[slow])
            nums[++slow] = nums[fast];
    }
    return slow + 1;
}
```

### 3.4.3 滑动窗口

滑动窗口（Sliding Window）是对"连续子数组/子串"问题的经典优化技术。维护一个窗口 $[l, r]$，通过移动指针而非重新计算来保持窗口状态。

**固定窗口**（窗口大小 $k$ 固定）：

```python
# 🎯 固定窗口：长度为 k 的子数组的最大和
def maxSumWindow(arr, k):
    # 初始化第一个窗口
    window_sum = sum(arr[:k])
    max_sum = window_sum
    # 滑动：移入 arr[r]，移出 arr[l-1]
    for l in range(1, len(arr) - k + 1):
        window_sum += arr[l + k - 1] - arr[l - 1]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

**可变窗口**（窗口大小根据条件动态收缩）：

```python
# 🎯 LeetCode #3 无重复字符的最长子串
def lengthOfLongestSubstring(s):
    char_idx = {}   # 字符 → 最近出现位置
    l = 0           # 窗口左边界
    res = 0
    for r, c in enumerate(s):
        if c in char_idx and char_idx[c] >= l:
            l = char_idx[c] + 1  # 右移左边界，跳过重复字符
        char_idx[c] = r
        res = max(res, r - l + 1)
    return res
# "abcabcbb" → 3 ("abc")

# 🎯 LeetCode #209 长度最小的子数组（和 ≥ target）
def minSubArrayLen(target, nums):
    l = 0
    window_sum = 0
    res = float('inf')
    for r, v in enumerate(nums):
        window_sum += v
        while window_sum >= target:   # 收缩左边界
            res = min(res, r - l + 1)
            window_sum -= nums[l]
            l += 1
    return 0 if res == float('inf') else res

# 🎯 LeetCode #76 最小覆盖子串（困难）
def minWindow(s, t):
    from collections import Counter
    need = Counter(t)
    have = {}
    formed = 0  # 已满足的字符种类数
    required = len(need)
    l = 0
    res = ""
    for r, c in enumerate(s):
        have[c] = have.get(c, 0) + 1
        if c in need and have[c] == need[c]:
            formed += 1
        while formed == required:  # 窗口有效，尝试收缩
            if not res or r - l + 1 < len(res):
                res = s[l:r+1]
            lc = s[l]
            have[lc] -= 1
            if lc in need and have[lc] < need[lc]:
                formed -= 1
            l += 1
    return res
```

```cpp
// C++ 滑动窗口：最长不重复子串
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> idx;
    int l = 0, res = 0;
    for (int r = 0; r < s.size(); r++) {
        if (idx.count(s[r]) && idx[s[r]] >= l)
            l = idx[s[r]] + 1;
        idx[s[r]] = r;
        res = max(res, r - l + 1);
    }
    return res;
}
```

**可变窗口设计模板**：

```python
def sliding_window_template(arr):
    l = 0
    state = {}  # 窗口内的状态（哈希表、计数器等）
    res = 0     # 或 float('inf')

    for r in range(len(arr)):
        # 1. 将 arr[r] 加入窗口，更新 state
        # state[arr[r]] = ...

        # 2. 判断窗口是否需要收缩（条件视题目而定）
        while <窗口不满足条件>:
            # 3. 将 arr[l] 移出窗口，更新 state
            # state[arr[l]] -= 1
            l += 1

        # 4. 此时窗口 [l, r] 满足条件，更新答案
        res = max(res, r - l + 1)

    return res
```

### 3.4.4 双指针时间复杂度分析

**关键观察**：不论是对撞指针还是快慢指针/滑动窗口，每个指针最多移动 $n$ 步。

- 对撞指针：$l$ 从 0 向右，$r$ 从 $n-1$ 向左，两者共移动 $n$ 步 → $O(n)$
- 快慢指针：`fast` 遍历 $n$ 个元素，`slow` 至多也走 $n$ 步 → $O(n)$
- 滑动窗口：`r` 向右 $n$ 步，`l` 向右至多 $n$ 步 → $O(n)$（注意内层 while 不会让 `l > r`）

<div data-component="TwoPointerRace"></div>

<div data-component="SlidingWindowDemo"></div>

---

## 3.5 常见数组技巧

### 3.5.1 原地修改（In-place）

```python
# 🎯 LeetCode #27 移除元素（in-place，O(1) 空间）
def removeElement(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow

# 🎯 LeetCode #28 在原地数组上找子串（KMP 思路预告）
# 暴力 O(mn)，KMP O(m+n)——见 Chapter 31（字符串章节）
```

### 3.5.2 数组旋转：三次翻转法

```python
# 🎯 LeetCode #189 轮转数组
# 将数组向右旋转 k 步：O(n) 时间，O(1) 空间
def rotate(nums, k):
    n = len(nums)
    k %= n  # 处理 k > n 的情况
    def reverse(l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1; r -= 1
    # 三次翻转：
    # 1. 翻转全部
    reverse(0, n - 1)    # [1,2,3,4,5,6,7] → [7,6,5,4,3,2,1]
    # 2. 翻转前 k 个
    reverse(0, k - 1)    # [7,6,5,4,3,2,1] → [5,6,7,4,3,2,1]
    # 3. 翻转后 n-k 个
    reverse(k, n - 1)    # [5,6,7,4,3,2,1] → [5,6,7,1,2,3,4]
```

```cpp
void rotate(vector<int>& nums, int k) {
    int n = nums.size();
    k %= n;
    reverse(nums.begin(), nums.end());
    reverse(nums.begin(), nums.begin() + k);
    reverse(nums.begin() + k, nums.end());
}
```

> **证明**：设原数组 $A = [a_0, \ldots, a_{n-k-1}, a_{n-k}, \ldots, a_{n-1}]$，目标 $B = [a_{n-k}, \ldots, a_{n-1}, a_0, \ldots, a_{n-k-1}]$。翻转全部得 $[a_{n-1}, \ldots, a_{n-k}, a_{n-k-1}, \ldots, a_0]$，再分别翻转两段即得 $B$。$\blacksquare$

### 3.5.3 摩尔投票法（Boyer-Moore Voting）

**问题**：找出超过 $\lfloor n/2 \rfloor$ 次出现的众数（保证存在）。

**核心思想**：候选者与非候选者"对抗抵消"，真正的众数一定活到最后。

```python
# 🎯 LeetCode #169 多数元素
def majorityElement(nums):
    candidate, count = None, 0
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    return candidate
# 时间 O(n)，空间 O(1)
```

**摩尔投票直觉**：想象每个非候选元素都"消耗"一个候选元素。由于众数超过半数，它消耗完所有非众数后还有剩余，必然成为最后的候选者。

### 3.5.4 Boyer-Moore 推广：n/3 问题

```python
# 🎯 LeetCode #229 多数元素 II（出现超过 n/3 次）
# 最多有 2 个这样的元素，维护两个候选者
def majorityElementII(nums):
    c1, c2, cnt1, cnt2 = None, None, 0, 0
    for num in nums:
        if num == c1:
            cnt1 += 1
        elif num == c2:
            cnt2 += 1
        elif cnt1 == 0:
            c1, cnt1 = num, 1
        elif cnt2 == 0:
            c2, cnt2 = num, 1
        else:
            cnt1 -= 1
            cnt2 -= 1
    # 验证候选者是否真的超过 n/3
    return [c for c in (c1, c2) if c is not None and nums.count(c) > len(nums) // 3]
```

---

## 3.6 经典题目精讲

### 题一：和为 K 的子数组（#560）

**问题**：给整数数组 `nums` 和整数 `k`，返回和为 `k` 的子数组个数。

**关键观察**：$\text{sum}(l, r) = k \Leftrightarrow \text{prefix}[r+1] - \text{prefix}[l] = k$。

```python
from collections import defaultdict

def subarraySum(nums, k):
    prefix_count = defaultdict(int)
    prefix_count[0] = 1   # 空前缀（sum=0）出现 1 次
    cur_sum = 0
    res = 0
    for num in nums:
        cur_sum += num
        # 查找有多少个 l 满足 prefix[l] = cur_sum - k
        res += prefix_count[cur_sum - k]
        prefix_count[cur_sum] += 1
    return res
# 时间 O(n)，空间 O(n)；暴力 O(n²)
```

### 题二：乘积不含自身（#238）

```python
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n
    # 前缀积（不含自身）
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    # 后缀积（不含自身）
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    return result
# 时间 O(n)，空间 O(1)（输出数组不计）
```

### 题三：最小覆盖子串（#76，困难）

见 3.4.3 节滑动窗口部分的完整实现。

---

## 3.7 本章小结

| 技术 | 查询/操作 | 预处理/额外空间 | 典型场景 |
|------|---------|---------------|---------|
| 前缀和 | $O(1)$ 区间查询 | $O(n)$ / $O(n)$ | 静态数组多次区间求和 |
| 二维前缀和 | $O(1)$ 矩形区域和 | $O(mn)$ / $O(mn)$ | 矩阵静态查询 |
| 差分数组 | $O(n)$ 还原，$O(1)$ 更新 | $O(1)$ / $O(n)$ | 批量区间更新后单次输出 |
| 双指针（对撞） | $O(n)$ | $O(1)$ | 有序数组配对问题 |
| 快慢指针 | $O(n)$ | $O(1)$ | 原地过滤/去重 |
| 滑动窗口 | $O(n)$ | $O(k)$ | 连续子数组/子串问题 |
| 摩尔投票 | $O(n)$ | $O(1)$ | 多数元素（无需排序/哈希）|

**核心设计原则**：
1. **前缀和**：用空间换时间，适合"静态数组 + 多次查询"。
2. **差分数组**：延迟计算，适合"批量更新 + 一次输出"。
3. **双指针/滑动窗口**：利用单调性，将 $O(n^2)$ 枚举降为 $O(n)$。

> **💡 思考题**：
> 1. 为什么动态数组扩容选择 2 倍而非 3 倍？3 倍是否也能保证摊销 $O(1)$？（提示：分析摊销代价 = 总复制次数/总追加次数）
> 2. 差分数组与前缀和可以组合使用吗？能同时支持 $O(1)$ 区间更新和 $O(1)$ 区间查询吗？（提示：了解树状数组/线段树）
> 3. 摩尔投票法为什么不能找"超过 $n/5$"的众数？如果要找，需要多少个候选者？

---

## 📚 参考资料

- CLRS 第4版 Chapter 10.1（基础数据结构），Chapter 17（摊销分析）
- Sedgewick《算法》第4版 1.3 节
- Skiena《算法设计手册》第3版 Chapter 3.1
- MIT 6.006 Lecture 2: Data Structures（Arrays, Linked Lists）
- LeetCode 题单：[前缀和专题](https://leetcode.cn/tag/prefix-sum/)、[双指针专题](https://leetcode.cn/tag/two-pointers/)、[滑动窗口专题](https://leetcode.cn/tag/sliding-window/)
