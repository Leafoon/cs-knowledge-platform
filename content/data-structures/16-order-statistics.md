# Chapter 16: 顺序统计量与选择算法（Order Statistics & Selection Algorithms）

> **学习目标**：  
> 理解"第 K 小"问题的本质，掌握三种解法（朴素排序、随机选择、BFPRT）及其时间复杂度的精确分析；能独立推导 RANDOMIZED-SELECT 的期望 $O(n)$ 证明（指示随机变量法）；理解 BFPRT 五元组分组策略为何保证最坏 $O(n)$；在面试与工程中能根据问题规模、数据特征和时间约束做出最优选择。

---

## 16.1 顺序统计量（Order Statistics）

### 16.1.1 什么是"顺序统计量"？

**生活比喻**：期末考试结束，老师手里有 40 份试卷，分数各不相同。家长问："我的孩子排名第 8，她的分数是多少？"这个问题就是：在 40 个数中，找出**第 8 小**的那个数。这就是**顺序统计量（Order Statistic）**问题。

**正式定义**：给定一个含 $n$ 个**互不相同**数的集合 $A$，第 $i$ 顺序统计量（$i$-th order statistic）是 **$A$ 中第 $i$ 小的元素**，记作 $A_{(i)}$。

$$A_{(1)} \leq A_{(2)} \leq \cdots \leq A_{(n)}$$

- $A_{(1)}$：**最小值**（minimum）
- $A_{(n)}$：**最大值**（maximum）
- $A_{(\lfloor (n+1)/2 \rfloor)}$：**下中位数**（lower median，又称"第 $k$ 小，$k = \lfloor(n+1)/2\rfloor$"）
- $A_{(\lceil (n+1)/2 \rceil)}$：**上中位数**（upper median）

当 $n$ 为奇数时，上下中位数相同；当 $n$ 为偶数时（如 $n=6$），下中位数是第 3 小，上中位数是第 4 小。**CLRS 通常用下中位数（$\lfloor (n+1)/2 \rfloor$）作为统一定义**。

**问题的核心**：我们的任务是找到 $A_{(i)}$——**不一定要对整个数组排序，只需找到那一个元素**。这启发我们：能不能比 $O(n \log n)$ 更快？答案是肯定的——期望 $O(n)$，甚至最坏 $O(n)$。

### 16.1.2 中位数的特殊地位

在统计学和算法中，**中位数**是顺序统计量里最常用的一个，原因在于：

1. **鲁棒性（Robustness）**：中位数不受极端值影响。工资数据中，几个亿万富翁会把**平均数**拉偏，但**中位数**反映的是普通人的真实水平。

2. **算法中的核心作用**：在快速排序和 BFPRT 算法中，选一个"好的"pivot（接近中位数的元素）是保证高效划分的关键。

3. **流式计算难题**：在数据流场景（数据一条条到来，不能完整存储）中实时维护中位数，是一个经典的优先队列应用（双堆维护法）。

**中位数的下标计算**（假设 $n$ 个元素，1-indexed）：

$$i_{\text{lower}} = \left\lfloor \frac{n+1}{2} \right\rfloor, \quad i_{\text{upper}} = \left\lceil \frac{n+1}{2} \right\rceil$$

| $n$ | 下中位数下标 | 上中位数下标 | 备注 |
|---|---|---|---|
| 5 | 3 | 3 | 奇数，唯一中位数 |
| 6 | 3 | 4 | 偶数，两个中位数 |
| 7 | 4 | 4 | 奇数，唯一中位数 |
| 8 | 4 | 5 | 偶数，两个中位数 |

### 16.1.3 朴素解法：先排序再索引

最直观的方法：**先把数组排好序，再直接读第 $i$ 个位置**。

```
朴素算法（Naive Selection）：
  输入：数组 A，目标排名 i（1-indexed）
  1. 对 A 排序（任意比较排序，如归并排序）
  2. 返回 A[i-1]（0-indexed 取第 i-1 位）
```

**时间复杂度**：$O(n \log n)$（排序主导）  
**空间复杂度**：$O(n)$（归并排序）或 $O(\log n)$（快速排序）  
**适用场景**：数组很小（$n \leq 10^4$）、或者后续需要反复查询不同排名时（排一次，多次使用）

**问题**：对于每次都需要重新查询的场景，$O(n \log n)$ 是必要的吗？考虑这个例子：

```
A = [3, 1, 9, 7, 4, 6, 2, 8, 5]，求第 5 小（即中位数）
排序后：[1, 2, 3, 4, 5, 6, 7, 8, 9]
答案是 A[4] = 5

但我们真的需要知道 1、2、3、4、6、7、8、9 在排序后的确切位置吗？
不需要！只要找到"5"就够了。
```

这正是选择算法（Selection Algorithm）比完整排序更快的根本原因。

```python
def naive_select(arr: list[int], k: int) -> int:
    """
    朴素选择：排序后取第 k 小（1-indexed）
    
    时间：O(n log n)
    空间：O(n)（Python sort 使用 TimSort，底层有 O(n) 辅助空间）
    
    参数：
        arr: 含 n 个元素的列表（不修改原数组）
        k: 目标排名，1-indexed（1 ≤ k ≤ n）
    返回：第 k 小的元素值
    """
    if not 1 <= k <= len(arr):
        raise ValueError(f"k={k} 超出范围 [1, {len(arr)}]")
    
    sorted_arr = sorted(arr)   # TimSort，O(n log n)，稳定
    return sorted_arr[k - 1]   # 1-indexed → 0-indexed

# 使用示例
arr = [3, 1, 9, 7, 4, 6, 2, 8, 5]
print(naive_select(arr, 1))   # 1（最小）
print(naive_select(arr, 5))   # 5（中位数）
print(naive_select(arr, 9))   # 9（最大）
```

```cpp
#include <vector>
#include <algorithm>
#include <stdexcept>
using namespace std;

/**
 * 朴素选择：排序后取第 k 小（1-indexed）
 * 
 * 时间：O(n log n)
 * 空间：O(n)（拷贝数组，不修改原数组）
 * 
 * @param arr 输入数组
 * @param k   目标排名，1-indexed（1 ≤ k ≤ n）
 * @return    第 k 小的元素值
 */
int naive_select(const vector<int>& arr, int k) {
    int n = static_cast<int>(arr.size());
    if (k < 1 || k > n) {
        throw invalid_argument("k 超出范围 [1, " + to_string(n) + "]");
    }
    
    vector<int> tmp(arr);          // 拷贝，不修改原数组
    sort(tmp.begin(), tmp.end());  // std::sort，introSort，O(n log n)
    return tmp[k - 1];             // 1-indexed → 0-indexed
}

// C++ 原地版本（会修改原数组）
// 使用 nth_element（内部即 QuickSelect 的标准库实现）
int nth_element_select(vector<int>& arr, int k) {
    // nth_element 保证：arr[k-1] 是第 k 小，其左边都 ≤ arr[k-1]，右边都 ≥ arr[k-1]
    // 平均 O(n)，最坏 O(n)（实现上通常用 IntroSelect，结合 BFPRT 思路）
    nth_element(arr.begin(), arr.begin() + k - 1, arr.end());
    return arr[k - 1];
}

int main() {
    vector<int> arr = {3, 1, 9, 7, 4, 6, 2, 8, 5};
    cout << naive_select(arr, 1) << endl;  // 1（最小）
    cout << naive_select(arr, 5) << endl;  // 5（中位数）
    cout << naive_select(arr, 9) << endl;  // 9（最大）
    return 0;
}
```

---

## 16.2 随机选择算法（RANDOMIZED-SELECT）

### 16.2.1 核心思路：借用快速排序的 PARTITION

回想快速排序（Chapter 15）：每次选一个 **pivot**（基准），把数组划分为：
- **左半**：所有 $< \text{pivot}$ 的元素
- **pivot**：恰好在正确位置 $q$
- **右半**：所有 $> \text{pivot}$ 的元素

划分后，我们立即知道 pivot 是**第 $q+1$ 小**的元素（0-indexed：位置 $q$ 意味着它前面有 $q$ 个比它小的元素）。

**关键洞察**：
- 如果 $q+1 = i$：找到了！直接返回 pivot
- 如果 $q+1 > i$：第 $i$ 小在**左半**，递归去左半找
- 如果 $q+1 < i$：第 $i$ 小在**右半**，递归去右半找**第 $i - (q+1)$ 小**

对比快速排序（两边都要递归）：**RANDOMIZED-SELECT 每次只递归一边**，这是它比归并排序更快的根本原因——平均只需处理一半。

```
RANDOMIZED-SELECT 伪代码（CLRS §9.2）：

RANDOMIZED-SELECT(A, p, r, i):
  // 找 A[p..r] 中第 i 小的元素
  
  if p == r:                     // 基础情况：只有一个元素
    return A[p]
  
  q ← RANDOMIZED-PARTITION(A, p, r)  // 随机选 pivot 并划分
  k ← q - p + 1                 // pivot 是当前子数组第 k 小
  
  if i == k:                     // pivot 恰好是我们要找的
    return A[q]
  elif i < k:                    // 目标在左半
    return RANDOMIZED-SELECT(A, p, q - 1, i)
  else:                          // 目标在右半（注意 i 的调整！）
    return RANDOMIZED-SELECT(A, q + 1, r, i - k)
```

**下标调整的理解**：当目标在右半时，我们在子数组 `A[q+1..r]` 中寻找第 $i - k$ 小。为什么减 $k$？因为左半（含 pivot）共 $k$ 个元素，它们都比右半小，所以原来的第 $i$ 小，在右半的局部排列中就是第 $i - k$ 小。

### 16.2.2 最坏情况：退化为 $\Theta(n^2)$

如果每次 PARTITION 都极不均匀（比如总选到最大或最小元素作为 pivot），那么：

```
A = [1, 2, 3, 4, 5, 6, 7, 8, 9]，求第 9 小（最大值）
不巧每次选到最小元素作为 pivot：

第 1 次划分：pivot=1，左=[空]，右=[2,3,4,5,6,7,8,9]，处理 n=9 个
第 2 次划分：pivot=2，左=[空]，右=[3,4,5,6,7,8,9]，处理 n=8 个
...
共处理 n + (n-1) + ... + 1 = Θ(n²) 次比较
```

**最坏时间**：$\Theta(n^2)$——与朴素排序一样糟糕（但比排序更糟，因为排序是 $O(n \log n)$！）

这就是为什么需要**随机化**：通过在 PARTITION 前随机选 pivot（均匀随机从 $[p, r]$ 中选一个），使得没有特定输入能稳定触发最坏情况。

### 16.2.3 期望时间 $O(n)$ 的严格证明

这是本章最重要的理论结果。我们用**指示随机变量（Indicator Random Variables）**来证明。

**设 $T(n)$ 为在大小为 $n$ 的数组上运行 RANDOMIZED-SELECT 的期望比较次数**。

**关键简化假设**：pivot 等可能地是第 1 小到第 $n$ 小的任何一个（随机化保证了这一点）。设 $X_k$（$k = 1, 2, \ldots, n$）为指示随机变量：

$$X_k = \mathbb{1}[\text{pivot 是第 } k \text{ 小}], \quad \Pr[X_k = 1] = \frac{1}{n}$$

当 pivot 是第 $k$ 小时，我们递归到更小的子问题：
- 若 $i < k$：递归子问题大小为 $k - 1$
- 若 $i > k$：递归子问题大小为 $n - k$
- 若 $i = k$：直接找到，递归大小为 $0$

**上界分析**（最保守估计：不假设知道 $i$，每次取两个方向的较大者）：

$$T(n) \leq \sum_{k=1}^{n} \frac{1}{n} \left( T(\max(k-1, n-k)) \right) + \Theta(n)$$

注意 $\max(k-1, n-k) \leq \max\left(\frac{n-1}{2}, \frac{n}{2}\right) = \frac{n}{2}$，当 $k$ 接近中间时。

更精确地：$\max(k-1, n-k)$ 在 $k \leq \lceil n/2 \rceil$ 时等于 $n-k$，在 $k > \lceil n/2 \rceil$ 时等于 $k-1$。对 $k$ 从 1 到 $n$ 求均值：

$$T(n) \leq \frac{1}{n} \sum_{k=\lceil n/2 \rceil}^{n-1} T(k) + \Theta(n)$$

（因为当 $k$ 在 $[\lceil n/2 \rceil, n-1]$ 范围时，$\max(k-1, n-k) = k$，而每个 $k$ 出现两次：一次作为 $k$，一次作为 $n-k$，合并后求和范围为 $\lceil n/2 \rceil$ 到 $n-1$）

**代入法证明 $T(n) = O(n)$**：设 $T(n) \leq cn$（$c$ 是待定常数）。代入：

$$T(n) \leq \frac{1}{n} \sum_{k=\lceil n/2 \rceil}^{n-1} ck + \Theta(n) = \frac{c}{n} \cdot \sum_{k=\lceil n/2 \rceil}^{n-1} k + \Theta(n)$$

$$\sum_{k=\lceil n/2 \rceil}^{n-1} k = \sum_{k=1}^{n-1}k - \sum_{k=1}^{\lceil n/2 \rceil - 1}k \leq \frac{(n-1)n}{2} - \frac{(\lfloor n/2 \rfloor)(\lfloor n/2 \rfloor + 1)}{2} \leq \frac{3n^2}{8}$$

所以：

$$T(n) \leq \frac{c}{n} \cdot \frac{3n^2}{8} + \Theta(n) = \frac{3cn}{8} + \Theta(n)$$

只要 $c$ 足够大（$c \geq \frac{\Theta(1)}{1 - 3/8} = \frac{8}{5} \cdot \Theta(1)$），就有 $T(n) \leq cn$。✓

**结论**：$\mathbb{E}[T(n)] = O(n)$，即 RANDOMIZED-SELECT 的**期望时间复杂度为 $O(n)$**。

> ⚠️ **陷阱**：这里证明的是**期望**，不是最坏。最坏仍是 $\Theta(n^2)$。但在实践中，触发最坏情况的概率极低（可以用鞅论证明高概率下不超过 $O(n \log n)$）。

### 16.2.4 Python 与 C++ 完整实现

实现上有两种风格：**原地版**（修改数组，$O(1)$ 额外空间）和**拷贝版**（不修改原数组，$O(n)$ 空间）。工程中通常用原地版。

```python
import random

def randomized_select(arr: list[int], k: int) -> int:
    """
    随机化选择：期望 O(n) 时间找第 k 小元素（1-indexed）

    时间复杂度：期望 O(n)，最坏 Θ(n²)（极低概率）
    空间复杂度：O(log n) 期望递归栈深度（最坏 O(n)）
    稳定性：不稳定（PARTITION 过程会移动元素）

    参数：
        arr: 待查询列表（会被修改！如不希望修改，请传 arr[:] 的拷贝）
        k: 目标排名，1-indexed（1 ≤ k ≤ len(arr)）
    返回：第 k 小的元素值
    """
    if not 1 <= k <= len(arr):
        raise ValueError(f"k={k} 超出范围")
    
    # 在子数组 arr[left..right] 中找第 i 小（内部 0-indexed）
    def _select(left: int, right: int, i: int) -> int:
        # 基础情况：只有一个元素
        if left == right:
            return arr[left]
        
        # 随机选 pivot 并移到末尾
        pivot_idx = random.randint(left, right)
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        
        # Lomuto 划分（pivot = arr[right]）
        pivot = arr[right]
        store = left  # store 指向"小于 pivot 区域"的下一个空位
        
        for j in range(left, right):
            if arr[j] <= pivot:
                arr[store], arr[j] = arr[j], arr[store]
                store += 1
        
        # 把 pivot 放到最终位置
        arr[store], arr[right] = arr[right], arr[store]
        pivot_pos = store   # pivot 现在在 arr[pivot_pos]
        
        # pivot 在当前子数组中是第 k_local 小
        k_local = pivot_pos - left + 1  # 1-indexed，相对于 arr[left]
        
        if i == k_local:
            return arr[pivot_pos]
        elif i < k_local:
            return _select(left, pivot_pos - 1, i)       # 去左半
        else:
            return _select(pivot_pos + 1, right, i - k_local)  # 去右半，调整排名
    
    # 注意：传入 k 本身就是 1-indexed，内部 i 参数也用 1-indexed
    return _select(0, len(arr) - 1, k)


# ——— 迭代版（避免栈溢出，适合大数组）———
def randomized_select_iterative(arr: list[int], k: int) -> int:
    """
    迭代版随机选择，避免极端情况下的递归栈溢出。
    时间/空间同递归版，但规避了 Python 默认递归深度限制。
    """
    arr = arr[:]   # 拷贝，不修改原数组
    left, right, i = 0, len(arr) - 1, k
    
    while left < right:
        # 随机 pivot，Lomuto 划分
        pivot_idx = random.randint(left, right)
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        
        pivot, store = arr[right], left
        for j in range(left, right):
            if arr[j] <= pivot:
                arr[store], arr[j] = arr[j], arr[store]
                store += 1
        arr[store], arr[right] = arr[right], arr[store]
        
        k_local = store - left + 1
        
        if i == k_local:
            return arr[store]
        elif i < k_local:
            right = store - 1           # 缩小右边界
        else:
            i -= k_local                # 调整排名
            left = store + 1            # 缩小左边界
    
    return arr[left]


# ——— 使用示例 ———
arr = [3, 1, 9, 7, 4, 6, 2, 8, 5]
print(randomized_select(arr[:], 1))   # 1（最小，用拷贝避免修改）
print(randomized_select(arr[:], 5))   # 5（中位数）
print(randomized_select(arr[:], 9))   # 9（最大）

# 迭代版：直接传原数组（内部已拷贝）
print(randomized_select_iterative(arr, 3))   # 3（第3小）
```

```cpp
#include <vector>
#include <random>
#include <stdexcept>
using namespace std;

// 全局随机引擎（比 rand() 质量更好，线程安全问题需注意）
static mt19937 rng(42);  // 固定种子，调试用；生产中用 random_device{}()

/**
 * Lomuto 随机化划分：选随机 pivot，划分 arr[left..right]
 * 返回：pivot 的最终位置（0-indexed）
 * 时间：O(right - left + 1)，修改 arr
 */
int randomized_partition(vector<int>& arr, int left, int right) {
    // 随机选 pivot 索引，换到末尾
    uniform_int_distribution<int> dist(left, right);
    int pivot_idx = dist(rng);
    swap(arr[pivot_idx], arr[right]);
    
    int pivot = arr[right];
    int store = left;   // "小于 pivot 区"的下一个空位
    
    for (int j = left; j < right; ++j) {
        if (arr[j] <= pivot) {
            swap(arr[store], arr[j]);
            ++store;
        }
    }
    swap(arr[store], arr[right]);   // pivot 就位
    return store;
}

/**
 * RANDOMIZED-SELECT（递归版）
 * 在 arr[left..right] 中找第 i 小（1-indexed，相对于左端）
 * 
 * 期望时间：O(n)，最坏 Θ(n²)
 * 空间：O(log n) 期望递归栈
 */
int randomized_select_helper(vector<int>& arr, int left, int right, int i) {
    if (left == right) return arr[left];   // 仅一个元素
    
    int q = randomized_partition(arr, left, right);
    int k = q - left + 1;   // pivot 是当前子数组第 k 小（1-indexed）
    
    if (i == k) return arr[q];
    else if (i < k) return randomized_select_helper(arr, left, q - 1, i);
    else             return randomized_select_helper(arr, q + 1, right, i - k);
}

/**
 * 对外接口：在 arr 中找第 k 小（1-indexed，修改原数组）
 */
int randomized_select(vector<int>& arr, int k) {
    int n = static_cast<int>(arr.size());
    if (k < 1 || k > n) throw invalid_argument("k 超出范围");
    return randomized_select_helper(arr, 0, n - 1, k);
}

/**
 * 迭代版 RANDOMIZED-SELECT（避免递归栈溢出，推荐用于生产环境）
 * 
 * 期望时间：O(n)，空间：O(1)（除数组外）
 */
int randomized_select_iterative(vector<int> arr, int k) {  // 传值，内部拷贝
    int n = static_cast<int>(arr.size());
    if (k < 1 || k > n) throw invalid_argument("k 超出范围");
    
    int left = 0, right = n - 1, i = k;
    
    while (left < right) {
        int q = randomized_partition(arr, left, right);
        int kk = q - left + 1;
        
        if (i == kk) return arr[q];
        else if (i < kk) right = q - 1;
        else { i -= kk; left = q + 1; }
    }
    return arr[left];
}

int main() {
    vector<int> arr = {3, 1, 9, 7, 4, 6, 2, 8, 5};
    
    vector<int> a1 = arr;
    cout << randomized_select(a1, 1) << endl;  // 1

    vector<int> a2 = arr;
    cout << randomized_select(a2, 5) << endl;  // 5

    cout << randomized_select_iterative(arr, 9) << endl;  // 9
    return 0;
}
```

<div data-component="RandomSelectTrace"></div>

---

## 16.3 线性时间选择：BFPRT 算法（中位数的中位数）

### 16.3.1 为什么 RANDOMIZED-SELECT 还不够好？

RANDOMIZED-SELECT 的期望是 $O(n)$，但它有**两个软肋**：

1. **最坏情况是 $\Theta(n^2)$**：虽然概率极低，但在对可靠性要求极高的场景（实时系统、嵌入式、竞赛）中，这不可接受。
2. **随机源**：依赖高质量随机数，有时候输入数据本身就是对随机 pivot 的"攻击"（adversarial input）。

**BFPRT 算法**（1973年，由 Blum、Floyd、Pratt、Rivest、Tarjan 五人提出，因此也叫"五位作者算法"或"Median-of-Medians"）解决了这个问题：**确定性地**选出一个"足够好"的 pivot，保证最坏情况 $O(n)$。

**核心思想**：不通过随机性，而是通过**精心设计的确定性策略**，保证 pivot **至少比 30% 的元素大，又至少比 30% 的元素小**。这样每次划分后，递归子问题的规模至多是原来的 70%，从而保证对数深度后总时间 $O(n)$。

### 16.3.2 五元组分组策略

**步骤一**：把 $n$ 个元素**每 5 个分成一组**（最后一组可能不足 5 个）：

```
n = 17 个元素，分成 4 组：
组 1：[a₁, a₂, a₃, a₄, a₅]   5 个元素
组 2：[a₆, a₇, a₈, a₉, a₁₀]  5 个元素
组 3：[a₁₁, a₁₂, a₁₃, a₁₄, a₁₅] 5 个元素
组 4：[a₁₆, a₁₇]              2 个元素（最后一组可能不足5个）

共 ⌈17/5⌉ = 4 组
```

**为什么选 5？** 这是一个精心权衡的常数：
- 太小（如 3）：保证的 pivot 质量不够好，无法得到 $O(n)$
- 太大（如 7）：组内排序代价增大，常数因子变大，实践变慢
- 5 是满足 $O(n)$ 最坏保证的**最小奇数**，且常数因子相对合理

**步骤二**：对**每一组**（至多 5 个元素）进行**插入排序**，找出各组的**中位数**（每组第 $\lceil 5/2 \rceil = 3$ 小的元素）：

```
组 1 排序后：[a₁≤a₂≤a₃≤a₄≤a₅]  →  中位数 = a₃（第3小）
组 2 排序后：[a₆≤a₇≤a₈≤a₉≤a₁₀] →  中位数 = a₈（第3小）
...
```

每组最多 5 个元素，插入排序代价为 $O(1)$（最多 $\binom{5}{2} = 10$ 次比较），共 $\lceil n/5 \rceil$ 组，所以这一步总耗时 $O(n)$。

**步骤三**：把所有组的中位数组成一个新数组 $M$（共 $\lceil n/5 \rceil$ 个元素），对 $M$ **递归调用 BFPRT 本身**，找出 $M$ 的中位数，记为 $m^*$（**中位数的中位数，Median of Medians**）。

<div data-component="BFPRTGrouping"></div>

### 16.3.3 为什么 $m^*$ 是一个"好的" pivot？

设 $m^*$ 是 $M$ 的中位数。由于 $|M| = \lceil n/5 \rceil$，$m^*$ 在 $M$ 中**至少比 $\lceil \lceil n/5 \rceil / 2 \rceil$ 个组的中位数大**（因为它是 $M$ 的中位数）。

设至少有 $t = \lceil \lceil n/5 \rceil / 2 \rceil \geq n/10$ 个组，其组中位数 $\leq m^*$。对于这些组：
- **该组中位数 $\leq m^*$**：说明该组有至少 3 个元素（中位数及其左边两个）$\leq m^*$

所以，**至少有 $3t \geq 3n/10$ 个元素 $\leq m^*$**。对称地，**至少有 $3n/10$ 个元素 $\geq m^*$**。

**结论**：以 $m^*$ 为 pivot 进行 PARTITION 后：
- 大于 $m^*$ 的元素：至多 $n - 3n/10 = 7n/10$ 个
- 小于 $m^*$ 的元素：至多 $7n/10$ 个

因此，两个子问题中**较大的那个**至多有 $7n/10 + 6$ 个元素（$+6$ 是处理不足 5 的最后一组的边界修正，参见 CLRS 书中的精确计算）。

**直觉可视化**：设 $n = 50$（10 组，每组 5 个），对每组排序并取中位数：

```
10个组中位数组成数组M，m* 是M的中位数（第5小）

                          m* 在M中
            ≤ m*的中位数       ≥ m*的中位数
            [组1] [组2] [组3] [组4] [组5]  [组6][组7][组8][组9][组10]
              ↓     ↓     ↓     ↓     ↓
每组排好序后：
[小 中 大]   [小 中 大]  [小 中 大]  ...（其中"中"= 该组中位数）

对 ≤ m* 的5个中位数所在组：每组至少3个元素 ≤ m*
→ 至少 5×3 = 15 个元素 ≤ m*（占50个的30%）

对 ≥ m* 的5个中位数所在组：每组至少3个元素 ≥ m*
→ 至少 5×3 = 15 个元素 ≥ m*（占50个的30%）

所以 PARTITION 后，两侧各至多 n - 15 = 35 ≤ 7n/10 个元素
```

### 16.3.4 时间复杂度递推与证明

BFPRT 的运行时间满足：

$$T(n) \leq T\!\left(\left\lceil \frac{n}{5} \right\rceil\right) + T\!\left(\frac{7n}{10} + 6\right) + O(n)$$

其中：
- $T(\lceil n/5 \rceil)$：对 $\lceil n/5 \rceil$ 个中位数递归调用 BFPRT 找 $m^*$
- $T(7n/10 + 6)$：递归到 PARTITION 后较大的那一侧
- $O(n)$：分组 + 各组排序 + PARTITION 本身

**代入法证明 $T(n) = O(n)$**：设 $T(n) \leq cn$ 对所有 $n \geq n_0$ 成立。代入：

$$T(n) \leq c \cdot \frac{n}{5} + c \cdot \left(\frac{7n}{10} + 6\right) + an$$

$$= cn \cdot \left(\frac{1}{5} + \frac{7}{10}\right) + 6c + an$$

$$= \frac{9cn}{10} + 6c + an$$

$$\leq cn \quad \Longleftrightarrow \quad \frac{9cn}{10} + 6c + an \leq cn$$

$$\Longleftrightarrow \quad 6c + an \leq \frac{cn}{10}$$

$$\Longleftrightarrow \quad c \geq \frac{10an + 60c}{n} = 10a + \frac{60c}{n}$$

当 $n \geq 120$（使 $60/n \leq 1/2$）时，只需 $c \geq 20a$（$a$ 是 $O(n)$ 的常数因子）即可满足。✓

**结论**：$T(n) = O(n)$，BFPRT 保证最坏情况线性时间！

> 📌 **关键点**：递推中的两项 $n/5$ 和 $7n/10$ 之和为 $9n/10 < n$，这保证了递推能"收敛"。如果选 3 元组（$T(n/3 + T(2n/3) + O(n)$），由于 $1/3 + 2/3 = 1$，这就成了 $T(n) = O(n \log n)$ 了！**5元组是能保证 $O(n)$ 的最小奇数分组大小**。

### 16.3.5 BFPRT 完整实现

```python
def insertion_sort_small(arr: list, left: int, right: int) -> None:
    """对 arr[left..right] 做插入排序（最多5个元素，O(1)）"""
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def bfprt(arr: list[int], k: int) -> int:
    """
    BFPRT（中位数的中位数）选择算法
    
    最坏时间：O(n)
    空间：O(log n) 递归栈 + O(n) 辅助（存中位数数组）
    
    参数：
        arr: 输入数组（会被修改！）
        k: 目标排名，1-indexed
    返回：第 k 小的元素值
    """
    def _bfprt(a: list, left: int, right: int, i: int) -> int:
        n = right - left + 1
        
        # 基础情况：≤5个元素，直接排序
        if n <= 5:
            insertion_sort_small(a, left, right)
            return a[left + i - 1]   # 第 i 小（1-indexed → left+i-1）
        
        # ─── 步骤1：每5个元素分一组，求各组中位数 ───
        medians = []
        g = left
        while g <= right:
            group_end = min(g + 4, right)     # 最后一组可能不足5个
            insertion_sort_small(a, g, group_end)
            group_size = group_end - g + 1
            median_idx = g + (group_size - 1) // 2   # 中位数下标（0-indexed）
            medians.append(a[median_idx])
            g += 5
        
        # ─── 步骤2：递归求中位数的中位数 m* ───
        median_of_medians = _bfprt(medians, 0, len(medians) - 1,
                                    (len(medians) + 1) // 2)   # 下中位数
        
        # ─── 步骤3：用 m* 做 PARTITION ───
        # 先把 m* 移到末尾
        for idx in range(left, right + 1):
            if a[idx] == median_of_medians:
                a[idx], a[right] = a[right], a[idx]
                break
        
        # Lomuto 划分
        pivot = a[right]
        store = left
        for j in range(left, right):
            if a[j] <= pivot:
                a[store], a[j] = a[j], a[store]
                store += 1
        a[store], a[right] = a[right], a[store]
        
        # ─── 步骤4：判断目标在哪侧 ───
        k_local = store - left + 1   # pivot 是第 k_local 小（相对当前子数组）
        
        if i == k_local:
            return a[store]
        elif i < k_local:
            return _bfprt(a, left, store - 1, i)
        else:
            return _bfprt(a, store + 1, right, i - k_local)
    
    if not 1 <= k <= len(arr):
        raise ValueError(f"k={k} 超出范围")
    
    arr_copy = arr[:]   # 不修改原数组
    return _bfprt(arr_copy, 0, len(arr_copy) - 1, k)


# 使用示例
arr = [3, 1, 9, 7, 4, 6, 2, 8, 5]
for k in range(1, 10):
    print(f"第{k}小: {bfprt(arr, k)}", end="  ")
# 第1小: 1  第2小: 2  第3小: 3  第4小: 4  第5小: 5  第6小: 6  第7小: 7  第8小: 8  第9小: 9
```

```cpp
#include <vector>
#include <algorithm>
#include <stdexcept>
using namespace std;

/**
 * 对 arr[left..right] 做插入排序（最多5个元素，O(1) 比较代价）
 */
void insertion_sort_small(vector<int>& arr, int left, int right) {
    for (int i = left + 1; i <= right; ++i) {
        int key = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

/**
 * BFPRT 内部递归实现
 * 
 * @param arr    工作数组（会被修改）
 * @param left   子数组左端
 * @param right  子数组右端
 * @param i      目标排名（1-indexed，相对于 left）
 * @return       第 i 小的元素值
 */
int bfprt_helper(vector<int>& arr, int left, int right, int i) {
    int n = right - left + 1;
    
    // 基础情况：≤5个元素直接排序
    if (n <= 5) {
        insertion_sort_small(arr, left, right);
        return arr[left + i - 1];  // 1-indexed → left + i - 1
    }
    
    // ─── 步骤1：每5个元素一组，求各组中位数 ───
    vector<int> medians;
    for (int g = left; g <= right; g += 5) {
        int group_end = min(g + 4, right);
        insertion_sort_small(arr, g, group_end);
        int group_size = group_end - g + 1;
        int median_idx = g + (group_size - 1) / 2;  // 下中位数（0-indexed）
        medians.push_back(arr[median_idx]);
    }
    
    // ─── 步骤2：递归求中位数的中位数 m* ───
    int m = static_cast<int>(medians.size());
    int mom = bfprt_helper(medians, 0, m - 1, (m + 1) / 2);  // 下中位数
    
    // ─── 步骤3：找到 mom 的位置，换到末尾，做 Lomuto 划分 ───
    for (int idx = left; idx <= right; ++idx) {
        if (arr[idx] == mom) {
            swap(arr[idx], arr[right]);
            break;
        }
    }
    
    int pivot = arr[right];
    int store = left;
    for (int j = left; j < right; ++j) {
        if (arr[j] <= pivot) {
            swap(arr[store], arr[j]);
            ++store;
        }
    }
    swap(arr[store], arr[right]);
    
    // ─── 步骤4：递归到目标侧 ───
    int k_local = store - left + 1;  // pivot 在当前子数组中是第 k_local 小
    
    if (i == k_local)      return arr[store];
    else if (i < k_local)  return bfprt_helper(arr, left, store - 1, i);
    else                   return bfprt_helper(arr, store + 1, right, i - k_local);
}

/**
 * BFPRT 对外接口
 * 
 * 最坏时间：O(n)（无需随机，确定性保证）
 * 空间：O(log n) 递归栈 + O(n) 辅助（中位数数组）
 */
int bfprt(const vector<int>& arr, int k) {
    int n = static_cast<int>(arr.size());
    if (k < 1 || k > n) throw invalid_argument("k 超出范围");
    
    vector<int> tmp(arr);  // 拷贝，不修改原数组
    return bfprt_helper(tmp, 0, n - 1, k);
}

int main() {
    vector<int> arr = {3, 1, 9, 7, 4, 6, 2, 8, 5};
    for (int k = 1; k <= 9; ++k) {
        cout << "第" << k << "小: " << bfprt(arr, k) << "  ";
    }
    // 第1小: 1  第2小: 2  ...  第9小: 9
    cout << endl;
    return 0;
}
```

### 16.3.6 BFPRT vs RANDOMIZED-SELECT：工程权衡

BFPRT 的最坏 $O(n)$ 在理论上无懈可击，但在工程中几乎从不使用。为什么？

| 对比维度 | RANDOMIZED-SELECT | BFPRT |
|---|---|---|
| 最坏时间 | $\Theta(n^2)$（极低概率） | $O(n)$（确定性保证） |
| 期望时间 | $O(n)$ | $O(n)$ |
| 实际常数因子 | 小（约 $3.38n$ 次比较） | 大（约 $22n$ 次比较） |
| 实现复杂度 | 简单（20行代码） | 复杂（多层递归调用） |
| 缓存局部性 | 较好（原地划分） | 较差（辅助数组跳跃访问） |
| 随机数依赖 | 是 | 否 |
| 适用场景 | 几乎所有工程场景 | 实时系统/安全关键/竞赛（最坏保证必须） |

> 💡 **思考题**：BFPRT 的比较次数常数因子约为 22（相比 RANDOMIZED-SELECT 的约 3.38），这是 6 倍的差距。对于 $n = 10^6$，RANDOMIZED-SELECT 做约 $3.38 \times 10^6$ 次比较，BFPRT 做约 $2.2 \times 10^7$ 次。在触发最坏情况概率远小于 $10^{-6}$ 的情况下，RANDOMIZED-SELECT 的"期望成本"远低于 BFPRT 的"最坏成本"，工程上自然选前者。

---

## 16.4 实际应用

### 16.4.1 QuickSelect：面试与工程的标准答案

**QuickSelect** 是 RANDOMIZED-SELECT 的工程化版本，是 C++ 标准库 `std::nth_element` 的核心思想。它的特点是：

1. **通常用三路划分**（Three-Way Partition）处理重复元素
2. **结合小数组优化**：当子数组小于某个阈值（如 20 个元素）时，切换为插入排序
3. **Median-of-Three**：从 `arr[left]`、`arr[mid]`、`arr[right]` 三个候选中选中间值作为 pivot，减少极端情况概率

```python
def quickselect(nums: list[int], k: int) -> int:
    """
    工程化 QuickSelect（三路划分 + 随机化 + 小数组插排）
    
    面试标准写法：简洁、高效、处理重复元素
    时间：期望 O(n)，空间：O(log n) 期望（递归栈）
    
    参数：
        nums: 输入列表（会被修改）
        k: 目标排名，1-indexed
    """
    import random
    
    def partition3(left: int, right: int, pivot_val: int):
        """
        三路划分（Dutch National Flag）：
        返回 (lt, gt)，使得：
          nums[left..lt-1] < pivot_val
          nums[lt..gt]     == pivot_val
          nums[gt+1..right] > pivot_val
        """
        lt, i, gt = left, left, right
        while i <= gt:
            if nums[i] < pivot_val:
                nums[lt], nums[i] = nums[i], nums[lt]
                lt += 1; i += 1
            elif nums[i] > pivot_val:
                nums[i], nums[gt] = nums[gt], nums[i]
                gt -= 1
            else:
                i += 1
        return lt, gt
    
    def _qs(left: int, right: int, i: int) -> int:
        if left >= right:
            return nums[left]
        
        # 随机化 pivot（防止顺序输入退化）
        pivot_idx = random.randint(left, right)
        pivot_val = nums[pivot_idx]
        
        lt, gt = partition3(left, right, pivot_val)
        # 现在：nums[left..lt-1] < pivot，nums[lt..gt] == pivot，nums[gt+1..right] > pivot
        
        # 目标排名 i（1-indexed，相对 left）
        count_left  = lt - left          # 比 pivot 小的元素数
        count_equal = gt - lt + 1        # 等于 pivot 的元素数
        
        if i <= count_left:
            return _qs(left, lt - 1, i)                # 去左边
        elif i <= count_left + count_equal:
            return pivot_val                            # 在 pivot 区
        else:
            return _qs(gt + 1, right, i - count_left - count_equal)  # 去右边
    
    nums = nums[:]   # 拷贝
    return _qs(0, len(nums) - 1, k)


# ——— LeetCode #215 第 K 个最大元素 ———
def findKthLargest(nums: list[int], k: int) -> int:
    """
    第 K 大 = 第 (n - k + 1) 小
    时间：期望 O(n)，面试最优解
    """
    n = len(nums)
    return quickselect(nums, n - k + 1)

# 测试
print(findKthLargest([3, 2, 1, 5, 6, 4], 2))   # 5（第2大）
print(findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # 4（第4大）
```

```cpp
#include <vector>
#include <tuple>
#include <random>
#include <algorithm>
using namespace std;

static mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

/**
 * 三路划分（Dutch National Flag）
 * 将 nums[left..right] 分为三段：< pivot | == pivot | > pivot
 * 返回 (lt, gt)：== pivot 的范围是 [lt, gt]
 */
pair<int,int> partition3(vector<int>& nums, int left, int right, int pivot_val) {
    int lt = left, i = left, gt = right;
    while (i <= gt) {
        if      (nums[i] < pivot_val) { swap(nums[lt++], nums[i++]); }
        else if (nums[i] > pivot_val) { swap(nums[i],    nums[gt--]); }
        else                          { ++i; }
    }
    return {lt, gt};
}

/**
 * 工程化 QuickSelect（三路划分 + 随机化）
 * 
 * @param nums  工作数组（会被修改）
 * @param left  子数组左端
 * @param right 子数组右端
 * @param i     目标排名（1-indexed，相对 left）
 * @return      第 i 小的元素值
 */
int quickselect_helper(vector<int>& nums, int left, int right, int i) {
    if (left >= right) return nums[left];
    
    // 随机化 pivot
    uniform_int_distribution<int> dist(left, right);
    int pivot_val = nums[dist(rng)];
    
    auto [lt, gt] = partition3(nums, left, right, pivot_val);
    
    int count_left  = lt - left;         // 小于 pivot 的元素数
    int count_equal = gt - lt + 1;       // 等于 pivot 的元素数
    
    if      (i <= count_left)                   return quickselect_helper(nums, left, lt - 1, i);
    else if (i <= count_left + count_equal)     return pivot_val;
    else    return quickselect_helper(nums, gt + 1, right, i - count_left - count_equal);
}

int quickselect(const vector<int>& nums, int k) {
    vector<int> tmp(nums);
    return quickselect_helper(tmp, 0, (int)tmp.size() - 1, k);
}

// LeetCode #215: 第 K 个最大元素
int findKthLargest(vector<int>& nums, int k) {
    int n = nums.size();
    return quickselect(nums, n - k + 1);   // 第 k 大 = 第 (n-k+1) 小
}

int main() {
    vector<int> a1 = {3, 2, 1, 5, 6, 4};
    cout << findKthLargest(a1, 2) << endl;   // 5

    vector<int> a2 = {3, 2, 3, 1, 2, 4, 5, 5, 6};
    cout << findKthLargest(a2, 4) << endl;   // 4
    return 0;
}
```

### 16.4.2 Top-K 问题：四种方法的全面对比

**问题描述**：给定 $n$ 个数，找出其中**最大的 $k$ 个数**（或最小的 $k$ 个数）。

**方法一：全排序**  
将数组完整排序，取前 $k$ 个。  
时间 $O(n \log n)$，空间 $O(1)$（原地排序）或 $O(n)$（归并排序）。  
**适用**：$k$ 接近 $n$，或后续还需要整个有序数组。

**方法二：最小堆（Min-Heap）**  
维护一个大小为 $k$ 的最小堆：遍历数组，若当前元素大于堆顶，则替换堆顶并调整。  
时间 $O(n \log k)$，空间 $O(k)$。  
**适用**：$k$ 远小于 $n$（如 $k = O(\sqrt{n})$）、**流式数据**（不能一次性存入内存）。

**方法三：QuickSelect**  
先用选择算法找到第 $(n-k+1)$ 小的数（即第 $k$ 大），然后一次线性扫描收集所有 $\geq$ 此数的元素。  
期望时间 $O(n)$，空间 $O(1)$（原地）。  
**适用**：$k$ 占 $n$ 的一定比例、数据可以一次性加载进内存、需要最优期望性能。

**方法四：BFPRT**  
同 QuickSelect，但保证最坏 $O(n)$。  
**适用**：对最坏情况要求严格的系统（实时控制、安全关键）。

**定量对比**（$n = 10^8$，$k = 10^3$，现代 CPU 约 $10^9$ 次操作/秒）：

| 方法 | 时间（估算） | 空间 | 是否稳定 | 适用场景 |
|---|---|---|---|---|
| 全排序（快速排序） | $\approx 2.7$ 秒 | $O(\log n)$ | ❌不稳定 | $k \approx n$ 或需完整排序 |
| 归并排序 | $\approx 2.7$ 秒 | $O(n)$ | ✅稳定 | 需要稳定 + 完整排序 |
| 堆（大小 $k$） | $\approx 0.7$ 秒 | $O(k)=O(10^3)$ | ❌不稳定 | 流式数据或 $k \ll n$ |
| **QuickSelect** | **$\approx 0.3$ 秒** | $O(1)$ | ❌不稳定 | **在线数据，最有效** |
| BFPRT | $\approx 2.2$ 秒 | $O(\log n)$ | ❌不稳定 | 最坏保证必须 |

> 注：为什么堆（$O(n \log k)$）在 $k=10^3$ 时比 QuickSelect（$O(n)$）还要快？因为 $\log(10^3) \approx 10$，$O(n \log k) \approx 10n$，而 QuickSelect 的实际常数因子约 $3.38$，所以 $10n > 3.38n$，QuickSelect 仍更快。当 $k$ 更小（如 $k = 10$）时，堆的 $O(n \log k) \approx 3.32n$ 与 QuickSelect $3.38n$ 几乎相当。堆的优势在于**流式/外存场景**，QuickSelect 的优势在于**随机访问内存 + $k$ 较大**时。

```python
import heapq
import random

def top_k_methods_comparison(nums: list[int], k: int):
    """演示 Top-K 的四种方法，对比结果是否一致"""
    n = len(nums)
    
    # 方法一：全排序
    sorted_result = sorted(nums, reverse=True)[:k]
    
    # 方法二：最小堆
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:   # 当前元素比堆中最小的还大
            heapq.heapreplace(heap, x)
    heap_result = sorted(heap, reverse=True)
    
    # 方法三：QuickSelect（先找第(n-k+1)小，再扫描）
    nums_copy = nums[:]
    threshold = quickselect(nums_copy, n - k + 1)   # 第 k 大的值
    qs_result = sorted([x for x in nums if x >= threshold], reverse=True)[:k]
    
    print(f"全排序结果:    {sorted_result}")
    print(f"堆方法结果:    {sorted(heap_result, reverse=True)}")
    print(f"QuickSelect:  {qs_result}")
    print(f"三种结果一致: {sorted_result == sorted(heap, reverse=True) == qs_result}")

# 测试
nums = [random.randint(1, 100) for _ in range(20)]
print(f"数组: {nums}")
top_k_methods_comparison(nums, k=5)
```

```cpp
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
using namespace std;

// ——— 方法二：最小堆（标准库 priority_queue）———
vector<int> top_k_heap(const vector<int>& nums, int k) {
    // STL priority_queue 默认最大堆；用 greater<int> 变为最小堆
    priority_queue<int, vector<int>, greater<int>> min_heap;
    
    for (int x : nums) {
        if ((int)min_heap.size() < k) {
            min_heap.push(x);
        } else if (x > min_heap.top()) {
            min_heap.pop();
            min_heap.push(x);
        }
    }
    
    vector<int> result;
    while (!min_heap.empty()) {
        result.push_back(min_heap.top());
        min_heap.pop();
    }
    sort(result.rbegin(), result.rend());   // 降序
    return result;
}

// ——— 方法三：QuickSelect ———
vector<int> top_k_quickselect(vector<int> nums, int k) {
    int n = nums.size();
    int threshold = quickselect(nums, n - k + 1);   // 第 k 大的阈值
    
    vector<int> result;
    for (int x : nums) {
        if (x >= threshold) result.push_back(x);
    }
    sort(result.rbegin(), result.rend());
    return result;
}

int main() {
    vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    int k = 4;
    
    // 方法一：全排序
    vector<int> tmp = nums;
    sort(tmp.rbegin(), tmp.rend());  // 降序
    cout << "全排序 Top-" << k << ": ";
    for (int i = 0; i < k; ++i) cout << tmp[i] << " ";
    cout << endl;   // 9 6 5 5
    
    // 方法二：堆
    auto heap_res = top_k_heap(nums, k);
    cout << "堆方法 Top-" << k << ": ";
    for (int x : heap_res) cout << x << " ";
    cout << endl;
    
    // 方法三：QuickSelect
    auto qs_res = top_k_quickselect(nums, k);
    cout << "QuickSelect Top-" << k << ": ";
    for (int x : qs_res) cout << x << " ";
    cout << endl;
    
    return 0;
}
```

<div data-component="TopKComparison"></div>

### 16.4.3 数据流中位数：双堆动态维护

这是结合堆与顺序统计量的经典综合题（LeetCode #295）。如果数据是流式到来的（不能完整存储），如何实时维护当前数据的**中位数**？

**策略**：用**两个堆**将数据分为"左半部分"和"右半部分"：
- `Max_Heap`（最大堆）：维护左半数据（较小的那一半），堆顶 = 左半最大值
- `Min_Heap`（最小堆）：维护右半数据（较大的那一半），堆顶 = 右半最小值

**不变量**：
1. `len(Max_Heap) >= len(Min_Heap)`（左半比右半多不超过 1 个）
2. `Max_Heap.top() <= Min_Heap.top()`（左半最大 ≤ 右半最小）

**中位数**：
- $n$ 为奇数：`Max_Heap.top()`（左半有 1 个"多出"的元素）
- $n$ 为偶数：`(Max_Heap.top() + Min_Heap.top()) / 2`

```python
import heapq

class MedianFinder:
    """
    数据流中位数（LeetCode #295）
    
    addNum：O(log n)
    findMedian：O(1)
    空间：O(n)
    """
    def __init__(self):
        # Python 只有最小堆，用负数模拟最大堆
        self.max_heap: list[int] = []   # 左半（最大堆，存负数）
        self.min_heap: list[int] = []   # 右半（最小堆）
    
    def addNum(self, num: int) -> None:
        # 步骤1：无条件先加入左半（最大堆）
        heapq.heappush(self.max_heap, -num)
        
        # 步骤2：维护 max_heap.top() <= min_heap.top()
        if self.min_heap and (-self.max_heap[0]) > self.min_heap[0]:
            # 左半最大值 > 右半最小值，需要把左半最大值移到右半
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        
        # 步骤3：维护 len(max_heap) >= len(min_heap)（且差值 ≤ 1）
        if len(self.max_heap) < len(self.min_heap):
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)
        elif len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
    
    def findMedian(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])   # 奇数个，取左半最大
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0   # 偶数个取均值

# 使用示例
mf = MedianFinder()
for num in [1, 2, 3, 4, 5, 6]:
    mf.addNum(num)
    print(f"加入{num}后，中位数 = {mf.findMedian()}")
# 加入1后，中位数 = 1.0
# 加入2后，中位数 = 1.5
# 加入3后，中位数 = 2.0
# 加入4后，中位数 = 2.5
# 加入5后，中位数 = 3.0
# 加入6后，中位数 = 3.5
```

```cpp
#include <queue>
#include <iostream>
using namespace std;

/**
 * 数据流中位数（LeetCode #295）
 * 双堆维护：max_heap（左半）+ min_heap（右半）
 * 
 * addNum：O(log n)，findMedian：O(1)，空间：O(n)
 * 
 * 不变量：
 *   1. max_heap.size() >= min_heap.size()（差值 ≤ 1）
 *   2. max_heap.top() <= min_heap.top()
 */
class MedianFinder {
private:
    priority_queue<int> max_heap;                            // 最大堆（左半）
    priority_queue<int, vector<int>, greater<int>> min_heap; // 最小堆（右半）

public:
    void addNum(int num) {
        // 步骤1：先加入左半（最大堆）
        max_heap.push(num);
        
        // 步骤2：维护 max_heap.top() <= min_heap.top()
        if (!min_heap.empty() && max_heap.top() > min_heap.top()) {
            min_heap.push(max_heap.top());
            max_heap.pop();
        }
        
        // 步骤3：平衡两堆大小
        if (max_heap.size() < min_heap.size()) {
            max_heap.push(min_heap.top());
            min_heap.pop();
        } else if (max_heap.size() > min_heap.size() + 1) {
            min_heap.push(max_heap.top());
            max_heap.pop();
        }
    }
    
    double findMedian() {
        if (max_heap.size() > min_heap.size()) {
            return static_cast<double>(max_heap.top());
        }
        return (max_heap.top() + min_heap.top()) / 2.0;
    }
};

int main() {
    MedianFinder mf;
    for (int num : {1, 2, 3, 4, 5, 6}) {
        mf.addNum(num);
        cout << "加入" << num << "后，中位数 = " << mf.findMedian() << endl;
    }
    return 0;
}
```

### 16.4.4 经典 LeetCode 题解思路

| 题目 | 核心思路 | 复杂度 |
|---|---|---|
| **#215 第K个最大元素** | QuickSelect：第 $k$ 大 = 第 $(n-k+1)$ 小；期望 $O(n)$ | 期望 $O(n)$ |
| **#973 最接近原点的 K 个点** | 以距离平方为关键词，QuickSelect 或堆；避免开方保精度 | 期望 $O(n)$ |
| **#347 前 K 高频元素** | 先哈希统计频率 $O(n)$，再对频率数组 QuickSelect 或堆 | $O(n)$ |
| **#295 数据流的中位数** | 双堆维护左右半；addNum $O(\log n)$，findMedian $O(1)$ | $O(\log n)$ |
| **#4 两个有序数组的中位数** | 二分法（见 Chapter 17）；不能用 QuickSelect | $O(\log(m+n))$ |

**题 #973 的细节**：距离"最接近"原点 = 欧几里得距离最小 = 平方距离最小（避免开平方浮点误差）。使用 QuickSelect 以**平方距离**为键，找第 $k$ 小，再扫描一遍收集：

```python
def kClosest(points: list[list[int]], k: int) -> list[list[int]]:
    """
    #973 最接近原点的 K 个点
    用 QuickSelect 以平方距离为键，期望 O(n)
    """
    def dist_sq(p: list[int]) -> int:
        return p[0]**2 + p[1]**2
    
    # 在 points[left..right] 中找第 i 小（按距离平方），原地修改
    import random
    
    def partition(left: int, right: int) -> int:
        pivot_idx = random.randint(left, right)
        pivot_dist = dist_sq(points[pivot_idx])
        points[pivot_idx], points[right] = points[right], points[pivot_idx]
        
        store = left
        for j in range(left, right):
            if dist_sq(points[j]) <= pivot_dist:
                points[store], points[j] = points[j], points[store]
                store += 1
        points[store], points[right] = points[right], points[store]
        return store
    
    def select(left: int, right: int, k_goal: int):
        if left >= right:
            return
        q = partition(left, right)
        if q == k_goal - 1:
            return                             # 前 k 个已就位
        elif q < k_goal - 1:
            select(q + 1, right, k_goal)
        else:
            select(left, q - 1, k_goal)
    
    select(0, len(points) - 1, k)
    return points[:k]

# 测试
print(kClosest([[1,3],[-2,2]], 1))  # [[-2,2]]（距离=√8 < √10）
print(kClosest([[3,3],[5,-1],[-2,4]], 2))  # [[3,3],[-2,4]]
```

---

## 16.5 边界情况与常见陷阱

### 16.5.1 重复元素的处理

当数组中存在**大量重复元素**时（如全相同），Lomuto 划分会退化：每次 pivot 都被放到末尾，partition 总是不均匀（$0$ vs $n-1$），退化到 $O(n^2)$。

**解决方案一**：使用**三路划分**（Dutch National Flag，如 16.4.1 中 `quickselect` 的实现），一次将 `== pivot` 的全部归入中间，直接跳过整个相等区域。

**解决方案二**：随机化 pivot + 随机打乱数组（Fisher-Yates Shuffle），使得这种极端情况的概率降低。

```python
# 危险：Lomuto + 全相同数组 → O(n²)
arr = [5] * 10000
# randomized_select(arr, 5000)  # 会很慢！

# 正确：三路划分的 quickselect（上面 16.4.1 中已实现）
print(quickselect([5] * 10000, 5000))  # 5，O(n)
```

### 16.5.2 原地实现时的排名同步

在原地区间递归时，$k$（目标排名）必须相对于当前子数组的**左端 `left`** 来计数，而不是相对于全局数组。一个常见错误是忘记在"去右半"时调整 $i$：

```python
# 错误示例：去右半时忘记减去 k_local
def wrong_select(arr, left, right, i):
    # ...
    elif i > k_local:
        return wrong_select(arr, q + 1, right, i)  # ❌ 应该是 i - k_local
        # 这样 i 仍然是全局排名，但右半子数组从 q+1 开始，下标不对应

# 正确示例
def correct_select(arr, left, right, i):
    # ...
    elif i > k_local:
        return correct_select(arr, q + 1, right, i - k_local)  # ✓ 调整为局部排名
```

### 16.5.3 BFPRT 中的等值处理

BFPRT 在找到 $m^*$ 后，通过线性扫描找到 $m^*$ **第一次**出现的位置做 PARTITION。若数组中有多个等于 $m^*$ 的元素，需要小心：

- 简化版（如上面代码）：找第一个等于 $m^*$ 的位置，用它做 pivot
- 严格版：用三路划分直接处理所有等于 $m^*$ 的元素

---

## 本章小结

| 算法 | 时间（最坏） | 时间（期望/平均） | 空间 | 核心技术 |
|---|---|---|---|---|
| 朴素排序选择 | $O(n \log n)$ | $O(n \log n)$ | 视排序 | 排序后索引 |
| RANDOMIZED-SELECT | $\Theta(n^2)$ | $O(n)$（期望） | $O(\log n)$ | 随机 pivot + 单侧递归 |
| BFPRT（中位数的中位数）| $O(n)$ | $O(n)$ | $O(\log n)$ | 确定性"好 pivot" |
| QuickSelect（工程版）| $\Theta(n^2)$ | $O(n)$（期望） | $O(1)$（迭代） | 三路划分 + 随机化 |
| 双堆（数据流中位数）| $O(\log n)$/次 | $O(\log n)$/次 | $O(n)$ | Max+Min 双堆不变量 |

**核心洞察**：
1. "选第 $k$ 小"比"完整排序"更容易，因为我们只需要保证 pivot 两侧分得对，不需要两侧都有序
2. 随机化是消除最坏情况的实用武器，BFPRT 是理论最优的"核武器"
3. Top-K 问题在不同数据规模和访问模式下有截然不同的最优解：流式用堆，随机访问用 QuickSelect

**下一章**（Chapter 17）将深入**二分搜索**：如何在有序结构中以 $O(\log n)$ 的代价精准定位目标，以及"对答案二分"这一强大设计范式。

---

**参考资料**：
- CLRS 第4版 Chapter 9（《Introduction to Algorithms》中文版第9章：中位数和顺序统计量）
- MIT 6.006 Lecture 9（Selection Sort, Heap Sort, AVL Sort）
- Blum, Floyd, Pratt, Rivest, Tarjan《Time Bounds for Selection》（1973，BFPRT 原始论文）
- LeetCode #215、#295、#347、#973（经典 Top-K 与中位数题目）
