# Chapter 17: 二分搜索与有序结构搜索（Binary Search & Ordered Search）

> **学习目标**：  
> 彻底掌握二分搜索的三种核心模板（精确查找 / 左边界 / 右边界）及其不变量，永远不再写出有 Bug 的二分；能识别"对答案二分"的适用场景并独立建立单调性判断；理解二分搜索在有序数组、旋转数组、连续函数等不同"搜索空间"中的统一本质；掌握指数搜索、三分搜索等扩展变体，并能在面试中快速选出正确策略。

---

## 17.1 二分搜索（Binary Search）

### 17.1.1 前提：有序数组与单调性

**生活比喻——猜数字游戏**：你和朋友玩一个游戏，朋友心里想了 1 到 100 之间的一个整数，每次你猜一个数，他只告诉你"大了"或"小了"或"猜对了"。最聪明的策略是什么？

不是从 1 开始一个个猜（这要平均猜 50 次），而是**每次猜中间的数**：先猜 50，如果大了，下次猜 25，如果小了猜 75……如此下去，最多猜 $\lceil \log_2 100 \rceil = 7$ 次就能找到答案。

这就是**二分搜索（Binary Search）**的直觉：每次把搜索范围缩小一半。

**正式前提**：二分搜索要求搜索空间具有**某种单调性（Monotonicity）**，具体来说：

1. **有序数组**：元素按升序（或降序）排列，"某元素是否 ≥ target"是一个关于下标的单调函数。
2. **单调判断函数**：更抽象地，存在一个判断函数 $f(idx)$，使得 $f(0), f(1), \ldots, f(n-1)$ 满足：存在某个分界点 $p$，$p$ 左边全部是 `False`，$p$ 右边全部是 `True`（或反之）。

> **通俗理解**：只要你能把所有候选位置分成"一定不是答案的区域"和"可能是答案的区域"，且这两段是连续的，你就能使用二分搜索。

**典型单调性示例**：

| 问题 | 判断函数 $f(x)$ | 单调性 |
|---|---|---|
| 有序数组查找 target | `arr[x] >= target` | $x$ 越大，$f$ 越可能为 True |
| 吃香蕉（#875）| `能在 h 小时内以速度 k 吃完` | $k$ 越大，越容易满足 |
| 给绳子分段（#1011）| `用 m 艘船能在 days 内运完` | $days$ 越大，越容易满足 |
| 划分 K 段（#410）| `最大段和 ≤ m 时，能分成 ≤ K 段` | $m$ 越大，越容易满足 |

**为什么单调性是充要条件**：如果没有单调性（比如搜索空间是"乱序"的），那么无论你选哪个中点，都无法通过该中点的信息推断出答案在左边还是右边，因而无法排除一半的搜索空间——二分就失去了意义。

---

### 17.1.2 迭代实现与死循环陷阱分析

二分搜索看似简单，但细节极其容易出错。Donald Knuth（《计算机程序设计艺术》作者）曾指出：二分搜索的思想在 1946 年就被提出，但第一个没有 Bug 的完整实现直到 1962 年才出现——整整 16 年！

**两种循环条件的本质差异**：

```
while l <= r   // 搜索空间：[l, r]（闭区间，至少有 1 个元素时继续）
while l < r    // 搜索空间：[l, r)（左闭右开，至少有 2 个元素时继续）
```

这两种写法本质上都正确，但**配套的边界收缩逻辑必须与之匹配**。混用是死循环 Bug 的第一大来源。

**死循环的根本原因**：搜索空间在某一步后没有减小。以下面的错误写法为例：

```
l = 0, r = 5（搜索空间有 6 个元素）
mid = (0 + 5) / 2 = 2
如果 arr[2] < target：错误地写 l = mid（而非 l = mid + 1）
此时 l = 2, r = 5，mid 仍为 2，l 不变 → 永远卡在这里！
```

**Rule of Thumb（经验法则）**：如果使用 `while l <= r`，两侧收缩都必须是 `l = mid + 1` 或 `r = mid - 1`（严格排除 mid）；如果使用 `while l < r`，则根据语义决定是否包含 mid。

**中点计算的溢出问题（重要！）**：

```python
# ❌ 危险写法（在大整数时可能溢出，C++ 中尤需警惕）
mid = (l + r) // 2

# ✅ 安全写法
mid = l + (r - l) // 2

# ✅ Python 中不需要担心整数溢出，但 C++ 必须使用安全写法
```

在 C++ 中，`int` 类型最大值约为 $2^{31} - 1 \approx 2.1 \times 10^9$，若 `l + r` 超过此值就会溢出。"对答案二分"时搜索空间可能很大（如 $l=10^9, r=10^9$），溢出风险极高。

---

### 17.1.3 循环不变式证明正确性

**循环不变式（Loop Invariant）**是一个在循环每次迭代前后都保持为真的命题。通过它，我们能严格证明算法的正确性。

**以精确查找为例（`while l <= r`，闭区间模板）**，定义不变式：

> **INV**：若 target 存在于数组中，则 `target` 一定在 `arr[l..r]`（即下标 $[l, r]$ 的子数组）中。

**证明三步骤**：

1. **初始化（Initialization）**：循环开始前，$l = 0, r = n - 1$，整个数组都在 $[l, r]$ 范围内。若 `target` 存在，它必然在 `arr[0..n-1]` 中。→ INV 成立 ✓

2. **保持（Maintenance）**：假设某次迭代开始时 INV 成立，令 $m = l + (r - l) / 2$，分三种情况：
   - 若 `arr[m] == target`：找到了，循环终止，INV 在终止时仍然成立（target 就在 $m$ 处）。
   - 若 `arr[m] < target`：因为数组有序，$[l, m]$ 中所有元素 $\leq arr[m] < \text{target}$，target 不可能在此区间。安全地令 $l = m + 1$，INV 仍成立 ✓
   - 若 `arr[m] > target`：同理，$[m, r]$ 中所有元素 $\geq arr[m] > \text{target}$，令 $r = m - 1$，INV 仍成立 ✓

3. **终止（Termination）**：当 $l > r$ 时循环终止，搜索空间为空，说明 target 不存在于数组中，返回 $-1$。

---

### 17.1.4 模板一：精确查找（找不到返回 -1）

**核心思想**：在有序数组中找一个特定值。如果存在，返回其下标；否则返回 $-1$。

**使用闭区间 `[l, r]` + `while l <= r`**：每次循环必定使 $r - l$ 减小（因为 `l = mid + 1` 或 `r = mid - 1`，mid 一定被排除），不会死循环。

**步骤图解**（数组 `[1, 3, 5, 7, 9, 11, 13]`，查找 `7`）：

```
初始：l=0, r=6, arr=[1,3,5,7,9,11,13]
          ↑                        ↑
          l                        r

第1次：mid=3, arr[3]=7 == target → 返回 3 ✓

（若查找 6）
第1次：mid=3, arr[3]=7 > 6 → r = mid-1 = 2
第2次：mid=1, arr[1]=3 < 6 → l = mid+1 = 2
第3次：mid=2, arr[2]=5 < 6 → l = mid+1 = 3
此时 l=3 > r=2，退出循环 → 返回 -1
```

```python
def binary_search(arr: list[int], target: int) -> int:
    """
    模板一：精确查找
    在有序数组 arr 中查找 target，返回下标（0-indexed）；找不到返回 -1。
    
    时间复杂度：O(log n)
    空间复杂度：O(1)
    
    注意：本函数假设 arr 已按升序排列。
    """
    l, r = 0, len(arr) - 1  # 搜索区间 [l, r]（闭区间）
    
    while l <= r:            # 区间非空时继续（含相等：单个元素也要检查）
        mid = l + (r - l) // 2  # 避免 (l + r) 整数溢出的安全写法
        
        if arr[mid] == target:
            return mid           # 找到，返回下标
        elif arr[mid] < target:
            l = mid + 1          # target 在右半，左边界右移（严格排除 mid）
        else:
            r = mid - 1          # target 在左半，右边界左移（严格排除 mid）
    
    return -1                    # 搜索空间耗尽，target 不存在

# 测试
arr = [1, 3, 5, 7, 9, 11, 13]
print(binary_search(arr, 7))   # 3
print(binary_search(arr, 6))   # -1
print(binary_search(arr, 1))   # 0（边界：最左元素）
print(binary_search(arr, 13))  # 6（边界：最右元素）
print(binary_search(arr, 0))   # -1（小于最小值）
print(binary_search(arr, 15))  # -1（大于最大值）
print(binary_search([], 5))    # -1（边界：空数组）
```

```cpp
#include <vector>
using namespace std;

/**
 * 模板一：精确查找
 * 在有序数组 arr 中查找 target，返回下标（0-indexed）；找不到返回 -1。
 *
 * 时间复杂度：O(log n)
 * 空间复杂度：O(1)
 *
 * @param arr    已升序排列的数组
 * @param target 目标值
 * @return       目标值的下标，或 -1（不存在时）
 */
int binary_search(const vector<int>& arr, int target) {
    int l = 0, r = static_cast<int>(arr.size()) - 1;  // 类型转为有符号整型，避免 r=-1 时与 l 比较出错
    
    while (l <= r) {               // 搜索区间 [l, r] 非空
        int mid = l + (r - l) / 2; // ✅ 安全中点计算，避免 l+r 溢出
        
        if (arr[mid] == target) {
            return mid;             // 找到目标
        } else if (arr[mid] < target) {
            l = mid + 1;            // 右移左边界（严格排除 mid）
        } else {
            r = mid - 1;            // 左移右边界（严格排除 mid）
        }
    }
    return -1;                      // 未找到
}

// 或直接使用 STL：std::binary_search（只返回是否存在）
// std::lower_bound（返回迭代器，指向第一个 >= target 的位置）
int using_stl(const vector<int>& arr, int target) {
    auto it = lower_bound(arr.begin(), arr.end(), target);
    if (it != arr.end() && *it == target) {
        return static_cast<int>(it - arr.begin());
    }
    return -1;
}

int main() {
    vector<int> arr = {1, 3, 5, 7, 9, 11, 13};
    cout << binary_search(arr, 7)  << endl;  // 3
    cout << binary_search(arr, 6)  << endl;  // -1
    cout << binary_search(arr, 1)  << endl;  // 0
    cout << binary_search(arr, 13) << endl;  // 6
    return 0;
}
```

---

### 17.1.5 模板二：左边界（第一个 ≥ target 的位置）

**经典需求**：有序数组中可能有**重复元素**，要找的是"第一个值为 target 的位置"，或者"当 target 不存在时，它应该插入的位置"（称为 **lower_bound**，即下界）。

**语义**：返回满足 `arr[i] >= target` 的**最小下标** $i$。

> 如果所有元素都 < target，则返回 `n`（表示 target 应在末尾插入）。

**为什么精确查找模板不够用**：若数组是 `[1, 3, 3, 3, 5]`，查找 3，精确查找可能返回任意一个 3 的位置（通常是中间那个），而**左边界模板**保证返回第一个 3 的位置（下标 1）。

**关键设计变化**：用 `while l < r` + 半开区间 `[l, r)`：
- 当 `arr[mid] >= target`：mid **有可能**是答案，不能排除 mid，令 `r = mid`
- 当 `arr[mid] < target`：mid 一定不是答案，令 `l = mid + 1`

循环结束时 `l == r`，此即所求下标。

```python
def lower_bound(arr: list[int], target: int) -> int:
    """
    模板二：左边界（lower_bound）
    返回满足 arr[i] >= target 的最小 i（0-indexed）。
    若所有元素均 < target，则返回 len(arr)。
    
    等价于 Python 的 bisect.bisect_left(arr, target)。
    
    时间复杂度：O(log n)
    空间复杂度：O(1)
    
    适用场景：
      - 查找 target 第一次出现的位置
      - 查找 target 的插入位置（保持有序）
      - 对答案二分时寻找"恰好满足条件"的最小值
    """
    l, r = 0, len(arr)  # 搜索区间 [l, r)，r 初始为 n（允许返回 n 表示末尾）
    
    while l < r:         # 区间至少有 2 个位置时继续（l == r 时停止）
        mid = l + (r - l) // 2
        
        if arr[mid] < target:
            l = mid + 1  # arr[mid] 太小，mid 不能是答案，左边界右移
        else:
            r = mid      # arr[mid] >= target，mid 有可能是答案，右边界收到 mid（不排除 mid）
    
    # 循环结束时 l == r，即为所求
    return l

# 测试
arr = [1, 3, 3, 3, 5, 7]
print(lower_bound(arr, 3))   # 1（第一个 3 的位置）
print(lower_bound(arr, 4))   # 4（4 应插在下标 4）
print(lower_bound(arr, 0))   # 0（比最小元素还小，插在开头）
print(lower_bound(arr, 8))   # 6（比最大元素还大，插在末尾 = len(arr)）

# 用 lower_bound 实现精确查找
def find_exact(arr, target):
    idx = lower_bound(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1

print(find_exact(arr, 3))  # 1
print(find_exact(arr, 4))  # -1
```

```cpp
#include <vector>
#include <algorithm>
using namespace std;

/**
 * 模板二：左边界（手写 lower_bound）
 * 返回满足 arr[i] >= target 的最小 i（0-indexed）。
 * 若所有元素均 < target，则返回 arr.size()。
 *
 * 等价于 std::lower_bound(arr.begin(), arr.end(), target) - arr.begin()。
 */
int lower_bound_impl(const vector<int>& arr, int target) {
    int l = 0, r = static_cast<int>(arr.size());  // 区间 [l, r)，r 初始为 n
    
    while (l < r) {
        int mid = l + (r - l) / 2;
        
        if (arr[mid] < target) {
            l = mid + 1;   // 太小，mid 一定不是答案
        } else {
            r = mid;       // arr[mid] >= target，mid 可能是答案，右边界收到 mid
        }
    }
    return l;  // l == r 时停止
}

// 使用 STL（推荐）：
// std::lower_bound(arr.begin(), arr.end(), target) - arr.begin()
// 效果完全等价，且针对随机访问迭代器做了优化

int main() {
    vector<int> arr = {1, 3, 3, 3, 5, 7};
    
    // 手写版
    cout << lower_bound_impl(arr, 3) << endl;  // 1
    cout << lower_bound_impl(arr, 4) << endl;  // 4
    cout << lower_bound_impl(arr, 8) << endl;  // 6
    
    // STL 版（结果相同）
    int idx = static_cast<int>(lower_bound(arr.begin(), arr.end(), 3) - arr.begin());
    cout << idx << endl;  // 1
    
    return 0;
}
```

---

### 17.1.6 模板三：右边界（最后一个 ≤ target 的位置）

**经典需求**：找数组中"最后一个值 ≤ target 的位置"（称为 **upper_bound - 1**，即上界减一）。

**语义**：返回满足 `arr[i] <= target` 的**最大下标** $i$。等价于：`upper_bound(arr, target) - 1`，其中 `upper_bound` 返回"第一个 > target"的位置。

**典型使用场景**：统计数组中 target 出现的次数 = `upper_bound(target) - lower_bound(target)`。

```python
def upper_bound(arr: list[int], target: int) -> int:
    """
    upper_bound：返回满足 arr[i] > target 的最小 i（0-indexed）。
    若所有元素均 <= target，则返回 len(arr)。
    
    等价于 bisect.bisect_right(arr, target)。
    
    右边界（最后一个 <= target 的位置）= upper_bound(arr, target) - 1。
    """
    l, r = 0, len(arr)   # 区间 [l, r)
    
    while l < r:
        mid = l + (r - l) // 2
        
        if arr[mid] <= target:  # 注意：这里是 <=，与 lower_bound 的 < 不同
            l = mid + 1
        else:
            r = mid
    
    return l  # upper_bound

def right_bound(arr: list[int], target: int) -> int:
    """
    右边界：返回满足 arr[i] == target 的最大下标。
    若 target 不存在，返回 -1。
    """
    idx = upper_bound(arr, target) - 1
    if idx >= 0 and arr[idx] == target:
        return idx
    return -1

# 综合应用：统计 target 出现次数
def count_occurrences(arr: list[int], target: int) -> int:
    return upper_bound(arr, target) - lower_bound(arr, target)

# 测试
arr = [1, 3, 3, 3, 5, 7]
print(right_bound(arr, 3))              # 3（最后一个 3 的位置）
print(right_bound(arr, 4))              # -1（不存在）
print(count_occurrences(arr, 3))        # 3（三个 3）
print(count_occurrences(arr, 4))        # 0（0 个 4）

# 综合使用：找 target 的范围 [first, last]
def search_range(arr: list[int], target: int) -> tuple[int, int]:
    """LeetCode #34：在排序数组中查找元素的第一个和最后一个位置"""
    first = lower_bound(arr, target)
    if first == len(arr) or arr[first] != target:
        return (-1, -1)
    last = upper_bound(arr, target) - 1
    return (first, last)

print(search_range(arr, 3))  # (1, 3)
print(search_range(arr, 4))  # (-1, -1)
```

```cpp
#include <vector>
#include <algorithm>
using namespace std;

/**
 * upper_bound：返回满足 arr[i] > target 的最小 i（0-indexed）。
 * 等价于 std::upper_bound(arr.begin(), arr.end(), target) - arr.begin()。
 */
int upper_bound_impl(const vector<int>& arr, int target) {
    int l = 0, r = static_cast<int>(arr.size());
    
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] <= target) {  // 注意：<= 而非 < ！
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

// 统计 target 出现次数
int count_occurrences(const vector<int>& arr, int target) {
    // upper_bound - lower_bound = 等于 target 的元素数量
    int ub = static_cast<int>(upper_bound(arr.begin(), arr.end(), target) - arr.begin());
    int lb = static_cast<int>(lower_bound(arr.begin(), arr.end(), target) - arr.begin());
    return ub - lb;
}

// 查找元素的范围 [first, last]（LeetCode #34）
pair<int,int> search_range(const vector<int>& arr, int target) {
    int first = static_cast<int>(lower_bound(arr.begin(), arr.end(), target) - arr.begin());
    if (first == (int)arr.size() || arr[first] != target) {
        return {-1, -1};
    }
    int last = static_cast<int>(upper_bound(arr.begin(), arr.end(), target) - arr.begin()) - 1;
    return {first, last};
}

int main() {
    vector<int> arr = {1, 3, 3, 3, 5, 7};
    cout << count_occurrences(arr, 3) << endl;   // 3
    
    auto [f, l] = search_range(arr, 3);
    cout << f << " " << l << endl;               // 1 3
    return 0;
}
```

---

### 17.1.7 三种模板的统一推导与差异对比

**从统一视角理解三种模板**：所有二分搜索本质上都是在寻找一个**分界点**——使得分界点左边满足条件 A，右边满足条件 B。三种模板的差异只在于：判断函数和分界点的定义不同。

```
模板一（精确查找）：
  左侧条件：arr[i] < target
  右侧条件：arr[i] > target
  分界点  ：arr[i] == target（若存在）

模板二（左边界 / lower_bound）：
  左侧条件：arr[i] < target
  右侧条件：arr[i] >= target  ← 分界点是右侧段的第一个元素
  终止时 l == r == 答案

模板三（上界 / upper_bound）：
  左侧条件：arr[i] <= target
  右侧条件：arr[i] > target   ← 分界点是右侧段的第一个元素
  终止时 l == r == upper_bound
```

| 特性 | 模板一（精确查找）| 模板二（左边界）| 模板三（上界）|
|---|---|---|---|
| 循环条件 | `while l <= r` | `while l < r` | `while l < r` |
| 初始 `r` | `n - 1` | `n` | `n` |
| `arr[mid] < target` | `l = mid + 1` | `l = mid + 1` | `l = mid + 1` |
| `arr[mid] == target` | `return mid` | `r = mid` | `l = mid + 1` |
| `arr[mid] > target` | `r = mid - 1` | `r = mid` | `r = mid` |
| 返回值 | `mid`（找到）或 `-1` | `l`（第一个 ≥ target）| `l`（第一个 > target）|
| 终止条件 | `l > r` | `l == r` | `l == r` |
| 重复元素处理 | 返回任意一个 | 返回最左边那个 | 返回最右边的下一个 |

**记忆口诀**：
- 精确查找：`<= r`，遇到相等直接返回
- 左边界：`< r`，遇到 `>=` 时收缩右侧（`r = mid`）
- 上界：`< r`，遇到 `>` 时收缩右侧（`r = mid`），遇到 `<=` 右移左侧

<div data-component="BinarySearchBoundaryTemplate"></div>

---

### 17.1.8 O(log n) 的严格证明

**定理**：对于含 $n$ 个元素的有序数组，二分搜索至多执行 $\lceil \log_2(n+1) \rceil$ 次比较。

**证明**：

设循环执行前搜索空间大小为 $W = r - l + 1$（闭区间）。每次循环取 $m = l + \lfloor W/2 \rfloor$：
- 若向右收缩：新空间大小 $W' = r - (m+1) + 1 = r - m = W - \lfloor W/2 \rfloor - 1 \leq \lfloor W/2 \rfloor$
- 若向左收缩：新空间大小 $W' = (m-1) - l + 1 = m - l = \lfloor W/2 \rfloor$

两种情况下，$W' \leq \lfloor W/2 \rfloor$，即**每次循环搜索空间至少减半**。

从初始 $W_0 = n$ 出发，经过 $k$ 次减半：

$$W_k \leq \left\lfloor \frac{n}{2^k} \right\rfloor$$

当 $W_k = 0$（循环终止）时：

$$\frac{n}{2^k} < 1 \implies k > \log_2 n \implies k \geq \lceil \log_2(n+1) \rceil$$

因此**比较次数 $= O(\log n)$**。

**具体数字**：

| 数组大小 $n$ | 最多比较次数 |
|---|---|
| 10 | 4 次 |
| 100 | 7 次 |
| 1,000 | 10 次 |
| $10^6$（一百万）| 20 次 |
| $10^9$（十亿）| 30 次 |
| $10^{18}$（**相当于遍历整个宇宙中的原子数量**）| 60 次 |

这个增长速度令人惊叹——哪怕数组大到足以给地球上每个原子分配一个元素，二分搜索只需约 60 次比较！

---

## 17.2 二分搜索的扩展

### 17.2.1 旋转有序数组中搜索（LeetCode #33）

**问题描述**：有序数组 `[0, 1, 2, 4, 5, 6, 7]` 在某个未知下标处被"旋转"，变成 `[4, 5, 6, 7, 0, 1, 2]`。在这个旋转后的数组中查找 target。

**直觉**：虽然整体不再有序，但把数组从中点 `mid` 分成两半后，**至少有一半是严格有序的**！

**判断哪半段有序**：
- 若 `arr[l] <= arr[mid]`：左半段 `arr[l..mid]` 有序（旋转点在右半段或不存在）
- 否则：右半段 `arr[mid..r]` 有序

确定哪半段有序后，检查 target 是否在有序段的范围内：若是，在有序段内继续搜索；否则，在另一半搜索。

```python
def search_rotated(arr: list[int], target: int) -> int:
    """
    LeetCode #33：搜索旋转有序数组（无重复元素）
    时间复杂度：O(log n)
    空间复杂度：O(1)
    
    核心思路：每次确定哪半段有序，再看 target 是否在有序段范围内。
    """
    l, r = 0, len(arr) - 1
    
    while l <= r:
        mid = l + (r - l) // 2
        
        if arr[mid] == target:
            return mid
        
        # 判断左半段 [l, mid] 是否有序
        if arr[l] <= arr[mid]:
            # 左半段有序：[arr[l], arr[mid]] 是单调递增序列
            if arr[l] <= target < arr[mid]:
                # target 在左半段范围内，去左半搜索
                r = mid - 1
            else:
                # target 不在左半段，去右半搜索
                l = mid + 1
        else:
            # 右半段有序：[arr[mid], arr[r]] 是单调递增序列
            if arr[mid] < target <= arr[r]:
                # target 在右半段范围内，去右半搜索
                l = mid + 1
            else:
                # target 不在右半段，去左半搜索
                r = mid - 1
    
    return -1

# 测试
arr = [4, 5, 6, 7, 0, 1, 2]
print(search_rotated(arr, 0))  # 4（0 在旋转点右侧）
print(search_rotated(arr, 3))  # -1（不存在）
print(search_rotated(arr, 4))  # 0（旋转点就是 target）
print(search_rotated(arr, 7))  # 3
```

```cpp
#include <vector>
using namespace std;

/**
 * LeetCode #33：搜索旋转有序数组（无重复元素）
 * 
 * 时间复杂度：O(log n)
 * 空间复杂度：O(1)
 */
int search_rotated(const vector<int>& arr, int target) {
    int l = 0, r = static_cast<int>(arr.size()) - 1;
    
    while (l <= r) {
        int mid = l + (r - l) / 2;
        
        if (arr[mid] == target) return mid;
        
        // 左半段 [l, mid] 有序？
        if (arr[l] <= arr[mid]) {
            // 左半有序：判断 target 是否在左半范围内
            if (arr[l] <= target && target < arr[mid]) {
                r = mid - 1;   // target 在左半
            } else {
                l = mid + 1;   // target 在右半
            }
        } else {
            // 右半段 [mid, r] 有序：判断 target 是否在右半范围内
            if (arr[mid] < target && target <= arr[r]) {
                l = mid + 1;   // target 在右半
            } else {
                r = mid - 1;   // target 在左半
            }
        }
    }
    return -1;
}

int main() {
    vector<int> arr = {4, 5, 6, 7, 0, 1, 2};
    cout << search_rotated(arr, 0) << endl;  // 4
    cout << search_rotated(arr, 3) << endl;  // -1
    return 0;
}
```

<div data-component="RotatedArraySearch"></div>

**处理含重复元素的变体（LeetCode #81）**：当数组中有重复元素时，`arr[l] == arr[mid]` 时无法判断哪半段有序（如 `[1, 3, 1, 1, 1]`）。此时退化为 `l++`（线性扫描该元素），最坏时间复杂度退化为 $O(n)$：

```python
def search_with_duplicates(arr: list[int], target: int) -> bool:
    """LeetCode #81：含重复元素的旋转有序数组搜索，返回是否存在"""
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if arr[mid] == target:
            return True
        # 核心差异：无法判断哪半段有序时，只能缩小一个位置
        if arr[l] == arr[mid] == arr[r]:
            l += 1; r -= 1   # 去掉两端的重复元素（最坏导致 O(n)）
        elif arr[l] <= arr[mid]:
            if arr[l] <= target < arr[mid]: r = mid - 1
            else: l = mid + 1
        else:
            if arr[mid] < target <= arr[r]: l = mid + 1
            else: r = mid - 1
    return False
```

---

### 17.2.2 查找峰值元素（LeetCode #162）

**问题描述**：给定数组，峰值元素的定义是"严格大于其左右相邻元素"的元素。假设数组边界外的元素为 $-\infty$，数组中至少存在一个峰值，找出任意一个峰值元素的下标。

**关键洞察**：如果向中点两侧"爬坡"，往更高的方向走，一定能找到峰值（就像沿着山坡往高处走，一定会到达某个山顶）。

**二分策略**：
- 若 `arr[mid] < arr[mid + 1]`：右侧比当前高，右边一定存在峰值（因为右侧至少有一个局部最高点），令 `l = mid + 1`
- 否则（`arr[mid] > arr[mid + 1]`）：当前比右侧高，当前或当前左侧存在峰值，令 `r = mid`

```python
def find_peak(arr: list[int]) -> int:
    """
    LeetCode #162：寻找峰值元素
    
    时间复杂度：O(log n)
    空间复杂度：O(1)
    
    不变量：arr[l-1] < arr[l]（左边界条件满足）
           arr[r+1] < arr[r]（右边界条件满足）
    → 峰值必存在于 [l, r] 中
    
    注意：题目保证 arr[-1] = arr[n] = -∞（边界外视为负无穷）
         以及没有相邻相等的元素（num[i] != num[i+1]）
    """
    l, r = 0, len(arr) - 1
    
    while l < r:
        mid = l + (r - l) // 2
        
        if arr[mid] < arr[mid + 1]:
            # mid 右边更高，峰值在右侧（含 mid+1）
            l = mid + 1
        else:
            # mid 右边更低（mid 可能就是峰值），向左收缩右边界
            r = mid
    
    # l == r，此即峰值下标
    return l

# 测试
print(find_peak([1, 2, 3, 1]))       # 2（峰值是 3）
print(find_peak([1, 2, 1, 3, 5]))    # 1 或 4（都是峰值，返回任意一个）
print(find_peak([1]))                # 0（单个元素就是峰值）
print(find_peak([3, 2, 1]))          # 0（单调递减，最左是峰值）
```

```cpp
#include <vector>
using namespace std;

/**
 * LeetCode #162：寻找峰值元素
 * 时间复杂度：O(log n)
 */
int find_peak(const vector<int>& arr) {
    int l = 0, r = static_cast<int>(arr.size()) - 1;
    
    while (l < r) {
        int mid = l + (r - l) / 2;
        
        if (arr[mid] < arr[mid + 1]) {
            l = mid + 1;  // 右边更高，峰值在右侧
        } else {
            r = mid;      // 当前比右侧高，峰值在 [l, mid]
        }
    }
    return l;  // l == r 即为峰值下标
}
```

**思考题**：为什么这个算法一定能找到峰值？提示：每次缩小搜索范围时，可以证明"当前范围内必存在峰值"这个不变量。

---

### 17.2.3 对答案二分（Answer Binary Search）

**这是二分搜索中最强大也最容易被忽视的思路。**

**核心思想**：当直接计算最优答案很困难时，将问题转化为：

> 给定一个候选答案 $x$，**判断 $x$ 是否可行**。若可行性关于 $x$ 具有单调性（越大越可行/越小越可行），则对 $x$ 进行二分搜索。

**单调性的直觉**：
- "以最大速度 $k$ 吃香蕉，能否在 $h$ 小时内吃完？"——$k$ 越大，越容易完成 → 单调递增
- "将货物分入载重量 $m$ 的船，能否在 $d$ 天内运完？"——$m$ 越大，越容易完成 → 单调递增
- "将数组分为 $k$ 段，最大段和是否 $\leq m$？"——$m$ 越大，越容易满足 → 单调递增

**通用解题框架**：

```
1. 确定答案的搜索空间 [lo, hi]
   - lo：答案的最小可能值（通常是"最宽松"的约束）
   - hi：答案的最大可能值（通常是"最严苛"的约束）

2. 定义 check(x) 函数，判断答案 x 是否可行
   - 通常是一个线性扫描的贪心验证，O(n) 时间

3. 用 lower_bound 找最小可行 x（或 upper_bound 找最大可行 x）

总时间复杂度：O(n log(hi - lo))
```

**经典例题一：吃香蕉（LeetCode #875）**

Koko 有 $n$ 堆香蕉，每堆 $piles[i]$ 根。她每小时可以吃 $k$ 根，每小时只能选一堆吃，吃不完可以留到下次。要在 $h$ 小时内吃完，求最小的 $k$。

```python
def min_eating_speed(piles: list[int], h: int) -> int:
    """
    LeetCode #875：吃香蕉
    
    时间复杂度：O(n log(max_pile))
    空间复杂度：O(1)
    
    搜索空间：k ∈ [1, max(piles)]
      - k=1：最慢，至少需要 sum(piles) 小时
      - k=max(piles)：最快，每小时能清空最大的一堆，共 n 小时
    
    check(k)：以速度 k 吃，是否能在 h 小时内吃完？
      → 每堆需要 ceil(pile / k) 小时，求总和是否 <= h
      → 贪心验证，O(n) 时间
    
    单调性：k 越大，需要的总时间越短，越容易满足 <= h
    → 找最小的 k 使得 check(k) 为 True
    → 这就是左边界！
    """
    import math
    
    def can_finish(k: int) -> bool:
        """check 函数：速度为 k 时，能否在 h 小时内吃完所有香蕉"""
        total_hours = sum(math.ceil(pile / k) for pile in piles)
        return total_hours <= h
    
    # 搜索空间：[1, max(piles)]
    l, r = 1, max(piles)
    
    # 找最小的 k 使得 can_finish(k) 为 True（左边界模板）
    while l < r:
        mid = l + (r - l) // 2
        if can_finish(mid):
            r = mid       # mid 可行，但可能有更小的可行值，收缩右边界
        else:
            l = mid + 1   # mid 不可行，最小可行值在右边
    
    return l  # l == r 即为最小可行速度

# 测试
print(min_eating_speed([3, 6, 7, 11], h=8))   # 4
print(min_eating_speed([30, 11, 23, 4, 20], h=5))  # 30
print(min_eating_speed([30, 11, 23, 4, 20], h=6))  # 23
```

```cpp
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

class Solution {
public:
    /**
     * LeetCode #875：吃香蕉
     * 时间复杂度：O(n log(max_pile))
     */
    int minEatingSpeed(vector<int>& piles, int h) {
        // check 函数：速度为 k 时，总共需要多少小时
        auto can_finish = [&](int k) -> bool {
            long long hours = 0;
            for (int pile : piles) {
                hours += (pile + k - 1) / k;  // ceil(pile / k)，避免 float 精度问题
                if (hours > h) return false;    // 提前退出（剪枝）
            }
            return hours <= h;
        };
        
        int l = 1, r = *max_element(piles.begin(), piles.end());
        
        // 左边界模板：找最小的可行 k
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (can_finish(mid)) {
                r = mid;      // mid 可行，找更小的
            } else {
                l = mid + 1;  // mid 不行，往右找
            }
        }
        return l;
    }
};
```

**经典例题二：分割数组的最大值（LeetCode #410）**

将数组分成 $k$ 段（非空连续子数组），使每段之和的**最大值**最小，求该最小值。

```python
def split_array(nums: list[int], k: int) -> int:
    """
    LeetCode #410：分割数组的最大值
    
    搜索空间：m ∈ [max(nums), sum(nums)]
      - m = max(nums)：每个元素一段（但要保证 k <= n）
      - m = sum(nums)：全部在一段
    
    check(m)：最大段和限制为 m 时，能否分成至多 k 段？
      → 贪心：从左到右，当前段加上下一个数不超过 m 就继续加，超过了就开新段
      → O(n) 贪心验证

    单调性：m 越大（限制越宽松），所需的段数越少，越容易 <= k
    → 找最小的 m 使得 check(m) 为 True（左边界）
    """
    def can_split(m: int) -> bool:
        """限制每段最大和为 m，最少需要几段？"""
        segments = 1
        current_sum = 0
        for num in nums:
            if current_sum + num > m:
                segments += 1   # 开新段
                current_sum = num
                if segments > k:
                    return False  # 段数超过限制，m 太小了
            else:
                current_sum += num
        return True
    
    l, r = max(nums), sum(nums)  # 搜索空间
    
    while l < r:
        mid = l + (r - l) // 2
        if can_split(mid):
            r = mid       # mid 可行，找更小的
        else:
            l = mid + 1   # mid 太小，不够分
    
    return l

# 测试
print(split_array([7, 2, 5, 10, 8], k=2))   # 18（[7,2,5] 和 [10,8]）
print(split_array([1, 2, 3, 4, 5], k=2))    # 9（[1,2,3] 和 [4,5]）
print(split_array([1, 4, 4], k=3))          # 4（每个元素单独一段）
```

```cpp
#include <vector>
#include <numeric>
using namespace std;

class Solution {
public:
    /**
     * LeetCode #410：分割数组的最大值
     * 时间复杂度：O(n log(sum - max))
     */
    int splitArray(vector<int>& nums, int k) {
        auto can_split = [&](long long m) -> bool {
            int segments = 1;
            long long cur = 0;
            for (int num : nums) {
                if (cur + num > m) {
                    ++segments;
                    cur = num;
                    if (segments > k) return false;
                } else {
                    cur += num;
                }
            }
            return true;
        };
        
        long long l = *max_element(nums.begin(), nums.end());
        long long r = accumulate(nums.begin(), nums.end(), 0LL);
        
        while (l < r) {
            long long mid = l + (r - l) / 2;
            if (can_split(mid)) r = mid;
            else l = mid + 1;
        }
        return static_cast<int>(l);
    }
};
```

<div data-component="AnswerBinarySearchDemo"></div>

**对答案二分的识别技巧**：题目中出现"最小化最大值"或"最大化最小值"，通常是对答案二分的信号。此外，若问题是"在满足某约束下，求某个量的最优值"，且该量与约束之间存在单调关系，也应考虑对答案二分。

---

### 17.2.4 浮点数二分（Precision Control）

当搜索空间是**连续实数域**时（如求平方根、求函数零点），需要对浮点数进行二分，重点在于**精度控制**。

**两种终止策略**：

1. **固定迭代次数**：执行固定次数（如 100 次）的循环，每次将搜索空间缩小一半。100 次后精度为初始范围的 $2^{-100} \approx 10^{-30}$，远超实际需求。这是最稳健的写法。

2. **精度判断**：当搜索空间小于阈值 $\varepsilon$（如 $10^{-9}$）时退出。需谨慎设置 $\varepsilon$ 以避免无限循环。

```python
def sqrt_float(x: float) -> float:
    """
    求 x 的平方根（浮点二分，精度 1e-9）
    等价于找 f(m) = m*m - x 的零点（x >= 0 时 f 单调递增）
    
    时间复杂度：O(log((hi - lo) / eps))，这里约 100 次循环
    """
    if x < 0:
        raise ValueError("不能对负数开方")
    if x == 0:
        return 0.0
    
    lo, hi = 0.0, max(1.0, x)  # 注意：x < 1 时 sqrt(x) > x，上界取 max(1, x)
    
    # 方法一：固定迭代次数（推荐）
    for _ in range(100):  # 100 次后精度约为 10^{-30}
        mid = (lo + hi) / 2
        if mid * mid < x:
            lo = mid
        else:
            hi = mid
    
    return (lo + hi) / 2

# 方法二：精度阈值
def sqrt_eps(x: float, eps: float = 1e-9) -> float:
    lo, hi = 0.0, max(1.0, x)
    while hi - lo > eps:
        mid = (lo + hi) / 2
        if mid * mid < x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

import math
print(f"{sqrt_float(2):.10f}")   # 1.4142135624
print(f"{math.sqrt(2):.10f}")    # 1.4142135624（对比标准库）
print(f"{sqrt_float(9):.10f}")   # 3.0000000000
print(f"{sqrt_float(0.25):.10f}") # 0.5000000000
```

```cpp
#include <cmath>
using namespace std;

/**
 * 浮点数二分：求平方根
 * 
 * 注意：实际使用中优先用 sqrt()，此处仅作教学演示。
 * 方法一（固定迭代次数）更安全，推荐使用。
 */
double sqrt_binary(double x) {
    if (x < 0) throw invalid_argument("Negative input");
    if (x == 0) return 0.0;
    
    double lo = 0.0, hi = max(1.0, x);
    
    // 方法一：固定 100 次迭代（精度约 10^-30，超过 double 精度）
    for (int i = 0; i < 100; ++i) {
        double mid = lo + (hi - lo) / 2.0;  // 避免 (lo+hi) 精度丢失
        if (mid * mid < x) lo = mid;
        else hi = mid;
    }
    return (lo + hi) / 2.0;
}

// 更实用：求整数平方根（LeetCode #69）
int int_sqrt(int x) {
    if (x < 2) return x;
    long long l = 1, r = x / 2;  // long long 防止 r*r 溢出
    while (l < r) {
        long long mid = l + (r - l + 1) / 2;  // 上取整 mid，防止死循环
        if (mid * mid <= x) l = mid;
        else r = mid - 1;
    }
    return static_cast<int>(l);
}

int main() {
    printf("%.10f\n", sqrt_binary(2.0));  // 1.4142135624
    printf("%d\n", int_sqrt(8));          // 2（floor(sqrt(8)) = 2）
    printf("%d\n", int_sqrt(9));          // 3
    return 0;
}
```

**整数平方根的边界处理细节**：求整数平方根时，使用右边界模板（找最大的 $m$ 使得 $m^2 \leq x$）。中点计算需上取整（`mid = l + (r - l + 1) / 2`），否则 `l = mid` 时 mid 不变导致死循环（当 `l = r - 1` 时）。

---

## 17.3 指数搜索（Exponential Search）

### 17.3.1 适用场景：无限数组与超大规模

**场景问题**：假设你有一个**无限长**的有序数组（或者数组极长，$n$ 未知），你需要查找 target。标准的二分搜索需要知道搜索范围 $[0, n-1]$，但现在你不知道 $n$。

**另一个场景**：搜索空间理论上无穷大（如实数域上的某个方程的根），但答案实际上很小（比如答案是 42，但搜索空间是 $[0, 10^{18}]$）。此时直接二分需要约 60 次迭代，而指数搜索只需约 $2 \log_2 42 \approx 10$ 次，更快。

**核心思路**：
1. **倍增阶段**：从下标 1 开始，以 $1, 2, 4, 8, 16, \ldots$ 的指数方式探测，找到满足 `arr[bound] >= target` 的最小 $2^k$。这个阶段最多需要 $\lceil \log_2 i \rceil$ 步，其中 $i$ 是 target 在数组中的实际下标。
2. **二分阶段**：在区间 $[2^{k-1}, 2^k]$ 内对 target 二分查找，最多再需要 $\log_2 2^k \approx \log_2 i$ 步。

**总时间复杂度**：$O(\log i)$，其中 $i$ 是目标元素的位置——和目标元素的实际位置成比例，对于"小下标"的目标非常高效。

### 17.3.2 实现与分析

```python
def exponential_search(arr: list[int], target: int) -> int:
    """
    指数搜索（Exponential Search / Doubling Search）
    
    时间复杂度：O(log i)，其中 i 是 target 在数组中的下标
    空间复杂度：O(1)（迭代实现）
    
    适用：
      - 无界 / 无限有序数组（不知道 n）
      - target 可能在数组前段（小下标处）的场景
      - 数据流中的有序序列搜索
    
    注意：arr 必须有序（升序），且本实现假设数组有下界（下标从 0 开始）。
    """
    n = len(arr)
    if n == 0:
        return -1
    
    # 边界特判：第 0 个元素
    if arr[0] == target:
        return 0
    
    # 阶段一：倍增，找到 target 可能在的区间上界
    bound = 1
    while bound < n and arr[bound] < target:
        bound *= 2   # 指数增长：1, 2, 4, 8, 16, ...
    
    # 阶段二：在 [bound//2, min(bound, n-1)] 内标准二分查找
    l = bound // 2
    r = min(bound, n - 1)
    
    while l <= r:
        mid = l + (r - l) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    
    return -1

# 测试
arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
print(exponential_search(arr, 3))   # 1（小下标，只需 2 次倍增）
print(exponential_search(arr, 21))  # 10（大下标，需要更多倍增）
print(exponential_search(arr, 8))   # -1（不存在）

# 效率对比（arr 长度为 100，查找第 3 个元素）
# 标准二分：约 log2(100) ≈ 7 次
# 指数搜索：约 2*log2(3) ≈ 3 次（仅当 target 在前部时更快）
```

```cpp
#include <vector>
#include <algorithm>
using namespace std;

/**
 * 指数搜索
 * 时间复杂度：O(log i)，i 为 target 的下标
 */
int exponential_search(const vector<int>& arr, int target) {
    int n = static_cast<int>(arr.size());
    if (n == 0) return -1;
    if (arr[0] == target) return 0;
    
    // 阶段一：倍增找上界
    int bound = 1;
    while (bound < n && arr[bound] < target) {
        bound *= 2;
    }
    
    // 阶段二：在 [bound/2, min(bound, n-1)] 内二分
    int l = bound / 2;
    int r = min(bound, n - 1);
    
    // 使用 STL lower_bound 完成二分部分
    auto it = lower_bound(arr.begin() + l, arr.begin() + r + 1, target);
    if (it != arr.end() && *it == target) {
        return static_cast<int>(it - arr.begin());
    }
    return -1;
}
```

**倍增的数学直觉**：假设目标下标为 $i$，倍增过程会在第 $\lceil \log_2(i+1) \rceil$ 步找到 $\text{bound} \geq i$。此时 $\text{bound}/2 < i \leq \text{bound}$，二分搜索的范围是 $[\text{bound}/2, \text{bound}]$，大小约为 $\text{bound}/2 \leq i$，因此二分也只需 $O(\log i)$ 步。两个阶段合计 $O(\log i)$。

**指数搜索 vs 标准二分**：
- 若 $i \approx n$（target 在末尾），两者性能相近（都是 $O(\log n)$）
- 若 $i \ll n$（target 在前部），指数搜索远快于标准二分

---

## 17.4 三分搜索（Ternary Search）

### 17.4.1 单峰 / 单谷函数求最值

**场景**：给定一个**单峰函数**（先单调递增后单调递减，或连续函数只有一个极大值），求其最大值所在的位置。注意：这类函数不再具有全局单调性，普通二分搜索无法直接应用。

**生活比喻**：你在爬"一座没有分叉路的山丘"，只能向前或向后走。你想找山顶。三分搜索的做法是：在当前视野 $[l, r]$ 中取两点 $m_1 = l + (r-l)/3$ 和 $m_2 = r - (r-l)/3$：
- 若 $f(m_1) < f(m_2)$：山顶在 $m_1$ 右边，令 $l = m_1 + 1$
- 若 $f(m_1) > f(m_2)$：山顶在 $m_2$ 左边，令 $r = m_2 - 1$

每次把搜索空间缩小到原来的 $2/3$。

**适用函数类型**：
- **单峰（Unimodal）**：存在唯一极大值，两侧单调递减
- **单谷（Unimodal depression）**：存在唯一极小值，两侧单调递增（形状翻转即可）
- **凸函数 / 凹函数**：凸函数（$f''(x) \geq 0$）有唯一极小值，凹函数有唯一极大值

### 17.4.2 实现与和二分的关系

```python
def ternary_search_max_int(arr: list[int]) -> int:
    """
    整数三分搜索：找单峰数组的峰值下标
    时间复杂度：O(log n)（每次缩小到 2/3）
    
    注意：对于整数三分，要特别注意 l, r 的终止条件（l+1 >= r 时直接比较）
    """
    l, r = 0, len(arr) - 1
    
    while r - l > 2:  # 搜索空间超过 3 个元素时继续
        m1 = l + (r - l) // 3
        m2 = r - (r - l) // 3
        
        if arr[m1] < arr[m2]:
            l = m1 + 1   # 峰值在 m1 右侧
        elif arr[m1] > arr[m2]:
            r = m2 - 1   # 峰值在 m2 左侧
        else:
            # arr[m1] == arr[m2]：峰值必在 [m1, m2] 内（同时收缩两侧）
            l = m1 + 1
            r = m2 - 1
    
    # 针对剩余 1~3 个元素，直接找最大值
    peak_idx = l
    for i in range(l + 1, r + 1):
        if arr[i] > arr[peak_idx]:
            peak_idx = i
    return peak_idx


def ternary_search_float(f, lo: float, hi: float, iterations: int = 200) -> float:
    """
    浮点数三分搜索：在 [lo, hi] 上找单峰函数 f 的极大值点
    
    时间复杂度：O(iterations)，每次区间缩小到 2/3
    精度：(hi - lo) * (2/3)^iterations
    
    iterations=200 时精度约为初始区间的 (2/3)^200 ≈ 10^{-35}
    """
    for _ in range(iterations):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        
        if f(m1) < f(m2):
            lo = m1   # 峰值在 m1 右侧
        else:
            hi = m2   # 峰值在 m2 左侧
    
    return (lo + hi) / 2  # 近似极大值点


# 测试：f(x) = -(x - 3)^2 + 9 是顶点在 x=3 处的开口向下抛物线
f = lambda x: -(x - 3) ** 2 + 9
peak_x = ternary_search_float(f, 0, 10)
print(f"极大值点 x ≈ {peak_x:.6f}，f(x) ≈ {f(peak_x):.6f}")
# 极大值点 x ≈ 3.000000，f(x) ≈ 9.000000

# 整数单峰数组
arr = [1, 2, 4, 7, 9, 6, 3, 1]
print(ternary_search_max_int(arr))  # 4（arr[4] = 9 是峰值）
```

```cpp
#include <functional>
using namespace std;

/**
 * 浮点数三分搜索：在 [lo, hi] 上找单峰函数 f 的极大值点
 * 
 * @param f          目标函数（单峰，求极大值）
 * @param lo         搜索下界
 * @param hi         搜索上界
 * @param iterations 迭代次数（默认 200，精度远超 double 表示范围）
 * @return           极大值点的近似位置
 */
double ternary_search(function<double(double)> f, double lo, double hi, int iterations = 200) {
    for (int i = 0; i < iterations; ++i) {
        double m1 = lo + (hi - lo) / 3.0;
        double m2 = hi - (hi - lo) / 3.0;
        
        if (f(m1) < f(m2)) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    return (lo + hi) / 2.0;
}

// 整数三分搜索：找单峰数组的峰值下标
int ternary_search_int(const vector<int>& arr) {
    int l = 0, r = static_cast<int>(arr.size()) - 1;
    
    while (r - l > 2) {
        int m1 = l + (r - l) / 3;
        int m2 = r - (r - l) / 3;
        
        if (arr[m1] < arr[m2]) l = m1 + 1;
        else if (arr[m1] > arr[m2]) r = m2 - 1;
        else { l = m1 + 1; r = m2 - 1; }
    }
    
    int peak = l;
    for (int i = l + 1; i <= r; ++i) {
        if (arr[i] > arr[peak]) peak = i;
    }
    return peak;
}

int main() {
    // 测试：f(x) = -(x-3)^2 + 9，极大值在 x=3
    auto f = [](double x) { return -(x - 3) * (x - 3) + 9; };
    double peak = ternary_search(f, 0.0, 10.0);
    printf("极大值点 x ≈ %.6f\n", peak);  // 3.000000
    
    vector<int> arr = {1, 2, 4, 7, 9, 6, 3, 1};
    printf("峰值下标：%d\n", ternary_search_int(arr));  // 4
    return 0;
}
```

**三分搜索 vs 二分求导**：在连续可微函数上，三分搜索和"对导数 $f'(x)$ 使用二分搜索找零点"是等价的思路。若 $f$ 是凹函数，则 $f' > 0$ 在极大值左侧，$f' < 0$ 在极大值右侧，单调性满足，可直接对 $f'$ 二分。实践中若能解析求导，则优先使用对 $f'$ 的二分（实现更简单，无需特判相等情况）。

**三分搜索的收敛速度分析**：

每次迭代后搜索空间缩至 $2/3$，经过 $k$ 次迭代后，精度为：

$$\text{精度} = (hi - lo) \times \left(\frac{2}{3}\right)^k$$

若要达到精度 $\varepsilon$，需要 $k \geq \frac{\ln((hi-lo)/\varepsilon)}{\ln(3/2)} \approx 2.41 \times \log_2 \frac{hi - lo}{\varepsilon}$ 次迭代。相比之下，二分搜索每次缩至 $1/2$，达到同等精度需要 $\log_2 \frac{hi - lo}{\varepsilon}$ 次——三分搜索需要**额外 2.41 倍**的迭代次数，但每次迭代需要**两次**函数求值（而二分只需一次），因此三分搜索的函数求值次数约为二分的 $2 \times 2.41 \approx 4.8$ 倍。

> **实践建议**：若能转化为"对导数二分"，优先选择这种方案；在只能黑盒访问函数值的场景（如竞赛题的评分函数），使用三分搜索。

---

## 本章小结

### 知识图谱

```
二分搜索
├── 17.1 三种核心模板
│   ├── 模板一：精确查找（while l <= r，相等时 return mid）
│   ├── 模板二：左边界 lower_bound（while l < r，>= 时 r = mid）
│   └── 模板三：上界 upper_bound（while l < r，> 时 r = mid）
│
├── 17.2 扩展应用
│   ├── 旋转有序数组（判断哪半段有序后二分）
│   ├── 峰值元素（局部单调性 + 爬坡法）
│   ├── 对答案二分（最优化 → 可行性判断）
│   └── 浮点数二分（固定迭代次数最安全）
│
├── 17.3 指数搜索
│   └── 倍增 + 二分，O(log i)，适合无界数组
│
└── 17.4 三分搜索
    └── 单峰/单谷函数最值，O(log n)，每步缩至 2/3
```

### 复杂度速查表

| 算法 | 时间复杂度 | 空间复杂度 | 前提 |
|---|---|---|---|
| 精确二分查找 | $O(\log n)$ | $O(1)$ | 有序数组 |
| lower_bound / upper_bound | $O(\log n)$ | $O(1)$ | 有序数组（含重复）|
| 旋转数组二分 | $O(\log n)$ | $O(1)$ | 旋转有序数组（无重复）|
| 峰值元素查找 | $O(\log n)$ | $O(1)$ | 局部单调性存在 |
| 对答案二分 | $O(n \log(\text{range}))$ | $O(1)$ | check 函数单调 |
| 浮点数二分 | $O(\log(1/\varepsilon))$ | $O(1)$ | 函数单调 |
| 指数搜索 | $O(\log i)$ | $O(1)$ | 有序数组，$i$ = 目标下标 |
| 三分搜索 | $O(\log n)$ | $O(1)$ | 单峰/单谷函数 |

### 常见错误与陷阱总结

> ⚠️ **陷阱一：溢出问题**
> 
> **错误**：`mid = (l + r) / 2`  
> **正确**：`mid = l + (r - l) / 2`  
> 在 C++ 中，`l + r` 可能溢出 `int` 范围（约 $2.1 \times 10^9$）。对答案二分时搜索范围经常接近 $10^9$，两个 $10^9$ 相加必然溢出。

> ⚠️ **陷阱二：边界收缩方向错误（死循环）**
> 
> **错误**（使用 `while l <= r`）：`l = mid`（应为 `l = mid + 1`）  
> **原因**：当 `l == r` 时，`mid == l`，`l = mid` 不变，无限循环。  
> **规则**：`while l <= r` 必须保证 `l = mid + 1` 和 `r = mid - 1`（严格排除 mid）。

> ⚠️ **陷阱三：找左边界时，遇到等于 target 就停止**
> 
> **错误**：找到第一个 `arr[mid] == target` 就 `return mid`。  
> **正确**：应继续令 `r = mid`，继续向左搜索，确保返回的是**最左边**那个。

> ⚠️ **陷阱四：对答案二分时搜索空间边界设置错误**
> 
> `lo` 应设为"答案的最小可能值"，`hi` 应设为"答案的最大可能值"。边界设窄会漏掉答案，设得太宽影响效率但不影响正确性。遇到不确定时，大胆地扩大搜索空间，如令 `hi = 2e18`。

> ⚠️ **陷阱五：check 函数的单调性方向判断错误**
> 
> 在对答案二分时，必须明确：check 函数是"越大越容易满足"还是"越小越容易满足"？对应地，找的是左边界还是右边界？搞反了，结果完全错误。

### 经典 LeetCode 题导引

| 题号 | 题目 | 知识点 | 难度 |
|---|---|---|---|
| #704 | Binary Search | 模板一：精确查找 | 简单 |
| #35 | Search Insert Position | 模板二：左边界 | 简单 |
| #34 | Find First and Last Position | lower_bound + upper_bound | 中等 |
| #33 | Search in Rotated Sorted Array | 旋转数组二分 | 中等 |
| #81 | Search in Rotated Sorted Array II | 含重复元素的旋转 | 中等 |
| #162 | Find Peak Element | 局部单调性二分 | 中等 |
| #69 | Sqrt(x) | 整数二分 + 边界细节 | 简单 |
| #875 | Koko Eating Bananas | 对答案二分 | 中等 |
| #1011 | Capacity to Ship Packages | 对答案二分 | 中等 |
| #410 | Split Array Largest Sum | 对答案二分 + 贪心 | 困难 |
| #4 | Median of Two Sorted Arrays | 对分割点二分（高难度）| 困难 |

### 思考题

> **💡 思考题一**："对答案二分"最关键的步骤是什么？如何证明 check 函数具有单调性？请用 LeetCode #410（分割数组的最大值）举例说明。
>
> **提示**：思考"最大段和上限 $m$ 增大"时，所需段数的变化方向。

> **💡 思考题二**：在三分搜索中，为什么不能用"两点分成三段"以外的方式（比如随机选择两点）来缩小搜索空间？
>
> **提示**：若选的两点 $m_1, m_2$ 不将 $[l, r]$ 三等分，极端情况下可能只缩小极少量空间，导致收敛极慢甚至不收敛。

> **💡 思考题三**：能否用二分搜索解决"在 $n \times n$ 矩阵中，每行每列均已排序，查找 target"（LeetCode #240）？如果不能，应该用什么策略？
>
> **提示**：矩阵不满足"一维有序"的前提，无法直接二分。可考虑从右上角开始的线性搜索策略，$O(n)$ 时间完成。
