# Chapter 9: 堆与优先队列（Heap & Priority Queue）

## 学习目标

- 理解堆的"完全二叉树 + 数组存储"结构，掌握父子索引公式
- 掌握 MAX-HEAPIFY 下沉操作的递归逻辑与 O(log n) 证明
- 理解 BUILD-MAX-HEAP 线性建堆的 O(n) 证明（级数求和）
- 能用 Python `heapq` 与 C++ `priority_queue` 解决 Top-K、合并 K 路、数据流中位数等高频题
- 理解堆排序 O(n log n) 与 O(1) 空间的权衡及缓存局部性缺陷

> **典型 LeetCode**：#215（第K大）、#295（数据流中位数）、#23（合并K路链表）、#347（前K高频）、#373、#378

---

## 9.1 堆的基本概念

### 9.1.1 直觉比喻：插队式优先排队

想象一个医院急诊室：病人来了不是按到达顺序就诊，而是**按病情紧急程度**（优先级）就诊。

- 最危重的病人永远排在队列头部（最大堆：最大值在顶）
- 新病人来了重新按优先级排位（INSERT）
- 医生每次接诊的都是当时最危重的（EXTRACT-MAX）

这就是**优先队列（Priority Queue）**的本质，而**堆（Heap）**是实现优先队列最高效的数据结构。

### 9.1.2 堆性质（Heap Property）

**最大堆（Max-Heap）**：每个节点的键值 ≥ 其所有子节点的键值

$$\forall i > 0: \quad A[\text{parent}(i)] \geq A[i]$$

这意味着**根永远是最大值**（全局最大一眼即知）。

**最小堆（Min-Heap）**：每个节点的键值 ≤ 其所有子节点的键值

$$\forall i > 0: \quad A[\text{parent}(i)] \leq A[i]$$

根永远是最小值。Python 的 `heapq` 是**最小堆**；C++ 的 `priority_queue` 默认是**最大堆**。

> ⚠️ **堆性质 ≠ 全排序**！堆只保证父 ≥ 子，不保证兄弟之间有序。给定最大堆，无法在 O(1) 内找到第 2 大值——只有根是确定的最大。

### 9.1.3 完全二叉树的数组存储（0-indexed）

堆用**一维数组**表示一棵**完全二叉树**（Complete Binary Tree）。

完全二叉树：除最后一层外每层全满，最后一层所有节点靠左排列。

```
数组：[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
索引：  0   1   2  3  4   5  6  7   8  9

对应树形：
              16 [0]
            /       \
          14 [1]   10 [2]
         /    \    /    \
       8 [3] 7 [4] 9 [5] 3 [6]
      / \ / \
    2[7]4[8]1[9]
```

### 9.1.4 父子索引关系（0-indexed）

| 关系 | 公式（0-indexed） | 例（i=3） |
|------|-----------------|---------|
| 父节点 | `parent(i) = (i-1) // 2` | `(3-1)//2 = 1` ✓ |
| 左子节点 | `left(i) = 2*i + 1` | `2*3+1 = 7` ✓ |
| 右子节点 | `right(i) = 2*i + 2` | `2*3+2 = 8` ✓ |

> 📌 CLRS 教材使用 1-indexed（parent=i//2, left=2i, right=2i+1），Python 实际实现用 0-indexed。

### 9.1.5 堆高度

$n$ 个节点的完全二叉树高度 $h = \lfloor \log_2 n \rfloor$。

**推导**：完全二叉树第 $0$ 层到第 $h-1$ 层共有 $2^h - 1$ 个节点，加上最后一层至多 $2^h$ 个，故满足 $2^h - 1 < n \leq 2^{h+1} - 1$，即 $h = \lfloor \log_2 n \rfloor$。

这是所有堆操作 O(log n) 的根本来源。

---

## 9.2 二叉堆操作（CLRS 风格）

### 9.2.1 MAX-HEAPIFY：下沉操作

**问题**：节点 `i` 的值可能比某个子节点小，违反堆性质。假设 `i` 的两棵子树均满足堆性质，通过下沉操作恢复。

**CLRS 伪代码**：

```
MAX-HEAPIFY(A, i):
    l = LEFT(i);  r = RIGHT(i)
    if l ≤ A.heap-size and A[l] > A[i]:
        largest = l
    else:
        largest = i
    if r ≤ A.heap-size and A[r] > A[largest]:
        largest = r
    if largest ≠ i:
        swap A[i] and A[largest]
        MAX-HEAPIFY(A, largest)   // 递归继续下沉
```

**正确性**：

- 每次将 `i`、左子、右子中最大的放到父位置
- 若 `largest ≠ i`，父处原值被移到了某子节点，继续对该子节点 HEAPIFY
- 子树已满足堆性质 + 父处放了三者最大值 → 父处满足堆性质
- 递归向下，最终终止（树高有限）

**双语实现**：

```python
def max_heapify(arr, n, i):
    """对 arr[0..n-1] 中以 i 为根的子树做下沉操作（0-indexed）"""
    largest = i
    l = 2 * i + 1   # 左子节点
    r = 2 * i + 2   # 右子节点

    # 找 i、左子、右子中的最大者
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]   # 交换
        max_heapify(arr, n, largest)                   # 递归下沉
```

```cpp
void max_heapify(vector<int>& arr, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;   // 左子节点
    int r = 2 * i + 2;   // 右子节点

    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        max_heapify(arr, n, largest);  // 递归下沉
    }
}
```

**复杂度分析**：

- 每层操作 O(1)（比较 + 可能交换）
- 最多递归 $h = \lfloor \log_2 n \rfloor$ 层
- **时间 O(log n)**，空间 O(log n)（递归栈）或改写迭代版 O(1)

**迭代版（避免栈溢出）**：

```python
def max_heapify_iter(arr, n, i):
    """迭代版 O(1) 额外空间"""
    while True:
        largest = i
        l, r = 2 * i + 1, 2 * i + 2
        if l < n and arr[l] > arr[largest]: largest = l
        if r < n and arr[r] > arr[largest]: largest = r
        if largest == i: break          # 堆性质已满足，停止
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest                     # 继续向下
```

```cpp
void max_heapify_iter(vector<int>& arr, int n, int i) {
    while (true) {
        int largest = i, l = 2*i+1, r = 2*i+2;
        if (l < n && arr[l] > arr[largest]) largest = l;
        if (r < n && arr[r] > arr[largest]) largest = r;
        if (largest == i) break;
        swap(arr[i], arr[largest]);
        i = largest;
    }
}
```

<div data-component="HeapifyAnimation"></div>

### 9.2.2 BUILD-MAX-HEAP：线性建堆

**目标**：将任意数组转换为满足最大堆性质的数组。

**关键洞察**：叶子节点（下标 $> \lfloor n/2 \rfloor - 1$）天然是单节点堆，无需 HEAPIFY。只需从 $\lfloor n/2 \rfloor - 1$ 倒序到 $0$，逐一调用 MAX-HEAPIFY。

**CLRS 伪代码**：

```
BUILD-MAX-HEAP(A):
    A.heap-size = A.length
    for i = ⌊A.length/2⌋ downto 1:    // CLRS 1-indexed
        MAX-HEAPIFY(A, i)
```

**双语实现**：

```python
def build_max_heap(arr):
    """原地将 arr 转换为最大堆，O(n) 时间"""
    n = len(arr)
    # 从最后一个非叶子节点开始，倒序调用 HEAPIFY
    for i in range(n // 2 - 1, -1, -1):
        max_heapify(arr, n, i)
    # 时间复杂度：O(n)（不是 O(n log n)，见 9.2.3 证明）
```

```cpp
void build_max_heap(vector<int>& arr) {
    int n = arr.size();
    // 从最后一个非叶子节点开始，倒序调用 HEAPIFY
    for (int i = n / 2 - 1; i >= 0; i--)
        max_heapify(arr, n, i);
    // 时间复杂度：O(n)
}
```

**为什么从 n//2-1 开始？**

- 完全二叉树中，叶子节点的下标范围是 $[\lfloor n/2 \rfloor, n-1]$（0-indexed）
- 叶子只有自己，天然满足堆性质，无需处理
- 第一个非叶子节点是 $\lfloor n/2 \rfloor - 1$（它是最后一个叶子的父节点）

**示例（n=10）**：

```
arr = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]

从 i=4 开始（arr[4]=16，左子=arr[9]=7无右子）：
  i=4: HEAPIFY(4) → 16>7，无需交换
  i=3: HEAPIFY(3) → 14>2，交换 arr[3]=14, arr[3]=2...
  i=2: HEAPIFY(2) → 10>3，...
  i=1: HEAPIFY(1) → ...
  i=0: HEAPIFY(0) → ...

结果：arr = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
```

### 9.2.3 线性建堆的 O(n) 证明

**关键定理**：BUILD-MAX-HEAP 时间复杂度为 $O(n)$，而非直觉上的 $O(n \log n)$。

**证明（CLRS 方法）**：

高度为 $h$ 的节点最多有 $\lceil n/2^{h+1} \rceil$ 个，每次 HEAPIFY 代价 $O(h)$。

$$T(n) = \sum_{h=0}^{\lfloor \log n \rfloor} \left\lceil \frac{n}{2^{h+1}} \right\rceil \cdot O(h)$$

$$\leq \sum_{h=0}^{\infty} \frac{n}{2^{h+1}} \cdot ch = \frac{cn}{2} \sum_{h=0}^{\infty} \frac{h}{2^h}$$

利用等比级数公式：$\sum_{h=0}^{\infty} h \cdot x^h = \frac{x}{(1-x)^2}$，令 $x = 1/2$：

$$\sum_{h=0}^{\infty} \frac{h}{2^h} = \frac{1/2}{(1-1/2)^2} = 2$$

$$\therefore T(n) \leq \frac{cn}{2} \cdot 2 = cn = O(n)$$

**直觉理解**：叶子节点（占总数的 ~1/2）高度为 0，代价为 0；越靠近根的节点代价越高但数量指数级减少。两者相乘再求和收敛于常数倍的 $n$。

<div data-component="BuildHeapLinearProof"></div>

### 9.2.4 HEAP-EXTRACT-MAX：弹出最大值

**弹出最大值**（根节点）后，需要维护堆性质：

1. 将根（最大值）与最后一个元素交换
2. 堆大小减 1（逻辑上删除末元素）
3. 对新根调用 MAX-HEAPIFY 恢复堆性质

**双语实现**：

```python
def heap_extract_max(arr, heap_size):
    """弹出并返回最大值，O(log n)"""
    if heap_size < 1:
        raise IndexError("heap underflow")
    max_val = arr[0]
    arr[0] = arr[heap_size - 1]   # 末元素放到根
    heap_size -= 1
    max_heapify(arr, heap_size, 0)  # 重新 HEAPIFY 根
    return max_val, heap_size
```

```cpp
int heap_extract_max(vector<int>& arr, int& heap_size) {
    if (heap_size < 1) throw runtime_error("heap underflow");
    int max_val = arr[0];
    arr[0] = arr[--heap_size];    // 末元素放到根，堆大小减 1
    max_heapify(arr, heap_size, 0);
    return max_val;
}
```

**复杂度**：交换 O(1)，HEAPIFY O(log n)，总计 **O(log n)**。

### 9.2.5 HEAP-INCREASE-KEY：增大键值（上浮）

**场景**：要增大某节点的键值（如优先级提升），增大后可能违反父 ≥ 子，需要向上"上浮"（Sift-Up / Bubble-Up）。

```
HEAP-INCREASE-KEY(A, i, key):
    if key < A[i]:  error "new key is smaller"
    A[i] = key
    while i > 0 and A[PARENT(i)] < A[i]:
        swap A[i] and A[PARENT(i)]
        i = PARENT(i)
```

**双语实现**：

```python
def heap_increase_key(arr, i, new_key):
    """增大 arr[i] 的键值到 new_key，O(log n)"""
    if new_key < arr[i]:
        raise ValueError("new key is smaller than current key")
    arr[i] = new_key
    # 上浮：只要父节点比自己小就交换
    while i > 0:
        parent = (i - 1) // 2
        if arr[parent] < arr[i]:
            arr[parent], arr[i] = arr[i], arr[parent]
            i = parent
        else:
            break
```

```cpp
void heap_increase_key(vector<int>& arr, int i, int new_key) {
    if (new_key < arr[i]) throw invalid_argument("new key smaller");
    arr[i] = new_key;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (arr[parent] < arr[i]) {
            swap(arr[parent], arr[i]);
            i = parent;
        } else break;
    }
}
```

**复杂度**：最多上浮 $h = O(\log n)$ 层，**O(log n)**。

### 9.2.6 MAX-HEAP-INSERT：插入

**策略**：先在末尾插入 $-\infty$，再调用 HEAP-INCREASE-KEY 设置实际值（等价于上浮）。

**双语实现**：

```python
def max_heap_insert(arr, key):
    """向最大堆插入新键值，O(log n)"""
    arr.append(float('-inf'))           # 先插入哨兵 -∞
    heap_increase_key(arr, len(arr)-1, key)  # 再上浮到正确位置
```

```cpp
void max_heap_insert(vector<int>& arr, int key) {
    arr.push_back(INT_MIN);               // 先插入哨兵
    heap_increase_key(arr, arr.size()-1, key);
}
```

**所有操作复杂度汇总**：

| 操作 | 时间复杂度 | 说明 |
|------|----------|------|
| BUILD-MAX-HEAP | O(n) | 线性建堆 |
| MAX-HEAPIFY | O(log n) | 下沉 |
| HEAP-EXTRACT-MAX | O(log n) | 弹出最大 + HEAPIFY |
| HEAP-INCREASE-KEY | O(log n) | 增大键值 + 上浮 |
| MAX-HEAP-INSERT | O(log n) | 插末 + 上浮 |
| HEAP-MAXIMUM | O(1) | 仅查根 |

---

## 9.3 堆排序（HeapSort）

### 9.3.1 堆排序完整算法

**思想**：

1. **BUILD-MAX-HEAP**：把整个数组建成最大堆（O(n)）
2. **n 次 EXTRACT-MAX**：每次将根（当前最大值）与末尾交换，堆大小减 1，HEAPIFY（O(log n)）

重复 $n-1$ 次后，数组升序排列。

**双语实现**：

```python
def heap_sort(arr):
    """堆排序，原地排序，O(n log n) 时间，O(1) 额外空间"""
    n = len(arr)
    # 阶段 1：建最大堆 O(n)
    build_max_heap(arr)
    # 阶段 2：逐步提取最大值 O(n log n)
    for i in range(n - 1, 0, -1):
        # 将最大值（根）放到末尾
        arr[0], arr[i] = arr[i], arr[0]
        # 堆大小减 1，对新根 HEAPIFY
        max_heapify(arr, i, 0)
    # 最终：arr 升序排列
```

```cpp
void heap_sort(vector<int>& arr) {
    int n = arr.size();
    // 阶段 1：建最大堆
    build_max_heap(arr);
    // 阶段 2：逐步提取
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);          // 最大值放末尾
        max_heapify(arr, i, 0);        // 堆大小减 1 后 HEAPIFY
    }
}
```

<div data-component="HeapSortTrace"></div>

### 9.3.2 复杂度分析

- **时间**：BUILD O(n) + $n$ 次 HEAPIFY O(log n) = **O(n log n)**
- **空间**：O(1)（原地排序，仅常数个辅助变量；递归 HEAPIFY 可改迭代）
- **稳定性**：**不稳定**（EXTRACT 时末元素放到根，破坏原有顺序）

### 9.3.3 堆排序 vs 快排 vs 归并排序

| 算法 | 时间（平均） | 时间（最坏） | 空间 | 稳定性 | 缓存友好 |
|-----|------------|------------|------|--------|---------|
| 堆排序 | O(n log n) | O(n log n) | O(1) | ❌ | ❌ 差 |
| 快速排序 | O(n log n) | O(n²) | O(log n) | ❌ | ✅ 好 |
| 归并排序 | O(n log n) | O(n log n) | O(n) | ✅ | ✅ 好 |
| 内省排序（introsort） | O(n log n) | O(n log n) | O(log n) | ❌ | ✅ 好 |

### 9.3.4 为什么堆排序缓存不友好？

堆排序在 HEAPIFY 时，访问模式是 `arr[i] → arr[2i+1] → arr[4i+3] → ...`，索引跳跃式增长（从数组前部跳到中部、后部），**与 CPU 缓存的空间局部性（Spatial Locality）相悖**。

快速排序的 partition 步骤从两端向中间扫描，访问连续内存，缓存命中率高得多。

这也是为什么工程实践中（C++ STL `std::sort`、Python Timsort）不直接用堆排序——它仅在需要绝对 O(n log n) 最坏保证且不在乎常数因子时才用（如内省排序的后备）。

---

## 9.4 优先队列（Priority Queue）应用

### 9.4.1 Python `heapq` 模块

Python 标准库 `heapq` 实现了**最小堆**（min-heap），接口简洁：

```python
import heapq

# ─── 基本操作 ───
h = []
heapq.heappush(h, 5)    # INSERT: O(log n)
heapq.heappush(h, 3)
heapq.heappush(h, 8)
print(h[0])              # PEEK(最小值): O(1), 输出 3
val = heapq.heappop(h)   # EXTRACT-MIN: O(log n), 弹出 3

# 从列表直接建堆: O(n)
data = [9, 1, 5, 3, 7]
heapq.heapify(data)      # 原地建最小堆
print(data[0])           # 输出 1

# 模拟最大堆：存 (-val, val)
max_heap = []
for v in [9, 1, 5, 3, 7]:
    heapq.heappush(max_heap, -v)   # 注意取负！
max_val = -heapq.heappop(max_heap) # 取出最大值：9

# heappushpop: PUSH 后 POP，比分开调用更快
result = heapq.heappushpop(h, 0)   # 先 push 0, 再 pop 最小

# nlargest / nsmallest
biggest_3 = heapq.nlargest(3, data)   # O(n + k log n)
```

```cpp
#include <queue>
#include <vector>
using namespace std;

// C++ priority_queue 默认 MAX-HEAP
priority_queue<int> max_pq;
max_pq.push(5);
max_pq.push(3);
max_pq.push(8);
int top = max_pq.top();    // 查看最大值: 8
max_pq.pop();              // 弹出最大值

// MIN-HEAP: 使用 greater<int>
priority_queue<int, vector<int>, greater<int>> min_pq;
min_pq.push(5);
min_pq.push(3);
min_pq.push(8);
int min_top = min_pq.top(); // 3

// 自定义比较（按第二元素排序）
using P = pair<int,int>;
priority_queue<P, vector<P>, greater<P>> custom_pq;
custom_pq.push({1, 10});
custom_pq.push({3, 5});
```

### 9.4.2 Top-K 问题（LeetCode #215）

**问题**：数组中第 K 大的元素。

**方法一：最小堆维护大小为 K 的窗口**（O(n log k)，适合数据流）

维护一个**大小为 K 的最小堆**，堆中始终是当前最大的 K 个元素，堆顶即第 K 大。

```python
import heapq

def findKthLargest_heap(nums, k):
    """最小堆维护 K 个最大元素，O(n log k) 时间，O(k) 空间"""
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # 维持大小为 k
    return min_heap[0]  # 堆顶即第 k 大
```

```cpp
int findKthLargest(vector<int>& nums, int k) {
    // min-heap，大小维持为 k
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int num : nums) {
        pq.push(num);
        if ((int)pq.size() > k) pq.pop();
    }
    return pq.top();
}
```

**方法二：快速选择（Quick Select）**（O(n) 期望，O(1) 额外空间，更快但不支持数据流）

```python
def findKthLargest_qs(nums, k):
    """快速选择，平均 O(n)，最坏 O(n^2)"""
    target = len(nums) - k   # 第 k 大 = 升序第 (n-k) 小（0-indexed）
    def quick_select(lo, hi):
        pivot = nums[hi]
        p = lo
        for i in range(lo, hi):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[hi] = nums[hi], nums[p]
        if p == target:   return nums[p]
        elif p < target:  return quick_select(p + 1, hi)
        else:             return quick_select(lo, p - 1)
    return quick_select(0, len(nums) - 1)
```

```cpp
int quickSelect(vector<int>& nums, int lo, int hi, int target) {
    int pivot = nums[hi], p = lo;
    for (int i = lo; i < hi; i++)
        if (nums[i] <= pivot) swap(nums[p++], nums[i]);
    swap(nums[p], nums[hi]);
    if (p == target) return nums[p];
    if (p < target)  return quickSelect(nums, p+1, hi, target);
    return quickSelect(nums, lo, p-1, target);
}
int findKthLargest(vector<int>& nums, int k) {
    return quickSelect(nums, 0, nums.size()-1, nums.size()-k);
}
```

### 9.4.3 合并 K 路有序链表（LeetCode #23）

将 K 个已排序链表合并为一个有序链表。

**思路**：用最小堆维护 K 个链表的当前最小头节点，每次弹出全局最小，将其仍有后继则入堆。

**时间**：O(N log K)（N = 总节点数）；**空间**：O(K)（堆大小）

```python
import heapq
from typing import Optional, List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists):
    """合并 K 路有序链表，O(N log K) 时间"""
    heap = []
    counter = 0  # 避免 ListNode 直接比较
    for node in lists:
        if node:
            heapq.heappush(heap, (node.val, counter, node))
            counter += 1
    dummy = ListNode()
    cur = dummy
    while heap:
        val, _, node = heapq.heappop(heap)  # 弹出当前全局最小
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1
    return dummy.next
```

```cpp
struct ListNode { int val; ListNode* next; };

struct Cmp {
    bool operator()(ListNode* a, ListNode* b) { return a->val > b->val; }
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    priority_queue<ListNode*, vector<ListNode*>, Cmp> pq;
    for (auto node : lists)
        if (node) pq.push(node);     // 初始化：K 个头节点入堆
    ListNode dummy, *cur = &dummy;
    while (!pq.empty()) {
        auto node = pq.top(); pq.pop();
        cur->next = node;
        cur = cur->next;
        if (node->next) pq.push(node->next);
    }
    return dummy.next;
}
```

### 9.4.4 数据流中位数（LeetCode #295）

**问题**：在线维护数据流的中位数，支持 `addNum` 和 `findMedian` 两个操作。

**双堆维护**：

- `lo`：**最大堆**（存较小的一半），堆顶 = 较小半的最大值
- `hi`：**最小堆**（存较大的一半），堆顶 = 较大半的最小值

**不变量**：

1. $|lo| = |hi|$ 或 $|lo| = |hi| + 1$（`lo` 可以多存 1 个）
2. $lo.\text{top} \leq hi.\text{top}（顺序不变量）$

**中位数**：`lo` 比 `hi` 多 1 个时 → 中位数 = `lo.top()`；相等时 → 中位数 = `(lo.top() + hi.top()) / 2`

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []   # max-heap（存负数模拟）：较小一半
        self.hi = []   # min-heap：较大一半

    def addNum(self, num):
        heapq.heappush(self.lo, -num)    # 先放入 lo（max-heap）
        # 顺序不变量修复：lo 顶若 > hi 顶，移过去
        if self.hi and -self.lo[0] > self.hi[0]:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        # 大小平衡：lo 最多比 hi 多 1 个
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])        # lo 多 1 个
        return (-self.lo[0] + self.hi[0]) / 2.0
```

```cpp
class MedianFinder {
    priority_queue<int> lo;                              // max-heap（较小半）
    priority_queue<int, vector<int>, greater<int>> hi;   // min-heap（较大半）
public:
    void addNum(int num) {
        lo.push(num);
        // 顺序不变量：lo.top 不能 > hi.top
        if (!hi.empty() && lo.top() > hi.top()) {
            hi.push(lo.top()); lo.pop();
        }
        // 大小平衡
        if (lo.size() > hi.size() + 1) { hi.push(lo.top()); lo.pop(); }
        else if (hi.size() > lo.size()) { lo.push(hi.top()); hi.pop(); }
    }
    double findMedian() {
        if (lo.size() > hi.size()) return lo.top();
        return (lo.top() + hi.top()) / 2.0;
    }
};
```

<div data-component="DualHeapMedian"></div>

### 9.4.5 任务调度与优先级队列

<div data-component="PriorityQueueScheduler"></div>

### 9.4.6 前 K 个高频元素（LeetCode #347）

**思路**：哈希表统计频率，再用最小堆维护大小为 K 的窗口。

```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    """前 K 个高频元素，O(n log k)"""
    freq = Counter(nums)           # O(n) 统计频率
    # 最小堆，按频率排，维持大小为 k
    heap = []
    for num, cnt in freq.items():
        heapq.heappush(heap, (cnt, num))
        if len(heap) > k:
            heapq.heappop(heap)    # 弹出频率最低的
    return [x[1] for x in heap]
```

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int,int> freq;
    for (int n : nums) freq[n]++;
    // min-heap 按频率排
    using P = pair<int,int>;  // {freq, val}
    priority_queue<P, vector<P>, greater<P>> pq;
    for (auto& [val, cnt] : freq) {
        pq.push({cnt, val});
        if ((int)pq.size() > k) pq.pop();
    }
    vector<int> res;
    while (!pq.empty()) { res.push_back(pq.top().second); pq.pop(); }
    return res;
}
```

---

## 9.5 d-叉堆（d-ary Heap）

### 9.5.1 d-叉堆结构

**d-叉堆**（d-ary Heap）将二叉堆推广到每个节点有 $d$ 个子节点（d=2 即普通二叉堆）。

**索引关系（0-indexed，d 叉）**：

- 父节点：`parent(i) = (i-1) // d`
- 第 $j$ 个子节点（$0 \leq j < d$）：`child(i, j) = d*i + j + 1`

**操作复杂度**：

| 操作 | d-叉堆 | d=2（二叉堆） |
|------|-------|------------|
| EXTRACT-MAX | O(d log_d n) | O(log n) |
| INSERT / DECREASE-KEY | O(log_d n) | O(log n) |
| BUILD | O(n) | O(n) |

**权衡**：$d$ 越大，树越矮（$O(\log_d n)$ 层），上浮操作更快；但下沉操作每层需比较 $d$ 个子节点，代价 $O(d \log_d n)$。

### 9.5.2 d=4 在 Dijkstra 中的优化

Dijkstra 算法对优先队列的操作：EXTRACT-MIN 次数少（每节点 1 次），DECREASE-KEY 次数多（每边 1 次）。

- 二叉堆：DECREASE-KEY = O(log n)，EXTRACT = O(log n)
- 4-叉堆：DECREASE-KEY = O(log_4 n) ≈ O(log n / 2)，EXTRACT = O(4 log_4 n) ≈ O(2 log n)

图边稠密时（E >> V），DECREASE-KEY 次数远多于 EXTRACT，4-叉堆将总时间从约 $O(V \log V + E \log V)$ 降低约 0.5 的常数因子。

```python
class DaryHeap:
    """d 叉最小堆实现"""
    def __init__(self, d=4):
        self.d = d
        self.data = []

    def _parent(self, i): return (i - 1) // self.d
    def _child(self, i, j): return self.d * i + j + 1

    def push(self, val):
        self.data.append(val)
        self._sift_up(len(self.data) - 1)

    def _sift_up(self, i):
        while i > 0:
            p = self._parent(i)
            if self.data[p] > self.data[i]:
                self.data[p], self.data[i] = self.data[i], self.data[p]
                i = p
            else: break

    def pop(self):
        if not self.data: raise IndexError("empty")
        self.data[0] = self.data[-1]
        self.data.pop()
        if self.data: self._sift_down(0)
        return  # (改造为返回最小值版本)

    def _sift_down(self, i):
        n = len(self.data)
        while True:
            smallest = i
            for j in range(self.d):
                c = self._child(i, j)
                if c < n and self.data[c] < self.data[smallest]:
                    smallest = c
            if smallest == i: break
            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            i = smallest
```

```cpp
class DaryHeap {
    int d;
    vector<int> data;
    int parent(int i) { return (i - 1) / d; }
    int child(int i, int j) { return d * i + j + 1; }

    void sift_up(int i) {
        while (i > 0) {
            int p = parent(i);
            if (data[p] > data[i]) { swap(data[p], data[i]); i = p; }
            else break;
        }
    }
    void sift_down(int i) {
        int n = data.size();
        while (true) {
            int smallest = i;
            for (int j = 0; j < d; j++) {
                int c = child(i, j);
                if (c < n && data[c] < data[smallest]) smallest = c;
            }
            if (smallest == i) break;
            swap(data[i], data[smallest]);
            i = smallest;
        }
    }
public:
    DaryHeap(int d = 4) : d(d) {}
    void push(int val) { data.push_back(val); sift_up(data.size()-1); }
    int top() { return data[0]; }
    void pop() { data[0] = data.back(); data.pop_back(); if (!data.empty()) sift_down(0); }
};
```

---

## 9.6 总结与复习

### 所有堆操作复杂度汇总

| 操作 | 二叉堆 | d-叉堆 | Python heapq |
|------|--------|--------|-------------|
| BUILD | O(n) | O(n) | `heapify(arr)` |
| INSERT | O(log n) | O(log_d n) | `heappush(h, val)` |
| EXTRACT-MIN/MAX | O(log n) | O(d log_d n) | `heappop(h)` |
| PEEK | O(1) | O(1) | `h[0]` |
| DECREASE-KEY | O(log n) | O(log_d n) | 无直接接口 |
| HEAP-SORT | O(n log n) | — | — |

### ⚠️ 常见错误汇总

| 错误 | 正确做法 |
|------|---------|
| Python heapq 当最大堆用 | 存 `(-val, val)` 或用比较类 |
| BUILD 从 n//2 而非 n-1 开始 | 叶子天然满足堆性质，第一个非叶子在 `n//2 - 1`（0-indexed） |
| EXTRACT 后忘记 HEAPIFY | 末元素放到根后必须重新 HEAPIFY |
| 线性建堆误认为 O(n log n) | 级数求和证明其为 O(n) |
| 中位数双堆忘记维持顺序不变量 | 插入后检查 lo.top ≤ hi.top |

### 💡 思考题

1. **O(n) 建堆等价于？**：将 n 个元素逐一 INSERT（一次 O(log n)）总代价是 O(n log n)，而 BUILD-MAX-HEAP 是 O(n)。两者的差异来源于哪里？
2. **堆支持 DECREASE-KEY 但不支持任意删除的原因？**：如果要在 O(log n) 内删除任意节点，需要记录每个元素在堆中的位置，维护一个哈希表，代价是 O(1) 额外存储。Dijkstra 的斐波那契堆正是这样做的。
3. **为什么用 top-K 最小堆而非最大堆？**：维护 K 个最大值的最小堆可以快速判断新元素是否值得入堆（只需和堆顶比较），若用最大堆则每次都不知道该淘汰谁。
4. **堆排序缓存不友好**：用具体例子（n=16 数组）计算 HEAPIFY 时跳跃的访问模式，与快速排序 partition 的顺序扫描对比，量化 cache miss 次数差异。

**参考资料**：CLRS 第4版 Chapter 6（堆）；Sedgewick 2.4（优先队列）；MIT 6.006 Lecture 4；LeetCode #215 #295 #23 #347
