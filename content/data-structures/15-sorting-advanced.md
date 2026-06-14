# Chapter 15: 高效排序算法（Advanced Sorting Algorithms）

> **学习目标**：  
> 理解归并排序分治思想与 $O(n \log n)$ 的主定理推导；掌握快速排序 PARTITION 的两种方案（Lomuto 与 Hoare）及随机化期望证明；通过决策树模型严格论证比较排序 $\Omega(n \log n)$ 下界；掌握突破下界的三种线性时间排序（计数/基数/桶）及其适用条件；能在面试与工程中做出合理算法选择并分析 TimSort 的现实优势。

---

## 15.1 归并排序（Merge Sort）

### 15.1.1 分治思想：把大问题劈成两半

归并排序是"分治法"（Divide and Conquer）的教科书级应用。面对一个长度为 $n$ 的数组，我们不直接处理它，而是：

1. **分（Divide）**：把数组从中间劈成两半，左半 $A[0..\lfloor n/2 \rfloor - 1]$，右半 $A[\lfloor n/2 \rfloor .. n-1]$
2. **治（Conquer）**：递归地把两半分别排好序（问题规模减半，边界条件：长度为 1 的数组天然有序）
3. **合（Combine）**：把两个已排好序的半数组**合并**（Merge）成一个有序数组

**直觉比喻**：想象你手里有两叠已经按面值排好序的扑克牌，现在把它们合并成一叠。你只需要反复比较两叠顶部的牌，取出较小的那张放入结果堆，直到某叠取完，再把另一叠整叠接上。这就是 MERGE 操作。

**图解过程**（以 `[5, 2, 4, 6, 1, 3]` 为例）：

```
分解阶段（自上而下）：
[5, 2, 4, 6, 1, 3]
    ↙            ↘
[5, 2, 4]      [6, 1, 3]
  ↙    ↘         ↙    ↘
[5, 2] [4]    [6, 1]  [3]
↙   ↘           ↙   ↘
[5] [2]        [6]  [1]

合并阶段（自下而上）：
[5] + [2] → [2, 5]       [6] + [1] → [1, 6]
[2, 5] + [4] → [2, 4, 5]  [1, 6] + [3] → [1, 3, 6]
[2, 4, 5] + [1, 3, 6] → [1, 2, 3, 4, 5, 6]  ✓
```

### 15.1.2 MERGE 过程——合并两个有序数组

MERGE 是归并排序的核心。它将两个相邻的已排序子数组 $A[l..m]$ 和 $A[m+1..r]$ 合并为 $A[l..r]$ 的一个有序数组。

**关键细节**：使用辅助数组 `L` 和 `R` 拷贝两半，再从头比较写回。两半末尾各加一个"哨兵"值（$+\infty$），避免边界判断。

**循环不变式**：每次循环开始前，$A[l..k-1]$ 包含 `L[0..i-1]` 和 `R[0..j-1]` 中的最小 $k-l$ 个元素，且已排序。

### 15.1.3 Python 与 C++ 完整实现

```python
def merge_sort(arr: list[int], l: int = 0, r: int = None) -> None:
    """
    原地归并排序（修改传入数组，辅助空间 O(n)）
    时间：Θ(n log n)，最好/平均/最坏全相同
    空间：O(n) 辅助数组 + O(log n) 递归栈
    稳定性：稳定（MERGE 中相等时优先取左半元素）

    参数：
        arr: 待排序列表
        l:   排序区间左端（含），默认 0
        r:   排序区间右端（含），默认 len(arr)-1
    """
    if r is None:
        r = len(arr) - 1

    # 递归基：区间长度 <= 1 时直接返回（天然有序）
    if l >= r:
        return

    # 分：找中点，防止 (l+r)//2 的整数溢出（Python 无所谓，C++ 要用 l+(r-l)//2）
    mid = l + (r - l) // 2

    # 治：递归排左半和右半
    merge_sort(arr, l, mid)
    merge_sort(arr, mid + 1, r)

    # 合：把排好的两半合并
    _merge(arr, l, mid, r)


def _merge(arr: list[int], l: int, mid: int, r: int) -> None:
    """
    合并 arr[l..mid] 和 arr[mid+1..r] 两个已排序子数组。
    时间：Θ(r - l + 1)，即 O(n)
    空间：O(r - l + 1) 辅助数组
    """
    # 拷贝左半和右半到辅助数组
    left  = arr[l : mid + 1]      # arr[l..mid]（含右端）
    right = arr[mid + 1 : r + 1]  # arr[mid+1..r]（含右端）

    i, j, k = 0, 0, l   # i 指向 left，j 指向 right，k 指向写入位置

    while i < len(left) and j < len(right):
        # 关键：<= 而非 <，相等时优先取左半 → 保证稳定性
        # 若改成 <，则相等时取右半，破坏稳定性
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    # 一半遍历完后，另一半剩余元素直接拷回（已排好序，无需再比较）
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


# === 使用示例 ===
arr = [5, 2, 4, 6, 1, 3]
merge_sort(arr)
print(arr)  # [1, 2, 3, 4, 5, 6]

# 验证稳定性：相同值的元素保持原来相对顺序
data = [(85, 'Alice'), (72, 'Bob'), (85, 'Carol')]
data.sort(key=lambda x: x[0])   # Python sort 用 TimSort（稳定）
print(data)  # [(72, 'Bob'), (85, 'Alice'), (85, 'Carol')]

# 归并排序计算逆序数（见 15.1.8）
def count_inversions(arr):
    """在归并排序过程中统计逆序对数，O(n log n)。"""
    if len(arr) <= 1:
        return arr[:], 0
    mid = len(arr) // 2
    left, cl = count_inversions(arr[:mid])
    right, cr = count_inversions(arr[mid:])
    merged, cm = [], 0
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            # left[i..] 都比 right[j] 大 → 产生 len(left)-i 个逆序对
            merged.append(right[j]); j += 1
            cm += len(left) - i
    merged += left[i:] + right[j:]
    return merged, cl + cr + cm

arr2 = [2, 3, 8, 6, 1]
_, inv = count_inversions(arr2)
print(f"逆序数：{inv}")   # 5
```

```cpp
#include <vector>
using namespace std;

/**
 * 归并两个已排序子数组 arr[l..mid] 和 arr[mid+1..r]。
 * 时间：O(r - l + 1)，空间：O(r - l + 1)
 * 稳定性：稳定（等号时取左半，即 arr[i] <= arr[j] 取左）
 */
void merge(vector<int>& arr, int l, int mid, int r) {
    // 拷贝两半到辅助数组
    vector<int> left(arr.begin() + l, arr.begin() + mid + 1);
    vector<int> right(arr.begin() + mid + 1, arr.begin() + r + 1);

    int i = 0, j = 0, k = l;

    while (i < (int)left.size() && j < (int)right.size()) {
        // <= 保证稳定性：相等时优先取左半
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }

    // 剩余部分直接拷回（已有序，无需再比较）
    while (i < (int)left.size())  arr[k++] = left[i++];
    while (j < (int)right.size()) arr[k++] = right[j++];
}

/**
 * 归并排序（递归版）
 * 时间：Θ(n log n)（最好/平均/最坏均相同）
 * 空间：O(n) 辅助 + O(log n) 栈
 */
void merge_sort(vector<int>& arr, int l, int r) {
    if (l >= r) return;   // 递归基：区间长度 <= 1

    // 防溢出写法：l + (r - l) / 2，而非 (l + r) / 2
    int mid = l + (r - l) / 2;

    merge_sort(arr, l, mid);       // 递归排左半
    merge_sort(arr, mid + 1, r);   // 递归排右半
    merge(arr, l, mid, r);         // 合并
}

// 对外接口（隐藏 l/r 参数）
void merge_sort(vector<int>& arr) {
    if (!arr.empty())
        merge_sort(arr, 0, (int)arr.size() - 1);
}

/*
int main() {
    vector<int> arr = {5, 2, 4, 6, 1, 3};
    merge_sort(arr);
    for (int x : arr) printf("%d ", x);  // 1 2 3 4 5 6
    return 0;
}
*/
```

### 15.1.4 时间复杂度：主定理推导

归并排序的递推式：

$$T(n) = \underbrace{2 \cdot T\!\left(\frac{n}{2}\right)}_{\text{两个子问题}} + \underbrace{\Theta(n)}_{\text{MERGE 代价}}$$

**主定理**（情形 2）：$a = 2$，$b = 2$，$f(n) = \Theta(n)$，$n^{\log_b a} = n^{\log_2 2} = n^1 = n$。

因为 $f(n) = \Theta(n^{\log_b a}) = \Theta(n)$，满足情形 2，故：

$$T(n) = \Theta(n \log n)$$

**递归树直观理解**：

```
层 0（根）：1 个问题，合并代价 cn（n 个元素）
层 1：2 个问题，合并代价各 c(n/2)，合计 cn
层 2：4 个问题，合并代价各 c(n/4)，合计 cn
...
层 k：2^k 个问题，合并代价各 c(n/2^k)，合计 cn
...
层 log₂n（叶）：n 个子问题，每个大小 1，代价 Θ(1)

共 log₂n + 1 层，每层合并代价均为 cn
总代价 = cn × (log₂n + 1) = Θ(n log n)
```

**关键性质**：归并排序的时间复杂度**与输入无关**——最好、平均、最坏均为 $\Theta(n \log n)$，这是快速排序不具备的。

<div data-component="MergeSortRecursionTree"></div>

### 15.1.5 归并排序的稳定性证明

MERGE 中，当 `left[i] == right[j]` 时，我们**优先取左半**（`arr[k] = left[i]`）。

设原数组中 $A[i] = A[j]$ 且 $i < j$（即 $A[i]$ 在左半，$A[j]$ 在右半），合并后 $A[i]$ 先被写入结果，$A[j]$ 在其之后——相对顺序不变。✓

> ⚠️ **陷阱**：若把条件改为 `left[i] < right[j]`（严格小于），则相等时优先取右半，破坏稳定性。这是实现归并排序时最常见的错误之一。

### 15.1.6 自底向上归并排序（Bottom-Up Merge Sort）

递归版归并需要 $O(\log n)$ 的调用栈空间。自底向上版从大小为 1 的子数组开始，逐轮将相邻子数组两两合并，不需要任何递归：

```python
def merge_sort_bottom_up(arr: list[int]) -> None:
    """
    自底向上归并排序（迭代版），无递归调用栈开销。
    时间：Θ(n log n)，空间：O(n) 辅助数组（调用栈降为 O(1)）
    适合：链表排序（LeetCode #148），避免链表随机访问的开销
    """
    n = len(arr)
    size = 1   # 当前子数组大小，从 1 开始，每轮翻倍

    while size < n:
        # 每轮：把所有相邻的 size 大小的子数组两两合并
        for l in range(0, n, 2 * size):
            mid = min(l + size - 1, n - 1)       # 左半末尾（不超过数组右边界）
            r   = min(l + 2 * size - 1, n - 1)   # 右半末尾（不超过数组右边界）
            if mid < r:   # 右半存在（不只有一个子数组）
                _merge(arr, l, mid, r)
        size *= 2   # 子数组大小翻倍
```

```cpp
void merge_sort_bottom_up(vector<int>& arr) {
    int n = arr.size();
    for (int size = 1; size < n; size *= 2) {
        for (int l = 0; l < n; l += 2 * size) {
            int mid = min(l + size - 1, n - 1);
            int r   = min(l + 2 * size - 1, n - 1);
            if (mid < r) {
                merge(arr, l, mid, r);
            }
        }
    }
}
```

### 15.1.7 链表上的归并排序（LeetCode #148）

归并排序天然适合链表：无需随机访问，只需修改指针，辅助空间可降为 $O(1)$（迭代版）。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def sort_list(head: ListNode) -> ListNode:
    """
    链表归并排序（LeetCode #148）
    时间：O(n log n)，空间：O(log n) 递归栈（迭代版可降为 O(1)）
    核心技巧：
      1. 快慢指针找链表中点（Floyd 判圈思路）
      2. 切断链表为两半（slow.next = None）
      3. 递归排两半，再合并两个有序链表
    """
    # 递归基：空链表或单节点，天然有序
    if not head or not head.next:
        return head

    # 快慢指针找中点：fast 每次走 2 步，slow 每次走 1 步
    # 当 fast 到末尾时，slow 在中点（偏左中位数）
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None   # 切断：左半 head..slow，右半 mid..末尾

    left  = sort_list(head)   # 递归排左半
    right = sort_list(mid)    # 递归排右半

    return _merge_list(left, right)   # 合并两个有序链表


def _merge_list(l1: ListNode, l2: ListNode) -> ListNode:
    """合并两个有序链表，返回合并后的头节点。空间 O(1)（只修改指针）"""
    dummy = ListNode(0)   # 哨兵节点简化头部处理
    cur = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 if l1 else l2   # 拼接剩余部分
    return dummy.next
```

```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy(0);
    ListNode* cur = &dummy;
    while (l1 && l2) {
        if (l1->val <= l2->val) { cur->next = l1; l1 = l1->next; }
        else                    { cur->next = l2; l2 = l2->next; }
        cur = cur->next;
    }
    cur->next = l1 ? l1 : l2;
    return dummy.next;
}

ListNode* sortList(ListNode* head) {
    if (!head || !head->next) return head;

    // 快慢指针找中点
    ListNode* slow = head;
    ListNode* fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    ListNode* mid = slow->next;
    slow->next = nullptr;   // 切断

    return mergeTwoLists(sortList(head), sortList(mid));
}
```

### 15.1.8 利用归并排序计算逆序数（进阶）

在 Chapter 14 我们提到逆序数可以 $O(n \log n)$ 计算。关键洞察：

> **在 MERGE 过程中**，当我们把 `right[j]` 放入结果时（即 `right[j] < left[i]`），说明 `left[i], left[i+1], ..., left[end]` 都比 `right[j]` 大，且它们在原数组中都在 `right[j]` 之前，因此这一次操作产生了 `len(left) - i` 个逆序对。

累加所有这样的事件，就得到总逆序数。Python 实现见 15.1.3 的 `count_inversions` 函数。这也是 LeetCode #315 和 #493 的核心。

---

## 15.2 快速排序（Quicksort）

### 15.2.1 算法思想：选基准，一次划分到位

快速排序的思想与归并排序的"分治"相似，但工作的重点不同：

- **归并排序**：倾向在"合"上努力（MERGE 是核心），"分"只是简单对半劈
- **快速排序**：倾向在"分"上努力（PARTITION 是核心），"合"不需要任何工作

**核心流程**：
1. 从数组中选一个**基准元素**（Pivot）
2. **划分**（PARTITION）：把所有小于 pivot 的元素移到左边，大于 pivot 的移到右边——pivot 到达它在最终排序结果中的**正确位置**
3. 递归地对 pivot 左边和右边的子数组重复

**直觉比喻**：想象你在整理一排书，拿起任意一本书（pivot），把比它薄的书都移到左边，比它厚的书都移到右边。这本书现在在正确位置了，再对两边递归处理。

### 15.2.2 Lomuto PARTITION 方案

Lomuto 划分以**最后一个元素**为 pivot，用单向扫描实现，代码直观但效率略低于 Hoare：

```
变量含义：
  pivot = arr[r]（最后一元素）
  i = "小于区域"的右边界（含），初始为 l-1
  j = 当前扫描指针，从 l 到 r-1

不变式：
  arr[l..i]   ≤ pivot（已确定在左边的元素）
  arr[i+1..j-1] > pivot（已确定在右边的元素）
  arr[j..r-1] 未处理

每步：若 arr[j] <= pivot，i++，交换 arr[i] 和 arr[j]
最后：交换 arr[i+1] 和 arr[r]（把 pivot 放到正确位置）
返回：i+1（pivot 的最终位置）
```

```python
import random

def _lomuto_partition(arr: list[int], l: int, r: int) -> int:
    """
    Lomuto 划分方案：以 arr[r] 为 pivot。
    返回 pivot 的最终下标 p，满足：
      arr[l..p-1] <= arr[p] <= arr[p+1..r]
    时间：Θ(r - l + 1)，空间：O(1)（原地）
    
    ⚠️ 注意事项：
    - 完全相同输入时退化为 O(n^2)（所有元素相等，每次 i 到达 r-1）
    - 逆序/有序输入时 pivot 永远是最大/最小值，同样退化
    → 实际使用时配合随机化（见 15.2.4）
    """
    pivot = arr[r]
    i = l - 1   # 小于等于区域右边界，初始为"空"

    for j in range(l, r):     # j 扫描 arr[l..r-1]，跳过 pivot 本身
        if arr[j] <= pivot:   # 当前元素属于左侧
            i += 1
            arr[i], arr[j] = arr[j], arr[i]   # 扩展左侧区域

    # 把 pivot 放到正确位置（左右分界处）
    arr[i + 1], arr[r] = arr[r], arr[i + 1]
    return i + 1   # 返回 pivot 的下标


def quicksort_lomuto(arr: list[int], l: int = 0, r: int = None) -> None:
    """
    快速排序（Lomuto + 随机化 pivot）
    期望时间：O(n log n)，最坏 O(n^2)（极罕见，随机化后）
    空间：O(log n) 期望递归栈（最坏 O(n)）
    稳定性：不稳定（交换操作破坏相对顺序）
    """
    if r is None:
        r = len(arr) - 1
    if l >= r:
        return

    # 随机化：把随机位置的元素交换到末尾作 pivot（防止有序输入退化）
    rand_idx = random.randint(l, r)
    arr[rand_idx], arr[r] = arr[r], arr[rand_idx]

    p = _lomuto_partition(arr, l, r)   # 划分，p 是 pivot 最终位置
    quicksort_lomuto(arr, l, p - 1)    # 递归左半
    quicksort_lomuto(arr, p + 1, r)    # 递归右半


# === 使用示例 ===
arr = [3, 6, 8, 10, 1, 2, 1]
quicksort_lomuto(arr)
print(arr)  # [1, 1, 2, 3, 6, 8, 10]
```

```cpp
#include <vector>
#include <algorithm>
#include <cstdlib>
using namespace std;

int lomuto_partition(vector<int>& arr, int l, int r) {
    int pivot = arr[r];
    int i = l - 1;

    for (int j = l; j < r; j++) {
        if (arr[j] <= pivot) {
            swap(arr[++i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[r]);   // pivot 归位
    return i + 1;
}

void quicksort_lomuto(vector<int>& arr, int l, int r) {
    if (l >= r) return;

    // 随机化 pivot：将随机下标的元素换到末尾
    int rand_idx = l + rand() % (r - l + 1);
    swap(arr[rand_idx], arr[r]);

    int p = lomuto_partition(arr, l, r);
    quicksort_lomuto(arr, l, p - 1);
    quicksort_lomuto(arr, p + 1, r);
}
```

### 15.2.3 Hoare PARTITION 方案

Hoare 的原始方案（1962）使用**双向扫描**，效率更高（平均交换次数约为 Lomuto 的 1/3）：

```
思路：
  pivot = arr[l]（第一个元素）
  i 从左向右扫，找到 arr[i] >= pivot 停下
  j 从右向左扫，找到 arr[j] <= pivot 停下
  交换 arr[i] 和 arr[j]，继续扫
  直到 i >= j，此时 arr[l..j] <= pivot <= arr[j+1..r]
  
⚠️ 注意：Hoare 返回的是 j（不是 pivot 的最终位置！）
  pivot 不一定在 j 位置，它仍在左半某处。
  因此递归时是 quicksort(l, j) 和 quicksort(j+1, r)
```

```python
def _hoare_partition(arr: list[int], l: int, r: int) -> int:
    """
    Hoare 划分方案：以 arr[l] 为 pivot（配合随机化使用）。
    返回 j，满足：arr[l..j] 的所有元素 <= arr[j+1..r] 的所有元素。
    注意：arr[j] 不一定就是 pivot，pivot 在 arr[l..j] 的某处。
    
    相比 Lomuto，每次划分的交换次数更少（平均少 2/3）。
    """
    pivot = arr[l]
    i = l - 1
    j = r + 1

    while True:
        # i 向右找第一个 >= pivot 的元素
        i += 1
        while arr[i] < pivot:
            i += 1

        # j 向左找第一个 <= pivot 的元素
        j -= 1
        while arr[j] > pivot:
            j -= 1

        if i >= j:
            return j   # 两指针相遇，划分完成

        arr[i], arr[j] = arr[j], arr[i]   # 交换


def quicksort_hoare(arr: list[int], l: int = 0, r: int = None) -> None:
    """Hoare 划分的快速排序（带随机化）"""
    if r is None:
        r = len(arr) - 1
    if l >= r:
        return

    # 随机化：把随机元素移到 l 位置作为 pivot
    rand_idx = random.randint(l, r)
    arr[rand_idx], arr[l] = arr[l], arr[rand_idx]

    j = _hoare_partition(arr, l, r)
    quicksort_hoare(arr, l, j)       # 注意：左半是 l..j（含 j）
    quicksort_hoare(arr, j + 1, r)   # 右半是 j+1..r
```

```cpp
int hoare_partition(vector<int>& arr, int l, int r) {
    int pivot = arr[l];
    int i = l - 1, j = r + 1;

    while (true) {
        do { i++; } while (arr[i] < pivot);
        do { j--; } while (arr[j] > pivot);
        if (i >= j) return j;
        swap(arr[i], arr[j]);
    }
}

void quicksort_hoare(vector<int>& arr, int l, int r) {
    if (l >= r) return;

    // 随机化
    int rand_idx = l + rand() % (r - l + 1);
    swap(arr[rand_idx], arr[l]);

    int j = hoare_partition(arr, l, r);
    quicksort_hoare(arr, l, j);
    quicksort_hoare(arr, j + 1, r);
}
```

<div data-component="QuicksortPartitionViz"></div>

### 15.2.4 最坏情况与随机化

**最坏情况 $\Theta(n^2)$**：

如果每次选的 pivot 都是当前子数组的**最大值或最小值**，划分极度不均衡：

```
输入 [1, 2, 3, 4, 5]，静态选最后元素为 pivot：
第 1 轮：pivot=5，左半 [1,2,3,4]，右半 []     → 规模减少 1
第 2 轮：pivot=4，左半 [1,2,3]，右半 []        → 规模减少 1
...
共 n-1 轮，第 k 轮处理 n-k+1 个元素，PARTITION 代价 Θ(n-k+1)
总代价 = Θ(n) + Θ(n-1) + ... + Θ(1) = Θ(n²)
```

**排序好的数组** ↔ **逆序数组** ↔ **所有元素相同** 都是退化输入！

**随机化解决退化**：

在 PARTITION 之前，随机从 `[l, r]` 中选一个下标，将该元素与 `arr[r]`（Lomuto）或 `arr[l]`（Hoare）互换，再进行划分。

随机化后，**没有任何固定输入**能使快速排序退化——因为对手不知道我们会选哪个 pivot。我们的"坏运气"来自纯随机，概率极低。

### 15.2.5 期望 $O(n \log n)$ 的证明思路

**指示随机变量法**（CLRS §7.4）：

设 $X_{ij}$ 为指示随机变量：当排序过程中第 $i$ 小和第 $j$ 小的元素被比较时 $X_{ij} = 1$，否则为 0。

**关键结论**：第 $i$ 小和第 $j$ 小（$i < j$）被比较，当且仅当在排名 $i, i+1, \ldots, j$ 的元素中，$i$ 或 $j$ 先被选为 pivot（共 $j - i + 1$ 个候选，恰有 2 个满足条件）：

$$\Pr[X_{ij} = 1] = \frac{2}{j - i + 1}$$

总期望比较次数：

$$E\!\left[\sum_{i<j} X_{ij}\right] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \frac{2}{j-i+1} = 2\sum_{i=1}^{n-1}\sum_{k=2}^{n-i+1}\frac{1}{k} \leq 2n\sum_{k=1}^{n}\frac{1}{k} = 2nH_n$$

其中 $H_n = \ln n + O(1)$ 是调和级数，因此期望比较次数为 $2n\ln n \approx 1.386 \cdot n\log n$。

**结论**：随机快速排序的期望时间复杂度为 $\Theta(n \log n)$。

### 15.2.6 三路划分（Dutch National Flag）——处理大量重复元素

当输入中有大量**相同元素**时，标准快排退化（所有与 pivot 相等的元素都被分到同一侧）。例如对 `[3, 3, 3, 3, 3]` 排序，每次 Lomuto 划分都把前 $n-1$ 个 3 分到左侧，pivot 放到最后，下一轮仍然要处理前 $n-1$ 个—— $O(n^2)$。

**三路划分**（Three-Way Partition / Dutch National Flag）：

将数组分为三个区域：小于 pivot / 等于 pivot / 大于 pivot。下一轮只需递归"小于"和"大于"两部分，"等于"部分跳过。

```python
def three_way_partition(arr: list[int], l: int, r: int) -> tuple[int, int]:
    """
    三路划分（Dijkstra's Dutch National Flag 算法）。
    
    定义三个区域，用 lt, gt, i 三个指针维护：
      arr[l..lt-1]  < pivot   （小于区，已处理）
      arr[lt..i-1] == pivot   （等于区，已处理）
      arr[i..gt]   未处理    （待扫描）
      arr[gt+1..r]  > pivot   （大于区，已处理）
    
    初始：lt = l, i = l, gt = r
    终止：i > gt，三个区域全部就位
    
    返回：(lt, gt)，下次递归对 [l, lt-1] 和 [gt+1, r] 处理
    时间：Θ(r - l + 1)
    """
    pivot = arr[l]
    lt = l      # arr[l..lt-1] < pivot
    gt = r      # arr[gt+1..r] > pivot
    i  = l      # 当前扫描指针

    while i <= gt:
        if arr[i] < pivot:
            # arr[i] 属于左侧（< pivot），与 lt 交换，lt 和 i 都右移
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
            i  += 1
        elif arr[i] > pivot:
            # arr[i] 属于右侧（> pivot），与 gt 交换，gt 左移（i 不动，因为换来的未知）
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            # arr[i] == pivot，属于中间区，i 右移
            i += 1

    return lt, gt   # arr[lt..gt] 全部 == pivot


def quicksort_3way(arr: list[int], l: int = 0, r: int = None) -> None:
    """
    三路快速排序（对大量重复元素极高效）
    最好情况：O(n)（所有元素相同时，一次 3-way partition 就结束）
    期望：O(n log n)（随机输入）
    空间：O(log n) 期望递归栈
    """
    if r is None:
        r = len(arr) - 1
    if l >= r:
        return

    # 随机选 pivot
    rand_idx = random.randint(l, r)
    arr[rand_idx], arr[l] = arr[l], arr[rand_idx]

    lt, gt = three_way_partition(arr, l, r)
    quicksort_3way(arr, l, lt - 1)   # 递归 < pivot 部分
    quicksort_3way(arr, gt + 1, r)   # 递归 > pivot 部分
    # arr[lt..gt] == pivot，已在正确位置，无需递归！


# === 使用示例 ===
arr = [3, 6, 8, 10, 1, 2, 1, 3, 3, 6]
quicksort_3way(arr)
print(arr)  # [1, 1, 2, 3, 3, 3, 6, 6, 8, 10]

# LeetCode #75 Sort Colors（三路划分经典应用）
def sort_colors(nums: list[int]) -> None:
    """
    对只含 0, 1, 2 的数组排序，O(n) 时间，O(1) 空间（单趟扫描）。
    本质是三路划分以 1 为 pivot 的特例。
    """
    lt, i, gt = 0, 0, len(nums) - 1
    while i <= gt:
        if   nums[i] == 0: nums[i], nums[lt] = nums[lt], nums[i]; lt += 1; i += 1
        elif nums[i] == 2: nums[i], nums[gt] = nums[gt], nums[i]; gt -= 1
        else:              i += 1

# 测试
colors = [2, 0, 2, 1, 1, 0]
sort_colors(colors)
print(colors)  # [0, 0, 1, 1, 2, 2]
```

```cpp
pair<int,int> three_way_partition(vector<int>& arr, int l, int r) {
    int pivot = arr[l];
    int lt = l, gt = r, i = l;

    while (i <= gt) {
        if (arr[i] < pivot) {
            swap(arr[i++], arr[lt++]);
        } else if (arr[i] > pivot) {
            swap(arr[i], arr[gt--]);   // i 不移动！
        } else {
            i++;
        }
    }
    return {lt, gt};
}

void quicksort_3way(vector<int>& arr, int l, int r) {
    if (l >= r) return;

    // 随机化
    int ri = l + rand() % (r - l + 1);
    swap(arr[ri], arr[l]);

    auto [lt, gt] = three_way_partition(arr, l, r);
    quicksort_3way(arr, l, lt - 1);
    quicksort_3way(arr, gt + 1, r);
}

// LeetCode #75 Sort Colors
void sortColors(vector<int>& nums) {
    int lt = 0, i = 0, gt = nums.size() - 1;
    while (i <= gt) {
        if      (nums[i] == 0) swap(nums[i++], nums[lt++]);
        else if (nums[i] == 2) swap(nums[i], nums[gt--]);
        else                   i++;
    }
}
```

### 15.2.7 为什么实践中选 Quicksort？

与归并排序和堆排序相比，快速排序在实践中通常最快，原因有三：

1. **缓存命中率高**：快排的访问模式几乎总是连续的（扫描相邻区间），而堆排序的访问模式是跳跃的（访问 $i, 2i, 2i+1, \ldots$），cache miss 率高得多

2. **常数因子小**：内层循环每步只需一次比较+一次条件跳转（Hoare 方案），而归并需要额外的辅助数组分配和写入

3. **原地排序**：$O(\log n)$ 栈空间远小于归并的 $O(n)$，对大型数组内存压力小

> **实际的 std::sort**（Introsort）结合了三者：默认用快速排序，递归深度超过 $2\log n$ 时切换为堆排序（防止最坏退化），子数组较小时切换为插入排序。

---

## 15.3 堆排序回顾与对比

### 15.3.1 堆排序简要回顾

堆排序在 Chapter 9 已详细讲过，这里仅回顾其关键特性以便与快排/归并对比：

1. **建堆**（BUILD-MAX-HEAP）：$O(n)$（紧界证明：$\sum_{h=0}^{\lfloor \log n \rfloor} \lceil n/2^{h+1} \rceil \cdot O(h) = O(n)$）
2. **n-1 次取最大值**：每次把堆顶（最大值）与末尾交换，再 HEAPIFY，$O(\log n)$ 一次，共 $O(n \log n)$
3. **总时间**：$\Theta(n \log n)$，**无论输入如何**（与归并相同，优于快排的最坏情况）
4. **空间**：$O(1)$（原地排序，优于归并的 $O(n)$）
5. **稳定性**：**不稳定**（取堆顶与末尾的交换会改变相对顺序）

### 15.3.2 堆排序比快排慢的原因

尽管时间复杂度相同，堆排序在实践中**始终慢于快速排序**，主要原因是**缓存局部性极差**：

```
n = 16 的最大堆：
层 0（根）：arr[1]
层 1：arr[2], arr[3]
层 2：arr[4], arr[5], arr[6], arr[7]
层 3：arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]

HEAPIFY 时，从 arr[i] 跳到 arr[2i] 或 arr[2i+1]。
当 i = 1（根），访问 arr[2] 和 arr[3]  → 相对连续
当 i = 8（倒数第二层），访问 arr[16] 和 arr[17] → 跳跃巨大！

对大型数组，这些跳跃几乎每次都 cache miss。
```

**实验数据**（典型场景，$n = 10^7$，仅供参考）：

| 算法 | 缓存命中率 | 实际耗时 |
|---|---|---|
| 快速排序 | ~90% | 1× |
| 归并排序 | ~85% | 1.2× |
| 堆排序 | ~40% | 2.5× |

---

## 15.4 基于比较排序的下界——$\Omega(n \log n)$

### 15.4.1 决策树模型

**核心思想**：任何基于比较的排序算法都可以被抽象为一棵**决策树**（Decision Tree）：

- **内部节点**：一次比较操作 $a_i : a_j$（"$a_i$ 与 $a_j$ 哪个更小？"）
- **边**：比较的两种结果（$a_i \leq a_j$ → 左子树，$a_i > a_j$ → 右子树）
- **叶节点**：确定的一种排列顺序（$a_{\pi(1)} \leq a_{\pi(2)} \leq \cdots \leq a_{\pi(n)}$）

对 $n = 3$（元素 $a_1, a_2, a_3$）的插入排序决策树：

```
                     a₁ : a₂
                   ≤/        \>
             a₂ : a₃         a₁ : a₃
           ≤/     \>        ≤/      \>
       a₁ : a₃  [a₂,a₃,a₁] a₂:a₃  [a₁,a₃,a₂]
      ≤/    \>             ≤/   \>
  [a₁,a₂,a₃] [a₁,a₃,a₂] [a₂,a₁,a₃] [a₃,a₁,a₂]  ← wait, wrong

（实际树形因算法而异，共 6 个叶子对应 3! = 6 种排列）
```

**关键约束**：
1. 正确的排序算法**一定能区分所有 $n!$ 种输入排列**，因此决策树至少有 $n!$ 个叶节点
2. 算法在某输入上的比较次数 = 决策树中对应**从根到叶子的路径长度**
3. 最坏情况比较次数 = 决策树的**高度**（最长路径）

### 15.4.2 下界推导

**定理**：任何基于比较的排序算法在最坏情况下至少需要 $\Omega(n \log n)$ 次比较。

**证明**：

设决策树高度为 $h$。高度为 $h$ 的二叉树至多有 $2^h$ 个叶子节点。

由于决策树至少需要 $n!$ 个叶子：

$$2^h \geq n! \implies h \geq \log_2(n!)$$

由 **Stirling 近似**：

$$n! \approx \sqrt{2\pi n} \cdot \left(\frac{n}{e}\right)^n \implies \log_2(n!) = \sum_{k=1}^n \log_2 k \geq \sum_{k=\lceil n/2 \rceil}^n \log_2 k \geq \frac{n}{2} \log_2 \frac{n}{2} = \Omega(n \log n)$$

精确地，$\log_2(n!) = n\log_2 n - n\log_2 e + O(\log n) = \Omega(n \log n)$。

因此 $h = \Omega(n \log n)$，即最坏情况的比较次数至少为 $\Omega(n \log n)$。□

<div data-component="SortDecisionTree"></div>

### 15.4.3 决策树下界的深层含义

**推论 1**：归并排序和堆排序是**渐进最优**的基于比较的排序——它们的 $\Theta(n \log n)$ 与下界匹配。

**推论 2**：排序算法的"优化"空间在于**常数因子**（实现细节、缓存友好性），而非渐进复杂度。

**推论 3**：线性时间排序（下一节）必须**不基于元素间两两比较**，而需要假设输入具有额外结构。

---

## 15.5 线性时间排序——突破下界

上一节证明了基于比较的排序下界是 $\Omega(n \log n)$。那么 $O(n)$ 排序如何可能？

**秘密**：线性时间排序**不使用元素间的两两比较**，而是利用输入数据的**额外结构**（如键值范围有限、输入均匀分布等）来提取信息，绕开了决策树下界。

### 15.5.1 计数排序（Counting Sort）

**适用条件**：键值为整数，范围在 $[0, k]$ 内，$k = O(n)$。

**思想**：不比较元素，而是**数**每个值出现了多少次，然后根据计数**直接重建**有序数组。

**三阶段流程**：
1. **Count**（计数）：遍历输入，`count[v]++`
2. **Prefix Sum**（前缀和）：`count[i] += count[i-1]`，令 `count[v]` 等于值 $\leq v$ 的元素总数
3. **Place**（放置）：**从右向左**遍历输入，把每个元素放到 `count[v]-1` 的位置，然后 `count[v]--`

**为什么从右向左**：保证稳定性。相同键值的元素按原始顺序从右到左依次放置，末尾的先放，先进先出形成左侧排先。

```python
def counting_sort(arr: list[int], k: int) -> list[int]:
    """
    计数排序。
    时间：Θ(n + k)
    空间：Θ(n + k)（count 数组 + 输出数组）
    稳定性：稳定（从右向左放置，保持等值元素的原始顺序）
    限制：键值必须是 0..k 的整数

    参数：
        arr: 输入数组，元素在 [0, k] 之间
        k:   最大键值（包含）
    返回：
        排好序的新数组
    """
    n = len(arr)
    count  = [0] * (k + 1)   # count[v] = 值 v 出现的次数
    output = [0] * n          # 输出数组

    # 阶段 1：计数
    for v in arr:
        count[v] += 1

    # 阶段 2：前缀和
    # 结束后 count[v] = arr 中值 <= v 的元素总数
    # = 值 v 的元素在输出中最右一位的下标 + 1
    for i in range(1, k + 1):
        count[i] += count[i - 1]

    # 阶段 3：从右向左放置（保证稳定性！）
    # 若从左向右，相同键值的后出现者会覆盖先出现者的位置，破坏稳定性
    for i in range(n - 1, -1, -1):
        v = arr[i]
        output[count[v] - 1] = arr[i]   # 放到 count[v]-1 位置
        count[v] -= 1                    # 该值的"槽"用掉一个

    return output


# === 使用示例 ===
arr = [2, 5, 3, 0, 2, 3, 0, 3]
k = 5
print(counting_sort(arr, k))   # [0, 0, 2, 2, 3, 3, 3, 5]

# 稳定性验证：对带附加信息的数据排序
data = [(2, 'b'), (5, 'e'), (2, 'a'), (0, 'd')]
# 按键值计数排序，相同键值的 'b' 和 'a'，'b' 先出现，结果应保持 'b' 在 'a' 前
# （从右向左放置时，'a' 先放，'b' 后放，因此 'b' 在左（较小下标）→ 'b' 在前）
```

```cpp
vector<int> counting_sort(const vector<int>& arr, int k) {
    int n = arr.size();
    vector<int> count(k + 1, 0), output(n);

    // 阶段 1：计数
    for (int v : arr) count[v]++;

    // 阶段 2：前缀和
    for (int i = 1; i <= k; i++) count[i] += count[i-1];

    // 阶段 3：从右向左放置（保稳定性）
    for (int i = n - 1; i >= 0; i--) {
        int v = arr[i];
        output[--count[v]] = arr[i];   // count[v]-- 后放置
    }
    return output;
}
```

**复杂度分析**：

- 阶段 1：$O(n)$（遍历 $n$ 个元素）
- 阶段 2：$O(k)$（遍历 $k+1$ 个计数格）
- 阶段 3：$O(n)$
- 总时间：$\Theta(n + k)$，当 $k = O(n)$ 时为 $\Theta(n)$

**实际应用**：
- 年龄排序（$k \leq 150$）
- 基数排序中对单个"位"排序（$k = 9$ 或 $k = 255$）
- 哈希桶分配

### 15.5.2 基数排序（Radix Sort）

**核心问题**：若键值范围 $k$ 很大（如 32 位整数，$k = 2^{32}$），计数排序的 $\Theta(n + k)$ 退化为 $\Theta(2^{32})$——完全不可用。

**基数排序的思路**：把每个键值视为 $d$ 位数（可以是十进制位、十六进制位、字节），从**最低有效位**（Least Significant Digit，LSD）到最高位，依次对每位做一次**稳定排序**（通常用计数排序）。

**为什么从低位到高位（而非高位到低位）？**

如果从高位开始（MSD），需要维护多个独立的"桶"，实现复杂。从低位开始（LSD），每轮稳定排序后，低位的顺序会被后续轮次保持——这正是稳定性的作用！

**图解**（对 3 个 3 位数排序）：

```
原始：[329, 457, 657, 839, 436, 720, 355]

按个位（b=0）稳定排序：
  0 → 720
  5 → 355
  6 → 436
  7 → 457, 657
  9 → 329, 839
结果：[720, 355, 436, 457, 657, 329, 839]

按十位（b=1）稳定排序：
  2 → 720, 329（329 的十位是 2？329→ 3是百位，2是十位，9是个位，对）
  3 → 436（十位是 3）
  5 → 355, 457, 657（十位是 5）
  3 → 839（十位是 3，排在 436 之后）
  ...
结果：[720, 329, 436, 839, 355, 457, 657]

按百位（b=2）稳定排序：
  3 → 329, 355
  4 → 436, 457
  6 → 657
  7 → 720
  8 → 839
结果：[329, 355, 436, 457, 657, 720, 839]  ✓
```

```python
def radix_sort(arr: list[int], base: int = 10) -> list[int]:
    """
    基数排序（LSD，从最低位到最高位）。
    时间：O(d × (n + base))，其中 d 是最大数的位数
    空间：O(n + base)（计数排序的辅助空间）
    稳定性：稳定（依赖内部计数排序的稳定性）
    限制：键值为非负整数

    参数：
        arr:  输入数组（非负整数）
        base: 基数，默认 10（十进制位）；可用 256 表示按字节处理
    返回：
        排好序的新数组
    """
    if not arr:
        return arr[:]

    max_val = max(arr)
    result = arr[:]

    exp = 1   # 当前处理的位（个位 exp=1，十位 exp=10，...）
    while max_val // exp > 0:
        result = _counting_sort_by_digit(result, exp, base)
        exp *= base

    return result


def _counting_sort_by_digit(arr: list[int], exp: int, base: int) -> list[int]:
    """
    以 arr[i] // exp % base 为键，对 arr 做计数排序。
    这是基数排序每轮调用的稳定排序。
    """
    n = len(arr)
    count  = [0] * base
    output = [0] * n

    # 计数
    for v in arr:
        digit = (v // exp) % base
        count[digit] += 1

    # 前缀和
    for i in range(1, base):
        count[i] += count[i - 1]

    # 从右向左放置（保稳定性）
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % base
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1

    return output


# === 使用示例 ===
arr = [170, 45, 75, 90, 802, 24, 2, 66]
print(radix_sort(arr))   # [2, 24, 45, 66, 75, 90, 170, 802]

# 对 32 位整数，用 base=256（按字节），只需 d=4 轮：
# 时间 O(4 × (n + 256)) = O(n)（n >> 256 时）
arr2 = [1000000000, 3, 999, 42, 12345678]
print(radix_sort(arr2, base=256))
```

```cpp
void counting_sort_by_digit(vector<int>& arr, int exp, int base) {
    int n = arr.size();
    vector<int> output(n), count(base, 0);

    for (int v : arr)
        count[(v / exp) % base]++;

    for (int i = 1; i < base; i++)
        count[i] += count[i-1];

    // 从右向左放置（稳定性关键！）
    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % base;
        output[--count[digit]] = arr[i];
    }
    arr = output;
}

void radix_sort(vector<int>& arr, int base = 10) {
    if (arr.empty()) return;
    int max_val = *max_element(arr.begin(), arr.end());

    for (int exp = 1; max_val / exp > 0; exp *= base)
        counting_sort_by_digit(arr, exp, base);
}

/*
int main() {
    vector<int> arr = {170, 45, 75, 90, 802, 24, 2, 66};
    radix_sort(arr);
    for (int x : arr) cout << x << " ";  // 2 24 45 66 75 90 170 802
    return 0;
}
*/
```

**复杂度分析**：

设数组有 $n$ 个元素，最大数有 $d$ 位，每位基数为 $b$：

- 每轮计数排序：$\Theta(n + b)$
- 共 $d$ 轮
- 总时间：$\Theta(d(n + b))$

**如何选择 $b$？** 以 $w$ 位的整数为例，令 $b = n$（用 $n$ 进制），则 $d = w / \log n$，总时间 $\Theta\!\left(\frac{wn}{\log n}\right)$。当 $w = O(\log n)$（键值为 $O(n)$ 量级），时间为 $\Theta(n)$。

### 15.5.3 桶排序（Bucket Sort）

**适用条件**：输入元素服从**均匀分布**，分布在 $[0, 1)$ 上。

**思想**：把 $[0, 1)$ 均匀划分为 $n$ 个**桶**（buckets），每个桶对应 $[i/n, (i+1)/n)$。将每个元素放入对应的桶，对每个桶内部排序（小桶用插入排序），最后拼接。

**为什么期望 $O(n)$？** 均匀分布保证每个桶期望包含 $O(1)$ 个元素，桶内排序期望 $O(1)$，$n$ 个桶合计 $O(n)$。

```python
def bucket_sort(arr: list[float]) -> list[float]:
    """
    桶排序。
    时间：期望 O(n)（均匀分布输入），最坏 O(n²)（所有元素落入同一桶）
    空间：O(n)（桶数组）
    稳定性：取决于桶内排序（用插入排序时稳定）
    限制：要求输入均匀分布在 [0, 1)（实际可归一化到此范围）

    参数：
        arr: 浮点数数组，元素在 [0, 1) 内
    返回：
        排好序的新数组
    """
    n = len(arr)
    if n == 0:
        return []

    # 创建 n 个空桶
    buckets: list[list[float]] = [[] for _ in range(n)]

    # 将每个元素放入对应桶
    for v in arr:
        bucket_idx = int(v * n)                   # 均匀映射到 [0, n-1]
        bucket_idx = min(bucket_idx, n - 1)       # 处理 v == 1.0 的边界
        buckets[bucket_idx].append(v)

    # 每个桶内部排序（少量元素，插入排序最优）
    for bucket in buckets:
        bucket.sort()   # Python 内置 TimSort，也可用插入排序

    # 拼接所有桶
    result = []
    for bucket in buckets:
        result.extend(bucket)
    return result


# === 使用示例 ===
import random
arr = [random.random() for _ in range(10)]
print(sorted(arr))          # 参考答案
print(bucket_sort(arr))     # 应一致

# 非 [0,1) 数据：先归一化
def bucket_sort_general(arr: list[float]) -> list[float]:
    """对任意浮点数组，先线性归一化再桶排序。"""
    if not arr:
        return []
    lo, hi = min(arr), max(arr)
    if lo == hi:
        return arr[:]
    # 线性变换 [lo, hi] → [0, 1)
    normalized = [(v - lo) / (hi - lo) for v in arr]
    sorted_norm = bucket_sort(normalized)
    # 逆变换
    return [v * (hi - lo) + lo for v in sorted_norm]
```

```cpp
void bucket_sort(vector<double>& arr) {
    int n = arr.size();
    if (n == 0) return;

    vector<vector<double>> buckets(n);

    for (double v : arr) {
        int idx = min((int)(v * n), n - 1);
        buckets[idx].push_back(v);
    }

    // 每个桶内插入排序（小规模最优）
    for (auto& b : buckets) {
        // 插入排序
        for (int i = 1; i < (int)b.size(); i++) {
            double key = b[i];
            int j = i - 1;
            while (j >= 0 && b[j] > key) { b[j+1] = b[j]; j--; }
            b[j+1] = key;
        }
    }

    int k = 0;
    for (auto& b : buckets)
        for (double v : b)
            arr[k++] = v;
}
```

**期望时间复杂度证明**（CLRS §8.4）：

设 $n_i$ 为第 $i$ 个桶中的元素数，桶内插入排序代价为 $O(n_i^2)$，总运行时间：

$$T(n) = \Theta(n) + \sum_{i=0}^{n-1} O(n_i^2)$$

由均匀分布，$E[n_i] = 1$，$E[n_i^2] = 1 + \frac{1}{n} \cdot \frac{n-1}{n} \cdot n = 2 - \frac{1}{n}$（精确计算）。

$$E[T(n)] = \Theta(n) + n \cdot O\!\left(2 - \frac{1}{n}\right) = \Theta(n)$$

### 15.5.4 三种线性时间排序适用条件对比

| 特性 | 计数排序 | 基数排序 | 桶排序 |
|---|---|---|---|
| **时间** | $\Theta(n+k)$ | $\Theta(d(n+b))$ | 期望 $O(n)$ |
| **空间** | $\Theta(n+k)$ | $\Theta(n+b)$ | $\Theta(n)$ |
| **稳定性** | ✅ 稳定 | ✅ 稳定 | ✅ 稳定（插入排序） |
| **键值要求** | 整数，范围 $[0,k]$ | 整数，可分位 | 均匀分布浮点数 |
| **最坏情况** | $O(n+k)$（无退化） | $O(d(n+b))$ | $O(n^2)$（极端集中） |
| **比较操作** | ❌ 不比较 | ❌ 不比较 | ✅ 桶内比较 |
| **适用场景** | 小整数键（年龄、评分）| 大整数/字符串 | 均匀实数、散列后的键 |

**选择指南**：
- 键值是有界整数，$k = O(n)$ → **计数排序**
- 键值是大整数（如 64 位），可按字节拆分 → **基数排序**
- 键值均匀分布（如浮点概率、散列值）→ **桶排序**
- 无法满足以上任何一条 → 退回比较排序（快排/归并）

<div data-component="LinearSortDemo"></div>

---

## 15.6 TimSort——现实世界的排序冠军

### 15.6.1 什么是 TimSort？

TimSort 由 Tim Peters 于 2002 年为 Python 设计，后被 Java（`Arrays.sort` 对象版）、Android、Rust 的标准库采用。它是一种**归并排序 + 插入排序**的混合算法，核心思想是：

> **现实中的数据往往"几乎有序"**——由若干个已经排好序的**片段**（Run）拼接而成。TimSort 利用这一结构，把排序代价转化为 $O(n \log r)$，其中 $r$ 是片段数。

### 15.6.2 TimSort 的核心机制

1. **检测自然 Run**：从左到右扫描，找出已经升序（或严格降序，则反转）的连续片段。若片段太短（小于 `MIN_RUN`，通常 32~64），用插入排序把它扩充到 `MIN_RUN` 长度。

2. **Run 栈与合并策略**：检测到的 Run 压入栈中。为了平衡合并树（避免某些 Run 被参与太多次合并），维护**两个不变式**：
   - $|Z| > |Y| + |X|$
   - $|Y| > |X|$（$X, Y, Z$ 为栈顶三个 Run 的长度）
   
   违反时立即合并相邻 Run。

3. **最终合并**：输入扫描完毕后，逐一合并栈中剩余 Run。

### 15.6.3 TimSort 的性能优势

| 输入类型 | 归并排序 | 快速排序 | TimSort |
|---|---|---|---|
| 完全随机 | $O(n\log n)$ | $O(n\log n)$ 期望 | $O(n\log n)$ |
| 已排序 | $O(n\log n)$ | $O(n^2)$（未随机化）| $O(n)$ |
| 逆序 | $O(n\log n)$ | $O(n^2)$ | $O(n)$ |
| 若干有序段 | $O(n\log n)$ | $O(n\log n)$ | $O(n\log r)$ |

TimSort 在"几乎有序"数据（$r$ 很小）时可以达到接近 $O(n)$ 的速度，这在实际工程中极为常见（数据库结果集、日志文件、用户操作历史……）。

---

## 15.7 综合对比与算法选择

### 15.7.1 完整对比表

| 算法 | 最好 | 平均 | 最坏 | 空间 | 稳定 | 缓存友好 | 实践评级 |
|---|---|---|---|---|---|---|---|
| 插入排序 | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | ✅ | ⭐⭐⭐ | 小数组首选 |
| 归并排序 | $O(n\log n)$ | $O(n\log n)$ | $O(n\log n)$ | $O(n)$ | ✅ | ⭐⭐ | 链表/稳定首选 |
| 快速排序 | $O(n\log n)$ | $O(n\log n)$ | $O(n^2)$ | $O(\log n)$ | ❌ | ⭐⭐⭐ | 通用首选 |
| 堆排序 | $O(n\log n)$ | $O(n\log n)$ | $O(n\log n)$ | $O(1)$ | ❌ | ⭐ | 严格空间场景 |
| 计数排序 | $O(n+k)$ | $O(n+k)$ | $O(n+k)$ | $O(k)$ | ✅ | ⭐⭐⭐ | 小整数键 |
| 基数排序 | $O(d(n+b))$ | $O(d(n+b))$ | $O(d(n+b))$ | $O(n+b)$ | ✅ | ⭐⭐ | 大整数/字符串 |
| 桶排序 | $O(n)$ | $O(n)$ | $O(n^2)$ | $O(n)$ | ✅ | ⭐⭐ | 均匀实数 |
| TimSort | $O(n)$ | $O(n\log n)$ | $O(n\log n)$ | $O(n)$ | ✅ | ⭐⭐⭐ | 通用稳定首选 |

<div data-component="SortingComplexityTable"></div>

### 15.7.2 工程决策树

```
需要稳定排序？
  └─ YES → 归并排序 或 TimSort（若用语言标准库）

数据规模很小（n < 20）？
  └─ YES → 插入排序

键值为有界整数（k = O(n)）？
  └─ YES → 计数排序

键值为大整数可按位分解？
  └─ YES → 基数排序

输入均匀分布浮点数？
  └─ YES → 桶排序

空间极度受限（O(1)）且不需稳定？
  └─ YES → 堆排序（最坏 O(n log n) 保证）

通用场景，需要最快实践速度？
  └─ YES → 随机快速排序 / Introsort（std::sort）
```

---

## 15.8 经典问题与面试考点

### 15.8.1 LeetCode 精选

| 题目 | 核心思想 |
|---|---|
| [#912 排序数组](https://leetcode.cn/problems/sort-an-array/) | 练习各排序算法，推荐归并和三路快排 |
| [#75 颜色分类](https://leetcode.cn/problems/sort-colors/) | Dutch National Flag 三路划分经典 |
| [#148 排序链表](https://leetcode.cn/problems/sort-list/) | 链表归并排序，快慢指针找中点 |
| [#315 计算右侧小于当前元素的个数](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/) | 归并排序计逆序数 |
| [#23 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/) | 多路归并（外排序核心思路） |
| [#179 最大数](https://leetcode.cn/problems/largest-number/) | 自定义比较函数的排序 |

### 15.8.2 LeetCode #148 完整解法

见 15.1.7 节的链表归并排序实现，以下补充迭代（$O(1)$ 空间）版本：

```python
def sort_list_iterative(head: ListNode) -> ListNode:
    """
    链表归并排序——自底向上迭代版，空间 O(1)（无递归栈）。
    
    思路：从步长 size=1 开始，每轮把链表中相邻的 size 长段两两合并，
    size 翻倍，直到 size >= 链表长度。
    """
    # 先求链表长度
    n, cur = 0, head
    while cur:
        n += 1
        cur = cur.next

    dummy = ListNode(0)
    dummy.next = head

    size = 1
    while size < n:
        prev, cur = dummy, dummy.next
        while cur:
            # 取出左半（size 个节点）
            left = cur
            right = _split(left, size)   # right 是右半起点，left 截止
            cur   = _split(right, size)  # cur 是下一组的起点，right 截止

            # 合并 left 和 right，接到 prev 后面
            merged_head, merged_tail = _merge_with_tail(left, right)
            prev.next = merged_head
            prev = merged_tail

        size *= 2

    return dummy.next


def _split(head: ListNode, k: int) -> ListNode:
    """从 head 开始走 k 步，截断后返回后半起点。"""
    for _ in range(k - 1):
        if head and head.next:
            head = head.next
        else:
            break
    if not head:
        return None
    rest = head.next
    head.next = None   # 截断
    return rest


def _merge_with_tail(l1: ListNode, l2: ListNode) -> tuple:
    """合并两链表，返回 (头节点, 尾节点)。"""
    dummy = ListNode(0)
    cur = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1; l1 = l1.next
        else:
            cur.next = l2; l2 = l2.next
        cur = cur.next
    cur.next = l1 if l1 else l2
    while cur.next:
        cur = cur.next   # 找尾节点
    return dummy.next, cur
```

### 15.8.3 LeetCode #315 用归并计算逆序数

```python
def count_smaller(nums: list[int]) -> list[int]:
    """
    LeetCode #315：计算每个数右侧有多少个比它小的数。
    
    思路：将 (值, 原始下标) 对归并排序，在 MERGE 阶段，
    每当右半元素 right[j] 被选中（right[j] < left[i]），
    说明 left[i..end_of_left] 都比 right[j] 大，
    但这是"下标更大"的 right[j] 比"下标更小"的 left[i..] 小
    → 对 right[j] 而言，贡献不是逆序数
    
    正确思路：统计每个 left[i] 右侧有多少 right[j] 比它小：
    当 right[j] 被放入结果时，统计 j 的计数（即右侧有 j 个元素已放入）
    """
    result = [0] * len(nums)
    indexed = list(enumerate(nums))   # [(0, val0), (1, val1), ...]

    def merge_count(pairs):
        if len(pairs) <= 1:
            return pairs
        mid = len(pairs) // 2
        left  = merge_count(pairs[:mid])
        right = merge_count(pairs[mid:])

        merged = []
        i, j = 0, 0
        right_count = 0   # 已从 right 放入 merged 的数量

        while i < len(left) and j < len(right):
            if left[i][1] <= right[j][1]:
                # left[i] 被放入，此时已有 right_count 个 right 元素比它小
                # 但等等——我们要的是 left[i] 右侧（大下标）比它小的数的个数
                # right 的元素下标都 > left 的元素下标
                # 若 right[j][1] < left[i][1]，说明右侧有元素比 left[i] 小
                result[left[i][0]] += right_count
                merged.append(left[i])
                i += 1
            else:
                # right[j][1] < left[i][1]，right_count + 1
                right_count += 1
                merged.append(right[j])
                j += 1

        # 处理 left 中剩余（right 已全放入，剩余 left 每个各加 right_count）
        while i < len(left):
            result[left[i][0]] += right_count
            merged.append(left[i])
            i += 1

        merged.extend(right[j:])
        return merged

    merge_count(indexed)
    return result


# 测试
print(count_smaller([5, 2, 6, 1]))  # [2, 1, 1, 0]
```

### 15.8.4 高频面试 Q&A

**Q1：快排和归并排序哪个更好？**

A：取决于场景：
- **通用场景**：快排更好（缓存友好、原地、常数因子小），工程中是首选
- **需要稳定排序**：归并
- **最坏情况保证**：归并（$\Theta(n\log n)$ 无退化）或堆排序（$O(1)$ 空间）
- **链表**：归并（链表不支持随机访问，快排 pivot 选取麻烦）
- **外排序（磁盘）**：归并（多路归并可直接延伸）

**Q2：为什么决策树下界只适用于"基于比较"的排序？**

A：决策树模型假设每次信息获取来自一次"$a_i$ vs $a_j$"的二元比较。计数/基数/桶排序获取信息的方式是**直接读取键值**（$O(1)$ 可得键值的精确数字），而非两两比较。因此它们不被决策树模型约束，可以突破 $\Omega(n\log n)$ 下界。

**Q3：TimSort 为什么是实践中最好的通用稳定排序？**

A：
1. **利用自然有序性**：现实数据往往包含已排好序的片段（Run），TimSort 检测并直接利用这些 Run，最优情况 $O(n)$
2. **稳定性**：基于归并，天然稳定
3. **精心调优的实现**：Run 最小长度（MIN_RUN）的选择保证归并树平衡，避免某些数据使归并不均衡
4. **兼顾小数组**：用插入排序处理小 Run，避免归并的常数因子开销

**Q4：对 10 亿个 IP 地址排序，如何设计方案？**

A：IP 地址可以视为 32 位无符号整数，范围 $[0, 2^{32})$。
- **方案 1**：基数排序，按字节（8位）分4轮，$O(4(n + 256)) = O(n)$；但 $n = 10^9$ 时内存约需 4GB（每个 int 4字节），需要分块
- **方案 2**：若内存不足，先按最高字节（$0\sim255$）分 256 块写入磁盘，每块再独立排序（计数排序），最后顺序输出——外排序思路
- **方案 3**：位图（Bitmap）：开 $2^{32}$ 位的位图（512MB），对每个 IP 置位，最后顺序扫描位图输出存在的 IP（时间 $O(2^{32} + n)$，不支持重复计数）

---

## 15.9 常见错误与调试

### ⚠️ 错误 1：MERGE 后忘记拷贝剩余元素

```python
# ❌ 错误：只处理了一半结束的情况，另一半剩余被遗忘
while i < len(left) and j < len(right):
    if left[i] <= right[j]: arr[k] = left[i]; i += 1
    else:                    arr[k] = right[j]; j += 1
    k += 1
# 循环结束后，left 或 right 可能还有剩余元素没写入 arr！

# ✅ 正确：循环结束后补全剩余
while i < len(left):  arr[k] = left[i];  i += 1; k += 1
while j < len(right): arr[k] = right[j]; j += 1; k += 1
```

### ⚠️ 错误 2：`(l + r) // 2` 整数溢出

```cpp
// ❌ 在 C++ 中，若 l 和 r 均为大整数，l+r 可能溢出
int mid = (l + r) / 2;

// ✅ 防溢出写法
int mid = l + (r - l) / 2;
```

Python 整数无溢出，但养成习惯，C++ 代码必须用防溢出写法。

### ⚠️ 错误 3：MERGE 稳定性被破坏——用 `<` 而非 `<=`

```python
# ❌ 错误：相等时取右半，破坏稳定性
if left[i] < right[j]:   # 严格小于
    arr[k] = left[i]; i += 1
else:
    arr[k] = right[j]; j += 1   # 相等时取右半！

# ✅ 正确：相等时取左半（保持原始顺序）
if left[i] <= right[j]:   # 小于等于
    arr[k] = left[i]; i += 1
else:
    arr[k] = right[j]; j += 1
```

### ⚠️ 错误 4：三路划分中 `i` 的移动条件

```python
# ❌ 错误：对 arr[i] > pivot 时也移动 i，导致右侧元素未检查
if arr[i] < pivot:   lt += 1; swap; i += 1
elif arr[i] > pivot: gt -= 1; swap; i += 1  # ← i 不该移动！
else:                i += 1

# ✅ 正确：> pivot 时 i 不动（因为从 gt 换来的元素还未检查）
if arr[i] < pivot:   lt += 1; swap; i += 1
elif arr[i] > pivot: gt -= 1; swap          # i 保持不动
else:                i += 1
```

### ⚠️ 错误 5：计数排序从左向右放置——破坏稳定性

```python
# ❌ 错误：从左向右，相同键值的后出现者会放到"先"找到的空位
for i in range(n):   # 从左向右
    output[count[arr[i]] - 1] = arr[i]
    count[arr[i]] -= 1

# ✅ 正确：从右向左，保证相同键值的元素保持原始相对顺序
for i in range(n - 1, -1, -1):   # 从右向左
    output[count[arr[i]] - 1] = arr[i]
    count[arr[i]] -= 1
```

---

## 本章小结

| 知识点 | 核心结论 |
|---|---|
| 归并排序 | 分治，$\Theta(n\log n)$ 最坏，$O(n)$ 辅助空间，稳定；MERGE 等号取左保稳定性 |
| 主定理应用 | $T(n) = 2T(n/2) + \Theta(n) \Rightarrow \Theta(n\log n)$（情形 2） |
| 快速排序 | 期望 $O(n\log n)$，原地，不稳定；随机化 pivot 防退化；三路划分解重复元素 |
| Lomuto vs Hoare | Lomuto 简单但交换多；Hoare 效率高但返回值不是 pivot 位置 |
| 堆排序 | $\Theta(n\log n)$，$O(1)$ 空间，不稳定；缓存命中差，实践慢 |
| 决策树下界 | $n!$ 叶子 → 高度 $\geq \log_2 n! = \Omega(n\log n)$，任何比较排序下界 |
| 计数排序 | $\Theta(n+k)$，从右向左放置保稳定；适合 $k=O(n)$ 整数 |
| 基数排序 | $\Theta(d(n+b))$，LSD 从低位到高位，依赖稳定子排序；突破 $O(n\log n)$ |
| 桶排序 | 期望 $O(n)$（均匀分布），最坏 $O(n^2)$；适合均匀实数 |
| TimSort | 归并 + 插入，利用自然 Run，实际数据接近 $O(n)$，Python/Java 标准库实现 |

**下一章**（Chapter 16）将深入"第 K 小"问题：从朴素的 $O(n\log n)$ 排序，到随机选择的期望 $O(n)$，再到 BFPRT 的最坏 $O(n)$——在不完全排序的前提下，如何找到你想要的那一个？

---

**参考资料**：
- CLRS 第4版 Chapter 2（归并排序）、Chapter 7（快速排序）、Chapter 8（线性时间排序）
- Sedgewick《Algorithms》第4版 2.2–2.3 节
- MIT 6.006 Lecture 7（归并）、Lecture 8（快排、堆排序、线性排序）
- Tim Peters, "ListSort.txt"（TimSort 原始设计文档）
