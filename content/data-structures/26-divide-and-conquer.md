# Chapter 26: 分治算法（Divide and Conquer）

## 章节导读

如果说前面的排序、图、树、并查集是在学习“具体武器”，那么分治算法（Divide and Conquer）更像是在学习一种**通用作战思想**。

它的核心观念并不复杂：

> **把一个大问题，拆成几个更小、结构相同的子问题；先分别解决小问题；再把它们的答案合并成大问题的答案。**

这听起来像什么？

- 整理一大摞试卷：先分成左右两堆，分别排好，再合并。
- 搜索一本厚词典：不是从第一页翻到最后一页，而是不断对半判断。
- 计算一个很大的幂：不是乘很多次，而是先算一半，再平方。
- 处理海量点集：先分别处理左右两半，再只检查边界附近的特殊情况。

这就是分治的魅力：

- **思路自然**：和递归天然契合。
- **结构清晰**：常能把复杂问题拆成统一模板。
- **复杂度可分析**：常写成递推式 $T(n)=aT(n/b)+f(n)$。
- **适合并行**：子问题彼此独立时，可以同时算。

但分治并不是“逢题就拆”。如果你把问题拆开之后，**合并特别昂贵**，或者**子问题之间并不独立、反而大量重叠**，那分治就不一定是好选择。这也是为什么本章后面要专门讲：

- 主定理（Master Theorem）
- Akra-Bazzi 定理
- 分治与动态规划（DP）的边界
- 非平衡分治与工程优化

本章你会看到两类非常典型的现象：

1. **分治把复杂问题“变小”**：如归并排序、最近点对。
2. **分治通过代数变形“减少工作量”**：如 Karatsuba、Strassen。

---

## 26.1 分治三步骤与终止条件

### 26.1.1 分治到底在做什么？先建立直觉

对于初学者来说，分治最容易和“递归”混在一起。

其实两者关系是：

- **递归**是一种实现方式；
- **分治**是一种设计思想。

也就是说：

- 你可以用递归实现分治；
- 但不是所有递归都是分治。

例如：

- 计算斐波那契数列的朴素递归是递归，但不是优秀的分治设计，因为子问题大量重叠；
- 归并排序是递归，而且是非常标准的分治算法，因为两个子数组排序后彼此独立，最后再线性合并。

分治通常包含三个动作：

1. **Divide（分解）**：把原问题拆成更小子问题；
2. **Conquer（解决）**：递归求解子问题；
3. **Combine（合并）**：把子问题答案拼成原问题答案。

可以把它理解成一个团队协作流程：

- 经理把大任务拆成多个小任务；
- 员工分别完成小任务；
- 经理把小任务结果汇总。

如果拆出来的任务大小差不多，而且汇总不太贵，整个过程通常就很高效。

---

### 26.1.2 分解（Divide）：把规模 $n$ 变成若干规模 $n/b$

最常见的分解方式是“对半拆分”：

- 归并排序：拆成左右两半；
- 二分搜索：拆成左半或右半；
- 最近点对：按中线拆成左半平面与右半平面；
- Strassen：把矩阵拆成四个子矩阵。

如果一个规模为 $n$ 的问题，被拆成 $a$ 个规模大约为 $n/b$ 的子问题，那么这一步可以抽象为：

$$
\text{原问题规模 } n \longrightarrow a \text{ 个子问题，每个规模约为 } n/b
$$

这里：

- $a$ 表示**子问题个数**；
- $b$ 表示**每个子问题缩小的比例**。

举几个最典型的例子：

| 算法 | 子问题个数 $a$ | 缩小比例 $b$ | 说明 |
|---|---:|---:|---|
| 归并排序 | 2 | 2 | 拆左右两半 |
| 二分搜索 | 1 | 2 | 只递归进入一半 |
| Karatsuba | 3 | 2 | 两个 $n$ 位数拆为高低两半后只做 3 次乘法 |
| Strassen | 7 | 2 | $2\times2$ 块矩阵只做 7 次子矩阵乘法 |

> **关键点**：分解这一步本身也有成本。
> 有些算法分解很容易（只算一个 mid），有些则要先排序、复制、切片、构造辅助结构。

所以我们不能只看“拆了几个子问题”，还要看“拆的代价是不是太大”。

---

### 26.1.3 解决（Conquer）：递归求解，基元情况直接返回

子问题变小之后，通常继续用**同样的方法**去解决。

这就是递归的本质：

> 大问题的解法，依赖于更小规模的同类问题的解法。

但递归不能无限继续，必须有一个**终止条件（Base Case）**。

常见终止条件：

- 数组长度为 0 或 1：已经天然有序；
- 指数为 0：$a^0=1$；
- 点数很少：直接暴力求解最近点对；
- 数字位数很少：直接做普通乘法。

一个非常标准的分治模板如下：

```python
# Python：分治通用模板

def divide_and_conquer(problem):
    # 1) Base Case：问题足够小，直接求解
    if small_enough(problem):
        return solve_directly(problem)

    # 2) Divide：拆成若干子问题
    subproblems = split(problem)

    # 3) Conquer：递归求解子问题
    subresults = [divide_and_conquer(sub) for sub in subproblems]

    # 4) Combine：合并子问题答案
    return merge(subresults)
```

```cpp
// C++：分治通用模板
Result divideAndConquer(const Problem& problem) {
    // 1) Base Case
    if (smallEnough(problem)) {
        return solveDirectly(problem);
    }

    // 2) Divide
    vector<Problem> subproblems = split(problem);

    // 3) Conquer
    vector<Result> subresults;
    subresults.reserve(subproblems.size());
    for (const auto& sub : subproblems) {
        subresults.push_back(divideAndConquer(sub));
    }

    // 4) Combine
    return merge(subresults);
}
```

> **新手特别容易犯的错**：
> 只会写递归调用，却忘记写“何时停止”。
> 这会导致无限递归，最终栈溢出（Stack Overflow）。

---

### 26.1.4 合并（Combine）：分治是否高效，往往看这一步

很多人学分治时，只记住“拆开递归”，却忽略了最关键的问题：

> **合并步骤到底贵不贵？**

比如：

- 归并排序：拆分几乎不要钱，真正花时间的是合并两个有序数组，代价是 $O(n)$；
- 最近点对：左右分别求完最近距离后，还要检查中间带状区域，但只需要额外 $O(n)$；
- 快速排序：它和归并不一样，它的“合并”几乎不要钱，主要成本在前面的 partition；
- Strassen：合并要做很多矩阵加减法，这部分虽比乘法便宜，但常数不小。

所以一个分治算法的总成本，通常可以拆成两部分：

1. 递归子问题成本；
2. 当前层的非递归成本（分解 + 合并 + 预处理）。

这就自然引出递推式。

---

### 26.1.5 递推式的一般形式：$T(n)=aT(n/b)+f(n)$

当一个问题被拆成：

- $a$ 个子问题；
- 每个子问题规模约为 $n/b$；
- 当前层额外工作量为 $f(n)$；

那么总时间复杂度常写为：

$$
T(n)=aT(n/b)+f(n)
$$

其中：

- $aT(n/b)$：表示所有子问题的总成本；
- $f(n)$：表示本层额外工作量（拆分、合并、扫描、加法等）。

#### 递归树直觉

把这个式子画成一棵递归树：

- 第 0 层：1 个规模 $n$ 的问题，代价 $f(n)$；
- 第 1 层：$a$ 个规模 $n/b$ 的问题，总代价 $a\,f(n/b)$；
- 第 2 层：$a^2$ 个规模 $n/b^2$ 的问题，总代价 $a^2 f(n/b^2)$；
- ……

整棵树总成本就是每层成本之和。

这套思维特别重要，因为它能帮你理解主定理，而不只是死记结论。

#### 常见递推式例子

1. **归并排序**

$$
T(n)=2T(n/2)+O(n)
$$

2. **二分搜索**

$$
T(n)=T(n/2)+O(1)
$$

3. **Karatsuba**

$$
T(n)=3T(n/2)+O(n)
$$

4. **Strassen**

$$
T(n)=7T(n/2)+O(n^2)
$$

这些式子，正是主定理最擅长解决的对象。

---

### 26.1.6 主定理（Master Theorem）：三种情形与应用

主定理解决的是这类标准递推式：

$$
T(n)=aT(n/b)+f(n), \quad a\ge 1,\; b>1
$$

它的核心比较对象是：

$$
n^{\log_b a}
$$

这个量可以理解为：**整棵递归树中“叶子层”的总量级**。

也就是说，主定理在比较：

- 当前层额外工作 $f(n)$
- 与递归展开后的“子问题总骨架量级” $n^{\log_b a}$

到底谁更大。

#### 情形一：$f(n)$ 比 $n^{\log_b a}$ 小很多

如果：

$$
f(n)=O\left(n^{\log_b a-\varepsilon}\right)
$$

其中 $\varepsilon>0$，说明当前层的额外工作比递归子问题总量级小一个多项式级别。

则：

$$
T(n)=\Theta\left(n^{\log_b a}\right)
$$

**例子：二分搜索**

$$
T(n)=T(n/2)+O(1)
$$

这里：

- $a=1, b=2$
- $n^{\log_2 1}=1$
- $f(n)=O(1)$

所以：

$$
T(n)=\Theta(\log n)
$$

> 直觉：每层花费都很小，真正决定成本的是树高。

#### 情形二：$f(n)$ 与 $n^{\log_b a}$ 同阶

如果：

$$
f(n)=\Theta\left(n^{\log_b a} \log^k n\right)
$$

那么：

$$
T(n)=\Theta\left(n^{\log_b a}\log^{k+1} n\right)
$$

最常见的子情况是 $k=0$，即：

$$
f(n)=\Theta(n^{\log_b a})
$$

那么：

$$
T(n)=\Theta(n^{\log_b a}\log n)
$$

**例子：归并排序**

$$
T(n)=2T(n/2)+O(n)
$$

这里：

- $a=2, b=2$
- $n^{\log_2 2}=n$
- $f(n)=O(n)$

所以：

$$
T(n)=\Theta(n\log n)
$$

> 直觉：每一层都花 $O(n)$，总共有 $O(\log n)$ 层。

#### 情形三：$f(n)$ 比 $n^{\log_b a}$ 大很多

如果：

$$
f(n)=\Omega\left(n^{\log_b a+\varepsilon}\right)
$$

并且满足**正则条件（regularity condition）**：

$$
a f(n/b) \le c f(n), \quad c<1
$$

则：

$$
T(n)=\Theta(f(n))
$$

**例子**：

$$
T(n)=2T(n/2)+n^2
$$

这里：

- $n^{\log_2 2}=n$
- $f(n)=n^2$ 明显更大

所以：

$$
T(n)=\Theta(n^2)
$$

> 直觉：合并成本太大，整棵递归树的成本被顶层/高层主导。

#### 一个新手最容易出错的点

很多同学会把：

$$
T(n)=2T(n/2)+n\log n
$$

直接套成“情形二”。

其实这时：

- $n^{\log_2 2}=n$
- $f(n)=n\log n = \Theta(n \log^1 n)$

这确实属于扩展形式的情形二，结论是：

$$
T(n)=\Theta(n\log^2 n)
$$

但如果题目变得更奇怪，比如：

$$
T(n)=T(n/2)+T(n/3)+n
$$

这就不是主定理的标准形式了，要用 Akra-Bazzi。

<div data-component="MasterTheoremCalculator"></div>

#### 主定理速查表

| 情形 | 条件 | 结论 | 直觉 |
|---|---|---|---|
| Case 1 | $f(n)$ 更小 | $\Theta(n^{\log_b a})$ | 子问题主导 |
| Case 2 | $f(n)$ 同阶 | $\Theta(n^{\log_b a}\log n)$ 或扩展版 | 每层差不多 |
| Case 3 | $f(n)$ 更大 | $\Theta(f(n))$ | 合并成本主导 |

---

### 26.1.7 Akra-Bazzi 定理：非均匀划分怎么办？

主定理很好用，但它有一个明显限制：

> 所有子问题必须“长得一样”，都是 $n/b$ 规模。

可现实中很多递推式不是这么整齐的。

例如：

$$
T(n)=T(n/2)+T(n/3)+n
$$

这时子问题规模并不统一，主定理直接失效。

Akra-Bazzi 定理可以处理更一般的形式：

$$
T(x)=\sum_{i=1}^{k} a_i T(b_i x) + g(x)
$$

其中：

- $0<b_i<1$
- $a_i>0$
- $g(x)$ 是当前层非递归成本

核心思想是：先找一个实数 $p$，使得

$$
\sum_{i=1}^{k} a_i b_i^p = 1
$$

然后复杂度近似为：

$$
T(x)=\Theta\left(x^p\left(1+\int_1^x \frac{g(u)}{u^{p+1}}\,du\right)\right)
$$

对于初学阶段，你**不用强行背这个积分形式**，更重要的是理解：

- 主定理适合“均匀拆分”；
- Akra-Bazzi 适合“非均匀拆分”；
- 它本质上仍是在比较“递归骨架规模”和“每层额外工作”。

#### 例子：$T(n)=T(n/2)+T(n/3)+n$

要求解：

$$
(1/2)^p + (1/3)^p = 1
$$

数值上可得 $p<1$，而当前层代价是 $g(n)=n$，因此整体复杂度仍由线性项主导：

$$
T(n)=\Theta(n)
$$

> 这类题在面试中不一定要求你精确套 Akra-Bazzi，但常常考你能否意识到：
> **这已经不能直接用主定理。**

---

## 26.2 经典分治算法

### 26.2.1 归并排序（复习强化）

#### 先用生活直觉理解

想象你要把一大摞扑克牌按数字排序。

一种自然思路是：

1. 先把牌堆一分为二；
2. 左半边排好，右半边排好；
3. 最后像拉拉链一样把两堆有序牌合并。

这就是归并排序（Merge Sort）。

#### 为什么它是分治的“教科书模板”？

因为它把三步体现得特别清楚：

- **Divide**：平分数组；
- **Conquer**：分别排序左右两半；
- **Combine**：线性合并两个有序数组。

递推式：

$$
T(n)=2T(n/2)+O(n)
$$

根据主定理情形二：

$$
T(n)=O(n\log n)
$$

#### 归并排序为什么稳定？

稳定排序的意思是：

> 若两个元素键值相等，排序后它们的相对顺序不变。

归并排序稳定的关键在于合并时：

- 若 `left[i] <= right[j]`，优先取左边；
- 这样可以保持左侧原有次序。

#### Python 实现

```python
from typing import List


def merge_sort(nums: List[int]) -> List[int]:
    """
    归并排序（返回新数组版本）

    时间复杂度: O(n log n)
    空间复杂度: O(n)

    设计说明：
    - 用切片写法更直观，适合教学；
    - 生产环境中若担心切片复制开销，可改用索引 + 辅助数组版本。
    """
    # Base Case：长度 0 或 1 的数组天然有序
    if len(nums) <= 1:
        return nums[:]

    mid = len(nums) // 2
    left_sorted = merge_sort(nums[:mid])
    right_sorted = merge_sort(nums[mid:])

    return merge(left_sorted, right_sorted)



def merge(left: List[int], right: List[int]) -> List[int]:
    """
    合并两个有序数组

    边界条件：
    - left 或 right 可能为空
    - 处理重复元素时使用 <= 保证稳定性
    """
    i = j = 0
    merged = []

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # 下面两句只会有一句真正追加非空部分
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


nums = [5, 2, 4, 6, 1, 3]
print(merge_sort(nums))  # [1, 2, 3, 4, 5, 6]
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

class MergeSort {
public:
    static void sort(vector<int>& nums) {
        if (nums.empty()) return;
        vector<int> temp(nums.size());
        mergeSort(nums, 0, (int)nums.size() - 1, temp);
    }

private:
    static void mergeSort(vector<int>& nums, int left, int right, vector<int>& temp) {
        // Base Case：单个元素区间天然有序
        if (left >= right) return;

        int mid = left + (right - left) / 2;
        mergeSort(nums, left, mid, temp);
        mergeSort(nums, mid + 1, right, temp);
        merge(nums, left, mid, right, temp);
    }

    static void merge(vector<int>& nums, int left, int mid, int right, vector<int>& temp) {
        int i = left;
        int j = mid + 1;
        int k = left;

        while (i <= mid && j <= right) {
            // 这里用 <= 保证稳定性
            if (nums[i] <= nums[j]) {
                temp[k++] = nums[i++];
            } else {
                temp[k++] = nums[j++];
            }
        }

        while (i <= mid) temp[k++] = nums[i++];
        while (j <= right) temp[k++] = nums[j++];

        for (int p = left; p <= right; ++p) {
            nums[p] = temp[p];
        }
    }
};

int main() {
    vector<int> nums = {5, 2, 4, 6, 1, 3};
    MergeSort::sort(nums);
    for (int x : nums) cout << x << ' ';
    cout << '\n';
    return 0;
}
```

#### 归并排序的优缺点

**优点**：

- 时间复杂度稳定：最坏、平均、最好都是 $O(n\log n)$；
- 稳定排序；
- 适合链表排序、外部排序、多路归并。

**缺点**：

- 需要额外 $O(n)$ 辅助空间；
- 对数组原地排序不如快速排序省空间；
- 对小数组常数不如插入排序。

---

### 26.2.2 快速排序（复习强化）

#### 先建立直觉

如果归并排序像“先各排各的，再合并”，那么快速排序（Quick Sort）更像：

> 先找一个“基准（pivot）”，把数组按它分成“小于它”和“大于它”的两边，然后分别继续排。

它的核心不是合并，而是 **partition（划分）**。

#### 为什么快速排序仍然算分治？

因为它依然满足三步：

- **Divide**：按照 pivot 把数组分成左右两部分；
- **Conquer**：递归排序左半和右半；
- **Combine**：几乎不需要额外合并，因为 pivot 的最终位置已确定。

#### 复杂度为什么“看起来漂亮，实际要小心”？

若每次 pivot 都比较平均地把数组分成两半：

$$
T(n)=2T(n/2)+O(n)=O(n\log n)
$$

但如果每次都极端不平衡，比如分成 $1$ 和 $n-1$：

$$
T(n)=T(n-1)+O(n)=O(n^2)
$$

所以随机化 pivot 很重要，它能让平均情况接近均衡划分。

#### Python 实现（随机 pivot + 原地划分）

```python
import random
from typing import List


def quick_sort(nums: List[int]) -> None:
    """
    原地快速排序

    平均时间复杂度: O(n log n)
    最坏时间复杂度: O(n^2)
    空间复杂度: O(log n)（递归栈，平均）
    """
    def partition(left: int, right: int) -> int:
        # 随机化 pivot，降低退化概率
        pivot_idx = random.randint(left, right)
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        pivot = nums[right]

        i = left  # nums[left:i] 都是 < pivot 的区域
        for j in range(left, right):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

        # 把 pivot 放到最终位置
        nums[i], nums[right] = nums[right], nums[i]
        return i

    def sort(left: int, right: int) -> None:
        if left >= right:
            return
        p = partition(left, right)
        sort(left, p - 1)
        sort(p + 1, right)

    sort(0, len(nums) - 1)


arr = [7, 3, 5, 2, 9, 1, 8]
quick_sort(arr)
print(arr)  # [1, 2, 3, 5, 7, 8, 9]
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

class QuickSort {
public:
    static void sort(vector<int>& nums) {
        if (nums.empty()) return;
        quickSort(nums, 0, (int)nums.size() - 1);
    }

private:
    static int partition(vector<int>& nums, int left, int right) {
        int pivotIdx = left + rand() % (right - left + 1);
        swap(nums[pivotIdx], nums[right]);
        int pivot = nums[right];

        int i = left;
        for (int j = left; j < right; ++j) {
            if (nums[j] < pivot) {
                swap(nums[i], nums[j]);
                ++i;
            }
        }
        swap(nums[i], nums[right]);
        return i;
    }

    static void quickSort(vector<int>& nums, int left, int right) {
        if (left >= right) return;
        int p = partition(nums, left, right);
        quickSort(nums, left, p - 1);
        quickSort(nums, p + 1, right);
    }
};

int main() {
    srand((unsigned)time(nullptr));
    vector<int> nums = {7, 3, 5, 2, 9, 1, 8};
    QuickSort::sort(nums);
    for (int x : nums) cout << x << ' ';
    cout << '\n';
    return 0;
}
```

#### 快速排序的工程地位为什么这么高？

虽然它最坏会到 $O(n^2)$，但在工程中依然极其常用，因为：

- 原地排序，额外空间小；
- cache locality 通常优于归并；
- 常数因子小；
- 与三路划分、插入排序、小区间优化结合后非常快。

---

### 26.2.3 大整数乘法——Karatsuba 算法

#### 先从“普通乘法”出发

假设我们要计算两个 $n$ 位数相乘：

$$
X = a\cdot 10^m + b, \qquad Y = c\cdot 10^m + d
$$

其中：

- $a, c$ 是高位部分；
- $b, d$ 是低位部分；
- 大约各自有 $m=n/2$ 位。

普通分治会展开为：

$$
XY = ac\cdot 10^{2m} + (ad+bc)\cdot 10^m + bd
$$

这需要 **4 次乘法**：

- $ac$
- $ad$
- $bc$
- $bd$

于是递推式为：

$$
T(n)=4T(n/2)+O(n)=O(n^2)
$$

这和普通竖式乘法并没有本质突破。

#### Karatsuba 的天才观察

Karatsuba 发现：

$$
(a+b)(c+d)=ac+ad+bc+bd
$$

所以：

$$
ad+bc=(a+b)(c+d)-ac-bd
$$

于是只需要 3 次乘法：

1. $z_2=ac$
2. $z_0=bd$
3. $z_1=(a+b)(c+d)$

最后：

$$
XY = z_2\cdot 10^{2m} + (z_1-z_2-z_0)\cdot 10^m + z_0
$$

递推式变成：

$$
T(n)=3T(n/2)+O(n)
$$

根据主定理情形一：

$$
T(n)=O(n^{\log_2 3}) \approx O(n^{1.585})
$$

这就是一个真正的复杂度突破。

<div data-component="KaratsubaLargeIntMult"></div>

#### Python 实现（整数版本，便于理解）

```python
def karatsuba(x: int, y: int) -> int:
    """
    Karatsuba 大整数乘法

    说明：
    - 这里直接使用 Python int 做拆分，利于理解公式；
    - 真正的大整数库会处理符号、进位、不同进制、阈值切换等细节。
    """
    # Base Case：数字足够小，直接乘
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    m = n // 2

    power = 10 ** m
    a, b = divmod(x, power)
    c, d = divmod(y, power)

    z2 = karatsuba(a, c)
    z0 = karatsuba(b, d)
    z1 = karatsuba(a + b, c + d)

    return z2 * (10 ** (2 * m)) + (z1 - z2 - z0) * power + z0


print(karatsuba(1234, 5678))  # 7006652
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

long long karatsuba(long long x, long long y) {
    // Base Case：小数直接乘
    if (x < 10 || y < 10) return x * y;

    int n = max((int)to_string(x).size(), (int)to_string(y).size());
    int m = n / 2;

    long long power = 1;
    for (int i = 0; i < m; ++i) power *= 10;

    long long a = x / power, b = x % power;
    long long c = y / power, d = y % power;

    long long z2 = karatsuba(a, c);
    long long z0 = karatsuba(b, d);
    long long z1 = karatsuba(a + b, c + d);

    return z2 * power * power + (z1 - z2 - z0) * power + z0;
}

int main() {
    cout << karatsuba(1234, 5678) << "\n"; // 7006652
    return 0;
}
```

#### Karatsuba 的现实意义

Karatsuba 告诉我们一个非常重要的思想：

> 分治不只是“拆问题”，还可以通过代数恒等变形，**减少递归分支数**。

这类思路在更高级的快速乘法、FFT、多项式乘法中会反复出现。

#### 注意事项

- 常数因子不小，小整数时可能反而比普通乘法慢；
- 工程实现通常设置阈值：位数小于某个值时切回普通乘法；
- 若用字符串实现，要特别注意补零、进位、负号处理。

---

### 26.2.4 Strassen 矩阵乘法

#### 普通矩阵乘法为什么是 8 次子块乘法？

设两个矩阵都拆成四块：

$$
A = \begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{bmatrix},
\qquad
B = \begin{bmatrix}B_{11} & B_{12} \\ B_{21} & B_{22}\end{bmatrix}
$$

那么：

$$
C_{11}=A_{11}B_{11}+A_{12}B_{21}
$$
$$
C_{12}=A_{11}B_{12}+A_{12}B_{22}
$$
$$
C_{21}=A_{21}B_{11}+A_{22}B_{21}
$$
$$
C_{22}=A_{21}B_{12}+A_{22}B_{22}
$$

看起来需要 **8 次子矩阵乘法**，于是有：

$$
T(n)=8T(n/2)+O(n^2)=O(n^3)
$$

#### Strassen 的突破

Strassen 发现，只用 7 次子矩阵乘法就够了。

定义：

$$
M_1=(A_{11}+A_{22})(B_{11}+B_{22})
$$
$$
M_2=(A_{21}+A_{22})B_{11}
$$
$$
M_3=A_{11}(B_{12}-B_{22})
$$
$$
M_4=A_{22}(B_{21}-B_{11})
$$
$$
M_5=(A_{11}+A_{12})B_{22}
$$
$$
M_6=(A_{21}-A_{11})(B_{11}+B_{12})
$$
$$
M_7=(A_{12}-A_{22})(B_{21}+B_{22})
$$

然后：

$$
C_{11}=M_1+M_4-M_5+M_7
$$
$$
C_{12}=M_3+M_5
$$
$$
C_{21}=M_2+M_4
$$
$$
C_{22}=M_1-M_2+M_3+M_6
$$

递推式变为：

$$
T(n)=7T(n/2)+O(n^2)
$$

所以：

$$
T(n)=O(n^{\log_2 7}) \approx O(n^{2.807})
$$

这第一次打破了 $O(n^3)$ 的矩阵乘法壁垒。

<div data-component="StrassenMatrixMult"></div>

#### Python 实现（教学版）

```python
from typing import List

Matrix = List[List[int]]


def add(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def sub(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def naive_mul(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def split(A: Matrix):
    n = len(A)
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    return A11, A12, A21, A22


def combine(C11: Matrix, C12: Matrix, C21: Matrix, C22: Matrix) -> Matrix:
    top = [a + b for a, b in zip(C11, C12)]
    bottom = [a + b for a, b in zip(C21, C22)]
    return top + bottom


def strassen(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)

    # Base Case：小矩阵直接用普通乘法，常数更优
    if n <= 2:
        return naive_mul(A, B)

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    M1 = strassen(add(A11, A22), add(B11, B22))
    M2 = strassen(add(A21, A22), B11)
    M3 = strassen(A11, sub(B12, B22))
    M4 = strassen(A22, sub(B21, B11))
    M5 = strassen(add(A11, A12), B22)
    M6 = strassen(sub(A21, A11), add(B11, B12))
    M7 = strassen(sub(A12, A22), add(B21, B22))

    C11 = add(sub(add(M1, M4), M5), M7)
    C12 = add(M3, M5)
    C21 = add(M2, M4)
    C22 = add(sub(add(M1, M3), M2), M6)

    return combine(C11, C12, C21, C22)


A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print(strassen(A, B))  # [[19, 22], [43, 50]]
```

#### C++ 实现（教学版）

```cpp
#include <bits/stdc++.h>
using namespace std;

using Matrix = vector<vector<int>>;

Matrix add(const Matrix& A, const Matrix& B) {
    int n = (int)A.size();
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix sub(const Matrix& A, const Matrix& B) {
    int n = (int)A.size();
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

Matrix naiveMul(const Matrix& A, const Matrix& B) {
    int n = (int)A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

void split(const Matrix& A, Matrix& A11, Matrix& A12, Matrix& A21, Matrix& A22) {
    int n = (int)A.size(), mid = n / 2;
    A11.assign(mid, vector<int>(mid));
    A12.assign(mid, vector<int>(mid));
    A21.assign(mid, vector<int>(mid));
    A22.assign(mid, vector<int>(mid));

    for (int i = 0; i < mid; ++i) {
        for (int j = 0; j < mid; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + mid];
            A21[i][j] = A[i + mid][j];
            A22[i][j] = A[i + mid][j + mid];
        }
    }
}

Matrix combine(const Matrix& C11, const Matrix& C12, const Matrix& C21, const Matrix& C22) {
    int mid = (int)C11.size();
    int n = mid * 2;
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < mid; ++i) {
        for (int j = 0; j < mid; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + mid] = C12[i][j];
            C[i + mid][j] = C21[i][j];
            C[i + mid][j + mid] = C22[i][j];
        }
    }
    return C;
}

Matrix strassen(const Matrix& A, const Matrix& B) {
    int n = (int)A.size();
    if (n <= 2) return naiveMul(A, B);

    Matrix A11, A12, A21, A22, B11, B12, B21, B22;
    split(A, A11, A12, A21, A22);
    split(B, B11, B12, B21, B22);

    Matrix M1 = strassen(add(A11, A22), add(B11, B22));
    Matrix M2 = strassen(add(A21, A22), B11);
    Matrix M3 = strassen(A11, sub(B12, B22));
    Matrix M4 = strassen(A22, sub(B21, B11));
    Matrix M5 = strassen(add(A11, A12), B22);
    Matrix M6 = strassen(sub(A21, A11), add(B11, B12));
    Matrix M7 = strassen(sub(A12, A22), add(B21, B22));

    Matrix C11 = add(sub(add(M1, M4), M5), M7);
    Matrix C12 = add(M3, M5);
    Matrix C21 = add(M2, M4);
    Matrix C22 = add(sub(add(M1, M3), M2), M6);

    return combine(C11, C12, C21, C22);
}

int main() {
    Matrix A = {{1,2},{3,4}};
    Matrix B = {{5,6},{7,8}};
    Matrix C = strassen(A, B);
    for (auto& row : C) {
        for (int x : row) cout << x << ' ';
        cout << '\n';
    }
    return 0;
}
```

#### 为什么工程里不常直接用 Strassen？

虽然理论上更快，但它有几个现实问题：

1. 常数因子大；
2. 需要大量矩阵加减法和临时空间；
3. 对数值误差更敏感（浮点矩阵时尤其明显）；
4. 只有矩阵规模很大时才可能体现优势。

所以：

- 理论上，Strassen 非常重要；
- 工程上，BLAS / cache blocking / SIMD / GPU 往往更关键。

---

### 26.2.5 最近点对（Closest Pair of Points）

#### 问题定义

给定平面上 $n$ 个点，求距离最近的两点。

最朴素的方法是枚举所有点对：

$$
O(n^2)
$$

当点很多时，这显然不够快。

#### 分治思路

1. 按 $x$ 坐标排序；
2. 从中线把点分为左右两半；
3. 递归求左半最近距离 $d_L$ 和右半最近距离 $d_R$；
4. 令

$$
d = \min(d_L, d_R)
$$

5. 最后只检查中线附近宽度为 $2d$ 的带状区域（strip）。

难点在于：

> 为什么只检查 strip 里有限个点就够了？

核心结论是著名的 **7 点引理**：

- 若 strip 内点按 $y$ 坐标有序；
- 对每个点，只需向后检查常数个点（最多 7 个）；
- 因此 strip 检查是 $O(n)$。

于是总体递推式：

$$
T(n)=2T(n/2)+O(n)
$$

所以总复杂度：

$$
O(n\log n)
$$

<div data-component="ClosestPairStrip"></div>

#### Python 实现

```python
from math import dist
from typing import List, Tuple

Point = Tuple[float, float]


def closest_pair(points: List[Point]) -> float:
    """
    最近点对分治算法

    时间复杂度: O(n log n)
    空间复杂度: O(n)
    """
    px = sorted(points)                 # 按 x 排序
    py = sorted(points, key=lambda p: p[1])  # 按 y 排序

    def solve(px: List[Point], py: List[Point]) -> float:
        n = len(px)

        # Base Case：点数很少时直接暴力
        if n <= 3:
            ans = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    ans = min(ans, dist(px[i], px[j]))
            return ans

        mid = n // 2
        mid_x = px[mid][0]

        left_px = px[:mid]
        right_px = px[mid:]

        left_set = set(left_px)
        left_py = [p for p in py if p in left_set]
        right_py = [p for p in py if p not in left_set]

        d_left = solve(left_px, left_py)
        d_right = solve(right_px, right_py)
        d = min(d_left, d_right)

        # 构造 strip：只保留离中线距离 < d 的点，仍按 y 有序
        strip = [p for p in py if abs(p[0] - mid_x) < d]

        # 7 点引理：每个点只需往后看常数个点
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 8, len(strip))):
                d = min(d, dist(strip[i], strip[j]))

        return d

    return solve(px, py)


pts = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
print(round(closest_pair(pts), 3))  # 1.414
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
};

double getDist(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double bruteForce(vector<Point>& pts, int l, int r) {
    double ans = 1e100;
    for (int i = l; i <= r; ++i)
        for (int j = i + 1; j <= r; ++j)
            ans = min(ans, getDist(pts[i], pts[j]));
    return ans;
}

double closestPairRec(vector<Point>& px, int l, int r) {
    if (r - l + 1 <= 3) return bruteForce(px, l, r);

    int mid = l + (r - l) / 2;
    double midx = px[mid].x;

    double d1 = closestPairRec(px, l, mid);
    double d2 = closestPairRec(px, mid + 1, r);
    double d = min(d1, d2);

    vector<Point> strip;
    for (int i = l; i <= r; ++i) {
        if (abs(px[i].x - midx) < d) strip.push_back(px[i]);
    }
    sort(strip.begin(), strip.end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
    });

    for (int i = 0; i < (int)strip.size(); ++i) {
        for (int j = i + 1; j < (int)strip.size() && j <= i + 7; ++j) {
            d = min(d, getDist(strip[i], strip[j]));
        }
    }
    return d;
}

double closestPair(vector<Point> pts) {
    sort(pts.begin(), pts.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });
    return closestPairRec(pts, 0, (int)pts.size() - 1);
}

int main() {
    vector<Point> pts = {{2,3},{12,30},{40,50},{5,1},{12,10},{3,4}};
    cout << fixed << setprecision(3) << closestPair(pts) << "\n";
    return 0;
}
```

#### 为什么这个算法经典？

因为它体现了分治算法的一个重要设计技巧：

> 拆完左右后，看似还要检查“跨边界”情况，
> 但如果能用几何性质把跨边界检查压到线性，就能维持整体 $O(n\log n)$。

这类思路在计算几何里非常常见。

---

### 26.2.6 快速幂（Binary Exponentiation）与矩阵快速幂

#### 先从最朴素想法开始

如果要计算 $a^n$，最直接的方法是乘 $n-1$ 次：

```text
res = a * a * a * ... * a
```

复杂度：

$$
O(n)
$$

当 $n$ 非常大时，这会很慢。

#### 分治视角：指数减半

关键观察：

- 若 $n$ 是偶数：

$$
a^n = (a^{n/2})^2
$$

- 若 $n$ 是奇数：

$$
a^n = a \cdot a^{n-1}
$$

于是每次递归都把指数减半，递推式近似：

$$
T(n)=T(n/2)+O(1)
$$

所以：

$$
T(n)=O(\log n)
$$

#### Python：标量快速幂

```python
def fast_pow(a: int, n: int) -> int:
    """
    分治递归版快速幂
    时间复杂度: O(log n)
    """
    if n == 0:
        return 1
    if n == 1:
        return a

    half = fast_pow(a, n // 2)
    if n % 2 == 0:
        return half * half
    else:
        return half * half * a


print(fast_pow(2, 10))  # 1024
```

#### C++：标量快速幂

```cpp
#include <bits/stdc++.h>
using namespace std;

long long fastPow(long long a, long long n) {
    if (n == 0) return 1;
    if (n == 1) return a;

    long long half = fastPow(a, n / 2);
    if (n % 2 == 0) return half * half;
    return half * half * a;
}

int main() {
    cout << fastPow(2, 10) << "\n"; // 1024
    return 0;
}
```

#### 矩阵快速幂为什么重要？

有些递推关系可以写成矩阵形式，比如斐波那契数列：

$$
\begin{bmatrix}F_{n+1} \\ F_n\end{bmatrix} =
\begin{bmatrix}1 & 1 \\ 1 & 0\end{bmatrix}^n
\begin{bmatrix}1 \\ 0\end{bmatrix}
$$

于是问题就变成：快速计算矩阵的 $n$ 次幂。

#### Python：矩阵快速幂求斐波那契

```python
from typing import List

Matrix2 = List[List[int]]


def mul(A: Matrix2, B: Matrix2) -> Matrix2:
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]


def mat_pow(M: Matrix2, n: int) -> Matrix2:
    # 单位矩阵
    result = [[1, 0], [0, 1]]
    base = M

    while n > 0:
        if n & 1:
            result = mul(result, base)
        base = mul(base, base)
        n >>= 1

    return result


def fib(n: int) -> int:
    if n == 0:
        return 0
    M = [[1, 1], [1, 0]]
    R = mat_pow(M, n - 1)
    return R[0][0]


print(fib(10))  # 55
```

#### C++：矩阵快速幂求斐波那契

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Matrix2 {
    long long a00, a01, a10, a11;
};

Matrix2 mul(const Matrix2& A, const Matrix2& B) {
    return {
        A.a00 * B.a00 + A.a01 * B.a10,
        A.a00 * B.a01 + A.a01 * B.a11,
        A.a10 * B.a00 + A.a11 * B.a10,
        A.a10 * B.a01 + A.a11 * B.a11
    };
}

Matrix2 matPow(Matrix2 base, long long n) {
    Matrix2 res{1, 0, 0, 1}; // 单位矩阵
    while (n > 0) {
        if (n & 1) res = mul(res, base);
        base = mul(base, base);
        n >>= 1;
    }
    return res;
}

long long fib(int n) {
    if (n == 0) return 0;
    Matrix2 M{1, 1, 1, 0};
    Matrix2 R = matPow(M, n - 1);
    return R.a00;
}

int main() {
    cout << fib(10) << "\n"; // 55
    return 0;
}
```

#### 快速幂的本质总结

快速幂之所以是分治，不是因为它“有递归”或者“有循环”，而是因为：

> 它把“求 $n$ 次幂”分解成“求一半指数的幂”，
> 再通过平方合并答案。

---

### 26.2.7 逆序对计数（Inversion Count via Merge Sort）

#### 什么是逆序对？

若数组中存在下标 $i<j$，但有：

$$
a_i > a_j
$$

则 $(i,j)$ 是一个逆序对。

例如：

```text
[3, 1, 2]
```

逆序对有：

- (3,1)
- (3,2)

共 2 个。

#### 为什么暴力法慢？

直接双重循环检查每一对：

$$
O(n^2)
$$

#### 分治的关键洞察

把数组分成左右两半后，逆序对分三类：

1. 完全在左半中；
2. 完全在右半中；
3. **跨左右两半**。

前两类可以递归统计。

真正精华在第三类：

- 左半和右半在递归返回后都已经有序；
- 合并时若 `left[i] > right[j]`，则 `left[i:]` 中所有元素都大于 `right[j]`；
- 因此可一次性加上：

$$
\text{mid} - i + 1
$$

这样就把跨区间逆序对计数压成了线性时间。

递推式：

$$
T(n)=2T(n/2)+O(n)
$$

因此总复杂度：

$$
O(n\log n)
$$

<div data-component="InversionCountMerge"></div>

#### Python 实现

```python
from typing import List, Tuple


def count_inversions(nums: List[int]) -> int:
    """
    逆序对计数（归并排序分治版）

    返回逆序对总数
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    """
    def sort_count(arr: List[int]) -> Tuple[List[int], int]:
        if len(arr) <= 1:
            return arr[:], 0

        mid = len(arr) // 2
        left_sorted, left_cnt = sort_count(arr[:mid])
        right_sorted, right_cnt = sort_count(arr[mid:])

        merged = []
        i = j = 0
        cross_cnt = 0

        while i < len(left_sorted) and j < len(right_sorted):
            if left_sorted[i] <= right_sorted[j]:
                merged.append(left_sorted[i])
                i += 1
            else:
                merged.append(right_sorted[j])
                # left_sorted[i:] 全都比 right_sorted[j] 大
                cross_cnt += len(left_sorted) - i
                j += 1

        merged.extend(left_sorted[i:])
        merged.extend(right_sorted[j:])

        return merged, left_cnt + right_cnt + cross_cnt

    _, ans = sort_count(nums)
    return ans


print(count_inversions([2, 4, 1, 3, 5]))  # 3
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

class InversionCounter {
public:
    static long long count(vector<int>& nums) {
        if (nums.empty()) return 0;
        vector<int> temp(nums.size());
        return mergeCount(nums, 0, (int)nums.size() - 1, temp);
    }

private:
    static long long mergeCount(vector<int>& nums, int left, int right, vector<int>& temp) {
        if (left >= right) return 0;

        int mid = left + (right - left) / 2;
        long long cnt = 0;
        cnt += mergeCount(nums, left, mid, temp);
        cnt += mergeCount(nums, mid + 1, right, temp);
        cnt += merge(nums, left, mid, right, temp);
        return cnt;
    }

    static long long merge(vector<int>& nums, int left, int mid, int right, vector<int>& temp) {
        int i = left, j = mid + 1, k = left;
        long long cnt = 0;

        while (i <= mid && j <= right) {
            if (nums[i] <= nums[j]) {
                temp[k++] = nums[i++];
            } else {
                temp[k++] = nums[j++];
                // nums[i..mid] 都比 nums[j-1] 大
                cnt += (mid - i + 1);
            }
        }

        while (i <= mid) temp[k++] = nums[i++];
        while (j <= right) temp[k++] = nums[j++];

        for (int p = left; p <= right; ++p) nums[p] = temp[p];
        return cnt;
    }
};

int main() {
    vector<int> nums = {2, 4, 1, 3, 5};
    cout << InversionCounter::count(nums) << "\n"; // 3
    return 0;
}
```

#### 这个问题为什么面试高频？

因为它考察的不是“会不会归并排序”，而是：

> 你能不能意识到：
> **排序过程本身就携带了很多结构信息，可以顺便统计答案。**

这是一种非常典型的算法思维升级。

---

## 26.3 分治算法设计技巧

### 26.3.1 关键原则：合并步骤不能过重

分治之所以快，往往是因为：

- 子问题规模迅速变小；
- 每一层的额外工作不过分夸张。

如果你设计了一个分治方案：

- 拆成两个子问题；
- 但合并居然要 $O(n^2)$；

那么递推式会变成：

$$
T(n)=2T(n/2)+O(n^2)
$$

最终复杂度就是：

$$
O(n^2)
$$

这时分治就没有明显优势了。

**经验法则**：

- 若想得到 $O(n\log n)$，通常希望每层总代价不超过 $O(n)$；
- 若合并代价远大于子问题缩小收益，分治就会被合并拖垮。

#### 一个简单对比

- 归并排序：合并 $O(n)$，成功达到 $O(n\log n)$；
- 假设你设计某算法，拆左右两半，但最后要把左右答案两两配对比较，合并变成 $O(n^2)$，那复杂度就会崩。

> 所以：
> **分治不是“敢拆就行”，而是“拆完能便宜地合回来”。**

---

### 26.3.2 非平衡分治：快速选择 vs 归并排序

很多教材默认分治是“平衡”的，即每次都对半拆。

但实际中常出现**非平衡分治**。

最典型例子就是快速选择（QuickSelect）：

- 它和快速排序很像，也是 partition；
- 但每次只递归进入一侧；
- 若 pivot 很理想，规模大幅缩小；
- 若 pivot 很糟糕，可能退化成 $n-1$ 和 $0$。

#### 为什么它平均很快？

因为平均情况下，pivot 会较均匀地切分数组，于是只追一边的代价接近：

$$
T(n)=T(n/2)+O(n)=O(n)
$$

但最坏情况下：

$$
T(n)=T(n-1)+O(n)=O(n^2)
$$

#### Python：QuickSelect

```python
import random
from typing import List


def quick_select(nums: List[int], k: int) -> int:
    """
    返回第 k 小元素（1-indexed）
    平均时间复杂度: O(n)
    最坏时间复杂度: O(n^2)
    """
    k -= 1  # 转成 0-indexed

    def partition(left: int, right: int) -> int:
        pivot_idx = random.randint(left, right)
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        pivot = nums[right]
        i = left
        for j in range(left, right):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i

    left, right = 0, len(nums) - 1
    while left <= right:
        p = partition(left, right)
        if p == k:
            return nums[p]
        elif p < k:
            left = p + 1
        else:
            right = p - 1

    raise ValueError("k out of range")


print(quick_select([7, 2, 1, 6, 8, 5, 3, 4], 3))  # 3
```

#### C++：QuickSelect

```cpp
#include <bits/stdc++.h>
using namespace std;

int partitionVec(vector<int>& nums, int left, int right) {
    int pivotIdx = left + rand() % (right - left + 1);
    swap(nums[pivotIdx], nums[right]);
    int pivot = nums[right];
    int i = left;
    for (int j = left; j < right; ++j) {
        if (nums[j] < pivot) swap(nums[i++], nums[j]);
    }
    swap(nums[i], nums[right]);
    return i;
}

int quickSelect(vector<int> nums, int k) { // k: 1-indexed
    --k;
    int left = 0, right = (int)nums.size() - 1;
    while (left <= right) {
        int p = partitionVec(nums, left, right);
        if (p == k) return nums[p];
        if (p < k) left = p + 1;
        else right = p - 1;
    }
    throw runtime_error("k out of range");
}

int main() {
    vector<int> nums = {7,2,1,6,8,5,3,4};
    cout << quickSelect(nums, 3) << "\n"; // 3
    return 0;
}
```

#### 核心结论

- 平衡分治通常容易得到稳定复杂度；
- 非平衡分治常在平均情况下很优，但最坏情况可能恶化；
- 随机化是工程上对抗退化的重要手段。

---

### 26.3.3 尾递归优化与记忆化：避免多余开销

#### 尾递归优化（Tail Recursion Optimization）

尾递归指的是：

> 递归调用是函数的最后一步操作，返回后不需要再做别的事。

例如快速幂的某些写法、快速排序中只保留一侧递归时，都可能转成更节省栈的版本。

在支持尾递归优化的语言/编译器中，尾递归有机会被编译成迭代，从而减少栈空间。

但要注意：

- Python **默认不做**尾递归优化；
- C++ 编译器是否优化，取决于具体实现和编译条件；
- 所以面试/工程中，若担心栈深度，最好直接改写成迭代。

#### 记忆化（Memoization）为什么放在这里讲？

因为很多初学者会把“递归 + 记忆化”和“分治”混为一谈。

区别在于：

- **分治**：子问题通常独立，不重叠；
- **记忆化/DP**：子问题重叠，需要缓存结果避免重复算。

例如：

- 归并排序不需要记忆化；
- 朴素斐波那契递归非常需要记忆化。

#### Python：斐波那契记忆化对比

```python
from functools import lru_cache


def fib_bad(n: int) -> int:
    # 大量重复子问题，不是好的分治
    if n <= 1:
        return n
    return fib_bad(n - 1) + fib_bad(n - 2)


@lru_cache(None)
def fib_memo(n: int) -> int:
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)
```

#### C++：记忆化版本

```cpp
#include <bits/stdc++.h>
using namespace std;

long long fibMemo(int n, vector<long long>& memo) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    return memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
}

int main() {
    int n = 10;
    vector<long long> memo(n + 1, -1);
    cout << fibMemo(n, memo) << "\n"; // 55
    return 0;
}
```

**要点总结**：

- 尾递归优化：解决“栈开销”问题；
- 记忆化：解决“重复子问题”问题；
- 它们都可能和分治一起出现，但解决的是不同层面的事情。

---

### 26.3.4 分治与 DP 的边界：子问题是否独立？

这是面试中特别高频的一个追问：

> 这个题到底该用分治还是动态规划？

一个非常实用的判断标准是：

## **子问题是否重叠？**

#### 适合分治的典型特征

- 子问题彼此独立；
- 算完一个子问题，不会反复依赖另一个同规模子问题；
- 最后只需要合并答案。

例子：

- 归并排序；
- 最近点对；
- 快速幂；
- Strassen。

#### 适合 DP 的典型特征

- 子问题会重复出现；
- 若不用缓存，会重复计算很多次；
- 更适合用表格或记忆化保存中间结果。

例子：

- 斐波那契；
- 最长公共子序列；
- 背包；
- 区间 DP。

#### 一个很重要的对比

| 问题 | 更适合 | 原因 |
|---|---|---|
| 归并排序 | 分治 | 左右排序互不重叠 |
| 最近点对 | 分治 | 左右部分独立，仅边界线性合并 |
| 斐波那契 | DP | 子问题大量重叠 |
| 矩阵链乘 | DP | 拆分位置不同但子区间重复出现 |

> 一句话记忆：
> **独立拆开 → 分治；重复出现 → DP。**

---

### 26.3.5 并行分治：天然适合多核与分布式

分治算法的一大工程优势是：

> **子问题往往天然独立，因此非常适合并行。**

例如归并排序：

- 左半排序和右半排序可同时执行；
- 最后再做一次合并。

若有两颗 CPU 核：

- 单核工作量（Work）仍接近 $O(n\log n)$；
- 关键路径长度（Span）会明显缩短。

在并行算法里，经常区分两个概念：

1. **工作量（Work）**：所有操作总数；
2. **跨度（Span）/ 临界路径（Critical Path）**：最长依赖链长度。

以并行归并排序为例：

- Work 仍为 $O(n\log n)$；
- 若合并也能并行优化，Span 可远小于单线程时间。

#### Python（概念示意，不强调性能）

```python
# 这里只展示“可并行性思路”，不作为高性能实现
# 真正并行常用 multiprocessing / joblib / Ray / C++ OpenMP / TBB 等

def parallel_merge_sort_concept(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # 概念上：left 和 right 可以并行排序
    left_sorted = parallel_merge_sort_concept(left)
    right_sorted = parallel_merge_sort_concept(right)

    return merge(left_sorted, right_sorted)
```

#### C++（OpenMP 风格概念示意）

```cpp
// 仅示意伪工程思路，实际项目需根据阈值控制任务粒度
#include <bits/stdc++.h>
using namespace std;

void parallelMergeSort(vector<int>& nums, int l, int r, vector<int>& temp) {
    if (l >= r) return;
    int mid = l + (r - l) / 2;

    // 概念上可以并行：
    // #pragma omp task
    parallelMergeSort(nums, l, mid, temp);

    // #pragma omp task
    parallelMergeSort(nums, mid + 1, r, temp);

    // #pragma omp taskwait
    // 然后 merge(nums, l, mid, r, temp)
}
```

#### 并行分治的现实问题

并行不是免费的，还要考虑：

- 任务创建开销；
- 线程同步开销；
- cache 竞争；
- 子任务太小时并行反而更慢。

所以工程中通常会设阈值：

- 当区间足够大时才并行；
- 小区间退回串行算法。

---

## 26.4 算法复杂度对比与总结

| 算法 | 递推式 | 时间复杂度 | 空间复杂度 | 核心思想 |
|---|---|---|---|---|
| 归并排序 | $2T(n/2)+O(n)$ | $O(n\log n)$ | $O(n)$ | 对半拆分 + 线性合并 |
| 快速排序 | 平均 $2T(n/2)+O(n)$ | 平均 $O(n\log n)$，最坏 $O(n^2)$ | 平均 $O(\log n)$ | partition 划分 |
| Karatsuba | $3T(n/2)+O(n)$ | $O(n^{1.585})$ | 递归栈 + 中间对象 | 3 次乘法代替 4 次 |
| Strassen | $7T(n/2)+O(n^2)$ | $O(n^{2.807})$ | 较大 | 7 次子矩阵乘法代替 8 次 |
| 最近点对 | $2T(n/2)+O(n)$ | $O(n\log n)$ | $O(n)$ | strip + 7 点引理 |
| 快速幂 | $T(n/2)+O(1)$ | $O(\log n)$ | $O(\log n)$ / 迭代 $O(1)$ | 指数减半 |
| 逆序对计数 | $2T(n/2)+O(n)$ | $O(n\log n)$ | $O(n)$ | 合并时顺便统计 |
| QuickSelect | 平均 $T(n/2)+O(n)$ | 平均 $O(n)$，最坏 $O(n^2)$ | 平均 $O(\log n)$ | 只递归一侧 |

**本章核心主线回顾**：

- 分治不只是“递归地拆”，而是“拆得合理、合得便宜”；
- 主定理是分析标准分治递推式的核心工具；
- Karatsuba / Strassen 展示了“代数变形减少递归分支”的高级技巧；
- 最近点对与逆序对计数展示了“在合并阶段顺便完成额外统计/筛选”的设计艺术；
- 分治与 DP 的边界，关键看**子问题是否重叠**；
- 分治非常适合并行，因为多个子问题天然独立。

---

## 本章常见错误与调试技巧

> ⚠️ **主定理乱套用**：只有当递推式能写成 $T(n)=aT(n/b)+f(n)$ 且所有子问题规模一致时，才能直接使用主定理。像 $T(n)=T(n/2)+T(n/3)+n$ 不能硬套。

> ⚠️ **忽略辅助空间**：归并排序、最近点对、Strassen 往往不是“纯时间问题”，它们都可能引入显著额外空间，面试中要主动说明。

> ⚠️ **快速排序最坏情况**：如果输入近乎有序，且总是选端点元素做 pivot，很容易退化到 $O(n^2)$。随机 pivot 或三路划分是常见修正方案。

> ⚠️ **逆序对计数 off-by-one**：当 `left[i] > right[j]` 时，增加的是“左半剩余元素个数”，即 `len(left) - i` 或 `mid - i + 1`，不要漏算或多算。

> ⚠️ **Strassen / Karatsuba 过早优化**：理论复杂度更好，不代表小规模就更快。工程中常需要“阈值切换”：小规模回退到朴素算法。

> ⚠️ **最近点对 strip 检查写成平方级**：若每个 strip 点都和后面所有点比较，就把算法写废了。正确做法是利用按 y 排序和 7 点引理，只比较常数个候选点。

---

## 面试高频问题

1. **为什么归并排序是稳定的，而快速排序通常不是？**
2. **主定理三种情形如何快速判断？**
3. **Karatsuba 为什么能把 4 次乘法减少到 3 次？**
4. **Strassen 为什么理论快但工程中不常直接用？**
5. **逆序对计数为什么能在 merge 阶段一次加一段？**
6. **分治与 DP 的根本区别是什么？**
7. **快速幂为什么是 $O(\log n)$？矩阵快速幂如何用于线性递推？**

---

## 练习与思考题

### 基础练习

1. 手写归并排序，并说明为什么它稳定。
2. 手写随机化快速排序，并解释为什么平均是 $O(n\log n)$。
3. 用快速幂计算 $3^{45}$，并写出递归展开过程。
4. 求数组 `[5,4,3,2,1]` 的逆序对数量。

### 进阶思考

1. 为什么 $T(n)=2T(n/2)+n\log n$ 的答案是 $O(n\log^2 n)$？
2. 若一个分治算法每层合并代价为 $O(n^2)$，它还有机会做到 $O(n\log n)$ 吗？为什么？
3. Strassen 通过“7 代 8”降低了复杂度，那么能不能继续做到 6 次？这为什么极其困难？
4. 最近点对为什么只需检查 strip 中常数个点？如果在三维空间中会发生什么变化？

---

## 扩展阅读

- **CLRS** 第 2、4、31 章：分治、主定理、矩阵乘法
- **Skiena《The Algorithm Design Manual》**：Divide and Conquer 章节
- **CP-Algorithms**：Karatsuba、最近点对、二分幂
- **LeetCode 高频题**：
  - #23 合并 K 个升序链表（多路归并思想）
  - #53 最大子数组（分治版可做思维训练）
  - #169 多数元素（分治版）
  - #315 计算右侧小于当前元素的个数（逆序对扩展）
  - #50 Pow(x, n)（快速幂）

下一章我们将进入**动态规划（Dynamic Programming）**。你会发现：

- 分治强调“拆成独立子问题”；
- DP 强调“重叠子问题要缓存”。

这两者既相似，又经常被混淆。把本章彻底吃透，你在下一章会轻松很多。
