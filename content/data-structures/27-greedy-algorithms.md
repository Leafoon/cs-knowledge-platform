# Chapter 27: 贪心算法（Greedy Algorithms）

## 章节导读

在 Chapter 26 我们学习了分治：把大问题拆开，各自解决，再合并。

这一章我们进入另一种非常高频、非常“工程化”的思想：**贪心算法（Greedy Algorithm）**。

很多同学第一次学贪心，会觉得它像“拍脑袋”策略：

- 每一步都选当前看起来最好的；
- 不回头，不后悔；
- 希望最后刚好是全局最优。

这听起来很危险，但在很多问题上它不仅正确，而且极其高效。

你可以把贪心理解成：

- **分治**像“先拆再合”；
- **动态规划（DP）**像“把所有可能状态都算全再取最优”；
- **贪心**像“每一步做一个局部最优决定，并证明这个决定不会让全局变差”。

本章重点不是背模板，而是建立两个能力：

1. **什么时候贪心可行？**
2. **如何证明贪心正确？**

这两件事正是面试和竞赛中真正拉开差距的地方。

---

## 27.1 贪心算法的核心思想

### 27.1.1 贪心选择性质（Greedy-Choice Property）

贪心算法最核心的前提叫：**贪心选择性质**。

它的意思是：

> 在某个问题中，存在一种“当前局部最优选择”，并且总能扩展成全局最优解。

这个定义有点抽象，我们用生活比喻理解：

- 你在安排一天会议，希望参加最多场。
- 如果你先选“最晚结束”的会议，后面几乎没空间了。
- 如果你先选“最早结束”的会议，你为后面留出了最大余地。

这就是活动选择问题里的典型贪心：**每次选结束最早的活动**。

所谓贪心，不是“随便选一个看起来不错的”，而是选一个有证明支撑的“安全选择（safe choice）”。

---

### 27.1.2 最优子结构（Optimal Substructure）

贪心和 DP 都常有“最优子结构”：

> 原问题的最优解，包含子问题的最优解。

但两者分水岭在于：

- DP：子问题通常**重叠**，需要记忆化或表格。
- 贪心：子问题通常不需要保存完整全局状态，只要一步步做“安全选择”。

例如：

- 活动选择（无权）：贪心可解。
- 带权活动选择：贪心失败，转 DP。

所以你不能只看到“有最优子结构”就上贪心。

---

### 27.1.3 贪心 vs 动态规划：怎么选？

实战判断顺序建议如下：

1. 先问：能不能构造一个“局部选择”始终不吃亏？
2. 若能，再问：能否通过交换论证/归纳证明其全局最优？
3. 若不能，且问题有重叠子问题，通常考虑 DP。

可用这张速判表：

| 特征 | 贪心 | DP |
|---|---|---|
| 每步只需做一个局部决策 | 常见 | 不一定 |
| 需要回看大量历史状态 | 少 | 多 |
| 子问题重叠 | 通常弱 | 通常强 |
| 证明方式 | 交换论证 / 切割性质 / 不变式 | 状态转移 + 归纳 |
| 时间复杂度 | 往往更低 | 往往更高但稳妥 |

<div data-component="GreedyVsDPDecisionMap"></div>

---

### 27.1.4 交换论证（Exchange Argument）

交换论证是贪心证明中最常见的武器。

套路是：

1. 假设存在一个最优解 OPT，但它第一步没选贪心选项；
2. 把 OPT 的第一步替换成贪心选项；
3. 证明替换后解不更差（可行且价值不降）；
4. 说明“存在一个最优解以贪心选择开头”；
5. 递归/归纳到剩余子问题。

这是一个非常强的思想：

> 你不需要证明“所有最优解都等于贪心”，只要证明“至少有一个最优解可以改造成以贪心开头”。

---

### 27.1.5 归纳法证明贪心最优性的标准框架

一个可复用的证明模板：

1. **定义问题规模**（如剩余活动数、剩余任务数）。
2. **Base Case**：规模很小（0/1）时显然最优。
3. **归纳假设**：规模 < n 的问题，贪心最优。
4. **归纳步骤**：
   - 用交换论证证明“贪心第一步是安全的”；
   - 去掉第一步后得到规模更小的同类问题；
   - 由归纳假设得剩余部分也最优；
   - 合并得整体最优。

你会发现，本章后面的活动选择、Huffman、MST 的正确性证明都在用这个骨架。

<div data-component="ExchangeArgumentAnimator"></div>

---

## 27.2 活动选择问题（Activity Selection）

### 27.2.1 问题定义

给定 $n$ 个活动，每个活动有开始时间 $s_i$ 和结束时间 $f_i$，活动区间写为：

$$
[s_i, f_i)
$$

目标：选出最多个互不重叠活动。

“互不重叠”定义为：若选了活动 $i$ 和 $j$，则必须满足

$$
f_i \le s_j \quad \text{或} \quad f_j \le s_i
$$

---

### 27.2.2 贪心策略：每次选结束最早的活动

直觉：

- 结束越早，给后续留下的时间越多；
- 因此“最早结束优先（Earliest Finish Time First）”很自然。

算法：

1. 按结束时间升序排序；
2. 从前往后扫；
3. 若当前活动开始时间不早于上次已选活动结束时间，就选它。

---

### 27.2.3 正确性证明（交换论证 + 归纳）

设贪心首选活动为 $g$（结束最早）。

对任意一个最优解 OPT，令它的第一个活动为 $o$。

因为 $g$ 结束最早，有：

$$
f_g \le f_o
$$

用 $g$ 替换 $o$ 后，后续活动可行性不变（甚至更宽松），因此可得到一个同样大小的最优解且以 $g$ 开头。

于是第一步是安全的。

删去 $g$ 后，剩下是同类子问题，归纳即可。

---

### 27.2.4 O(n log n) 实现

#### Python 实现

```python
from typing import List, Tuple


def activity_selection(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    无权活动选择：最多活动数

    思路：按结束时间排序 + 单次扫描
    时间复杂度：O(n log n)
    空间复杂度：O(n)（排序输出）

    Edge Cases:
    - 空输入
    - 相同结束时间
    - 零长度区间 [x, x)
    """
    if not intervals:
        return []

    # 结束时间升序，若相同可按开始时间升序稳定化
    intervals = sorted(intervals, key=lambda it: (it[1], it[0]))

    chosen = []
    last_end = float('-inf')

    for s, f in intervals:
        # 关键条件：当前开始时间 >= 上次结束时间
        if s >= last_end:
            chosen.append((s, f))
            last_end = f

    return chosen


arr = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11)]
ans = activity_selection(arr)
print(ans)
print("count =", len(ans))
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<pair<int,int>> activitySelection(vector<pair<int,int>> intervals) {
    // Edge case
    if (intervals.empty()) return {};

    // 按结束时间升序，结束相同按开始时间升序
    sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });

    vector<pair<int,int>> chosen;
    int lastEnd = INT_MIN;

    for (auto [s, f] : intervals) {
        if (s >= lastEnd) {
            chosen.push_back({s, f});
            lastEnd = f;
        }
    }
    return chosen;
}

int main() {
    vector<pair<int,int>> arr = {{1,4},{3,5},{0,6},{5,7},{3,9},{5,9},{6,10},{8,11}};
    auto ans = activitySelection(arr);
    for (auto [s,f] : ans) cout << "[" << s << "," << f << ") ";
    cout << "\ncount = " << ans.size() << "\n";
    return 0;
}
```

<div data-component="ActivitySelectionTimeline"></div>

---

### 27.2.5 变体：带权活动选择（贪心失效）

如果每个活动有收益 $w_i$，目标从“最多数量”变成“最大总收益”，贪心策略可能失败。

这就是 LeetCode #1235（Maximum Profit in Job Scheduling）类型。

经典做法：

- 按结束时间排序；
- 对每个活动二分找最后一个不冲突活动；
- 做 DP：

$$
dp[i]=\max(dp[i-1], dp[p(i)] + w_i)
$$

#### Python 实现（DP + 二分）

```python
from bisect import bisect_right
from typing import List, Tuple


def weighted_activity_selection(jobs: List[Tuple[int, int, int]]) -> int:
    """
    jobs: (start, end, profit)
    返回最大收益
    """
    if not jobs:
        return 0

    jobs.sort(key=lambda x: x[1])
    ends = [e for _, e, _ in jobs]
    n = len(jobs)

    dp = [0] * (n + 1)
    # dp[i]: 前 i 个工作（1..i）最大收益

    for i in range(1, n + 1):
        s, e, w = jobs[i - 1]
        # 找到最后一个 end <= s 的工作数量 idx
        idx = bisect_right(ends, s)
        dp[i] = max(dp[i - 1], dp[idx] + w)

    return dp[n]


print(weighted_activity_selection([(1,3,50),(3,5,20),(6,19,100),(2,100,200)]))
```

#### C++ 实现（DP + 二分）

```cpp
#include <bits/stdc++.h>
using namespace std;

int weightedActivitySelection(vector<array<int,3>> jobs) {
    // jobs[i] = {start, end, profit}
    if (jobs.empty()) return 0;

    sort(jobs.begin(), jobs.end(), [](auto& a, auto& b) {
        return a[1] < b[1];
    });

    vector<int> ends;
    for (auto& j : jobs) ends.push_back(j[1]);

    int n = (int)jobs.size();
    vector<int> dp(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        int s = jobs[i - 1][0], w = jobs[i - 1][2];
        int idx = upper_bound(ends.begin(), ends.end(), s) - ends.begin();
        dp[i] = max(dp[i - 1], dp[idx] + w);
    }
    return dp[n];
}

int main() {
    vector<array<int,3>> jobs = {{1,3,50},{3,5,20},{6,19,100},{2,100,200}};
    cout << weightedActivitySelection(jobs) << "\n";
    return 0;
}
```

这节非常重要：**同一个题型，目标函数一变，贪心可能立刻失效。**

---

## 27.3 Huffman 编码（Huffman Coding）

### 27.3.1 最优前缀码

目标：给每个字符分配二进制码，满足：

1. **前缀码（Prefix Code）**：任何一个码字不是另一个码字前缀（保证可无歧义解码）；
2. **加权路径长度最小**（平均码长最短）。

设字符频率为 $f_i$，码长为 $l_i$，目标最小化：

$$
\sum_i f_i l_i
$$

---

### 27.3.2 Huffman 贪心构建

贪心策略：每次取频率最小的两个节点合并。

流程：

1. 所有字符作为叶子入最小堆；
2. 重复 $n-1$ 次：
   - 取两个最小频率节点 $x,y$；
   - 新建父节点 $z$，频率 $f_z=f_x+f_y$；
   - 把 $z$ 放回堆；
3. 剩下的根即 Huffman 树。

复杂度：$O(n\log n)$。

---

### 27.3.3 正确性引理一

在某棵最优前缀树中，频率最低的两个字符可以安排为最深层兄弟节点。

直觉：

- 深层节点码长更长；
- 低频字符放更深“惩罚更小”；
- 通过交换高频与低频叶子可不变差。

---

### 27.3.4 正确性引理二（归纳）

把两个最低频字符 $x,y$ 合并成一个伪字符 $z$，频率 $f_z=f_x+f_y$。

若在缩小后的问题上得到最优树，再把 $z$ 拆回 $x,y$，即可得到原问题最优树。

这说明“先合并最低频两项”是安全的。

这就是贪心选择性质 + 最优子结构。

<div data-component="HuffmanTreeBuilder"></div>

---

### 27.3.5 Python / C++ 实现

#### Python 实现

```python
import heapq
from collections import Counter
from typing import Dict, Optional


class Node:
    def __init__(self, freq: int, ch: Optional[str] = None, left=None, right=None):
        self.freq = freq
        self.ch = ch
        self.left = left
        self.right = right

    def __lt__(self, other):
        # heapq 需要可比较
        return self.freq < other.freq


def build_huffman_codes(text: str) -> Dict[str, str]:
    """
    返回字符 -> Huffman 编码

    Edge Cases:
    - 空串
    - 只有一种字符（编码通常约定为 '0'）
    """
    if not text:
        return {}

    freq = Counter(text)
    heap = [Node(f, ch) for ch, f in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        only = heap[0]
        return {only.ch: '0'}

    while len(heap) > 1:
        x = heapq.heappop(heap)
        y = heapq.heappop(heap)
        parent = Node(x.freq + y.freq, None, x, y)
        heapq.heappush(heap, parent)

    root = heap[0]
    codes: Dict[str, str] = {}

    def dfs(node: Node, path: str):
        if node.ch is not None:
            codes[node.ch] = path
            return
        dfs(node.left, path + '0')
        dfs(node.right, path + '1')

    dfs(root, '')
    return codes


s = "huffman greedy coding"
codes = build_huffman_codes(s)
print(codes)
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Node {
    int freq;
    char ch;
    Node* left;
    Node* right;
    Node(int f, char c='\0', Node* l=nullptr, Node* r=nullptr)
        : freq(f), ch(c), left(l), right(r) {}
};

struct Cmp {
    bool operator()(Node* a, Node* b) const {
        return a->freq > b->freq; // min-heap
    }
};

void buildCode(Node* root, const string& path, unordered_map<char,string>& codes) {
    if (!root) return;
    if (!root->left && !root->right) {
        // 叶子
        codes[root->ch] = path.empty() ? "0" : path;
        return;
    }
    buildCode(root->left, path + "0", codes);
    buildCode(root->right, path + "1", codes);
}

unordered_map<char,string> huffmanCodes(const string& s) {
    unordered_map<char,int> freq;
    for (char c : s) freq[c]++;

    priority_queue<Node*, vector<Node*>, Cmp> pq;
    for (auto& [c,f] : freq) pq.push(new Node(f, c));

    if (pq.empty()) return {};
    if (pq.size() == 1) {
        auto* only = pq.top();
        return {{only->ch, "0"}};
    }

    while (pq.size() > 1) {
        Node* x = pq.top(); pq.pop();
        Node* y = pq.top(); pq.pop();
        Node* p = new Node(x->freq + y->freq, '\0', x, y);
        pq.push(p);
    }

    Node* root = pq.top();
    unordered_map<char,string> codes;
    buildCode(root, "", codes);
    return codes;
}

int main() {
    string s = "huffman greedy coding";
    auto codes = huffmanCodes(s);
    for (auto& [c, code] : codes) {
        cout << c << " -> " << code << "\n";
    }
    return 0;
}
```

---

### 27.3.6 工业应用

- gzip / DEFLATE：LZ77 + Huffman
- JPEG：熵编码阶段常用 Huffman
- 网络传输、日志压缩、嵌入式传感数据压缩

注意：现代压缩器常结合多种技术，Huffman 不总是单独出现。

---

## 27.4 任务调度与区间问题

### 27.4.1 区间着色（最少机器数）

问题：给一堆区间任务，每个任务占用一台机器，互相重叠不能共用机器。求最少机器数。

结论：最少机器数 = 任意时刻的最大重叠区间数。

贪心实现：

- 按开始时间排序；
- 用最小堆维护“正在占用机器的任务结束时间”；
- 新任务开始时，若最早结束机器已空闲则复用，否则新增机器。

#### Python 实现

```python
import heapq
from typing import List, Tuple


def min_machines(intervals: List[Tuple[int, int]]) -> int:
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])  # 按开始时间
    heap = []  # 机器释放时间（结束时间）最小堆

    for s, e in intervals:
        if heap and heap[0] <= s:
            heapq.heappop(heap)  # 复用一台机器
        heapq.heappush(heap, e)

    return len(heap)


print(min_machines([(1,4),(2,5),(7,9),(3,6)]))  # 3
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

int minMachines(vector<pair<int,int>> intervals) {
    if (intervals.empty()) return 0;

    sort(intervals.begin(), intervals.end()); // 按开始时间

    priority_queue<int, vector<int>, greater<int>> pq; // 最早结束时间

    for (auto [s,e] : intervals) {
        if (!pq.empty() && pq.top() <= s) pq.pop();
        pq.push(e);
    }
    return (int)pq.size();
}

int main() {
    vector<pair<int,int>> arr = {{1,4},{2,5},{7,9},{3,6}};
    cout << minMachines(arr) << "\n"; // 3
    return 0;
}
```

<div data-component="IntervalColoringScheduler"></div>

---

### 27.4.2 EDF（Earliest Deadline First）

EDF 是实时调度中经典贪心：优先处理截止时间最早的任务。

在某些模型下（可抢占、单机等），EDF 具有可证明最优性；在离散非抢占场景中，也常作为高质量启发式。

这里给一个“最大按时完成任务数”版本（LeetCode #630 思路）：

- 按 deadline 排序；
- 累加已选任务时长；
- 若超时，删除已选中耗时最长任务（最大堆）。

#### Python 实现

```python
import heapq
from typing import List, Tuple


def schedule_max_tasks(tasks: List[Tuple[int, int]]) -> int:
    """
    tasks: (duration, deadline)
    返回最多能按时完成的任务数
    """
    tasks.sort(key=lambda x: x[1])
    total = 0
    max_heap = []  # Python 用负号模拟最大堆

    for dur, ddl in tasks:
        total += dur
        heapq.heappush(max_heap, -dur)
        if total > ddl:
            removed = -heapq.heappop(max_heap)
            total -= removed

    return len(max_heap)


print(schedule_max_tasks([(100,200),(200,1300),(1000,1250),(2000,3200)]))
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

int scheduleMaxTasks(vector<pair<int,int>> tasks) {
    // tasks: {duration, deadline}
    sort(tasks.begin(), tasks.end(), [](auto& a, auto& b){
        return a.second < b.second;
    });

    long long total = 0;
    priority_queue<int> maxHeap; // 已选任务中的最长时长

    for (auto [dur, ddl] : tasks) {
        total += dur;
        maxHeap.push(dur);
        if (total > ddl) {
            total -= maxHeap.top();
            maxHeap.pop();
        }
    }
    return (int)maxHeap.size();
}

int main() {
    vector<pair<int,int>> tasks = {{100,200},{200,1300},{1000,1250},{2000,3200}};
    cout << scheduleMaxTasks(tasks) << "\n";
    return 0;
}
```

---

### 27.4.3 分数背包（Fractional Knapsack）

和 0/1 背包不同，分数背包允许“切分物品”。

贪心策略：按单位价值 $v_i/w_i$ 从大到小装。

因为可切分，局部最优选择能直接扩展为全局最优。

#### Python 实现

```python
from typing import List, Tuple


def fractional_knapsack(items: List[Tuple[int, int]], capacity: int) -> float:
    """
    items: (value, weight)
    可分割背包最大价值
    """
    items = sorted(items, key=lambda x: x[0] / x[1], reverse=True)
    total_value = 0.0
    remain = capacity

    for v, w in items:
        if remain == 0:
            break
        if w <= remain:
            total_value += v
            remain -= w
        else:
            frac = remain / w
            total_value += v * frac
            remain = 0

    return total_value


print(fractional_knapsack([(60,10),(100,20),(120,30)], 50))  # 240.0
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

double fractionalKnapsack(vector<pair<int,int>> items, int capacity) {
    // items: {value, weight}
    sort(items.begin(), items.end(), [](auto& a, auto& b){
        return (double)a.first / a.second > (double)b.first / b.second;
    });

    double ans = 0.0;
    int remain = capacity;

    for (auto [v,w] : items) {
        if (remain == 0) break;
        if (w <= remain) {
            ans += v;
            remain -= w;
        } else {
            ans += (double)v * remain / w;
            remain = 0;
        }
    }
    return ans;
}

int main() {
    vector<pair<int,int>> items = {{60,10},{100,20},{120,30}};
    cout << fixed << setprecision(1) << fractionalKnapsack(items, 50) << "\n";
    return 0;
}
```

<div data-component="FractionalKnapsackPicker"></div>

---

### 27.4.4 跳跃游戏（Jump Game, #55）

问题：数组 `nums[i]` 表示从位置 i 最多跳多远，问能否到达末尾。

贪心维护：

- 当前能到达的最远下标 `far`；
- 逐个扫描 i，若 `i > far` 说明断层，失败；
- 否则更新 `far = max(far, i + nums[i])`。

#### Python 实现

```python
from typing import List


def can_jump(nums: List[int]) -> bool:
    far = 0
    for i, step in enumerate(nums):
        if i > far:
            return False
        far = max(far, i + step)
    return True


print(can_jump([2,3,1,1,4]))  # True
print(can_jump([3,2,1,0,4]))  # False
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

bool canJump(const vector<int>& nums) {
    int far = 0;
    for (int i = 0; i < (int)nums.size(); ++i) {
        if (i > far) return false;
        far = max(far, i + nums[i]);
    }
    return true;
}

int main() {
    cout << boolalpha << canJump({2,3,1,1,4}) << "\n";
    cout << boolalpha << canJump({3,2,1,0,4}) << "\n";
    return 0;
}
```

---

### 27.4.5 Gas Station（#134）

问题：环形加油站，`gas[i]` 提供油量，`cost[i]` 去下一站消耗，求可行起点。

贪心关键观察：

- 若从起点 `start` 出发在 `i` 失败（油量为负），
- 则 `start..i` 中任意一点都不可能作为可行起点，
- 因此可以直接把起点跳到 `i+1`。

#### Python 实现

```python
from typing import List


def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    if sum(gas) < sum(cost):
        return -1

    start = 0
    tank = 0

    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0

    return start


print(can_complete_circuit([1,2,3,4,5], [3,4,5,1,2]))  # 3
```

#### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int totalGas = accumulate(gas.begin(), gas.end(), 0);
    int totalCost = accumulate(cost.begin(), cost.end(), 0);
    if (totalGas < totalCost) return -1;

    int start = 0, tank = 0;
    for (int i = 0; i < (int)gas.size(); ++i) {
        tank += gas[i] - cost[i];
        if (tank < 0) {
            start = i + 1;
            tank = 0;
        }
    }
    return start;
}

int main() {
    vector<int> gas = {1,2,3,4,5};
    vector<int> cost = {3,4,5,1,2};
    cout << canCompleteCircuit(gas, cost) << "\n"; // 3
    return 0;
}
```

<div data-component="GasStationGreedyScan"></div>

---

## 27.5 Dijkstra 与 Prim 的贪心本质

### 27.5.1 Dijkstra 的贪心选择性质

Dijkstra 每一步选当前未确定节点中距离源点最小者 $u$，并“锁定”它。

贪心正确性（非负边权前提）核心是：

- 若存在更短路径到 $u$，该路径必先经过某个未确定节点；
- 但未确定节点的暂定距离不可能比 $u$ 更小（与选取规则矛盾）；
- 所以 $u$ 一旦被取出就不会再被改进。

这就是最短路径中的安全选择。

---

### 27.5.2 Prim 的切割性质（Cut Property）

对任意割 $(S, V\setminus S)$，横跨该割的最小权边一定属于某棵 MST。

Prim 每次从“当前树”和“外部节点”之间选最小边，本质就是不断应用 cut property。

这也是一个典型贪心安全边选择。

---

### 27.5.3 Kruskal 的环性质（Cycle Property）

对任意环，环上权重最大的边不可能出现在任意 MST 中。

Kruskal 按边权从小到大选边，遇到成环就跳过，实质是在利用这个性质避免“坏边”。

> 所以 Dijkstra / Prim / Kruskal 虽然题型不同，底层都是“局部安全选择 + 全局最优证明”的贪心范式。

<div data-component="CutPropertyVisualizer"></div>

---

## 27.6 贪心失效案例与判断方法

### 27.6.1 0/1 背包：贪心失效

0/1 背包不能分割物品，按单位价值排序未必最优。

反例：容量 50，物品

- (value=60, weight=10)
- (value=100, weight=20)
- (value=120, weight=30)

按单位价值贪心会选前两件，总价值 160，剩余容量 20 放不下第三件。

但最优是选后两件：100 + 120 = 220。

这说明“可分割”与“不可分割”是贪心可行性的分水岭。

---

### 27.6.2 找零反例：[1,3,4] 找 6

贪心策略“每次选最大面额”：

- 6 -> 4 + 1 + 1（3 枚）

但最优是：

- 6 -> 3 + 3（2 枚）

所以贪心并不总适用于任意硬币系统。

#### Python：DP 求最少硬币数（对比贪心）

```python
from typing import List


def coin_change_dp(coins: List[int], amount: int) -> int:
    INF = 10**9
    dp = [INF] * (amount + 1)
    dp[0] = 0

    for x in range(1, amount + 1):
        for c in coins:
            if x - c >= 0:
                dp[x] = min(dp[x], dp[x - c] + 1)

    return dp[amount] if dp[amount] < INF else -1


print(coin_change_dp([1,3,4], 6))  # 2
```

#### C++：DP 求最少硬币数

```cpp
#include <bits/stdc++.h>
using namespace std;

int coinChangeDP(vector<int>& coins, int amount) {
    const int INF = 1e9;
    vector<int> dp(amount + 1, INF);
    dp[0] = 0;

    for (int x = 1; x <= amount; ++x) {
        for (int c : coins) {
            if (x - c >= 0) {
                dp[x] = min(dp[x], dp[x - c] + 1);
            }
        }
    }
    return dp[amount] >= INF ? -1 : dp[amount];
}

int main() {
    vector<int> coins = {1,3,4};
    cout << coinChangeDP(coins, 6) << "\n"; // 2
    return 0;
}
```

---

### 27.6.3 判断贪心是否可行：实战检查清单

可以用下面这个清单快速判断：

1. 能否定义一个“局部最好”的决策标准？
2. 能否用交换论证证明：若最优解没这么选，可以交换成这么选且不变差？
3. 选择一步后，剩余问题是否是同类子问题？
4. 问题是否缺乏重叠子问题（否则 DP 更稳）？
5. 能否构造一个小反例打破该贪心？

如果第 2 条证明不出来，通常不要硬上贪心。

<div data-component="GreedyCounterexampleLab"></div>

---

## 27.7 本章复杂度对比与总结

| 问题 | 策略 | 时间复杂度 | 空间复杂度 | 备注 |
|---|---|---|---|---|
| 活动选择（无权） | 结束时间最早优先 | $O(n\log n)$ | $O(1)\sim O(n)$ | 贪心正确 |
| 带权活动选择 | DP + 二分 | $O(n\log n)$ | $O(n)$ | 贪心一般失效 |
| Huffman 编码 | 最小堆反复合并最小频率 | $O(n\log n)$ | $O(n)$ | 最优前缀码 |
| 区间着色（最少机器） | 最小堆维护释放时间 | $O(n\log n)$ | $O(n)$ | 贪心正确 |
| 分数背包 | 按价值密度排序 | $O(n\log n)$ | $O(1)\sim O(n)$ | 可分割是关键 |
| Jump Game | 维护最远可达 | $O(n)$ | $O(1)$ | 贪心正确 |
| Gas Station | 环形单扫重置起点 | $O(n)$ | $O(1)$ | 贪心正确 |
| 0/1 背包 | DP | 典型 $O(nW)$ | $O(W)$ | 贪心失效 |
| 找零（一般币制） | DP | $O(amount\cdot n)$ | $O(amount)$ | 贪心不总对 |

**本章核心主线**：

- 贪心不是“猜”，而是“可证明的局部安全选择”；
- 交换论证是贪心证明主力；
- 同一题型目标变化后，贪心可能立即失效；
- Dijkstra / Prim / Kruskal 都是贪心本质；
- 一旦交换论证走不通，要果断考虑 DP。

---

## 本章常见错误与调试技巧

> ⚠️ **把“看起来合理”当成“可证明正确”**：贪心最怕直觉正确但证明错误。先找反例，再写代码。

> ⚠️ **活动选择排序字段写错**：必须按结束时间排序，不是开始时间。

> ⚠️ **Huffman 比较器写反**：最小堆写成最大堆会导致完全错误编码树。

> ⚠️ **Jump Game 忘记断层判断**：若 `i > far` 应立即返回 False。

> ⚠️ **Gas Station 忘记总量先验**：若 `sum(gas) < sum(cost)`，必无解，直接 -1。

> ⚠️ **分数背包与 0/1 背包混淆**：前者可切割可贪心，后者不可切割通常要 DP。

---

## 面试高频问题

1. 为什么活动选择选“结束最早”而不是“开始最早”？
2. 如何用交换论证证明一个贪心策略正确？
3. 为什么分数背包可贪心，而 0/1 背包不行？
4. Dijkstra 的贪心正确性依赖什么前提？
5. Prim / Kruskal 分别对应哪条图论性质？
6. 找零问题什么时候贪心成立，什么时候失败？

---

## 练习与思考题

### 基础题

1. 实现无权活动选择，并输出具体活动集合。
2. 用最小堆实现区间着色，返回最少机器数。
3. 实现 Huffman 编码并计算平均码长。
4. 实现 Jump Game 与 Gas Station 两个线性贪心题。

### 进阶题

1. 给定一个贪心策略，先写反例再写证明（或证明失败）。
2. 对比带权活动选择的贪心与 DP，给出失败样例。
3. 构造一个硬币系统使“最大面额优先找零”失效。
4. 证明：若边权允许负数，Dijkstra 的贪心可能失效。

---

## 扩展阅读

- **CLRS** 第 16 章（Greedy Algorithms）
- **Kleinberg & Tardos**：贪心算法与交换论证
- **CP-Algorithms**：Huffman、区间调度、MST 性质
- LeetCode：
  - #55 Jump Game
  - #134 Gas Station
  - #435 Non-overlapping Intervals
  - #452 Minimum Number of Arrows to Burst Balloons
  - #630 Course Schedule III
  - #1235 Maximum Profit in Job Scheduling（DP）

下一章将进入**动态规划（Dynamic Programming）**，你会更清晰地看到：

- 贪心：一步一步做安全选择；
- DP：显式保存状态，系统枚举最优转移。
