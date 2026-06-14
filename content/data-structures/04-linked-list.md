# Chapter 4: 链表（Linked List）

> **前置知识**：Chapter 1 复杂度分析、Chapter 3 数组基础（用于对比）
>
> **难度**：🟢 初级
>
> **核心目标**：掌握单/双向链表的指针操作细节，用快慢指针解决环检测、找中点等经典问题，深刻理解链表与数组在缓存局部性上的根本差异。

---

## 4.1 单向链表

### 4.1.1 节点结构与内存散布模型

数组在内存中是**连续**的一整块，而链表的节点**散布在堆（heap）的任意位置**，每个节点通过 `next` 指针串联。

```python
# Python 单向链表节点
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 构建链表 1 → 2 → 3
head = ListNode(1, ListNode(2, ListNode(3)))
```

```cpp
// C++ 单向链表节点
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int v = 0, ListNode* n = nullptr) : val(v), next(n) {}
};

// 构建链表 1 → 2 → 3
ListNode* head = new ListNode(1, new ListNode(2, new ListNode(3)));
```

**内存模型**：

```
栈帧：head → [addr:0x1000]
堆：
  0x1000: [val=1 | next=0x2080]
  0x2080: [val=2 | next=0x3F40]
  0x3F40: [val=3 | next=nullptr]
```

节点地址不连续，这是链表**不支持随机访问**的根本原因。

### 4.1.2 插入、删除、查找及其复杂度

| 操作 | 条件 | 时间复杂度 | 说明 |
|------|------|-----------|------|
| 头部插入 | — | $O(1)$ | 新节点 next = head，head = 新节点 |
| 尾部插入 | 无 tail 指针 | $O(n)$ | 需遍历到末尾 |
| 尾部插入 | 有 tail 指针 | $O(1)$ | 直接 tail.next = 新节点 |
| 中间插入 | 已知前驱 | $O(1)$ | prev.next 重指 |
| 中间插入 | 需找前驱 | $O(n)$ | 先遍历 |
| 删除头节点 | — | $O(1)$ | head = head.next |
| 删除任意节点 | 已知前驱 | $O(1)$ | prev.next = curr.next |
| 删除任意节点 | 无前驱 | $O(n)$ | 先遍历找前驱 |
| 查找元素 | — | $O(n)$ | 顺序遍历 |

```python
# 链表遍历模板：O(n) 逐节点访问
def traverse(head: ListNode):
    curr = head
    while curr:
        print(curr.val)   # 处理当前节点
        curr = curr.next  # 向后推进，直到 None

# 头部插入 O(1)：新节点指向原 head，再更新 head
def insert_head(head: ListNode, val: int) -> ListNode:
    new_node = ListNode(val)
    new_node.next = head   # ① 新节点的 next = 旧头
    return new_node        # ② 返回新头（调用方更新 head）

# 在已知节点 prev 之后插入 O(1)：只需修改两根指针
def insert_after(prev: ListNode, val: int):
    new_node = ListNode(val)
    new_node.next = prev.next  # ① 先保存 prev 原后继（必须先做！）
    prev.next = new_node       # ② 再让 prev 指向新节点

# 删除 prev 的后继节点 O(1)：跳过该节点即可
def delete_after(prev: ListNode):
    if prev.next:              # 确保后继存在
        prev.next = prev.next.next  # 跨过目标节点

# 按值查找节点 O(n)：顺序遍历
def find(head: ListNode, val: int) -> ListNode:
    curr = head
    while curr:
        if curr.val == val:
            return curr
        curr = curr.next
    return None   # 未找到

# 尾部插入（维护 tail 指针时 O(1)，否则需先遍历 O(n)）
def insert_tail(tail: ListNode, val: int) -> ListNode:
    new_node = ListNode(val)
    tail.next = new_node   # 当前尾节点指向新节点
    return new_node        # 新节点成为新的 tail
```

```cpp
// C++ 链表基本操作
void traverse(ListNode* head) {
    for (ListNode* curr = head; curr; curr = curr->next)
        std::cout << curr->val << " ";
}

// 头部插入 O(1)
ListNode* insertHead(ListNode* head, int val) {
    ListNode* node = new ListNode(val);
    node->next = head;   // ① 新节点指向旧头
    return node;         // ② 返回新头（调用方更新 head）
}

// 在 prev 之后插入 O(1)（顺序非常重要：先设 new->next，再改 prev->next）
void insertAfter(ListNode* prev, int val) {
    ListNode* node = new ListNode(val);
    node->next = prev->next;  // ① 保存 prev 的原后继
    prev->next = node;        // ② prev 指向新节点
}

// 删除 prev 的后继节点 O(1)
void deleteAfter(ListNode* prev) {
    if (!prev->next) return;
    ListNode* target = prev->next;
    prev->next = target->next;
    delete target;   // C++ 需手动释放内存
}

// 按值查找 O(n)
ListNode* findByVal(ListNode* head, int val) {
    for (ListNode* curr = head; curr; curr = curr->next)
        if (curr->val == val) return curr;
    return nullptr;
}
```

> ⚠️ **常见错误**：`insert_after` 必须先令 `new_node.next = prev.next`，再令 `prev.next = new_node`。若顺序颠倒，`prev.next` 已被覆盖，原后继节点的地址丢失，链表断裂。

### 4.1.3 哨兵节点（Dummy Head）

处理链表头部操作时，常需要单独判断 `head` 是否为 `None`。引入**哨兵节点**（也叫虚节点/dummy head）可消除这一边界情况：

哨兵节点本身不存储有效数据，只借用其 `next` 指针充当稳定的「前驱」——这样链表任意位置（包括原 head）都可以用统一的"删除 prev.next"逻辑处理。

```python
# 无哨兵版：删除所有值为 val 的节点，需特殊判断 head 是否要删
def remove_elements_no_dummy(head: ListNode, val: int) -> ListNode:
    # 先处理 head 本身可能需要删除的情况
    while head and head.val == val:
        head = head.next
    # 再处理后续节点
    curr = head
    while curr and curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next  # 跳过待删节点
        else:
            curr = curr.next
    return head

# 哨兵版：dummy.next 充当稳定前驱，所有节点统一处理，无需特判
def remove_elements(head: ListNode, val: int) -> ListNode:
    dummy = ListNode(-1)      # 哨兵（val=-1 无意义）
    dummy.next = head
    curr = dummy
    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next  # 统一逻辑
        else:
            curr = curr.next
    return dummy.next         # dummy.next 即为新 head
```

```cpp
// C++ 有哨兵版（注意 C++ 中哨兵在栈上分配，自动回收）
ListNode* removeElements(ListNode* head, int val) {
    ListNode dummy(0, head);   // 栈上哨兵
    ListNode* curr = &dummy;
    while (curr->next) {
        if (curr->next->val == val) {
            ListNode* del = curr->next;
            curr->next = del->next;
            delete del;        // 手动释放
        } else {
            curr = curr->next;
        }
    }
    return dummy.next;
}
```

> 💡 **原则**：凡是头部可能被删除或修改的链表操作，优先引入 dummy head，避免分类讨论。哨兵节点的值无关紧要，关键是它的 `next` 指针能稳定充当所有节点的「前驱入口」。

### 4.1.4 头插法 vs 尾插法

两种建链方式在总体时间上均为 $O(n)$，但产生的链表**顺序相反**：

- **头插法**：每次把新节点插到最前面，最终链表与输入序列**逆序**。时间 $O(1)$/次，无需维护额外指针。常用于快速构造逆序结构（如浏览器历史栈）。
- **尾插法**：每次追加到末尾，借助 `tail` 指针维护尾部，最终保留**原序**。时间 $O(1)$/次（需 dummy + tail），无尾指针时退化为 $O(n)$/次。

```python
# 头插法构建链表（结果逆序）
def build_by_head(vals):
    head = None
    for v in vals:
        node = ListNode(v)
        node.next = head    # 新节点链接到旧头
        head = node         # head 更新为新节点
    return head             # 输入 [1,2,3] → 链表 3→2→1

# 尾插法构建链表（结果保留原序）
def build_by_tail(vals):
    dummy = ListNode()      # 哨兵节点，val 无意义
    tail = dummy
    for v in vals:
        tail.next = ListNode(v)
        tail = tail.next    # tail 始终指向最后一个真实节点
    return dummy.next       # 输入 [1,2,3] → 链表 1→2→3
```

```cpp
// 头插法（逆序）
ListNode* buildByHead(const std::vector<int>& vals) {
    ListNode* head = nullptr;
    for (int v : vals) {
        ListNode* node = new ListNode(v);
        node->next = head;  // 新节点链接旧头
        head = node;        // 更新头指针
    }
    return head;  // [1,2,3] → 3→2→1
}

// 尾插法（保留原序）：使用 dummy + tail 维护尾部指针
ListNode* buildByTail(const std::vector<int>& vals) {
    ListNode dummy;
    ListNode* tail = &dummy;
    for (int v : vals) {
        tail->next = new ListNode(v);
        tail = tail->next;  // 移动尾指针
    }
    return dummy.next;  // [1,2,3] → 1→2→3
}
```

> 💡 **应用场景**：头插法常用于实现与输入顺序相反的结构（如栈、反转操作）；尾插法用于「合并有序链表」、「复制带随机指针的链表」等需要保持原序的场景。

---

## 4.2 双向链表

### 4.2.1 节点结构

```python
class DListNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
```

```cpp
struct DListNode {
    int key, val;
    DListNode* prev;
    DListNode* next;
    DListNode(int k=0, int v=0): key(k), val(v), prev(nullptr), next(nullptr) {}
};
```

### 4.2.2 O(1) 删除已知节点

单向链表删除一个节点，必须先找到它的**前驱节点**才能修改 `prev.next`——若没有外部途径获得前驱，需从 head 遍历，最坏 $O(n)$。

双向链表每个节点的 `prev` 字段就是前驱地址，调用方传入 node 指针即可 $O(1)$ 完成删除，无需知道 head：

```
删除节点 B（A ↔ B ↔ C）：
  A.next = C    （绕过 B 向前）
  C.prev = A    （绕过 B 向后）
```

引入哨兵头尾节点后，`node.prev` 和 `node.next` 必然非空（哨兵充当边界），代码更简洁：

```python
def remove_node(node: DListNode):
    """O(1) 删除双向链表中任意节点（使用哨兵头尾可省略 None 判断）"""
    node.prev.next = node.next   # 前驱绕过 node
    node.next.prev = node.prev   # 后继的 prev 也更新
    # Python GC 自动回收 node
```

```cpp
// C++ 双向链表 O(1) 删除节点（配合哨兵使用，prev/next 必然非空）
void removeNode(DListNode* node) {
    node->prev->next = node->next;  // 前驱跳过 node
    node->next->prev = node->prev;  // 后继更新 prev
    delete node;                    // C++ 手动释放
}
```

> 对比：删除单向链表中的节点 B（`A→B→C`），需先找到 A（$O(n)$），然后 `A.next = C`。双向链表直接通过 `B.prev` 找到 A，$O(1)$ 完成。

### 4.2.3 双向哨兵头尾技巧

在双向链表两端各放一个**哨兵节点**（dummy head 和 dummy tail），使得任意真实节点的 `prev` 和 `next` 始终非空，彻底消除边界判断：

```
哨兵头 ↔ node1 ↔ node2 ↔ ... ↔ nodeN ↔ 哨兵尾
```

`add_to_front` 的四步指针操作顺序（设当前第一个真实节点为 `first`）：
1. `node.next = first`（新节点指向原 first）
2. `node.prev = 哨兵头`（新节点回指哨兵头）
3. `first.prev = node`（原 first 的 prev 改为新节点）
4. `哨兵头.next = node`（哨兵头前进指向新节点）

步骤 3、4 必须在 1、2 之后执行，否则 `first` 的旧地址会被覆盖而丢失。

```python
class DoublyLinkedList:
    def __init__(self):
        self.head = DListNode()   # 哨兵头（val 无意义）
        self.tail = DListNode()   # 哨兵尾（val 无意义）
        self.head.next = self.tail
        self.tail.prev = self.head
        # 初始状态：head ↔ tail（链表为空）

    def add_to_front(self, node: DListNode):
        """在哨兵头后插入（链表最前端），O(1)"""
        first = self.head.next
        node.next = first          # ① new → 原first
        node.prev = self.head      # ② new ← 哨兵头
        first.prev = node          # ③ 原first ← new
        self.head.next = node      # ④ 哨兵头 → new

    def remove(self, node: DListNode):
        """O(1) 删除任意节点（哨兵保证 prev/next 必然非空）"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def remove_last(self) -> DListNode:
        """移除并返回最后一个真实节点（哨兵尾的前驱），O(1)"""
        if self.tail.prev is self.head:
            return None            # 链表为空
        last = self.tail.prev
        self.remove(last)
        return last

    def is_empty(self) -> bool:
        return self.head.next is self.tail
```

```cpp
// C++ 双向链表（使用哨兵头尾）
struct DoublyLinkedList {
    DListNode* head;
    DListNode* tail;

    DoublyLinkedList() {
        head = new DListNode();
        tail = new DListNode();
        head->next = tail;
        tail->prev = head;
    }

    // 在链表最前端插入 O(1)
    void addToFront(DListNode* node) {
        DListNode* first = head->next;
        node->next = first;    // ① new → 原first
        node->prev = head;     // ② new ← 哨兵头
        first->prev = node;    // ③ 原first ← new
        head->next = node;     // ④ 哨兵头 → new
    }

    // O(1) 删除任意节点
    void remove(DListNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        // 调用方负责决定是否 delete node
    }

    // 移除并返回最尾真实节点 O(1)
    DListNode* removeLast() {
        if (tail->prev == head) return nullptr;  // 空链表
        DListNode* last = tail->prev;
        remove(last);
        return last;
    }

    bool isEmpty() { return head->next == tail; }
};
```

### 4.2.4 LRU Cache = 双向链表 + 哈希表

**LRU（Least Recently Used，最近最少使用）** 缓存淘汰策略：
- `get(key)`：存在则返回值并将节点移到链表最前（最近使用）
- `put(key, value)`：存在则更新并移最前；不存在则新建放最前，容量满时淘汰链表尾部节点

**关键洞察**：需要 $O(1)$ 的"访问节点"和"删除/移动节点"——哈希表提供前者，双向链表提供后者。

<div data-component="LRUCacheSim"></div>

```python
from collections import OrderedDict

# 方法一：借助 Python OrderedDict（面试不推荐，说明原理更重要）
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)   # 标记最近使用
        return self.cache[key]

    def put(self, key: int, value: int):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)  # 淘汰最旧（front）

# 方法二：手写双向链表 + 哈希表（推荐，展示原理）
class LRUCache2:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}                       # key → DListNode
        self.dll = DoublyLinkedList()       # 最近使用在头，最旧在尾

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        node = self.map[key]
        self.dll.remove(node)
        self.dll.add_to_front(node)         # 移到最前
        return node.val

    def put(self, key: int, value: int):
        if key in self.map:
            node = self.map[key]
            node.val = value
            self.dll.remove(node)
            self.dll.add_to_front(node)
        else:
            node = DListNode(key, value)
            self.map[key] = node
            self.dll.add_to_front(node)
            if len(self.map) > self.cap:
                last = self.dll.remove_last()   # 淘汰最旧
                del self.map[last.key]
```

```cpp
// C++ 手写 LRU（与上述 Python 思路完全等价）
class LRUCache {
    int cap;
    list<pair<int,int>> dll;              // {key, val}，front = 最近
    unordered_map<int, list<pair<int,int>>::iterator> map;
public:
    LRUCache(int capacity) : cap(capacity) {}

    int get(int key) {
        if (!map.count(key)) return -1;
        auto it = map[key];
        dll.splice(dll.begin(), dll, it); // O(1) 移到头部
        return it->second;
    }

    void put(int key, int value) {
        if (map.count(key)) {
            auto it = map[key];
            it->second = value;
            dll.splice(dll.begin(), dll, it);
        } else {
            dll.push_front({key, value});
            map[key] = dll.begin();
            if ((int)map.size() > cap) {
                map.erase(dll.back().first);
                dll.pop_back();
            }
        }
    }
};
```

**复杂度**：`get` 和 `put` 均为 $O(1)$（哈希查找 + 链表指针操作）。

---

## 4.3 循环链表

### 4.3.1 结构与遍历

循环链表的最后一个节点的 `next` 指向头节点（而非 `None`），形成闭环。与普通链表相比：

- **优点**：从任意节点出发均可访问全表；尾节点到头节点 $O(1)$（`tail.next = head`）；适合「轮转」类问题。
- **危险**：遍历时必须人为设置终止条件（记录起点或节点计数），否则**无限循环**。

常用两种遍历写法：
1. 记录起点，当再次到达起点时停止（`do-while` 风格）
2. 计数器（已知长度时用）

```python
# 循环链表遍历（从 start 出发绕一圈）
def traverse_circular(start: ListNode):
    if not start:
        return
    curr = start
    while True:
        print(curr.val)
        curr = curr.next
        if curr is start:   # 回到起点，终止（注意用 is，不是 ==）
            break

# 构建循环链表：最后一个节点的 next 指回 head
def build_circular(vals: list) -> ListNode:
    if not vals:
        return None
    nodes = [ListNode(v) for v in vals]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    nodes[-1].next = nodes[0]   # 尾 → 头，封口成环
    return nodes[0]

# 在循环链表尾部插入（维护 tail 指针时 O(1)）
def insert_after_tail(tail: ListNode, val: int) -> ListNode:
    new_node = ListNode(val)
    new_node.next = tail.next   # 新节点指向 head
    tail.next = new_node        # 原尾指向新节点
    return new_node             # 新节点成为新 tail
```

```cpp
// C++ 循环链表遍历（do-while 天然适合「至少执行一次」的环形结构）
void traverseCircular(ListNode* start) {
    if (!start) return;
    ListNode* curr = start;
    do {
        std::cout << curr->val << " ";
        curr = curr->next;
    } while (curr != start);  // 回到起点时终止
}

// 构建循环链表
ListNode* buildCircular(const std::vector<int>& vals) {
    if (vals.empty()) return nullptr;
    ListNode* head = new ListNode(vals[0]);
    ListNode* tail = head;
    for (size_t i = 1; i < vals.size(); i++) {
        tail->next = new ListNode(vals[i]);
        tail = tail->next;
    }
    tail->next = head;  // 尾 → 头，封闭环
    return head;
}
```

> ⚠️ **常见陷阱**：循环链表删除节点前，若只有一个节点（`head.next == head`），需单独处理，不能简单地将 `prev.next = curr.next`（可能导致整个环断掉或引用悬空）。

### 4.3.2 约瑟夫问题（Josephus Problem）

**问题描述**：$n$ 个人编号 $1\ldots n$ 围成一圈，从 1 号开始报数，每报到第 $m$ 个人就出圈，下一个人重新从 1 开始报数，求最后留下的人的编号。

**方法一：循环链表模拟** — 时间 $O(n \cdot m)$，空间 $O(n)$

每轮找目标用 $m-1$ 步，删除后继续。适合理解过程，但大 $n,m$ 时性能差。

**方法二：数学递推（Josephus 公式）** — 时间 $O(n)$，空间 $O(1)$

设 $f(n, m)$ 为 $n$ 人时最终幸存者的 **0-indexed 位置**，递推关系为：

$$f(1, m) = 0, \quad f(n, m) = \bigl(f(n-1, m) + m\bigr) \bmod n$$

**直觉**：每淘汰一个人后，剩余 $n-1$ 人重新编号（起始位置偏移 $m$），子问题的答案只需加上偏移量再取模就能映射回原始编号。

```python
# 方法一：循环链表模拟 O(n·m)
def josephus_list(n: int, m: int) -> int:
    nodes = [ListNode(i) for i in range(1, n + 1)]
    for i in range(n):
        nodes[i].next = nodes[(i + 1) % n]   # 构建循环链表
    curr = nodes[0]
    while curr.next != curr:   # 只剩一个人时停止
        # 向前走 m-1 步：curr 停在「将被淘汰者的前驱」
        for _ in range(m - 1):
            curr = curr.next
        curr.next = curr.next.next  # 删除第 m 个人
        curr = curr.next            # 从下一个人开始新一轮
    return curr.val

# 方法二：数学递推 O(n)（推荐）
# f(1,m)=0; f(n,m)=(f(n-1,m)+m)%n  →  1-indexed 答案 = f(n,m)+1
def josephus_math(n: int, m: int) -> int:
    pos = 0                       # f(1,m)=0（0-indexed，1人时盲目留下）
    for i in range(2, n + 1):     # 从 2 人逐步扩展到 n 人
        pos = (pos + m) % i
    return pos + 1                # 转为 1-indexed

# 验证：n=5, m=3 → 依次出圈 3,1,5,2，最后剩 4
# josephus_math(5, 3) → 4  ✓
```

```cpp
// C++ 数学递推（O(n) 时间，O(1) 空间）——面试首选
int josephusMath(int n, int m) {
    int pos = 0;                  // f(1,m) = 0（0-indexed）
    for (int i = 2; i <= n; i++)
        pos = (pos + m) % i;
    return pos + 1;               // 转为 1-indexed
}

// C++ 链表模拟（O(n·m) 仅用于理解过程）
int josephusList(int n, int m) {
    std::vector<ListNode*> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = new ListNode(i + 1);
    for (int i = 0; i < n; i++) nodes[i]->next = nodes[(i + 1) % n];
    ListNode* curr = nodes[0];
    while (curr->next != curr) {
        for (int i = 0; i < m - 1; i++) curr = curr->next;
        ListNode* del = curr->next;
        curr->next = del->next;
        delete del;
        curr = curr->next;
    }
    int ans = curr->val;
    delete curr;
    return ans;
}
```

| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| 链表模拟 | $O(n \cdot m)$ | $O(n)$ | 理解过程、$n,m$ 很小 |
| 数学递推 | $O(n)$ | $O(1)$ | 竞赛/面试首选 |

---

## 4.4 链表经典算法

<div data-component="LinkedListOperations"></div>

### 4.4.1 反转链表

**迭代三指针法**（$O(n)$ 时间，$O(1)$ 空间）：

```python
def reverse_list(head: ListNode) -> ListNode:
    prev, curr = None, head
    while curr:
        nxt = curr.next    # ① 先保存 next（关键！避免断链）
        curr.next = prev   # ② 反转指针
        prev = curr        # ③ prev 前进
        curr = nxt         # ④ curr 前进
    return prev

# 步骤图示（1→2→3→None 变成 3→2→1→None）：
# 初始：prev=None  curr=1  nxt=?
# 第1轮：nxt=2, 1→None, prev=1, curr=2
# 第2轮：nxt=3, 2→1,    prev=2, curr=3
# 第3轮：nxt=None, 3→2, prev=3, curr=None
# 返回 prev=3
```

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *curr = head;
    while (curr) {
        ListNode* nxt = curr->next;
        curr->next = prev;
        prev = curr;
        curr = nxt;
    }
    return prev;
}
```

**递归法**（$O(n)$ 时间，$O(n)$ 空间——递归栈）：

递归的核心思路：先递归反转 `head.next` 之后的所有节点，然后仅将 `head.next.next = head`（后继指回 head），最后断掉 `head.next = None`。

```python
def reverse_list_rec(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head                         # 基础情况：0 或 1 个节点
    new_head = reverse_list_rec(head.next)  # 后半已经反转，new_head 是新头
    head.next.next = head   # 后继（反转后变成当前的"尾"）指回 head
    head.next = None        # head 成为新链尾，cut 掉旧 next 防止环
    return new_head         # 返回反转后链表的头
```

```cpp
// C++ 递归反转
ListNode* reverseListRec(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* newHead = reverseListRec(head->next);
    head->next->next = head;  // 后继指回 head
    head->next = nullptr;     // head 成为新链尾
    return newHead;
}
```

<div data-component="LinkedListReversal"></div>

### 4.4.2 链表中点（快慢指针）

**核心思想**：快指针走 2 步，慢指针走 1 步，快指针到终点时慢指针恰好在中点。

存在两种常见的「中点」定义，对应两种循环条件：

| 循环条件 | 奇数长（5节点）慢指针停在 | 偶数长（4节点）慢指针停在 | 典型用途 |
|---------|----------------------|----------------------|---------|
| `while fast and fast.next` | 节点3（**正中**） | 节点3（**后半首个**）| 一般找中点 |
| `while fast.next and fast.next.next` | 节点2（**前半末**） | 节点2（**前半末**）| 回文检测/连接时断链 |

```python
# 版本一：fast 走到末尾（slow 停在「后半起点」或「正中」）
def find_middle_v1(head: ListNode) -> ListNode:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next        # 慢指针每次 1 步
        fast = fast.next.next   # 快指针每次 2 步
    return slow
    # 奇数长 1→2→3→4→5：slow 停在 3（中点）
    # 偶数长 1→2→3→4  ：slow 停在 3（后半第一个）

# 版本二：fast 提前一步停（slow 停在「前半末尾」，便于断链）
def find_middle_v2(head: ListNode) -> ListNode:
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow
    # 奇数长 1→2→3→4→5：slow 停在 2
    # 偶数长 1→2→3→4  ：slow 停在 2（前半末尾）
    # 用法：right = slow.next; slow.next = None  断开前后两半
```

```cpp
// C++ 版本一：slow 停在后半首个节点（或正中）
ListNode* findMiddle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}

// C++ 版本二：slow 停在前半最后一个节点（适用于断链后分别操作两段）
ListNode* findMiddleForSplit(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    // slow->next 是后半链表的头
    return slow;
}
```

> 💡 **正确性**：设链表长 $n$，每轮 slow +1、fast +2。当 fast 抵达末尾时经过了 $\lfloor n/2 \rfloor$ 轮，slow 也走了 $\lfloor n/2 \rfloor$ 步，恰好处于中间位置。

### 4.4.3 Floyd 判环算法（数学证明）

**问题**：判断链表是否有环，若有，返回环的入口节点。

**思路**：慢指针每次走 1 步，快指针每次走 2 步。若有环，两者必然在环内相遇。

**数学证明（找环入口）**：

设：
- 头部到环入口距离 = $a$
- 环长 = $b$
- 相遇时慢指针在环内走了 $c$ 步

相遇时：
$$\text{快指针路程} = 2 \times \text{慢指针路程}$$
$$a + b \cdot k_1 + c = 2(a + c) \quad (k_1 \geq 1)$$
$$b \cdot k_1 = a + c$$
$$a = b \cdot k_1 - c$$

即：**从头部走 $a$ 步 = 从相遇点沿环方向再走 $b k_1 - c$ 步**。因为环长 $b$ 的倍数绕了整圈不改变位置，所以从相遇点继续走 $a$ 步就到达入口——这解释了为什么将快指针重置到头部、两者同步各走 1 步，相遇点即为环入口。

<div data-component="FloydCycleDetection"></div>

```python
def detect_cycle(head: ListNode) -> ListNode:
    slow = fast = head
    # 第一阶段：判断是否有环
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None          # 无环

    # 第二阶段：找环入口
    fast = head              # 快指针重置到头
    while slow != fast:
        slow = slow.next
        fast = fast.next     # 两者均走 1 步
    return slow              # 相遇点即为入口
```

```cpp
ListNode* detectCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            fast = head;
            while (slow != fast) { slow = slow->next; fast = fast->next; }
            return slow;
        }
    }
    return nullptr;
}
```

### 4.4.4 合并两个有序链表

```python
# 迭代版（O(m+n) 时间，O(1) 空间）
def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode()
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2    # 拼接剩余
    return dummy.next

# 递归版（O(m+n) 时间，O(m+n) 空间——调用栈）
def merge_two_lists_rec(l1: ListNode, l2: ListNode) -> ListNode:
    if not l1: return l2
    if not l2: return l1
    if l1.val <= l2.val:
        l1.next = merge_two_lists_rec(l1.next, l2)
        return l1
    else:
        l2.next = merge_two_lists_rec(l1, l2.next)
        return l2
```

```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy, *curr = &dummy;
    while (l1 && l2) {
        if (l1->val <= l2->val) { curr->next = l1; l1 = l1->next; }
        else                    { curr->next = l2; l2 = l2->next; }
        curr = curr->next;
    }
    curr->next = l1 ? l1 : l2;
    return dummy.next;
}
```

### 4.4.5 K 个一组反转链表（LeetCode #25）

将链表每 $k$ 个节点一组进行反转，最后不足 $k$ 个的保持原顺序。

**核心步骤**：
1. 用 `has_k_nodes` 检查剩余节点是否满 $k$ 个（不足则退出）
2. 组内用三指针法迭代反转 $k$ 个节点，**记录 `group_tail`**（反转前的「组头」就是反转后的「组尾」）
3. 重连四条关键链接：`group_prev → 新组头`，`新组尾 → 下一组头`
4. `group_prev` 移动到新组尾，准备下一轮

```python
def reverse_k_group(head: ListNode, k: int) -> ListNode:
    """每 k 个节点一组翻转，最后不足 k 个保持原样。O(n) 时间，O(1) 空间。"""
    def has_k_nodes(node, k):
        while node and k > 0:
            node = node.next
            k -= 1
        return k == 0

    dummy = ListNode(0, head)
    group_prev = dummy

    while has_k_nodes(group_prev.next, k):
        prev, curr = None, group_prev.next
        group_tail = curr  # 反转前的「组头」= 反转后的「组尾」
        for _ in range(k):
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        # 重连四条边：
        group_prev.next = prev      # ① group_prev → 新组头（prev）
        group_tail.next = curr      # ② 新组尾（原组头）→ 下一组
        group_prev = group_tail     # ③ 更新 group_prev

    return dummy.next

# 示例：[1,2,3,4,5], k=2 → [2,1,4,3,5]
# 示例：[1,2,3,4,5], k=3 → [3,2,1,4,5]
```

```cpp
// C++ K 个一组反转链表
ListNode* reverseKGroup(ListNode* head, int k) {
    auto hasKNodes = [&](ListNode* node, int cnt) -> bool {
        while (node && cnt > 0) { node = node->next; cnt--; }
        return cnt == 0;
    };

    ListNode dummy(0, head);
    ListNode* groupPrev = &dummy;

    while (hasKNodes(groupPrev->next, k)) {
        ListNode *prev = nullptr, *curr = groupPrev->next;
        ListNode* groupTail = curr;   // 记录组头（反转后变组尾）
        for (int i = 0; i < k; i++) {
            ListNode* nxt = curr->next;
            curr->next = prev;
            prev = curr;
            curr = nxt;
        }
        groupPrev->next = prev;       // 接入新组头
        groupTail->next = curr;       // 组尾连接下一组
        groupPrev = groupTail;
    }
    return dummy.next;
}
```

### 4.4.6 链表排序（归并排序）

对链表用归并排序：$O(n \log n)$ 时间，Top-down 版本 $O(\log n)$ 递归栈空间（Bottom-up 版可降至 $O(1)$）。

```python
def sort_list(head: ListNode) -> ListNode:
    """自顶向下归并排序。O(n log n) 时间，O(log n) 空间（递归栈）。"""
    if not head or not head.next:
        return head                # 基底：0 或 1 节点天然有序

    # 找中点（fast 从 head.next 出发，slow 停在前半末尾便于断链）
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None               # 断链：left=head...slow, right=mid...

    left  = sort_list(head)        # 递归排序左半
    right = sort_list(mid)         # 递归排序右半
    return merge_two_lists(left, right)
```

```cpp
// C++ 自顶向下归并排序
ListNode* sortList(ListNode* head) {
    if (!head || !head->next) return head;

    // 快慢指针找前半末尾（fast 从 head->next 出发）
    ListNode *slow = head, *fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    ListNode* mid = slow->next;
    slow->next = nullptr;          // 断链

    return mergeTwoLists(sortList(head), sortList(mid));
}

// Bottom-up 归并排序（O(1) 空间，避免递归栈）
ListNode* sortListBottomUp(ListNode* head) {
    if (!head || !head->next) return head;
    int len = 0;
    for (ListNode* p = head; p; p = p->next) len++;

    ListNode dummy(0, head);
    for (int sz = 1; sz < len; sz *= 2) {   // 子段长度从 1 倍增到 n
        ListNode* prev = &dummy;
        ListNode* curr = dummy.next;
        while (curr) {
            // 切出左段（sz 个节点）
            ListNode* left = curr;
            for (int i = 1; i < sz && curr->next; i++) curr = curr->next;
            ListNode* right = curr->next;
            curr->next = nullptr;
            // 切出右段（最多 sz 个节点）
            curr = right;
            for (int i = 1; i < sz && curr && curr->next; i++) curr = curr->next;
            ListNode* next = curr ? curr->next : nullptr;
            if (curr) curr->next = nullptr;
            // 合并左右段并接在 prev 后
            prev->next = mergeTwoLists(left, right);
            while (prev->next) prev = prev->next;
            curr = next;
        }
    }
    return dummy.next;
}
```

**为什么归并排序适合链表？**
- **快速排序**找基准需要随机访问（`a[(l+r)/2]`），链表定位需 $O(n)$，整体退化至 $O(n^2)$。
- **归并排序**只需顺序步进（快慢指针找中点，双指针合并），天然契合链表结构。
- Bottom-up 版本可将空间降至 $O(1)$，是链表排序的工业级写法（LeetCode #148 官方 follow-up）。

### 4.4.7 链表回文检测

**思路**：找中点 → 反转后半 → 双指针比较 → 还原后半（可选，面试需问清是否须保持结构）

以 `1→2→3→2→1` 为例：
1. 快慢指针找中点，slow 停在节点 `3`
2. 从 `3` 开始反转后半，得到 `1→2→3`（后半反转链）
3. 两指针从 head 和后半头分别出发，逐一比较
4. 可选：再次翻转恢复原链表结构

为何不复制到数组再比较？链表本身可以原地反转，$O(1)$ 额外空间，避免 $O(n)$ 的数组拷贝开销。

```python
def is_palindrome(head: ListNode) -> bool:
    if not head or not head.next:
        return True

    # 快慢指针找中点（slow 停在正中或后半第一个）
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # 反转后半链表
    def reverse(node):
        prev, curr = None, node
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        return prev

    second_half = reverse(slow)
    copy = second_half   # 保存以便还原

    # 双指针比较
    p1, p2 = head, second_half
    result = True
    while p2:
        if p1.val != p2.val:
            result = False
            break
        p1 = p1.next
        p2 = p2.next

    # 还原后半链表（恢复原链表结构）
    reverse(copy)

    return result
```

```cpp
// C++ 链表回文检测（O(n) 时间，O(1) 空间）
bool isPalindrome(ListNode* head) {
    if (!head || !head->next) return true;

    // 快慢指针找中点
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }

    // 反转后半
    auto reverseList = [](ListNode* node) -> ListNode* {
        ListNode* prev = nullptr;
        while (node) {
            ListNode* nxt = node->next;
            node->next = prev;
            prev = node;
            node = nxt;
        }
        return prev;
    };

    ListNode* secondHalf = reverseList(slow);
    ListNode* copy = secondHalf;

    // 对比
    ListNode *p1 = head, *p2 = secondHalf;
    bool result = true;
    while (p2) {
        if (p1->val != p2->val) { result = false; break; }
        p1 = p1->next;
        p2 = p2->next;
    }

    // 还原后半
    reverseList(copy);

    return result;
}
```

---

## 4.5 数组 vs 链表深度对比

### 4.5.1 操作复杂度对比表

| 操作 | 数组（连续内存）| 链表（散布内存）| 说明 |
|------|--------------|--------------|------|
| 随机访问 `a[i]` | $O(1)$ | $O(n)$ | 数组直接计算地址 |
| 头部插入 | $O(n)$ | $O(1)$ | 数组需整体右移 |
| 尾部插入（平摊） | $O(1)$ | $O(1)$（有tail时）| 动态数组扩容摊销 |
| 中间插入 | $O(n)$ | $O(1)$（已知位置）| 链表只改指针 |
| 删除头部 | $O(n)$ | $O(1)$ | 数组整体左移 |
| 删除中间 | $O(n)$ | $O(1)$（已知节点）| 链表只改指针 |
| 搜索（无序）| $O(n)$ | $O(n)$ | 都需顺序遍历 |
| 额外空间 | 无 | 每节点 1 个指针 | 单向链表 +1指针/节点 |

### 4.5.2 缓存局部性分析

<div data-component="LinkedListVsArrayBenchmark"></div>

**缓存行（Cache Line）** 通常为 64 字节，CPU 读取一个地址时会把周围的数据一起加载进 L1/L2 缓存。

**数组遍历**：
```
内存地址：[0x100, 0x104, 0x108, 0x10C, ...]  ← 连续
访问 a[0] 时，cache line 会一次加载 0x100~0x13F 共 16 个 int
之后 a[1]~a[15] 全部命中缓存 → ✅ 高缓存命中率
```

**链表遍历**：
```
0x100 → 0x2080 → 0x3F40 → 0x15C0 → ...  ← 随机跳跃
访问 node1 时 cache line 加载 0x100 附近
访问 node2（0x2080）时 cache miss，重新加载
→ ⚠️ 高 cache miss 率，实际速度可比数组慢 3~10×
```

**实测规律**（n = 10^7 元素）：

| 结构 | 顺序遍历耗时（参考）| Cache miss 率 |
|------|-----------------|--------------|
| `int[]` 数组 | ~20 ms | <1% |
| `LinkedList` | ~150 ms | ~40% |

### 4.5.3 工程选择建议

在实际工程中，选择数组还是链表，需要结合**访问模式**、**插删频率**和**内存约束**综合判断：

- **频繁随机访问**（如 `a[i]`）→ 数组/`vector`，$O(1)$ 下标访问，缓存友好。
- **频繁头部插入/删除**（如 LIFO 队列、滑动窗口）→ 双端队列（`collections.deque` / `std::deque`），或链表。
- **频繁中间插删且已持有节点指针**（如 LRU、跳跃表操作层）→ 双向链表，$O(1)$ 操作。
- **LRU / LFU 缓存**（同时需 $O(1)$ 查找 + $O(1)$ 删除）→ **双向链表 + 哈希表** 的经典组合。
- **实现栈**（LIFO）→ Python `list.append/pop` 或 C++ `std::stack`（底层数组），不需要链表。
- **实现队列**（FIFO）→ `collections.deque` 或 C++ `std::queue`（底层 deque），单链表也可但缓存不友好。
- **内存极度敏感**场景 → 优先数组，链表每个节点至少多存 1 个指针（8B），双向还多 1 个（16B 额外）。

```
[决策树]
访问模式是否需要随机访问（下标）？
  ├─ 是 → 数组/vector
  └─ 否 → 会频繁插删？
            ├─ 头/尾 → deque（兼顾两端 O(1)）
            └─ 中间且已知位置 → 双向链表
                                是否还需 O(1) 查找？
                                  └─ 是 → 链表 + 哈希表（LRU 模式）
```

---

## 4.6 经典题精讲

### 题1：反转链表 II（LeetCode #92）：指定区间 `[left, right]` 反转

**思路（头插法原地翻转）**：先走到 `left-1` 位置得到前驱 `pre`，然后对区间内节点反复将「`curr` 的后继」摘出并插到 `pre` 后面，循环 `right-left` 次，$O(n)$ 时间，$O(1)$ 空间。

以 `[1,2,3,4,5]`，`left=2`，`right=4` 为例：
- `pre=1, curr=2`
- 第1轮：取出 `3` 插到 `pre` 后 → `[1,3,2,4,5]`
- 第2轮：取出 `4` 插到 `pre` 后 → `[1,4,3,2,5]`
- 结果：`[1,4,3,2,5]` ✓

```python
def reverse_between(head: ListNode, left: int, right: int) -> ListNode:
    dummy = ListNode(0, head)
    pre = dummy
    for _ in range(left - 1):
        pre = pre.next                # 走到区间前驱
    curr = pre.next                   # curr 始终是「已翻转段」的末尾
    for _ in range(right - left):
        nxt = curr.next               # nxt 是下一个待挪节点
        curr.next = nxt.next          # 从链中摘出 nxt
        nxt.next = pre.next           # nxt 插到 pre 后面
        pre.next = nxt
    return dummy.next
```

```cpp
ListNode* reverseBetween(ListNode* head, int left, int right) {
    ListNode dummy(0, head);
    ListNode* pre = &dummy;
    for (int i = 0; i < left - 1; i++) pre = pre->next;
    ListNode* curr = pre->next;
    for (int i = 0; i < right - left; i++) {
        ListNode* nxt = curr->next;
        curr->next = nxt->next;
        nxt->next = pre->next;
        pre->next = nxt;
    }
    return dummy.next;
}
```

### 题2：删除链表倒数第 N 个节点（LeetCode #19）

**思路（快慢指针一次遍历）**：fast 先走 `n+1` 步（领先 slow `n+1` 位），然后两者同步向后走，直到 fast 为 None。此时 slow 恰好在待删节点的**前驱**，直接修改 `slow.next`。

为什么是 `n+1` 而不是 `n`？因为我们需要的是**前驱**，而不是待删节点本身。

```python
def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):    # fast 先走 n+1 步（从 dummy 出发）
        fast = fast.next
    while fast:               # 同步走，fast 到 None 时 slow 在前驱
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next  # 跳过待删节点
    return dummy.next
```

```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0, head);
    ListNode *fast = &dummy, *slow = &dummy;
    for (int i = 0; i <= n; i++) fast = fast->next;  // fast 走 n+1 步
    while (fast) { slow = slow->next; fast = fast->next; }
    ListNode* del = slow->next;
    slow->next = del->next;
    delete del;
    return dummy.next;
}
```

### 题3：子链表和等于 K（哈希前缀和）

**思路（前缀和 + 哈希表）**：与数组「子数组和为 K」等价。  
`prefix[j] - prefix[i] = k` ⟺ `prefix[i] = prefix[j] - k`  
用哈希表记录每个前缀和第一次出现的索引（或次数），遍历一次即 $O(n)$。

链表版本直接在遍历指针时累加前缀和，无需转为数组。

```python
def subarray_sum_k(head: ListNode, k: int) -> int:
    """返回节点值之和等于 k 的连续子链表个数。O(n) 时间，O(n) 空间。"""
    count = 0
    prefix = 0
    seen = {0: 1}   # 前缀和为 0 出现 1 次（代表空前缀）
    curr = head
    while curr:
        prefix += curr.val
        # 若 prefix - k 曾出现，则该差值对应的每个起点都能构成一段和为 k 的子链
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
        curr = curr.next
    return count
```

```cpp
int subarraySumK(ListNode* head, int k) {
    std::unordered_map<int, int> seen;
    seen[0] = 1;   // 空前缀
    int count = 0, prefix = 0;
    for (ListNode* curr = head; curr; curr = curr->next) {
        prefix += curr->val;
        auto it = seen.find(prefix - k);
        if (it != seen.end()) count += it->second;
        seen[prefix]++;
    }
    return count;
}
```

---

## 4.7 本章小结

| 知识点 | 核心结论 |
|--------|---------|
| 单向链表插入/删除（已知位置）| $O(1)$ |
| 链表随机访问 | $O(n)$ |
| 哨兵节点 | 消除头部边界特判，代码更简洁 |
| Floyd 判环 | 慢1步快2步，相遇后快指针重置头部，再次相遇即入口 |
| LRU Cache | 双向链表（最近到最旧）+ 哈希表，get/put 均 $O(1)$ |
| 链表归并排序 | $O(n\log n)$ 时间，适合链表的排序算法 |
| 缓存友好性 | 数组远优于链表，实测快 3~10 倍 |

**🎯 面试高频题单**：
- `#206` 反转链表（必会）
- `#141/#142` 判环/找入口（Floyd 完整推导）
- `#21` 合并有序链表
- `#146` LRU Cache（手写双向链表版，最高频设计题）
- `#25` K 个一组翻转（综合难）
- `#19` 删除倒数第 N 个
- `#234` 回文链表

**💡 思考题**：
1. 双向链表比单向链表多一个 `prev` 指针，使每个节点内存多 8 字节（64 位系统）。这个开销在 LRU Cache 中值得吗？试从"是否存在 $O(1)$ 单链表替代方案"角度分析。
2. `sort_list` 用 Top-down 归并需要 $O(\log n)$ 栈空间。如何改成 Bottom-up 归并使空间降到 $O(1)$？

> **参考资料**：CLRS 第4版 Chapter 10.2–10.3；Sedgewick 1.3；《算法4》1.3 节；LeetCode 题解 #146、#25
