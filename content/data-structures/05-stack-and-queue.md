# Chapter 5: 栈与队列（Stack & Queue）

> **前置知识**：Chapter 3 数组基础、Chapter 4 链表基础
>
> **难度**：🟢 初级
>
> **核心目标**：深刻理解 LIFO/FIFO 语义及其实现细节；掌握单调栈和单调队列解决"下一个更大元素"与"滑动窗口最值"；理解栈在系统底层（调用栈、表达式求值）中的核心地位。

---

## 5.1 栈（Stack）

### 5.1.1 LIFO 原则与抽象接口（push/pop/peek/isEmpty）

**栈（Stack）** 是一种遵循 **后进先出（LIFO, Last In First Out）** 原则的数据结构：最后压入的元素最先被弹出。

想象一摞盘子：只能从最顶层放入或取走盘子，不能从中间或底部操作。

**抽象接口（ADT）**：

| 操作 | 说明 | 时间复杂度 |
|------|------|-----------|
| `push(x)` | 将 x 压入栈顶 | $O(1)$ |
| `pop()` | 移除并返回栈顶元素 | $O(1)$ |
| `peek()` / `top()` | 查看栈顶元素（不移除）| $O(1)$ |
| `isEmpty()` | 判断栈是否为空 | $O(1)$ |
| `size()` | 返回元素个数 | $O(1)$ |

**注意**：`pop()` 和 `peek()` 在栈为空时均应抛出异常（underflow）。防御性编程中应始终先检查 `isEmpty()`。

LIFO 的自然应用场景：
- **函数调用栈**：最后调用的函数最先返回
- **括号匹配**：最后遇到的左括号需要最先被对应右括号闭合
- **撤销操作（Undo）**：最后执行的操作最先被撤销
- **DFS/回溯**：通过显式栈模拟递归

### 5.1.2 数组实现（顶部指针 top，O(1) push/pop）

最简单的栈实现：用**数组 + top 指针**记录栈顶位置。`top` 指向最后一个有效元素的索引，初始为 `-1`（空栈）。

```python
class ArrayStack:
    def __init__(self, capacity: int = 100):
        self._data = [None] * capacity  # 静态数组（固定容量）
        self._top = -1                  # -1 表示空栈

    def push(self, val) -> None:
        if self._top == len(self._data) - 1:
            raise OverflowError("Stack overflow")
        self._top += 1              # 先移动 top
        self._data[self._top] = val  # 再写入数据

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack underflow")
        val = self._data[self._top]
        self._top -= 1              # top 后退（数据可以不清零，被覆盖即可）
        return val

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack underflow")
        return self._data[self._top]

    def is_empty(self) -> bool:
        return self._top == -1

    def size(self) -> int:
        return self._top + 1

# 动态扩容版本（类似 Python list）
class DynamicArrayStack:
    def __init__(self):
        self._data = []  # Python list 作底层，动态扩容由 list 负责

    def push(self, val) -> None:
        self._data.append(val)   # list.append = O(1) 均摊

    def pop(self):
        if not self._data:
            raise IndexError("Stack underflow")
        return self._data.pop()  # list.pop() = O(1)

    def peek(self):
        if not self._data:
            raise IndexError("Stack underflow")
        return self._data[-1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def size(self) -> int:
        return len(self._data)
```

```cpp
#include <vector>
#include <stdexcept>

// C++ 数组栈（底层使用 std::vector 动态扩容）
template<typename T>
class ArrayStack {
    std::vector<T> data;
public:
    // push O(1) 均摊
    void push(const T& val) { data.push_back(val); }

    // pop O(1)
    T pop() {
        if (isEmpty()) throw std::underflow_error("Stack underflow");
        T top = data.back();
        data.pop_back();
        return top;
    }

    // peek O(1)（返回引用，避免拷贝）
    const T& peek() const {
        if (isEmpty()) throw std::underflow_error("Stack underflow");
        return data.back();
    }

    bool isEmpty() const { return data.empty(); }
    int size() const { return (int)data.size(); }
};

// 使用方式
// ArrayStack<int> s;
// s.push(1); s.push(2); s.push(3);
// s.pop();   // 返回 3
// s.peek();  // 返回 2
```

**复杂度总结**：静态数组实现，所有操作 $O(1)$，但固定容量；动态数组（`std::vector` 或 Python `list`）均摊 $O(1)$，容量自动翻倍扩容。

### 5.1.3 链表实现（头插头删，O(1) push/pop）

用**单向链表的头部**作为栈顶：`push` = 头插，`pop` = 删除头节点。无需担心容量限制，按需分配内存。

```python
class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class LinkedStack:
    def __init__(self):
        self._head = None   # 头节点 = 栈顶
        self._size = 0

    def push(self, val) -> None:
        """头插 O(1)：新节点指向旧头，更新 head"""
        self._head = Node(val, self._head)
        self._size += 1

    def pop(self):
        """头删 O(1)：保存头节点值，head 移到 head.next"""
        if self.is_empty():
            raise IndexError("Stack underflow")
        val = self._head.val
        self._head = self._head.next  # Python GC 自动回收旧头节点
        self._size -= 1
        return val

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack underflow")
        return self._head.val

    def is_empty(self) -> bool:
        return self._head is None

    def size(self) -> int:
        return self._size
```

```cpp
// C++ 链表栈
template<typename T>
class LinkedStack {
    struct Node {
        T val;
        Node* next;
        Node(T v, Node* n = nullptr) : val(v), next(n) {}
    };
    Node* head = nullptr;
    int _size = 0;

public:
    ~LinkedStack() {
        while (head) { Node* tmp = head; head = head->next; delete tmp; }
    }

    void push(const T& val) {
        head = new Node(val, head);  // 头插 O(1)
        _size++;
    }

    T pop() {
        if (!head) throw std::underflow_error("Stack underflow");
        T val = head->val;
        Node* old = head;
        head = head->next;
        delete old;              // C++ 手动释放
        _size--;
        return val;
    }

    const T& peek() const {
        if (!head) throw std::underflow_error("Stack underflow");
        return head->val;
    }

    bool isEmpty() const { return !head; }
    int size() const { return _size; }
};
```

**对比**：

| 特性 | 数组栈 | 链表栈 |
|------|--------|--------|
| 内存布局 | 连续（缓存友好）| 离散（缓存不友好）|
| 容量 | 固定或倍增扩容 | 按需分配，无上限 |
| push/pop | $O(1)$ 均摊 | $O(1)$ 严格 |
| 额外开销 | 数组可能有空槽 | 每个节点多一个指针（8字节）|
| 实践推荐 | ✅ 首选（缓存命中率高）| 需要链表特性时才用 |

### 5.1.4 Python `list` 作栈的惯例（append/pop）

Python 内置的 `list` 已经是一个高效的动态数组栈。面试和工程中直接使用即可，无需手写：

```python
stack = []

# push → append
stack.append(1)
stack.append(2)
stack.append(3)
# stack = [1, 2, 3]

# pop → pop()（默认弹出最后一个）
val = stack.pop()   # val = 3, stack = [1, 2]

# peek → stack[-1]
top = stack[-1]     # top = 2（不弹出）

# isEmpty → not stack
if not stack:
    print("empty")

# 切勿用 stack.pop(0) 代替 dequeue！
# pop(0) 时间 O(n)（整体左移），应使用 collections.deque
```

```cpp
// C++ 推荐使用 std::stack（内部默认用 std::deque）
#include <stack>

std::stack<int> stk;
stk.push(1); stk.push(2); stk.push(3);
stk.top();    // 查看栈顶（不弹出），返回 3
stk.pop();    // 弹出栈顶（无返回值！）
int val = stk.top(); stk.pop();  // 获取并弹出 = top() + pop()
stk.empty();  // true / false
stk.size();   // 元素数量

// 或用 vector 手动模拟（更透明）
#include <vector>
std::vector<int> s;
s.push_back(1);
int top = s.back(); s.pop_back();  // 等价于 push/pop
```

> ⚠️ **C++ 陷阱**：`std::stack::pop()` 不返回值！需先 `top()` 取值，再 `pop()` 删除。这与 Python 的 `list.pop()` 不同。

### 5.1.5 栈在系统中的应用：函数调用栈、浏览器历史、撤销操作

**1. 函数调用栈（Call Stack）**

每当一个函数被调用时，操作系统在「调用栈」上压入一个**栈帧（Stack Frame）**，记录：
- 返回地址
- 局部变量
- 实参值

函数返回时，对应栈帧被弹出，控制权回到调用方。

```python
def factorial(n):
    # 调用 factorial(3) 的调用栈（从底到顶）：
    # [factorial(3)] → [factorial(2)] → [factorial(1)] → [factorial(0)]
    # factorial(0) 返回后，逐层弹出栈帧，乘法逐级计算
    if n == 0:
        return 1
    return n * factorial(n - 1)

# 若 n 过大（如 factorial(100000)），Python 抛出 RecursionError（栈溢出）
# 解决：改成尾递归迭代或显式栈模拟
```

**2. 浏览器前进/后退**

用两个栈实现：
- `back_stack`：存访问历史（每次跳转，把当前页压入 back_stack）
- `forward_stack`：存"可以前进的页面"（每次后退，把当前页压入 forward_stack）

```python
back_stack = []
forward_stack = []
current = "home"

def navigate(url: str):
    global current
    back_stack.append(current)  # 当前页压入历史
    forward_stack.clear()       # 新跳转清空前进历史
    current = url

def go_back():
    global current
    if not back_stack: return
    forward_stack.append(current)
    current = back_stack.pop()

def go_forward():
    global current
    if not forward_stack: return
    back_stack.append(current)
    current = forward_stack.pop()

navigate("page_a"); navigate("page_b"); navigate("page_c")
go_back()     # current = "page_b", back=[home,page_a], forward=[page_c]
go_back()     # current = "page_a", back=[home], forward=[page_c,page_b]
go_forward()  # current = "page_b", back=[home,page_a], forward=[page_c]
```

**3. 撤销/重做（Undo/Redo）**

编辑器的 Undo/Redo 也是经典的双栈应用：操作执行时压入 undo_stack；撤销时弹出并压入 redo_stack；新操作执行时清空 redo_stack。

### 5.1.6 单调栈（Monotonic Stack）

<div data-component="MonotonicStackTrace"></div>

**单调栈**：栈内元素始终保持**严格单调**（递增或递减）的栈。它通过在每次 `push` 之前弹出「打破单调性」的元素，从而 $O(1)$ 均摊地维护单调序列。

**核心模式：找"下一个更大元素"（Next Greater Element, NGE）**

```
给定数组 [2, 1, 5, 6, 2, 3]，求每个元素右侧第一个比它大的元素：
 2 → 5
 1 → 5
 5 → 6
 6 → -1
 2 → 3
 3 → -1
```

**算法**：维护一个**单调递减栈**（栈顶到栈底递减）：

1. 遍历数组，当前元素 `x`
2. 若 `x > stack.top()`，则 stack.top() 找到了自己的 NGE = `x`，弹出并记录答案；重复直到栈为空或 `x ≤ stack.top()`
3. 将 `x`（或其下标）压栈

```python
def next_greater_element(nums: list[int]) -> list[int]:
    """O(n) 时间，O(n) 空间——每个元素最多入栈/出栈各一次"""
    n = len(nums)
    result = [-1] * n
    stack = []   # 存下标（单调递减：stack[0] 最大值）

    for i in range(n):
        # 弹出所有比 nums[i] 小的元素（它们的 NGE 就是 nums[i]）
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    # 栈内剩余元素无 NGE，result 中已初始化为 -1

    return result

# 示例：
# nums   = [2, 1, 5, 6, 2, 3]
# result = [5, 5, 6,-1, 3,-1]

# 变体：找下一个更小元素 → 改为维护单调递增栈（弹出条件改为 nums[top] > nums[i]）
def next_smaller_element(nums: list[int]) -> list[int]:
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and nums[stack[-1]] > nums[i]:  # 递增栈：弹更大的
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result
```

```cpp
// C++ 下一个更大元素
std::vector<int> nextGreaterElement(const std::vector<int>& nums) {
    int n = nums.size();
    std::vector<int> result(n, -1);
    std::stack<int> stk;   // 存下标，维护单调递减（值递减）
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && nums[stk.top()] < nums[i]) {
            result[stk.top()] = nums[i];
            stk.pop();
        }
        stk.push(i);
    }
    return result;
}
```

**为何是 $O(n)$？**
每个元素**最多入栈一次、出栈一次**，总操作次数 $\leq 2n$，均摊每次 $O(1)$。

**单调栈总结**：

| 目标 | 栈的单调性 | 弹出条件 |
|------|-----------|---------|
| 下一个更大元素 | 单调递减（值） | `nums[top] < nums[i]` |
| 下一个更小元素 | 单调递增（值） | `nums[top] > nums[i]` |
| 上一个更大元素 | 单调递减（值） | 从右往左遍历，同上逻辑 |

---

## 5.2 队列（Queue）

### 5.2.1 FIFO 原则与抽象接口（enqueue/dequeue/front/isEmpty）

**队列（Queue）** 遵循 **先进先出（FIFO, First In First Out）** 原则：最早进入的元素最先被取出。

类比日常生活的排队购票：先到的人先买票，后到的人在队尾等候。

**抽象接口（ADT）**：

| 操作 | 说明 | 时间复杂度 |
|------|------|-----------|
| `enqueue(x)` | 在队尾插入 x | $O(1)$ |
| `dequeue()` | 移除并返回队头元素 | $O(1)$ |
| `front()` / `peek()` | 查看队头元素（不移除）| $O(1)$ |
| `isEmpty()` | 判断队列是否为空 | $O(1)$ |
| `size()` | 返回元素个数 | $O(1)$ |

**FIFO 的自然应用场景**：
- **BFS（广度优先搜索）**：按层序扩展节点，最先发现的节点最先被处理
- **操作系统任务调度**：FCFS（先来先服务）调度队列
- **打印机缓冲队列**：先提交的打印任务先执行
- **网络数据包缓冲**：按接收顺序处理数据包

### 5.2.2 循环数组实现（避免假溢出，head/tail 指针取模）

<div data-component="CircularQueueVisualizer"></div>

朴素数组队列的问题：`dequeue` 之后数组头部空出空间，但数据都在右边，导致「假溢出」——空间明明还有，却因 `tail` 到达末尾而无法插入。

**循环队列**：让 `head` 和 `tail` 指针绕环行走（取模），循环利用空间：

```
初始：head = 0, tail = 0（空）
enqueue(A)：data[0]=A, tail=1
enqueue(B)：data[1]=B, tail=2
enqueue(C)：data[2]=C, tail=3
dequeue()：返回 A, head=1
enqueue(D)：data[3]=D, tail=4
dequeue()：返回 B, head=2
enqueue(E)：data[4] → 但 tail=4 到达末尾？
→ tail = (4+1) % 5 = 0（绕回 0 号槽）
```

**判满与判空**：最常见约定是**预留一个空槽**，这样满与空的判断条件不同：
- 空：`head == tail`
- 满：`(tail + 1) % capacity == head`

```python
class CircularQueue:
    def __init__(self, capacity: int):
        self._data = [None] * (capacity + 1)  # 多一个空槽用于区分满/空
        self._head = 0
        self._tail = 0
        self._cap = capacity + 1

    def enqueue(self, val) -> bool:
        if self.is_full():
            return False           # 或 raise
        self._data[self._tail] = val
        self._tail = (self._tail + 1) % self._cap   # 取模循环
        return True

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue underflow")
        val = self._data[self._head]
        self._head = (self._head + 1) % self._cap   # 取模循环
        return val

    def front(self):
        if self.is_empty():
            raise IndexError("Queue underflow")
        return self._data[self._head]

    def is_empty(self) -> bool:
        return self._head == self._tail

    def is_full(self) -> bool:
        return (self._tail + 1) % self._cap == self._head

    def size(self) -> int:
        return (self._tail - self._head + self._cap) % self._cap
```

```cpp
// C++ 循环队列
template<typename T>
class CircularQueue {
    std::vector<T> data;
    int head = 0, tail = 0, cap;
public:
    CircularQueue(int capacity) : cap(capacity + 1), data(capacity + 1) {}

    bool enqueue(const T& val) {
        if (isFull()) return false;
        data[tail] = val;
        tail = (tail + 1) % cap;
        return true;
    }

    T dequeue() {
        if (isEmpty()) throw std::underflow_error("Queue underflow");
        T val = data[head];
        head = (head + 1) % cap;
        return val;
    }

    const T& front() const {
        if (isEmpty()) throw std::underflow_error("Queue underflow");
        return data[head];
    }

    bool isEmpty() const { return head == tail; }
    bool isFull() const { return (tail + 1) % cap == head; }
    int size() const { return (tail - head + cap) % cap; }
};
```

> ⚠️ **常见混淆**：满条件是 `(tail+1)%cap == head`，空条件是 `head == tail`。两者容易记反。记忆技巧：满时 tail "追上" head 前一格（保留一格空槽）；空时 tail == head（指向同一格）。

### 5.2.3 链表实现（尾插头删，O(1) 所有操作）

用链表实现队列：**队头 = 链表头**，**队尾 = 链表尾**。
- `enqueue` = 尾插（需维护 `tail` 指针）
- `dequeue` = 头删

```python
class Node:
    def __init__(self, val, nxt=None):
        self.val = val
        self.next = nxt

class LinkedQueue:
    def __init__(self):
        self._head = None   # 队头（dequeue 端）
        self._tail = None   # 队尾（enqueue 端）
        self._size = 0

    def enqueue(self, val) -> None:
        """尾插 O(1)"""
        node = Node(val)
        if self._tail is None:          # 空队列
            self._head = self._tail = node
        else:
            self._tail.next = node      # 原尾节点指向新节点
            self._tail = node           # 更新尾指针
        self._size += 1

    def dequeue(self):
        """头删 O(1)"""
        if self.is_empty():
            raise IndexError("Queue underflow")
        val = self._head.val
        self._head = self._head.next
        if self._head is None:          # 队列变为空
            self._tail = None
        self._size -= 1
        return val

    def front(self):
        if self.is_empty():
            raise IndexError("Queue underflow")
        return self._head.val

    def is_empty(self) -> bool:
        return self._head is None

    def size(self) -> int:
        return self._size
```

```cpp
// C++ 链表队列
template<typename T>
class LinkedQueue {
    struct Node { T val; Node* next; Node(T v) : val(v), next(nullptr) {} };
    Node* head = nullptr;
    Node* tail = nullptr;
    int _size = 0;
public:
    ~LinkedQueue() { while (head) { Node* t=head; head=head->next; delete t; } }

    void enqueue(const T& val) {
        Node* node = new Node(val);
        if (!tail) head = tail = node;
        else { tail->next = node; tail = node; }
        _size++;
    }

    T dequeue() {
        if (!head) throw std::underflow_error("Queue underflow");
        T val = head->val;
        Node* old = head;
        head = head->next;
        if (!head) tail = nullptr;
        delete old;
        _size--;
        return val;
    }

    const T& front() const { return head->val; }
    bool isEmpty() const { return !head; }
    int size() const { return _size; }
};
```

### 5.2.4 Python `collections.deque` 的性能优势

Python 标准库的 `collections.deque` 是**双端循环链表 + 固定大小的块数组（block deque）**实现，所有两端操作均为严格 $O(1)$，远优于 `list`：

```python
from collections import deque

# 作为队列使用
q = deque()
q.append(1)        # 右端入队 O(1)
q.append(2)
q.append(3)
q.popleft()        # 左端出队 O(1) ← 关键！list.pop(0) 是 O(n)
q.appendleft(0)    # 左端入队 O(1)（双端特性）
q.pop()            # 右端出队 O(1)

# 限制最大容量（满时自动弹出另一端）
ring = deque(maxlen=3)
ring.append(1); ring.append(2); ring.append(3)
ring.append(4)   # 自动弹出左端 1，ring = deque([2,3,4])

# 性能对比（n = 10^6 次 dequeue）：
# list.pop(0)     → O(n) 每次 → 总 O(n²) ≈ 几十秒
# deque.popleft() → O(1) 每次 → 总 O(n) ≈ 零点几秒
```

```cpp
// C++ 推荐使用 std::queue（底层默认 std::deque）
#include <queue>
std::queue<int> q;
q.push(1); q.push(2); q.push(3);
q.front();    // 查看队头 = 1
q.pop();      // 弹出队头（无返回值）
q.back();     // 查看队尾 = 3
q.empty();    // true/false
q.size();     // 元素数量
```

### 5.2.5 BFS 中的队列角色（层序遍历、最短路径）

队列是 BFS 的核心数据结构：每轮从队头取出节点，处理后将其邻居加入队尾——天然保证**按层（距离）顺序**探索。

```python
from collections import deque

def bfs(graph: dict, start: int) -> list[int]:
    """图的 BFS 遍历，返回访问顺序"""
    visited = {start}
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()    # 队头出队 O(1)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)  # 邻居入队尾 O(1)
    return order

# 二叉树层序遍历（同样的 BFS 模式）
def level_order(root) -> list[list[int]]:
    if not root: return []
    queue = deque([root])
    result = []
    while queue:
        level_size = len(queue)     # 当前层节点数
        level = []
        for _ in range(level_size): # 只处理当前层
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

---

## 5.3 双端队列（Deque）

### 5.3.1 Deque 接口（在两端都支持 O(1) 插入/删除）

**双端队列（Deque，Double-Ended Queue）** 是栈与队列的超集：在**两端**都能以 $O(1)$ 完成插入和删除。

| 操作 | 说明 | 时间复杂度 |
|------|------|-----------|
| `push_front(x)` / `appendleft(x)` | 从左端插入 | $O(1)$ |
| `push_back(x)` / `append(x)` | 从右端插入 | $O(1)$ |
| `pop_front()` / `popleft()` | 移除并返回左端元素 | $O(1)$ |
| `pop_back()` / `pop()` | 移除并返回右端元素 | $O(1)$ |
| `front()` | 查看左端元素 | $O(1)$ |
| `back()` | 查看右端元素 | $O(1)$ |

Deque 可以模拟栈（只操作一端）和队列（一端进一端出），是最通用的线性容器。

```python
from collections import deque

d = deque()
d.append(2)        # 右端插入：[2]
d.appendleft(1)    # 左端插入：[1, 2]
d.append(3)        # 右端插入：[1, 2, 3]
d.popleft()        # 左端删除：返回 1, d=[2, 3]
d.pop()            # 右端删除：返回 3, d=[2]
d[0]               # 查看左端 = 2（不删除）
d[-1]              # 查看右端 = 2（不删除）
```

```cpp
#include <deque>
std::deque<int> d;
d.push_front(1);   // 左端插入
d.push_back(3);    // 右端插入
d.push_back(2);
d.front();         // 查看左端
d.back();          // 查看右端
d.pop_front();     // 移除左端
d.pop_back();      // 移除右端
```

### 5.3.2 单调队列（Monotonic Deque）：维护窗口最值

**单调队列**是专为解决**滑动窗口最值**问题设计的 Deque 变体，队列内元素始终保持单调（如递减）：

- 窗口右边界右移时：从队尾弹出所有比当前元素小的值（破坏单调性），然后从队尾插入当前元素
- 窗口左边界右移时：若队首元素的下标已不在窗口内，从队首弹出

这样，队首始终是当前窗口的最大值，每次查询 $O(1)$，整体 $O(n)$。

### 5.3.3 滑动窗口最值问题：O(n) 解法

<div data-component="DequeWindowMaxDemo"></div>

**题目**（LeetCode #239）：给定数组 `nums` 和窗口大小 `k`，求每个窗口的最大值。

```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """
    单调队列（递减），O(n) 时间，O(k) 空间
    deque 中保存下标，保证 nums[dq[0]] 始终是当前窗口最大值
    """
    dq = deque()       # 存下标，队首到队尾 nums 值递减
    result = []

    for i in range(len(nums)):
        # ① 弹出所有不如 nums[i] 大的队尾元素（它们在当前窗口内不可能是最大值）
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()          # 从队尾弹出

        dq.append(i)          # 将当前下标从队尾加入

        # ② 若队首元素已滑出窗口（下标 < i-k+1），弹出
        if dq[0] < i - k + 1:
            dq.popleft()      # 从队首弹出

        # ③ 窗口已满（i >= k-1），队首即当前窗口最大值
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# 示例：nums=[1,3,-1,-3,5,3,6,7], k=3
# 窗口  [1,3,-1]  → max=3
# 窗口  [3,-1,-3] → max=3
# 窗口  [-1,-3,5] → max=5
# 窗口  [-3,5,3]  → max=5
# 窗口  [5,3,6]   → max=6
# 窗口  [3,6,7]   → max=7
# 输出：[3,3,5,5,6,7]
```

```cpp
// C++ 单调队列滑动窗口最大值
std::vector<int> maxSlidingWindow(const std::vector<int>& nums, int k) {
    std::deque<int> dq;  // 存下标，值单调递减
    std::vector<int> result;

    for (int i = 0; i < (int)nums.size(); i++) {
        // ① 弹出队尾所有不如 nums[i] 大的元素
        while (!dq.empty() && nums[dq.back()] <= nums[i])
            dq.pop_back();
        dq.push_back(i);

        // ② 队首超出窗口
        if (dq.front() < i - k + 1)
            dq.pop_front();

        // ③ 窗口已满，记录最大值
        if (i >= k - 1)
            result.push_back(nums[dq.front()]);
    }
    return result;
}
```

**为何是 $O(n)$？** 每个下标最多入队一次、出队一次，总操作 $\leq 2n$。

### 5.3.4 单调队列 vs 堆（复杂度与适用场景对比）

| 方案 | 时间复杂度 | 空间复杂度 | 优点 | 缺点 |
|------|-----------|-----------|------|------|
| 暴力（每窗口遍历）| $O(nk)$ | $O(1)$ | 简单 | $n,k$ 大时超时 |
| 大顶堆（延迟删除）| $O(n \log k)$ | $O(k)$ | 灵活（多种查询）| 实现稍复杂 |
| 单调队列 | $O(n)$ | $O(k)$ | 最优时间复杂度 | 只支持单一最值查询 |

**使用建议**：
- 需要窗口最值 → **单调队列**（$O(n)$ 最优）
- 需要窗口中位数或多种聚合 → 两个堆或线段树
- 仅单次查询或 $k$ 很小 → 暴力即可

---

## 5.4 用栈模拟队列 / 用队列模拟栈

### 5.4.1 两个栈实现队列（均摊 O(1) dequeue）

<div data-component="TwoStackQueue"></div>

**思路**：用两个栈 `inbox`（入队栈）和 `outbox`（出队栈）：
- `enqueue(x)`：压入 `inbox`（$O(1)$）
- `dequeue()`：若 `outbox` 为空，把 `inbox` 所有元素倒入 `outbox`（翻转顺序 = 恢复 FIFO）；然后从 `outbox.pop()`

**均摊分析**：每个元素最多被移动一次（从 `inbox` 到 `outbox`），总移动次数 $\leq n$，均摊 $O(1)$。

```python
class MyQueue:
    """用两个栈模拟队列（LeetCode #232）"""
    def __init__(self):
        self.inbox = []    # 入队栈：push 到这里
        self.outbox = []   # 出队栈：pop 从这里

    def push(self, x: int) -> None:
        self.inbox.append(x)   # 直接压入 inbox，O(1)

    def _transfer(self) -> None:
        """当 outbox 为空时，将 inbox 全部转移（只在需要时转移！）"""
        if not self.outbox:
            while self.inbox:
                self.outbox.append(self.inbox.pop())
            # inbox 全部倒入 outbox 后，outbox 顶部 = 最早入队的元素

    def pop(self) -> int:
        self._transfer()
        if not self.outbox:
            raise IndexError("Queue underflow")
        return self.outbox.pop()

    def peek(self) -> int:
        self._transfer()
        return self.outbox[-1]

    def empty(self) -> bool:
        return not self.inbox and not self.outbox
```

```cpp
// C++ 两个栈模拟队列（LeetCode #232）
class MyQueue {
    std::stack<int> inbox, outbox;

    void transfer() {
        if (outbox.empty()) {
            while (!inbox.empty()) {
                outbox.push(inbox.top());
                inbox.pop();
            }
        }
    }
public:
    void push(int x) { inbox.push(x); }

    int pop() {
        transfer();
        int val = outbox.top();
        outbox.pop();
        return val;
    }

    int peek() {
        transfer();
        return outbox.top();
    }

    bool empty() { return inbox.empty() && outbox.empty(); }
};
```

> ⚠️ **常见错误**：每次 `pop/peek` 都把所有元素搬来搬去！正确做法是只有 `outbox` 为空时才转移，且每次全量转移。若频繁在两者之间来回倒，实际复杂度退化为 $O(n)$。

### 5.4.2 两个队列实现栈（O(n) 的 pop/top）

**思路**：主队列 `q1` 存所有元素，辅助队列 `q2` 协助旋转：
- `push(x)`：压入 `q1`（$O(1)$）
- `pop()`：把 `q1` 前 `n-1` 个元素转移到 `q2`，取出 `q1` 剩余的最后一个（= 栈顶），然后交换 `q1` 和 `q2`

```python
from collections import deque

class MyStack:
    """用两个队列模拟栈（LeetCode #225），O(n) pop/top"""
    def __init__(self):
        self.q1 = deque()   # 主队列
        self.q2 = deque()   # 辅助队列

    def push(self, x: int) -> None:
        self.q1.append(x)   # 直接入队 q1，O(1)

    def pop(self) -> int:
        # 将 q1 前 n-1 个元素转移到 q2
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        val = self.q1.popleft()    # q1 剩余的最后一个 = 栈顶
        self.q1, self.q2 = self.q2, self.q1  # 交换引用，不是复制
        return val

    def top(self) -> int:
        # 同 pop，但最后把值再压回 q1
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        val = self.q1[0]
        self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
        return val

    def empty(self) -> bool:
        return not self.q1
```

```cpp
// C++ 用两个队列模拟栈
class MyStack {
    std::queue<int> q1, q2;
public:
    void push(int x) { q1.push(x); }

    int pop() {
        while (q1.size() > 1) { q2.push(q1.front()); q1.pop(); }
        int val = q1.front(); q1.pop();
        std::swap(q1, q2);   // O(1) 交换
        return val;
    }

    int top() {
        while (q1.size() > 1) { q2.push(q1.front()); q1.pop(); }
        int val = q1.front();
        q2.push(q1.front()); q1.pop();
        std::swap(q1, q2);
        return val;
    }

    bool empty() { return q1.empty(); }
};
```

### 5.4.3 应用价值与面试场景

| 互模拟 | push | pop/dequeue | 均摊复杂度 |
|--------|------|-------------|-----------|
| 双栈→队列 | $O(1)$ | 均摊 $O(1)$ | 最优 |
| 双队→栈 | $O(1)$ | $O(n)$ | 无法避免 |

**为何需要掌握这类题？**
1. 面试官通过它考察对 LIFO/FIFO 本质的理解，以及均摊复杂度分析能力
2. 某些底层系统（如 Actor 模型消息传递）只提供一种原语，需要用它模拟另一种
3. 理解均摊分析的真实案例——银行家论证（每个元素"赚取"两个信用分，最多消耗两次操作）

---

## 5.5 栈与队列应用

### 5.5.1 括号匹配（多种括号的通用解法）

**题目**（LeetCode #20）：给定包含 `()[]{}`  的字符串，判断括号是否有效。

**核心洞察**：遇到左括号压栈；遇到右括号时，若栈为空或栈顶不匹配则无效；否则弹出栈顶。最后栈为空则有效。

```python
def is_valid(s: str) -> bool:
    """O(n) 时间，O(n) 空间（最坏情况全是左括号）"""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}  # 右括号 → 对应左括号

    for ch in s:
        if ch in '([{':
            stack.append(ch)      # 左括号压栈
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False      # 栈为空 or 顶部不匹配
            stack.pop()           # 匹配成功，弹出
    return not stack              # 所有左括号都被匹配则有效

# 测试用例：
# "()"          → True
# "()[]{}"      → True
# "(]"          → False
# "([)]"        → False
# "{[]}"        → True
# "("           → False（栈不为空）
```

```cpp
bool isValid(const std::string& s) {
    std::stack<char> stk;
    for (char ch : s) {
        if (ch == '(' || ch == '[' || ch == '{') {
            stk.push(ch);
        } else {
            if (stk.empty()) return false;
            char top = stk.top(); stk.pop();
            if (ch == ')' && top != '(') return false;
            if (ch == ']' && top != '[') return false;
            if (ch == '}' && top != '{') return false;
        }
    }
    return stk.empty();
}
```

**扩展变体**：
- 最长有效括号子串（#32）：用栈存下标，dp 解
- 括号生成（#22）：回溯 + 剪枝

### 5.5.2 逆波兰表达式求值（后缀表达式）

<div data-component="ExpressionEval"></div>

**后缀表达式（Reverse Polish Notation, RPN）** 将运算符放在两个操作数之后，无需括号即可表达运算优先级：

```
中缀（人类可读）：3 + 4 × 2 = 11
后缀（计算机友好）：3 4 2 × + = 11

中缀：(5 + 1) × (2 + 4) × 3 = 108
后缀：5 1 + 2 4 + × 3 ×
```

**求值算法**：遍历后缀表达式的每个 token：
- 是数字 → 压栈
- 是运算符 → 弹出两个操作数计算，结果压回栈

```python
def eval_rpn(tokens: list[str]) -> int:
    """逆波兰表达式求值，O(n) 时间，O(n) 空间"""
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in tokens:
        if token not in operators:
            stack.append(int(token))    # 数字压栈
        else:
            b = stack.pop()             # 第二个操作数（后压入的）
            a = stack.pop()             # 第一个操作数（先压入的）
            if token == '+':   stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            elif token == '/': stack.append(int(a / b))  # 向零取整
    return stack[0]

# 示例：["2","1","+","3","*"]
# 步骤：
# token="2" → stack=[2]
# token="1" → stack=[2,1]
# token="+" → pop 1,2 → 2+1=3 → stack=[3]
# token="3" → stack=[3,3]
# token="*" → pop 3,3 → 3*3=9 → stack=[9]
# 返回 9（即 (2+1)*3）

# 注意：Python 的整除 // 是向下取整，-7//2=-4≠-3
# 题意要求向零取整，因此用 int(-7/2) = -3
```

```cpp
int evalRPN(const std::vector<std::string>& tokens) {
    std::stack<long long> stk;
    for (const std::string& t : tokens) {
        if (t == "+" || t == "-" || t == "*" || t == "/") {
            long long b = stk.top(); stk.pop();
            long long a = stk.top(); stk.pop();
            if      (t == "+") stk.push(a + b);
            else if (t == "-") stk.push(a - b);
            else if (t == "*") stk.push(a * b);
            else                stk.push(a / b);   // C++ 整除天然向零
        } else {
            stk.push(std::stoll(t));
        }
    }
    return (int)stk.top();
}
```

### 5.5.3 中缀表达式转后缀（调度场算法 Shunting Yard）

**调度场算法（Shunting Yard Algorithm，Dijkstra 1961）**：将人类书写的中缀表达式转换为便于计算的后缀形式，时间 $O(n)$。

**运算符优先级表**：

| 运算符 | 优先级 | 结合性 |
|--------|--------|--------|
| `(`    | —      | —      |
| `+` `-` | 1    | 左结合 |
| `*` `/` | 2    | 左结合 |
| `^`（幂）| 3   | 右结合 |

```python
def infix_to_postfix(expression: str) -> list[str]:
    """
    调度场算法：中缀 → 后缀
    输入：字符串如 "3 + 4 * 2 / ( 1 - 5 ) ^ 2"（空格分隔）
    输出：后缀 token 列表
    """
    prec = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    right_assoc = {'^'}   # 右结合运算符

    op_stack = []       # 运算符栈
    output = []         # 输出队列（后缀结果）

    tokens = expression.split()
    for tok in tokens:
        if tok.lstrip('-').isdigit():      # 数字 → 直接输出
            output.append(tok)
        elif tok == '(':                    # 左括号 → 压栈
            op_stack.append(tok)
        elif tok == ')':                    # 右括号 → 弹出直到左括号
            while op_stack and op_stack[-1] != '(':
                output.append(op_stack.pop())
            op_stack.pop()                  # 弹出 '('（不输出）
        else:                               # 运算符
            # 弹出优先级更高（或相同且左结合）的栈顶运算符
            while (op_stack and op_stack[-1] != '('
                   and (prec.get(op_stack[-1], 0) > prec[tok]
                        or (prec.get(op_stack[-1], 0) == prec[tok]
                            and tok not in right_assoc))):
                output.append(op_stack.pop())
            op_stack.append(tok)

    # 弹出剩余运算符
    while op_stack:
        output.append(op_stack.pop())

    return output

# 示例：
# infix_to_postfix("3 + 4 * 2")  → ['3', '4', '2', '*', '+']
# infix_to_postfix("( 1 + 2 ) * 3") → ['1', '2', '+', '3', '*']
```

```cpp
// C++ 调度场算法（简化版，仅处理 +−×÷ 和括号）
std::string infixToPostfix(const std::string& expr) {
    std::map<char, int> prec = {{'+'，1}, {'-'，1}, {'*'，2}, {'/'，2}};
    std::stack<char> ops;
    std::string result;

    for (char ch : expr) {
        if (ch == ' ') continue;
        if (isdigit(ch)) {
            result += ch; result += ' ';
        } else if (ch == '(') {
            ops.push(ch);
        } else if (ch == ')') {
            while (!ops.empty() && ops.top() != '(') {
                result += ops.top(); result += ' '; ops.pop();
            }
            ops.pop();  // 弹出 '('
        } else {
            while (!ops.empty() && ops.top() != '('
                   && prec.count(ops.top()) && prec[ops.top()] >= prec[ch]) {
                result += ops.top(); result += ' '; ops.pop();
            }
            ops.push(ch);
        }
    }
    while (!ops.empty()) { result += ops.top(); result += ' '; ops.pop(); }
    return result;
}
```

### 5.5.4 接雨水（Monotonic Stack 经典应用，LeetCode #42）

**题目**：给定柱状图高度数组，计算两侧柱子之间能够积水的总量。

**单调栈思路**：维护递减单调栈（存下标），遇到比栈顶高的柱子时：「当前柱子 + 栈顶 + 新栈顶」形成一个凹槽，可以积水。

```python
def trap(height: list[int]) -> int:
    """单调栈法：O(n) 时间，O(n) 空间"""
    stack = []   # 单调递减（从底到顶高度递减）
    water = 0

    for i in range(len(height)):
        # 当 height[i] > 栈顶高度，说明出现了积水区域
        while stack and height[stack[-1]] < height[i]:
            bottom = stack.pop()         # 凹槽底部（需要积水的槽）
            if not stack:
                break                    # 左侧没有边界，无法积水
            left = stack[-1]             # 左边界
            right = i                    # 右边界（当前 i）
            width = right - left - 1
            bounded_height = min(height[left], height[right]) - height[bottom]
            water += width * bounded_height

        stack.append(i)

    return water

# 示例：height = [0,1,0,2,1,0,1,3,2,1,2,1]
# 输出：6

# 对比：双指针法（O(n) 时间，O(1) 空间，更优）
def trap_two_ptr(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water
```

```cpp
// C++ 接雨水（双指针法 O(1) 空间）
int trap(const std::vector<int>& height) {
    int left = 0, right = (int)height.size() - 1;
    int leftMax = 0, rightMax = 0, water = 0;
    while (left < right) {
        if (height[left] < height[right]) {
            leftMax = std::max(leftMax, height[left]);
            water += leftMax - height[left];
            left++;
        } else {
            rightMax = std::max(rightMax, height[right]);
            water += rightMax - height[right];
            right--;
        }
    }
    return water;
}
```

### 5.5.5 柱状图中最大矩形（单调栈 #84）

**题目**（LeetCode #84）：给定柱状图的每根柱子高度，求可以形成的最大矩形面积。

**核心思路**：对每根柱子，找到以它为高度时，向左/右能延伸到哪里（即左右两侧第一根比它矮的柱子的位置）。这正好是「前一个更小元素」（left单调递增栈）和「下一个更小元素」（right单调递增栈）的典型应用。

```python
def largest_rectangle(heights: list[int]) -> int:
    """单调递增栈，O(n) 时间，O(n) 空间"""
    # 在首尾各添加哨兵高度 0，避免处理边界特殊情况
    heights = [0] + heights + [0]
    n = len(heights)
    stack = [0]   # 存下标，栈内高度单调递增
    max_area = 0

    for i in range(1, n):
        # 当遇到比栈顶更矮的柱子，栈顶柱子的「右边界」就是 i
        while heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]       # 以栈顶为高度
            w = i - stack[-1] - 1         # 宽度 = 右边界 - 左边界 - 1
            max_area = max(max_area, h * w)
        stack.append(i)

    return max_area

# 示例：heights = [2,1,5,6,2,3]
# 加哨兵后：[0,2,1,5,6,2,3,0]
# 最大矩形面积 = 10（高度5和6的宽度各1，但高度2延伸5格宽）
```

```cpp
int largestRectangleArea(std::vector<int>& heights) {
    heights.insert(heights.begin(), 0);  // 左哨兵
    heights.push_back(0);                // 右哨兵
    int n = heights.size();
    std::stack<int> stk;
    stk.push(0);
    int maxArea = 0;
    for (int i = 1; i < n; i++) {
        while (heights[i] < heights[stk.top()]) {
            int h = heights[stk.top()]; stk.pop();
            int w = i - stk.top() - 1;
            maxArea = std::max(maxArea, h * w);
        }
        stk.push(i);
    }
    return maxArea;
}
```

**接雨水 vs 柱状图最大矩形对比**：

| 题目 | 栈的单调性 | 触发弹出条件 | 计算内容 |
|------|-----------|------------|---------|
| 接雨水（#42）| 单调递减 | `height[i] > stack.top()` | 积水面积 |
| 柱状图（#84）| 单调递增 | `heights[i] < stack.top()` | 矩形面积 |

两道题都是单调栈，但维护方向相反：接雨水找「被夹住的凹槽」，需要递减栈（等待比自己高的来）；柱状图找「同高延伸距离」，需要递增栈（等待比自己矮的来划定右边界）。

### 5.5.6 最小栈设计（O(1) 获取最小值）

**题目**（LeetCode #155）：设计一个支持 `push`、`pop`、`top` 和 `getMin` 的栈，所有操作均 $O(1)$。

**关键思路**：使用一个**辅助栈 `min_stack`** 同步维护到目前为止的最小值序列：

```python
class MinStack:
    """O(1) push/pop/top/getMin，O(n) 空间（辅助栈）"""
    def __init__(self):
        self._stack = []
        self._min_stack = []   # min_stack[i] = main stack 前 i+1 个元素的最小值

    def push(self, val: int) -> None:
        self._stack.append(val)
        # 辅助栈记录"到目前为止的最小值"
        if not self._min_stack:
            self._min_stack.append(val)
        else:
            self._min_stack.append(min(val, self._min_stack[-1]))

    def pop(self) -> None:
        self._stack.pop()
        self._min_stack.pop()   # 同步弹出（两栈高度始终相同）

    def top(self) -> int:
        return self._stack[-1]

    def getMin(self) -> int:
        return self._min_stack[-1]   # 直接取辅助栈顶 O(1)

# 模拟 push(5)  → stack=[5],       min_stack=[5]
# push(3)       → stack=[5,3],     min_stack=[5,3]
# push(7)       → stack=[5,3,7],   min_stack=[5,3,3]  ← 7 > 3，最小仍是 3
# getMin() = 3
# pop()         → stack=[5,3],     min_stack=[5,3]
# pop()         → stack=[5],       min_stack=[5]
# getMin() = 5  ← 正确！
```

```cpp
// C++ MinStack
class MinStack {
    std::stack<int> stk, minStk;
public:
    void push(int val) {
        stk.push(val);
        if (minStk.empty() || val <= minStk.top())
            minStk.push(val);
        else
            minStk.push(minStk.top());   // 当前最小值保持不变，也压入
    }

    void pop() {
        stk.pop();
        minStk.pop();   // 同步弹出
    }

    int top() { return stk.top(); }
    int getMin() { return minStk.top(); }
};
```

**空间优化版（只在 min 减小时才压辅助栈）**：

```python
class MinStackOptimized:
    """辅助栈只在新 min 出现时才压，pop 时只在值等于当前 min 才弹出"""
    def __init__(self):
        self._stack = []
        self._min_stack = []

    def push(self, val: int) -> None:
        self._stack.append(val)
        if not self._min_stack or val <= self._min_stack[-1]:
            self._min_stack.append(val)   # 只有新 min 才压

    def pop(self) -> None:
        val = self._stack.pop()
        if val == self._min_stack[-1]:    # 弹出的是当前 min
            self._min_stack.pop()

    def top(self) -> int:
        return self._stack[-1]

    def getMin(self) -> int:
        return self._min_stack[-1]
```

---

## 5.6 经典题精讲

### 题1：有效括号（LeetCode #20）

已在 5.5.1 节详细讲解（括号匹配通用解法）。

**关键边界**：
- `"]"` → 栈为空就立即返回 False，不要等到最后
- `"([])"` → True：栈内 `[` 被 `]` 弹出后，栈内剩 `(`，最后被 `)` 弹出，栈空 → True
- `"(]"` → 栈顶是 `(`，但遇到 `]` 不匹配 → False

### 题2：移掉 K 位数字（LeetCode #402）

**题目**：给一个字符串形式的非负整数 `num` 和整数 `k`，删去 `k` 位后使剩余数字组成的整数最小。

**贪心 + 单调递增栈**：尽量让高位更小。遇到比栈顶小的数字时，弹出栈顶（即删除那一位），直到已删 `k` 个或输入耗尽。

```python
def remove_k_digits(num: str, k: int) -> str:
    """O(n) 时间，O(n) 空间"""
    stack = []
    for digit in num:
        # 若当前数字比栈顶小，且还能删（k > 0），则弹出栈顶（删除更大位）
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)

    # 若 k 还有剩余，从末尾删（此时栈已单调不减，末尾是最大的）
    if k > 0:
        stack = stack[:-k]

    # 移除前导零，且结果不为空则返回，否则返回 "0"
    result = ''.join(stack).lstrip('0')
    return result if result else "0"

# 示例：
# num="1432219", k=3 → "1219"（删去 4,3,2 高位大数）
# num="10200", k=1   → "200" → "200"（删去 1）→ 再 lstrip → "200"
# num="10", k=2      → "" → "0"
```

```cpp
std::string removeKdigits(std::string num, int k) {
    std::string stk;   // 用 string 模拟栈（输出更方便）
    for (char d : num) {
        while (k > 0 && !stk.empty() && stk.back() > d) {
            stk.pop_back();
            k--;
        }
        stk.push_back(d);
    }
    // 末尾多余位
    while (k-- > 0) stk.pop_back();
    // 去前导零
    int start = 0;
    while (start < (int)stk.size() - 1 && stk[start] == '0') start++;
    return stk.substr(start);
}
```

---

## 5.7 本章小结

| 知识点 | 核心结论 |
|--------|---------|
| 栈（LIFO）数组实现 | push/pop $O(1)$，缓存友好，推荐首选 |
| 栈（LIFO）链表实现 | push/pop 严格 $O(1)$，但指针开销大 |
| 循环队列 | 用 `(idx+1)%cap` 取模解决假溢出；空 `head==tail`，满 `(tail+1)%cap==head` |
| Python 实践 | 栈用 `list`，队列用 `collections.deque`（deque.popleft() 是 $O(1)$）|
| 单调栈 | 每元素最多入/出栈各一次，$O(n)$ 均摊；递减栈找 NGE，递增栈找 NSE |
| 单调队列 | 滑动窗口最值 $O(n)$，优于堆 $O(n \log k)$ |
| 双栈→队列 | enqueue $O(1)$，dequeue 均摊 $O(1)$；`outbox` 为空时才批量转移 |
| 接雨水 | 单调递减栈（寻找凹槽）或双指针（$O(1)$ 空间） |
| 柱状图最大矩形 | 单调递增栈（寻找延伸范围），加哨兵简化边界 |
| 最小栈 | 辅助栈同步维护当前最小值，getMin $O(1)$ |

**🎯 面试高频题单**：
- `#20` 有效括号（必会，栈基础）
- `#155` 最小栈（辅助栈设计）
- `#239` 滑动窗口最大值（单调队列 $O(n)$）
- `#42` 接雨水（单调栈 or 双指针）
- `#84` 柱状图最大矩形（单调栈，有哨兵技巧）
- `#232` 用栈实现队列（均摊分析）
- `#402` 移掉 K 位数字（贪心 + 单调栈）

**💡 思考题**：
1. 单调栈在「接雨水」和「柱状图最大矩形」中的角色有什么本质不同？两道题分别维护递减还是递增栈？
2. `collections.deque` 在 Python 中底层是什么数据结构？为什么 `appendleft/popleft` 是严格 $O(1)$ 而 `list.insert(0, x)` 是 $O(n)$？
3. 双栈模拟队列的均摊复杂度分析：如果连续进行 $n$ 次 `enqueue` 再 $n$ 次 `dequeue`，总操作次数是多少？

> **参考资料**：CLRS 第4版 Chapter 10.1；Sedgewick 1.3（Stacks and Queues）；Skiena 3.1–3.2；LeetCode 题解 #42、#84、#239
