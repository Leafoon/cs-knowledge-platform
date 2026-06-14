# Chapter 12: 哈希表（Hash Table）

> **学习目标**：  
> 理解哈希表是如何以"以空间换时间"的核心思路将查找、插入、删除操作从 $O(\log n)$ 降到期望 $O(1)$ 的；掌握哈希函数设计的核心原则；能区分链地址法与开放寻址法的适用场景与性能特征；理解布隆过滤器的设计动机与误判率公式，并能合理选择参数。

---

## 12.1 直接寻址与哈希的动机

### 12.1.1 直接寻址表：最理想的随机访问

假设我们要维护一个学生成绩库，学生 ID 的范围是 $[0, 999]$（共 1000 个可能的 ID）。最直接的做法是开一个大小为 1000 的数组 `table[0..999]`，将学生 ID 直接作为下标，查找/插入/删除均为 $O(1)$。这就是**直接寻址表（Direct-Address Table）**。

**直接寻址表的美好与局限**：

| 性质 | 详情 |
|------|------|
| 查找 | $O(1)$，直接 `return table[key]` |
| 插入 | $O(1)$，直接 `table[key] = value` |
| 删除 | $O(1)$，直接 `table[key] = None` |
| 空间 | $\Theta(\|U\|)$，$|U|$ 为全集大小 |

问题出在空间上。如果键的全集 $|U|$ 很大（比如学生 ID 是 18 位身份证号，$|U| \approx 10^{18}$），但实际存储的键数 $n$ 很小（只有 100 名学生），那么需要分配 $10^{18}$ 个槽位，这是完全不可接受的。

**关键矛盾**：

$$\text{实际键数 } n \ll \text{全集大小 } |U|$$

直接寻址表在 $|U|$ 很大时**空间浪费极其严重**，甚至根本无法分配。

### 12.1.2 哈希函数：压缩键空间

**哈希表（Hash Table）**的核心思想是引入一个**哈希函数（Hash Function）**：

$$h: U \rightarrow \{0, 1, \ldots, m-1\}$$

它将庞大的键全集 $U$ **压缩映射**到一个大小为 $m$ 的有限槽位数组（称为**哈希表**）中。

- **键（Key）**：原始的键值（如身份证号、字符串、对象等）
- **哈希值（Hash Value）**：$h(k)$，键 $k$ 经哈希函数映射后的整数，作为数组下标
- **槽（Slot）**：哈希表中的一个存储位置，`table[h(k)]` 存放键为 $k$ 的元素

**生活比喻**：把键空间想象成全国的快递单号（约 20 位数字），哈希表是一排有编号的储物柜（只有 100 个）。哈希函数就是把快递单号"折叠"成一个 0-99 的数字，指定放进哪个柜子。优秀的哈希函数让快递均匀分配到各个柜子，差的哈希函数让所有包裹都挤进同一个柜子。

**期望性能**：若哈希函数设计良好，每个槽平均存储 $n/m$ 个元素，查找、插入、删除均期望 $O(1)$。

### 12.1.3 负载因子 α 与性能的关系

**负载因子（Load Factor）**定义为：

$$\alpha = \frac{n}{m}$$

其中 $n$ 是哈希表中存储的元素个数，$m$ 是哈希表的槽数。

- $\alpha < 1$：平均每个槽不到 1 个元素，冲突少
- $\alpha = 1$：平均每个槽正好 1 个元素  
- $\alpha > 1$：（**链地址法**允许，开放寻址法不允许）平均每个槽超过 1 个元素，冲突增多

负载因子是哈希表**时间与空间权衡**的核心参数：
- $\alpha$ 太小 → 空间浪费大，性能好
- $\alpha$ 太大 → 空间利用率高，但查找性能急剧下降
- 工程实践：链地址法通常维持 $\alpha \leq 1$；开放寻址法通常维持 $\alpha \leq 0.7$

### 12.1.4 碰撞不可避免：鸽巢原理

当 $n > m$（元素数超过槽数）时，由**鸽巢原理**（Pigeonhole Principle），必然有至少两个不同的键映射到同一个槽，发生**碰撞（Collision）**。

更深刻的是：即使 $n \leq m$，碰撞也很可能发生——这正是著名的"**生日悖论**"：在一个班级（57 人）中，两人同天生日的概率超过 99%。对于哈希表，当 $m = 365$（天数），$n = 23$ 时碰撞概率已超 50%。

$$P(\text{至少一次碰撞}) = 1 - \frac{m!/(m-n)!}{m^n} \approx 1 - e^{-n^2/(2m)}$$

当 $n \approx \sqrt{2m \ln(1/\delta)}$ 时，碰撞概率约为 $1 - \delta$。  
对于 $m = 10^6, \delta = 0.01$：仅需 $n \approx 3033$ 个元素就有 99% 的概率发生碰撞。

**结论**：碰撞是哈希表必须面对的现实，哈希表的设计核心在于**如何高效地处理碰撞**，而不是消除它。

---

## 12.2 哈希函数设计

**好的哈希函数**需要满足以下性质：
1. **确定性**：相同的键每次产生相同的哈希值
2. **均匀性**：输出尽量均匀分布在 $[0, m-1]$ 上，减少碰撞
3. **高效性**：计算时间 $O(|key|)$（不能比读键还慢）
4. **雪崩效应**：键的微小变化导致哈希值的巨大变化（避免相似键聚集到相邻槽）

### 12.2.1 除法散列法

$$h(k) = k \bmod m$$

**实现最简单**，直接取余数。但 $m$ 的选取至关重要：

- **避免 $m = 2^p$**：此时 $h(k)$ 只取决于 $k$ 的最低 $p$ 位，高位信息完全丢失。若键的低位分布不均匀（如很多键是 8 的倍数），冲突会极其严重。
- **选质数（Prime）**：$m$ 选质数能更充分地利用键的所有位，分布更均匀。Knuth 建议选**离 $2^p$ 既不太近也不太远**的质数。

例如：若哈希表预期存 $n = 2000$ 个元素，取 $\alpha \approx 1$，则 $m$ 选 **2003**（质数，比 $2048 = 2^{11}$ 小但不太近）。

常见误区：$m = 2^p$ 的问题在实际系统中频繁出现，Java 早期 HashMap 因此收到过批评（后来改为使用"扰动函数"弥补）。

```python
def division_hash(key: int, m: int) -> int:
    """
    除法散列法：h(k) = k mod m
    要求：
      - m 应选质数，避免 2 的幂次
      - key 应为非负整数，若为负数先取绝对值或做位运算处理
    """
    return key % m  # Python 的 % 对负数也能返回正值（与 C++ 不同）

# 演示：m = 7（质数）vs m = 8（2的幂次）
keys = [16, 24, 32, 40, 48, 56]  # 全是 8 的倍数
print("m=7（质数）:", [k % 7 for k in keys])   # → [2, 3, 4, 5, 6, 0] 均匀
print("m=8（2^3）: ", [k % 8 for k in keys])   # → [0, 0, 0, 0, 0, 0] 全部冲突！
```
```cpp
#include <iostream>
#include <vector>
using namespace std;

int divisionHash(long long key, int m) {
    // 注意：C++ 中负数取模结果可能为负，需处理
    // ((key % m) + m) % m 保证结果在 [0, m-1]
    return (int)(((key % m) + m) % m);
}

int main() {
    // 演示：m=7 vs m=8，键全为 8 的倍数
    vector<int> keys = {16, 24, 32, 40, 48, 56};
    cout << "m=7（质数）: ";
    for (int k : keys) cout << divisionHash(k, 7) << " ";  // 2 3 4 5 6 0 均匀
    cout << "\nm=8（2^3）:  ";
    for (int k : keys) cout << divisionHash(k, 8) << " ";  // 全为 0！严重冲突
    cout << endl;
}
```

### 12.2.2 乘法散列法

$$h(k) = \lfloor m \cdot (kA \bmod 1) \rfloor$$

其中先计算 $kA$ 的小数部分（$kA \bmod 1$，即取 $kA$ 中小数点后面的部分），再乘以 $m$ 向下取整。

**$m$ 的选取**：乘法散列法中 $m$ 的选取不像除法那么讲究，通常直接取 $2^p$ 方便位运算实现。

**$A$ 的最优选取**：Knuth 建议 $A = (\sqrt{5} - 1)/2 \approx 0.6180339887$（黄金分割比，与斐波那契数列密切相关）。这个值能让输出分布最均匀。

**位运算实现**（实际系统中非常高效）：设 $w$ 为机器字长（如 64 位），$p = \log_2 m$：
$$h(k) = \text{高} p \text{ 位}(k \cdot s)$$
其中 $s = \lfloor A \cdot 2^w \rfloor$，整个运算只需一次乘法 + 右移，极快。

**乘法散列法的优势**：对 $m$ 的值不敏感，且雪崩效应好（键的任何一位变化都会影响所有乘积位）。

```python
import math

def multiply_hash(key: int, m: int, A: float = 0.6180339887) -> int:
    """
    乘法散列法：h(k) = floor(m * frac(k * A))
    其中 frac(x) = x - floor(x) 取小数部分
    A = (sqrt(5) - 1) / 2 ≈ 0.6180339887（黄金分割比，Knuth 推荐）
    """
    # 计算 k*A 的小数部分
    fractional = (key * A) % 1.0   # Python 浮点 % 1.0 取小数部分
    return int(m * fractional)      # 乘以 m 后向下取整

# 位运算版本（模拟 64 位系统，更高效）
def multiply_hash_bitwise(key: int, m: int) -> int:
    """
    利用整数乘法溢出实现乘法散列法（m 必须是 2 的幂次）
    s = floor((sqrt(5)-1)/2 * 2^64)，Knuth 推荐的魔数
    """
    assert (m & (m - 1)) == 0, "m 必须是 2 的幂次"
    w = 64                                              # 机器字长
    p = int(math.log2(m))                              # m = 2^p
    # 黄金分割比对应的魔数（64位版本）
    s = 0x9E3779B97F4A7C15                             # ≈ (√5-1)/2 * 2^64
    product = (key * s) & 0xFFFFFFFFFFFFFFFF           # 模拟 64 位溢出
    return product >> (w - p)                           # 取高 p 位

# 验证两种实现结果相似
m = 16  # 必须 2 的幂次
keys = [1234, 5678, 9999, 42, 100]
for k in keys:
    h1 = multiply_hash(k, m)
    h2 = multiply_hash_bitwise(k, m)
    print(f"  key={k:4d}: float版={h1:2d}, bitwise版={h2:2d}")
```
```cpp
#include <cstdint>
#include <cmath>
#include <iostream>
using namespace std;

// 浮点版本（直观）
int multiplyHash(uint64_t key, int m, double A = 0.6180339887) {
    double frac = fmod((double)key * A, 1.0);   // 取小数部分
    return (int)(m * frac);
}

// 位运算版本（高效，m 必须是 2 的幂）
int multiplyHashBitwise(uint64_t key, int m) {
    // Knuth 魔数：floor((√5-1)/2 * 2^64) = 0x9E3779B97F4A7C15
    const uint64_t s = 0x9E3779B97F4A7C15ULL;
    // 求 log2(m)，即 m = 2^p 中的 p
    int p = 0;
    int tmp = m;
    while (tmp > 1) { tmp >>= 1; p++; }
    // 64 位自然溢出，取高 p 位
    uint64_t product = key * s;             // 64 位乘法，高位自动溢出丢弃
    return (int)(product >> (64 - p));      // 右移取高 p 位
}

int main() {
    int m = 16;
    int keys[] = {1234, 5678, 9999, 42, 100};
    for (int k : keys) {
        cout << "key=" << k
             << " float=" << multiplyHash(k, m)
             << " bitwise=" << multiplyHashBitwise(k, m) << "\n";
    }
}
```

### 12.2.3 全域哈希（Universal Hashing）

**问题**：任何固定的哈希函数都存在一组"最坏情况输入"——恶意构造的键集合使所有键都映射到同一个槽（对于除法散列，取 $k = 0, m, 2m, \ldots$），导致查找退化为 $O(n)$。

**解决方案**：不固定使用一个哈希函数，而是从一个**函数族（Family）**$\mathcal{H}$ 中**随机选取**一个函数来使用。

**全域哈希族（Universal Hash Family）**的定义：一族哈希函数 $\mathcal{H}$（都将 $U$ 映射到 $\{0,\ldots,m-1\}$），若对任意两个不同的键 $x, y \in U$：

$$\Pr_{h \in \mathcal{H}}[h(x) = h(y)] \leq \frac{1}{m}$$

即随机从 $\mathcal{H}$ 中挑一个函数，任意两个不同键碰撞的概率**不超过随机选槽的概率** $1/m$。

**经典构造**（CLRS 11.3.3）：取 $m$ 为质数，将键 $k$ 表示为 $r+1$ 位的 $m$ 进制数 $\langle k_0, k_1, \ldots, k_r \rangle$，随机选 $\mathbf{a} = \langle a_0, a_1, \ldots, a_r \rangle$（每个 $a_i$ 独立均匀随机选自 $\{0, \ldots, m-1\}$），定义：

$$h_{\mathbf{a}}(k) = \left(\sum_{i=0}^{r} a_i k_i\right) \bmod m$$

**期望冲突次数**：若使用全域哈希，对固定的 $x$，与其碰撞的其他 $n-1$ 个元素的期望碰撞次数：

$$E[\text{与 x 碰撞的元素数}] = (n-1) \cdot \frac{1}{m} = \frac{n-1}{m} < \frac{n}{m} = \alpha$$

因此，不论输入如何构造，期望每个查找的碰撞次数 $< \alpha$，查找期望 $O(1 + \alpha)$。

下面的组件演示了多次随机选取哈希函数后，冲突分布的期望特性：

<div data-component="UniversalHashDemo"></div>

```python
import random

def universal_hash_family(m: int, key_size_bits: int = 32):
    """
    生成一个全域哈希函数（全域性质的随机线性哈希）
    
    构造：将 key 分成 r+1 个 b 位的 "数字"（b = ceil(log2(m))）
    随机选 a_0, ..., a_r，然后 h(k) = (a_0*k_0 + ... + a_r*k_r) mod m
    
    参数：
      m：哈希表大小（应为质数）
      key_size_bits：key 的最大位数
      
    返回：一个随机哈希函数 h(key) -> [0, m-1]
    """
    import math
    # 每个 "数字" 的位数（让每个 a_i 在 [0, m-1] 内）
    b = max(1, math.ceil(math.log2(m + 1)))
    # 需要多少个 "数字"
    r = (key_size_bits + b - 1) // b
    # 随机系数向量 a
    a = [random.randint(0, m - 1) for _ in range(r + 1)]
    
    def h(key: int) -> int:
        # 将 key 分成 r+1 个 b 位的片段
        result = 0
        for i in range(r + 1):
            k_i = (key >> (i * b)) & ((1 << b) - 1)  # 第 i 个 b 位片段
            result = (result + a[i] * k_i) % m
        return result
    
    return h

# 演示：多次随机选择哈希函数，统计碰撞次数
m = 100
keys = list(range(200))        # 200 个键，n/m = 2.0
num_trials = 1000
total_collisions = 0

for _ in range(num_trials):
    h = universal_hash_family(m)
    buckets = [0] * m
    for k in keys:
        buckets[h(k)] += 1
    # 碰撞数 = 每个桶内 C(count, 2) 之和
    collisions = sum(c * (c - 1) // 2 for c in buckets)
    total_collisions += collisions

avg_collisions = total_collisions / num_trials
expected = len(keys) * (len(keys) - 1) / (2 * m)  # 期望碰撞对数
print(f"平均碰撞对数: {avg_collisions:.2f}，理论期望: {expected:.2f}")
```
```cpp
#include <vector>
#include <random>
#include <functional>
#include <iostream>
using namespace std;

// 返回一个全域哈希函数（通用线性哈希族）
// m 为质数，key_bits 为键的位数
function<int(uint64_t)> makeUniversalHash(int m, int key_bits = 32) {
    int b = 1;
    while ((1 << b) < m) b++;          // b = ceil(log2(m))
    int r = (key_bits + b - 1) / b;    // 需要多少个片段

    mt19937 rng(random_device{}());
    uniform_int_distribution<int> dist(0, m - 1);
    vector<int> a(r + 1);
    for (auto& ai : a) ai = dist(rng); // 随机系数

    return [a, b, r, m](uint64_t key) -> int {
        long long result = 0;
        for (int i = 0; i <= r; i++) {
            int k_i = (int)((key >> (i * b)) & ((1LL << b) - 1)); // 第 i 个片段
            result = (result + (long long)a[i] * k_i) % m;
        }
        return (int)result;
    };
}

int main() {
    int m = 100;
    vector<int> keys(200);
    iota(keys.begin(), keys.end(), 0); // keys = [0, 1, ..., 199]
    
    int total_collisions = 0;
    int trials = 1000;
    
    for (int t = 0; t < trials; t++) {
        auto h = makeUniversalHash(m);
        vector<int> buckets(m, 0);
        for (int k : keys) buckets[h(k)]++;
        int c = 0;
        for (int cnt : buckets) c += cnt * (cnt - 1) / 2; // C(cnt, 2)
        total_collisions += c;
    }
    
    double avg = (double)total_collisions / trials;
    double expected = (double)keys.size() * (keys.size() - 1) / (2.0 * m);
    cout << "平均碰撞对: " << avg << ", 期望: " << expected << endl;
}
```

### 12.2.4 密码学哈希 vs 哈希表哈希

这是一个经常被混淆的重要区别：

| 维度 | 哈希表哈希（MurmurHash、FNV、xxHash） | 密码学哈希（SHA-256、SHA-3） |
|------|------|------|
| **核心目标** | 均匀分布、快速计算 | 单向性、抗碰撞攻击 |
| **速度** | 极快（GB/s 级吞吐） | 慢（MB/s 级，刻意增加计算量） |
| **碰撞抵抗** | 只要碰撞少即可，允许被找到 | 计算上不可能找到碰撞（安全假设） |
| **单向性** | 不需要（可从哈希值反推接近的键） | 严格要求（无法从哈希值得到任何关于原像的信息） |
| **雪崩效应** | 需要（分布均匀） | 需要（安全性要求） |
| **应用** | `dict`/`map`、数据分片、布隆过滤器 | 数字签名、密码存储（+salt）、区块链 |

**重要警告**：在哈希表中使用密码学哈希函数是**过度设计**（SHA-256 比 MurmurHash 慢 10–100 倍）；反之，用哈希表哈希来验证文件完整性是**安全漏洞**（MurmurHash 碰撞极易被构造）。

现代语言的随机哈希机制：Python 从 3.3 起，字符串的哈希值在每次启动时加入随机种子（`PYTHONHASHSEED`），以防止"哈希洪水攻击"（Hash Flood Attack）——攻击者通过精心构造大量碰撞键使服务器的 `dict` 操作降级为 $O(n)$，从而进行拒绝服务攻击。

### 12.2.5 字符串哈希：多项式滚动哈希

对于字符串 $s = s_0 s_1 \cdots s_{n-1}$，**多项式哈希（Polynomial Hash）**将字符串看作多项式：

$$h(s) = \left(\sum_{i=0}^{n-1} s_i \cdot p^i\right) \bmod M$$

其中 $p$ 是一个质数（哈希进制，通常取 31 或 131），$M$ 是取模大的质数（通常 $10^9+7$ 或 $10^9+9$）。

**滚动哈希（Rabin-Karp）**：对于长度为 $L$ 的窗口，当窗口右移一位时，新增一个字符、去掉最左字符，$O(1)$ 更新哈希值，是字符串匹配算法的基础。

$$h(s[i+1..i+L]) = \frac{h(s[i..i+L-1]) - s_i \cdot p^0}{p} \cdot p^0 + s_{i+L} \cdot p^{L}$$

（精确实现需要模逆元，实践中用变形形式更简便）

```python
class PolynomialHash:
    """
    多项式滚动哈希（双哈希减少碰撞概率）
    
    使用两套独立的 (p, M) 参数，误判概率从 1/M 降到 1/M^2
    对长度 ~10^6 的字符串，单哈希 10^9 量级的 M 仍有约 10^-3 碰撞率
    双哈希将其降到 ~10^-12，工程上可视为无碰撞
    """
    def __init__(self):
        self.p1, self.M1 = 131, 10**9 + 7    # 第一套参数
        self.p2, self.M2 = 137, 10**9 + 9    # 第二套参数（独立）
    
    def hash_string(self, s: str) -> tuple:
        """计算字符串的双哈希值（O(n)）"""
        h1 = h2 = 0
        p1_pow, p2_pow = 1, 1
        for c in s:
            val = ord(c)
            h1 = (h1 + val * p1_pow) % self.M1
            h2 = (h2 + val * p2_pow) % self.M2
            p1_pow = p1_pow * self.p1 % self.M1
            p2_pow = p2_pow * self.p2 % self.M2
        return (h1, h2)
    
    def rolling_hash(self, s: str, window: int):
        """
        O(n) 滚动哈希，枚举字符串 s 中所有长度为 window 的子串哈希值
        
        应用：Rabin-Karp 字符串匹配，时间 O(n+m) 期望
        """
        n = len(s)
        if window > n:
            return
        
        # 计算初始窗口的哈希值
        h1, h2 = 0, 0
        pw1, pw2 = 1, 1
        for i in range(window):
            h1 = (h1 + ord(s[i]) * pw1) % self.M1
            h2 = (h2 + ord(s[i]) * pw2) % self.M2
            if i < window - 1:
                pw1 = pw1 * self.p1 % self.M1
                pw2 = pw2 * self.p2 % self.M2
        
        yield 0, (h1, h2)   # 第一个窗口
        
        # 预计算 p^window 用于去除最左字符
        # 由于公式对应从低次到高次，直接加新字符更方便
        # 此处用从高次到低次的变形：h(s[i..i+L]) = p * h(s[i+1..i+L]) + s[i]
        # 滚动：h_new = (h_old - s[i] * p^(L-1)) * p + s[i+L]，需要模逆元
        # 以下用"重新计算"演示概念（生产中应用模逆元）
        for i in range(1, n - window + 1):
            # 去掉 s[i-1]（最低幂次），加上 s[i+window-1]
            # 这里省略了精确实现（需模逆元）
            pass

# Rabin-Karp 字符串匹配
def rabin_karp(text: str, pattern: str) -> list:
    """
    Rabin-Karp 算法：字符串匹配
    期望 O(n+m)，最坏 O(nm)（哈希碰撞退化）
    使用双哈希使最坏情况概率极低
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    p, MOD = 131, 10**9 + 7
    
    # 计算模式串哈希值
    pat_hash = 0
    high_pow = 1               # p^(m-1) 用于去掉最高次项
    for i in range(m):
        pat_hash = (pat_hash * p + ord(pattern[i])) % MOD
        if i < m - 1:
            high_pow = high_pow * p % MOD
    
    # 计算初始窗口（text[0..m-1]）哈希值
    win_hash = 0
    for i in range(m):
        win_hash = (win_hash * p + ord(text[i])) % MOD
    
    results = []
    for i in range(n - m + 1):
        if win_hash == pat_hash:
            # 哈希值相同，逐字符验证（避免哈希碰撞误判）
            if text[i:i + m] == pattern:
                results.append(i)
        if i < n - m:
            # 滚动：去掉 text[i]，加入 text[i+m]
            win_hash = (win_hash - ord(text[i]) * high_pow % MOD + MOD) % MOD
            win_hash = (win_hash * p + ord(text[i + m])) % MOD
    
    return results

# 测试
text = "abcababcab"
pattern = "abc"
print(f"'{pattern}' 在 '{text}' 中出现位置: {rabin_karp(text, pattern)}")
# 输出: [0, 5]
```
```cpp
#include <string>
#include <vector>
#include <iostream>
using namespace std;

// Rabin-Karp 字符串匹配
vector<int> rabinKarp(const string& text, const string& pattern) {
    int n = text.size(), m = pattern.size();
    if (m > n) return {};
    
    const long long p = 131, MOD = 1e9 + 7;
    
    // 计算模式串哈希
    long long patHash = 0, highPow = 1;
    for (int i = 0; i < m; i++) {
        patHash = (patHash * p + pattern[i]) % MOD;
        if (i < m - 1) highPow = highPow * p % MOD;  // p^(m-1)
    }
    
    // 初始窗口哈希
    long long winHash = 0;
    for (int i = 0; i < m; i++)
        winHash = (winHash * p + text[i]) % MOD;
    
    vector<int> results;
    for (int i = 0; i <= n - m; i++) {
        if (winHash == patHash) {
            // 哈希相同，逐字符验证
            if (text.substr(i, m) == pattern)
                results.push_back(i);
        }
        if (i < n - m) {
            // 滚动哈希：去掉 text[i]，加入 text[i+m]
            winHash = (winHash - (long long)text[i] * highPow % MOD + MOD) % MOD;
            winHash = (winHash * p + text[i + m]) % MOD;
        }
    }
    return results;
}

int main() {
    string text = "abcababcab", pattern = "abc";
    auto res = rabinKarp(text, pattern);
    cout << "'" << pattern << "' 出现位置: ";
    for (int pos : res) cout << pos << " ";
    cout << endl;  // 输出: 0 5
}
```

---

## 12.3 冲突解决：链地址法（Chaining）

### 12.3.1 结构与基本操作

**链地址法（Separate Chaining）**的核心思想：哈希表的每个槽不是直接存键，而是存一个**链表的头指针**。所有被哈希到同一槽的键形成一条链表（"链"）。

```
哈希表（m=7）：
 slot 0: → [15] → [22] → NULL
 slot 1: → [8]  → NULL
 slot 2: → NULL
 slot 3: → [3]  → [10] → [17] → NULL
 slot 4: → [11] → NULL
 slot 5: → NULL
 slot 6: → [6]  → NULL
```

**三个基本操作**：

| 操作 | 步骤 | 时间复杂度 |
|------|------|------|
| INSERT(k, v) | 计算 h(k)，将 (k,v) 插入 `table[h(k)]` 链表头 | $O(1)$（头插法） |
| SEARCH(k) | 计算 h(k)，遍历 `table[h(k)]` 链表找键 k | $O(1+\alpha)$ 期望 |
| DELETE(k) | 计算 h(k)，从链表中移除键 k 的节点 | $O(1+\alpha)$ 期望 |

**注意插入用头插法**：新元素插到链表头，$O(1)$。若用尾插则需遍历到尾部，$O(\text{链表长度})$。

<div data-component="HashChainVisualizer"></div>

```python
class HashTableChaining:
    """
    链地址法实现的哈希表
    支持任意可哈希的键，使用 Python 内置 hash() + 除法散列
    
    内部结构：
      self.table: list of list，每个槽是一个列表（模拟链表）
      每个元素是 (key, value) 元组
    """
    
    def __init__(self, initial_capacity: int = 16, max_load: float = 1.0):
        """
        initial_capacity：初始槽数（建议为质数或 2 的幂）
        max_load：触发扩容的负载因子阈值（链地址法通常 1.0）
        """
        # 找最近的质数作为初始容量（更均匀分布）
        self._m = initial_capacity
        self._n = 0                        # 存储的元素数
        self._max_load = max_load
        self._table = [[] for _ in range(self._m)]  # 槽数组，每槽一个列表
    
    def _hash(self, key) -> int:
        """除法散列：h(k) = hash(k) mod m"""
        return hash(key) % self._m         # Python hash() 内置处理了随机种子
    
    def insert(self, key, value):
        """
        插入 (key, value)。若 key 已存在则更新值。
        
        边界条件：
          - 键需要是可哈希类型（不可变，如 int, str, tuple）
          - 插入后检查负载因子，超过阈值则自动扩容（rehash）
        """
        slot = self._hash(key)
        chain = self._table[slot]
        
        # 先检查键是否已存在（防止重复插入）
        for i, (k, v) in enumerate(chain):
            if k == key:
                chain[i] = (key, value)    # 更新已有键的值
                return
        
        # 键不存在，头插（插入到列表末尾模拟，O(1) 均摊）
        chain.append((key, value))
        self._n += 1
        
        # 负载因子超阈值，触发 rehash 扩容
        if self._n / self._m > self._max_load:
            self._rehash()
    
    def search(self, key):
        """
        查找键 key 对应的值，未找到返回 None。
        
        期望 O(1 + α)，其中 α = n/m 是负载因子。
        最坏情况 O(n)（全部碰撞到同一槽，极罕见）。
        """
        slot = self._hash(key)
        for k, v in self._table[slot]:
            if k == key:
                return v
        return None
    
    def delete(self, key) -> bool:
        """
        删除键 key。返回 True 表示成功删除，False 表示键不存在。
        链地址法的删除没有 Tombstone 问题（直接移除节点）。
        """
        slot = self._hash(key)
        chain = self._table[slot]
        for i, (k, v) in enumerate(chain):
            if k == key:
                chain.pop(i)               # O(链表长度) 但期望 O(1)
                self._n -= 1
                return True
        return False
    
    def _rehash(self):
        """
        扩容：新槽数 = 旧槽数 * 2（或 * 2 后取下一个质数）
        将所有元素重新哈希插入新表，O(n + m) 时间。
        这是一次性的较大开销，但摊销到每次插入上是 O(1)。
        """
        old_table = self._table
        self._m = self._m * 2              # 翻倍
        self._n = 0
        self._table = [[] for _ in range(self._m)]
        
        # 重新插入所有元素
        for chain in old_table:
            for key, value in chain:
                self.insert(key, value)    # 调用 insert（不再 rehash，因为 n/m 变小了）
    
    def __repr__(self):
        items = []
        for chain in self._table:
            items.extend(f"{k}:{v}" for k, v in chain)
        return f"HashTable(n={self._n}, m={self._m}, α={self._n/self._m:.2f})" \
               f" {{{', '.join(items)}}}"


# 使用演示
ht = HashTableChaining(initial_capacity=7)
ht.insert("apple", 1)
ht.insert("banana", 2)
ht.insert("cherry", 3)
ht.insert("apple", 99)      # 更新 apple
print(ht.search("apple"))   # → 99
print(ht.search("grape"))   # → None
ht.delete("banana")
print(ht)
```
```cpp
#include <vector>
#include <list>
#include <string>
#include <functional>
#include <optional>
#include <iostream>
using namespace std;

template<typename K, typename V>
class HashTableChaining {
    // 每个槽是一个键值对的链表（std::list 支持 O(1) 删除中间节点）
    vector<list<pair<K, V>>> table;
    int m;           // 槽数（capacity）
    int n;           // 存储的元素数
    double maxLoad;  // 扩容阈值

    int hashKey(const K& key) const {
        // 使用 std::hash（对基本类型和 std::string 已内置）
        return (int)(hash<K>{}(key) % (size_t)m);
    }

    void rehash() {
        auto oldTable = table;
        m *= 2;
        n = 0;
        table.assign(m, list<pair<K, V>>{});
        for (auto& chain : oldTable)
            for (auto& [k, v] : chain)
                insert(k, v);  // 重新插入
    }

public:
    HashTableChaining(int capacity = 16, double maxLoad = 1.0)
        : m(capacity), n(0), maxLoad(maxLoad),
          table(capacity, list<pair<K, V>>{}) {}

    void insert(const K& key, const V& value) {
        int slot = hashKey(key);
        // 检查键是否已存在
        for (auto& [k, v] : table[slot]) {
            if (k == key) { v = value; return; } // 更新
        }
        // 头插（push_front 实现头插法，O(1)）
        table[slot].push_front({key, value});
        n++;
        // 检查负载
        if ((double)n / m > maxLoad) rehash();
    }

    optional<V> search(const K& key) const {
        int slot = hashKey(key);
        for (const auto& [k, v] : table[slot])
            if (k == key) return v;
        return nullopt;  // C++17 optional：意味着"没有值"
    }

    bool remove(const K& key) {
        int slot = hashKey(key);
        auto& chain = table[slot];
        for (auto it = chain.begin(); it != chain.end(); ++it) {
            if (it->first == key) {
                chain.erase(it);  // list 的 erase 是 O(1)
                n--;
                return true;
            }
        }
        return false;
    }

    void printStats() const {
        cout << "HashTable(n=" << n << ", m=" << m
             << ", α=" << (double)n/m << ")\n";
    }
};

int main() {
    HashTableChaining<string, int> ht(7);
    ht.insert("apple", 1);
    ht.insert("banana", 2);
    ht.insert("apple", 99);  // 更新
    auto v = ht.search("apple");
    if (v) cout << "apple=" << *v << "\n";  // 99
    ht.remove("banana");
    ht.printStats();
}
```

### 12.3.2 SUHA 下的期望查找复杂度分析

**简单均匀哈希假设（Simple Uniform Hashing Assumption，SUHA）**：每个键等概率独立地被映射到 $m$ 个槽中的任意一个，即对任意键 $k$ 和槽 $j$：$P[h(k) = j] = 1/m$。

**定理**：在 SUHA 下，使用链地址法的哈希表，**不成功查找**的期望时间为 $\Theta(1 + \alpha)$。

**证明思路**：
- 查找键 $k$ 时，访问槽 $h(k)$（一次 $O(1)$ 计算）
- 需要遍历该槽链表（一次比较 = 一次"工作单元"）
- 在 SUHA 下，任意其他键 $k'$ 落入 $h(k)$ 的概率 = $1/m$
- 期望该槽包含的其他键数 = $(n-1)/m < n/m = \alpha$
- 加上初始的 $O(1)$ 哈希计算，总期望 = $\Theta(1 + \alpha)$

**成功查找**的期望比较次数稍微复杂（需要对所有可能被查找的键取平均），结论是 $\Theta(1 + \alpha/2)$，仍然是 $\Theta(1 + \alpha)$。

### 12.3.3 最坏情况分析：$O(n)$ 的退化

尽管期望 $O(1+\alpha)$，链地址法的**最坏情况**是所有 $n$ 个键都映射到同一个槽（这对于固定的哈希函数，被攻击者恶意构造键时是可能的）。

最坏情况下：
- 一个槽的链表长度为 $n$
- 查找某个在链尾的键需要 $O(n)$ 次比较

**防御措施**：
1. 使用全域哈希（12.2.3 节）：随机性保证无论何种输入，期望 $O(1)$
2. Python 3.3+ 的 `PYTHONHASHSEED`：字符串哈希加随机种子
3. 链表换成红黑树（Java 8 的 HashMap）：链表超过 8 个节点时转为红黑树，最坏 $O(\log n)$

### 12.3.4 负载因子控制与动态扩容（Rehash）

**触发扩容的条件**：$\alpha > \alpha_{\max}$（链地址法通常 $\alpha_{\max} = 1.0$ 或 0.75）

**扩容过程（Rehash）**：
1. 分配一个大小约为 $2m$ 的新表
2. 遍历旧表所有链表的所有元素（共 $n$ 个）
3. 将每个元素用新的哈希函数（基于新表大小 $2m$）重新插入新表
4. 释放旧表内存

**摊销分析**：每次插入的摊销 $O(1)$（势能法，势能 = 当前元素数 × 常数）。从上次扩容（表大小 $m/2$，元素数 $m/2$）到下次扩容（表大小 $m$，元素数 $m$），共 $m/2$ 次插入，触发一次代价 $O(m)$ 的 rehash，每次插入的摊销代价 = $O(m)/(m/2) = O(1)$。

**Python `dict` 的扩容策略**：  
- 初始容量 8，扩容因子约 2/3
- 当 $n > 2m/3$（即 $\alpha > 0.667$）时扩容
- CPython 3.6+ 引入"紧凑字典"优化：分离 `indices` 数组（存哈希槽位置）和 `entries` 数组（存实际键值对），保持插入顺序同时节省内存

---

## 12.4 冲突解决：开放寻址法（Open Addressing）

开放寻址法与链地址法的根本区别：**所有元素都存储在哈希表本身中**，不使用额外的链表。当碰撞发生时，通过某种**探测序列（Probe Sequence）**找到下一个空槽。

**关键约束**：开放寻址法要求负载因子 $\alpha < 1$（表不能装满，否则找不到空槽）。

探测函数的一般形式：

$$h(k, i) = (\hat{h}(k) + f(i)) \bmod m, \quad i = 0, 1, 2, \ldots$$

其中 $\hat{h}$ 是辅助哈希函数，$f(i)$ 是探测步长，$i$ 是探测次数（$i = 0$ 时不偏移）。

理想情况：$i = 0, 1, \ldots, m-1$ 时产生 $\{0, 1, \ldots, m-1\}$ 的排列（每个槽都恰好被探测到一次）。

### 12.4.1 线性探测（Linear Probing）

$$h(k, i) = (\hat{h}(k) + i) \bmod m$$

探测序列：$\hat{h}(k), \hat{h}(k)+1, \hat{h}(k)+2, \ldots$（步长为 1，循环）

**优点**：
- 实现最简单
- **缓存友好**：探测的连续位置在内存中相邻，缓存命中率高

**缺点：一次聚集（Primary Clustering）**  
已被占用的槽形成连续"簇"，簇越长，新来的键落在簇中的概率越大，导致簇越来越长，查找性能急剧下降。

**直觉**：想象一排停车位。原本空旷时大家随机停（$\hat{h}(k)$ 均匀）。一旦某段连续停满，后来的车必须继续往后找，使那段更长；下一辆车碰到那段的概率更大……恶性循环。

**探测次数期望**（均匀哈希假设下）：
- 不成功查找：$\approx \frac{1}{2}\left(1 + \frac{1}{(1-\alpha)^2}\right)$
- 成功查找：$\approx \frac{1}{2}\left(1 + \frac{1}{1-\alpha}\right)$

当 $\alpha = 0.9$ 时，不成功查找期望 $\approx 50.5$ 次探测——性能非常差！

### 12.4.2 二次探测（Quadratic Probing）

$$h(k, i) = (\hat{h}(k) + c_1 i + c_2 i^2) \bmod m$$

探测序列：$\hat{h}(k),\ \hat{h}(k)+c_1+c_2,\ \hat{h}(k)+2c_1+4c_2,\ \ldots$

常见选取：$c_1 = c_2 = 1/2, m = 2^p$（此时探测序列覆盖 $m/2$ 个槽）；或 $c_1 = 0, c_2 = 1, m$ 为质数与 $4 \bmod 4 = 3$ 的质数（此时覆盖所有 $m$ 个槽）。

**优点**：缓解了一次聚集（探测步长非常数，不会形成连续簇）

**缺点：二次聚集（Secondary Clustering）**  
两个键若 $\hat{h}(k_1) = \hat{h}(k_2)$（哈希到同一初始槽），它们的探测序列完全相同。因此，一次碰撞的键会形成另一种聚集，但比线性探测的聚集更"散"。

**限制**：若 $m$ 不是质数或 $c_1, c_2$ 选取不当，探测序列可能无法覆盖所有槽，导致即使 $\alpha < 1$ 也可能找不到空槽。

### 12.4.3 双重哈希（Double Hashing）

$$h(k, i) = (h_1(k) + i \cdot h_2(k)) \bmod m$$

其中 $h_1, h_2$ 是两个独立的辅助哈希函数。

**关键要求**：$h_2(k)$ 必须与 $m$ 互质（$\gcd(h_2(k), m) = 1$），才能保证探测序列覆盖所有 $m$ 个槽。

**两种保证互质的方法**：
1. $m$ 取质数，$h_2(k) = 1 + (k \bmod (m-1))$（因为 $1 \leq h_2(k) \leq m-1$ 且 $\gcd(h_2(k), m) = 1$ 当 $m$ 质数时）
2. $m$ 取 $2^p$，$h_2(k)$ 取奇数（奇数与 $2^p$ 互质）

**优点**：探测序列依赖于两个独立哈希值，几乎消除了聚集现象。双重哈希的性能最接近**均匀哈希假设**的理论预测（每个探测序列都是独立随机排列）——这也是为什么双重哈希是开放寻址法中**性能最优**的方案。

**缺点**：
- 两次哈希计算，缓存局部性比线性探测差
- 实现略复杂（需要确保 $h_2(k)$ 与 $m$ 互质）

<div data-component="OpenAddressingProbe"></div>

```python
from enum import Enum

class Slot(Enum):
    EMPTY = "EMPTY"
    DELETED = "DELETED"  # Tombstone 标记（见 12.4.5 节）

class OpenAddressHashTable:
    """
    开放寻址哈希表（支持三种探测方式）
    
    重要设计决策：
      1. DELETED（Tombstone）槽区别于 EMPTY：EMPTY 表示从未使用，DELETED 表示
         曾经使用但已删除。搜索时 DELETED 继续探测，EMPTY 时停止。
      2. 负载因子必须 < 1（通常控制在 ≤ 0.7）
      3. rehash 时将 DELETED 槽清除（减少无效探测）
    """
    
    def __init__(self, capacity: int = 17, mode: str = "double"):
        """
        capacity：槽数（建议质数）
        mode：探测方式，"linear" / "quadratic" / "double"
        """
        assert mode in ("linear", "quadratic", "double")
        self._m = capacity
        self._n = 0        # 有效元素数
        self._mode = mode
        self._keys = [Slot.EMPTY] * self._m
        self._vals = [None] * self._m
    
    def _h1(self, key) -> int:
        """主哈希函数"""
        return hash(key) % self._m
    
    def _h2(self, key) -> int:
        """辅助哈希函数（双重哈希用）"""
        # h2 必须与 m 互质（m 为质数时，h2 在 [1, m-1] 内均与 m 互质）
        return 1 + (hash(key) % (self._m - 1))
    
    def _probe(self, key, i: int) -> int:
        """根据探测模式计算第 i 次探测的槽位"""
        h1 = self._h1(key)
        if self._mode == "linear":
            return (h1 + i) % self._m
        elif self._mode == "quadratic":
            # c1=0, c2=1：h(k,i) = (h1 + i^2) mod m
            return (h1 + i * i) % self._m
        else:  # double hashing
            h2 = self._h2(key)
            return (h1 + i * h2) % self._m
    
    def insert(self, key, value) -> bool:
        """
        插入 (key, value)。
        
        返回 False 表示表已满（极罕见，正常应在高负载前 rehash）。
        插入到 EMPTY 或 DELETED 槽（DELETED 槽可复用）。
        """
        first_deleted = None   # 记录第一个 DELETED 槽（可以插在此处）
        for i in range(self._m):
            slot = self._probe(key, i)
            k = self._keys[slot]
            
            if k is Slot.EMPTY:
                # 找到空槽，插入
                insert_slot = first_deleted if first_deleted is not None else slot
                self._keys[insert_slot] = key
                self._vals[insert_slot] = value
                self._n += 1
                return True
            elif k is Slot.DELETED:
                # 记录第一个 DELETED 槽（后面可能还有相同键）
                if first_deleted is None:
                    first_deleted = slot
            elif k == key:
                # 键已存在，更新值
                self._vals[slot] = value
                return True
        
        if first_deleted is not None:
            # 所有探测位置都非 EMPTY，但有 DELETED 槽可复用
            self._keys[first_deleted] = key
            self._vals[first_deleted] = value
            self._n += 1
            return True
        
        return False  # 表已满
    
    def search(self, key):
        """
        查找键 key。
        
        遇到 EMPTY 停止（该键一定不存在）；
        遇到 DELETED 继续探测（不能停，该键可能在后面）。
        这是 Tombstone 机制的核心逻辑。
        """
        for i in range(self._m):
            slot = self._probe(key, i)
            k = self._keys[slot]
            if k is Slot.EMPTY:
                return None             # 确定不存在
            elif k is Slot.DELETED:
                continue                # 越过 Tombstone 继续探测
            elif k == key:
                return self._vals[slot]  # 找到
        return None
    
    def delete(self, key) -> bool:
        """
        删除键 key（软删除：置为 DELETED/Tombstone）。
        
        ⚠️ 不能直接置为 EMPTY！否则会截断后续键的探测链路，
        导致那些键"消失"（search 遇到 EMPTY 误以为不存在）。
        """
        for i in range(self._m):
            slot = self._probe(key, i)
            k = self._keys[slot]
            if k is Slot.EMPTY:
                return False            # 确定不存在
            elif k is Slot.DELETED:
                continue
            elif k == key:
                self._keys[slot] = Slot.DELETED  # Tombstone！
                self._vals[slot] = None
                self._n -= 1
                return True
        return False
    
    def load_factor(self) -> float:
        return self._n / self._m
    
    def __repr__(self):
        items = [f"{self._keys[i]}:{self._vals[i]}"
                 for i in range(self._m)
                 if self._keys[i] not in (Slot.EMPTY, Slot.DELETED)]
        return (f"OpenAddr({self._mode}, n={self._n}, m={self._m}, "
                f"α={self.load_factor():.2f}) [{', '.join(items)}]")


# 测试 Tombstone 的正确性
ht = OpenAddressHashTable(17, mode="linear")
ht.insert("a", 1)
ht.insert("b", 2)    # 假设 b 与 a 冲突，排在 a 后面
ht.insert("c", 3)    # 假设 c 与 a 冲突，排在 b 后面
ht.delete("b")       # 删除 b（置为 DELETED，不是 EMPTY）
print(ht.search("c"))   # 应输出 3（c 在 b 后面，b 的 DELETED 不影响探测）
print(ht)
```
```cpp
#include <vector>
#include <string>
#include <optional>
#include <iostream>
using namespace std;

enum class SlotState { EMPTY, DELETED, OCCUPIED };

template<typename K, typename V>
class OpenAddressHashTable {
    struct Slot {
        K key;
        V val;
        SlotState state = SlotState::EMPTY;
    };
    
    vector<Slot> table;
    int m, n;
    string mode;  // "linear" / "quadratic" / "double"

    int h1(const K& key) const {
        return (int)(hash<K>{}(key) % (size_t)m);
    }
    int h2(const K& key) const {
        return 1 + (int)(hash<K>{}(key) % (size_t)(m - 1));
    }
    int probe(const K& key, int i) const {
        int base = h1(key);
        if (mode == "linear")    return (base + i) % m;
        if (mode == "quadratic") return (base + i * i) % m;
        // double hashing
        return (base + i * h2(key)) % m;
    }

public:
    OpenAddressHashTable(int capacity = 17, string mode = "double")
        : m(capacity), n(0), mode(mode), table(capacity) {}

    bool insert(const K& key, const V& val) {
        int firstDeleted = -1;
        for (int i = 0; i < m; i++) {
            int slot = probe(key, i);
            if (table[slot].state == SlotState::EMPTY) {
                int ins = (firstDeleted != -1) ? firstDeleted : slot;
                table[ins] = {key, val, SlotState::OCCUPIED};
                n++; return true;
            } else if (table[slot].state == SlotState::DELETED) {
                if (firstDeleted == -1) firstDeleted = slot;
            } else if (table[slot].key == key) {
                table[slot].val = val;  // 更新
                return true;
            }
        }
        if (firstDeleted != -1) {
            table[firstDeleted] = {key, val, SlotState::OCCUPIED};
            n++; return true;
        }
        return false;  // 表满
    }

    optional<V> search(const K& key) const {
        for (int i = 0; i < m; i++) {
            int slot = probe(key, i);
            if (table[slot].state == SlotState::EMPTY) return nullopt;
            if (table[slot].state == SlotState::DELETED) continue;
            if (table[slot].key == key) return table[slot].val;
        }
        return nullopt;
    }

    bool remove(const K& key) {
        for (int i = 0; i < m; i++) {
            int slot = probe(key, i);
            if (table[slot].state == SlotState::EMPTY) return false;
            if (table[slot].state == SlotState::DELETED) continue;
            if (table[slot].key == key) {
                table[slot].state = SlotState::DELETED;  // Tombstone
                n--;
                return true;
            }
        }
        return false;
    }

    double loadFactor() const { return (double)n / m; }
};
```

### 12.4.4 均匀哈希假设下的探测次数期望

在**均匀哈希假设（Uniform Hashing Assumption）**下（每个键的探测序列是 $0 \ldots m-1$ 的等可能随机排列），探测次数期望：

| 操作 | 期望探测次数 | 公式 |
|------|------|------|
| **不成功查找** | $\dfrac{1}{1-\alpha}$ | 几何分布：每个探测有 $1-\alpha$ 的概率找到空槽 |
| **成功查找** | $\dfrac{1}{\alpha} \ln \dfrac{1}{1-\alpha}$ | 对插入时刻取平均 |

**数值对比**：

| $\alpha$ | 不成功查找（期望探测次数） | 成功查找（期望探测次数） |
|------|------|------|
| 0.5 | 2 | 1.39 |
| 0.7 | 3.33 | 2.04 |
| 0.9 | 10 | 3.91 |
| 0.95 | 20 | 4.94 |
| 0.99 | 100 | 6.88 |

**关键结论**：当 $\alpha$ 接近 1 时，探测次数爆炸式增长（从工程角度，$\alpha > 0.7$ 就该扩容了）。

<div data-component="LoadFactorPerformance"></div>

### 12.4.5 开放寻址法删除：Tombstone 机制

**致命错误**：直接将被删除槽设为 EMPTY。

**问题重现**：
```
初始：slot 0: A   slot 1: B   slot 2: EMPTY
（A 和 B 的 h1 均为 0，所以 B 被线性探测到 slot 1）

操作：delete(A) → 直接设 slot 0 为 EMPTY
结果：slot 0: EMPTY   slot 1: B

查找 B：h1(B)=0，探测 slot 0，发现 EMPTY → 返回"不存在"！
但 B 明明在 slot 1！
```

**Tombstone（墓碑）机制**：删除时将槽标记为 `DELETED`（既不是 `EMPTY` 也不是 `OCCUPIED`）。
- **搜索时**：遇到 `DELETED` **继续探测**（不视为结束）；遇到 `EMPTY` 才停止
- **插入时**：遇到 `DELETED` 可以将新元素插入该槽（复用空间）；同时继续探测确认键不重复

**Tombstone 的代价**：随着删除操作增多，`DELETED` 槽越来越多，实际有效元素虽少但探测链变长。**解决方案**：在 rehash 时，`DELETED` 槽当作空槽处理（彻底清除），定期或手动触发 rehash 恢复性能。

**CPython `dict` 的特殊处理**：Python 字典的 key 删除后会在内部数据结构留下 "dummy" 节点（等价于 Tombstone），并在 rehash 时清理。

### 12.4.6 动态扩容（Rehash）与装载策略

**何时触发 rehash**：

| 策略 | 阈值 | 代表系统 |
|------|------|------|
| 链地址法 | $\alpha > 0.75$ | Java HashMap（默认） |
| 线性探测 | $\alpha > 0.5$ | 推荐（一次聚集严重） |
| 双重哈希 | $\alpha > 0.7$ | 推荐（性能平滑下降） |
| Python dict | $\alpha > 0.667$ | CPython |

**扩容策略**：
- 通常翻倍（$m \rightarrow 2m$）：保证摊销 $O(1)$ 插入
- 也可选下一个质数（如 `m = next_prime(2m)`）：更优的分布
- **注意**：不能直接翻倍后继续用旧哈希函数（因为 $h(k) = k \bmod m$ 会变）必须**重新哈希**所有元素

**为什么 Java HashMap 选 0.75**？这是一个在时间（探测次数/链表长度）和空间（内存利用率）之间的经验最优点。$\alpha = 0.75$ 时，链地址法的平均链长 $< 1$，插入/查找期望接近 $O(1)$，同时内存使用率达 75%（不太浪费）。严格来说，0.75 没有严格数学最优性——它是 Knuth 在《计算机程序设计艺术》中推荐的经验值。

---

## 12.5 完美哈希（Perfect Hashing）

### 12.5.1 静态场景下的理想哈希

**完美哈希（Perfect Hashing）**适用于**键集合固定（静态）**的场景：键集合在运行前已知，构造时可以一次性设计一个无冲突的哈希函数。

**完美哈希的定义**：对键集合 $S = \{k_1, \ldots, k_n\}$，找到一个哈希函数 $h$，使得 $h(k_i) \neq h(k_j)$（$i \neq j$），即**零碰撞**。

**代价**：

| 目标 | 代价 |
|------|------|
| 最坏查找 | $O(1)$（无碰撞，直接访问） |
| 构造时间 | $O(n)$ 期望（随机化算法） |
| 空间 | $O(n)$ 期望 |

**应用场景**：
- 编译器/解释器的**关键字表**（Python 的保留关键字 `if, for, while, def, ...` 等 36 个）
- 操作系统的**系统调用表**（固定的 ~300 个系统调用）
- DNS 资源记录的静态缓存
- 任何**只读词典**（构建后只查询、不增删）

### 12.5.2 两级哈希方案（Fredman-Komlós-Szemerédi，FKS 构造）

**FKS 完美哈希**（1984 年）是解决完美哈希问题的经典方法，实现了 $O(n)$ 空间和 $O(1)$ 最坏查找：

**第一级**：用一个大小为 $m = n$ 的哈希表，使用全域哈希函数 $h_1$，将 $n$ 个键映射到 $m$ 个槽（可能有碰撞）。某些槽可能有多个键（设第 $j$ 槽有 $n_j$ 个键）。

**关键引理（Birthday Paradox 的逆向应用）**：若 $m = n$，全域哈希下，碰撞对数的期望 $\leq n(n-1)/(2m) = (n-1)/2 < n/2$。  
更强的结论：选择满足**总碰撞数 $< n$** 的 $h_1$，期望尝试次数为常数（因为随机 $h_1$ 满足此条件的概率 $\geq 1/2$）。

**第二级**：对第一级的每个槽 $j$（含 $n_j$ 个键），使用大小为 $m_j = n_j^2$ 的辅助哈希表，再次随机选全域哈希函数 $h_j^{(2)}$，直到得到该 $n_j$ 个键的无碰撞映射。

**无碰撞的概率（生日悖论）**：选 $n_j$ 个键映射到 $n_j^2$ 个槽，任意两键碰撞的概率：

$$P(\text{至少一次碰撞}) < \binom{n_j}{2} \cdot \frac{1}{n_j^2} = \frac{n_j (n_j-1)}{2n_j^2} < \frac{1}{2}$$

因此，每次随机选 $h_j^{(2)}$ 有至少 $1/2$ 的概率无碰撞，期望选 $\leq 2$ 次即可得到完美哈希。

**总空间**：$m + \sum_j m_j = n + \sum_j n_j^2$。由第一级的碰撞数 $< n$ 可得 $\sum_j \binom{n_j}{2} < n$，即 $\sum_j n_j(n_j-1) < 2n$，故 $\sum_j n_j^2 < 2n + n = 3n$。总空间 $O(n)$。

**查找过程**（完全 $O(1)$）：
```
PERFECT-HASH-SEARCH(key):
    j = h1(key)              // 第一级哈希，O(1)
    i = h2_j(key)            // 第二级哈希，O(1)
    if table2[j][i].key == key:
        return table2[j][i].value
    return NOT_FOUND
```

```python
import random
from typing import List, Tuple, Any, Optional

class PerfectHashTable:
    """
    FKS 两级完美哈希表（静态场景）
    
    构造流程：
      1. 随机选第一级全域哈希 h1，直到总碰撞数 < n
      2. 对每个碰撞桶 j，随机选 h2_j（大小 n_j^2），直到无碰撞
      3. 构造完成后查找是最坏 O(1)
      
    注意：这是教学实现，生产中应使用优化过的版本（如 gperf 工具）
    """
    
    def __init__(self, keys: List[Any], values: List[Any]):
        assert len(keys) == len(values), "键值数量必须相同"
        self.n = len(keys)
        self.m = max(self.n, 1)  # 第一级大小 = n
        
        pairs = list(zip(keys, values))
        
        # ── 步骤 1：找满足总碰撞数 < n 的第一级哈希 ──
        while True:
            self._a1 = random.randint(1, self.m * 100)  # 简单哈希参数
            self._p = self._next_prime(self.m * 10)     # 大质数
            
            buckets = [[] for _ in range(self.m)]
            for k, v in pairs:
                h = self._h1(k)
                buckets[h].append((k, v))
            
            # 统计总碰撞数（= 每桶 C(n_j, 2) 之和）
            total_collisions = sum(len(b) * (len(b) - 1) // 2 for b in buckets)
            if total_collisions < self.n:
                break   # 找到合适的第一级哈希
        
        # ── 步骤 2：对每个碰撞桶构造第二级完美哈希 ──
        self._table2 = []      # 第二级哈希表（列表的列表）
        self._h2_params = []   # 第二级哈希参数 (a, size)
        
        for bucket in buckets:
            nj = len(bucket)
            if nj == 0:
                self._table2.append([])
                self._h2_params.append((0, 0))
                continue
            
            mj = max(nj * nj, 1)  # 第二级大小 = n_j^2
            
            # 随机选 h2_j 直到无碰撞（期望 2 次）
            while True:
                a2 = random.randint(1, mj * 100)
                level2 = [None] * mj
                collision = False
                
                for k, v in bucket:
                    h = (a2 * hash(k)) % mj
                    if level2[h] is not None:
                        collision = True
                        break
                    level2[h] = (k, v)
                
                if not collision:
                    self._table2.append(level2)
                    self._h2_params.append((a2, mj))
                    break
    
    def _next_prime(self, n: int) -> int:
        """找 >= n 的下一个质数（简单暴力实现，仅用于演示）"""
        from sympy import isprime  # 实际中可用 Miller-Rabin
        if n < 2:
            return 2
        n = n | 1  # 从奇数开始
        while True:
            if all(n % i != 0 for i in range(2, int(n**0.5) + 1)):
                return n
            n += 2
    
    def _h1(self, key) -> int:
        """第一级哈希"""
        return (self._a1 * hash(key)) % self.m
    
    def search(self, key) -> Optional[Any]:
        """
        完美哈希查找 - 严格 O(1) 最坏情况
        两次哈希，两次数组访问，无任何循环
        """
        j = self._h1(key)                     # 定位第一级桶
        a2, mj = self._h2_params[j]
        if mj == 0:
            return None
        i = (a2 * hash(key)) % mj             # 定位第二级槽
        entry = self._table2[j][i]
        if entry is not None and entry[0] == key:
            return entry[1]
        return None


# 演示：Python 关键字表的完美哈希
python_keywords = ["if", "else", "elif", "for", "while", "def", "class",
                   "return", "import", "from", "as", "with", "try", "except"]
values = list(range(len(python_keywords)))

pht = PerfectHashTable(python_keywords, values)
print(pht.search("def"))      # → 6（对应的索引值）
print(pht.search("while"))    # → 4
print(pht.search("lambda"))   # → None（不存在）
```
```cpp
#include <vector>
#include <string>
#include <optional>
#include <random>
#include <functional>
#include <iostream>
using namespace std;

// 简化版 FKS 完美哈希（教学用）
template<typename K, typename V>
class PerfectHashTable {
    struct Entry { K key; V val; bool occupied = false; };
    
    int n, m;
    long long a1;                               // 第一级哈希参数
    vector<vector<Entry>> table2;              // 第二级哈希表
    vector<pair<long long, int>> h2Params;     // (a2, mj)

    int h1(const K& key) const {
        return (int)((a1 * (long long)hash<K>{}(key)) % m);
    }

public:
    PerfectHashTable(const vector<K>& keys, const vector<V>& vals) {
        n = m = (int)keys.size();
        if (n == 0) return;
        
        mt19937_64 rng(42);

        // 步骤 1：找总碰撞 < n 的 a1
        while (true) {
            a1 = rng() % (n * 100) + 1;
            vector<vector<pair<K, V>>> buckets(m);
            for (int i = 0; i < n; i++)
                buckets[h1(keys[i])].push_back({keys[i], vals[i]});

            int totalCollisions = 0;
            for (auto& b : buckets)
                totalCollisions += (int)b.size() * ((int)b.size() - 1) / 2;
            if (totalCollisions < n) {
                // 步骤 2：对每个桶构造第二级完美哈希
                table2.resize(m);
                h2Params.resize(m);
                for (int j = 0; j < m; j++) {
                    auto& bucket = buckets[j];
                    int nj = bucket.size();
                    int mj = max(nj * nj, 1);
                    if (nj == 0) { h2Params[j] = {0, 0}; continue; }
                    while (true) {
                        long long a2 = rng() % (mj * 100) + 1;
                        vector<Entry> level2(mj);
                        bool collision = false;
                        for (auto& [k, v] : bucket) {
                            int idx = (int)((a2 * (long long)hash<K>{}(k)) % mj);
                            if (level2[idx].occupied) { collision = true; break; }
                            level2[idx] = {k, v, true};
                        }
                        if (!collision) {
                            table2[j] = level2;
                            h2Params[j] = {a2, mj};
                            break;
                        }
                    }
                }
                break;
            }
        }
    }

    optional<V> search(const K& key) const {
        int j = h1(key);
        auto [a2, mj] = h2Params[j];
        if (mj == 0) return nullopt;
        int i = (int)((a2 * (long long)hash<K>{}(key)) % mj);
        const auto& e = table2[j][i];
        if (e.occupied && e.key == key) return e.val;
        return nullopt;
    }
};

int main() {
    vector<string> kws = {"if", "else", "for", "while", "def", "class", "return"};
    vector<int> ids = {0, 1, 2, 3, 4, 5, 6};
    PerfectHashTable<string, int> pht(kws, ids);
    auto v = pht.search("def");
    if (v) cout << "def → " << *v << "\n";   // 4
    auto u = pht.search("lambda");
    cout << (u ? "found" : "not found") << "\n";  // not found
}
```

---

## 12.6 布隆过滤器（Bloom Filter）

### 12.6.1 结构与设计直觉

**布隆过滤器（Bloom Filter）**是 Burton Howard Bloom 于 1970 年提出的一种**概率型数据结构**，用于回答"**某个元素是否在集合中**"这一判断问题。

**核心特征**：
- **可能假阳性**：有元素**不在集合中但被判断为在集合中**的可能（False Positive）
- **绝无假阴性**：若元素确实在集合中，**一定会被判断为在集合中**（No False Negative）
- **无法删除**（基本版）：插入操作不可逆，删除会影响其他元素
- **极省空间**：相比完整存储所有键，节省 10–100 倍内存

**数据结构**：
- 一个 $m$ 位的**位数组（Bit Array）**，初始全为 0
- $k$ 个**独立的哈希函数** $h_1, h_2, \ldots, h_k$（每个将元素映射到 $[0, m-1]$）

**插入操作**：对元素 $x$，计算 $h_1(x), h_2(x), \ldots, h_k(x)$，将这 $k$ 个位置的位全部置为 1。

**查询操作**：对查询元素 $q$，检查 $h_1(q), h_2(q), \ldots, h_k(q)$ 这 $k$ 个位置：
- 若**全部为 1**：$q$ **可能**在集合中（False Positive 可能）
- 若**存在 0**：$q$ **一定不在**集合中（确定的否定）

**生活比喻**：想象一张"签到表"（位数组），每个人到场时在 $k$ 个随机格子打勾（$k$ 个哈希）。查询某人是否到场时，检查他对应的 $k$ 个格子是否都有勾。若都有勾，他**可能**来过（也可能是别人"碰巧"打了这些格子）；若有格子没勾，他**一定**没来。

<div data-component="BloomFilterDemo"></div>

### 12.6.2 误判率（False Positive Rate）推导

**设**：$m$ 位数组，$k$ 个哈希函数，插入 $n$ 个元素。

**第一步**：插入一个元素后，某一特定位仍为 0 的概率是：

$$P[\text{某位为 0 after 1 element}] = \left(1 - \frac{1}{m}\right)^k \approx e^{-k/m}$$

（每个哈希函数不命中该特定位的概率是 $1 - 1/m$，$k$ 次独立，用 $\lim_{m\to\infty}(1-1/m)^m = e^{-1}$ 近似）

**第二步**：插入 $n$ 个元素后，某一特定位仍为 0 的概率：

$$P[\text{某位为 0 after n elements}] = \left(1 - \frac{1}{m}\right)^{kn} \approx e^{-kn/m}$$

**第三步**：某一特定位为 1 的概率：

$$p = 1 - e^{-kn/m}$$

**第四步**：查询一个不在集合中的元素时，要求 $k$ 个哈希位**全部为 1**（每个独立，概率 $p$）：

$$\boxed{P(\text{false positive}) \approx \left(1 - e^{-kn/m}\right)^k}$$

这个公式非常优美——它只依赖于 $m/n$（每个元素分配的位数，也叫"位宽"）和 $k$（哈希函数个数）。

### 12.6.3 最优哈希函数个数 k 的推导

对给定的 $m/n$，最优 $k$ 使误判率最小化：

令 $f(k) = (1 - e^{-kn/m})^k$，对 $k$ 求导令 $f'(k) = 0$，求解得：

$$k^* = \frac{m}{n} \ln 2 \approx 0.693 \cdot \frac{m}{n}$$

代入误判率公式，最优误判率（当 $k = k^*$）为：

$$P^*(\text{false positive}) \approx \left(\frac{1}{2}\right)^{k^*} = \left(\frac{1}{2}\right)^{(m/n)\ln 2} = \left(0.6185\right)^{m/n}$$

**实用设计表**：

| 目标误判率 $\varepsilon$ | 所需 $m/n$（位/元素）| 最优 $k$ | 说明 |
|------|------|------|------|
| 1% | 9.6 位 | 6 或 7 | 10 位/元素是常用设计 |
| 0.1% | 14.4 位 | 10 | 14-15 位/元素 |
| 0.01% | 19.2 位 | 13 | |
| 0.1% ($10^{-3}$) | 14.4 位 | 10 | |
| $10^{-6}$ | 28.8 位 | 20 | ~4 字节/元素，极低误判率 |

**结论**：存储 $10^8$ 个 URL（平均 60 字节），完整存储需 6 GB；用布隆过滤器（误判率 1%，10 位/元素）只需 **125 MB**（节省约 50 倍），还能保证绝无假阴性。

### 12.6.4 应用：缓存加速、URL 去重、Redis

**1. 数据库缓存穿透防护（Redis + Bloom Filter）**

**问题**：用户请求一个不存在的 ID（如 `user_id = -1`），Redis 缓存没有（因为未缓存"不存在"），请求穿透到数据库，大量这样的请求会打垮数据库（"缓存穿透"攻击）。

**解决方案**：在 Redis 缓存前加一层布隆过滤器，存储所有**合法且存在的** ID。请求先经过布隆过滤器：
- 布隆说"不在集合中" → 该 ID 一定不存在，直接返回空，**无需访问 Redis 和数据库**
- 布隆说"在集合中" → 去 Redis 查（可能是假阳性，但概率很低）

**2. 爬虫 URL 去重**

网络爬虫已访问数十亿 URL，每抓到新 URL 需判断是否已访问过。完整 HashSet 需要 TB 级内存，而布隆过滤器（误判率 0.1%，每 URL 14.4 位）只需 **约 21 GB 处理 1000 亿 URL**，而且：
- 假阳性（已访问但误判为未访问）：会重复抓取（问题不大，幂等操作）
- 假阴性：不存在！一定能抓到真正未访问的 URL

**3. Redis 内置 Bloom Filter（通过 RedisBloom 模块）**

```bash
# 安装 RedisBloom 后
BF.RESERVE myfilter 0.001 1000000   # 误判率 0.1%，预期 100 万元素
BF.ADD myfilter "www.example.com"   # 添加 URL
BF.EXISTS myfilter "www.example.com"  # → 1（存在）
BF.EXISTS myfilter "www.notexist.com" # → 0（不存在）
```

```python
import hashlib
import math
from typing import List

class BloomFilter:
    """
    布隆过滤器实现
    
    使用 SHA-256 截断模拟 k 个独立哈希函数：
      h_i(x) = SHA256(x + str(i)) mod m
    这是工程中常见的"一个哈希函数模拟多个"的技巧，
    实际生产中应使用 MurmurHash3 或 xxHash 等更快的非密码学哈希。
    
    空间表示：Python 的 bytearray（8 位/字节），实际每位对应 1 bit
    更高效的实现应使用位操作（详见下方）
    """
    
    def __init__(self, n: int, epsilon: float = 0.01):
        """
        n：预期插入的元素数
        epsilon：目标误判率（默认 1%）
        
        自动计算最优 m 和 k：
          m = -n * ln(ε) / (ln 2)^2
          k = (m/n) * ln 2
        """
        # 计算最优 m（位数组大小）
        self.m = int(-n * math.log(epsilon) / (math.log(2) ** 2)) + 1
        # 计算最优 k（哈希函数个数）
        self.k = int(self.m / n * math.log(2)) + 1
        # 位数组（用 bytearray 存储，每字节 8 位）
        self._bits = bytearray(math.ceil(self.m / 8))
        self._n_inserted = 0
        
        print(f"布隆过滤器: n={n}, ε={epsilon}")
        print(f"  → 位数组大小 m={self.m:,} 位 ({self.m/8/1024:.1f} KB)")
        print(f"  → 哈希函数个数 k={self.k}")
        print(f"  → 每元素占用 {self.m/n:.1f} 位")
    
    def _get_positions(self, item: str) -> List[int]:
        """
        计算 item 对应的 k 个哈希位置。
        技术：用一个 sha256 生成足够的随机字节，分段截取。
        这比调用 k 次不同哈希函数更高效（一次 SHA 调用）。
        """
        # 使用 SHA256(item || seed) 生成大随机数
        positions = []
        seed = 0
        while len(positions) < self.k:
            h = hashlib.sha256(f"{item}{seed}".encode()).digest()
            # 每 8 字节读取一个 64 位整数，对 m 取模
            for i in range(0, len(h) - 7, 8):
                val = int.from_bytes(h[i:i+8], 'big')
                positions.append(val % self.m)
                if len(positions) >= self.k:
                    break
            seed += 1
        return positions[:self.k]
    
    def _set_bit(self, pos: int):
        """设置第 pos 位为 1"""
        self._bits[pos // 8] |= (1 << (pos % 8))
    
    def _get_bit(self, pos: int) -> bool:
        """读取第 pos 位"""
        return bool(self._bits[pos // 8] & (1 << (pos % 8)))
    
    def add(self, item: str):
        """插入元素：将 k 个哈希位全部置 1"""
        for pos in self._get_positions(item):
            self._set_bit(pos)
        self._n_inserted += 1
    
    def might_contain(self, item: str) -> bool:
        """
        查询元素是否可能在集合中。
        
        返回 True：可能在集合中（可能假阳性）
        返回 False：一定不在集合中（确定无误）
        
        绝无假阴性：若 item 确实已 add，k 个位都已置 1，
        查询时一定返回 True。
        """
        return all(self._get_bit(pos) for pos in self._get_positions(item))
    
    def actual_false_positive_rate(self) -> float:
        """计算当前实际误判率上界（基于插入元素数 n）"""
        p = 1 - math.exp(-self.k * self._n_inserted / self.m)
        return p ** self.k
    
    def memory_usage_bytes(self) -> int:
        return len(self._bits)


# 演示：爬虫 URL 去重
bf = BloomFilter(n=1_000_000, epsilon=0.001)  # 100 万 URL，误判率 0.1%
print()

# 插入 100 万个 URL（模拟）
print("插入 100 万个 URL...")
for i in range(1_000_000):
    bf.add(f"https://example.com/page/{i}")

# 测试假阳性率
false_positives = sum(
    1 for i in range(1_000_000, 1_100_000)  # 未插入的 URL
    if bf.might_contain(f"https://example.com/page/{i}")
)
print(f"误判率 (实测): {false_positives / 100_000:.4f}")
print(f"误判率 (理论): {bf.actual_false_positive_rate():.4f}")
print(f"内存占用: {bf.memory_usage_bytes() / 1024:.1f} KB")
```
```cpp
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <iostream>
using namespace std;

class BloomFilter {
    vector<bool> bits;   // C++ vector<bool> 自动按位存储（紧凑）
    int m;               // 位数组大小
    int k;               // 哈希函数个数
    int nInserted;

    // 使用 h1 * i + h2 的双哈希技巧模拟 k 个独立哈希函数
    // 这是 Kirsch & Mitzenmacher (2006) 证明误判率不增的标准方法
    int getPos(const string& item, int i) const {
        size_t h1 = hash<string>{}(item);
        size_t h2 = hash<string>{}(item + "salt");  // 简单二次哈希
        return (int)((h1 + (size_t)i * h2) % (size_t)m);
    }

public:
    BloomFilter(int n, double epsilon = 0.01) : nInserted(0) {
        // m = -n * ln(ε) / (ln2)^2
        m = (int)(-n * log(epsilon) / (log(2) * log(2))) + 1;
        // k = (m/n) * ln2
        k = (int)((double)m / n * log(2)) + 1;
        bits.assign(m, false);
        
        cout << "BloomFilter: m=" << m << " bits ("
             << m / 8 / 1024 << " KB), k=" << k << "\n";
    }

    void add(const string& item) {
        for (int i = 0; i < k; i++)
            bits[getPos(item, i)] = true;
        nInserted++;
    }

    bool mightContain(const string& item) const {
        for (int i = 0; i < k; i++)
            if (!bits[getPos(item, i)]) return false;  // 有 0 则一定不在
        return true;  // 全 1 则可能在（可能假阳性）
    }

    double theoreticalFPR() const {
        double p = 1.0 - exp(-(double)k * nInserted / m);
        return pow(p, k);
    }
};

int main() {
    BloomFilter bf(1000000, 0.001);  // 100 万元素，误判率 0.1%
    
    // 插入 100 万个 URL
    for (int i = 0; i < 1000000; i++)
        bf.add("https://example.com/page/" + to_string(i));
    
    // 测试假阳性率
    int fp = 0;
    for (int i = 1000000; i < 1100000; i++)
        if (bf.mightContain("https://example.com/page/" + to_string(i)))
            fp++;
    
    cout << "实测误判率: " << (double)fp / 100000 << "\n";
    cout << "理论误判率: " << bf.theoreticalFPR() << "\n";
}
```

---

## 12.7 总结与练习

### 12.7.1 核心知识点回顾

| 知识点 | 关键结论 |
|--------|---------|
| 直接寻址表 | $O(1)$ 所有操作，但空间 $|U|$ 不可行 |
| 哈希 + 链地址法 | 期望 $O(1+\alpha)$，允许 $\alpha > 1$，支持动态扩容 |
| 哈希 + 开放寻址 | 期望 $O(1/(1-\alpha))$，要求 $\alpha < 1$，缓存友好 |
| 线性探测 vs 双重哈希 | 前者缓存友好但聚集严重，后者接近均匀哈希理论 |
| 全域哈希 | 保证期望 $O(1)$ 与输入无关（防攻击） |
| 完美哈希（FKS） | 静态 $O(n)$ 空间 + $O(1)$ 最坏查找 |
| 布隆过滤器 | 概率型：无假阴性，误判率 $\approx (1-e^{-kn/m})^k$ |
| Tombstone 机制 | 开放寻址法删除的必要手段，不可直接置 EMPTY |

### 12.7.2 常见陷阱

1. **开放寻址法忘记 Tombstone**：删除后直接置 EMPTY，后续 search 会找不到存活的键——这是正确性 bug，不只是性能问题。

2. **m 选 2 的幂 + 除法散列**：若键有规律（如所有键都是偶数），低位的均匀性决定了一切。必须加扰动函数（Java HashMap 的做法）或改用质数 m。

3. **布隆过滤器的"假阳性 vs 假阴性"混淆**：布隆过滤器只有假阳性（查到"存在"但实际不存在），**绝无假阴性**（若实际存在，一定被查到）。反方向的结论是错误的。

4. **开放寻址法在 α > 0.7 时的性能崩溃**：不成功查找期望探测次数 $\approx 1/(1-0.7)^2 = 11$ 次，性能急剧下降。必须在此之前 rehash。

5. **哈希函数的"随机性"混淆**：全域哈希的随机性是对哈希函数选取的随机性（防止最坏情况键集），而 Python 的 `PYTHONHASHSEED` 是防止哈希洪水攻击的机制——两者目标不同，不要混淆。

### 12.7.3 面试高频考点

| 题目 | 要点 |
|------|------|
| 哈希表 O(1) 的条件？ | 期望 O(1)，依赖均匀分布假设；全域哈希保证期望，完美哈希保证最坏 |
| 链地址 vs 开放寻址 怎么选？ | 链地址：$\alpha$ 可 > 1，删除简单，CPU 缓存稍差；开放寻址：缓存友好，要求 $\alpha < 1$，删除复杂 |
| Java HashMap 负载因子 0.75 的意义？ | 时间-空间权衡的经验值，链表平均长 < 1，性能稳定 |
| 布隆过滤器的使用场景？ | 缓存穿透防护、URL 去重、大规模集合成员查询（允许极低误判率、不允许假阴性） |
| 设计 LRU Cache（LeetCode 146）？ | HashMap + 双向链表：O(1) 访问（map 找节点）+ O(1) 移动（双向链表移到头部） |

### 12.7.4 LeetCode 经典题目

| 题号 | 题目 | 知识点 |
|------|------|------|
| #1 | Two Sum | 哈希表补数查找 |
| #49 | Group Anagrams | 字符串哈希（排序后作键） |
| #128 | Longest Consecutive Sequence | 哈希集合 O(n) 解法 |
| #146 | LRU Cache | HashMap + 双向链表 |
| #705/706 | 设计哈希集合/映射 | 手写链地址法哈希表 |
| #290 | Word Pattern | 双向哈希映射 |

> **💡 思考题**：  
> 1. 全域哈希和密码学哈希都有"随机性"，但目标完全不同。全域哈希关心的是什么？密码学哈希关心的是什么？为什么哈希表中使用 SHA-256 是过度设计？  
> 2. 若你要为一个存储 10 亿条记录的系统设计布隆过滤器，且允许误判率为 0.01%，需要多少 MB 内存？最优哈希个数是多少？（$\ln 2 \approx 0.693$）  
> 3. 为什么 FKS 完美哈希的第二级大小选 $n_j^2$ 而不是 $n_j$？ 如果选 $n_j$，第二级无碰撞的概率是多少？

---

> **参考资料**：  
> - 《算法导论》（CLRS）第 4 版 Chapter 11（Hash Tables）  
> - Knuth, D.E. *The Art of Computer Programming, Vol. 3: Sorting and Searching* (3.4 Hashing)  
> - Bloom, B.H. *Space/Time Trade-offs in Hash Coding with Allowable Errors* (1970)  
> - Fredman, M.; Komlós, J.; Szemerédi, E. *Storing a Sparse Table with O(1) Worst Case Access Time* (1984)  
> - Kirsch, A.; Mitzenmacher, M. *Less Hashing, Same Performance* (2006)（双哈希模拟 k 个哈希函数）  
> - MIT 6.006 Lecture 4, 8: Hashing  
> - Python `dict` 内部实现：https://github.com/python/cpython/blob/main/Objects/dictobject.c
