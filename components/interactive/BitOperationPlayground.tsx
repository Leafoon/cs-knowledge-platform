"use client";

import React, { useState, useCallback } from "react";

// 将 32-bit 整数转为 8 位分组的二进制字符串（含符号扩展）
function toBin32(n: number): string {
  const unsigned = n >>> 0; // 转无符号 32 位
  return unsigned.toString(2).padStart(32, "0");
}

// 显示为 16 位（教学用，避免 32 位太宽）
function toBin16(n: number): string {
  const unsigned = n & 0xffff;
  return unsigned.toString(2).padStart(16, "0");
}

function toHex(n: number): string {
  return "0x" + (n >>> 0).toString(16).toUpperCase().padStart(8, "0");
}

// ── 位运算操作定义 ────────────────────────────────────────────────────────────
interface BitOp {
  id: string;
  label: string;
  symbol: string;
  desc: string;
  compute: (a: number, b: number) => number;
  needsB: boolean;
  tip: string;
}

const BIT_OPS: BitOp[] = [
  {
    id: "and",
    label: "按位与",
    symbol: "a & b",
    desc: "两位都为 1 才为 1",
    compute: (a, b) => a & b,
    needsB: true,
    tip: "用途：取指定位（掩码操作）、判断奇偶 (n & 1)",
  },
  {
    id: "or",
    label: "按位或",
    symbol: "a | b",
    desc: "任意一位为 1 即为 1",
    compute: (a, b) => a | b,
    needsB: true,
    tip: "用途：置位操作（将某位设为 1）",
  },
  {
    id: "xor",
    label: "按位异或",
    symbol: "a ^ b",
    desc: "两位不同则为 1，相同为 0",
    compute: (a, b) => a ^ b,
    needsB: true,
    tip: "用途：翻转位、无临时变量交换、找只出现一次的数",
  },
  {
    id: "not",
    label: "按位取反",
    symbol: "~a",
    desc: "所有位取反（0→1，1→0）",
    compute: (a) => ~a,
    needsB: false,
    tip: "注意：~n = -(n+1)，因为 CPU 用补码",
  },
  {
    id: "shl",
    label: "左移",
    symbol: "a << b",
    desc: "所有位左移 b 位，右补 0",
    compute: (a, b) => a << Math.min(b, 31),
    needsB: true,
    tip: "a << b 等价于 a × 2^b（高位溢出丢弃）",
  },
  {
    id: "shr",
    label: "算术右移",
    symbol: "a >> b",
    desc: "所有位右移 b 位，左补符号位",
    compute: (a, b) => a >> Math.min(b, 31),
    needsB: true,
    tip: "a >> b 等价于 a ÷ 2^b（向下取整）",
  },
  {
    id: "ushr",
    label: "逻辑右移",
    symbol: "a >>> b",
    desc: "所有位右移 b 位，左补 0",
    compute: (a, b) => (a >>> Math.min(b, 31)),
    needsB: true,
    tip: "无符号右移，负数变正（Java/JS 有 >>>，C++ 用 unsigned）",
  },
];

// ── 常用位技巧 ────────────────────────────────────────────────────────────────
interface Trick {
  label: string;
  expr: (n: number) => string;
  result: (n: number) => string;
  desc: string;
}

function isPowerOf2(n: number) {
  return n > 0 && (n & (n - 1)) === 0;
}

function countOnes(n: number) {
  let count = 0;
  let x = n >>> 0;
  while (x) {
    x &= x - 1;
    count++;
  }
  return count;
}

function lowbit(n: number) {
  return n & -n;
}

const TRICKS: Trick[] = [
  {
    label: "判断奇偶",
    expr: (n) => `${n} & 1 = ${n & 1}`,
    result: (n) => (n & 1 ? "奇数" : "偶数"),
    desc: "最低位为 1 → 奇数",
  },
  {
    label: "判断 2 的幂",
    expr: (n) => `${n} & (${n}-1) = ${n & (n - 1)}`,
    result: (n) => (isPowerOf2(n) ? "✅ 是 2 的幂" : "❌ 不是 2 的幂"),
    desc: "2 的幂二进制只有一个 1",
  },
  {
    label: "清最低位 1",
    expr: (n) => `${n} & (${n}-1) = ${n & (n - 1)}`,
    result: (n) => `→ ${(n & (n - 1)).toString(2).padStart(8,'0')}`,
    desc: "Brian Kernighan：用于 popcount",
  },
  {
    label: "取最低位 1（lowbit）",
    expr: (n) => `${n} & (-${n}) = ${lowbit(n)}`,
    result: (n) => `= ${lowbit(n).toString(2).padStart(8,'0')}₂`,
    desc: "树状数组（BIT）的核心操作",
  },
  {
    label: "计数 1 的个数",
    expr: (n) => `popcount(${n})`,
    result: (n) => `= ${countOnes(n)} 个 1`,
    desc: "Brian Kernighan 法：循环次数=1的个数",
  },
  {
    label: "乘以 2",
    expr: (n) => `${n} << 1 = ${n << 1}`,
    result: (n) => `= ${n << 1}`,
    desc: "左移 1 位等价于 ×2（速度更快）",
  },
  {
    label: "除以 2",
    expr: (n) => `${n} >> 1 = ${n >> 1}`,
    result: (n) => `= ${n >> 1}`,
    desc: "右移 1 位等价于 ÷2（向下取整）",
  },
  {
    label: "快速幂（3^13 % 1000）",
    expr: () => `base=3, exp=13=1101₂`,
    result: () => `= ${[...Array(14)].reduce((acc, _, i) => i === 0 ? { r: 1, b: 3 } : { r: (13 >> (i-1)) & 1 ? acc.r * acc.b % 1000 : acc.r, b: acc.b * acc.b % 1000 }, { r: 1, b: 3 }).r}`,
    desc: "逐位处理 exp，O(log exp) 次乘法",
  },
];

// ── 位显示组件 ────────────────────────────────────────────────────────────────
function BinaryDisplay({
  value,
  label,
  color,
  bits = 16,
}: {
  value: number;
  label: string;
  color: string;
  bits?: number;
}) {
  const bin = bits === 16 ? toBin16(value) : toBin32(value);
  const groups = bin.match(/.{1,4}/g) ?? [];

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3">
      <div className="flex justify-between items-center mb-2">
        <span className={`text-xs font-mono font-semibold ${color}`}>{label}</span>
        <div className="flex gap-2 text-xs text-text-secondary font-mono">
          <span>十进制: <span className={color}>{value}</span></span>
          <span>十六进制: <span className={color}>{toHex(value)}</span></span>
        </div>
      </div>
      <div className="flex gap-1.5 flex-wrap">
        {groups.map((grp, gi) => (
          <div key={gi} className="flex gap-0.5">
            {grp.split("").map((bit, bi) => {
              const pos = (groups.length - 1 - gi) * 4 + (3 - bi); // 位索引（从低位到高位）
              return (
                <div
                  key={bi}
                  title={`第 ${pos} 位（2^${pos} = ${Math.pow(2, pos)}）`}
                  className={`w-6 h-7 rounded flex items-center justify-center text-xs font-mono font-bold cursor-help transition-colors
                    ${bit === "1"
                      ? `bg-gradient-to-b ${color.includes("blue") ? "from-blue-500/40 to-blue-600/20 text-blue-700 dark:text-blue-200" : color.includes("emerald") ? "from-emerald-500/40 to-emerald-600/20 text-emerald-700 dark:text-emerald-200" : "from-violet-500/40 to-violet-600/20 text-violet-700 dark:text-violet-200"}`
                      : "bg-bg-tertiary text-text-tertiary"
                    }`}
                >
                  {bit}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="flex gap-1.5 mt-0.5 flex-wrap">
        {groups.map((grp, gi) => (
          <div key={gi} className="flex gap-0.5">
            {grp.split("").map((_, bi) => {
              const pos = (groups.length - 1 - gi) * 4 + (3 - bi);
              return (
                <div key={bi} className="w-6 text-center text-[9px] text-text-tertiary font-mono">
                  {pos}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── 主组件 ────────────────────────────────────────────────────────────────────
export default function BitOperationPlayground() {
  const [numA, setNumA] = useState(42);
  const [numB, setNumB] = useState(15);
  const [selectedOp, setSelectedOp] = useState<string>("and");
  const [tab, setTab] = useState<"calc" | "tricks">("calc");

  const op = BIT_OPS.find((o) => o.id === selectedOp)!;
  const result = op.compute(numA, numB);

  const handleInput = useCallback((setter: (v: number) => void, raw: string) => {
    const v = parseInt(raw, 10);
    if (!isNaN(v) && v >= -32768 && v <= 65535) setter(v);
    else if (raw === "" || raw === "-") setter(0);
  }, []);

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-6 my-6 shadow-sm">
      {/* 标题 */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-emerald-500/20 flex items-center justify-center text-xl">
          🔢
        </div>
        <div>
          <h3 className="font-bold text-text-primary text-base">位运算交互演示</h3>
          <p className="text-xs text-text-secondary">输入数字，实时查看二进制表示与各种位运算结果</p>
        </div>
        <div className="ml-auto flex rounded-lg overflow-hidden border border-border-subtle text-xs">
          <button onClick={() => setTab("calc")}
            className={`px-3 py-1.5 font-medium transition-colors ${tab === "calc" ? "bg-emerald-500/30 text-emerald-700 dark:text-emerald-200" : "bg-bg-tertiary text-text-secondary hover:text-text-secondary"}`}>
            计算器
          </button>
          <button onClick={() => setTab("tricks")}
            className={`px-3 py-1.5 font-medium transition-colors ${tab === "tricks" ? "bg-emerald-500/30 text-emerald-700 dark:text-emerald-200" : "bg-bg-tertiary text-text-secondary hover:text-text-secondary"}`}>
            常用技巧
          </button>
        </div>
      </div>

      {/* ──── 计算器视图 ──── */}
      {tab === "calc" && (
        <div className="space-y-4">
          {/* 操作选择 */}
          <div className="flex flex-wrap gap-2">
            {BIT_OPS.map((o) => (
              <button
                key={o.id}
                onClick={() => setSelectedOp(o.id)}
                className={`px-3 py-1.5 rounded-lg border text-xs font-mono font-medium transition-all ${selectedOp === o.id
                  ? "bg-emerald-500/25 border-emerald-400/50 text-emerald-700 dark:text-emerald-200"
                  : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-secondary"
                  }`}
              >
                {o.symbol}
              </button>
            ))}
          </div>

          {/* 操作说明 */}
          <div className="rounded-xl border border-emerald-400/20 bg-emerald-500/5 px-4 py-2.5 flex items-start gap-3">
            <span className="text-emerald-400 text-lg mt-0.5">ℹ</span>
            <div>
              <div className="text-sm font-semibold text-emerald-700 dark:text-emerald-200">{op.label}（{op.symbol}）</div>
              <div className="text-xs text-text-secondary mt-0.5">{op.desc}</div>
              <div className="text-xs text-emerald-400/80 mt-1">💡 {op.tip}</div>
            </div>
          </div>

          {/* 输入 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-text-secondary mb-1.5 block">操作数 a</label>
              <input
                type="number" value={numA} min={-32768} max={65535}
                onChange={(e) => handleInput(setNumA, e.target.value)}
                className="w-full bg-bg-tertiary border border-border-subtle rounded-lg px-3 py-2 text-blue-600 dark:text-blue-300 font-mono text-sm focus:outline-none focus:border-blue-500 dark:focus:border-blue-400/60"
              />
            </div>
            {op.needsB && (
              <div>
                <label className="text-xs text-text-secondary mb-1.5 block">操作数 b</label>
                <input
                  type="number" value={numB} min={0} max={31}
                  onChange={(e) => handleInput(setNumB, e.target.value)}
                  className="w-full bg-bg-tertiary border border-border-subtle rounded-lg px-3 py-2 text-violet-600 dark:text-violet-300 font-mono text-sm focus:outline-none focus:border-violet-500 dark:focus:border-violet-400/60"
                />
              </div>
            )}
          </div>

          {/* 二进制显示 */}
          <div className="space-y-2">
            <BinaryDisplay value={numA} label="a" color="text-blue-500 dark:text-blue-300" />
            {op.needsB && (
              <BinaryDisplay value={numB & 0xffff} label="b" color="text-violet-500 dark:text-violet-300" />
            )}
            {/* 分隔线 */}
            <div className="flex items-center gap-2 px-2">
              <div className="flex-1 border-t border-border-subtle" />
              <span className="text-text-secondary font-mono text-sm font-bold">{op.symbol}</span>
              <div className="flex-1 border-t border-border-subtle" />
            </div>
            <BinaryDisplay value={result} label={op.symbol} color="text-emerald-300" />
          </div>

          {/* 结果摘要 */}
          <div className="rounded-lg bg-bg-tertiary px-4 py-3 border border-border-subtle font-mono text-sm">
            <span className="text-blue-500 dark:text-blue-300">{numA}</span>
            {op.needsB && <><span className="text-text-tertiary mx-2">{op.symbol.replace("a", "").replace("b", "")}</span><span className="text-violet-500 dark:text-violet-300">{numB}</span></>}
            <span className="text-text-tertiary mx-2">=</span>
            <span className="text-emerald-300 font-bold">{result}</span>
            <span className="text-text-tertiary ml-3 text-xs">
              ({toBin16(result >>> 0).replace(/(.{4})/g, "$1 ").trim()}₂)
            </span>
          </div>
        </div>
      )}

      {/* ──── 常用技巧视图 ──── */}
      {tab === "tricks" && (
        <div>
          {/* 输入数字 */}
          <div className="mb-4">
            <label className="text-xs text-text-secondary mb-1.5 block">输入一个整数（观察各技巧的结果）</label>
            <input
              type="number" value={numA} min={-1000} max={1000}
              onChange={(e) => handleInput(setNumA, e.target.value)}
              className="w-52 bg-bg-tertiary border border-border-subtle rounded-lg px-3 py-2 text-blue-600 dark:text-blue-300 font-mono text-sm focus:outline-none focus:border-blue-500 dark:focus:border-blue-400/60"
            />
          </div>

          <BinaryDisplay value={numA} label={`当前值：${numA}`} color="text-blue-500 dark:text-blue-300" />

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
            {TRICKS.map((trick, i) => (
              <div key={i} className="rounded-xl border border-border-subtle bg-bg-tertiary p-3">
                <div className="text-xs font-semibold text-text-primary mb-1">{trick.label}</div>
                <div className="font-mono text-xs text-emerald-400 mb-0.5">{trick.expr(numA)}</div>
                <div className="font-mono text-xs text-blue-300 mb-1.5">{trick.result(numA)}</div>
                <div className="text-[11px] text-text-tertiary">{trick.desc}</div>
              </div>
            ))}
          </div>

          {/* 快速幂演示 */}
          <div className="mt-4 rounded-xl border border-amber-400/20 bg-amber-500/5 p-4">
            <div className="text-sm font-bold text-amber-300 mb-2">⚡ 快速幂演示：{numA}^10 mod 1000</div>
            <div className="font-mono text-xs text-text-secondary space-y-1">
              <div>exp = 10 = <span className="text-amber-300">1010</span>₂</div>
              {(() => {
                const steps: string[] = [];
                let base = ((numA % 1000) + 1000) % 1000;
                let result = 1;
                let exp = 10;
                let step = 0;
                while (exp > 0) {
                  if (exp & 1) {
                    steps.push(`第${step}位=1：result = ${result} × ${base} mod 1000 = ${result * base % 1000}`);
                    result = result * base % 1000;
                  } else {
                    steps.push(`第${step}位=0：跳过（base=${base}）`);
                  }
                  base = base * base % 1000;
                  exp >>= 1;
                  step++;
                }
                return steps.map((s, i) => <div key={i} className="text-text-secondary">{s}</div>);
              })()}
              <div className="text-emerald-300 font-bold mt-1">
                最终结果 = {(() => {
                  let base = ((numA % 1000) + 1000) % 1000;
                  let result = 1;
                  let exp = 10;
                  while (exp > 0) {
                    if (exp & 1) result = result * base % 1000;
                    base = base * base % 1000;
                    exp >>= 1;
                  }
                  return result;
                })()}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
