"use client";

import { useState } from "react";

/* ─── Operation definitions ─────────────────────────────────────────────── */

type Op = "AND" | "OR" | "XOR" | "NOT" | "LSH" | "RSH";

const OPERATIONS: { key: Op; label: string; symbol: string; unary?: boolean }[] = [
  { key: "AND", label: "AND（与）", symbol: "&" },
  { key: "OR", label: "OR（或）", symbol: "|" },
  { key: "XOR", label: "XOR（异或）", symbol: "^" },
  { key: "NOT", label: "NOT（取反）", symbol: "~", unary: true },
  { key: "LSH", label: "左移", symbol: "<<" },
  { key: "RSH", label: "右移", symbol: ">>" },
];

const TRUTH_TABLES: Record<Op, { a?: number; b?: number; r: number }[]> = {
  AND: [
    { a: 0, b: 0, r: 0 }, { a: 0, b: 1, r: 0 },
    { a: 1, b: 0, r: 0 }, { a: 1, b: 1, r: 1 },
  ],
  OR: [
    { a: 0, b: 0, r: 0 }, { a: 0, b: 1, r: 1 },
    { a: 1, b: 0, r: 1 }, { a: 1, b: 1, r: 1 },
  ],
  XOR: [
    { a: 0, b: 0, r: 0 }, { a: 0, b: 1, r: 1 },
    { a: 1, b: 0, r: 1 }, { a: 1, b: 1, r: 0 },
  ],
  NOT: [
    { a: 0, r: 1 }, { a: 1, r: 0 },
  ],
  LSH: [],
  RSH: [],
};

/* ─── Helpers ───────────────────────────────────────────────────────────── */

function toBits(val: number): number[] {
  const bits: number[] = [];
  for (let i = 7; i >= 0; i--) {
    bits.push((val >> i) & 1);
  }
  return bits;
}

function bitsToDec(bits: number[]): number {
  return bits.reduce((acc, b) => (acc << 1) | b, 0);
}

function applyOp(a: number, b: number, op: Op): number {
  switch (op) {
    case "AND": return a & b;
    case "OR":  return a | b;
    case "XOR": return a ^ b;
    case "NOT": return (~a) & 0xff;
    case "LSH": return (a << 1) & 0xff;
    case "RSH": return (a >> 1) & 0xff;
  }
}

/* ─── Bit box component ────────────────────────────────────────────────── */

function BitBox({
  bit,
  variant,
  highlight,
  animate,
}: {
  bit: number;
  variant: "inputA" | "inputB" | "result";
  highlight?: boolean;
  animate?: boolean;
}) {
  const colorMap = {
    inputA: {
      bg: bit === 1
        ? "bg-emerald-100 dark:bg-emerald-900/60 border-emerald-400 dark:border-emerald-600 text-emerald-800 dark:text-emerald-200"
        : "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-500 dark:text-slate-400",
      label: "A",
    },
    inputB: {
      bg: bit === 1
        ? "bg-blue-100 dark:bg-blue-900/60 border-blue-400 dark:border-blue-600 text-blue-800 dark:text-blue-200"
        : "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-500 dark:text-slate-400",
      label: "B",
    },
    result: {
      bg: bit === 1
        ? "bg-amber-100 dark:bg-amber-900/60 border-amber-400 dark:border-amber-600 text-amber-800 dark:text-amber-200"
        : "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-500 dark:text-slate-400",
      label: "R",
    },
  };

  const { bg } = colorMap[variant];

  return (
    <div
      className={`
        w-9 h-9 flex items-center justify-center rounded-md border-2
        font-mono text-sm font-bold transition-all duration-300
        ${bg}
        ${highlight ? "ring-2 ring-amber-400 dark:ring-amber-500 scale-110" : ""}
        ${animate ? "animate-pulse" : ""}
      `}
    >
      {bit}
    </div>
  );
}

/* ─── Main component ───────────────────────────────────────────────────── */

export function BitwiseOperationsDemo() {
  const [inputA, setInputA] = useState("10110101");
  const [inputB, setInputB] = useState("11001010");
  const [op, setOp] = useState<Op>("AND");
  const [activeBit, setActiveBit] = useState<number | null>(null);
  const [animating, setAnimating] = useState(false);

  const opInfo = OPERATIONS.find((o) => o.key === op)!;
  const isUnary = op === "NOT";
  const isShift = op === "LSH" || op === "RSH";

  // Parse inputs
  const aVal = parseInt(inputA, 2) & 0xff;
  const bVal = parseInt(inputB, 2) & 0xff;
  const aValid = /^[01]{1,8}$/.test(inputA);
  const bValid = /^[01]{1,8}$/.test(inputB);

  const aBits = toBits(aVal);
  const bBits = toBits(bVal);
  const result = applyOp(aVal, bVal, op);
  const resultBits = toBits(result);

  // Shift-specific: the result bits shifted by one position
  const shiftedBits: number[] = op === "LSH"
    ? [...aBits.slice(1), 0]
    : op === "RSH"
      ? [0, ...aBits.slice(0, 7)]
      : [];

  // Animation: step through each bit position
  const handleAnimate = () => {
    if (animating) return;
    setAnimating(true);
    setActiveBit(0);
    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step >= 8) {
        clearInterval(interval);
        setTimeout(() => {
          setActiveBit(null);
          setAnimating(false);
        }, 600);
      } else {
        setActiveBit(step);
      }
    }, 400);
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Title */}
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">
        位运算演示器
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">
        Bitwise Operations Demo — 逐位可视化 AND / OR / XOR / NOT / Shift
      </p>

      {/* ── Input Section ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
        {/* Input A */}
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            输入 A（8-bit 二进制）
          </label>
          <input
            type="text"
            value={inputA}
            maxLength={8}
            onChange={(e) => setInputA(e.target.value.replace(/[^01]/g, ""))}
            className="w-full px-3 py-2 font-mono text-lg rounded-lg border
              border-slate-300 dark:border-slate-600
              bg-white dark:bg-slate-800
              text-slate-800 dark:text-slate-100
              focus:outline-none focus:ring-2 focus:ring-cyan-400"
            placeholder="例如 10110101"
          />
          {aValid && (
            <span className="text-xs text-slate-500 dark:text-slate-400 mt-1 inline-block">
              = {aVal}（十进制）
            </span>
          )}
          {!aValid && inputA.length > 0 && (
            <span className="text-xs text-red-500 mt-1 inline-block">
              请输入 0 和 1 组成的二进制数（最多 8 位）
            </span>
          )}
        </div>

        {/* Input B (hidden for unary / shift) */}
        {!isUnary && !isShift && (
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
              输入 B（8-bit 二进制）
            </label>
            <input
              type="text"
              value={inputB}
              maxLength={8}
              onChange={(e) => setInputB(e.target.value.replace(/[^01]/g, ""))}
              className="w-full px-3 py-2 font-mono text-lg rounded-lg border
                border-slate-300 dark:border-slate-600
                bg-white dark:bg-slate-800
                text-slate-800 dark:text-slate-100
                focus:outline-none focus:ring-2 focus:ring-cyan-400"
              placeholder="例如 11001010"
            />
            {bValid && (
              <span className="text-xs text-slate-500 dark:text-slate-400 mt-1 inline-block">
                = {bVal}（十进制）
              </span>
            )}
            {!bValid && inputB.length > 0 && (
              <span className="text-xs text-red-500 mt-1 inline-block">
                请输入 0 和 1 组成的二进制数（最多 8 位）
              </span>
            )}
          </div>
        )}
      </div>

      {/* ── Operation Selector ────────────────────────────────────────── */}
      <div className="mb-5">
        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          选择运算
        </label>
        <div className="flex flex-wrap gap-2">
          {OPERATIONS.map((o) => (
            <button
              key={o.key}
              onClick={() => { setOp(o.key); setActiveBit(null); }}
              className={`
                px-3 py-1.5 rounded-lg text-sm font-medium border transition-all
                ${op === o.key
                  ? "bg-cyan-600 text-white border-cyan-600 shadow-md"
                  : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600 hover:border-cyan-400"}
              `}
            >
              {o.label} <span className="font-mono ml-1 opacity-70">{o.symbol}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── Bit-by-bit Visualization ──────────────────────────────────── */}
      <div className="mb-5 overflow-x-auto">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
            逐位运算结果
          </span>
          <button
            onClick={handleAnimate}
            disabled={animating}
            className="px-3 py-1 text-xs rounded-md bg-cyan-100 dark:bg-cyan-900/40
              text-cyan-700 dark:text-cyan-300 border border-cyan-300 dark:border-cyan-700
              hover:bg-cyan-200 dark:hover:bg-cyan-800/50 transition-colors
              disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {animating ? "动画播放中…" : "▶ 逐位动画"}
          </button>
        </div>

        {/* Bit position labels */}
        <div className="flex items-center gap-1 mb-2 pl-24">
          {[7, 6, 5, 4, 3, 2, 1, 0].map((pos) => (
            <div
              key={pos}
              className="w-9 text-center text-[10px] text-slate-400 dark:text-slate-500 font-mono"
            >
              {pos}
            </div>
          ))}
        </div>

        {/* Row A */}
        <div className="flex items-center gap-1 mb-1.5">
          <span className="w-20 text-right text-sm font-mono text-emerald-700 dark:text-emerald-400 pr-3 shrink-0">
            A
          </span>
          {aBits.map((bit, i) => (
            <BitBox
              key={i}
              bit={bit}
              variant="inputA"
              highlight={activeBit === i}
              animate={animating && activeBit === i}
            />
          ))}
          <span className="ml-3 text-xs text-slate-500 dark:text-slate-400 font-mono shrink-0">
            = {aVal}
          </span>
        </div>

        {/* Row B (or shift arrow) */}
        {isShift ? (
          <div className="flex items-center gap-1 mb-1.5">
            <span className="w-20 text-right text-sm font-mono text-slate-500 dark:text-slate-400 pr-3 shrink-0">
              {op === "LSH" ? "左移 1 位" : "右移 1 位"}
            </span>
            {shiftedBits.map((bit, i) => (
              <BitBox
                key={i}
                bit={bit}
                variant="inputB"
                highlight={activeBit === i}
              />
            ))}
          </div>
        ) : !isUnary ? (
          <div className="flex items-center gap-1 mb-1.5">
            <span className="w-20 text-right text-sm font-mono text-blue-700 dark:text-blue-400 pr-3 shrink-0">
              B
            </span>
            {bBits.map((bit, i) => (
              <BitBox
                key={i}
                bit={bit}
                variant="inputB"
                highlight={activeBit === i}
                animate={animating && activeBit === i}
              />
            ))}
            <span className="ml-3 text-xs text-slate-500 dark:text-slate-400 font-mono shrink-0">
              = {bVal}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-1 mb-1.5">
            <span className="w-20 text-right text-sm font-mono text-slate-500 dark:text-slate-400 pr-3 shrink-0">
              ~A（取反）
            </span>
            {aBits.map((_, i) => (
              <BitBox
                key={i}
                bit={aBits[i] ^ 1}
                variant="inputB"
                highlight={activeBit === i}
              />
            ))}
          </div>
        )}

        {/* Operator line */}
        <div className="flex items-center gap-1 mb-1.5">
          <span className="w-20 text-right text-sm font-mono text-slate-500 dark:text-slate-400 pr-3 shrink-0">
            {opInfo.symbol}
          </span>
          <div className="flex gap-1">
            {[7, 6, 5, 4, 3, 2, 1, 0].map((pos) => (
              <div
                key={pos}
                className="w-9 h-5 flex items-center justify-center text-xs font-mono text-slate-400 dark:text-slate-500"
              >
                {opInfo.symbol}
              </div>
            ))}
          </div>
        </div>

        {/* Separator */}
        <div className="flex items-center gap-1 mb-1.5 pl-24">
          <div className="flex gap-1">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="w-9 border-b-2 border-slate-300 dark:border-slate-600" />
            ))}
          </div>
        </div>

        {/* Result row */}
        <div className="flex items-center gap-1">
          <span className="w-20 text-right text-sm font-mono text-amber-700 dark:text-amber-400 font-bold pr-3 shrink-0">
            结果
          </span>
          {(isShift ? shiftedBits : resultBits).map((bit, i) => (
            <BitBox
              key={i}
              bit={bit}
              variant="result"
              highlight={activeBit === i}
              animate={animating && activeBit === i}
            />
          ))}
          <span className="ml-3 text-xs text-slate-500 dark:text-slate-400 font-mono shrink-0">
            = {isShift ? bitsToDec(shiftedBits) : result}
          </span>
        </div>
      </div>

      {/* ── Decimal Summary ───────────────────────────────────────────── */}
      <div className="mb-5 p-4 rounded-lg bg-white/60 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
        <div className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          十进制等价表达
        </div>
        <div className="font-mono text-sm text-slate-800 dark:text-slate-200 space-y-1">
          {isUnary ? (
            <>
              <div>
                <span className="text-emerald-600 dark:text-emerald-400">A</span> = {aVal}
              </div>
              <div>
                <span className="text-amber-600 dark:text-amber-400">~A</span> = (~{aVal}) &amp; 0xFF ={" "}
                <span className="font-bold">{result}</span>
              </div>
            </>
          ) : isShift ? (
            <>
              <div>
                <span className="text-emerald-600 dark:text-emerald-400">A</span> = {aVal}
              </div>
              <div>
                <span className="text-amber-600 dark:text-amber-400">
                  {op === "LSH" ? "A << 1" : "A >> 1"}
                </span>{" "}
                = {bitsToDec(shiftedBits)}{" "}
                <span className="text-slate-400 dark:text-slate-500">
                  （{op === "LSH" ? "相当于 ×2" : "相当于 ÷2 取整"}）
                </span>
              </div>
            </>
          ) : (
            <>
              <div>
                <span className="text-emerald-600 dark:text-emerald-400">A</span> = {aVal},{" "}
                <span className="text-blue-600 dark:text-blue-400">B</span> = {bVal}
              </div>
              <div>
                <span className="text-emerald-600 dark:text-emerald-400">{aVal}</span>{" "}
                <span className="text-slate-500">{opInfo.symbol}</span>{" "}
                <span className="text-blue-600 dark:text-blue-400">{bVal}</span>{" "}
                = <span className="font-bold">{result}</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Truth Table ───────────────────────────────────────────────── */}
      {TRUTH_TABLES[op].length > 0 && (
        <div>
          <div className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            真值表 — {opInfo.label}
          </div>
          <div className="inline-block overflow-hidden rounded-lg border border-slate-200 dark:border-slate-700">
            <table className="text-sm font-mono">
              <thead>
                <tr className="bg-slate-100 dark:bg-slate-800">
                  {isUnary ? (
                    <>
                      <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
                        A
                      </th>
                      <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
                        ~A
                      </th>
                    </>
                  ) : (
                    <>
                      <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
                        A
                      </th>
                      <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
                        B
                      </th>
                      <th className="px-4 py-2 text-left text-slate-600 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
                        {opInfo.symbol} 结果
                      </th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {TRUTH_TABLES[op].map((row, i) => (
                  <tr
                    key={i}
                    className={i % 2 === 0
                      ? "bg-white dark:bg-slate-900"
                      : "bg-slate-50 dark:bg-slate-800/50"}
                  >
                    {isUnary ? (
                      <>
                        <td className="px-4 py-1.5 border-b border-slate-100 dark:border-slate-800">
                          <span className={row.a === 1 ? "text-emerald-600 dark:text-emerald-400 font-bold" : ""}>
                            {row.a}
                          </span>
                        </td>
                        <td className="px-4 py-1.5 border-b border-slate-100 dark:border-slate-800">
                          <span className={row.r === 1 ? "text-amber-600 dark:text-amber-400 font-bold" : ""}>
                            {row.r}
                          </span>
                        </td>
                      </>
                    ) : (
                      <>
                        <td className="px-4 py-1.5 border-b border-slate-100 dark:border-slate-800">
                          <span className={row.a === 1 ? "text-emerald-600 dark:text-emerald-400 font-bold" : ""}>
                            {row.a}
                          </span>
                        </td>
                        <td className="px-4 py-1.5 border-b border-slate-100 dark:border-slate-800">
                          <span className={row.b === 1 ? "text-blue-600 dark:text-blue-400 font-bold" : ""}>
                            {row.b}
                          </span>
                        </td>
                        <td className="px-4 py-1.5 border-b border-slate-100 dark:border-slate-800">
                          <span className={row.r === 1 ? "text-amber-600 dark:text-amber-400 font-bold" : ""}>
                            {row.r}
                          </span>
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
