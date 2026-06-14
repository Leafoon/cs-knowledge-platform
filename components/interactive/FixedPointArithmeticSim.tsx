"use client";

import { useState, useMemo } from "react";

/* ── helpers ────────────────────────────────────────────────────────── */

/** Pad a binary string to `n` bits (unsigned). */
function padBin(val: number, n: number): string {
  return val.toString(2).padStart(n, "0").slice(-n);
}

/** Absolute-value binary of `|v|` in `n` bits (for sign-magnitude). */
function absBin(v: number, n: number): string {
  return padBin(Math.abs(v), n - 1);
}

/** True when the decimal value is representable in n-bit two's complement. */
function inRange(v: number, n: number): boolean {
  const lo = -(1 << (n - 1));
  const hi = (1 << (n - 1)) - 1;
  return v >= lo && v <= hi;
}

/** Return the n-bit unsigned representation used internally. */
function toUnsigned(v: number, n: number): number {
  return v < 0 ? (1 << n) + v : v;
}

/* ── four code representations ──────────────────────────────────────── */

function getOriginalCode(v: number, n: number): string {
  // 原码: sign bit + magnitude
  if (v === 0) return "0".repeat(n);
  const sign = v < 0 ? "1" : "0";
  return sign + absBin(v, n);
}

function getOnesComplement(v: number, n: number): string {
  // 反码: positive same as 原码; negative = invert all bits of abs
  if (v === 0) return "0".repeat(n);
  if (v > 0) return padBin(v, n);
  // v < 0: invert bits of abs(v) then set sign bit
  const pos = padBin(Math.abs(v), n - 1);
  const inverted = pos
    .split("")
    .map((b) => (b === "0" ? "1" : "0"))
    .join("");
  return "1" + inverted;
}

function getTwosComplement(v: number, n: number): string {
  // 补码: use modular arithmetic
  const u = toUnsigned(v, n);
  return padBin(u, n);
}

function getExcessCode(v: number, n: number): string {
  // 移码: add bias 2^(n-1), then represent as unsigned
  const bias = 1 << (n - 1);
  const encoded = v + bias;
  return padBin(encoded, n);
}

/* ── conversion step explanations ───────────────────────────────────── */

function originalCodeSteps(v: number, n: number): string[] {
  if (v === 0) return [`0 的原码有 +0 和 -0 两种表示，这里统一用全 0`];
  const sign = v < 0 ? "1 (负)" : "0 (正)";
  const mag = Math.abs(v).toString(2).padStart(n - 1, "0");
  return [
    `符号位 = ${sign}`,
    `数值部分 = |${v}| 的二进制 = ${mag}`,
    `原码 = 符号位 + 数值部分 = ${getOriginalCode(v, n)}`,
  ];
}

function onesComplementSteps(v: number, n: number): string[] {
  if (v === 0) return [`0 的反码统一用全 0 表示`];
  if (v > 0)
    return [
      `正数的反码与原码相同`,
      `反码 = ${getOnesComplement(v, n)}`,
    ];
  const absB = absBin(v, n);
  const inverted = absB
    .split("")
    .map((b) => (b === "0" ? "1" : "0"))
    .join("");
  return [
    `负数的反码 = 原码的符号位不变，数值位按位取反`,
    `|${v}| 二进制 = ${absB}`,
    `数值位取反 → ${inverted}`,
    `反码 = 1${inverted}`,
  ];
}

function twosComplementSteps(v: number, n: number): string[] {
  if (v >= 0)
    return [
      `正数的补码与原码相同`,
      `补码 = ${getTwosComplement(v, n)}`,
    ];
  // negative: ones' complement + 1
  const oc = getOnesComplement(v, n);
  const tc = getTwosComplement(v, n);
  return [
    `负数的补码 = 反码 + 1`,
    `反码 = ${oc}`,
    `最低位加 1 → ${tc}`,
    `补码 = ${tc}`,
  ];
}

function excessCodeSteps(v: number, n: number): string[] {
  const bias = 1 << (n - 1);
  const encoded = v + bias;
  return [
    `移码 = 补码的符号位取反（等价于真值 + 偏置值 ${bias}）`,
    `${v} + ${bias} = ${encoded}`,
    `${encoded} 的二进制 = ${padBin(encoded, n)}`,
    `移码 = ${getExcessCode(v, n)}`,
  ];
}

/* ── addition helpers ───────────────────────────────────────────────── */

interface AddStep {
  bitIdx: number;
  aBit: string;
  bBit: string;
  carryIn: string;
  sum: string;
  carryOut: string;
}

function performAddition(
  a: number,
  b: number,
  n: number
): { steps: AddStep[]; result: string; carryOut: string; overflow: boolean } {
  const aBin = getTwosComplement(a, n);
  const bBin = getTwosComplement(b, n);

  const steps: AddStep[] = [];
  let carry = 0;

  for (let i = n - 1; i >= 0; i--) {
    const aBit = parseInt(aBin[i], 2);
    const bBit = parseInt(bBin[i], 2);
    const total = aBit + bBit + carry;
    const sumBit = total % 2;
    const carryOut = Math.floor(total / 2);
    steps.push({
      bitIdx: i,
      aBit: String(aBit),
      bBit: String(bBit),
      carryIn: String(carry),
      sum: String(sumBit),
      carryOut: String(carryOut),
    });
    carry = carryOut;
  }

  const resultUnsigned = toUnsigned(a, n) + toUnsigned(b, n);
  const resultBin = padBin(resultUnsigned, n);
  const cout = String(carry);

  // Overflow: both operands same sign, result different sign
  const aSign = aBin[0];
  const bSign = bBin[0];
  const rSign = resultBin[0];
  const overflow = aSign === bSign && aSign !== rSign;

  return { steps, result: resultBin, carryOut: cout, overflow };
}

/* ── bit-box renderer ───────────────────────────────────────────────── */

function BitBoxes({
  bits,
  signOnly = false,
}: {
  bits: string;
  signOnly?: boolean;
}) {
  return (
    <div className="flex gap-0.5">
      {bits.split("").map((b, i) => (
        <span
          key={i}
          className={`inline-flex items-center justify-center w-7 h-7 rounded text-xs font-bold font-mono
            ${i === 0
              ? "bg-red-200 text-red-800 dark:bg-red-900 dark:text-red-200"
              : "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
            }
            border border-slate-300 dark:border-slate-600`}
        >
          {b}
        </span>
      ))}
    </div>
  );
}

/* ── main component ─────────────────────────────────────────────────── */

export function FixedPointArithmeticSim() {
  const [bits, setBits] = useState<8 | 16>(8);
  const [inputVal, setInputVal] = useState<string>("-42");
  const [tab, setTab] = useState<"repr" | "add">("repr");

  // Addition state
  const [addA, setAddA] = useState<string>("-25");
  const [addB, setAddB] = useState<string>("-30");

  const n = bits;
  const lo = -(1 << (n - 1));
  const hi = (1 << (n - 1)) - 1;

  const parsedVal = useMemo(() => {
    const v = parseInt(inputVal, 10);
    return isNaN(v) ? null : v;
  }, [inputVal]);

  const parsedA = useMemo(() => {
    const v = parseInt(addA, 10);
    return isNaN(v) ? null : v;
  }, [addA]);

  const parsedB = useMemo(() => {
    const v = parseInt(addB, 10);
    return isNaN(v) ? null : v;
  }, [addB]);

  /* ── representation data ─────────────────────────────────────────── */
  const reprData = useMemo(() => {
    if (parsedVal === null || !inRange(parsedVal, n)) return null;
    return {
      original: getOriginalCode(parsedVal, n),
      onesComp: getOnesComplement(parsedVal, n),
      twosComp: getTwosComplement(parsedVal, n),
      excess: getExcessCode(parsedVal, n),
      stepsOriginal: originalCodeSteps(parsedVal, n),
      stepsOnes: onesComplementSteps(parsedVal, n),
      stepsTwos: twosComplementSteps(parsedVal, n),
      stepsExcess: excessCodeSteps(parsedVal, n),
    };
  }, [parsedVal, n]);

  /* ── addition data ───────────────────────────────────────────────── */
  const addData = useMemo(() => {
    if (parsedA === null || parsedB === null) return null;
    if (!inRange(parsedA, n) || !inRange(parsedB, n)) return null;
    return performAddition(parsedA, parsedB, n);
  }, [parsedA, parsedB, n]);

  const addSumDecimal = useMemo(() => {
    if (parsedA === null || parsedB === null) return null;
    return parsedA + parsedB;
  }, [parsedA, parsedB]);

  /* ── flags ───────────────────────────────────────────────────────── */
  const flags = useMemo(() => {
    if (!addData || parsedA === null || parsedB === null) return null;
    const resultBin = addData.result;
    const resultVal = addData.overflow
      ? null
      : (() => {
          const u = parseInt(resultBin, 2);
          return u >= (1 << (n - 1)) ? u - (1 << n) : u;
        })();

    const ZF = resultBin.split("").every((b) => b === "0") ? 1 : 0;
    const SF = parseInt(resultBin[0], 2);
    const CF = parseInt(addData.carryOut, 2);
    const OF = addData.overflow ? 1 : 0;
    return { ZF, SF, CF, OF, resultVal };
  }, [addData, parsedA, parsedB, n]);

  /* ── render ──────────────────────────────────────────────────────── */
  return (
    <div className="my-8 p-6 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-slate-900 dark:to-rose-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-1 text-slate-800 dark:text-slate-100">
        定点数表示与运算模拟器
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
        探索原码、反码、补码、移码的表示与补码加法运算
      </p>

      {/* ── controls ────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-end gap-4 mb-5">
        {/* bit-width toggle */}
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
            位宽
          </label>
          <div className="flex rounded-lg overflow-hidden border border-slate-300 dark:border-slate-600">
            {([8, 16] as const).map((b) => (
              <button
                key={b}
                onClick={() => setBits(b)}
                className={`px-4 py-1.5 text-sm font-medium transition-colors
                  ${bits === b
                    ? "bg-rose-500 text-white"
                    : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                  }`}
              >
                {b}-bit
              </button>
            ))}
          </div>
        </div>

        {/* decimal input */}
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
            十进制值 ({lo} ~ {hi})
          </label>
          <input
            type="number"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            min={lo}
            max={hi}
            className="w-28 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600
              bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100
              font-mono text-sm focus:outline-none focus:ring-2 focus:ring-rose-400"
          />
        </div>
      </div>

      {/* ── tabs ─────────────────────────────────────────────────────── */}
      <div className="flex gap-1 mb-5 border-b border-slate-200 dark:border-slate-700">
        {[
          { id: "repr" as const, label: "编码表示" },
          { id: "add" as const, label: "补码加法" },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors
              ${tab === t.id
                ? "border-rose-500 text-rose-600 dark:text-rose-400"
                : "border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
              }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Representation Tab ──────────────────────────────────────── */}
      {tab === "repr" && (
        <div>
          {parsedVal === null ? (
            <p className="text-sm text-red-500">请输入一个合法的整数。</p>
          ) : !inRange(parsedVal, n) ? (
            <p className="text-sm text-red-500">
              {parsedVal} 超出 {n}-bit 表示范围 [{lo}, {hi}]。
            </p>
          ) : reprData ? (
            <div className="space-y-5">
              {/* code type cards */}
              {[
                {
                  name: "原码",
                  nameEn: "Sign-Magnitude",
                  bits: reprData.original,
                  steps: reprData.stepsOriginal,
                  desc: "最高位为符号位，其余位为数值的绝对值",
                },
                {
                  name: "反码",
                  nameEn: "Ones' Complement",
                  bits: reprData.onesComp,
                  steps: reprData.stepsOnes,
                  desc: "正数同原码；负数 = 符号位不变 + 数值位按位取反",
                },
                {
                  name: "补码",
                  nameEn: "Two's Complement",
                  bits: reprData.twosComp,
                  steps: reprData.stepsTwos,
                  desc: "正数同原码；负数 = 反码 + 1（计算机内部使用）",
                },
                {
                  name: "移码",
                  nameEn: "Excess / Bias",
                  bits: reprData.excess,
                  steps: reprData.stepsExcess,
                  desc: `补码的符号位取反（真值 + 偏置值 ${1 << (n - 1)}）`,
                },
              ].map((code) => (
                <div
                  key={code.name}
                  className="rounded-lg bg-white/70 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 p-4"
                >
                  <div className="flex flex-wrap items-center gap-3 mb-2">
                    <span className="text-sm font-bold text-slate-700 dark:text-slate-200">
                      {code.name}
                    </span>
                    <span className="text-xs text-slate-400 dark:text-slate-500">
                      {code.nameEn}
                    </span>
                    <span className="ml-auto text-xs text-slate-400 dark:text-slate-500 max-w-xs text-right hidden sm:block">
                      {code.desc}
                    </span>
                  </div>

                  {/* bit boxes */}
                  <BitBoxes bits={code.bits} />

                  {/* binary string */}
                  <p className="mt-2 font-mono text-sm text-slate-600 dark:text-slate-300">
                    {code.bits}
                  </p>

                  {/* conversion steps */}
                  <details className="mt-2 group">
                    <summary className="text-xs text-rose-600 dark:text-rose-400 cursor-pointer select-none hover:underline">
                      查看转换步骤
                    </summary>
                    <ul className="mt-1 space-y-0.5 text-xs text-slate-600 dark:text-slate-400 font-mono">
                      {code.steps.map((s, i) => (
                        <li key={i} className="pl-3 border-l-2 border-rose-300 dark:border-rose-700">
                          {s}
                        </li>
                      ))}
                    </ul>
                  </details>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}

      {/* ── Addition Tab ────────────────────────────────────────────── */}
      {tab === "add" && (
        <div>
          <div className="flex flex-wrap items-end gap-4 mb-5">
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                操作数 A ({lo} ~ {hi})
              </label>
              <input
                type="number"
                value={addA}
                onChange={(e) => setAddA(e.target.value)}
                min={lo}
                max={hi}
                className="w-28 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600
                  bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100
                  font-mono text-sm focus:outline-none focus:ring-2 focus:ring-rose-400"
              />
            </div>
            <span className="text-lg font-bold text-slate-400 dark:text-slate-500 pb-1">+</span>
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                操作数 B ({lo} ~ {hi})
              </label>
              <input
                type="number"
                value={addB}
                onChange={(e) => setAddB(e.target.value)}
                min={lo}
                max={hi}
                className="w-28 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600
                  bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100
                  font-mono text-sm focus:outline-none focus:ring-2 focus:ring-rose-400"
              />
            </div>
          </div>

          {parsedA === null || parsedB === null ? (
            <p className="text-sm text-red-500">请输入合法的整数。</p>
          ) : !inRange(parsedA, n) || !inRange(parsedB, n) ? (
            <p className="text-sm text-red-500">
              操作数超出 {n}-bit 表示范围 [{lo}, {hi}]。
            </p>
          ) : addData ? (
            <div className="space-y-4">
              {/* operands in complement form */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {[
                  { label: `A = ${parsedA}`, bits: getTwosComplement(parsedA, n) },
                  { label: `B = ${parsedB}`, bits: getTwosComplement(parsedB, n) },
                ].map((op) => (
                  <div
                    key={op.label}
                    className="rounded-lg bg-white/70 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 p-3"
                  >
                    <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                      {op.label} 的补码
                    </p>
                    <BitBoxes bits={op.bits} />
                    <p className="mt-1 font-mono text-sm text-slate-600 dark:text-slate-300">
                      {op.bits}
                    </p>
                  </div>
                ))}
              </div>

              {/* addition process */}
              <div className="rounded-lg bg-white/70 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 p-4 overflow-x-auto">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-3">
                  逐位加法过程（从低位到高位）
                </p>

                {/* column headers — rendered right-to-left matching bit positions */}
                <table className="font-mono text-xs w-full min-w-[360px]">
                  <thead>
                    <tr className="text-slate-400 dark:text-slate-500">
                      <th className="text-left pr-3 py-0.5">位</th>
                      {[...Array(n)].map((_, i) => (
                        <th key={i} className="w-7 text-center py-0.5">
                          {n - 1 - i}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {/* A */}
                    <tr>
                      <td className="text-slate-500 dark:text-slate-400 pr-3 py-0.5">A</td>
                      {addData.steps.map((s, i) => (
                        <td
                          key={i}
                          className={`text-center py-0.5 ${
                            i === 0
                              ? "text-red-600 dark:text-red-400 font-bold"
                              : "text-blue-600 dark:text-blue-400"
                          }`}
                        >
                          {s.aBit}
                        </td>
                      ))}
                    </tr>
                    {/* B */}
                    <tr>
                      <td className="text-slate-500 dark:text-slate-400 pr-3 py-0.5">B</td>
                      {addData.steps.map((s, i) => (
                        <td
                          key={i}
                          className={`text-center py-0.5 ${
                            i === 0
                              ? "text-red-600 dark:text-red-400 font-bold"
                              : "text-blue-600 dark:text-blue-400"
                          }`}
                        >
                          {s.bBit}
                        </td>
                      ))}
                    </tr>
                    {/* Carry in */}
                    <tr className="border-t border-dashed border-slate-300 dark:border-slate-600">
                      <td className="text-slate-400 dark:text-slate-500 pr-3 py-0.5 text-[10px]">
                        C<sub>in</sub>
                      </td>
                      {addData.steps.map((s, i) => (
                        <td
                          key={i}
                          className="text-center py-0.5 text-amber-500 dark:text-amber-400"
                        >
                          {s.carryIn}
                        </td>
                      ))}
                    </tr>
                    {/* Sum */}
                    <tr className="border-t-2 border-slate-300 dark:border-slate-600">
                      <td className="text-slate-600 dark:text-slate-300 pr-3 py-0.5 font-bold">
                        Sum
                      </td>
                      {addData.steps.map((s, i) => (
                        <td
                          key={i}
                          className={`text-center py-0.5 font-bold ${
                            i === 0
                              ? "text-red-600 dark:text-red-400"
                              : "text-blue-700 dark:text-blue-300"
                          }`}
                        >
                          {s.sum}
                        </td>
                      ))}
                    </tr>
                    {/* Carry out */}
                    <tr>
                      <td className="text-slate-400 dark:text-slate-500 pr-3 py-0.5 text-[10px]">
                        C<sub>out</sub>
                      </td>
                      {addData.steps.map((s, i) => (
                        <td
                          key={i}
                          className="text-center py-0.5 text-amber-500 dark:text-amber-400"
                        >
                          {s.carryOut}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>

                {/* final carry */}
                <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  最终进位 C<sub>out</sub> ={" "}
                  <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                    {addData.carryOut}
                  </span>
                </p>
              </div>

              {/* result */}
              <div className="rounded-lg bg-white/70 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 p-4">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">
                  运算结果
                </p>
                <div className="flex flex-wrap items-center gap-3 mb-2">
                  <BitBoxes bits={addData.result} />
                  <span className="font-mono text-sm text-slate-600 dark:text-slate-300">
                    = {addData.result}
                  </span>
                </div>
                {flags && (
                  <p className="text-sm text-slate-700 dark:text-slate-200">
                    作为补码解读 ={" "}
                    <span className="font-mono font-bold">
                      {flags.resultVal !== null
                        ? flags.resultVal
                        : "溢出 (Overflow)"}
                    </span>
                    {addSumDecimal !== null && (
                      <span className="text-slate-400 dark:text-slate-500 ml-2">
                        （算术值: {parsedA} + {parsedB} = {addSumDecimal}
                        {addData.overflow &&
                        !inRange(addSumDecimal, n)
                          ? "，超出范围"
                          : ""}
                        ）
                      </span>
                    )}
                  </p>
                )}
              </div>

              {/* flags */}
              {flags && (
                <div className="rounded-lg bg-white/70 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 p-4">
                  <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-3">
                    标志位 (Flags)
                  </p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      {
                        name: "ZF (零标志)",
                        value: flags.ZF,
                        desc: flags.ZF ? "结果为 0" : "结果非 0",
                      },
                      {
                        name: "SF (符号标志)",
                        value: flags.SF,
                        desc: flags.SF ? "结果为负" : "结果非负",
                      },
                      {
                        name: "CF (进位标志)",
                        value: flags.CF,
                        desc: flags.CF ? "最高位有进位" : "无进位",
                      },
                      {
                        name: "OF (溢出标志)",
                        value: flags.OF,
                        desc: flags.OF ? "溢出！" : "无溢出",
                      },
                    ].map((f) => (
                      <div
                        key={f.name}
                        className={`rounded-lg p-3 text-center border ${
                          f.value
                            ? f.name.startsWith("OF") && f.value
                              ? "bg-red-50 dark:bg-red-950 border-red-300 dark:border-red-700"
                              : "bg-amber-50 dark:bg-amber-950 border-amber-300 dark:border-amber-700"
                            : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700"
                        }`}
                      >
                        <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">
                          {f.name}
                        </p>
                        <p
                          className={`text-2xl font-bold font-mono ${
                            f.value
                              ? f.name.startsWith("OF") && f.value
                                ? "text-red-600 dark:text-red-400"
                                : "text-amber-600 dark:text-amber-400"
                              : "text-slate-400 dark:text-slate-500"
                          }`}
                        >
                          {f.value}
                        </p>
                        <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">
                          {f.desc}
                        </p>
                      </div>
                    ))}
                  </div>

                  {/* overflow explanation */}
                  {addData.overflow && (
                    <div className="mt-3 p-3 rounded-lg bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800">
                      <p className="text-xs text-red-700 dark:text-red-300 font-medium mb-1">
                        溢出检测
                      </p>
                      <p className="text-xs text-red-600 dark:text-red-400 font-mono">
                        符号位 A[{getTwosComplement(parsedA, n)[0]}] = B[
                        {getTwosComplement(parsedB, n)[0]}]
                        ，但结果符号位 = {addData.result[0]}
                        <br />
                        两个同号数相加，结果符号不同 → 溢出
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}

      {/* legend */}
      <div className="mt-5 flex flex-wrap items-center gap-4 text-[11px] text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-4 rounded bg-red-200 dark:bg-red-900 border border-slate-300 dark:border-slate-600" />
          符号位
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-4 rounded bg-blue-100 dark:bg-blue-900 border border-slate-300 dark:border-slate-600" />
          数值位
        </span>
      </div>
    </div>
  );
}
