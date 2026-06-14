"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";

function padBin(val: number, n: number): string {
  return ((val < 0 ? (1 << n) + val : val) >>> 0).toString(2).padStart(n, "0").slice(-n);
}

export function OverflowDetectionDemo() {
  const [bits, setBits] = useState(8);
  const [a, setA] = useState("100");
  const [b, setB] = useState("50");

  const n = bits;
  const lo = -(1 << (n - 1));
  const hi = (1 << (n - 1)) - 1;
  const aVal = parseInt(a, 10);
  const bVal = parseInt(b, 10);
  const validA = !isNaN(aVal) && aVal >= lo && aVal <= hi;
  const validB = !isNaN(bVal) && bVal >= lo && bVal <= hi;
  const valid = validA && validB;

  const result = useMemo(() => {
    if (!valid) return null;
    const aBin = padBin(aVal, n);
    const bBin = padBin(bVal, n);
    const sum = aVal + bVal;

    // binary addition
    let carry = 0;
    const resultBits: string[] = [];
    const carries: number[] = [];
    for (let i = n - 1; i >= 0; i--) {
      const ab = parseInt(aBin[i]);
      const bb = parseInt(bBin[i]);
      const total = ab + bb + carry;
      resultBits.unshift(String(total % 2));
      carry = Math.floor(total / 2);
      carries.unshift(carry);
    }
    const resultBin = resultBits.join("");
    const cout = carry;

    // interpret result as signed
    const resultUnsigned = parseInt(resultBin, 2);
    const resultSigned = resultUnsigned >= (1 << (n - 1)) ? resultUnsigned - (1 << n) : resultUnsigned;

    // overflow detection
    const signA = parseInt(aBin[0]);
    const signB = parseInt(bBin[0]);
    const signR = parseInt(resultBin[0]);

    // method 1: single sign bit
    const overflow1 = signA === signB && signA !== signR;

    // method 2: double sign bit (carry into sign vs carry out of sign)
    const carryIntoSign = carries.length >= 2 ? carries[carries.length - 2] : 0;
    const carryOutOfSign = cout;
    const overflow2 = carryIntoSign !== carryOutOfSign;

    return {
      aBin, bBin, resultBin, sum, resultSigned, cout,
      overflow1, overflow2, signA, signB, signR, carryIntoSign, carryOutOfSign,
      inRange: sum >= lo && sum <= hi,
    };
  }, [aVal, bVal, n, valid]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        溢出检测演示
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        输入两个数进行补码加法，展示单符号位和双符号位溢出检测
      </p>

      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">位宽</label>
          <div className="flex rounded-lg overflow-hidden border border-border-subtle">
            {([8, 16] as const).map((b) => (
              <button key={b} onClick={() => setBits(b)}
                className={`px-4 py-1.5 text-sm font-medium transition-colors ${bits === b ? "bg-accent-primary text-white" : "bg-bg-secondary text-text-secondary"}`}>
                {b}-bit
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">操作数 A ({lo}~{hi})</label>
          <input type="number" value={a} onChange={(e) => setA(e.target.value)} min={lo} max={hi}
            className="w-24 px-3 py-1.5 rounded border border-border-subtle bg-bg-secondary text-text-primary font-mono text-sm" />
        </div>
        <span className="self-end pb-2 text-lg font-bold text-text-secondary">+</span>
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">操作数 B ({lo}~{hi})</label>
          <input type="number" value={b} onChange={(e) => setB(e.target.value)} min={lo} max={hi}
            className="w-24 px-3 py-1.5 rounded border border-border-subtle bg-bg-secondary text-text-primary font-mono text-sm" />
        </div>
      </div>

      {!valid && (
        <p className="text-sm text-red-500">请输入合法的 {n}-bit 范围内的整数</p>
      )}

      {valid && result && (
        <div className="space-y-4">
          {/* operands display */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { label: `A = ${aVal}`, bits: result.aBin },
              { label: `B = ${bVal}`, bits: result.bBin },
              { label: `结果 = ${result.resultSigned}`, bits: result.resultBin },
            ].map((op, i) => (
              <div key={i} className="rounded-lg border border-border-subtle bg-bg-secondary p-3 text-center">
                <p className="text-xs text-text-secondary mb-1">{op.label}</p>
                <div className="flex gap-0.5 justify-center">
                  {op.bits.split("").map((bit, j) => (
                    <span key={j} className={`inline-flex items-center justify-center w-7 h-7 rounded text-xs font-bold font-mono border ${
                      j === 0 ? "bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 border-red-300 dark:border-red-700"
                        : "bg-blue-50 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border-blue-300 dark:border-blue-700"
                    }`}>
                      {bit}
                    </span>
                  ))}
                </div>
                <p className="font-mono text-xs text-text-secondary mt-1">{op.bits}</p>
              </div>
            ))}
          </div>

          {/* overflow detection methods */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* method 1: single sign bit */}
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className={`rounded-lg border p-4 ${
                result.overflow1
                  ? "bg-red-50 dark:bg-red-950 border-red-300 dark:border-red-700"
                  : "bg-green-50 dark:bg-green-950 border-green-300 dark:border-green-700"
              }`}
            >
              <h4 className="font-semibold text-sm text-text-primary mb-2">
                方法一：单符号位检测
              </h4>
              <p className="text-xs text-text-secondary mb-2">
                若两个同号数相加，结果符号与操作数不同，则溢出
              </p>
              <div className="font-mono text-xs space-y-1">
                <p>A符号={result.signA}, B符号={result.signB}, 结果符号={result.signR}</p>
                <p className={result.overflow1 ? "text-red-600 dark:text-red-400 font-bold" : "text-green-600 dark:text-green-400 font-bold"}>
                  {result.overflow1
                    ? `${result.signA}=${result.signB} ≠ ${result.signR} → 溢出！`
                    : `符号位无冲突 → 无溢出`}
                </p>
              </div>
            </motion.div>

            {/* method 2: double sign bit (Cf ⊕ C1) */}
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className={`rounded-lg border p-4 ${
                result.overflow2
                  ? "bg-red-50 dark:bg-red-950 border-red-300 dark:border-red-700"
                  : "bg-green-50 dark:bg-green-950 border-green-300 dark:border-green-700"
              }`}
            >
              <h4 className="font-semibold text-sm text-text-primary mb-2">
                方法二：双符号位检测 (Cf⊕C1)
              </h4>
              <p className="text-xs text-text-secondary mb-2">
                最高位进位(Cf)与次高位进位(C1)异或为 1 则溢出
              </p>
              <div className="font-mono text-xs space-y-1">
                <p>Cf(符号位进位)={result.carryOutOfSign}, C1(次高位进位)={result.carryIntoSign}</p>
                <p className={result.overflow2 ? "text-red-600 dark:text-red-400 font-bold" : "text-green-600 dark:text-green-400 font-bold"}>
                  {result.carryIntoSign} ⊕ {result.carryOutOfSign} = {result.carryIntoSign ^ result.carryOutOfSign}
                  {result.overflow2 ? " → 溢出！" : " → 无溢出"}
                </p>
              </div>
            </motion.div>
          </div>

          {/* arithmetic check */}
          <div className={`rounded-lg p-3 text-sm ${result.inRange ? "bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-300" : "bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300"}`}>
            算术值：{aVal} + {bVal} = {result.sum}
            {result.inRange ? `（在 ${n}-bit 范围 [${lo}, ${hi}] 内）` : `（超出 ${n}-bit 范围 [${lo}, ${hi}]，溢出！）`}
          </div>
        </div>
      )}
    </div>
  );
}
