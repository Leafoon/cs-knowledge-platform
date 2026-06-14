"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";

function padBin(val: number, n: number): string {
  return ((val < 0 ? (1 << n) + val : val) >>> 0).toString(2).padStart(n, "0").slice(-n);
}

export function SignedUnsignedComparison() {
  const [bits, setBits] = useState(8);
  const [inputBin, setInputBin] = useState("10110100");

  const n = bits;
  const validBits = inputBin.split("").filter((b) => b === "0" || b === "1").slice(0, n);
  const padded = validBits.join("").padStart(n, "0").slice(-n);

  const unsignedVal = useMemo(() => parseInt(padded, 2), [padded]);
  const signedVal = useMemo(() => {
    const u = parseInt(padded, 2);
    return u >= (1 << (n - 1)) ? u - (1 << n) : u;
  }, [padded, n]);

  const bitRange = useMemo(() => {
    return Array.from({ length: n }, (_, i) => n - 1 - i);
  }, [n]);

  function handleBitFlip(i: number) {
    const arr = padded.split("");
    arr[i] = arr[i] === "0" ? "1" : "0";
    setInputBin(arr.join(""));
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        有符号 vs 无符号对比
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        点击位可翻转，观察同一二进制在不同解释下的含义
      </p>

      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">位宽</label>
          <div className="flex rounded-lg overflow-hidden border border-border-subtle">
            {([8, 16] as const).map((b) => (
              <button key={b} onClick={() => { setBits(b); setInputBin("0".repeat(b)); }}
                className={`px-4 py-1.5 text-sm font-medium transition-colors ${bits === b ? "bg-accent-primary text-white" : "bg-bg-secondary text-text-secondary"}`}>
                {b}-bit
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">二进制输入</label>
          <input
            type="text"
            value={inputBin}
            onChange={(e) => setInputBin(e.target.value.replace(/[^01]/g, "").slice(0, n))}
            maxLength={n}
            className="px-3 py-1.5 rounded border border-border-subtle bg-bg-secondary text-text-primary font-mono text-sm w-40"
          />
        </div>
      </div>

      {/* interactive bit boxes */}
      <div className="mb-6">
        <p className="text-xs text-text-secondary mb-2">点击翻转每一位</p>
        <div className="flex gap-0.5">
          {padded.split("").map((b, i) => (
            <div key={i} className="flex flex-col items-center">
              <span className="text-[10px] text-text-secondary mb-1">{bitRange[i]}</span>
              <motion.button
                onClick={() => handleBitFlip(i)}
                className={`w-9 h-9 rounded flex items-center justify-center font-mono font-bold text-sm border-2 cursor-pointer transition-colors ${
                  i === 0
                    ? "bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 border-red-400 dark:border-red-600 hover:border-red-500"
                    : "bg-blue-50 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border-blue-300 dark:border-blue-700 hover:border-blue-500"
                }`}
                whileTap={{ scale: 0.9 }}
              >
                {b}
              </motion.button>
            </div>
          ))}
        </div>
        <div className="flex gap-0.5 mt-1">
          {padded.split("").map((_, i) => (
            <span key={i} className="w-9 text-center text-[9px] text-text-secondary">
              {i === 0 ? "符号位" : ""}
            </span>
          ))}
        </div>
      </div>

      {/* comparison side by side */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* unsigned */}
        <motion.div
          key={`u-${unsignedVal}`}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-lg border border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-950 p-4"
        >
          <h4 className="font-semibold text-sm text-blue-700 dark:text-blue-300 mb-2">无符号解释 (Unsigned)</h4>
          <div className="text-center mb-3">
            <span className="text-3xl font-bold font-mono text-blue-800 dark:text-blue-200">{unsignedVal}</span>
          </div>
          <p className="text-xs text-blue-600 dark:text-blue-400 font-mono">
            Σ b<sub>i</sub> × 2<sup>i</sup> = {padded.split("").map((b, i) => b === "1" ? `2^${n - 1 - i}` : null).filter(Boolean).join(" + ") || "0"} = {unsignedVal}
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
            范围：[0, {(1 << n) - 1}]
          </p>
        </motion.div>

        {/* signed (two's complement) */}
        <motion.div
          key={`s-${signedVal}`}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="rounded-lg border border-green-300 dark:border-green-700 bg-green-50 dark:bg-green-950 p-4"
        >
          <h4 className="font-semibold text-sm text-green-700 dark:text-green-300 mb-2">有符号解释 (Signed / 补码)</h4>
          <div className="text-center mb-3">
            <span className="text-3xl font-bold font-mono text-green-800 dark:text-green-200">{signedVal}</span>
          </div>
          <p className="text-xs text-green-600 dark:text-green-400 font-mono">
            -b<sub>{n - 1}</sub>×2<sup>{n - 1}</sup> + Σ b<sub>i</sub>×2<sup>i</sup> = {padded[0] === "1" ? `-2^${n - 1}` : "0"}{padded.split("").slice(1).map((b, i) => b === "1" ? ` + 2^${n - 2 - i}` : null).filter(Boolean).join("")} = {signedVal}
          </p>
          <p className="text-xs text-green-600 dark:text-green-400 mt-1">
            范围：[{-(1 << (n - 1))}, {(1 << (n - 1)) - 1}]
          </p>
        </motion.div>
      </div>

      {/* binary string display */}
      <div className="mt-4 rounded-lg bg-bg-secondary border border-border-subtle p-3">
        <p className="text-xs text-text-secondary">
          <span className="font-semibold text-text-primary">二进制：</span>
          <span className="font-mono ml-1">{padded}</span>
          <span className="ml-3">十六进制：</span>
          <span className="font-mono ml-1">0x{unsignedVal.toString(16).toUpperCase().padStart(Math.ceil(n / 4), "0")}</span>
        </p>
      </div>
    </div>
  );
}
