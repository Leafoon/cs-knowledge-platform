"use client";

import { useState } from "react";
import { motion } from "framer-motion";

function padBin(val: number, n: number): string {
  return val.toString(2).padStart(n, "0").slice(-n);
}

function toUnsigned(v: number, n: number): number {
  return v < 0 ? (1 << n) + v : v;
}

function getOriginalCode(v: number, n: number): string {
  if (v === 0) return "0".repeat(n);
  const sign = v < 0 ? "1" : "0";
  return sign + padBin(Math.abs(v), n - 1);
}

function getOnesComplement(v: number, n: number): string {
  if (v === 0) return "0".repeat(n);
  if (v > 0) return padBin(v, n);
  const pos = padBin(Math.abs(v), n - 1);
  const inv = pos.split("").map((b) => (b === "0" ? "1" : "0")).join("");
  return "1" + inv;
}

function getTwosComplement(v: number, n: number): string {
  return padBin(toUnsigned(v, n), n);
}

export function TwosComplementVisualizer() {
  const [bits, setBits] = useState(8);
  const [inputVal, setInputVal] = useState("-42");

  const n = bits;
  const lo = -(1 << (n - 1));
  const hi = (1 << (n - 1)) - 1;
  const v = parseInt(inputVal, 10);
  const valid = !isNaN(v) && v >= lo && v <= hi;

  const codes = valid
    ? [
        { name: "原码", nameEn: "Sign-Magnitude", bits: getOriginalCode(v, n), color: "#667eea" },
        { name: "反码", nameEn: "Ones' Complement", bits: getOnesComplement(v, n), color: "#f59e0b" },
        { name: "补码", nameEn: "Two's Complement", bits: getTwosComplement(v, n), color: "#10b981" },
      ]
    : [];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        补码表示可视化
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        输入十进制数，查看原码、反码、补码的二进制表示
      </p>

      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">位宽</label>
          <div className="flex rounded-lg overflow-hidden border border-border-subtle">
            {([8, 16] as const).map((b) => (
              <button
                key={b}
                onClick={() => setBits(b)}
                className={`px-4 py-1.5 text-sm font-medium transition-colors ${
                  bits === b
                    ? "bg-accent-primary text-white"
                    : "bg-bg-secondary text-text-secondary hover:bg-bg-elevated"
                }`}
              >
                {b}-bit
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">
            十进制值 ({lo} ~ {hi})
          </label>
          <input
            type="number"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            min={lo}
            max={hi}
            className="w-28 px-3 py-1.5 rounded border border-border-subtle bg-bg-secondary text-text-primary font-mono text-sm"
          />
        </div>
      </div>

      {!valid && inputVal !== "" && (
        <p className="text-sm text-red-500 mb-4">
          {v} 超出 {n}-bit 范围 [{lo}, {hi}]
        </p>
      )}

      {valid && (
        <div className="space-y-4">
          {/* value label */}
          <motion.div
            key={v}
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <span className="text-3xl font-bold text-text-primary">{v}</span>
            <span className="text-sm text-text-secondary ml-2">
              ({v >= 0 ? "正数" : "负数"})
            </span>
          </motion.div>

          {/* three representations */}
          {codes.map((code, ci) => (
            <motion.div
              key={code.name}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: ci * 0.1 }}
              className="rounded-lg border border-border-subtle bg-bg-secondary p-4"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: code.color }} />
                <span className="font-semibold text-sm text-text-primary">{code.name}</span>
                <span className="text-xs text-text-secondary">{code.nameEn}</span>
              </div>
              <div className="flex gap-0.5 mb-2">
                {code.bits.split("").map((b, i) => (
                  <motion.span
                    key={i}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: ci * 0.1 + i * 0.02 }}
                    className={`inline-flex items-center justify-center w-8 h-8 rounded text-xs font-bold font-mono border ${
                      i === 0
                        ? "bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 border-red-300 dark:border-red-700"
                        : "bg-blue-50 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border-blue-300 dark:border-blue-700"
                    }`}
                  >
                    {b}
                  </motion.span>
                ))}
              </div>
              <p className="font-mono text-sm text-text-secondary">{code.bits}</p>
            </motion.div>
          ))}

          {/* conversion steps for negative */}
          {v < 0 && valid && (
            <div className="rounded-lg bg-bg-secondary border border-border-subtle p-4">
              <p className="text-xs font-medium text-text-secondary mb-2">转换过程（负数）</p>
              <div className="space-y-1 text-xs font-mono text-text-secondary">
                <p>① 原码 → 反码：符号位不变，数值位按位取反</p>
                <p>② 反码 → 补码：反码 + 1</p>
                <p className="mt-1 text-text-primary">
                  {getOriginalCode(v, n)} → {getOnesComplement(v, n)} → {getTwosComplement(v, n)}
                </p>
              </div>
            </div>
          )}

          {/* legend */}
          <div className="flex flex-wrap gap-4 text-[11px] text-text-secondary">
            <span className="flex items-center gap-1">
              <span className="inline-block w-4 h-4 rounded bg-red-100 dark:bg-red-900 border border-red-300 dark:border-red-700" />
              符号位
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-4 h-4 rounded bg-blue-50 dark:bg-blue-900 border border-blue-300 dark:border-blue-700" />
              数值位
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
