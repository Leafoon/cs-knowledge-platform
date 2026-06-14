"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";

export function ParityCheckerDemo() {
  const [dataBits, setDataBits] = useState("1011001");
  const [errorBit, setErrorBit] = useState<number | null>(null);
  const [parityType, setParityType] = useState<"even" | "odd">("even");

  const bits = dataBits.split("").filter((b) => b === "0" || b === "1");
  const ones = bits.filter((b) => b === "1").length;

  const parityBit = useMemo(() => {
    if (bits.length === 0) return 0;
    return parityType === "even" ? (ones % 2 === 0 ? 0 : 1) : (ones % 2 === 0 ? 1 : 0);
  }, [bits, ones, parityType]);

  const transmitted = [...bits, String(parityBit)];

  const received = useMemo(() => {
    if (errorBit === null || errorBit < 0 || errorBit >= transmitted.length) return transmitted;
    return transmitted.map((b, i) => (i === errorBit ? (b === "0" ? "1" : "0") : b));
  }, [transmitted, errorBit]);

  const receivedOnes = received.filter((b) => b === "1").length;
  const checkResult = parityType === "even" ? receivedOnes % 2 === 0 : receivedOnes % 2 === 1;

  function toggleError(i: number) {
    setErrorBit(errorBit === i ? null : i);
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        奇偶校验演示
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        输入数据位，生成校验位，点击模拟传输错误并检测
      </p>

      {/* controls */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">数据位 (0/1 序列)</label>
          <input
            type="text"
            value={dataBits}
            onChange={(e) => { setDataBits(e.target.value.replace(/[^01]/g, "")); setErrorBit(null); }}
            maxLength={16}
            className="px-3 py-2 rounded border border-border-subtle bg-bg-secondary text-text-primary font-mono text-sm w-44"
            placeholder="例: 1011001"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">校验方式</label>
          <div className="flex rounded-lg overflow-hidden border border-border-subtle">
            {(["even", "odd"] as const).map((t) => (
              <button
                key={t}
                onClick={() => { setParityType(t); setErrorBit(null); }}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  parityType === t
                    ? "bg-accent-primary text-white"
                    : "bg-bg-secondary text-text-secondary"
                }`}
              >
                {t === "even" ? "偶校验" : "奇校验"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {bits.length > 0 && (
        <div className="space-y-6">
          {/* step 1: original data */}
          <div>
            <p className="text-xs font-medium text-text-secondary mb-2">① 原始数据</p>
            <div className="flex gap-1">
              {bits.map((b, i) => (
                <div key={i} className="w-9 h-9 rounded flex items-center justify-center font-mono font-bold text-sm bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border border-blue-300 dark:border-blue-700">
                  {b}
                </div>
              ))}
            </div>
            <p className="text-xs text-text-secondary mt-1">
              1 的个数 = {ones}（{ones % 2 === 0 ? "偶数" : "奇数"}）
            </p>
          </div>

          {/* step 2: generate parity */}
          <div>
            <p className="text-xs font-medium text-text-secondary mb-2">
              ② 生成{parityType === "even" ? "偶" : "奇"}校验位
            </p>
            <div className="flex gap-1 items-end">
              {bits.map((b, i) => (
                <div key={i} className="w-9 h-9 rounded flex items-center justify-center font-mono font-bold text-sm bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border border-blue-300 dark:border-blue-700">
                  {b}
                </div>
              ))}
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-9 h-9 rounded flex items-center justify-center font-mono font-bold text-sm bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 border-2 border-green-500"
              >
                {parityBit}
              </motion.div>
            </div>
            <p className="text-xs text-text-secondary mt-1">
              校验位 = {parityBit}（使总 1 的个数为{parityType === "even" ? "偶数" : "奇数"}）
            </p>
          </div>

          {/* step 3: transmitted */}
          <div>
            <p className="text-xs font-medium text-text-secondary mb-2">
              ③ 传输数据 — 点击某位模拟传输错误
            </p>
            <div className="flex gap-1">
              {transmitted.map((b, i) => (
                <motion.button
                  key={i}
                  onClick={() => toggleError(i)}
                  className={`w-9 h-9 rounded flex items-center justify-center font-mono font-bold text-sm border-2 cursor-pointer transition-colors ${
                    errorBit === i
                      ? "bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 border-red-500"
                      : i === transmitted.length - 1
                        ? "bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 border-green-500"
                        : "bg-bg-secondary text-text-primary border-border-subtle hover:border-accent-primary"
                  }`}
                  whileTap={{ scale: 0.9 }}
                >
                  {errorBit === i ? (b === "0" ? "1" : "0") : b}
                </motion.button>
              ))}
            </div>
            {errorBit !== null && (
              <p className="text-xs text-red-500 mt-1">
                第 {errorBit + 1} 位发生翻转：{transmitted[errorBit]} → {received[errorBit]}
              </p>
            )}
          </div>

          {/* step 4: check result */}
          <div>
            <p className="text-xs font-medium text-text-secondary mb-2">④ 接收端校验</p>
            <div className="flex gap-1 items-end">
              {received.map((b, i) => (
                <div
                  key={i}
                  className={`w-9 h-9 rounded flex items-center justify-center font-mono font-bold text-sm border ${
                    errorBit === i
                      ? "bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 border-red-500"
                      : i === received.length - 1
                        ? "bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 border-green-500"
                        : "bg-bg-secondary text-text-primary border-border-subtle"
                  }`}
                >
                  {b}
                </div>
              ))}
            </div>
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mt-2 px-3 py-2 rounded-lg text-sm font-medium ${
                checkResult
                  ? "bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-300 border border-green-200 dark:border-green-800"
                  : "bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800"
              }`}
            >
              {checkResult
                ? `✓ 校验通过 — 接收数据中 1 的个数为 ${receivedOnes}（${parityType === "even" ? "偶数" : "奇数"}），无检测到错误`
                : `✗ 校验失败 — 接收数据中 1 的个数为 ${receivedOnes}（${parityType === "even" ? "非偶数" : "非奇数"}），检测到错误！`}
            </motion.div>
          </div>
        </div>
      )}
    </div>
  );
}
