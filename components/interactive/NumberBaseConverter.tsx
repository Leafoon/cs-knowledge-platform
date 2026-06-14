"use client";

import { useState } from "react";

const BASES = [
  { name: "十进制", radix: 10, prefix: "", label: "DEC" },
  { name: "二进制", radix: 2, prefix: "0b", label: "BIN" },
  { name: "八进制", radix: 8, prefix: "0o", label: "OCT" },
  { name: "十六进制", radix: 16, prefix: "0x", label: "HEX" },
];

const EXAMPLES = [
  { value: "2025", base: 10 },
  { value: "11111101001", base: 2 },
  { value: "3F.A", base: 16 },
  { value: "0.6875", base: 10 },
];

function convertInteger(num: string, fromRadix: number, toRadix: number): string {
  const decimal = parseInt(num, fromRadix);
  if (isNaN(decimal)) return "无效输入";
  return decimal.toString(toRadix).toUpperCase();
}

function convertFraction(frac: string, fromRadix: number, toRadix: number, precision = 8): string {
  let decimal = 0;
  for (let i = 0; i < frac.length; i++) {
    const digit = parseInt(frac[i], fromRadix);
    if (isNaN(digit)) return "无效输入";
    decimal += digit / Math.pow(fromRadix, i + 1);
  }
  let result = "";
  let remaining = decimal;
  const seen = new Set<number>();
  for (let i = 0; i < precision; i++) {
    if (remaining === 0) break;
    remaining *= toRadix;
    const digit = Math.floor(remaining);
    if (seen.has(remaining)) { result += "..."; break; }
    seen.add(remaining);
    result += digit.toString(toRadix).toUpperCase();
    remaining -= digit;
  }
  return result || "0";
}

function getConversionSteps(num: string, fromRadix: number, toRadix: number): string[] {
  const parts = num.split(".");
  const intPart = parts[0];
  const fracPart = parts[1];
  const steps: string[] = [];

  if (fromRadix === 10 && toRadix === 2) {
    if (intPart && intPart !== "0") {
      steps.push(`整数部分 — 除 ${toRadix} 取余法（逆序排列）：`);
      let n = parseInt(intPart, 10);
      const remainders: string[] = [];
      while (n > 0) {
        const r = n % toRadix;
        remainders.push(`${n} ÷ ${toRadix} = ${Math.floor(n / toRadix)} … 余 ${r}`);
        n = Math.floor(n / toRadix);
      }
      steps.push(...remainders);
      steps.push(`逆序排列余数 → ${remainders.map(r => r.split("余 ")[1]).reverse().join("")}`);
    }
    if (fracPart) {
      steps.push(`小数部分 — 乘 ${toRadix} 取整法（顺序排列）：`);
      let f = parseInt(fracPart, 10) / Math.pow(10, fracPart.length);
      for (let i = 0; i < 6; i++) {
        if (f === 0) break;
        const product = f * toRadix;
        const digit = Math.floor(product);
        steps.push(`${f.toFixed(4)} × ${toRadix} = ${product.toFixed(4)} … 取整 ${digit}`);
        f = product - digit;
      }
    }
  } else if (toRadix === 10) {
    steps.push(`按权展开法：`);
    const allChars = num.replace(".", "");
    const intLen = intPart.length;
    const terms: string[] = [];
    for (let i = 0; i < allChars.length; i++) {
      const digit = parseInt(allChars[i], fromRadix);
      const power = intLen - 1 - i;
      terms.push(`${allChars[i]}×${fromRadix}${power < 0 ? `⁻${Math.abs(power)}` : `^${power}`}`);
    }
    steps.push(terms.join(" + "));
  } else if ((fromRadix === 2 && toRadix === 16) || (fromRadix === 16 && toRadix === 2)) {
    const groupSize = toRadix === 16 ? 4 : 3;
    steps.push(`分组法：每 ${groupSize} 位二进制为一组`);
    const padded = intPart.padStart(Math.ceil(intPart.length / groupSize) * groupSize, "0");
    const groups: string[] = [];
    for (let i = 0; i < padded.length; i += groupSize) {
      groups.push(padded.slice(i, i + groupSize));
    }
    steps.push(`分组：${groups.join(" ")}`);
    steps.push(`分别转换：${groups.map(g => `${g} → ${parseInt(g, 2).toString(toRadix).toUpperCase()}`).join(", ")}`);
  } else {
    steps.push(`先将 ${num}(基数${fromRadix}) 转换为十进制，再转为基数${toRadix}`);
  }
  return steps;
}

export function NumberBaseConverter() {
  const [inputValue, setInputValue] = useState("2025");
  const [sourceBase, setSourceBase] = useState(10);
  const [showSteps, setShowSteps] = useState(true);

  const validateInput = (val: string, radix: number): boolean => {
    const clean = val.replace(".", "");
    for (const ch of clean) {
      const d = parseInt(ch, radix);
      if (isNaN(d) || d >= radix) return false;
    }
    return clean.length > 0;
  };

  const isValid = validateInput(inputValue, sourceBase);

  const conversions = isValid
    ? BASES.filter((b) => b.radix !== sourceBase).map((b) => {
        const parts = inputValue.split(".");
        const intResult = convertInteger(parts[0], sourceBase, b.radix);
        const fracResult = parts[1] ? "." + convertFraction(parts[1], sourceBase, b.radix) : "";
        return { ...b, result: intResult + fracResult };
      })
    : [];

  const steps = isValid ? getConversionSteps(inputValue, sourceBase, conversions[0]?.radix ?? 2) : [];

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-6 text-slate-900 dark:text-white">
        进制转换器
      </h3>

      {/* Source base selector */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
          选择源进制
        </label>
        <div className="flex gap-2 flex-wrap">
          {BASES.map((b) => (
            <button
              key={b.radix}
              onClick={() => setSourceBase(b.radix)}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
                sourceBase === b.radix
                  ? "bg-blue-600 text-white shadow-md"
                  : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-600 hover:border-blue-400"
              }`}
            >
              {b.label} ({b.name})
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
          输入数值
        </label>
        <div className="flex gap-3">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value.toUpperCase())}
            placeholder={`输入${BASES.find((b) => b.radix === sourceBase)?.name}数值...`}
            className={`flex-1 px-4 py-3 border rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-lg ${
              isValid
                ? "border-slate-300 dark:border-slate-600 focus:border-blue-500"
                : "border-red-400 dark:border-red-600"
            } focus:outline-none focus:ring-2 focus:ring-blue-500/20`}
          />
          <button
            onClick={() => {
              const ex = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
              setInputValue(ex.value);
              setSourceBase(ex.base);
            }}
            className="px-4 py-3 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded-lg hover:bg-indigo-200 dark:hover:bg-indigo-900/50 transition text-sm font-semibold"
          >
            示例
          </button>
        </div>
        {!isValid && (
          <p className="mt-2 text-sm text-red-500">
            输入包含无效字符，请检查
          </p>
        )}
      </div>

      {/* Results */}
      {isValid && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {conversions.map((c) => (
            <div
              key={c.radix}
              className="p-4 bg-white dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700"
            >
              <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                {c.label} · {c.name}
              </div>
              <div className="text-lg font-bold font-mono text-blue-600 dark:text-blue-400 break-all">
                {c.prefix}
                {c.result}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Conversion steps */}
      {isValid && steps.length > 0 && (
        <div>
          <button
            onClick={() => setShowSteps(!showSteps)}
            className="text-sm font-semibold text-blue-600 dark:text-blue-400 hover:underline mb-3"
          >
            {showSteps ? "收起转换过程 ▲" : "展开转换过程 ▼"}
          </button>
          {showSteps && (
            <div className="p-4 bg-slate-50 dark:bg-slate-800/30 rounded-lg border border-slate-200 dark:border-slate-700">
              <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
                转换过程
              </h4>
              <div className="space-y-1">
                {steps.map((step, i) => (
                  <div
                    key={i}
                    className="text-sm font-mono text-slate-600 dark:text-slate-400"
                  >
                    {step}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
