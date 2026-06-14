"use client";

import { useState } from "react";

type BCDType = "8421" | "excess3" | "2421";

const BCD_TABLE: Record<number, Record<BCDType, string>> = {
  0: { "8421": "0000", excess3: "0011", "2421": "0000" },
  1: { "8421": "0001", excess3: "0100", "2421": "0001" },
  2: { "8421": "0010", excess3: "0101", "2421": "0010" },
  3: { "8421": "0011", excess3: "0110", "2421": "0011" },
  4: { "8421": "0100", excess3: "0111", "2421": "0100" },
  5: { "8421": "0101", excess3: "1000", "2421": "1011" },
  6: { "8421": "0110", excess3: "1001", "2421": "1100" },
  7: { "8421": "0111", excess3: "1010", "2421": "1101" },
  8: { "8421": "1000", excess3: "1011", "2421": "1110" },
  9: { "8421": "1001", excess3: "1100", "2421": "1111" },
};

const BCD_INFO: Record<BCDType, { name: string; desc: string; color: string }> = {
  "8421": { name: "8421码", desc: "有权码，权值 8-4-2-1", color: "blue" },
  excess3: { name: "余3码", desc: "无权码，8421码+3，具有自补性", color: "emerald" },
  "2421": { name: "2421码", desc: "有权码，权值 2-4-2-1，具有自补性", color: "violet" },
};

const COLOR_MAP: Record<string, { bg: string; border: string; text: string; bit: string }> = {
  blue: { bg: "bg-blue-50 dark:bg-blue-900/20", border: "border-blue-300 dark:border-blue-700", text: "text-blue-700 dark:text-blue-300", bit: "bg-blue-500" },
  emerald: { bg: "bg-emerald-50 dark:bg-emerald-900/20", border: "border-emerald-300 dark:border-emerald-700", text: "text-emerald-700 dark:text-emerald-300", bit: "bg-emerald-500" },
  violet: { bg: "bg-violet-50 dark:bg-violet-900/20", border: "border-violet-300 dark:border-violet-700", text: "text-violet-700 dark:text-violet-300", bit: "bg-violet-500" },
};

function BitBox({ bit, color }: { bit: string; color: string }) {
  return (
    <div className={`w-8 h-8 ${color} text-white text-xs font-mono font-bold rounded flex items-center justify-center shadow-sm`}>
      {bit}
    </div>
  );
}

export function BCDEncoder() {
  const [bcdType, setBcdType] = useState<BCDType>("8421");
  const [inputValue, setInputValue] = useState("2025");
  const [addA, setAddA] = useState("46");
  const [addB, setAddB] = useState("37");

  const digits = inputValue.split("").map(Number).filter((d) => !isNaN(d) && d >= 0 && d <= 9);
  const encoded = digits.map((d) => BCD_TABLE[d][bcdType]);
  const info = BCD_INFO[bcdType];
  const colors = COLOR_MAP[info.color];

  const aDigits = addA.split("").map(Number).filter((d) => !isNaN(d) && d >= 0 && d <= 9);
  const bDigits = addB.split("").map(Number).filter((d) => !isNaN(d) && d >= 0 && d <= 9);
  const maxLen = Math.max(aDigits.length, bDigits.length);
  const paddedA = Array(maxLen - aDigits.length).fill(0).concat(aDigits);
  const paddedB = Array(maxLen - bDigits.length).fill(0).concat(bDigits);

  const additionSteps = paddedA.map((a, i) => {
    const b = paddedB[i];
    const sum = a + b;
    const encoded_a = BCD_TABLE[a]["8421"];
    const encoded_b = BCD_TABLE[b]["8421"];
    const rawSum = (a + b).toString(2).padStart(5, "0");
    const needsCorrection = sum > 9;
    const correctedValue = needsCorrection ? sum + 6 : sum;
    const correctedBinary = correctedValue.toString(2).padStart(needsCorrection ? 5 : 4, "0");
    const carry = needsCorrection ? 1 : 0;
    return {
      a, b, sum, encoded_a, encoded_b, rawSum, needsCorrection,
      correctedBinary, carry, resultDigit: sum % 10,
    };
  });

  const finalResult = additionSteps.reduce((acc, step, i) => {
    return acc + step.resultDigit.toString();
  }, "");
  const hasCarryOut = additionSteps.some((s) => s.carry > 0 && s.sum >= 10);

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-6 text-slate-900 dark:text-white">
        BCD码编码器
      </h3>

      {/* BCD type selector */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {(Object.keys(BCD_INFO) as BCDType[]).map((type) => {
          const c = COLOR_MAP[BCD_INFO[type].color];
          return (
            <button
              key={type}
              onClick={() => setBcdType(type)}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
                bcdType === type
                  ? `${c.bg} ${c.text} ${c.border} border shadow-md`
                  : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-600 hover:border-green-400"
              }`}
            >
              {BCD_INFO[type].name}
            </button>
          );
        })}
      </div>

      {/* Input */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
          输入十进制数（0-9999）
        </label>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value.replace(/[^0-9]/g, "").slice(0, 4))}
          className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-lg focus:outline-none focus:ring-2 focus:ring-green-500/20"
        />
      </div>

      {/* Info badge */}
      <div className={`mb-4 p-3 ${colors.bg} rounded-lg border ${colors.border}`}>
        <span className={`text-sm font-semibold ${colors.text}`}>
          {info.name}：{info.desc}
        </span>
      </div>

      {/* Encoding result */}
      {digits.length > 0 && (
        <div className="mb-6">
          <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
            编码结果
          </h4>
          <div className="flex flex-wrap items-center gap-2 mb-3">
            {digits.map((d, i) => (
              <div key={i} className="flex items-center gap-1">
                <span className="text-sm font-semibold text-slate-500 dark:text-slate-400 mr-1">
                  {d} →
                </span>
                <div className="flex gap-0.5">
                  {encoded[i].split("").map((bit, j) => (
                    <BitBox key={j} bit={bit} color={colors.bit} />
                  ))}
                </div>
                {i < digits.length - 1 && (
                  <span className="text-slate-400 mx-1">|</span>
                )}
              </div>
            ))}
          </div>
          <div className="p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
            <span className="text-xs text-slate-500 dark:text-slate-400">
              完整编码：
            </span>
            <span className="font-mono font-bold text-slate-800 dark:text-white ml-2">
              {encoded.join("")}
            </span>
          </div>
        </div>
      )}

      {/* Comparison table */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
          三种BCD码对照表
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-slate-100 dark:bg-slate-800">
                <th className="px-3 py-2 border border-slate-300 dark:border-slate-600 text-left">十进制</th>
                <th className="px-3 py-2 border border-slate-300 dark:border-slate-600 text-center">8421码</th>
                <th className="px-3 py-2 border border-slate-300 dark:border-slate-600 text-center">余3码</th>
                <th className="px-3 py-2 border border-slate-300 dark:border-slate-600 text-center">2421码</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 10 }, (_, d) => {
                const isHighlighted = digits.includes(d);
                return (
                  <tr
                    key={d}
                    className={isHighlighted ? "bg-green-50 dark:bg-green-900/20" : ""}
                  >
                    <td className="px-3 py-2 border border-slate-300 dark:border-slate-600 font-semibold">
                      {d}
                    </td>
                    {(["8421", "excess3", "2421"] as BCDType[]).map((type) => (
                      <td
                        key={type}
                        className={`px-3 py-2 border border-slate-300 dark:border-slate-600 text-center font-mono ${
                          bcdType === type ? "font-bold text-green-700 dark:text-green-400" : ""
                        }`}
                      >
                        {BCD_TABLE[d][type]}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* BCD Addition Demo */}
      <div>
        <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
          BCD加法演示（8421码 + 加6修正）
        </h4>
        <div className="flex gap-4 mb-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1">加数 A</label>
            <input
              type="text"
              value={addA}
              onChange={(e) => setAddA(e.target.value.replace(/[^0-9]/g, "").slice(0, 2))}
              className="w-20 px-3 py-2 border border-slate-300 dark:border-slate-600 rounded bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-sm"
            />
          </div>
          <div className="flex items-center text-xl font-bold text-slate-400 pt-4">+</div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">加数 B</label>
            <input
              type="text"
              value={addB}
              onChange={(e) => setAddB(e.target.value.replace(/[^0-9]/g, "").slice(0, 2))}
              className="w-20 px-3 py-2 border border-slate-300 dark:border-slate-600 rounded bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-sm"
            />
          </div>
        </div>

        {additionSteps.length > 0 && (
          <div className="space-y-2">
            {additionSteps.map((step, i) => (
              <div
                key={i}
                className="p-3 bg-white dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700"
              >
                <div className="flex items-center gap-4 flex-wrap">
                  <span className="text-sm font-semibold text-slate-500">
                    位 {i + 1}：
                  </span>
                  <span className="font-mono text-sm">
                    <span className="text-blue-600 dark:text-blue-400">{step.encoded_a}</span>
                    {" + "}
                    <span className="text-violet-600 dark:text-violet-400">{step.encoded_b}</span>
                    {" = "}
                    <span className="text-slate-800 dark:text-white">{step.rawSum}</span>
                  </span>
                  {step.needsCorrection && (
                    <>
                      <span className="text-rose-600 dark:text-rose-400 text-sm font-semibold">
                        ({step.sum} &gt; 9，需修正)
                      </span>
                      <span className="font-mono text-sm">
                        + 0110 = <span className="text-green-600 dark:text-green-400 font-bold">{step.correctedBinary}</span>
                      </span>
                    </>
                  )}
                  <span className="text-sm text-slate-500">
                    → 结果位：<span className="font-bold text-green-600 dark:text-green-400">{step.resultDigit}</span>
                  </span>
                </div>
              </div>
            ))}
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <span className="text-sm font-semibold text-green-700 dark:text-green-300">
                最终结果：{finalResult}（验证：{addA} + {addB} = {parseInt(addA || "0") + parseInt(addB || "0")}）
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
