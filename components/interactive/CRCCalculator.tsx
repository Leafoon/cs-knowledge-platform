"use client";

import { useState } from "react";

const PRESETS = [
  { name: "CRC-8", poly: "100000111", desc: "x⁸+x²+x+1" },
  { name: "CRC-16", poly: "11000000000000101", desc: "x¹⁶+x¹⁵+x²+1" },
  { name: "CRC-CCITT", poly: "10001000000100001", desc: "x¹⁶+x¹²+x⁵+1" },
];

function xorStrings(a: string, b: string): string {
  let result = "";
  for (let i = 0; i < b.length; i++) {
    result += a[i] === b[i] ? "0" : "1";
  }
  return result;
}

interface DivisionStep {
  step: number;
  current: string;
  divisor: string;
  xorResult: string;
  bringDown: string;
  quotient: string;
}

function crcDivide(dividend: string, divisor: string): { remainder: string; steps: DivisionStep[] } {
  const n = divisor.length;
  let current = dividend.slice(0, n);
  const steps: DivisionStep[] = [];
  let stepNum = 0;

  for (let i = n; i <= dividend.length; i++) {
    stepNum++;
    const quotientBit = current[0] === "1" ? "1" : "0";
    let xorResult: string;

    if (current[0] === "1") {
      xorResult = xorStrings(current, divisor);
    } else {
      xorResult = current.replace(/^./, "0").slice(0, n);
      xorResult = current;
    }

    const bringDown = i < dividend.length ? dividend[i] : "";

    steps.push({
      step: stepNum,
      current: current,
      divisor: current[0] === "1" ? divisor : "0".repeat(n),
      xorResult: current[0] === "1" ? xorStrings(current, divisor) : current,
      bringDown,
      quotient: quotientBit,
    });

    if (current[0] === "1") {
      current = xorStrings(current, divisor);
    }

    if (i < dividend.length) {
      current = current.slice(1) + dividend[i];
    }
  }

  return { remainder: current.slice(1), steps };
}

function getCRC(data: string, poly: string): { crc: string; steps: DivisionStep[] } {
  const r = poly.length - 1;
  const paddedData = data + "0".repeat(r);
  const { remainder, steps } = crcDivide(paddedData, poly);
  return { crc: remainder, steps };
}

export function CRCCalculator() {
  const [data, setData] = useState("110101");
  const [poly, setPoly] = useState("1011");
  const [verifyData, setVerifyData] = useState("");
  const [activePreset, setActivePreset] = useState<string | null>(null);

  const validData = /^[01]+$/.test(data) && data.length > 0;
  const validPoly = /^[01]+$/.test(poly) && poly.length >= 2 && poly[0] === "1";
  const canCalculate = validData && validPoly;

  const result = canCalculate ? getCRC(data, poly) : null;
  const transmittedFrame = canCalculate && result ? data + result.crc : "";

  const verifyValid = verifyData.length > 0 && /^[01]+$/.test(verifyData) && canCalculate;
  const verifyResult = verifyValid && result ? getCRC(verifyData, poly) : null;
  const verifyPassed = verifyResult ? /^0*$/.test(verifyResult.crc) : null;

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-purple-50 to-violet-50 dark:from-slate-900 dark:to-purple-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-6 text-slate-900 dark:text-white">
        CRC码计算与校验
      </h3>

      {/* Presets */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
          常用生成多项式预设
        </label>
        <div className="flex gap-2 flex-wrap">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => { setPoly(p.poly); setActivePreset(p.name); }}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                activePreset === p.name
                  ? "bg-purple-600 text-white shadow-md"
                  : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-600 hover:border-purple-400"
              }`}
            >
              {p.name} ({p.desc})
            </button>
          ))}
        </div>
      </div>

      {/* Data input */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
            数据位（二进制）
          </label>
          <input
            type="text"
            value={data}
            onChange={(e) => setData(e.target.value.replace(/[^01]/g, ""))}
            className={`w-full px-4 py-3 border rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-lg focus:outline-none focus:ring-2 focus:ring-purple-500/20 ${
              validData ? "border-slate-300 dark:border-slate-600" : "border-red-400"
            }`}
            placeholder="例如：110101"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
            生成多项式（二进制，最高位为1）
          </label>
          <input
            type="text"
            value={poly}
            onChange={(e) => { setPoly(e.target.value.replace(/[^01]/g, "")); setActivePreset(null); }}
            className={`w-full px-4 py-3 border rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-lg focus:outline-none focus:ring-2 focus:ring-purple-500/20 ${
              validPoly ? "border-slate-300 dark:border-slate-600" : "border-red-400"
            }`}
            placeholder="例如：1011"
          />
        </div>
      </div>

      {canCalculate && result && (
        <>
          {/* Info */}
          <div className="mb-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
            <span className="text-sm text-purple-700 dark:text-purple-300">
              生成多项式阶数 r = {poly.length - 1}，在数据后补 {poly.length - 1} 个 0
            </span>
          </div>

          {/* Division steps */}
          <div className="mb-6">
            <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
              模2除法过程
            </h4>
            <div className="p-4 bg-slate-50 dark:bg-slate-800/30 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
              <div className="font-mono text-sm space-y-1">
                <div className="text-slate-500 dark:text-slate-400 mb-2">
                  被除数：{data}{"0".repeat(poly.length - 1)}（数据后补{poly.length - 1}个0）
                </div>
                <div className="text-slate-500 dark:text-slate-400 mb-3">
                  除数：{poly}
                </div>
                {result.steps.filter((_, i) => i < 20).map((step, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-slate-400 w-6 text-right shrink-0">{step.step}.</span>
                    <div>
                      <span className="text-blue-600 dark:text-blue-400">{step.current}</span>
                      {step.quotient === "1" && (
                        <>
                          <span className="text-slate-400 mx-1">⊕</span>
                          <span className="text-purple-600 dark:text-purple-400">{step.divisor}</span>
                        </>
                      )}
                      <span className="text-slate-400 mx-1">→</span>
                      <span className="text-green-600 dark:text-green-400">{step.xorResult}</span>
                      {step.bringDown && (
                        <span className="text-slate-400 ml-1">↓{step.bringDown}</span>
                      )}
                      <span className="text-slate-400 ml-2">商{step.quotient}</span>
                    </div>
                  </div>
                ))}
                {result.steps.length > 20 && (
                  <div className="text-slate-400">...（共 {result.steps.length} 步）</div>
                )}
              </div>
            </div>
          </div>

          {/* Result */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <div className="text-xs font-semibold text-green-600 dark:text-green-400 mb-1">
                CRC校验码（余数）
              </div>
              <div className="font-mono text-lg font-bold text-green-700 dark:text-green-300">
                {result.crc}
              </div>
            </div>
            <div className="p-4 bg-violet-50 dark:bg-violet-900/20 rounded-lg border border-violet-200 dark:border-violet-800">
              <div className="text-xs font-semibold text-violet-600 dark:text-violet-400 mb-1">
                发送帧（数据 + CRC）
              </div>
              <div className="font-mono text-lg font-bold text-violet-700 dark:text-violet-300 break-all">
                <span className="text-slate-800 dark:text-white">{data}</span>
                <span className="text-violet-600 dark:text-violet-400">{result.crc}</span>
              </div>
            </div>
          </div>

          {/* Verification */}
          <div>
            <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
              CRC校验（输入接收到的帧）
            </h4>
            <div className="flex gap-3 mb-3">
              <input
                type="text"
                value={verifyData}
                onChange={(e) => setVerifyData(e.target.value.replace(/[^01]/g, ""))}
                className="flex-1 px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono focus:outline-none focus:ring-2 focus:ring-purple-500/20"
                placeholder="输入接收到的帧..."
              />
              <button
                onClick={() => setVerifyData(transmittedFrame)}
                className="px-4 py-3 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-200 text-sm font-semibold"
              >
                填入正确帧
              </button>
              <button
                onClick={() => {
                  const flipped = transmittedFrame.split("");
                  const pos = Math.floor(Math.random() * flipped.length);
                  flipped[pos] = flipped[pos] === "0" ? "1" : "0";
                  setVerifyData(flipped.join(""));
                }}
                className="px-4 py-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg hover:bg-red-200 text-sm font-semibold"
              >
                模拟1位错误
              </button>
            </div>

            {verifyResult && (
              <div className={`p-4 rounded-lg border ${
                verifyPassed
                  ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                  : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
              }`}>
                <div className="text-sm font-semibold mb-1">
                  校验余数 = <span className="font-mono">{verifyResult.crc}</span>
                </div>
                {verifyPassed ? (
                  <div className="text-green-700 dark:text-green-300 font-semibold">
                    余数为0，数据传输正确 ✓
                  </div>
                ) : (
                  <div className="text-red-700 dark:text-red-300 font-semibold">
                    余数非0，检测到错误 ✗
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
