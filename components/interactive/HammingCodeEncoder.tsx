"use client";

import { useState } from "react";

function getParityBits(n: number): number {
  let r = 0;
  while (Math.pow(2, r) < n + r + 1) r++;
  return r;
}

function encodeHamming(dataBits: number[]): { code: number[]; parityPositions: number[]; dataPositions: number[]; groups: number[][] } {
  const n = dataBits.length;
  const r = getParityBits(n);
  const totalLen = n + r;

  const parityPositions: number[] = [];
  const dataPositions: number[] = [];
  for (let i = 1; i <= totalLen; i++) {
    if ((i & (i - 1)) === 0) parityPositions.push(i);
    else dataPositions.push(i);
  }

  const code = new Array(totalLen + 1).fill(0);
  let dataIdx = 0;
  for (const pos of dataPositions) {
    code[pos] = dataBits[dataIdx++];
  }

  const groups: number[][] = [];
  for (let p = 0; p < r; p++) {
    const parityPos = Math.pow(2, p);
    const group: number[] = [];
    for (let i = 1; i <= totalLen; i++) {
      if (i & parityPos) group.push(i);
    }
    let parity = 0;
    for (const pos of group) parity ^= code[pos];
    code[parityPos] = parity;
    groups.push(group);
  }

  return { code: code.slice(1), parityPositions, dataPositions, groups: groups.map(g => g.map(x => x)) };
}

function detectAndCorrect(received: number[], r: number): { syndrome: number; errorPos: number; corrected: number[]; groupChecks: { group: number[]; parity: number; ok: boolean }[] } {
  const totalLen = received.length;
  const code = [0, ...received];
  const groupChecks: { group: number[]; parity: number; ok: boolean }[] = [];
  let syndrome = 0;

  for (let p = 0; p < r; p++) {
    const parityPos = Math.pow(2, p);
    const group: number[] = [];
    let parity = 0;
    for (let i = 1; i <= totalLen; i++) {
      if (i & parityPos) {
        group.push(i);
        parity ^= code[i];
      }
    }
    groupChecks.push({ group, parity, ok: parity === 0 });
    if (parity !== 0) syndrome |= parityPos;
  }

  const corrected = [...received];
  const errorPos = syndrome;
  if (errorPos > 0 && errorPos <= totalLen) {
    corrected[errorPos - 1] ^= 1;
  }

  return { syndrome, errorPos, corrected, groupChecks };
}

const GROUP_COLORS = [
  { bg: "bg-rose-100 dark:bg-rose-900/30", text: "text-rose-700 dark:text-rose-300", border: "border-rose-300 dark:border-rose-700" },
  { bg: "bg-emerald-100 dark:bg-emerald-900/30", text: "text-emerald-700 dark:text-emerald-300", border: "border-emerald-300 dark:border-emerald-700" },
  { bg: "bg-sky-100 dark:bg-sky-900/30", text: "text-sky-700 dark:text-sky-300", border: "border-sky-300 dark:border-sky-700" },
  { bg: "bg-violet-100 dark:bg-violet-900/30", text: "text-violet-700 dark:text-violet-300", border: "border-violet-300 dark:border-violet-700" },
];

export function HammingCodeEncoder() {
  const [dataInput, setDataInput] = useState("1011");
  const [flipPos, setFlipPos] = useState<number | null>(null);

  const dataBits = dataInput.split("").map(Number).filter((d) => d === 0 || d === 1);
  const validLength = dataBits.length >= 1 && dataBits.length <= 26;
  const r = validLength ? getParityBits(dataBits.length) : 0;
  const canEncode = validLength && r > 0;

  const encoded = canEncode ? encodeHamming(dataBits) : null;
  const totalLen = canEncode ? dataBits.length + r : 0;

  const received = encoded
    ? flipPos !== null && flipPos >= 1 && flipPos <= totalLen
      ? encoded.code.map((b, i) => (i === flipPos - 1 ? b ^ 1 : b))
      : encoded.code
    : [];

  const detection = received.length > 0 ? detectAndCorrect(received, r) : null;

  const handleFlip = (pos: number) => {
    setFlipPos(flipPos === pos ? null : pos);
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-slate-900 dark:to-orange-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-6 text-slate-900 dark:text-white">
        海明码编码与纠错演示
      </h3>

      {/* Input */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
          输入数据位（1~26位二进制）
        </label>
        <div className="flex gap-3">
          <input
            type="text"
            value={dataInput}
            onChange={(e) => { setDataInput(e.target.value.replace(/[^01]/g, "").slice(0, 26)); setFlipPos(null); }}
            className="flex-1 px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-lg focus:outline-none focus:ring-2 focus:ring-orange-500/20"
            placeholder="例如：1011"
          />
          <button
            onClick={() => { setDataInput("1011"); setFlipPos(null); }}
            className="px-4 py-3 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 rounded-lg hover:bg-orange-200 dark:hover:bg-orange-900/50 transition text-sm font-semibold"
          >
            重置
          </button>
        </div>
        {canEncode && (
          <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
            数据位 n={dataBits.length}，校验位 r={r}（2<sup>{r}</sup>={Math.pow(2, r)} ≥ {dataBits.length}+{r}+1={dataBits.length + r + 1}），总长度 {totalLen} 位
          </p>
        )}
      </div>

      {canEncode && encoded && (
        <>
          {/* Step 1: Bit positions */}
          <div className="mb-6">
            <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
              步骤1：位号分配（校验位在2的幂次方位置）
            </h4>
            <div className="flex gap-1 flex-wrap">
              {Array.from({ length: totalLen }, (_, i) => i + 1).map((pos) => {
                const isParity = encoded.parityPositions.includes(pos);
                return (
                  <div key={pos} className="flex flex-col items-center">
                    <div className="text-xs text-slate-400 mb-1">{pos}</div>
                    <div
                      className={`w-10 h-10 rounded flex items-center justify-center font-mono font-bold text-sm ${
                        isParity
                          ? "bg-amber-400 dark:bg-amber-600 text-amber-900 dark:text-amber-100"
                          : "bg-blue-500 dark:bg-blue-600 text-white"
                      }`}
                    >
                      {encoded.code[pos - 1]}
                    </div>
                    <div className="text-xs mt-1 font-semibold">
                      {isParity ? (
                        <span className="text-amber-600 dark:text-amber-400">P{Math.log2(pos)}</span>
                      ) : (
                        <span className="text-blue-600 dark:text-blue-400">D</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Step 2: Parity groups */}
          <div className="mb-6">
            <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
              步骤2：校验组覆盖关系
            </h4>
            <div className="space-y-2">
              {encoded.groups.map((group, p) => {
                const color = GROUP_COLORS[p % GROUP_COLORS.length];
                const parityPos = Math.pow(2, p);
                return (
                  <div key={p} className={`p-3 rounded-lg border ${color.border} ${color.bg}`}>
                    <span className={`text-sm font-semibold ${color.text}`}>
                      P{p}（位{parityPos}）覆盖：
                    </span>
                    <span className="font-mono text-sm ml-2">
                      {group.join(", ")}
                    </span>
                    <span className={`text-sm ml-2 ${color.text}`}>
                      → 值 = {encoded.code[parityPos - 1]}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Step 3: Final Hamming code */}
          <div className="mb-6">
            <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
              步骤3：最终海明码
            </h4>
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
              <div className="flex gap-1 flex-wrap">
                {encoded.code.map((bit, i) => (
                  <div
                    key={i}
                    className="w-10 h-10 bg-orange-500 dark:bg-orange-600 text-white rounded flex items-center justify-center font-mono font-bold text-sm"
                  >
                    {bit}
                  </div>
                ))}
              </div>
              <p className="mt-2 text-sm font-mono text-orange-700 dark:text-orange-300">
                海明码 = {encoded.code.join("")}
              </p>
            </div>
          </div>

          {/* Error detection section */}
          <div className="mb-4">
            <h4 className="font-semibold mb-3 text-slate-800 dark:text-slate-200">
              错误检测与纠错（点击任意位模拟错误）
            </h4>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">
              点击某一位可翻转其值，模拟传输错误。再次点击恢复。
            </p>

            <div className="flex gap-1 flex-wrap mb-4">
              {Array.from({ length: totalLen }, (_, i) => i + 1).map((pos) => {
                const isParity = encoded.parityPositions.includes(pos);
                const isFlipped = flipPos === pos;
                const isCorrected = detection && detection.errorPos === pos && flipPos !== null;
                return (
                  <div key={pos} className="flex flex-col items-center">
                    <div className="text-xs text-slate-400 mb-1">{pos}</div>
                    <button
                      onClick={() => handleFlip(pos)}
                      className={`w-10 h-10 rounded flex items-center justify-center font-mono font-bold text-sm transition-all cursor-pointer ${
                        isFlipped
                          ? "bg-red-500 text-white animate-pulse shadow-lg shadow-red-500/30"
                          : isCorrected
                            ? "bg-green-500 text-white shadow-lg shadow-green-500/30"
                            : isParity
                              ? "bg-amber-400 dark:bg-amber-600 text-amber-900 dark:text-amber-100 hover:ring-2 hover:ring-red-400"
                              : "bg-blue-500 dark:bg-blue-600 text-white hover:ring-2 hover:ring-red-400"
                      }`}
                    >
                      {received[pos - 1]}
                    </button>
                  </div>
                );
              })}
            </div>

            {detection && flipPos !== null && (
              <div className="space-y-3">
                {/* Group checks */}
                <div className="space-y-1">
                  {detection.groupChecks.map((check, p) => {
                    const color = GROUP_COLORS[p % GROUP_COLORS.length];
                    return (
                      <div key={p} className={`p-2 rounded border text-sm ${check.ok ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800" : `${color.bg} ${color.border}`}`}>
                        <span className="font-semibold">S{p + 1} = </span>
                        <span className="font-mono">
                          {check.group.map((pos) => received[pos - 1]).join(" ⊕ ")}
                        </span>
                        <span className="mx-2">=</span>
                        <span className={`font-bold ${check.ok ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
                          {check.parity}
                        </span>
                        <span className="ml-2">
                          {check.ok ? "✓" : "✗"}
                        </span>
                      </div>
                    );
                  })}
                </div>

                {/* Syndrome */}
                <div className={`p-4 rounded-lg border ${
                  detection.errorPos === 0
                    ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                    : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                }`}>
                  <div className="text-sm font-semibold mb-1">
                    校验子（Syndrome）= S{r}...S2S1 = {detection.groupChecks.map(c => c.parity).reverse().join("")}<sub>2</sub>
                  </div>
                  {detection.errorPos === 0 ? (
                    <div className="text-green-700 dark:text-green-300 font-semibold">
                      校验子为0，无错误
                    </div>
                  ) : (
                    <>
                      <div className="text-red-700 dark:text-red-300 font-semibold">
                        错误位号 = {detection.errorPos}
                      </div>
                      <div className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                        纠正：将第 {detection.errorPos} 位取反 → 纠正后为 {detection.corrected.join("")}
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {detection && flipPos === null && (
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                <span className="text-green-700 dark:text-green-300 font-semibold">
                  无错误。点击任意位可模拟传输错误。
                </span>
              </div>
            )}
          </div>
        </>
      )}

      {!canEncode && dataInput.length > 0 && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
          <span className="text-red-700 dark:text-red-300 text-sm">
            请输入合法的二进制数据（1~26位）
          </span>
        </div>
      )}
    </div>
  );
}
