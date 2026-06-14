"use client";

import React, { useState, useCallback } from "react";

/** 逆波兰表达式求值（后缀表达式）步进演示 */

const PRESETS: Record<string, string> = {
  "(2+1)×3": "2 1 + 3 *",
  "5+1×(2+4)×3": "5 1 2 4 + * 3 * +",
  "(4+5÷2)×(7-2)": "4 5 2 / + 7 2 - *",
  "简单减法": "10 3 - 4 2 / +",
  "含负数": "15 7 1 1 + - / 3 * 2 1 1 + + -",
};

interface Step {
  tokenIdx: number;
  token: string;
  action: "number" | "operator";
  stackSnapshot: number[];
  operands?: [number, number];
  result?: number;
  description: string;
}

function buildSteps(tokens: string[]): Step[] {
  const stack: number[] = [];
  const steps: Step[] = [];

  for (let ti = 0; ti < tokens.length; ti++) {
    const token = tokens[ti];
    if (["+", "-", "*", "/"].includes(token)) {
      const b = stack.pop()!;
      const a = stack.pop()!;
      let res: number;
      if (token === "+") res = a + b;
      else if (token === "-") res = a - b;
      else if (token === "*") res = a * b;
      else res = Math.trunc(a / b);
      stack.push(res);
      steps.push({
        tokenIdx: ti, token, action: "operator",
        stackSnapshot: [...stack],
        operands: [a, b], result: res,
        description: `弹出 ${b}（b）和 ${a}（a），计算 ${a} ${token} ${b} = ${res}，结果压栈`,
      });
    } else {
      const val = parseInt(token);
      stack.push(val);
      steps.push({
        tokenIdx: ti, token, action: "number",
        stackSnapshot: [...stack],
        description: `数字 ${val}，直接压栈`,
      });
    }
  }
  return steps;
}

function formatExpr(tokens: string[], activeIdx: number): React.ReactNode {
  return tokens.map((t, i) => {
    const isActive = i === activeIdx;
    const isOp = ["+", "-", "*", "/"].includes(t);
    return (
      <span key={i}
        className={`inline-block px-1.5 py-0.5 mx-0.5 rounded text-xs font-mono font-bold transition-all ${
          isActive
            ? isOp ? "bg-orange-500 text-white scale-110" : "bg-green-500 text-white scale-110"
            : i < activeIdx
            ? "bg-bg-tertiary text-text-tertiary"
            : isOp
            ? "bg-orange-500/20 text-orange-300 border border-orange-500/30"
            : "bg-blue-500/20 text-blue-300 border border-blue-500/30"
        }`}
      >
        {t}
      </span>
    );
  });
}

export default function ExpressionEval() {
  const [preset, setPreset] = useState("(2+1)×3");
  const [customInput, setCustomInput] = useState("");
  const [stepIdx, setStepIdx] = useState(0);

  const rawExpr = customInput.trim() || PRESETS[preset];
  const tokens = rawExpr.split(/\s+/).filter(Boolean);

  // 验证：确保 tokens 是有效的 RPN
  const isValidRPN = (() => {
    let cnt = 0;
    for (const t of tokens) {
      if (["+", "-", "*", "/"].includes(t)) { cnt--; if (cnt < 0) return false; }
      else if (!isNaN(Number(t))) cnt++;
      else return false;
    }
    return cnt === 1;
  })();

  const steps = isValidRPN ? buildSteps(tokens) : [];
  const step = steps[stepIdx];

  const reset = () => setStepIdx(0);
  const prev = () => setStepIdx((s) => Math.max(0, s - 1));
  const next = () => setStepIdx((s) => Math.min(steps.length - 1, s + 1));

  const calcExprPreview = useCallback(() => {
    const m: Record<string, string> = {
      "(2+1)×3": "(2+1)×3 = 9",
      "5+1×(2+4)×3": "5+1×(2+4)×3 = 23",
      "(4+5÷2)×(7-2)": "(4+5÷2)×(7-2) = 32",
      "简单减法": "10−3 + 4÷2 = 9",
      "含负数": "(15÷(7−1−1))×3 − (2+1+1) = 5",
    };
    return m[preset] || "";
  }, [preset]);

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-base font-bold text-text-primary">🧮 逆波兰表达式求值</h3>
          <p className="text-xs text-text-tertiary mt-0.5">
            {calcExprPreview() && <span>中缀：{calcExprPreview()}</span>}
          </p>
        </div>
      </div>

      {/* 预设选择 */}
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-xs text-text-tertiary">预设：</span>
        {Object.keys(PRESETS).map((p) => (
          <button key={p} onClick={() => { setPreset(p); setCustomInput(""); reset(); }}
            className={`px-2 py-1 rounded text-xs border transition-colors ${
              preset === p && !customInput
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"
            }`}>
            {p}
          </button>
        ))}
      </div>

      {/* 自定义输入 */}
      <div className="flex gap-2 items-center">
        <input
          type="text"
          value={customInput}
          onChange={(e) => { setCustomInput(e.target.value); reset(); }}
          placeholder="自定义 RPN（如：3 4 2 * + ），空格分隔"
          className="flex-1 px-3 py-1.5 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-xs focus:outline-none focus:border-blue-400"
        />
        {!isValidRPN && customInput && (
          <span className="text-red-400 text-xs">⚠️ 无效 RPN</span>
        )}
      </div>

      {/* Token 序列展示 */}
      <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
        <div className="text-xs text-text-tertiary mb-2">后缀表达式（逐 token 处理）：</div>
        <div className="flex flex-wrap gap-0.5">
          {formatExpr(tokens, step?.tokenIdx ?? -1)}
        </div>
        <div className="flex gap-4 text-[10px] text-text-tertiary mt-2">
          <span><span className="inline-block px-1 rounded bg-green-500 text-white text-[9px] mr-1">42</span>数字 → 压栈</span>
          <span><span className="inline-block px-1 rounded bg-orange-500 text-white text-[9px] mr-1">＋</span>运算符 → 弹出两数计算</span>
          <span><span className="inline-block px-1 rounded bg-bg-secondary text-text-tertiary text-[9px] border border-border-subtle mr-1">·</span>已处理</span>
        </div>
      </div>

      {/* 步骤说明 */}
      <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle min-h-[64px]">
        <div className="flex justify-between text-xs text-text-tertiary mb-1">
          <span>步骤 {stepIdx + 1} / {steps.length}</span>
          {step && (
            <span className={`font-bold ${step.action === "number" ? "text-green-400" : "text-orange-400"}`}>
              {step.action === "number" ? "PUSH 数字" : `计算 ${step.token}`}
            </span>
          )}
        </div>
        <p className="text-text-primary text-sm">{step?.description ?? "点击「下一步」开始"}</p>
        {step?.action === "operator" && step.operands && (
          <div className="mt-2 text-sm text-text-secondary">
            <span className="text-blue-300">{step.operands[0]}</span>
            <span className="text-orange-400 mx-2 font-bold">{step.token}</span>
            <span className="text-blue-300">{step.operands[1]}</span>
            <span className="text-text-tertiary mx-2">=</span>
            <span className="text-green-400 font-bold text-base">{step.result}</span>
          </div>
        )}
      </div>

      {/* 栈可视化 */}
      <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
        <div className="text-xs text-text-tertiary mb-2">栈状态（底→顶）：</div>
        <div className="flex gap-2 items-end min-h-[60px]">
          {(!step || step.stackSnapshot.length === 0) ? (
            <span className="text-text-tertiary text-xs italic">（空栈）</span>
          ) : (
            step.stackSnapshot.map((v, i) => {
              const isTop = i === step.stackSnapshot.length - 1;
              return (
                <div key={i}
                  className={`flex flex-col items-center px-3 py-2 rounded border transition-all duration-300 ${
                    isTop && step.action === "operator"
                      ? "bg-green-500/20 border-green-500 text-green-200 scale-110"
                      : isTop
                      ? "bg-blue-500/20 border-blue-500 text-blue-200"
                      : "bg-bg-secondary border-border-subtle text-text-secondary"
                  }`}>
                  <span className="font-bold text-sm">{v}</span>
                  {isTop && <span className="text-[9px] opacity-60 mt-0.5">栈顶</span>}
                </div>
              );
            })
          )}
        </div>
        {step && step.stackSnapshot.length > 0 && (
          <div className="text-[10px] text-text-tertiary mt-1">← 栈底 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 栈顶 →</div>
        )}
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-2 justify-center">
        <button onClick={reset}
          className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs transition-colors">
          ↩ 重置
        </button>
        <button onClick={prev} disabled={stepIdx === 0}
          className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs transition-colors disabled:opacity-40">
          ← 上一步
        </button>
        <button onClick={next} disabled={stepIdx === steps.length - 1 || steps.length === 0}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white text-xs hover:bg-blue-700 transition-colors disabled:opacity-40">
          下一步 →
        </button>
      </div>

      {/* 最终结果 */}
      {stepIdx === steps.length - 1 && steps.length > 0 && (
        <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-4 text-center">
          <div className="text-green-400 text-xs font-semibold mb-1">✅ 求值完成！</div>
          <div className="text-green-300 text-3xl font-bold">
            {step?.stackSnapshot[0]}
          </div>
          <div className="text-text-tertiary text-xs mt-1">
            后缀：{rawExpr}
          </div>
        </div>
      )}
    </div>
  );
}
