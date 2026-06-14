"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Play } from "lucide-react";

type Op = "+" | "-" | "×";

function toBin8(n: number): string {
  if (n < 0) return (256 + n).toString(2).padStart(8, "0");
  return n.toString(2).padStart(8, "0");
}

function checkOverflow(a: number, b: number, op: Op): { result: number; overflow: boolean; details: string } {
  let r: number;
  if (op === "+") r = a + b;
  else if (op === "-") r = a - b;
  else r = a * b;

  const overflow = r > 127 || r < -128;
  const masked = ((r + 128) % 256 + 256) % 256 - 128;
  const details = overflow
    ? `结果 ${r} 超出 [-128, 127] 范围，发生溢出！截断为 ${masked}`
    : `结果 ${r} 在范围内，无溢出。`;

  return { result: r, overflow, details };
}

function detectOverflowByBits(a: number, b: number, op: Op): string {
  const signA = a < 0 ? 1 : 0;
  const signB = (op === "-" ? -b : b) < 0 ? 1 : 0;
  const r = op === "+" ? a + b : op === "-" ? a - b : a * b;
  const signR = r < 0 ? 1 : 0;
  if (op === "×") return "乘法溢出需检查结果位宽";
  if (signA === signB && signA !== signR) return "同号相加（减）结果异号 → 溢出！";
  return "符号位一致 → 无溢出";
}

export function ArithmeticOverflowSuite() {
  const [a, setA] = useState(100);
  const [b, setB] = useState(50);
  const [op, setOp] = useState<Op>("+");
  const [result, setResult] = useState<ReturnType<typeof checkOverflow> | null>(null);
  const [bitCheck, setBitCheck] = useState("");

  const run = () => {
    setResult(checkOverflow(a, b, op));
    setBitCheck(detectOverflowByBits(a, b, op));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <AlertTriangle className="w-5 h-5 text-yellow-500" />
        算术溢出综合测试
      </h3>

      <div className="flex flex-wrap items-end gap-4 mb-4">
        <div>
          <label className="block text-xs text-text-muted mb-1">操作数 A</label>
          <input type="number" value={a} onChange={e => setA(Number(e.target.value))}
            className="w-24 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
          <div className="text-xs text-text-muted mt-1 font-mono">{toBin8(a)}</div>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">运算符</label>
          <div className="flex gap-1">
            {(["+", "-", "×"] as Op[]).map(o => (
              <button key={o} onClick={() => setOp(o)}
                className={`px-3 py-1 rounded text-sm font-mono ${op === o ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
                {o}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">操作数 B</label>
          <input type="number" value={b} onChange={e => setB(Number(e.target.value))}
            className="w-24 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
          <div className="text-xs text-text-muted mt-1 font-mono">{toBin8(b)}</div>
        </div>
        <button onClick={run} className="px-4 py-1.5 rounded bg-blue-500 text-white text-sm flex items-center gap-1 hover:bg-blue-600">
          <Play className="w-4 h-4" /> 检测
        </button>
      </div>

      {result && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          className={`p-4 rounded border ${result.overflow ? "border-red-500 bg-red-500/10" : "border-green-500 bg-green-500/10"}`}>
          <div className="flex items-center gap-2 mb-2">
            {result.overflow
              ? <AlertTriangle className="w-5 h-5 text-red-400" />
              : <span className="w-5 h-5 text-green-400">✓</span>}
            <span className={`font-semibold ${result.overflow ? "text-red-400" : "text-green-400"}`}>
              {result.overflow ? "溢出！" : "无溢出"}
            </span>
          </div>
          <p className="text-sm text-text-secondary mb-1">{result.details}</p>
          <p className="text-sm text-text-secondary">符号位检测：{bitCheck}</p>
        </motion.div>
      )}
    </div>
  );
}
