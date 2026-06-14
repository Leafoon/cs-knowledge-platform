"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Calculator } from "lucide-react";

type Op = "+" | "-" | "×" | "÷";

function toTwos(n: number, bits: number): string {
  if (n >= 0) return n.toString(2).padStart(bits, "0");
  return ((1 << bits) + n).toString(2).padStart(bits, "0");
}

function parseSigned(bin: string): number {
  if (bin[0] === "0") return parseInt(bin, 2);
  return parseInt(bin, 2) - (1 << bin.length);
}

function addBin(a: string, b: string): { result: string; steps: string[] } {
  const bits = a.length;
  const steps: string[] = [];
  let carry = 0;
  let result = "";
  for (let i = bits - 1; i >= 0; i--) {
    const bitA = parseInt(a[i]);
    const bitB = parseInt(b[i]);
    const sum = bitA + bitB + carry;
    result = (sum % 2) + result;
    const oldCarry = carry;
    carry = Math.floor(sum / 2);
    if (i >= bits - 4 || i === 0) {
      steps.push(`位${i}: ${bitA} + ${bitB} + ${oldCarry} = ${sum} → 写${sum % 2}, 进位${carry}`);
    }
  }
  return { result: result.slice(-bits), steps };
}

export function BinaryArithmeticLab() {
  const [a, setA] = useState("0110");
  const [b, setB] = useState("0011");
  const [bits, setBits] = useState(8);
  const [op, setOp] = useState<Op>("+");
  const [output, setOutput] = useState<{ result: string; steps: string[]; decimal: number } | null>(null);

  const validate = (s: string) => /^[01]+$/.test(s) && s.length <= bits;

  const compute = () => {
    const pa = a.padStart(bits, "0").slice(-bits);
    const pb = b.padStart(bits, "0").slice(-bits);
    let resultBin: string;
    let steps: string[] = [];
    let decResult: number;

    const da = parseSigned(pa);
    const db = parseSigned(pb);

    switch (op) {
      case "+": {
        const r = addBin(pa, pb);
        resultBin = r.result;
        steps = r.steps;
        decResult = parseSigned(resultBin);
        break;
      }
      case "-": {
        const negB = toTwos(-db, bits);
        const r = addBin(pa, negB);
        resultBin = r.result;
        steps = [`取反加1: ${pb} → ${negB}`, ...r.steps];
        decResult = da - db;
        break;
      }
      case "×": {
        decResult = da * db;
        resultBin = toTwos(decResult, bits);
        steps = [`${da} × ${db} = ${decResult}`, `二进制: ${resultBin}`];
        break;
      }
      case "÷": {
        if (db === 0) {
          setOutput({ result: "ERROR", steps: ["除数不能为零！"], decimal: 0 });
          return;
        }
        decResult = Math.trunc(da / db);
        resultBin = toTwos(decResult, bits);
        steps = [`${da} ÷ ${db} = ${decResult}`, `二进制: ${resultBin}`];
        break;
      }
    }
    setOutput({ result: resultBin, steps, decimal: decResult });
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Calculator className="w-5 h-5 text-cyan-500" />
        二进制算术实验室
      </h3>

      <div className="flex flex-wrap gap-4 mb-4 items-end">
        <div>
          <label className="block text-xs text-text-muted mb-1">位宽</label>
          <select value={bits} onChange={e => setBits(Number(e.target.value))}
            className="px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm">
            {[4, 8, 16].map(b => <option key={b} value={b}>{b}位</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">操作数 A</label>
          <input value={a} onChange={e => setA(e.target.value.replace(/[^01]/g, ""))}
            className={`w-32 px-2 py-1 rounded border text-sm font-mono bg-bg-surface ${validate(a) ? "border-border-subtle" : "border-red-500"}`} />
        </div>
        <div className="flex gap-1">
          {(["+", "-", "×", "÷"] as Op[]).map(o => (
            <button key={o} onClick={() => setOp(o)}
              className={`w-8 h-8 rounded text-sm font-mono ${op === o ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
              {o}
            </button>
          ))}
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">操作数 B</label>
          <input value={b} onChange={e => setB(e.target.value.replace(/[^01]/g, ""))}
            className={`w-32 px-2 py-1 rounded border text-sm font-mono bg-bg-surface ${validate(b) ? "border-border-subtle" : "border-red-500"}`} />
        </div>
        <button onClick={compute} className="px-4 py-1.5 rounded bg-blue-500 text-white text-sm hover:bg-blue-600">
          计算
        </button>
      </div>

      <AnimatePresence>
        {output && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs text-text-muted mb-1">二进制结果</div>
                <div className="font-mono text-lg text-blue-400 break-all">{output.result}</div>
              </div>
              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs text-text-muted mb-1">十进制结果</div>
                <div className="font-mono text-lg text-green-400">{output.decimal}</div>
              </div>
            </div>
            {output.steps.length > 0 && (
              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs text-text-muted mb-2">计算过程</div>
                {output.steps.map((s, i) => (
                  <div key={i} className="text-xs font-mono text-text-secondary leading-relaxed">{s}</div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
