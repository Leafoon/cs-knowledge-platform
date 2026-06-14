"use client";
import { useState } from "react";

interface Instruction {
  op: string;
  code: string;
  desc: string;
  jt: number;
  jf: number;
  k: number;
}

const programs: { name: string; desc: string; instructions: Instruction[] }[] = [
  {
    name: "捕获所有TCP流量",
    desc: "过滤所有TCP协议数据包",
    instructions: [
      { op: "LD", code: "BPF_LD+BPF_H+BPF_ABS", desc: "加载以太网类型字段", jt: 0, jf: 0, k: 12 },
      { op: "JEQ", code: "BPF_JMP+BPF_JEQ+BPF_K", desc: "是否为IPv4 (0x0800)?", jt: 0, jf: 4, k: 2048 },
      { op: "LD", code: "BPF_LD+BPF_B+BPF_ABS", desc: "加载IP协议字段", jt: 0, jf: 0, k: 23 },
      { op: "JEQ", code: "BPF_JMP+BPF_JEQ+BPF_K", desc: "是否为TCP (6)?", jt: 1, jf: 0, k: 6 },
      { op: "RET", code: "BPF_RET+BPF_K", desc: "接受 - 返回完整数据包", jt: 0, jf: 0, k: 65535 },
      { op: "RET", code: "BPF_RET+BPF_K", desc: "丢弃 - 返回0", jt: 0, jf: 0, k: 0 },
    ],
  },
  {
    name: "捕获HTTP端口流量",
    desc: "过滤目标端口80的TCP数据包",
    instructions: [
      { op: "LD", code: "BPF_LD+BPF_H+BPF_ABS", desc: "加载EtherType", jt: 0, jf: 0, k: 12 },
      { op: "JEQ", code: "BPF_JMP+BPF_JEQ+BPF_K", desc: "IPv4?", jt: 0, jf: 7, k: 2048 },
      { op: "LD", code: "BPF_LD+BPF_B+BPF_ABS", desc: "加载IP协议", jt: 0, jf: 0, k: 23 },
      { op: "JEQ", code: "BPF_JMP+BPF_JEQ+BPF_K", desc: "TCP?", jt: 0, jf: 5, k: 6 },
      { op: "LDH", code: "BPF_LD+BPF_H+BPF_ABS", desc: "加载目标端口", jt: 0, jf: 0, k: 22 },
      { op: "JEQ", code: "BPF_JMP+BPF_JEQ+BPF_K", desc: "端口80?", jt: 1, jf: 0, k: 80 },
      { op: "RET", code: "BPF_RET+BPF_K", desc: "丢弃", jt: 0, jf: 0, k: 0 },
      { op: "RET", code: "BPF_RET+BPF_K", desc: "接受 - 返回全部字节", jt: 0, jf: 0, k: 65535 },
    ],
  },
];

export function BPFExplorer() {
  const [progIdx, setProgIdx] = useState(0);
  const [pc, setPC] = useState(-1);
  const [acc, setAcc] = useState(0);
  const [result, setResult] = useState<string | null>(null);

  const prog = programs[progIdx];

  const reset = () => { setPC(-1); setAcc(0); setResult(null); };

  const step = () => {
    if (pc === -1) { setPC(0); setResult(null); return; }
    const inst = prog.instructions[pc];
    if (inst.op === "RET") {
      setResult(inst.k > 0 ? `ACCEPT (${inst.k} bytes)` : "DROP");
      return;
    }
    if (inst.op === "LD" || inst.op === "LDH") { setAcc(inst.k); }
    const nextPC = inst.op === "JEQ" ? (acc === inst.k ? pc + 1 + inst.jt : pc + 1 + inst.jf) : pc + 1;
    if (nextPC >= prog.instructions.length) { setResult("END"); return; }
    setPC(nextPC);
  };

  const runAll = () => {
    reset();
    let curPC = 0;
    let curAcc = 0;
    for (let i = 0; i < 20; i++) {
      const inst = prog.instructions[curPC];
      if (inst.op === "RET") { setResult(inst.k > 0 ? `ACCEPT (${inst.k} bytes)` : "DROP"); setPC(curPC); return; }
      if (inst.op === "LD" || inst.op === "LDH") curAcc = inst.k;
      curPC = inst.op === "JEQ" ? (curAcc === inst.k ? curPC + 1 + inst.jt : curPC + 1 + inst.jf) : curPC + 1;
      if (curPC >= prog.instructions.length) { setResult("END"); setPC(curPC); return; }
    }
    setPC(curPC);
    setAcc(curAcc);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">BPF 过滤器指令集</h3>
      <div className="flex gap-2 mb-4">
        {programs.map((p, i) => (
          <button key={i} onClick={() => { setProgIdx(i); reset(); }}
            className={`px-3 py-1.5 rounded text-sm ${progIdx === i ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {p.name}
          </button>
        ))}
      </div>
      <p className="text-xs text-text-secondary mb-3">{prog.desc}</p>
      <div className="mb-3 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-text-secondary border-b border-border-subtle">
              <th className="py-1 px-2 text-left">PC</th>
              <th className="py-1 px-2 text-left">操作码</th>
              <th className="py-1 px-2 text-left">描述</th>
              <th className="py-1 px-2 text-center">JT</th>
              <th className="py-1 px-2 text-center">JF</th>
              <th className="py-1 px-2 text-center">K</th>
            </tr>
          </thead>
          <tbody>
            {prog.instructions.map((inst, i) => (
              <tr key={i} className={`border-t border-border-subtle transition-colors ${i === pc ? "bg-blue-50 dark:bg-blue-900/20" : ""}`}>
                <td className="py-1 px-2 font-mono text-text-primary">{i}</td>
                <td className="py-1 px-2 font-mono text-text-primary">{inst.op}</td>
                <td className="py-1 px-2 text-text-secondary">{inst.desc}</td>
                <td className="py-1 px-2 text-center font-mono text-text-primary">{inst.jt}</td>
                <td className="py-1 px-2 text-center font-mono text-text-primary">{inst.jf}</td>
                <td className="py-1 px-2 text-center font-mono text-text-primary">{inst.k}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex gap-2 items-center mb-3">
        <button onClick={step} className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">单步执行</button>
        <button onClick={runAll} className="px-4 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded text-sm">运行全部</button>
        <button onClick={reset} className="px-4 py-1.5 bg-gray-500 hover:bg-gray-600 text-white rounded text-sm">重置</button>
        <span className="text-xs text-text-secondary ml-2">PC: {pc >= 0 ? pc : "-"} | A: {acc}</span>
      </div>
      {result && (
        <div className={`p-3 rounded text-sm font-medium ${result.includes("ACCEPT") ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300" : "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300"}`}>
          结果: {result}
        </div>
      )}
      <p className="text-xs text-text-secondary mt-3">BPF (Berkeley Packet Filter) 在内核态执行过滤，避免将不需要的数据包复制到用户态。</p>
    </div>
  );
}
export default BPFExplorer;
