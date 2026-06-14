"use client";

import { useState } from "react";

const tirdItems = [
  {
    id: "primfunc",
    name: "PrimFunc",
    type: "顶层",
    desc: "TIR 函数的根节点",
    detail: "PrimFunc 是 TIR 的入口\n包含:\n- 参数 (Buffer/Var)\n- 属性 (target, etc.)\n- 语句体 (body)",
    children: ["params", "body"],
  },
  {
    id: "params",
    name: "参数列表",
    type: "参数",
    desc: "Buffer 和标量参数",
    detail: "params: List[Buffer | Var]\n- Buffer: A[T.Buffer[(128, 128), 'float32']]\n- Var: n: T.int32",
    children: [],
  },
  {
    id: "body",
    name: "语句体",
    type: "语句",
    desc: "函数体，由语句组成",
    detail: "body: Stmt\n可能是:\n- Block: 数据并行块\n- For: 循环语句\n- SeqStmt: 语句序列\n- IfThenElse: 条件语句",
    children: ["for_loop", "block"],
  },
  {
    id: "for_loop",
    name: "For 循环",
    type: "语句",
    desc: "循环迭代语句",
    detail: "For(loop_var, min, extent, kind, body)\n- loop_var: 循环变量 vi\n- min: 起始值 0\n- extent: 迭代次数 128\n- kind: Serial/Parallel/Vectorized",
    children: ["block"],
  },
  {
    id: "block",
    name: "Block",
    type: "语句",
    desc: "计算块，包含读写和计算",
    detail: "Block:\n- reads: [A[vi, vj]]\n- writes: [B[vi, vj]]\n- body: BufferStore\n  B[vi, vj] = A[vi, vj] * 2.0",
    children: ["store"],
  },
  {
    id: "store",
    name: "BufferStore",
    type: "表达式",
    desc: "写入 Buffer 的表达式",
    detail: "BufferStore(buffer, value, indices)\n- buffer: B\n- value: A[vi, vj] * 2.0\n- indices: [vi, vj]\n\n子表达式:\n  BufferLoad(A, [vi, vj])\n  Mul(BufferLoad, 2.0f)",
    children: [],
  },
];

export function TIRInspectionTool() {
  const [selected, setSelected] = useState("primfunc");

  const item = tirdItems.find((t) => t.id === selected)!;

  const typeColors: Record<string, string> = {
    "顶层": "bg-indigo-500",
    "参数": "bg-blue-500",
    "语句": "bg-emerald-500",
    "表达式": "bg-amber-500",
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">TIR 检查工具</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">PrimFunc 的语句和表达式树结构</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">AST 节点</h4>
          <div className="space-y-1.5">
            {tirdItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setSelected(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all ${
                  selected === item.id
                    ? "bg-indigo-100 dark:bg-indigo-900/40 border border-indigo-300 dark:border-indigo-600"
                    : "hover:bg-slate-50 dark:hover:bg-slate-700/50"
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${typeColors[item.type]}`} />
                <div className="flex-1">
                  <div className="text-xs font-bold text-slate-700 dark:text-slate-200">{item.name}</div>
                  <div className="text-[10px] text-slate-500 dark:text-slate-400">{item.desc}</div>
                </div>
                <span className={`text-[10px] px-1.5 py-0.5 rounded text-white ${typeColors[item.type]}`}>
                  {item.type}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">{item.name}</h4>
            <pre className="text-xs text-slate-600 dark:text-slate-300 whitespace-pre-wrap font-mono leading-relaxed">
              {item.detail}
            </pre>
            {item.children.length > 0 && (
              <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
                <span className="text-xs text-slate-500 dark:text-slate-400">子节点: </span>
                {item.children.map((c) => (
                  <button
                    key={c}
                    onClick={() => setSelected(c)}
                    className="text-xs text-indigo-500 hover:text-indigo-700 dark:hover:text-indigo-300 ml-1 underline"
                  >
                    {tirdItems.find((t) => t.id === c)?.name}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
            <pre>{`@T.prim_func
def main(A: T.Buffer[(128, 128), "float32"],
         B: T.Buffer[(128, 128), "float32"]):
    for vi in T.serial(128):      # For
      for vj in T.serial(128):    # For
        with T.block("B"):        # Block
          B[vi, vj] = A[vi, vj] * 2.0  # BufferStore`}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
