"use client";

export function ModuleHierarchyDiagram() {
  const levels = [
    { name: "IRModule", desc: "顶层模块，包含所有函数定义", indent: 0, color: "text-blue-600 dark:text-blue-400", bg: "bg-blue-50 dark:bg-blue-900/30", border: "border-blue-200 dark:border-blue-800" },
    { name: "Function", desc: "Relay Function 或 PrimFunc", indent: 1, color: "text-indigo-600 dark:text-indigo-400", bg: "bg-indigo-50 dark:bg-indigo-900/30", border: "border-indigo-200 dark:border-indigo-800" },
    { name: "Body", desc: "函数体 (Expr / Stmt)", indent: 2, color: "text-purple-600 dark:text-purple-400", bg: "bg-purple-50 dark:bg-purple-900/30", border: "border-purple-200 dark:border-purple-800" },
    { name: "Call", desc: "算子调用表达式", indent: 3, color: "text-violet-600 dark:text-violet-400", bg: "bg-violet-50 dark:bg-violet-900/30", border: "border-violet-200 dark:border-violet-800" },
    { name: "Var", desc: "变量引用", indent: 4, color: "text-fuchsia-600 dark:text-fuchsia-400", bg: "bg-fuchsia-50 dark:bg-fuchsia-900/30", border: "border-fuchsia-200 dark:border-fuchsia-800" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        Module 层次结构
      </h3>
      <div className="flex flex-col gap-2">
        {levels.map((level, i) => (
          <div key={i} className={`ml-${level.indent * 6} flex items-center gap-3`}>
            <div className={`border-l-2 ${level.border} ${level.bg} rounded-r-lg px-4 py-3 flex-1`}>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400 font-mono w-4">{">"}</span>
                <span className={`font-bold text-sm ${level.color}`}>{level.name}</span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 ml-6 mt-1">{level.desc}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>IR 层次：</strong>IRModule 包含多个 Function，每个 Function 有 Body（由 Call 和 Var 组成的表达式树）。
      </div>
    </div>
  );
}
