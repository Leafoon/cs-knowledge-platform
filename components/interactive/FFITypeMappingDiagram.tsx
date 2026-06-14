"use client";

import { useState } from "react";

const mappings = [
  { python: "int", tvm: "TVMValue.v_int64", cpp: "int64_t", color: "from-violet-400 to-purple-500" },
  { python: "float", tvm: "TVMValue.v_float64", cpp: "double", color: "from-indigo-400 to-blue-500" },
  { python: "str", tvm: "TVMValue.v_str", cpp: "const char*", color: "from-blue-400 to-cyan-500" },
  { python: "bool", tvm: "TVMValue.v_int64", cpp: "int64_t (0/1)", color: "from-cyan-400 to-teal-500" },
  { python: "NDArray", tvm: "TVMValue.v_handle", cpp: "DLTensor*", color: "from-teal-400 to-green-500" },
  { python: "PackedFunc", tvm: "TVMValue.v_handle", cpp: "TVMFunctionHandle", color: "from-green-400 to-emerald-500" },
  { python: "None", tvm: "TVMValue.v_handle=nullptr", cpp: "void* nullptr", color: "from-slate-400 to-slate-500" },
];

export function FFITypeMappingDiagram() {
  const [active, setActive] = useState<number | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        FFI 类型映射: Python → TVM → C++
      </h3>

      <div className="flex items-center justify-between mb-4">
        <span className="text-sm font-bold text-violet-600 dark:text-violet-400 w-24 text-center">
          Python
        </span>
        <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
        <span className="text-sm font-bold text-indigo-600 dark:text-indigo-400 w-24 text-center">
          TVMValue
        </span>
        <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
        <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-24 text-center">
          C++
        </span>
      </div>

      <div className="space-y-2">
        {mappings.map((m, i) => (
          <div
            key={m.python}
            onClick={() => setActive(active === i ? null : i)}
            className={`grid grid-cols-[1fr_auto_1fr_auto_1fr] items-center gap-2 p-3 rounded-lg cursor-pointer transition-all ${
              active === i
                ? "bg-white dark:bg-slate-800 shadow-md border border-indigo-300 dark:border-indigo-700"
                : "hover:bg-white/50 dark:hover:bg-slate-800/50"
            }`}
          >
            <div className={`text-center px-3 py-1 rounded bg-gradient-to-r ${m.color} text-white text-sm font-mono font-bold`}>
              {m.python}
            </div>
            <span className="text-slate-400">→</span>
            <div className="text-center px-3 py-1 rounded bg-slate-200 dark:bg-slate-700 text-sm font-mono text-slate-700 dark:text-slate-300">
              {m.tvm}
            </div>
            <span className="text-slate-400">→</span>
            <div className="text-center px-3 py-1 rounded bg-slate-800 dark:bg-slate-900 text-sm font-mono text-green-400">
              {m.cpp}
            </div>
          </div>
        ))}
      </div>

      {active !== null && (
        <div className="mt-4 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
          <p className="text-sm text-indigo-700 dark:text-indigo-300 font-mono">
            Python <strong>{mappings[active].python}</strong> 通过 FFI 层打包为{" "}
            <strong>{mappings[active].tvm}</strong>，C++ 端解包为{" "}
            <strong>{mappings[active].cpp}</strong>
          </p>
        </div>
      )}
    </div>
  );
}
