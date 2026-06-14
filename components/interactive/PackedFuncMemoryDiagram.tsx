"use client";

import { useState } from "react";

export function PackedFuncMemoryDiagram() {
  const [highlight, setHighlight] = useState<string | null>(null);

  const stackItems = [
    { name: "TVMValue[0]", offset: "0x00", content: "v_int64 = 42", type: "kInt", color: "from-violet-500 to-purple-600" },
    { name: "TVMValue[1]", offset: "0x08", content: "v_float64 = 3.14", type: "kFloat", color: "from-indigo-500 to-blue-600" },
    { name: "TVMValue[2]", offset: "0x10", content: "v_handle = 0x7f...", type: "kHandle", color: "from-blue-500 to-cyan-600" },
    { name: "TVMValue[3]", offset: "0x18", content: 'v_str = "hello"', type: "kStr", color: "from-cyan-500 to-teal-600" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        PackedFunc 内存布局
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
            TVMArgs 栈布局
          </h4>
          <div className="relative">
            {stackItems.map((item, i) => (
              <div
                key={item.name}
                className={`flex items-center gap-3 p-3 mb-1 rounded-lg cursor-pointer transition-all border ${
                  highlight === item.name
                    ? "bg-white dark:bg-slate-800 shadow-lg border-indigo-400"
                    : "bg-white/50 dark:bg-slate-800/50 border-transparent hover:border-slate-300 dark:hover:border-slate-600"
                }`}
                onMouseEnter={() => setHighlight(item.name)}
                onMouseLeave={() => setHighlight(null)}
              >
                <div className="text-xs font-mono text-slate-400 w-12">
                  {item.offset}
                </div>
                <div className={`px-3 py-1 rounded bg-gradient-to-r ${item.color} text-white text-xs font-bold w-28 text-center`}>
                  {item.name}
                </div>
                <div className="flex-1">
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-300">
                    {item.content}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    type_code = {item.type}
                  </div>
                </div>
              </div>
            ))}
            <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-slate-200 dark:bg-slate-700 -z-10" />
          </div>
        </div>

        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
            TVMValue 联合体
          </h4>
          <pre className="bg-slate-900 text-green-400 p-4 rounded-xl text-xs font-mono">
{`union TVMValue {
  int64_t   v_int64;    // 8 bytes
  double    v_float64;  // 8 bytes
  void*     v_handle;   // 8 bytes
  const char* v_str;    // 8 bytes (ptr)
};

// 每个 TVMValue 都是 8 bytes
// 通过 type_code 区分实际类型
struct TVMArgs {
  TVMValue* values;     // 指向栈
  int*      type_codes; // 类型标记
  int       num_args;   // 参数个数
};`}
          </pre>

          {highlight && (
            <div className="mt-3 p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
              <p className="text-xs text-indigo-700 dark:text-indigo-300">
                <strong>{highlight}</strong>: 8 字节的 TVMValue 联合体，
                通过 type_code 指示当前存储的实际类型
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 flex gap-4 justify-center">
        <div className="px-4 py-2 bg-slate-800 dark:bg-slate-900 rounded-lg text-xs font-mono text-amber-400">
          总大小: num_args × 16 bytes (value + type_code)
        </div>
      </div>
    </div>
  );
}
