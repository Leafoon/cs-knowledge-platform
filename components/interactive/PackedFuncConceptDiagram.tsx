"use client";

import { useState } from "react";

export function PackedFuncConceptDiagram() {
  const [callType, setCallType] = useState<"python" | "cpp" | "relay">("python");

  const examples = {
    python: {
      label: "Python 调用",
      code: `# 获取全局 PackedFunc
func = tvm.get_global_func("tvm.contrib.sort.argsort")

# 调用 - 参数自动打包
result = func(input_tensor, is_ascend=True)

# 返回值自动解包
print(result)  # NDArray`,
      color: "from-violet-500 to-purple-600",
    },
    cpp: {
      label: "C++ 调用",
      code: `// 获取全局函数
TVMFuncHandle func;
TVMFuncGetGlobal("tvm.contrib.sort.argsort", &func);

// 打包参数
TVMValue values[2];
int type_codes[2];
values[0].v_handle = (void*)input;
type_codes[0] = kArrayHandle;
values[1].v_int64 = 1;
type_codes[1] = kInt;

// 调用
TVMFuncCall(func, values, type_codes, 2, &ret_val, &ret_tcode);`,
      color: "from-blue-500 to-cyan-600",
    },
    relay: {
      label: "Relay IR 调用",
      code: `// Relay 表达式中引用 PackedFunc
%0 = fn (%x: Tensor[(3, 4), float32]) {
  %1 = call @argsort(%x, is_ascend=True)
  %1
}

// 编译时绑定到具体 PackedFunc
// 运行时通过函数指针调用`,
      color: "from-green-500 to-emerald-600",
    },
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        PackedFunc: 统一可调用接口
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
        PackedFunc 是 TVM 的核心抽象，提供跨语言的统一函数调用接口
      </p>

      <div className="flex items-center justify-center mb-6">
        <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-6 py-3 rounded-xl font-bold shadow-lg">
          PackedFunc
        </div>
      </div>

      <div className="flex gap-2 mb-4">
        {(["python", "cpp", "relay"] as const).map((key) => (
          <button
            key={key}
            onClick={() => setCallType(key)}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${
              callType === key
                ? `bg-gradient-to-r ${examples[key].color} text-white`
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700"
            }`}
          >
            {examples[key].label}
          </button>
        ))}
      </div>

      <div className="bg-slate-900 rounded-xl p-4">
        <pre className="text-green-400 text-sm font-mono overflow-x-auto">
          {examples[callType].code}
        </pre>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3">
        <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-lg font-bold text-indigo-600 dark:text-indigo-400">TVMValue</div>
          <div className="text-xs text-slate-500">统一参数容器</div>
        </div>
        <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-lg font-bold text-purple-600 dark:text-purple-400">TypeCode</div>
          <div className="text-xs text-slate-500">类型标记</div>
        </div>
        <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-lg font-bold text-blue-600 dark:text-blue-400">TVMArgs</div>
          <div className="text-xs text-slate-500">参数包</div>
        </div>
      </div>
    </div>
  );
}
