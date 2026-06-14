"use client";

export function PackedFuncArchitecture() {
  const layers = [
    { name: "Python", desc: "用户调用 PackedFunc", detail: "func(arg1, arg2)", color: "from-blue-500 to-blue-600" },
    { name: "FFI", desc: "Foreign Function Interface", detail: "TVMValue + type_code", color: "from-indigo-500 to-indigo-600" },
    { name: "C++", desc: "PackedFunc 实现", detail: "TypedPackedFunc<F>", color: "from-purple-500 to-purple-600" },
    { name: "Device", desc: "硬件执行", detail: "kernel launch", color: "from-violet-500 to-violet-600" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        PackedFunc 调用链
      </h3>
      <div className="flex flex-col gap-0">
        {layers.map((layer, i) => (
          <div key={i}>
            <div className="flex items-center gap-4 p-4 bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-indigo-100 dark:border-indigo-900">
              <div className={`bg-gradient-to-br ${layer.color} text-white w-14 h-14 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 shadow-md`}>
                {layer.name}
              </div>
              <div className="flex-1">
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{layer.desc}</div>
                <code className="text-xs text-indigo-600 dark:text-indigo-400 font-mono">{layer.detail}</code>
              </div>
            </div>
            {i < layers.length - 1 && (
              <div className="flex justify-center py-1">
                <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>跨语言调用：</strong>PackedFunc 通过 FFI 桥接 Python 和 C++，实现零拷贝高效调用。
      </div>
    </div>
  );
}
