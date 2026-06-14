"use client";

export function TEtoTIRLowering() {
  const steps = [
    {
      step: "1",
      title: "Compute",
      desc: "定义计算语义",
      detail: "te.compute(shape, lambda)",
      color: "from-blue-500 to-blue-600",
    },
    {
      step: "2",
      title: "Schedule",
      desc: "应用调度变换",
      detail: "s.split / s.reorder / s.vectorize",
      color: "from-indigo-500 to-indigo-600",
    },
    {
      step: "3",
      title: "PrimFunc",
      desc: "生成 TIR 函数",
      detail: "tir.PrimFunc(params, body)",
      color: "from-purple-500 to-purple-600",
    },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        TE → TIR 降级
      </h3>
      <div className="flex flex-col gap-4">
        {steps.map((s, i) => (
          <div key={i} className="flex items-center gap-4">
            <div className={`bg-gradient-to-br ${s.color} text-white w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg shrink-0 shadow-lg`}>
              {s.step}
            </div>
            <div className="flex-1 bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border border-indigo-100 dark:border-indigo-900">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800 dark:text-slate-100">{s.title}</span>
                <code className="text-xs text-indigo-600 dark:text-indigo-400 font-mono">{s.detail}</code>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">{s.desc}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>三步降级：</strong>TE 定义计算 → Schedule 变换循环结构 → 生成 TIR PrimFunc。
      </div>
    </div>
  );
}
