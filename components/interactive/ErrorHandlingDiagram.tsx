"use client";

export function ErrorHandlingDiagram() {
  const stages = [
    {
      title: "错误类型",
      color: "from-red-500 to-red-600",
      items: [
        "类型错误 (TypeMismatch)",
        "形状错误 (ShapeError)",
        "未注册算子 (OpNotRegistered)",
        "调度冲突 (ScheduleError)",
      ],
    },
    {
      title: "诊断",
      color: "from-orange-500 to-orange-600",
      items: [
        "错误位置追踪",
        "IR 层级回溯",
        "类型推导日志",
        "Pass 名称标注",
      ],
    },
    {
      title: "修复",
      color: "from-green-500 to-green-600",
      items: [
        "修正输入形状",
        "添加缺失算子注册",
        "调整调度策略",
        "更新类型注解",
      ],
    },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        错误处理流程
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {stages.map((stage, i) => (
          <div key={i} className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-indigo-100 dark:border-indigo-900 overflow-hidden">
            <div className={`bg-gradient-to-r ${stage.title === "错误类型" ? "from-red-500 to-red-600" : stage.title === "诊断" ? "from-orange-500 to-orange-600" : "from-green-500 to-green-600"} text-white px-4 py-3 font-bold text-center`}>
              {stage.title}
            </div>
            <div className="p-4 space-y-2">
              {stage.items.map((item, j) => (
                <div key={j} className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
                  {item}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>三阶段：</strong>识别错误类型 → 诊断定位原因 → 应用修复策略。
      </div>
    </div>
  );
}
