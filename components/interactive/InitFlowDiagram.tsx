"use client";

export function InitFlowDiagram() {
  const steps = [
    {
      title: "环境初始化",
      color: "from-blue-500 to-blue-600",
      items: ["加载 runtime 模块", "注册 Target 后端", "初始化内存池", "设置日志级别"],
    },
    {
      title: "编译准备",
      color: "from-indigo-500 to-indigo-600",
      items: ["导入模型 (ONNX/PyTorch)", "构建 Relay IR", "应用优化 Pass", "生成执行图"],
    },
    {
      title: "运行验证",
      color: "from-purple-500 to-purple-600",
      items: ["分配输入张量", "执行推理", "验证输出形状", "性能基准测试"],
    },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        TVM 初始化流程
      </h3>
      <div className="flex flex-col gap-4">
        {steps.map((step, i) => (
          <div key={i} className="flex items-start gap-4">
            <div className={`bg-gradient-to-br ${step.color} text-white w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm shrink-0 shadow-lg mt-1`}>
              {i + 1}
            </div>
            <div className="flex-1 bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border border-indigo-100 dark:border-indigo-900">
              <div className="font-bold text-slate-800 dark:text-slate-100 mb-2">{step.title}</div>
              <div className="grid grid-cols-2 gap-2">
                {step.items.map((item, j) => (
                  <div key={j} className="text-xs text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-700/50 rounded-lg px-3 py-2">
                    {item}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>三步初始化：</strong>环境搭建 → 编译模型 → 验证正确性，确保从加载到执行的完整链路。
      </div>
    </div>
  );
}
