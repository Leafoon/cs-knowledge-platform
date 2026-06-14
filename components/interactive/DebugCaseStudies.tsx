"use client";

import { useState } from "react";

const cases = [
    {
        type: "Shape 错误",
        icon: "📐",
        color: "from-red-500 to-pink-500",
        symptom: "matmul shape mismatch: (32,64) vs (128,64)",
        diagnosis: "矩阵维度不匹配：A的列数(64) ≠ B的行数(128)",
        fix: "检查输入张量shape，确保 A.shape[1] == B.shape[0]",
        code: "A: (32, 64)  B: (128, 64)\n修正: B 应为 (64, 128)",
    },
    {
        type: "Dtype 错误",
        icon: "🔢",
        color: "from-amber-500 to-orange-500",
        symptom: "Cannot add float32 and int32 tensors",
        diagnosis: "数据类型不匹配：混合了 float32 和 int32 操作",
        fix: "使用 .astype() 统一数据类型",
        code: "A: float32  B: int32\n修正: B = B.astype('float32')",
    },
    {
        type: "Op 错误",
        icon: "⚙️",
        color: "from-purple-500 to-indigo-500",
        symptom: "Operator conv2d not registered for target cuda",
        diagnosis: "算子未在目标设备上注册",
        fix: "检查算子是否支持目标后端，或使用替代算子",
        code: '# 检查: tvm.ir_pass.check_legal(target="cuda")\n# 修正: 使用 relay.nn.conv2d 替代自定义算子',
    },
];

export function DebugCaseStudies() {
    const [selected, setSelected] = useState(0);
    const c = cases[selected];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">调试案例诊断</h3>

            <div className="flex gap-2 mb-6">
                {cases.map((cs, i) => (
                    <button
                        key={i}
                        onClick={() => setSelected(i)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
                            selected === i
                                ? `bg-gradient-to-r ${cs.color} text-white shadow-lg`
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        <span>{cs.icon}</span>
                        {cs.type}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-red-600 dark:text-red-400 mb-3">错误信息</h4>
                    <pre className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3 text-xs font-mono text-red-700 dark:text-red-300 mb-4">
                        {c.symptom}
                    </pre>
                    <h4 className="text-base font-bold text-amber-600 dark:text-amber-400 mb-2">诊断</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">{c.diagnosis}</p>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-emerald-600 dark:text-emerald-400 mb-3">修复方案</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">{c.fix}</p>
                    <pre className="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 text-xs font-mono text-emerald-700 dark:text-emerald-400">
                        {c.code}
                    </pre>
                </div>
            </div>
        </div>
    );
}
