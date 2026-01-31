"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function SelfVerificationLoop() {
    const [attempt, setAttempt] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    const maxAttempts = 4;

    const attempts = [
        {
            id: 0,
            solution: "等待开始...",
            verification: "",
            passed: false,
            confidence: 0
        },
        {
            id: 1,
            solution: "2x + 3 = 11\n直接得出 x = 4",
            verification: "检查：推理跳步，缺少中间计算。不正确。",
            passed: false,
            confidence: 0.3
        },
        {
            id: 2,
            solution: "2x + 3 = 11\n2x = 11 - 3 = 7\nx = 7/2 = 3.5",
            verification: "检查：计算错误，11-3=8而非7。不正确。",
            passed: false,
            confidence: 0.5
        },
        {
            id: 3,
            solution: "2x + 3 = 11\n2x = 11 - 3\n2x = 8\nx = 8/2\nx = 4",
            verification: "检查：\n1. 减法正确：11-3=8 ✓\n2. 除法正确：8/2=4 ✓\n3. 验证：2(4)+3=11 ✓\n通过！",
            passed: true,
            confidence: 0.95
        },
        {
            id: 4,
            solution: "2x + 3 = 11\n2x = 11 - 3\n2x = 8\nx = 8/2\nx = 4",
            verification: "检查：\n1. 减法正确：11-3=8 ✓\n2. 除法正确：8/2=4 ✓\n3. 验证：2(4)+3=11 ✓\n通过！",
            passed: true,
            confidence: 0.95
        }
    ];

    const currentAttempt = attempts[attempt];

    const handleRun = () => {
        if (isRunning) {
            setIsRunning(false);
        } else {
            setIsRunning(true);
            const interval = setInterval(() => {
                setAttempt(prev => {
                    if (prev >= maxAttempts) {
                        setIsRunning(false);
                        clearInterval(interval);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 2000);
        }
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-slate-900 dark:to-rose-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    自我验证循环
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    生成-验证-迭代，直到通过验证
                </p>
            </div>

            {/* 控制 */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex gap-3">
                    <button
                        onClick={handleRun}
                        className={`px-4 py-2 rounded-lg font-semibold transition ${isRunning ? "bg-orange-500 text-white" : "bg-rose-600 text-white hover:bg-rose-700"
                            }`}
                    >
                        {isRunning ? "⏸ 暂停" : "▶ 开始验证"}
                    </button>
                    <button
                        onClick={() => { setIsRunning(false); setAttempt(0); }}
                        className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition"
                    >
                        🔄 重置
                    </button>
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                    尝试: <strong className="text-rose-600 dark:text-rose-400">{attempt}/{maxAttempts}</strong>
                </div>
            </div>

            {/* 流程图 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">验证流程</h4>

                <div className="flex items-center justify-center gap-4">
                    {[
                        { step: "1. 生成", icon: "✏️", active: attempt > 0 },
                        { step: "→", icon: "", active: attempt > 0 },
                        { step: "2. 验证", icon: "🔍", active: attempt > 0 },
                        { step: "→", icon: "", active: true },
                        { step: currentAttempt.passed ? "3. ✅ 完成" : "3. 🔄 重试", icon: "", active: true }
                    ].map((item, idx) => (
                        <div key={idx} className="flex items-center gap-4">
                            {item.step !== "→" ? (
                                <div className={`p-4 rounded-xl border-2 transition ${item.active
                                        ? "border-rose-500 bg-rose-50 dark:bg-rose-900/20"
                                        : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
                                    }`}>
                                    <div className="text-2xl text-center mb-1">{item.icon}</div>
                                    <div className="text-sm font-semibold text-center text-slate-800 dark:text-slate-100">
                                        {item.step}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-3xl text-rose-500">→</div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* 当前尝试 */}
            {attempt > 0 && (
                <motion.div
                    key={attempt}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4 mb-6"
                >
                    {/* 生成的解 */}
                    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                        <div className="flex items-center justify-between mb-3">
                            <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                                生成的解答（尝试 {attempt}）
                            </h4>
                            <span className="text-sm text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 px-3 py-1 rounded-full">
                                LLM生成
                            </span>
                        </div>
                        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-300 dark:border-blue-700">
                            <pre className="text-sm text-slate-800 dark:text-slate-100 whitespace-pre-wrap font-mono">
                                {currentAttempt.solution}
                            </pre>
                        </div>
                    </div>

                    {/* 验证结果 */}
                    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                        <div className="flex items-center justify-between mb-3">
                            <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                                自我验证
                            </h4>
                            <span className={`text-sm px-3 py-1 rounded-full font-semibold ${currentAttempt.passed
                                    ? "text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20"
                                    : "text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20"
                                }`}>
                                {currentAttempt.passed ? "✅ 通过" : "❌ 未通过"}
                            </span>
                        </div>
                        <div className={`p-4 rounded-lg border ${currentAttempt.passed
                                ? "bg-green-50 dark:bg-green-900/20 border-green-500"
                                : "bg-red-50 dark:bg-red-900/20 border-red-500"
                            }`}>
                            <pre className="text-sm text-slate-800 dark:text-slate-100 whitespace-pre-wrap">
                                {currentAttempt.verification}
                            </pre>
                        </div>

                        {/* 置信度 */}
                        <div className="mt-4">
                            <div className="flex justify-between text-sm mb-2">
                                <span className="text-slate-600 dark:text-slate-400">置信度</span>
                                <span className="font-mono text-slate-800 dark:text-slate-100">
                                    {(currentAttempt.confidence * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                <motion.div
                                    className={`h-full ${currentAttempt.passed ? "bg-green-600" : "bg-red-600"
                                        }`}
                                    animate={{ width: `${currentAttempt.confidence * 100}%` }}
                                    transition={{ duration: 0.5 }}
                                />
                            </div>
                        </div>
                    </div>
                </motion.div>
            )}

            {/* 历史记录 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">尝试历史</h4>

                <div className="space-y-2">
                    {attempts.slice(1, attempt + 1).map((att) => (
                        <div
                            key={att.id}
                            className={`p-3 rounded-lg border ${att.passed
                                    ? "bg-green-50 dark:bg-green-900/20 border-green-500"
                                    : "bg-red-50 dark:bg-red-900/20 border-red-500"
                                }`}
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${att.passed ? "bg-green-600" : "bg-red-600"
                                        }`}>
                                        {att.id}
                                    </div>
                                    <div className="text-sm text-slate-700 dark:text-slate-300">
                                        {att.passed ? "验证通过" : "验证失败，重新生成"}
                                    </div>
                                </div>
                                <div className="text-sm text-slate-600 dark:text-slate-400">
                                    置信度: {(att.confidence * 100).toFixed(0)}%
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {currentAttempt.passed && (
                <div className="mt-6 bg-green-100 dark:bg-green-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                    🎉 <strong>验证成功！</strong> 经过{attempt}次尝试，找到了正确的解答
                </div>
            )}

            {!currentAttempt.passed && attempt > 0 && attempt < maxAttempts && (
                <div className="mt-6 bg-orange-100 dark:bg-orange-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                    🔄 验证失败，继续尝试...
                </div>
            )}
        </div>
    );
}
