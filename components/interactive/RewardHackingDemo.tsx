"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type HackingType = "length" | "repetition" | "format" | "normal";

export function RewardHackingDemo() {
    const [hackingType, setHackingType] = useState<HackingType>("normal");

    const examples = {
        normal: {
            title: "正常回复",
            text: "量子计算是利用量子力学原理进行信息处理的计算方式，具有指数级加速潜力。",
            rmScore: 7.5,
            quality: 85,
            color: "green",
            analysis: {
                length: "适中（23字）",
                uniqueness: "98%",
                coherence: "高",
                hacking: "无"
            }
        },
        length: {
            title: "长度Hacking",
            text: "量子计算是一种非常重要的计算方式。量子计算利用了量子力学的原理。量子计算可以进行非常快速的计算。量子计算是未来的发展方向。量子计算有很多应用场景。量子计算需要特殊的硬件支持。量子计算是一个复杂的领域。量子计算正在快速发展...",
            rmScore: 8.2,
            quality: 45,
            color: "orange",
            analysis: {
                length: "过长（120字）",
                uniqueness: "65%",
                coherence: "低（冗余）",
                hacking: "通过堆砌无意义内容获取高分"
            }
        },
        repetition: {
            title: "重复Hacking",
            text: "量子计算量子计算量子计算是一种计算方式是一种计算方式，量子量子量子力学原理原理原理...",
            rmScore: 6.8,
            quality: 25,
            color: "red",
            analysis: {
                length: "正常",
                uniqueness: "35%",
                coherence: "极低（大量重复）",
                hacking: "重复关键词以增加确定性"
            }
        },
        format: {
            title: "格式Hacking",
            text: "**量子计算**：\n1. 定义：\n   - 利用量子原理\n   - 超快速计算\n2. 特点：\n   ✓ 并行\n   ✓ 高效\n3. 应用：...",
            rmScore: 8.5,
            quality: 60,
            color: "yellow",
            analysis: {
                length: "正常",
                uniqueness: "85%",
                coherence: "中（过度格式化）",
                hacking: "利用奖励模型对列表格式的偏好"
            }
        }
    };

    const current = examples[hackingType];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-red-50 to-orange-50 dark:from-slate-900 dark:to-red-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    奖励 Hacking 演示
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    策略如何"欺骗"奖励模型获取高分
                </p>
            </div>

            {/* 类型选择 */}
            <div className="grid grid-cols-4 gap-3 mb-6">
                {(Object.keys(examples) as HackingType[]).map((type) => (
                    <button
                        key={type}
                        onClick={() => setHackingType(type)}
                        className={`p-3 rounded-lg border-2 transition ${hackingType === type
                                ? `border-${examples[type].color}-500 bg-${examples[type].color}-50 dark:bg-${examples[type].color}-900/20`
                                : "border-gray-200 dark:border-gray-700 bg-white dark:bg-slate-800"
                            }`}
                    >
                        <div className={`font-semibold text-sm ${hackingType === type
                                ? `text-${examples[type].color}-700 dark:text-${examples[type].color}-400`
                                : "text-slate-600 dark:text-slate-400"
                            }`}>
                            {examples[type].title}
                        </div>
                    </button>
                ))}
            </div>

            {/* 示例展示 */}
            <motion.div
                key={hackingType}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6"
            >
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                        {current.title}
                    </h4>
                    <div className={`px-3 py-1 rounded-full text-sm font-semibold bg-${current.color}-100 text-${current.color}-700 dark:bg-${current.color}-900/30 dark:text-${current.color}-400`}>
                        {hackingType === "normal" ? "✅ 正常" : "⚠️ Hacking"}
                    </div>
                </div>

                {/* 生成的文本 */}
                <div className={`p-4 rounded-lg border-2 mb-4 ${hackingType === "normal"
                        ? "border-green-300 dark:border-green-700 bg-green-50 dark:bg-green-900/10"
                        : "border-orange-300 dark:border-orange-700 bg-orange-50 dark:bg-orange-900/10"
                    }`}>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">生成的文本：</div>
                    <div className="text-slate-800 dark:text-slate-100 leading-relaxed">
                        {current.text}
                    </div>
                </div>

                {/* 指标对比 */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-600 dark:text-slate-400">奖励模型分数</span>
                            <span className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                                {current.rmScore.toFixed(1)}
                            </span>
                        </div>
                        <div className="h-3 bg-purple-200 dark:bg-purple-900 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-purple-600"
                                initial={{ width: 0 }}
                                animate={{ width: `${(current.rmScore / 10) * 100}%` }}
                                transition={{ duration: 0.5 }}
                            />
                        </div>
                    </div>

                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-600 dark:text-slate-400">实际质量</span>
                            <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                {current.quality}%
                            </span>
                        </div>
                        <div className="h-3 bg-blue-200 dark:bg-blue-900 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-blue-600"
                                initial={{ width: 0 }}
                                animate={{ width: `${current.quality}%` }}
                                transition={{ duration: 0.5 }}
                            />
                        </div>
                    </div>
                </div>

                {/* 差距警告 */}
                {hackingType !== "normal" && (
                    <div className="bg-red-100 dark:bg-red-900/30 border-2 border-red-300 dark:border-red-700 rounded-lg p-3 mb-4">
                        <div className="flex items-center gap-2">
                            <span className="text-2xl">⚠️</span>
                            <div>
                                <div className="font-semibold text-red-700 dark:text-red-400">
                                    奖励-质量差距：{(current.rmScore * 10 - current.quality).toFixed(0)}分
                                </div>
                                <div className="text-sm text-red-600 dark:text-red-300">
                                    模型获得高奖励但实际质量低
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* 分析 */}
                <div className="grid grid-cols-4 gap-3 text-sm">
                    {Object.entries(current.analysis).map(([key, value]) => (
                        <div key={key} className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
                            <div className="text-slate-600 dark:text-slate-400 mb-1 capitalize">{key}</div>
                            <div className="font-semibold text-slate-800 dark:text-slate-100">{value}</div>
                        </div>
                    ))}
                </div>
            </motion.div>

            {/* 防御方法 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">防御方法</h4>

                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300 dark:border-green-700">
                        <h5 className="font-semibold text-green-700 dark:text-green-400 mb-3 flex items-center gap-2">
                            <span>🛡️</span> 训练时防御
                        </h5>
                        <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                            <li className="flex items-start gap-2">
                                <span className="text-green-600">•</span>
                                <div>
                                    <strong>KL惩罚</strong>：限制偏离参考模型
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-green-600">•</span>
                                <div>
                                    <strong>长度归一化</strong>：奖励除以长度
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-green-600">•</span>
                                <div>
                                    <strong>重复惩罚</strong>：检测n-gram重复
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-green-600">•</span>
                                <div>
                                    <strong>奖励裁剪</strong>：限制奖励范围
                                </div>
                            </li>
                        </ul>
                    </div>

                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-2 border-blue-300 dark:border-blue-700">
                        <h5 className="font-semibold text-blue-700 dark:text-blue-400 mb-3 flex items-center gap-2">
                            <span>🎯</span> 奖励模型改进
                        </h5>
                        <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                            <li className="flex items-start gap-2">
                                <span className="text-blue-600">•</span>
                                <div>
                                    <strong>多样化数据</strong>：覆盖各种模式
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-blue-600">•</span>
                                <div>
                                    <strong>集成模型</strong>：多个RM平均
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-blue-600">•</span>
                                <div>
                                    <strong>对抗训练</strong>：在Hacking样本上训练
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-blue-600">•</span>
                                <div>
                                    <strong>人工审核</strong>：定期检查生成质量
                                </div>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-orange-100 dark:bg-orange-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                ⚡ 奖励Hacking是RLHF的主要挑战，需要多种防御机制共同应对
            </div>
        </div>
    );
}
