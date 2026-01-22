"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Tip {
    title: string;
    content: string;
}

interface Quiz {
    question: string;
    answer: boolean;
    explanation: string;
}

interface SectionContent {
    tips: Tip[];
    quizzes: Quiz[];
}

const DEFAULT_CONTENT: SectionContent = {
    tips: [
        { title: "PyTorch 哲学", content: "Explicit is better than implicit. PyTorch 倾向于让用户显式地控制 Tensor 的行为。" },
        { title: "学习建议", content: "多动手写代码，特别是 Tensor 的维度变换，光看是学不会的。" }
    ],
    quizzes: [
        { question: "PyTorch 是动态图框架吗？", answer: true, explanation: "是的，PyTorch 采用 Eager Execution 模式，代码像 Python 一样逐行执行。" }
    ]
};

// Map URL hash (section IDs) to content
// IDs come from the headings in markdown, e.g. "chapter-1-tensor" -> "1.1 什么是 Tensor？"
const CONTENT_DB: Record<string, SectionContent> = {
    "chapter-0": {
        tips: [
            { title: "环境配置", content: "Conda 是管理 Python 环境的神器，强烈建议为每个项目创建独立的 environment。" },
            { title: "CUDA 版本", content: "安装 PyTorch 时，CUDA 版本必须小于等于你显卡驱动支持的最高版本 (nvidia-smi)。" }
        ],
        quizzes: [
            { question: "Mac M1/M2 可以加速 PyTorch 吗？", answer: true, explanation: "可以！使用 MPS (Metal Performance Shaders) 后端即可加速。" }
        ]
    },
    "chapter-1": {
        tips: [
            { title: "View vs Reshape", content: "tensor.view() 要求内存连续，而 reshape() 则没有此限制。不确定时用 reshape() 更安全。" },
            { title: "广播机制", content: "维度为 1 的轴会自动扩展。小心隐式广播导致的维度错误！" },
            { title: "In-place 操作", content: "像 x.add_() 这样带下划线的方法会直接修改原数据，慎用！Autograd 可能会报错。" }
        ],
        quizzes: [
            { question: "tensor.view() 会发生内存拷贝吗？", answer: false, explanation: "通常不会。它是原存储的'视图'。除非数据不连续强制 contiguous()。" },
            { question: "x * y 是矩阵乘法吗？", answer: false, explanation: "不是！* 是元素级乘法 (Hadamard product)。矩阵乘法用 @ 或 torch.matmul。" }
        ]
    },
    "chapter-2": {
        tips: [
            { title: "梯度累加", content: "默认情况下 .backward() 会累加梯度。常用于变相增大 Batch Size。" },
            { title: "叶子节点", content: "只有 requires_grad=True 的叶子节点 (Leaf Node) 才会保留 .grad 属性。" }
        ],
        quizzes: [
            { question: "optimizer.step() 会清零梯度吗？", answer: false, explanation: "不会！必须手动调用 optimizer.zero_grad()。" },
            { question: "推理时应该用 no_grad 吗？", answer: true, explanation: "是的，这能显著减少显存占用并加速计算。" }
        ]
    },
    "chapter-3": {
        tips: [
            { title: "Module 模式", content: "记得调用 model.eval()！不然 Dropout 和 BatchNorm 会继续更新状态，导致推理结果错误。" },
            { title: "Shape Mismatch", content: "Linear 层的输入特征数必须精确匹配。不知多少层合适？先 print(x.shape) 看看。" }
        ],
        quizzes: [
            { question: "nn.ReLU() 有需要学习的参数吗？", answer: false, explanation: "没有。激活函数通常是无参的。" },
            { question: "forward() 函数能直接调用吗？", answer: false, explanation: "永远不要直接调用 model.forward(x)，请使用 model(x) 以确保护钩子 (Hooks) 正常工作。" }
        ]
    },
    "chapter-4": {
        tips: [
            { title: "Num Workers", content: "Windows 上多进程 DataLoader 经常报错？先把 num_workers 设为 0 试试。" },
            { title: "Collate Fn", content: "处理变长文本或特殊数据结构时，必须重写 collate_fn。" }
        ],
        quizzes: [
            { question: "Dataset 必须把所有图片读到内存吗？", answer: false, explanation: "不需要。通常只存储路径，在 __getitem__ 时才实时读取。" }
        ]
    },
    "chapter-5": {
        tips: [
            { title: "Adam vs SGD", content: "Adam 收敛快但可能掉入局部最优；SGD+Momentum 收敛慢但泛化通常更好。" },
            { title: "NaN Loss", content: "Loss 变成 NaN 了？检查一下是否忘记 zero_grad，或者是学习率太大爆炸了。" }
        ],
        quizzes: [
            { question: "CrossEntropyLoss 需要先手动 Softmax 吗？", answer: false, explanation: "不需要！它内部集成了 LogSoftmax，直接传 Logits 即可。" }
        ]
    }
};

interface ModuleRightSidebarProps {
    currentSection?: string;
}

export function ModuleRightSidebar({ currentSection = "" }: ModuleRightSidebarProps) {
    const [content, setContent] = useState<SectionContent>(DEFAULT_CONTENT);
    const [tipIndex, setTipIndex] = useState(0);
    const [quizIndex, setQuizIndex] = useState(0);
    const [showAnswer, setShowAnswer] = useState<boolean | null>(null);
    const [mounted, setMounted] = useState(false);

    // Detect context based on active ID
    useEffect(() => {
        // Simple matching logic: find the first key that is a substring of currentSection
        // e.g. "chapter-2-autograd" matches "chapter-2"
        const matchedKey = Object.keys(CONTENT_DB).find(key => currentSection.includes(key));

        if (matchedKey) {
            setContent(CONTENT_DB[matchedKey]);
            // Reset indices when chapter changes
            setTipIndex(0);
            setQuizIndex(0);
            setShowAnswer(null);
        }
    }, [currentSection]);

    useEffect(() => {
        setMounted(true);
        const timer = setInterval(() => {
            setTipIndex(i => (i + 1) % content.tips.length);
        }, 10000); // Rotate tips every 10s
        return () => clearInterval(timer);
    }, [content.tips.length]);

    if (!mounted) return null;

    const currentTip = content.tips[tipIndex % content.tips.length];
    const currentQuiz = content.quizzes[quizIndex % content.quizzes.length];

    return (
        <aside className="fixed w-64 space-y-6 pl-4 pt-4">
            {/* 1. Learning Streaks / Status */}
            <div className="bg-bg-elevated/80 backdrop-blur border border-border-subtle rounded-xl p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-text-secondary uppercase tracking-wider">当前状态</span>
                    <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                    </span>
                </div>
                <div className="flex items-center gap-3">
                    <div className="text-2xl font-black text-text-primary">Learning</div>
                    <div className="flex flex-col">
                        <div className="text-xs text-text-tertiary">
                            专注模式开启
                        </div>
                        {currentSection && (
                            <div className="text-[10px] text-accent-primary font-mono truncate w-28">
                                #{currentSection}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* 2. Context-Aware Tips */}
            <div className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 border border-indigo-100 dark:border-indigo-800 rounded-xl p-4 shadow-sm relative overflow-hidden group min-h-[140px]">
                <div className="absolute -right-4 -top-4 w-16 h-16 bg-indigo-200/30 rounded-full blur-xl group-hover:scale-150 transition-transform duration-700" />

                <h4 className="text-xs font-bold text-indigo-600 dark:text-indigo-400 mb-2 flex items-center gap-2">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    {currentTip.title}
                </h4>
                <AnimatePresence mode='wait'>
                    <motion.p
                        key={currentTip.content}
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        className="text-sm text-text-secondary leading-relaxed"
                    >
                        {currentTip.content}
                    </motion.p>
                </AnimatePresence>

                {content.tips.length > 1 && (
                    <div className="absolute bottom-4 left-4 flex gap-1">
                        {content.tips.map((_, i) => (
                            <div key={i} className={`h-1 rounded-full transition-all duration-300 ${i === tipIndex % content.tips.length ? 'w-4 bg-indigo-500' : 'w-1 bg-indigo-200'}`} />
                        ))}
                    </div>
                )}
            </div>

            {/* 3. Context-Aware Mini Quiz */}
            <div className="bg-bg-elevated/80 backdrop-blur border border-border-subtle rounded-xl p-4 shadow-sm">
                <h4 className="text-xs font-bold text-text-secondary uppercase tracking-wider mb-3">
                    Daily Quiz
                </h4>

                <AnimatePresence mode="wait">
                    <motion.p
                        key={currentQuiz.question}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-sm font-medium text-text-primary mb-4"
                    >
                        {currentQuiz.question}
                    </motion.p>
                </AnimatePresence>

                {showAnswer === null ? (
                    <div className="flex gap-2">
                        <button
                            onClick={() => setShowAnswer(true)}
                            className="flex-1 py-1.5 px-3 bg-green-50 hover:bg-green-100 text-green-700 text-xs rounded-lg border border-green-200 transition-colors"
                        >
                            Yes
                        </button>
                        <button
                            onClick={() => setShowAnswer(false)}
                            className="flex-1 py-1.5 px-3 bg-red-50 hover:bg-red-100 text-red-700 text-xs rounded-lg border border-red-200 transition-colors"
                        >
                            No
                        </button>
                    </div>
                ) : (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className={`rounded-lg p-3 text-xs ${showAnswer === currentQuiz.answer
                                ? 'bg-green-50 text-green-800 border border-green-200'
                                : 'bg-red-50 text-red-800 border border-red-200'
                            }`}
                    >
                        <div className="font-bold mb-1">
                            {showAnswer === currentQuiz.answer ? "🎉 Correct!" : "❌ Oops!"}
                        </div>
                        {currentQuiz.explanation}

                        <button
                            onClick={() => {
                                setShowAnswer(null);
                                setQuizIndex(i => (i + 1) % content.quizzes.length);
                            }}
                            className="mt-2 w-full py-1 bg-white/50 hover:bg-white/80 rounded text-center"
                        >
                            Next Question →
                        </button>
                    </motion.div>
                )}
            </div>

            <div className="text-[10px] text-text-tertiary text-center">
                Content Context: {content === DEFAULT_CONTENT ? "General" : "Matched"}
            </div>
        </aside>
    );
}
