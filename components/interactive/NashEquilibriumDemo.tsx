"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function NashEquilibriumDemo() {
    // 囚徒困境收益矩阵
    const payoffMatrix = {
        CC: [3, 3], // 双方合作
        CD: [0, 5], // 我合作，对方背叛
        DC: [5, 0], // 我背叛，对方合作
        DD: [1, 1], // 双方背叛（Nash均衡）
    };

    const [player1Choice, setPlayer1Choice] = useState<"C" | "D" | null>(null);
    const [player2Choice, setPlayer2Choice] = useState<"C" | "D" | null>(null);
    const [showResult, setShowResult] = useState(false);
    const [history, setHistory] = useState<Array<{ p1: string, p2: string, payoff: number[] }>>([]);

    const getPayoff = (p1: "C" | "D", p2: "C" | "D") => {
        return payoffMatrix[`${p1}${p2}` as keyof typeof payoffMatrix];
    };

    const playRound = () => {
        if (player1Choice && player2Choice) {
            const payoff = getPayoff(player1Choice, player2Choice);
            setHistory([...history, {
                p1: player1Choice,
                p2: player2Choice,
                payoff
            }]);
            setShowResult(true);
        }
    };

    const reset = () => {
        setPlayer1Choice(null);
        setPlayer2Choice(null);
        setShowResult(false);
    };

    const resetAll = () => {
        reset();
        setHistory([]);
    };

    const isNashEquilibrium = (p1: "C" | "D", p2: "C" | "D") => {
        return p1 === "D" && p2 === "D";
    };

    const totalPayoffs = history.reduce((acc, h) => ({
        p1: acc.p1 + h.payoff[0],
        p2: acc.p2 + h.payoff[1]
    }), { p1: 0, p2: 0 });

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-slate-900 dark:to-rose-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Nash 均衡（交互式博弈）
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    玩囚徒困境，体验Nash均衡
                </p>
            </div>

            {/* 收益矩阵 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">收益矩阵</h4>
                <table className="w-full text-sm border-collapse">
                    <thead>
                        <tr>
                            <th className="border-2 border-slate-300 dark:border-slate-600 p-3"></th>
                            <th className="border-2 border-slate-300 dark:border-slate-600 p-3 bg-blue-50 dark:bg-blue-900/20">
                                对方合作 (C)
                            </th>
                            <th className="border-2 border-slate-300 dark:border-slate-600 p-3 bg-blue-50 dark:bg-blue-900/20">
                                对方背叛 (D)
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td className="border-2 border-slate-300 dark:border-slate-600 p-3 bg-rose-50 dark:bg-rose-900/20 font-bold">
                                我合作 (C)
                            </td>
                            <td className="border-2 border-slate-300 dark:border-slate-600 p-3 text-center">
                                <div className="text-lg font-bold">(3, 3)</div>
                                <div className="text-xs text-slate-500">双赢</div>
                            </td>
                            <td className="border-2 border-slate-300 dark:border-slate-600 p-3 text-center">
                                <div className="text-lg font-bold">(0, 5)</div>
                                <div className="text-xs text-slate-500">我被背叛</div>
                            </td>
                        </tr>
                        <tr>
                            <td className="border-2 border-slate-300 dark:border-slate-600 p-3 bg-rose-50 dark:bg-rose-900/20 font-bold">
                                我背叛 (D)
                            </td>
                            <td className="border-2 border-slate-300 dark:border-slate-600 p-3 text-center">
                                <div className="text-lg font-bold">(5, 0)</div>
                                <div className="text-xs text-slate-500">我背叛对方</div>
                            </td>
                            <td className="border-2 border-slate-300 dark:border-slate-600 p-3 text-center bg-green-100 dark:bg-green-900/30">
                                <div className="text-lg font-bold">(1, 1)</div>
                                <div className="text-xs text-green-700 dark:text-green-400 font-bold">
                                    Nash 均衡 ⭐
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>

            {/* 交互博弈 */}
            {!showResult ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                        <h4 className="text-lg font-bold mb-4 text-center">👤 你的选择</h4>
                        <div className="space-y-3">
                            <motion.button
                                onClick={() => setPlayer1Choice("C")}
                                className={`w-full p-4 rounded-lg font-bold ${player1Choice === "C"
                                        ? "bg-blue-600 text-white"
                                        : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                                    }`}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                🤝 合作 (Cooperate)
                            </motion.button>
                            <motion.button
                                onClick={() => setPlayer1Choice("D")}
                                className={`w-full p-4 rounded-lg font-bold ${player1Choice === "D"
                                        ? "bg-red-600 text-white"
                                        : "bg-red-100 text-red-700 hover:bg-red-200"
                                    }`}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                ⚔️ 背叛 (Defect)
                            </motion.button>
                        </div>
                    </div>

                    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                        <h4 className="text-lg font-bold mb-4 text-center">🤖 对手选择</h4>
                        <div className="space-y-3">
                            <motion.button
                                onClick={() => setPlayer2Choice("C")}
                                className={`w-full p-4 rounded-lg font-bold ${player2Choice === "C"
                                        ? "bg-blue-600 text-white"
                                        : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                                    }`}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                🤝 合作 (Cooperate)
                            </motion.button>
                            <motion.button
                                onClick={() => setPlayer2Choice("D")}
                                className={`w-full p-4 rounded-lg font-bold ${player2Choice === "D"
                                        ? "bg-red-600 text-white"
                                        : "bg-red-100 text-red-700 hover:bg-red-200"
                                    }`}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                ⚔️ 背叛 (Defect)
                            </motion.button>
                        </div>
                    </div>
                </div>
            ) : (
                <motion.div
                    className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg mb-6"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                >
                    <h4 className="text-2xl font-bold mb-4 text-center">本轮结果</h4>
                    <div className="grid grid-cols-2 gap-6 mb-6">
                        <div className="text-center">
                            <div className="text-6xl mb-2">
                                {player1Choice === "C" ? "🤝" : "⚔️"}
                            </div>
                            <div className="text-xl font-bold">你</div>
                            <div className="text-3xl font-bold text-blue-600 mt-2">
                                {player1Choice && player2Choice ? getPayoff(player1Choice, player2Choice)[0] : 0}
                            </div>
                        </div>
                        <div className="text-center">
                            <div className="text-6xl mb-2">
                                {player2Choice === "C" ? "🤝" : "⚔️"}
                            </div>
                            <div className="text-xl font-bold">对手</div>
                            <div className="text-3xl font-bold text-rose-600 mt-2">
                                {player1Choice && player2Choice ? getPayoff(player1Choice, player2Choice)[1] : 0}
                            </div>
                        </div>
                    </div>
                    {player1Choice && player2Choice && isNashEquilibrium(player1Choice, player2Choice) && (
                        <div className="text-center p-4 bg-green-100 dark:bg-green-900/30 rounded-lg">
                            <div className="text-xl font-bold text-green-700 dark:text-green-400">
                                ⭐ 这是Nash均衡！⭐
                            </div>
                            <div className="text-sm mt-2">
                                双方都没有动机单方面改变策略
                            </div>
                        </div>
                    )}
                </motion.div>
            )}

            {/* 控制按钮 */}
            <div className="flex justify-center gap-4 mb-6">
                {!showResult ? (
                    <button
                        onClick={playRound}
                        disabled={!player1Choice || !player2Choice}
                        className="px-8 py-3 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-300 text-white font-bold text-lg transition-colors"
                    >
                        🎲 揭晓结果
                    </button>
                ) : (
                    <>
                        <button
                            onClick={reset}
                            className="px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-bold transition-colors"
                        >
                            ▶ 下一轮
                        </button>
                        <button
                            onClick={resetAll}
                            className="px-6 py-3 rounded-lg bg-slate-600 hover:bg-slate-700 text-white font-bold transition-colors"
                        >
                            🔄 重新开始
                        </button>
                    </>
                )}
            </div>

            {/* 历史记录 */}
            {history.length > 0 && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4">
                        游戏历史（共 {history.length} 轮）
                    </h4>
                    <div className="grid grid-cols-3 gap-4 mb-4">
                        <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <div className="text-sm text-slate-600 dark:text-slate-400">你的总分</div>
                            <div className="text-3xl font-bold text-blue-600">{totalPayoffs.p1}</div>
                        </div>
                        <div className="text-center p-4 bg-slate-50 dark:bg-slate-700 rounded">
                            <div className="text-sm text-slate-600 dark:text-slate-400">平均每轮</div>
                            <div className="text-2xl font-bold">
                                {(totalPayoffs.p1 / history.length).toFixed(1)}
                            </div>
                        </div>
                        <div className="text-center p-4 bg-rose-50 dark:bg-rose-900/20 rounded">
                            <div className="text-sm text-slate-600 dark:text-slate-400">对手总分</div>
                            <div className="text-3xl font-bold text-rose-600">{totalPayoffs.p2}</div>
                        </div>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                        {history.map((h, i) => (
                            <div
                                key={i}
                                className={`px-3 py-2 rounded text-sm ${isNashEquilibrium(h.p1 as "C" | "D", h.p2 as "C" | "D")
                                        ? "bg-green-100 dark:bg-green-900/30 border-2 border-green-500"
                                        : "bg-slate-100 dark:bg-slate-700"
                                    }`}
                            >
                                {h.p1 === "C" ? "🤝" : "⚔️"} vs {h.p2 === "C" ? "🤝" : "⚔️"}: {h.payoff[0]}-{h.payoff[1]}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 Nash均衡(D,D)是理性选择，但不是Pareto最优（双方都能通过合作获得更好结果）
            </div>
        </div>
    );
}
