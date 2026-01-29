"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function MCReturnEstimation() {
    const [numEpisodes, setNumEpisodes] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [estimates, setEstimates] = useState<any[]>([]);
    const [currentEpisode, setCurrentEpisode] = useState<number[]>([]);

    const trueValue = 5.0; // 真实价值
    const gamma = 0.9;
    const maxEpisodes = 100;

    const generateEpisode = () => {
        // 模拟生成一个 episode 的奖励序列
        const length = Math.floor(Math.random() * 8) + 3; // 3-10 步
        const rewards = Array(length).fill(0).map(() =>
            Math.random() > 0.5 ? 1 : -1
        );
        return rewards;
    };

    const calculateReturn = (rewards: number[]) => {
        let G = 0;
        for (let i = rewards.length - 1; i >= 0; i--) {
            G = rewards[i] + gamma * G;
        }
        return G;
    };

    useEffect(() => {
        if (!isPlaying || numEpisodes >= maxEpisodes) {
            if (numEpisodes >= maxEpisodes) setIsPlaying(false);
            return;
        }

        const timer = setTimeout(() => {
            const episode = generateEpisode();
            const G = calculateReturn(episode);

            // 计算当前平均估计
            const currentMean = estimates.length > 0
                ? (estimates[estimates.length - 1].mean * estimates.length + G) / (estimates.length + 1)
                : G;

            setEstimates(prev => [
                ...prev,
                {
                    episode: numEpisodes + 1,
                    return: G,
                    mean: currentMean,
                    error: Math.abs(currentMean - trueValue),
                }
            ]);

            setCurrentEpisode(episode);
            setNumEpisodes(numEpisodes + 1);
        }, 300);

        return () => clearTimeout(timer);
    }, [isPlaying, numEpisodes, estimates]);

    const reset = () => {
        setNumEpisodes(0);
        setIsPlaying(false);
        setEstimates([]);
        setCurrentEpisode([]);
    };

    const currentEstimate = estimates.length > 0
        ? estimates[estimates.length - 1].mean
        : 0;

    const error = Math.abs(currentEstimate - trueValue);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    MC Return 估计过程
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    观察样本平均如何收敛到真实值
                </p>
            </div>

            {/* 控制面板 */}
            <div className="flex justify-center items-center gap-4 mb-6">
                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    disabled={numEpisodes >= maxEpisodes}
                    className="px-6 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-300 text-white font-semibold transition-colors"
                >
                    {isPlaying ? "⏸ 暂停" : "▶ 播放"}
                </button>
                <button
                    onClick={reset}
                    className="px-6 py-2 rounded-lg bg-slate-600 hover:bg-slate-700 text-white font-semibold transition-colors"
                >
                    🔄 重置
                </button>
            </div>

            {/* 统计面板 */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4 text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Episodes</div>
                    <div className="text-2xl font-bold text-emerald-600">{numEpisodes}</div>
                </div>
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4 text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">真实值</div>
                    <div className="text-2xl font-bold text-blue-600">{trueValue.toFixed(2)}</div>
                </div>
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4 text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">当前估计</div>
                    <div className="text-2xl font-bold text-teal-600">{currentEstimate.toFixed(2)}</div>
                </div>
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4 text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">误差</div>
                    <div className="text-2xl font-bold text-orange-600">{error.toFixed(3)}</div>
                </div>
            </div>

            {/* 当前 Episode 可视化 */}
            {currentEpisode.length > 0 && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                        当前 Episode（T = {currentEpisode.length}）
                    </h4>
                    <div className="flex items-center justify-center gap-2 flex-wrap">
                        {currentEpisode.map((reward, idx) => {
                            const cumulative = calculateReturn(currentEpisode.slice(idx));
                            return (
                                <motion.div
                                    key={idx}
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    className="flex flex-col items-center"
                                >
                                    <div
                                        className={`w-16 h-16 rounded-lg flex items-center justify-center font-bold text-white ${reward > 0 ? "bg-green-500" : "bg-red-500"
                                            }`}
                                    >
                                        {reward > 0 ? "+1" : "-1"}
                                    </div>
                                    <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                                        t={idx}
                                    </div>
                                    <div className="text-xs text-slate-500 dark:text-slate-400">
                                        G={cumulative.toFixed(1)}
                                    </div>
                                    {idx < currentEpisode.length - 1 && (
                                        <div className="text-lg text-slate-400">→</div>
                                    )}
                                </motion.div>
                            );
                        })}
                    </div>
                    <div className="mt-4 text-center">
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                            Return: G = R₁ + γR₂ + γ²R₃ + ... = {calculateReturn(currentEpisode).toFixed(2)}
                        </div>
                    </div>
                </div>
            )}

            {/* 收敛曲线 */}
            {estimates.length > 1 && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                        估计收敛曲线
                    </h4>
                    <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={estimates}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey="episode"
                                label={{ value: 'Episode 数量', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis
                                label={{ value: '价值估计', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip />
                            <Legend />
                            <Line
                                type="monotone"
                                dataKey="mean"
                                stroke="#10b981"
                                strokeWidth={3}
                                name="样本平均"
                                dot={false}
                            />
                            <Line
                                type="monotone"
                                dataKey={() => trueValue}
                                stroke="#3b82f6"
                                strokeWidth={2}
                                strokeDasharray="5 5"
                                name="真实值"
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                    <div className="mt-4 text-sm text-slate-500 dark:text-slate-400">
                        <p>💡 大数定律：lim(n→∞) (1/n)Σ Gᵢ = V^π(s)</p>
                        <p className="mt-1">📊 标准误差：SE = σ / √n （误差以 1/√n 速度下降）</p>
                    </div>
                </div>
            )}

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：MC 通过样本平均估计价值函数，无偏但方差较大
            </div>
        </div>
    );
}
