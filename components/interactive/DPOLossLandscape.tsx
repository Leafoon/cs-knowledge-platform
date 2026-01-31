"use client";

import { useState } from "react";

export function DPOLossLandscape() {
    const [beta, setBeta] = useState(0.1);
    const [klCoef, setKlCoef] = useState(0.01);

    // 生成损失地形
    const generateLandscape = () => {
        const points = [];
        const gridSize = 30;

        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                // 策略log比率 (x轴: -3 to 3)
                const logRatioChosen = -3 + (i / gridSize) * 6;
                // y轴: -3 to 3
                const logRatioRejected = -3 + (j / gridSize) * 6;

                // DPO损失计算
                const rewardDiff = beta * (logRatioChosen - logRatioRejected);
                const dpoLoss = -Math.log(1 / (1 + Math.exp(-rewardDiff)));

                // KL惩罚
                const klPenalty = klCoef * (Math.abs(logRatioChosen) + Math.abs(logRatioRejected));

                const totalLoss = dpoLoss + klPenalty;

                points.push({
                    x: i,
                    y: j,
                    loss: totalLoss,
                    logRatioChosen,
                    logRatioRejected
                });
            }
        }

        return points;
    };

    const landscape = generateLandscape();

    // 找到最小和最大损失用于归一化颜色
    const minLoss = Math.min(...landscape.map(p => p.loss));
    const maxLoss = Math.max(...landscape.map(p => p.loss));

    const getColor = (loss: number) => {
        const normalized = (loss - minLoss) / (maxLoss - minLoss);

        if (normalized < 0.33) {
            // 低损失：绿色
            const intensity = Math.floor(normalized / 0.33 * 255);
            return `rgb(${255 - intensity}, 255, ${255 - intensity})`;
        } else if (normalized < 0.67) {
            // 中损失：黄色
            const intensity = Math.floor((normalized - 0.33) / 0.34 * 255);
            return `rgb(255, ${255 - intensity}, 0)`;
        } else {
            // 高损失：红色
            const intensity = Math.floor((normalized - 0.67) / 0.33 * 255);
            return `rgb(255, ${Math.max(0, 255 - intensity * 2)}, 0)`;
        }
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    DPO 损失地形图
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    可视化DPO优化目标的损失曲面
                </p>
            </div>

            {/* 参数控制 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">超参数调整</h4>

                <div className="grid grid-cols-2 gap-6">
                    <div>
                        <div className="flex justify-between mb-2">
                            <span className="font-semibold text-purple-600 dark:text-purple-400">
                                β (温度参数)
                            </span>
                            <span className="font-mono text-purple-600 dark:text-purple-400">{beta.toFixed(3)}</span>
                        </div>
                        <input
                            type="range"
                            min="0.01"
                            max="0.5"
                            step="0.01"
                            value={beta}
                            onChange={(e) => setBeta(parseFloat(e.target.value))}
                            className="w-full h-3 bg-purple-200 rounded-lg appearance-none cursor-pointer dark:bg-purple-900"
                        />
                        <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                            控制奖励差异的敏感度
                        </div>
                    </div>

                    <div>
                        <div className="flex justify-between mb-2">
                            <span className="font-semibold text-orange-600 dark:text-orange-400">
                                KL系数
                            </span>
                            <span className="font-mono text-orange-600 dark:text-orange-400">{klCoef.toFixed(3)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="0.1"
                            step="0.001"
                            value={klCoef}
                            onChange={(e) => setKlCoef(parseFloat(e.target.value))}
                            className="w-full h-3 bg-orange-200 rounded-lg appearance-none cursor-pointer dark:bg-orange-900"
                        />
                        <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                            KL惩罚强度
                        </div>
                    </div>
                </div>
            </div>

            {/* 损失地形可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">损失曲面</h4>

                <div className="relative bg-gray-100 dark:bg-gray-900 rounded-lg p-4">
                    {/* Heatmap */}
                    <div className="grid grid-cols-30 gap-0" style={{ gridTemplateColumns: 'repeat(30, 1fr)' }}>
                        {landscape.map((point, idx) => (
                            <div
                                key={idx}
                                className="aspect-square"
                                style={{
                                    backgroundColor: getColor(point.loss)
                                }}
                                title={`Loss: ${point.loss.toFixed(3)}\nChosen: ${point.logRatioChosen.toFixed(2)}\nRejected: ${point.logRatioRejected.toFixed(2)}`}
                            />
                        ))}
                    </div>

                    {/* 轴标签 */}
                    <div className="mt-4 text-center text-sm text-slate-600 dark:text-slate-400">
                        log[π<sub>θ</sub>(y<sub>chosen</sub>) / π<sub>ref</sub>(y<sub>chosen</sub>)] →
                    </div>
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-sm text-slate-600 dark:text-slate-400">
                        ← log[π<sub>θ</sub>(y<sub>reject</sub>) / π<sub>ref</sub>(y<sub>reject</sub>)]
                    </div>

                    {/* 最优点标记 */}
                    <div className="absolute top-6 right-6 bg-green-500 w-4 h-4 rounded-full border-2 border-white"></div>
                    <div className="absolute top-6 right-24 text-xs text-green-700 dark:text-green-400 font-semibold">
                        ← 最优解
                    </div>
                </div>

                {/* 颜色图例 */}
                <div className="mt-4 flex items-center justify-center gap-2">
                    <span className="text-xs text-slate-600 dark:text-slate-400">低损失</span>
                    <div className="flex h-6 w-64 rounded-full overflow-hidden">
                        {Array.from({ length: 50 }, (_, i) => {
                            const normalized = i / 50;
                            let color;
                            if (normalized < 0.33) {
                                const intensity = Math.floor(normalized / 0.33 * 255);
                                color = `rgb(${255 - intensity}, 255, ${255 - intensity})`;
                            } else if (normalized < 0.67) {
                                const intensity = Math.floor((normalized - 0.33) / 0.34 * 255);
                                color = `rgb(255, ${255 - intensity}, 0)`;
                            } else {
                                const intensity = Math.floor((normalized - 0.67) / 0.33 * 255);
                                color = `rgb(255, ${Math.max(0, 255 - intensity * 2)}, 0)`;
                            }
                            return (
                                <div
                                    key={i}
                                    className="flex-1 h-full"
                                    style={{ backgroundColor: color }}
                                />
                            );
                        })}
                    </div>
                    <span className="text-xs text-slate-600 dark:text-slate-400">高损失</span>
                </div>
            </div>

            {/* 关键观察 */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-500">
                    <h5 className="font-semibold text-green-700 dark:text-green-400 mb-2 flex items-center gap-2">
                        <span>🎯</span> 最优区域
                    </h5>
                    <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                        <li>• chosen的log比率 &gt; 0（提高概率）</li>
                        <li>• rejected的log比率 &lt; 0（降低概率）</li>
                        <li>• 损失最小化在右下角</li>
                    </ul>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-2 border-blue-500">
                    <h5 className="font-semibold text-blue-700 dark:text-blue-400 mb-2 flex items-center gap-2">
                        <span>📊</span> 参数影响
                    </h5>
                    <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                        <li>• β↑：损失曲面更陡峭</li>
                        <li>• KL系数↑：惩罚偏离参考模型</li>
                        <li>• 平衡二者实现稳定优化</li>
                    </ul>
                </div>
            </div>

            <div className="mt-6 bg-amber-100 dark:bg-amber-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 DPO的凸性质保证了稳定收敛到全局最优
            </div>
        </div>
    );
}
