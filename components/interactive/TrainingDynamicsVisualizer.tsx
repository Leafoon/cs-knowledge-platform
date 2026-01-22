"use client";

import React, { useState } from 'react';

const TrainingDynamicsVisualizer = () => {
    const [epochs, setEpochs] = useState(50);

    // Simulate typical learning curves
    // x: 0 to 100 epochs
    // Train Loss: Exponential decay
    // Val Loss: Decay then rise (Overfitting)

    // We render an SVG based on 'epochs' slider
    // The visualization shows the state at 'epochs'

    const width = 400;
    const height = 250;
    const padding = 30;
    const maxEpochs = 100;

    const getData = (e: number) => {
        // e is current epoch cursor (0-100)
        // Generate full history curve points
        const trainPts = [];
        const valPts = [];
        for (let i = 0; i <= maxEpochs; i++) {
            const tLoss = 1.0 * Math.exp(-0.05 * i) + 0.05;
            // Val loss: drops then rises
            // Ideal at around epoch 30
            const vLoss = 1.0 * Math.exp(-0.04 * i) + 0.0005 * Math.pow(Math.max(0, i - 30), 2) + 0.1;

            trainPts.push({ x: i, y: tLoss });
            valPts.push({ x: i, y: vLoss });
        }
        return { trainPts, valPts };
    };

    const { trainPts, valPts } = getData(epochs);

    const scaleX = (x: number) => padding + (x / maxEpochs) * (width - 2 * padding);
    const scaleY = (y: number) => height - padding - (y / 1.2) * (height - 2 * padding); // Max loss ~1.1

    const toPath = (pts: any[]) => {
        return pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.x)} ${scaleY(p.y)}`).join(' ');
    };

    const currentTrainLoss = trainPts[epochs].y;
    const currentValLoss = valPts[epochs].y;

    let status = "Underfitting";
    if (epochs > 20 && epochs < 45) status = "Optimal (Sweet Spot)";
    if (epochs >= 45) status = "Overfitting";

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">训练动态：过拟合与欠拟合</h3>

            <div className="relative border border-border-subtle rounded-lg bg-bg-surface overflow-hidden">
                <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
                    {/* Zones */}
                    <rect x={scaleX(45)} y={0} width={width - scaleX(45) - padding} height={height} fill="rgba(255, 0, 0, 0.05)" />
                    <text x={scaleX(70)} y={30} fontSize="10" fill="red" opacity="0.5">Overfitting Zone</text>

                    {/* Axes */}
                    <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#ccc" />
                    <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#ccc" />

                    {/* Curves (Full ghost traces) */}
                    <path d={toPath(trainPts)} fill="none" stroke="blue" strokeWidth="1" strokeDasharray="4 4" opacity="0.3" />
                    <path d={toPath(valPts)} fill="none" stroke="orange" strokeWidth="1" strokeDasharray="4 4" opacity="0.3" />

                    {/* Active Curves (up to current epoch) */}
                    <path d={toPath(trainPts.slice(0, epochs + 1))} fill="none" stroke="blue" strokeWidth="2" />
                    <path d={toPath(valPts.slice(0, epochs + 1))} fill="none" stroke="orange" strokeWidth="2" />

                    {/* Current Points */}
                    <circle cx={scaleX(epochs)} cy={scaleY(currentTrainLoss)} r="4" fill="blue" />
                    <circle cx={scaleX(epochs)} cy={scaleY(currentValLoss)} r="4" fill="orange" />

                    {/* Legend */}
                    <g transform={`translate(${width - 100}, ${padding})`}>
                        <rect width="10" height="10" fill="blue" />
                        <text x="15" y="9" fontSize="10" fill="gray">Train Loss</text>
                        <rect y="15" width="10" height="10" fill="orange" />
                        <text x="15" y="24" fontSize="10" fill="gray">Val Loss</text>
                    </g>
                </svg>
            </div>

            <div className="mt-6 flex gap-8 items-center">
                <div className="flex-1">
                    <label className="block text-sm font-bold text-text-secondary mb-2">Training Epochs: {epochs}</label>
                    <input
                        type="range"
                        min="0" max="100"
                        value={epochs}
                        onChange={(e) => setEpochs(parseInt(e.target.value))}
                        className="w-full accent-accent-primary"
                    />
                </div>

                <div className="w-48 p-3 rounded-lg border text-center transition-colors duration-300"
                    style={{
                        backgroundColor: status === 'Overfitting' ? '#fef2f2' : status === 'Optimal (Sweet Spot)' ? '#f0fdf4' : '#eff6ff',
                        borderColor: status === 'Overfitting' ? '#fca5a5' : status === 'Optimal (Sweet Spot)' ? '#86efac' : '#bfdbfe',
                    }}
                >
                    <div className="text-xs text-text-tertiary uppercase tracking-wide">Current State</div>
                    <div className="font-bold text-sm"
                        style={{
                            color: status === 'Overfitting' ? '#dc2626' : status === 'Optimal (Sweet Spot)' ? '#16a34a' : '#2563eb',
                        }}
                    >{status}</div>
                </div>
            </div>

            <div className="mt-4 text-xs text-text-tertiary leading-relaxed">
                拖动滑块观察：
                <ul className="list-disc pl-4 mt-1 space-y-1">
                    <li><strong>Underfitting (前期)</strong>: Train 和 Val Loss 都很高，模型还没学会特征。</li>
                    <li><strong>Optimal (中期)</strong>: Train Loss 继续下降，Val Loss 达到最低点。这是停止训练的最佳时机 (Early Stopping)。</li>
                    <li><strong>Overfitting (后期)</strong>: Train Loss 继续下降（死记硬背），但 Val Loss 开始反弹（泛化能力变差）。</li>
                </ul>
            </div>
        </div>
    );
};

export default TrainingDynamicsVisualizer;
