"use client";

import { useState } from "react";

export function FeatureEngineeringVisualizer() {
    const [featureType, setFeatureType] = useState<"polynomial" | "tile" | "rbf">("polynomial");

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    特征工程可视化
                </h3>
            </div>

            {/* 特征类型选择 */}
            <div className="flex justify-center gap-4 mb-6">
                <button onClick={() => setFeatureType("polynomial")} className={`px-6 py-2 rounded-lg ${featureType === "polynomial" ? "bg-blue-600 text-white" : "bg-blue-100 text-blue-700"}`}>
                    多项式特征
                </button>
                <button onClick={() => setFeatureType("tile")} className={`px-6 py-2 rounded-lg ${featureType === "tile" ? "bg-indigo-600 text-white" : "bg-indigo-100 text-indigo-700"}`}>
                    Tile Coding
                </button>
                <button onClick={() => setFeatureType("rbf")} className={`px-6 py-2 rounded-lg ${featureType === "rbf" ? "bg-purple-600 text-white" : "bg-purple-100 text-purple-700"}`}>
                    径向基函数
                </button>
            </div>

            {/* 特征说明 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                {featureType === "polynomial" && (
                    <div>
                        <h4 className="text-lg font-bold mb-3">多项式特征</h4>
                        <p className="mb-3">x(s) = [1, s, s², s³, ...]</p>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                            <strong>优点:</strong> 简单、全局特征<br />
                            <strong>缺点:</strong> 维度爆炸、边界效应
                        </div>
                    </div>
                )}
                {featureType === "tile" && (
                    <div>
                        <h4 className="text-lg font-bold mb-3">Tile Coding（瓦片编码）</h4>
                        <p className="mb-3">多个偏移的网格覆盖，二进制激活</p>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                            <strong>优点:</strong> 局部泛化、计算高效、可调分辨率<br />
                            <strong>缺点:</strong> 需要调参（瓦片数、偏移）
                        </div>
                    </div>
                )}
                {featureType === "rbf" && (
                    <div>
                        <h4 className="text-lg font-bold mb-3">径向基函数（RBF）</h4>
                        <p className="mb-3">x<sub>i</sub>(s) = exp(-||s - c<sub>i</sub>||² / 2σ²)</p>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                            <strong>优点:</strong> 平滑、局部泛化<br />
                            <strong>缺点:</strong> 需选择中心点和宽度
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
