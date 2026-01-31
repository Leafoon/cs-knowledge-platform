"use client";

export function CurriculumProgression() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-slate-900 dark:to-yellow-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    课程学习进度
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">难度递进</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                        <strong>Level 1: 简单</strong><br />
                        <div className="text-sm mt-2">静态目标、固定位置</div>
                        <div className="text-xs text-green-600">成功率: 95%</div>
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded border-2 border-yellow-500">
                        <strong>Level 2: 中等</strong><br />
                        <div className="text-sm mt-2">随机位置、轻微扰动</div>
                        <div className="text-xs text-yellow-600">成功率: 75%</div>
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded border-2 border-orange-500">
                        <strong>Level 3: 困难</strong><br />
                        <div className="text-sm mt-2">运动目标、动态障碍</div>
                        <div className="text-xs text-orange-600">成功率: 50%</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">课程学习优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 学习效率</strong><br />
                        渐进式学习比直接学难任务快
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 避免失败</strong><br />
                        减少早期探索的挫折
                    </div>
                </div>
            </div>
        </div>
    );
}
