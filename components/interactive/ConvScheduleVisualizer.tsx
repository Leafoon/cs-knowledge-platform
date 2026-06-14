"use client";

import { useState } from "react";

const tileSizes = [
    { label: "No Tiling", outer: 1, inner: 1, vectorize: false },
    { label: "Tile=4", outer: 4, inner: 4, vectorize: false },
    { label: "Tile=4+Vec", outer: 4, inner: 4, vectorize: true },
    { label: "Tile=8+Vec", outer: 8, inner: 8, vectorize: true },
];

const buildLines = (s: (typeof tileSizes)[number]) => {
    const lines: { text: string; indent: number; hl?: boolean }[] = [
        { text: "for n in range(N):", indent: 0 },
        { text: "for c in range(C):", indent: 1 },
    ];
    if (s.outer === 1) {
        lines.push({ text: "for h in range(H):", indent: 2 });
        lines.push({ text: "for w in range(W):", indent: 3 });
        lines.push({ text: "output[n,c,h,w] = conv(input, kernel)", indent: 4, hl: true });
    } else {
        lines.push({ text: "for h_o in range(0, H, TILE):", indent: 2 });
        lines.push({ text: "for w_o in range(0, W, TILE):", indent: 3 });
        lines.push({ text: "for h_i in range(h_o, min(h_o+TILE, H)):", indent: 4 });
        if (s.vectorize) {
            lines.push({ text: "for w_i in range(w_o, w_o+TILE, VEC):", indent: 5, hl: true });
            lines.push({ text: "output[...,w_i:w_i+VEC] = conv_vec(...)", indent: 6, hl: true });
        } else {
            lines.push({ text: "for w_i in range(w_o, min(w_o+TILE, W)):", indent: 5 });
            lines.push({ text: "output[n,c,h_i,w_i] = conv(...)", indent: 6, hl: true });
        }
    }
    return lines;
};

export function ConvScheduleVisualizer() {
    const [sel, setSel] = useState(0);
    const cfg = tileSizes[sel];
    const lines = buildLines(cfg);
    const score = cfg.vectorize ? (cfg.outer === 8 ? 95 : 80) : cfg.outer === 1 ? 30 : 60;

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">卷积调度 — NCHW 分块策略</h3>

            <div className="flex gap-2 mb-6 flex-wrap">
                {tileSizes.map((t, i) => (
                    <button
                        key={i}
                        onClick={() => setSel(i)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            sel === i
                                ? "bg-indigo-600 text-white shadow-lg"
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        {t.label}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-3">循环嵌套</h4>
                    <div className="bg-slate-100 dark:bg-slate-900 rounded-lg p-4 font-mono text-sm">
                        {lines.map((l, i) => (
                            <div
                                key={i}
                                className={`py-0.5 ${l.hl ? "text-emerald-600 dark:text-emerald-400 font-semibold" : "text-slate-700 dark:text-slate-300"}`}
                                style={{ paddingLeft: l.indent * 16 }}
                            >
                                {l.text}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                        <h4 className="text-base font-bold text-purple-600 dark:text-purple-400 mb-3">调度参数</h4>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between"><span className="text-slate-500">Tile Size</span><span className="font-mono text-indigo-600 dark:text-indigo-400">{cfg.outer}</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">向量化</span><span className={cfg.vectorize ? "text-emerald-600 dark:text-emerald-400" : "text-red-500"}>{cfg.vectorize ? "是" : "否"}</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">循环深度</span><span className="font-mono text-indigo-600 dark:text-indigo-400">{lines.length}</span></div>
                        </div>
                    </div>

                    <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                        <h4 className="text-base font-bold text-blue-600 dark:text-blue-400 mb-2">性能提升</h4>
                        <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-4 overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500" style={{ width: `${score}%` }} />
                        </div>
                        <div className="text-right text-xs text-slate-500 mt-1">{score}%</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
