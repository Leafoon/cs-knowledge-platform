"use client";

import { useState } from "react";

const tree = {
    name: "Expr",
    color: "from-indigo-500 to-purple-500",
    children: [
        {
            name: "Var",
            desc: "变量节点，持有类型和名称",
            color: "from-blue-500 to-indigo-500",
            children: [],
        },
        {
            name: "Constant",
            desc: "常量节点，持有字面值",
            color: "from-purple-500 to-pink-500",
            children: [],
        },
        {
            name: "Call",
            desc: "函数调用节点，包含操作符和参数列表",
            color: "from-emerald-500 to-teal-500",
            children: [
                { name: "Op", desc: "操作符标识", color: "from-cyan-500 to-blue-500" },
                { name: "args[]", desc: "参数表达式列表", color: "from-teal-500 to-emerald-500" },
            ],
        },
        {
            name: "Tuple",
            desc: "元组节点，包含多个字段表达式",
            color: "from-amber-500 to-orange-500",
            children: [
                { name: "fields[]", desc: "字段表达式列表", color: "from-orange-500 to-red-500" },
            ],
        },
    ],
};

export function ClassStructureDiagram() {
    const [expanded, setExpanded] = useState<Set<string>>(new Set(["Expr", "Call", "Tuple"]));

    const toggle = (name: string) => {
        setExpanded((prev) => {
            const next = new Set(prev);
            if (next.has(name)) next.delete(name);
            else next.add(name);
            return next;
        });
    };

    const renderNode = (node: any, depth: number = 0) => (
        <div key={node.name} style={{ paddingLeft: depth * 24 }}>
            <button
                onClick={() => toggle(node.name)}
                className={`w-full text-left flex items-center gap-3 p-3 rounded-lg mb-2 transition-all bg-white dark:bg-slate-800 shadow ${
                    expanded.has(node.name) ? "ring-1 ring-indigo-300 dark:ring-indigo-700" : ""
                }`}
            >
                <div className={`w-8 h-8 rounded-lg bg-gradient-to-r ${node.color} flex items-center justify-center text-white text-xs font-bold shrink-0`}>
                    {node.name[0]}
                </div>
                <div className="flex-1">
                    <span className="text-sm font-bold text-slate-800 dark:text-slate-100 font-mono">{node.name}</span>
                    {node.desc && <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{node.desc}</p>}
                </div>
                {node.children?.length > 0 && (
                    <span className="text-slate-400 text-xs">{expanded.has(node.name) ? "▼" : "▶"}</span>
                )}
            </button>
            {expanded.has(node.name) && node.children?.map((child: any) => renderNode(child, depth + 1))}
        </div>
    );

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">类继承结构</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-4">Expr 类层次</h4>
                {renderNode(tree)}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                    { name: "Expr", desc: "基类", color: "text-indigo-600 dark:text-indigo-400" },
                    { name: "Var", desc: "变量", color: "text-blue-600 dark:text-blue-400" },
                    { name: "Constant", desc: "常量", color: "text-purple-600 dark:text-purple-400" },
                    { name: "Call", desc: "调用", color: "text-emerald-600 dark:text-emerald-400" },
                ].map((t, i) => (
                    <div key={i} className="bg-white dark:bg-slate-800 rounded-xl p-3 shadow-lg text-center">
                        <div className={`text-sm font-bold font-mono ${t.color}`}>{t.name}</div>
                        <div className="text-xs text-slate-500 mt-1">{t.desc}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
