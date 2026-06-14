"use client";

import { useState } from "react";

interface Node {
  id: string;
  label: string;
  type: "op" | "var" | "const";
  x: number;
  y: number;
}

interface Edge {
  from: string;
  to: string;
  label: string;
}

const nodes: Node[] = [
  { id: "x", label: "%x\nfloat32", type: "var", x: 50, y: 80 },
  { id: "w1", label: "W1\n(128,64)", type: "const", x: 50, y: 200 },
  { id: "dense1", label: "nn.dense", type: "op", x: 200, y: 140 },
  { id: "relu", label: "nn.relu", type: "op", x: 350, y: 140 },
  { id: "w2", label: "W2\n(64,10)", type: "const", x: 350, y: 250 },
  { id: "dense2", label: "nn.dense", type: "op", x: 480, y: 180 },
  { id: "out", label: "output", type: "var", x: 600, y: 180 },
];

const edges: Edge[] = [
  { from: "x", to: "dense1", label: "" },
  { from: "w1", to: "dense1", label: "" },
  { from: "dense1", to: "relu", label: "" },
  { from: "relu", to: "dense2", label: "" },
  { from: "w2", to: "dense2", label: "" },
  { from: "dense2", to: "out", label: "" },
];

const nodeColors: Record<string, { bg: string; border: string; text: string }> = {
  op: { bg: "fill-indigo-100 dark:fill-indigo-900/60", border: "stroke-indigo-500", text: "text-indigo-700 dark:text-indigo-300" },
  var: { bg: "fill-emerald-100 dark:fill-emerald-900/60", border: "stroke-emerald-500", text: "text-emerald-700 dark:text-emerald-300" },
  const: { bg: "fill-amber-100 dark:fill-amber-900/60", border: "stroke-amber-500", text: "text-amber-700 dark:text-amber-300" },
};

export function RelayVisualizationDemo() {
  const [selected, setSelected] = useState<string | null>(null);

  const nodeDetails: Record<string, string> = {
    x: "输入变量: Tensor[(128,), float32]\n模型输入，绑定到实际数据",
    w1: "常量权重: Tensor[(128, 64), float32]\n第一层全连接的权重矩阵",
    dense1: "nn.dense(x, W1)\n矩阵乘法: (128,) × (128, 64) → (64,)\n计算: out = x @ W1^T",
    relu: "nn.relu(dense1)\n逐元素 ReLU 激活\nmax(0, x)",
    w2: "常量权重: Tensor[(64, 10), float32]\n第二层全连接的权重矩阵",
    dense2: "nn.dense(relu_out, W2)\n矩阵乘法: (64,) × (64, 10) → (10,)\n计算: out = relu @ W2^T",
    out: "输出: Tensor[(10,), float32]\n模型最终输出，10类分类",
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">Relay DAG 可视化</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">节点=算子，边=数据流，构成计算有向无环图</p>

      <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700 mb-5">
        <svg viewBox="0 0 680 300" className="w-full h-auto">
          {edges.map((e, i) => {
            const from = nodes.find((n) => n.id === e.from)!;
            const to = nodes.find((n) => n.id === e.to)!;
            return (
              <line
                key={i}
                x1={from.x + 40}
                y1={from.y + 15}
                x2={to.x}
                y2={to.y + 15}
                className="stroke-slate-300 dark:stroke-slate-600"
                strokeWidth={2}
                markerEnd="url(#arrow)"
              />
            );
          })}
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" className="fill-slate-400 dark:fill-slate-500" />
            </marker>
          </defs>
          {nodes.map((n) => (
            <g key={n.id} onClick={() => setSelected(n.id)} className="cursor-pointer">
              <rect
                x={n.x}
                y={n.y}
                width={80}
                height={30}
                rx={6}
                className={`${nodeColors[n.type].bg} ${nodeColors[n.type].border} ${selected === n.id ? "stroke-[3]" : "stroke-[1.5]"}`}
              />
              <text
                x={n.x + 40}
                y={n.y + 19}
                textAnchor="middle"
                className={`text-[11px] font-bold ${nodeColors[n.type].text}`}
              >
                {n.label.split("\n")[0]}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">图例</h4>
          <div className="space-y-2">
            {[
              { type: "op", label: "算子节点", desc: "nn.dense, nn.relu 等" },
              { type: "var", label: "变量节点", desc: "输入/输出张量" },
              { type: "const", label: "常量节点", desc: "权重、偏置等参数" },
            ].map((l, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded ${nodeColors[l.type].bg} ${nodeColors[l.type].border} border-2`} />
                <span className="text-xs font-bold text-slate-700 dark:text-slate-200">{l.label}</span>
                <span className="text-xs text-slate-500 dark:text-slate-400">{l.desc}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">
            {selected ? `节点: ${nodes.find((n) => n.id === selected)?.label.split("\n")[0]}` : "点击节点查看详情"}
          </h4>
          <pre className="text-xs text-slate-600 dark:text-slate-300 whitespace-pre-wrap font-mono">
            {selected ? nodeDetails[selected] : "选择 DAG 中的节点以查看其详细信息"}
          </pre>
        </div>
      </div>
    </div>
  );
}
