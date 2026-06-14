"use client";

import { useState } from "react";

const graphNodes = [
  { id: 0, name: "data", op: "input", color: "from-slate-400 to-slate-500" },
  { id: 1, name: "weight", op: "input", color: "from-slate-400 to-slate-500" },
  { id: 2, name: "conv2d", op: "nn.conv2d", inputs: [0, 1], color: "from-violet-500 to-purple-600" },
  { id: 3, name: "bias", op: "input", color: "from-slate-400 to-slate-500" },
  { id: 4, name: "add", op: "add", inputs: [2, 3], color: "from-indigo-500 to-blue-600" },
  { id: 5, name: "relu", op: "nn.relu", inputs: [4], color: "from-blue-500 to-cyan-600" },
];

export function GraphJSONDiagram() {
  const [selectedNode, setSelectedNode] = useState<number | null>(null);

  const jsonStr = `{
  "nodes": [
    {"op": "null", "name": "data"},      // 0
    {"op": "null", "name": "weight"},     // 1
    {"op": "nn.conv2d", "inputs": [0,1]},// 2
    {"op": "null", "name": "bias"},       // 3
    {"op": "add", "inputs": [2,3]},       // 4
    {"op": "nn.relu", "inputs": [4]}      // 5
  ],
  "arg_nodes": [0, 1, 3],
  "heads": [5],
  "attrs": {"shape": ["1,3,28,28"]}
}`;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        图 JSON 结构
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
            计算图
          </h4>
          <div className="flex flex-wrap gap-2 justify-center">
            {graphNodes.map((node) => (
              <button
                key={node.id}
                onClick={() => setSelectedNode(selectedNode === node.id ? null : node.id)}
                className={`px-3 py-2 rounded-lg transition-all ${
                  selectedNode === node.id
                    ? "ring-2 ring-indigo-400 ring-offset-2 dark:ring-offset-slate-900 scale-110"
                    : "hover:scale-105"
                }`}
              >
                <div className={`bg-gradient-to-r ${node.color} text-white px-3 py-2 rounded-lg shadow-md text-center`}>
                  <div className="text-[10px] opacity-70">#{node.id}</div>
                  <div className="text-sm font-bold">{node.name}</div>
                  <div className="text-[10px] opacity-70">{node.op}</div>
                </div>
              </button>
            ))}
          </div>

          <div className="mt-4 flex gap-4 justify-center text-xs">
            <span className="px-2 py-1 bg-slate-200 dark:bg-slate-700 rounded text-slate-600 dark:text-slate-400">
              arg_nodes: [0, 1, 3] (输入)
            </span>
            <span className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/40 rounded text-indigo-600 dark:text-indigo-400">
              heads: [5] (输出)
            </span>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
            JSON 表示
          </h4>
          <pre className="bg-slate-900 text-green-400 p-4 rounded-xl text-xs font-mono overflow-y-auto max-h-72">
            {jsonStr}
          </pre>
        </div>
      </div>

      {selectedNode !== null && (
        <div className="mt-4 p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <span className={`px-3 py-1 rounded bg-gradient-to-r ${graphNodes[selectedNode].color} text-white text-sm font-bold`}>
              Node #{selectedNode}
            </span>
            <span className="font-bold text-slate-800 dark:text-slate-100">
              {graphNodes[selectedNode].name}
            </span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 mt-2">
            op: {graphNodes[selectedNode].op}
            {graphNodes[selectedNode].inputs && (
              <>, inputs: [{(graphNodes[selectedNode].inputs as number[]).join(", ")}]</>
            )}
          </p>
        </div>
      )}
    </div>
  );
}
