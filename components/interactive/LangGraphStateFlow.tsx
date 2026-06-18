"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Workflow, Play, RotateCcw, Check } from "lucide-react";

interface GraphNode {
  id: string;
  label: string;
  type: "start" | "agent" | "tool" | "end";
  x: number;
  y: number;
}

interface GraphEdge {
  from: string;
  to: string;
  label?: string;
}

const NODES: GraphNode[] = [
  { id: "start", label: "START", type: "start", x: 50, y: 150 },
  { id: "agent", label: "Agent 节点", type: "agent", x: 200, y: 150 },
  { id: "tool", label: "Tool 节点", type: "tool", x: 400, y: 150 },
  { id: "end", label: "END", type: "end", x: 600, y: 150 },
];

const EDGES: GraphEdge[] = [
  { from: "start", to: "agent" },
  { from: "agent", to: "tool", label: "需要工具" },
  { from: "agent", to: "end", label: "生成回答" },
  { from: "tool", to: "agent", label: "返回结果" },
];

const NODE_COLORS: Record<GraphNode["type"], string> = {
  start: "#10b981",
  agent: "#3b82f6",
  tool: "#f59e0b",
  end: "#ef4444",
};

export function LangGraphStateFlow() {
  const [activeNode, setActiveNode] = useState<string>("start");
  const [isRunning, setIsRunning] = useState(false);
  const [state, setState] = useState<Record<string, string>>({
    messages: "[]",
    current_step: "0",
  });

  const handleRun = () => {
    if (isRunning) return;
    setIsRunning(true);

    const sequence = ["start", "agent", "tool", "agent", "end"];
    let idx = 0;

    const interval = setInterval(() => {
      setActiveNode(sequence[idx]);
      setState({
        messages: idx === 0 ? "[]" : `["用户问题"]`,
        current_step: String(idx),
      });
      idx++;
      if (idx >= sequence.length) {
        clearInterval(interval);
        setIsRunning(false);
      }
    }, 800);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Workflow className="w-6 h-6 text-blue-500" />
        LangGraph 状态图执行
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        LangGraph 将 Agent 行为建模为有向图，支持循环、条件分支和状态管理。
      </p>

      <div className="flex gap-3 mb-6">
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          {isRunning ? "执行中..." : "执行图"}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 图可视化 */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
          <svg width="100%" height="200" viewBox="0 0 650 200">
            {/* 绘制边 */}
            {EDGES.map((edge, i) => {
              const from = NODES.find((n) => n.id === edge.from)!;
              const to = NODES.find((n) => n.id === edge.to)!;
              return (
                <g key={i}>
                  <line
                    x1={from.x + 40}
                    y1={from.y}
                    x2={to.x - 10}
                    y2={to.y}
                    stroke="#cbd5e1"
                    strokeWidth="2"
                    markerEnd="url(#arrowhead)"
                  />
                  {edge.label && (
                    <text
                      x={(from.x + to.x) / 2 + 20}
                      y={from.y - 15}
                      className="text-xs fill-slate-500"
                      textAnchor="middle"
                    >
                      {edge.label}
                    </text>
                  )}
                </g>
              );
            })}
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#cbd5e1" />
              </marker>
            </defs>

            {/* 绘制节点 */}
            {NODES.map((node) => (
              <g key={node.id}>
                <rect
                  x={node.x - 10}
                  y={node.y - 25}
                  width={node.type === "start" || node.type === "end" ? 60 : 100}
                  height={50}
                  rx="8"
                  fill={activeNode === node.id ? NODE_COLORS[node.type] : "#f1f5f9"}
                  stroke={NODE_COLORS[node.type]}
                  strokeWidth={activeNode === node.id ? 3 : 1}
                />
                <text
                  x={node.x + (node.type === "start" || node.type === "end" ? 20 : 40)}
                  y={node.y + 5}
                  textAnchor="middle"
                  className={`text-sm font-medium ${activeNode === node.id ? "fill-white" : "fill-slate-700"}`}
                >
                  {node.label}
                </text>
              </g>
            ))}
          </svg>
        </div>

        {/* 状态面板 */}
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-4">当前状态</h4>
          <div className="space-y-3">
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
              <span className="text-xs text-slate-500 block mb-1">messages</span>
              <code className="text-sm text-blue-600 dark:text-blue-400 break-all">{state.messages}</code>
            </div>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
              <span className="text-xs text-slate-500 block mb-1">current_step</span>
              <code className="text-sm text-blue-600 dark:text-blue-400">{state.current_step}</code>
            </div>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
              <span className="text-xs text-slate-500 block mb-1">active_node</span>
              <code className="text-sm text-emerald-600 dark:text-emerald-400">{activeNode}</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
