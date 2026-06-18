"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Network, Users, ArrowRight } from "lucide-react";

interface Pattern {
  id: string;
  name: string;
  description: string;
  nodes: { x: number; y: number; label: string }[];
}

const PATTERNS: Pattern[] = [
  {
    id: "hierarchical",
    name: "主从模式",
    description: "一个主管Agent管理多个工作Agent",
    nodes: [
      { x: 200, y: 50, label: "主管" },
      { x: 80, y: 150, label: "工作1" },
      { x: 200, y: 150, label: "工作2" },
      { x: 320, y: 150, label: "工作3" },
    ],
  },
  {
    id: "peer",
    name: "对等模式",
    description: "多个Agent平等协作",
    nodes: [
      { x: 100, y: 100, label: "Agent A" },
      { x: 300, y: 100, label: "Agent B" },
      { x: 200, y: 180, label: "Agent C" },
    ],
  },
];

export function MultiAgentPatterns() {
  const [selected, setSelected] = useState(0);
  const pattern = PATTERNS[selected];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Network className="w-6 h-6 text-indigo-500" />
        多Agent协作模式
      </h3>

      <div className="flex gap-3 mb-6">
        {PATTERNS.map((p, idx) => (
          <button
            key={p.id}
            onClick={() => setSelected(idx)}
            className={`px-4 py-2 rounded-lg transition-all ${
              selected === idx
                ? "bg-indigo-600 text-white"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">{pattern.name}</h4>
        <p className="text-slate-600 dark:text-slate-300 mb-4">{pattern.description}</p>
        <svg width="100%" height="200" viewBox="0 0 400 200">
          {pattern.nodes.map((node, i) => (
            <g key={i}>
              <circle cx={node.x} cy={node.y} r="30" fill="#818cf8" />
              <text x={node.x} y={node.y + 5} textAnchor="middle" fill="white" fontSize="12">
                {node.label}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
