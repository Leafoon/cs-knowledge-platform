"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Table, CheckCircle, XCircle } from "lucide-react";

const methods = [
  { name: "程序查询", color: "red" },
  { name: "中断驱动", color: "yellow" },
  { name: "DMA", color: "blue" },
  { name: "通道", color: "green" },
];

const attributes = [
  {
    name: "CPU参与程度",
    values: ["全程参与", "数据传送时参与", "仅初始化/结束", "仅启动通道"],
    scores: [1, 2, 3, 4],
  },
  {
    name: "数据传送单位",
    values: ["字", "字", "数据块", "数据块/记录"],
    scores: [1, 1, 3, 4],
  },
  {
    name: "并行度",
    values: ["无并行", "CPU与I/O部分并行", "CPU与I/O高度并行", "完全并行"],
    scores: [1, 2, 3, 4],
  },
  {
    name: "系统复杂度",
    values: ["最简单", "较简单", "较复杂", "最复杂"],
    scores: [4, 3, 2, 1],
  },
  {
    name: "适用场景",
    values: ["简单嵌入式系统", "一般I/O设备", "高速批量传输", "大型计算机系统"],
    scores: [1, 2, 3, 4],
  },
  {
    name: "实时性",
    values: ["差（轮询延迟）", "较好（中断响应）", "好", "最好"],
    scores: [1, 2, 3, 4],
  },
  {
    name: "成本",
    values: ["最低", "低", "中等", "最高"],
    scores: [4, 3, 2, 1],
  },
];

export function IOSystemComparison() {
  const [highlight, setHighlight] = useState<number | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Table className="w-5 h-5 text-amber-400" />
        <h3 className="text-lg font-semibold">I/O 方式综合对比</h3>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left p-2 text-gray-400 text-xs">特性</th>
              {methods.map((m, i) => (
                <th key={m.name}
                  onMouseEnter={() => setHighlight(i)}
                  onMouseLeave={() => setHighlight(null)}
                  className={`p-2 text-xs text-center cursor-pointer transition-colors ${
                    highlight === i ? `bg-${m.color}-500/10` : ""
                  }`}
                >
                  <span className={`text-${m.color}-400`}>{m.name}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {attributes.map((attr) => (
              <tr key={attr.name} className="border-t border-gray-700/50">
                <td className="p-2 text-xs text-gray-300">{attr.name}</td>
                {attr.values.map((val, i) => (
                  <td key={i}
                    onMouseEnter={() => setHighlight(i)}
                    onMouseLeave={() => setHighlight(null)}
                    className={`p-2 text-xs text-center transition-colors ${
                      highlight === i ? `bg-${methods[i].color}-500/10` : ""
                    }`}
                  >
                    <div className="text-gray-300">{val}</div>
                    <div className="flex justify-center gap-0.5 mt-1">
                      {Array.from({ length: 4 }).map((_, j) => (
                        <motion.div key={j}
                          className="w-1.5 h-1.5 rounded-full"
                          animate={{
                            backgroundColor: j < attr.scores[i] ? "#facc15" : "#374151",
                          }}
                        />
                      ))}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 p-3 bg-gray-800/30 rounded-lg">
        <div className="text-xs text-gray-400 mb-2">发展趋势:</div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-red-300">程序查询</span>
          <span className="text-gray-500">→</span>
          <span className="text-yellow-300">中断</span>
          <span className="text-gray-500">→</span>
          <span className="text-blue-300">DMA</span>
          <span className="text-gray-500">→</span>
          <span className="text-green-300">通道</span>
          <span className="text-gray-400 ml-2">CPU逐步解放，I/O效率不断提高</span>
        </div>
      </div>
    </div>
  );
}
