"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GitBranch, ChevronDown, ChevronRight, Star } from "lucide-react";

interface ThoughtNode {
  id: string;
  content: string;
  score: number;
  children: ThoughtNode[];
}

const SAMPLE_TREE: ThoughtNode = {
  id: "root",
  content: "问题：如何设计一个高并发系统？",
  score: 0,
  children: [
    {
      id: "1",
      content: "使用微服务架构拆分服务",
      score: 8,
      children: [
        { id: "1-1", content: "服务注册与发现 + 负载均衡", score: 9, children: [] },
        { id: "1-2", content: "消息队列解耦服务间通信", score: 8.5, children: [] },
      ],
    },
    {
      id: "2",
      content: "引入缓存层减少数据库压力",
      score: 9,
      children: [
        { id: "2-1", content: "Redis集群 + 本地缓存", score: 9.5, children: [] },
        { id: "2-2", content: "CDN加速静态资源", score: 7, children: [] },
      ],
    },
    {
      id: "3",
      content: "数据库读写分离 + 分库分表",
      score: 7.5,
      children: [
        { id: "3-1", content: "主从复制 + 读写分离", score: 8, children: [] },
      ],
    },
  ],
};

function TreeNode({ node, level = 0 }: { node: ThoughtNode; level?: number }) {
  const [expanded, setExpanded] = useState(level < 2);
  const hasChildren = node.children.length > 0;
  const isBest = node.score >= 9;

  return (
    <div className={`${level > 0 ? "ml-6 border-l-2 border-slate-200 dark:border-slate-700 pl-4" : ""}`}>
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: level * 0.1 }}
        className={`p-3 rounded-lg mb-2 cursor-pointer transition-all ${
          isBest
            ? "bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700"
            : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-purple-300"
        }`}
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          {hasChildren && (
            expanded ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />
          )}
          {!hasChildren && <div className="w-4" />}
          <span className="text-sm text-slate-700 dark:text-slate-200 flex-1">{node.content}</span>
          {node.score > 0 && (
            <span className={`flex items-center gap-1 text-sm font-medium ${
              isBest ? "text-green-600 dark:text-green-400" : "text-slate-500"
            }`}>
              {isBest && <Star className="w-4 h-4 fill-current" />}
              {node.score.toFixed(1)}
            </span>
          )}
        </div>
      </motion.div>

      <AnimatePresence>
        {expanded && hasChildren && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            {node.children.map((child) => (
              <TreeNode key={child.id} node={child} level={level + 1} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function ToTVisualizer() {
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <GitBranch className="w-6 h-6 text-indigo-500" />
        Tree of Thoughts 可视化
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        探索多条推理路径，评估每条路径的质量，选择最优路径继续推理。
      </p>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <TreeNode node={SAMPLE_TREE} />
      </div>

      <div className="mt-4 flex items-center gap-4 text-sm text-slate-500">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-100 border border-green-300 rounded" />
          <span>最优路径 (≥9分)</span>
        </div>
        <div className="flex items-center gap-2">
          <Star className="w-4 h-4 text-green-500 fill-current" />
          <span>星标为最佳节点</span>
        </div>
        <span>点击节点展开/折叠</span>
      </div>
    </div>
  );
}
