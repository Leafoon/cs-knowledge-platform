"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Code2, Check, X, Info } from "lucide-react";

interface Principle {
  id: string;
  name: string;
  description: string;
  good: string[];
  bad: string[];
  example: {
    good: string;
    bad: string;
  };
}

const principles: Principle[] = [
  {
    id: "simplicity",
    name: "简洁性 (Simplicity)",
    description: "API 应该简单直观，最小化学习成本",
    good: ["函数名清晰表达功能", "参数数量合理 (≤4)", "避免过度设计"],
    bad: ["函数名模糊", "参数过多", "过度抽象"],
    example: {
      good: "open(filename, mode)",
      bad: "file_operation_handler(filename, mode, flags, options, callback, context)"
    }
  },
  {
    id: "consistency",
    name: "一致性 (Consistency)",
    description: "相似功能应使用相似的命名和参数模式",
    good: ["命名风格统一", "参数顺序一致", "错误处理统一"],
    bad: ["混用命名风格", "参数顺序混乱", "错误处理不一致"],
    example: {
      good: "read(fd, buf, size) / write(fd, buf, size)",
      bad: "read(fd, buf, size) / write(size, fd, buf)"
    }
  },
  {
    id: "orthogonality",
    name: "正交性 (Orthogonality)",
    description: "功能独立，修改一处不影响其他部分",
    good: ["单一职责", "功能独立", "最小耦合"],
    bad: ["功能重叠", "相互依赖", "副作用多"],
    example: {
      good: "open() / read() / write() 独立",
      bad: "open() 自动 read()，write() 自动 close()"
    }
  },
  {
    id: "safety",
    name: "安全性 (Safety)",
    description: "防止误用，提供错误检查和清晰的错误信息",
    good: ["类型检查", "边界检查", "明确错误码"],
    bad: ["无类型检查", "缓冲区溢出", "返回值模糊"],
    example: {
      good: "返回 -1 + errno 详细错误",
      bad: "返回 0/1/-1 不明确含义"
    }
  },
  {
    id: "efficiency",
    name: "高效性 (Efficiency)",
    description: "API 调用开销小，避免不必要的复制和转换",
    good: ["零拷贝设计", "批量操作", "异步接口"],
    bad: ["频繁拷贝", "强制同步", "单次操作"],
    example: {
      good: "writev() 批量写",
      bad: "多次 write() 单字节写"
    }
  }
];

export default function APIDesignPrinciples() {
  const [selectedPrinciple, setSelectedPrinciple] = useState<string | null>(principles[0].id);

  const selected = principles.find(p => p.id === selectedPrinciple);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Code2 className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          系统 API 设计原则
        </h3>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 原则列表 */}
        <div className="lg:col-span-1 space-y-2">
          {principles.map((principle) => (
            <motion.button
              key={principle.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setSelectedPrinciple(principle.id)}
              className={`
                w-full text-left p-4 rounded-lg transition-all
                ${selectedPrinciple === principle.id
                  ? "bg-indigo-600 text-white shadow-lg"
                  : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-indigo-50 dark:hover:bg-slate-700"
                }
              `}
            >
              <div className="font-semibold">{principle.name}</div>
              <div className={`text-sm mt-1 ${selectedPrinciple === principle.id ? "text-indigo-100" : "text-slate-500 dark:text-slate-400"}`}>
                {principle.description}
              </div>
            </motion.button>
          ))}
        </div>

        {/* 详细内容 */}
        {selected && (
          <motion.div
            key={selected.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2 space-y-4"
          >
            {/* 好的实践 */}
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-3 flex items-center gap-2">
                <Check className="w-5 h-5" />
                好的实践
              </h4>
              <ul className="space-y-2">
                {selected.good.map((item, i) => (
                  <li key={i} className="text-sm text-green-700 dark:text-green-200 flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 mt-1.5" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>

            {/* 不好的实践 */}
            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
              <h4 className="font-semibold text-red-800 dark:text-red-300 mb-3 flex items-center gap-2">
                <X className="w-5 h-5" />
                应避免的做法
              </h4>
              <ul className="space-y-2">
                {selected.bad.map((item, i) => (
                  <li key={i} className="text-sm text-red-700 dark:text-red-200 flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500 mt-1.5" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>

            {/* 代码示例对比 */}
            <div className="p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
              <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
                <Info className="w-5 h-5" />
                代码示例对比
              </h4>
              <div className="grid gap-3">
                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-800">
                  <div className="text-xs font-mono text-green-700 dark:text-green-300 mb-1">✓ 推荐</div>
                  <code className="text-sm font-mono text-green-900 dark:text-green-100">
                    {selected.example.good}
                  </code>
                </div>
                <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800">
                  <div className="text-xs font-mono text-red-700 dark:text-red-300 mb-1">✗ 不推荐</div>
                  <code className="text-sm font-mono text-red-900 dark:text-red-100 break-all">
                    {selected.example.bad}
                  </code>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* 总结 */}
      <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
        <p className="text-sm text-indigo-900 dark:text-indigo-100">
          <strong>设计哲学：</strong> UNIX 哲学强调 "做一件事并做好"、"简单胜于复杂"、"提供机制而非策略"。
          好的 API 设计能让程序员高效工作，减少错误，提升系统整体质量。
        </p>
      </div>
    </div>
  );
}
