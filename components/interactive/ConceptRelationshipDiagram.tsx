"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Network } from "lucide-react";

interface Concept {
  id: string;
  name: string;
  color: string;
  dimension: string;
  definition: string;
  example: string;
  related: string[];
}

const concepts: Concept[] = [
  { id: "sync", name: "同步", color: "#3b82f6", dimension: "结果交付", definition: "调用者在当前流程中等待任务结果，再继续后续逻辑", example: "result = fetch_data()", related: ["blocking", "async"] },
  { id: "async", name: "异步", color: "#8b5cf6", dimension: "结果交付", definition: "任务结果不会立即获得，调用者可以先处理其他工作", example: "task = create_task(fetch); await task", related: ["sync", "nonblocking", "concurrency"] },
  { id: "blocking", name: "阻塞", color: "#ef4444", dimension: "线程行为", definition: "当前执行线程因为等待而无法继续执行其他代码", example: "time.sleep(3)", related: ["sync", "nonblocking"] },
  { id: "nonblocking", name: "非阻塞", color: "#f59e0b", dimension: "线程行为", definition: "调用立即返回，不会一直卡住当前线程", example: "result = try_get(); if not ready: ...", related: ["blocking", "async"] },
  { id: "concurrency", name: "并发", color: "#10b981", dimension: "时间模型", definition: "多个任务在同一段时间内交替推进", example: "asyncio event loop", related: ["parallelism", "async"] },
  { id: "parallelism", name: "并行", color: "#06b6d4", dimension: "时间模型", definition: "多个任务在同一时刻真正执行", example: "multiprocessing", related: ["concurrency"] },
];

const combinations = [
  { label: "同步 + 阻塞", desc: "发起操作后一直等待结果", example: "time.sleep()", color: "#ef4444" },
  { label: "同步 + 非阻塞", desc: "发起操作后主动查询状态", example: "轮询模式", color: "#f59e0b" },
  { label: "异步 + 非阻塞", desc: "发起任务后去做别的事", example: "await asyncio.sleep()", color: "#10b981" },
  { label: "异步中夹杂阻塞", desc: "async def 内部错误使用阻塞调用", example: "async def: time.sleep()", color: "#ef4444" },
];

export function ConceptRelationshipDiagram() {
  const [selected, setSelected] = useState<string | null>(null);
  const active = concepts.find((c) => c.id === selected);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Network className="w-5 h-5" />
        六个核心概念关系图
      </h3>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6">
        {concepts.map((c) => (
          <button key={c.id} onClick={() => setSelected(selected === c.id ? null : c.id)}
            className={`p-3 rounded-xl border-2 transition-all text-center ${
              selected === c.id ? "border-current shadow-lg scale-105" : "border-transparent hover:border-slate-200 dark:hover:border-slate-700"
            }`}
            style={{ color: c.color, backgroundColor: selected === c.id ? `${c.color}15` : undefined }}>
            <div className="text-lg font-bold">{c.name}</div>
            <div className="text-[10px] opacity-60">{c.dimension}</div>
          </button>
        ))}
      </div>
      <AnimatePresence>
        {active && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
            className="bg-white dark:bg-slate-800 rounded-xl border p-5 mb-6" style={{ borderColor: active.color }}>
            <h4 className="font-bold text-lg mb-2" style={{ color: active.color }}>{active.name}</h4>
            <p className="text-sm text-slate-700 dark:text-slate-300 mb-2">{active.definition}</p>
            <div className="text-xs text-slate-500 bg-slate-50 dark:bg-slate-900 rounded p-2">
              示例：<code className="text-indigo-600 dark:text-indigo-400">{active.example}</code>
            </div>
            <div className="mt-2 text-xs text-slate-500">相关概念：{active.related.map((r) => concepts.find((c) => c.id === r)?.name).join("、")}</div>
          </motion.div>
        )}
      </AnimatePresence>
      <h4 className="font-bold text-sm text-slate-700 dark:text-slate-300 mb-3">常见组合</h4>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {combinations.map((c, i) => (
          <div key={i} className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-3">
            <div className="text-xs font-bold mb-1" style={{ color: c.color }}>{c.label}</div>
            <div className="text-[11px] text-slate-600 dark:text-slate-400">{c.desc}</div>
            <code className="text-[10px] text-indigo-500 mt-1 block">{c.example}</code>
          </div>
        ))}
      </div>
    </div>
  );
}
