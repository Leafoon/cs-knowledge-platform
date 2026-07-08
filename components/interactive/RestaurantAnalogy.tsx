"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Utensils, Clock, Users, ChefHat } from "lucide-react";

interface Scenario {
  id: number;
  title: string;
  type: string;
  description: string;
  totalTime: number;
  steps: { label: string; start: number; end: number; color: string }[];
}

const scenarios: Scenario[] = [
  {
    id: 1,
    title: "站在窗口一直等",
    type: "同步 + 阻塞",
    description: "你点完餐后站在窗口前什么也不做，直到面做好。当前执行者停下来等待。",
    totalTime: 5,
    steps: [{ label: "等待出餐", start: 0, end: 5, color: "#ef4444" }],
  },
  {
    id: 2,
    title: "点餐后拿号码牌",
    type: "异步",
    description: "点完餐拿到号码牌，回座位玩手机。面做好后服务员叫号再去取餐。",
    totalTime: 5,
    steps: [
      { label: "点餐", start: 0, end: 0.5, color: "#3b82f6" },
      { label: "玩手机", start: 0.5, end: 4.5, color: "#10b981" },
      { label: "取餐", start: 4.5, end: 5, color: "#f59e0b" },
    ],
  },
  {
    id: 3,
    title: "不断去窗口问",
    type: "非阻塞 + 轮询",
    description: "每隔一分钟过来问好了吗，没好就继续做别的事。每次查询都是非阻塞的。",
    totalTime: 5,
    steps: [
      { label: "问一次", start: 0, end: 0.3, color: "#8b5cf6" },
      { label: "做别的", start: 0.3, end: 1.3, color: "#10b981" },
      { label: "问一次", start: 1.3, end: 1.6, color: "#8b5cf6" },
      { label: "做别的", start: 1.6, end: 2.6, color: "#10b981" },
      { label: "问一次", start: 2.6, end: 2.9, color: "#8b5cf6" },
      { label: "做别的", start: 2.9, end: 3.9, color: "#10b981" },
      { label: "取餐", start: 3.9, end: 5, color: "#f59e0b" },
    ],
  },
  {
    id: 4,
    title: "一个厨师交替做三份菜",
    type: "并发",
    description: "厨师先切第一份菜，等待水烧开期间去处理第二份菜，再处理第三份菜。多任务交替推进。",
    totalTime: 6,
    steps: [
      { label: "切菜A", start: 0, end: 1, color: "#ef4444" },
      { label: "切菜B", start: 1, end: 2, color: "#3b82f6" },
      { label: "切菜C", start: 2, end: 3, color: "#10b981" },
      { label: "炒A", start: 3, end: 4, color: "#ef4444" },
      { label: "炒B", start: 4, end: 5, color: "#3b82f6" },
      { label: "炒C", start: 5, end: 6, color: "#10b981" },
    ],
  },
  {
    id: 5,
    title: "三个厨师同时做三份菜",
    type: "并行",
    description: "三个厨师各自占用一个灶台，同时工作。真正的同时执行。",
    totalTime: 3,
    steps: [
      { label: "厨师1做A", start: 0, end: 3, color: "#ef4444" },
      { label: "厨师2做B", start: 0, end: 3, color: "#3b82f6" },
      { label: "厨师3做C", start: 0, end: 3, color: "#10b981" },
    ],
  },
];

export function RestaurantAnalogy() {
  const [selected, setSelected] = useState<number>(0);
  const scenario = scenarios[selected];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Utensils className="w-5 h-5" />
        餐厅点餐类比
      </h3>
      <div className="flex flex-wrap gap-2 mb-6">
        {scenarios.map((s, i) => (
          <button
            key={s.id}
            onClick={() => setSelected(i)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              selected === i
                ? "bg-indigo-600 text-white shadow-md"
                : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700"
            }`}
          >
            情况 {s.id}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div
          key={selected}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg font-bold text-slate-900 dark:text-slate-100">{scenario.title}</span>
            <span className="px-2 py-0.5 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-xs rounded-full font-medium">
              {scenario.type}
            </span>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">{scenario.description}</p>
          <div className="relative h-32 bg-slate-50 dark:bg-slate-900 rounded-lg p-3 overflow-hidden">
            <div className="absolute left-0 right-0 bottom-6 h-px bg-slate-200 dark:bg-slate-700" />
            {Array.from({ length: Math.ceil(scenario.totalTime) + 1 }, (_, i) => (
              <div key={i} className="absolute bottom-0 text-[10px] text-slate-400" style={{ left: `${(i / scenario.totalTime) * 100}%` }}>
                {i}s
              </div>
            ))}
            {scenario.steps.map((step, i) => (
              <motion.div
                key={i}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: i * 0.15, duration: 0.4 }}
                className="absolute h-6 rounded flex items-center justify-center text-[10px] text-white font-medium"
                style={{
                  left: `${(step.start / scenario.totalTime) * 100}%`,
                  width: `${((step.end - step.start) / scenario.totalTime) * 100}%`,
                  backgroundColor: step.color,
                  top: `${(i % 3) * 28 + 12}px`,
                }}
              >
                {step.label}
              </motion.div>
            ))}
          </div>
          <div className="mt-3 flex items-center gap-2 text-sm">
            <Clock className="w-4 h-4 text-amber-500" />
            <span className="text-slate-700 dark:text-slate-300">总耗时：<strong>{scenario.totalTime} 秒</strong></span>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
