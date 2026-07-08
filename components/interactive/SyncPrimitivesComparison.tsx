"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Lock, Users, Bell, Package, GitBranch, Shield, ChevronDown, ChevronUp, Code2 } from "lucide-react";

interface Primitive {
  name: string;
  icon: React.ReactNode;
  color: string;
  useCase: string;
  methods: string;
  code: string;
  description: string;
}

const PRIMITIVES: Primitive[] = [
  {
    name: "Lock (互斥锁)",
    icon: <Lock className="w-4 h-4" />,
    color: "blue",
    useCase: "保护共享资源，同一时刻只允许一个线程访问",
    methods: "acquire() / release()",
    code: `async with lock:\n    # 临界区\n    shared_resource += 1`,
    description: "最基本的同步原语。确保同一时刻只有一个协程可以执行临界区代码，防止竞态条件。",
  },
  {
    name: "Semaphore (信号量)",
    icon: <Users className="w-4 h-4" />,
    color: "purple",
    useCase: "限制同时访问资源的协程数量",
    methods: "acquire() / release()",
    code: `sem = asyncio.Semaphore(3)\nasync with sem:\n    # 最多3个协程同时执行\n    await access_resource()`,
    description: "带计数器的锁。允许最多 N 个协程同时访问资源，常用于连接池、速率限制。",
  },
  {
    name: "Event (事件)",
    icon: <Bell className="w-4 h-4" />,
    color: "amber",
    useCase: "通知一个或多个协程某个条件已满足",
    methods: "set() / clear() / wait()",
    code: `event = asyncio.Event()\n# 生产者\nevent.set()\n# 消费者\nawait event.wait()`,
    description: "简单的标志位同步。一个协程设置事件，其他协程等待事件触发。适合一次性通知场景。",
  },
  {
    name: "Queue (队列)",
    icon: <Package className="w-4 h-4" />,
    color: "indigo",
    useCase: "协程间安全传递数据",
    methods: "put() / get() / task_done()",
    code: `queue = asyncio.Queue(maxsize=10)\nawait queue.put(item)\nitem = await queue.get()`,
    description: "线程安全的数据通道。支持有界队列（满时阻塞 put）和任务跟踪（task_done/join）。",
  },
  {
    name: "Condition (条件变量)",
    icon: <GitBranch className="w-4 h-4" />,
    color: "green",
    useCase: "等待特定条件成立后继续执行",
    methods: "wait() / notify() / notify_all()",
    code: `cond = asyncio.Condition()\nasync with cond:\n    await cond.wait()\n    # 条件满足后继续`,
    description: "与锁结合使用的高级同步原语。协程可以释放锁并等待条件通知，适合生产者-消费者的精确控制。",
  },
  {
    name: "Barrier (屏障)",
    icon: <Shield className="w-4 h-4" />,
    color: "rose",
    useCase: "所有协程到达同步点后一起继续",
    methods: "wait()",
    code: `barrier = asyncio.Barrier(3)\n# 每个协程\nawait barrier.wait()\n# 所有协程同时继续`,
    description: "多协程同步点。所有参与者必须都到达屏障处才能继续执行，适合并行计算的阶段性同步。",
  },
];

const colorMap: Record<string, { bg: string; border: string; text: string; light: string }> = {
  blue: { bg: "bg-blue-500", border: "border-blue-300 dark:border-blue-700", text: "text-blue-700 dark:text-blue-300", light: "bg-blue-50 dark:bg-blue-900/20" },
  purple: { bg: "bg-purple-500", border: "border-purple-300 dark:border-purple-700", text: "text-purple-700 dark:text-purple-300", light: "bg-purple-50 dark:bg-purple-900/20" },
  amber: { bg: "bg-amber-500", border: "border-amber-300 dark:border-amber-700", text: "text-amber-700 dark:text-amber-300", light: "bg-amber-50 dark:bg-amber-900/20" },
  indigo: { bg: "bg-indigo-500", border: "border-indigo-300 dark:border-indigo-700", text: "text-indigo-700 dark:text-indigo-300", light: "bg-indigo-50 dark:bg-indigo-900/20" },
  green: { bg: "bg-green-500", border: "border-green-300 dark:border-green-700", text: "text-green-700 dark:text-green-300", light: "bg-green-50 dark:bg-green-900/20" },
  rose: { bg: "bg-rose-500", border: "border-rose-300 dark:border-rose-700", text: "text-rose-700 dark:text-rose-300", light: "bg-rose-50 dark:bg-rose-900/20" },
};

export function SyncPrimitivesComparison() {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Code2 className="w-5 h-5 text-teal-500" />
        同步原语对比
      </h3>

      <div className="space-y-3">
        {PRIMITIVES.map((p, i) => {
          const c = colorMap[p.color];
          const isOpen = expanded === i;
          return (
            <motion.div key={p.name} layout className={`rounded-xl border ${c.border} overflow-hidden`}>
              <button
                onClick={() => setExpanded(isOpen ? null : i)}
                className={`w-full flex items-center gap-3 p-4 text-left hover:${c.light} transition-colors`}
              >
                <div className={`w-8 h-8 rounded-lg ${c.bg} flex items-center justify-center text-white`}>
                  {p.icon}
                </div>
                <div className="flex-1">
                  <div className={`font-semibold text-sm ${c.text}`}>{p.name}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">{p.useCase}</div>
                </div>
                <span className="text-xs font-mono text-slate-400 mr-2">{p.methods}</span>
                {isOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              <AnimatePresence>
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 space-y-3">
                      <p className="text-sm text-slate-600 dark:text-slate-400">{p.description}</p>
                      <div className="rounded-lg bg-slate-900 dark:bg-slate-950 p-3">
                        <pre className="text-xs text-green-400 font-mono whitespace-pre-wrap">{p.code}</pre>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
