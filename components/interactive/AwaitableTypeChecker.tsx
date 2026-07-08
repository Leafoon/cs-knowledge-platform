"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle, XCircle, Search, RotateCcw } from "lucide-react";

interface Expression {
  id: number;
  code: string;
  awaitable: boolean;
  type: string;
  explanation: string;
}

const expressions: Expression[] = [
  { id: 1, code: "asyncio.sleep(2)", awaitable: true, type: "coroutine", explanation: "asyncio.sleep() 是协程函数，调用后返回协程对象，是 awaitable 的" },
  { id: 2, code: "time.sleep(2)", awaitable: false, type: "NoneType", explanation: "time.sleep() 是普通函数，返回 None，不是 awaitable 的" },
  { id: 3, code: "print('hello')", awaitable: false, type: "NoneType", explanation: "print() 是普通内置函数，返回 None，不是 awaitable 的" },
  { id: 4, code: "123", awaitable: false, type: "int", explanation: "整数字面量不是 awaitable 的，直接返回自身" },
  { id: 5, code: "async def f(): pass\nf()", awaitable: true, type: "coroutine", explanation: "调用 async def 函数返回协程对象，协程对象是 awaitable 的" },
  { id: 6, code: "asyncio.gather(coro1, coro2)", awaitable: true, type: "Future", explanation: "asyncio.gather() 返回一个 Future 对象，是 awaitable 的" },
  { id: 7, code: "asyncio.ensure_future(coro)", awaitable: true, type: "Task", explanation: "ensure_future() 返回一个 Task 对象（Future 子类），是 awaitable 的" },
  { id: 8, code: "'hello world'", awaitable: false, type: "str", explanation: "字符串不是 awaitable 的" },
  { id: 9, code: "[1, 2, 3]", awaitable: false, type: "list", explanation: "列表不是 awaitable 的" },
  { id: 10, code: "asyncio.Queue()", awaitable: false, type: "Queue", explanation: "Queue 对象本身不是 awaitable 的，但它的 get() 方法返回协程" },
];

export function AwaitableTypeChecker() {
  const [results, setResults] = useState<Record<number, boolean | null>>({});
  const [revealed, setRevealed] = useState<Set<number>>(new Set());

  const checkExpression = (expr: Expression) => {
    setRevealed((prev) => new Set([...prev, expr.id]));
  };

  const reset = () => {
    setResults({});
    setRevealed(new Set());
  };

  const correct = expressions.filter((e) => revealed.has(e.id)).length;

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Search className="w-5 h-5 text-emerald-500" />
        Awaitable 类型检查器
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        点击表达式检查它是否是 awaitable 的。已检查: {correct}/{expressions.length}
      </p>

      <div className="flex gap-3 mb-6">
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg flex items-center gap-2 text-sm">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
        <button onClick={() => { const all = new Set(expressions.map((e) => e.id)); setRevealed(all); }}
          className="px-3 py-2 bg-indigo-600 text-white rounded-lg text-sm">
          全部检查
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {expressions.map((expr) => {
          const isRevealed = revealed.has(expr.id);
          return (
            <motion.div key={expr.id} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              onClick={() => !isRevealed && checkExpression(expr)}
              className={`rounded-xl border p-4 cursor-pointer transition-all ${
                isRevealed
                  ? expr.awaitable
                    ? "bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700"
                    : "bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700"
                  : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:border-indigo-300 dark:hover:border-indigo-700"
              }`}>
              <code className="text-sm font-mono text-slate-800 dark:text-slate-200 block mb-2">
                {expr.code}
              </code>
              {isRevealed ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <div className="flex items-center gap-2 mb-2">
                    {expr.awaitable ? (
                      <CheckCircle className="w-5 h-5 text-green-600" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-600" />
                    )}
                    <span className={`font-bold text-sm ${expr.awaitable ? "text-green-700 dark:text-green-300" : "text-red-700 dark:text-red-300"}`}>
                      {expr.awaitable ? "Awaitable" : "Not Awaitable"}
                    </span>
                    <span className="text-xs bg-slate-200 dark:bg-slate-700 px-2 py-0.5 rounded">
                      {expr.type}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 dark:text-slate-400">{expr.explanation}</p>
                </motion.div>
              ) : (
                <div className="text-xs text-slate-400 flex items-center gap-1">
                  <Search className="w-3 h-3" /> 点击检查
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 p-4 rounded">
        <h4 className="font-bold text-blue-800 dark:text-blue-300 mb-2">awaitable 的三种类型</h4>
        <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-1 list-disc list-inside">
          <li><strong>协程 (Coroutine)</strong>：async def 函数调用后的返回值</li>
          <li><strong>Task</strong>：通过 asyncio.create_task() 创建的任务对象</li>
          <li><strong>Future</strong>：底层的异步结果容器，Task 的父类</li>
        </ul>
      </div>
    </div>
  );
}
