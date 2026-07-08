"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, ArrowRight, Zap, Clock } from "lucide-react";

export function CoroutineVsFunction() {
  const [funcResult, setFuncResult] = useState<string | null>(null);
  const [coroutineResult, setCoroutineResult] = useState<string | null>(null);
  const [funcRunning, setFuncRunning] = useState(false);
  const [coroutineRunning, setCoroutineRunning] = useState(false);

  const callFunction = () => {
    setFuncRunning(true);
    setFuncResult(null);
    setTimeout(() => {
      setFuncResult("42");
      setFuncRunning(false);
    }, 800);
  };

  const callCoroutine = () => {
    setCoroutineRunning(true);
    setCoroutineResult(null);
    setTimeout(() => {
      setCoroutineResult("<coroutine object my_coro at 0x10a3b2c40>");
      setCoroutineRunning(false);
    }, 800);
  };

  const reset = () => {
    setFuncResult(null);
    setCoroutineResult(null);
    setFuncRunning(false);
    setCoroutineRunning(false);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Zap className="w-5 h-5 text-amber-500" />
        普通函数 vs 协程函数
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
        点击按钮调用两种函数，观察返回值的本质区别
      </p>

      <div className="flex gap-3 mb-6">
        <button onClick={callFunction} disabled={funcRunning}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> 调用普通函数
        </button>
        <button onClick={callCoroutine} disabled={coroutineRunning}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> 调用协程函数
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg">
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Regular Function */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-blue-200 dark:border-blue-800 p-5">
          <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4" /> 普通函数
          </h4>
          <pre className="bg-slate-900 text-green-400 p-3 rounded-lg text-xs mb-4 overflow-x-auto">
{`def my_func():
    return 42

result = my_func()
print(type(result))
print(result)`}
          </pre>
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
              <ArrowRight className="w-4 h-4" />
              <span>类型: <code className="bg-slate-100 dark:bg-slate-700 px-1 rounded">&lt;class &apos;int&apos;&gt;</code></span>
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
              <ArrowRight className="w-4 h-4" />
              <span>立即执行，直接返回结果</span>
            </div>
          </div>
          <AnimatePresence>
            {funcRunning && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="mt-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg p-3 text-sm text-blue-700 dark:text-blue-300">
                函数体立即执行...
              </motion.div>
            )}
            {funcResult && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="mt-4 bg-green-50 dark:bg-green-900/30 rounded-lg p-3">
                <div className="text-xs text-slate-500 mb-1">输出:</div>
                <div className="text-sm font-mono text-green-700 dark:text-green-300">
                  &lt;class &apos;int&apos;&gt;<br />{funcResult}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Coroutine Function */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-purple-200 dark:border-purple-800 p-5">
          <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3 flex items-center gap-2">
            <Clock className="w-4 h-4" /> 协程函数 (async def)
          </h4>
          <pre className="bg-slate-900 text-green-400 p-3 rounded-lg text-xs mb-4 overflow-x-auto">
{`async def my_coro():
    return 42

result = my_coro()
print(type(result))
print(result)`}
          </pre>
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
              <ArrowRight className="w-4 h-4" />
              <span>类型: <code className="bg-slate-100 dark:bg-slate-700 px-1 rounded">&lt;class &apos;coroutine&apos;&gt;</code></span>
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
              <ArrowRight className="w-4 h-4" />
              <span>不执行函数体，返回协程对象</span>
            </div>
          </div>
          <AnimatePresence>
            {coroutineRunning && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="mt-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg p-3 text-sm text-purple-700 dark:text-purple-300">
                创建协程对象（函数体未执行）...
              </motion.div>
            )}
            {coroutineResult && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="mt-4 bg-amber-50 dark:bg-amber-900/30 rounded-lg p-3">
                <div className="text-xs text-slate-500 mb-1">输出:</div>
                <div className="text-sm font-mono text-amber-700 dark:text-amber-300">
                  &lt;class &apos;coroutine&apos;&gt;<br />{coroutineResult}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <div className="mt-6 bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 dark:text-amber-300 mb-2">关键区别</h4>
        <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-1 list-disc list-inside">
          <li><strong>普通函数</strong>：调用时立即执行函数体，返回计算结果</li>
          <li><strong>协程函数</strong>：调用时仅创建协程对象，需要 <code>await</code> 或事件循环才会执行</li>
          <li>协程对象是 awaitable 类型，可以用 <code>await</code> 获取结果</li>
        </ul>
      </div>
    </div>
  );
}
