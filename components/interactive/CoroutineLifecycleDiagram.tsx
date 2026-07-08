"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, ArrowRight, CircleDot } from "lucide-react";

interface LifecycleState {
  id: string;
  label: string;
  sublabel: string;
  color: string;
  bgColor: string;
}

const states: LifecycleState[] = [
  { id: "define", label: "async def 定义", sublabel: "定义协程函数", color: "text-slate-600", bgColor: "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600" },
  { id: "call", label: "调用函数", sublabel: "my_coro()", color: "text-blue-600", bgColor: "bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700" },
  { id: "object", label: "协程对象", sublabel: "<coroutine>", color: "text-purple-600", bgColor: "bg-purple-50 dark:bg-purple-900/30 border-purple-300 dark:border-purple-700" },
  { id: "await", label: "await / 事件循环", sublabel: "调度执行", color: "text-amber-600", bgColor: "bg-amber-50 dark:bg-amber-900/30 border-amber-300 dark:border-amber-700" },
  { id: "execute", label: "执行函数体", sublabel: "运行代码", color: "text-green-600", bgColor: "bg-green-50 dark:bg-green-900/30 border-green-300 dark:border-green-700" },
  { id: "result", label: "返回结果", sublabel: "StopIteration", color: "text-emerald-600", bgColor: "bg-emerald-50 dark:bg-emerald-900/30 border-emerald-300 dark:border-emerald-700" },
];

export function CoroutineLifecycleDiagram() {
  const [activeIdx, setActiveIdx] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const play = () => {
    setActiveIdx(0);
    setIsPlaying(true);
  };

  useEffect(() => {
    if (isPlaying && activeIdx < states.length - 1) {
      timerRef.current = setTimeout(() => {
        setActiveIdx((prev) => prev + 1);
      }, 1200);
    } else if (activeIdx >= states.length - 1) {
      setIsPlaying(false);
    }
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [isPlaying, activeIdx]);

  const reset = () => {
    setActiveIdx(-1);
    setIsPlaying(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <CircleDot className="w-5 h-5 text-purple-500" />
        协程生命周期
      </h3>

      <div className="flex gap-3 mb-6">
        <button onClick={play} disabled={isPlaying}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> {isPlaying ? "播放中..." : "播放动画"}
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg">
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Lifecycle Steps */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 mb-6">
        <div className="flex flex-wrap items-center justify-center gap-2">
          {states.map((state, i) => (
            <React.Fragment key={state.id}>
              <motion.div
                animate={{
                  scale: activeIdx === i ? 1.05 : 1,
                  opacity: activeIdx >= 0 && i > activeIdx ? 0.4 : 1,
                }}
                className={`px-4 py-3 rounded-xl border-2 text-center min-w-[120px] transition-colors ${
                  activeIdx === i ? state.bgColor : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700"
                }`}>
                <div className={`text-sm font-bold ${activeIdx === i ? state.color : "text-slate-600 dark:text-slate-400"}`}>
                  {state.label}
                </div>
                <div className="text-xs text-slate-500 mt-1">{state.sublabel}</div>
              </motion.div>
              {i < states.length - 1 && (
                <ArrowRight className={`w-5 h-5 ${activeIdx > i ? "text-indigo-500" : "text-slate-300 dark:text-slate-600"}`} />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Detail Panel */}
      <AnimatePresence mode="wait">
        {activeIdx >= 0 && activeIdx < states.length && (
          <motion.div key={activeIdx} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
            className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className={`font-bold mb-2 ${states[activeIdx].color}`}>
                  阶段 {activeIdx + 1}: {states[activeIdx].label}
                </h4>
                <div className="text-sm text-slate-600 dark:text-slate-400 space-y-2">
                  {activeIdx === 0 && <p>使用 <code>async def</code> 关键字定义协程函数。此时函数还未被调用。</p>}
                  {activeIdx === 1 && <p>像普通函数一样调用 <code>my_coro()</code>，但不会执行函数体。</p>}
                  {activeIdx === 2 && <p>返回一个协程对象 <code>&lt;coroutine object&gt;</code>，类型为 <code>coroutine</code>。</p>}
                  {activeIdx === 3 && <p>通过 <code>await</code> 或事件循环调度，将协程交给事件循环管理。</p>}
                  {activeIdx === 4 && <p>事件循环开始执行协程函数体，遇到 <code>await</code> 时可能暂停。</p>}
                  {activeIdx === 5 && <p>函数执行完毕，抛出 <code>StopIteration</code> 异常，携带返回值。</p>}
                </div>
              </div>
              <pre className="bg-slate-900 text-green-400 p-3 rounded-lg text-xs overflow-x-auto">
                {activeIdx === 0 && `async def my_coro():\n    await asyncio.sleep(1)\n    return "result"`}
                {activeIdx === 1 && `# 调用协程函数\nobj = my_coro()`}
                {activeIdx === 2 && `# 返回协程对象\nprint(type(obj))\n# <class 'coroutine'>`}
                {activeIdx === 3 && `# 通过 await 调度\nresult = await obj\n# 或: asyncio.run(obj)`}
                {activeIdx === 4 && `# 事件循环执行函数体\n# 执行到 await sleep(1) 时暂停\n# 1秒后恢复执行`}
                {activeIdx === 5 && `# 返回结果\nprint(result)  # "result"\n# 协程对象变为 done 状态`}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {activeIdx === -1 && (
        <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-dashed border-slate-300 dark:border-slate-600 p-8 text-center text-slate-500">
          点击"播放动画"查看协程的完整生命周期
        </div>
      )}
    </div>
  );
}
