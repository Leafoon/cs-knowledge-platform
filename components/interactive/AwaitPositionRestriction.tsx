"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, XCircle, HelpCircle, RotateCcw } from "lucide-react";

interface QuizItem {
  id: number;
  code: string;
  valid: boolean;
  explanation: string;
  errorMsg?: string;
}

const quizItems: QuizItem[] = [
  {
    id: 1,
    code: `async def main():
    result = await asyncio.sleep(1)`,
    valid: true,
    explanation: "await 可以直接出现在 async def 函数体中",
  },
  {
    id: 2,
    code: `def main():
    result = await asyncio.sleep(1)`,
    valid: false,
    explanation: "await 不能出现在普通函数中",
    errorMsg: "SyntaxError: 'await' outside async function",
  },
  {
    id: 3,
    code: `async def main():
    if True:
        result = await fetch_data()`,
    valid: true,
    explanation: "await 可以出现在 async 函数内的 if 语句中",
  },
  {
    id: 4,
    code: `async def main():
    x = [await i for i in range(3)]`,
    valid: true,
    explanation: "await 可以出现在 async 函数内的列表推导式中（Python 3.6+）",
  },
  {
    id: 5,
    code: `x = [await i for i in range(3)]`,
    valid: false,
    explanation: "顶层代码中不能使用 await",
    errorMsg: "SyntaxError: 'await' outside function",
  },
  {
    id: 6,
    code: `async def main():
    def inner():
        return await fetch_data()`,
    valid: false,
    explanation: "await 不能出现在嵌套的普通函数中",
    errorMsg: "SyntaxError: 'await' outside async function",
  },
  {
    id: 7,
    code: `async def main():
    async def inner():
        return await fetch_data()
    result = await inner()`,
    valid: true,
    explanation: "await 可以出现在嵌套的 async 函数中",
  },
  {
    id: 8,
    code: `async def main():
    for i in range(10):
        await process(i)`,
    valid: true,
    explanation: "await 可以出现在 async 函数内的 for 循环中",
  },
];

export function AwaitPositionRestriction() {
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState<(boolean | null)[]>(Array(quizItems.length).fill(null));
  const [showResult, setShowResult] = useState(false);

  const item = quizItems[current];

  const answer = (choice: boolean) => {
    const newAnswers = [...answers];
    newAnswers[current] = choice;
    setAnswers(newAnswers);
    setShowResult(true);
  };

  const next = () => {
    if (current < quizItems.length - 1) {
      setCurrent(current + 1);
      setShowResult(false);
    }
  };

  const prev = () => {
    if (current > 0) {
      setCurrent(current - 1);
      setShowResult(false);
    }
  };

  const reset = () => {
    setCurrent(0);
    setAnswers(Array(quizItems.length).fill(null));
    setShowResult(false);
  };

  const correct = answers.filter((a, i) => a === quizItems[i].valid).length;
  const answered = answers.filter((a) => a !== null).length;

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <HelpCircle className="w-5 h-5 text-purple-500" />
        await 位置限制测验
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        判断以下代码中 await 的使用是否合法
      </p>

      <div className="flex gap-3 mb-4 items-center">
        <span className="text-sm text-slate-500">进度: {current + 1}/{quizItems.length}</span>
        <span className="text-sm text-slate-500">正确: {correct}/{answered}</span>
        <button onClick={reset} className="ml-auto px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg">
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Progress dots */}
      <div className="flex gap-1 mb-6">
        {quizItems.map((_, i) => (
          <button key={i} onClick={() => { setCurrent(i); setShowResult(false); }}
            className={`w-8 h-8 rounded-full text-xs font-bold flex items-center justify-center ${
              i === current ? "bg-indigo-600 text-white" :
              answers[i] === null ? "bg-slate-200 dark:bg-slate-700 text-slate-500" :
              answers[i] === quizItems[i].valid ? "bg-green-500 text-white" : "bg-red-500 text-white"
            }`}>
            {i + 1}
          </button>
        ))}
      </div>

      {/* Code Card */}
      <motion.div key={current} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
          {item.code}
        </pre>

        {!showResult ? (
          <div className="flex gap-4 mt-4">
            <button onClick={() => answer(true)}
              className="flex-1 py-3 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-lg font-medium hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors flex items-center justify-center gap-2">
              <CheckCircle className="w-5 h-5" /> 合法 (Valid)
            </button>
            <button onClick={() => answer(false)}
              className="flex-1 py-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg font-medium hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors flex items-center justify-center gap-2">
              <XCircle className="w-5 h-5" /> 非法 (Invalid)
            </button>
          </div>
        ) : (
          <AnimatePresence>
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className={`mt-4 p-4 rounded-lg ${
                answers[current] === item.valid
                  ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
                  : "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
              }`}>
              <div className="flex items-center gap-2 mb-2">
                {answers[current] === item.valid ? (
                  <CheckCircle className="w-5 h-5 text-green-600" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-600" />
                )}
                <span className={`font-bold ${answers[current] === item.valid ? "text-green-700 dark:text-green-300" : "text-red-700 dark:text-red-300"}`}>
                  {answers[current] === item.valid ? "正确!" : "错误!"}
                </span>
              </div>
              <p className="text-sm text-slate-700 dark:text-slate-300">{item.explanation}</p>
              {item.errorMsg && (
                <pre className="mt-2 bg-slate-900 text-red-400 p-2 rounded text-xs">{item.errorMsg}</pre>
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </motion.div>

      {/* Navigation */}
      <div className="flex justify-between">
        <button onClick={prev} disabled={current === 0}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg disabled:opacity-50">
          上一题
        </button>
        <button onClick={next} disabled={current === quizItems.length - 1}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg disabled:opacity-50">
          下一题
        </button>
      </div>
    </div>
  );
}
