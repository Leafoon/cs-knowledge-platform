"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { GitBranch, Play } from "lucide-react";

export default function ForkReturnValueDemo() {
  const [isExecuted, setIsExecuted] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const handleExecute = () => {
    setIsExecuted(true);
    setTimeout(() => setShowExplanation(true), 1000);
  };

  const handleReset = () => {
    setIsExecuted(false);
    setShowExplanation(false);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <GitBranch className="w-7 h-7 text-purple-600" />
        fork() 返回值演示
      </h3>

      {/* Code Example */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <h4 className="font-bold text-slate-800 mb-3">示例代码</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
{`#include <stdio.h>
#include <unistd.h>

int main() {
    pid_t pid = fork();  // 创建子进程
    
    if (pid < 0) {
        // fork() 失败
        perror("fork failed");
        return 1;
    } else if (pid == 0) {
        // 子进程：fork() 返回 0
        printf("子进程: pid = %d, fork() 返回 %d\\n", getpid(), pid);
    } else {
        // 父进程：fork() 返回子进程的 PID
        printf("父进程: pid = %d, fork() 返回 %d\\n", getpid(), pid);
    }
    return 0;
}`}
        </pre>
      </div>

      {/* Execute Button */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={handleExecute}
          disabled={isExecuted}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play className="w-5 h-5" />
          执行 fork()
        </button>
        {isExecuted && (
          <button
            onClick={handleReset}
            className="px-6 py-3 bg-slate-600 text-white rounded-lg font-semibold hover:bg-slate-700"
          >
            重置
          </button>
        )}
      </div>

      {/* Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Before fork() */}
        <motion.div
          initial={{ opacity: 1 }}
          animate={{ opacity: isExecuted ? 0.5 : 1 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h4 className="font-bold text-slate-800 mb-4 text-center">fork() 之前</h4>
          <div className="flex justify-center">
            <div className="bg-blue-100 border-2 border-blue-400 rounded-lg p-6 w-48">
              <div className="text-center font-bold text-blue-700 mb-2">父进程</div>
              <div className="text-sm text-slate-700">PID: 1234</div>
              <div className="text-sm text-slate-700 mt-2">
                <code className="bg-slate-200 px-2 py-1 rounded">pid_t pid</code>
              </div>
            </div>
          </div>
        </motion.div>

        {/* After fork() */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: isExecuted ? 1 : 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h4 className="font-bold text-slate-800 mb-4 text-center">fork() 之后</h4>
          <div className="flex flex-col gap-4">
            {/* Parent */}
            <motion.div
              initial={{ x: -50, opacity: 0 }}
              animate={isExecuted ? { x: 0, opacity: 1 } : {}}
              transition={{ delay: 0.2 }}
              className="bg-green-100 border-2 border-green-400 rounded-lg p-4"
            >
              <div className="font-bold text-green-700 mb-2">父进程</div>
              <div className="text-sm text-slate-700">PID: 1234</div>
              <div className="text-sm font-mono bg-white p-2 rounded mt-2 border border-green-300">
                pid = <span className="text-purple-600 font-bold">1235</span>
              </div>
              <div className="text-xs text-slate-600 mt-1">↑ 子进程的 PID</div>
            </motion.div>

            {/* Child */}
            <motion.div
              initial={{ x: 50, opacity: 0 }}
              animate={isExecuted ? { x: 0, opacity: 1 } : {}}
              transition={{ delay: 0.4 }}
              className="bg-purple-100 border-2 border-purple-400 rounded-lg p-4"
            >
              <div className="font-bold text-purple-700 mb-2">子进程</div>
              <div className="text-sm text-slate-700">PID: 1235</div>
              <div className="text-sm font-mono bg-white p-2 rounded mt-2 border border-purple-300">
                pid = <span className="text-blue-600 font-bold">0</span>
              </div>
              <div className="text-xs text-slate-600 mt-1">↑ 固定返回 0</div>
            </motion.div>
          </div>
        </motion.div>
      </div>

      {/* Output */}
      {isExecuted && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-4 mb-6"
        >
          <h4 className="font-bold text-slate-800 mb-3">程序输出</h4>
          <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm">
{`父进程: pid = 1234, fork() 返回 1235
子进程: pid = 1235, fork() 返回 0`}
          </pre>
        </motion.div>
      )}

      {/* Explanation */}
      {showExplanation && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h4 className="font-bold text-slate-800 mb-4">返回值详解</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
              <div className="font-bold text-red-700 mb-2">pid &lt; 0</div>
              <div className="text-sm text-slate-700 mb-2">fork() 失败</div>
              <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
                <li>内存不足</li>
                <li>进程数限制</li>
                <li>权限问题</li>
              </ul>
            </div>

            <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
              <div className="font-bold text-purple-700 mb-2">pid == 0</div>
              <div className="text-sm text-slate-700 mb-2">子进程</div>
              <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
                <li>子进程执行此分支</li>
                <li>始终返回 0</li>
                <li>可用 getpid() 获取自身 PID</li>
              </ul>
            </div>

            <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
              <div className="font-bold text-green-700 mb-2">pid &gt; 0</div>
              <div className="text-sm text-slate-700 mb-2">父进程</div>
              <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
                <li>父进程执行此分支</li>
                <li>返回子进程 PID</li>
                <li>可用于 wait() / waitpid()</li>
              </ul>
            </div>
          </div>

          {/* Why Design */}
          <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
            <h5 className="font-bold text-amber-800 mb-2">为什么这样设计？</h5>
            <ul className="text-sm text-slate-700 space-y-2">
              <li>
                <strong>子进程返回 0：</strong>子进程不需要知道自己的 PID（可通过 getpid() 获取），
                而 0 是一个不可能的 PID 值（PID 从 1 开始），因此可作为特殊标识。
              </li>
              <li>
                <strong>父进程返回子 PID：</strong>父进程通常需要知道子进程的 PID 来管理它（wait、kill 等），
                而父进程可能创建多个子进程，需要区分它们。
              </li>
              <li>
                <strong>一次调用，两次返回：</strong>fork() 在父进程中调用，但返回两次——一次在父进程，一次在子进程，
                体现了"创建了一个几乎完全相同的进程副本"这一语义。
              </li>
            </ul>
          </div>
        </motion.div>
      )}

      {/* Pattern */}
      <div className="mt-6 bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
        <h5 className="font-bold text-blue-800 mb-2">典型使用模式</h5>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto">
{`pid_t pid = fork();
if (pid == 0) {
    // 子进程逻辑
    execve("/bin/ls", ...);  // 执行新程序
    exit(0);
} else if (pid > 0) {
    // 父进程逻辑
    wait(NULL);  // 等待子进程结束
} else {
    // 错误处理
    perror("fork");
}`}
        </pre>
      </div>
    </div>
  );
}
