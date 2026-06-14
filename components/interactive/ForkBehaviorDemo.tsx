"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { GitFork, Play, RotateCcw } from "lucide-react";

export default function ForkBehaviorDemo() {
  const [hasForked, setHasForked] = useState(false);
  const [parentValue, setParentValue] = useState(100);
  const [childValue, setChildValue] = useState(100);

  const handleFork = () => {
    setHasForked(true);
    setParentValue(300);
    setChildValue(200);
  };

  const handleReset = () => {
    setHasForked(false);
    setParentValue(100);
    setChildValue(100);
  };

  const codeExample = `#include <stdio.h>
#include <unistd.h>

int main() {
    int x = 100;
    
    printf("Before fork: x=%d\\n", x);
    
    pid_t pid = fork();
    
    if (pid < 0) {
        perror("fork failed");
        return 1;
    } else if (pid == 0) {
        // 子进程
        x = 200;
        printf("Child: PID=%d, x=%d\\n", getpid(), x);
    } else {
        // 父进程
        x = 300;
        printf("Parent: PID=%d, Child PID=%d, x=%d\\n", 
               getpid(), pid, x);
    }
    
    printf("After fork: PID=%d, x=%d\\n", getpid(), x);
    
    return 0;
}`;

  const output = hasForked ? `Before fork: x=100
Parent: PID=1234, Child PID=1235, x=300
After fork: PID=1234, x=300
Child: PID=1235, x=200
After fork: PID=1235, x=200` : `Before fork: x=100
(等待 fork() 执行...)`;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <GitFork className="w-7 h-7 text-green-600" />
        fork() 行为演示
      </h3>

      {/* Control Buttons */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={handleFork}
          disabled={hasForked}
          className="px-6 py-3 rounded-lg font-semibold bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
        >
          <Play className="w-5 h-5" />
          执行 fork()
        </button>
        <button
          onClick={handleReset}
          className="px-6 py-3 rounded-lg font-semibold bg-slate-600 text-white hover:bg-slate-700 transition-all flex items-center gap-2"
        >
          <RotateCcw className="w-5 h-5" />
          重置
        </button>
      </div>

      {/* Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Before Fork */}
        {!hasForked && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-lg shadow-md p-6 lg:col-span-2"
          >
            <h4 className="font-bold text-slate-800 mb-4">fork() 执行前</h4>
            <div className="bg-blue-100 border-2 border-blue-400 rounded-lg p-4">
              <div className="font-bold text-blue-700 mb-2">进程 (PID 1234)</div>
              <div className="text-sm text-slate-700 space-y-1">
                <div className="flex items-center gap-2">
                  <div className="font-mono bg-blue-200 px-2 py-1 rounded">x = {parentValue}</div>
                  <span className="text-slate-600">变量 x 的值</span>
                </div>
                <div className="text-xs text-slate-600 mt-2">
                  只有一个进程在运行
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* After Fork */}
        {hasForked && (
          <>
            {/* Parent Process */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-lg shadow-md p-6"
            >
              <h4 className="font-bold text-slate-800 mb-4">父进程</h4>
              <div className="bg-green-100 border-2 border-green-400 rounded-lg p-4">
                <div className="font-bold text-green-700 mb-2">进程 (PID 1234)</div>
                <div className="text-sm text-slate-700 space-y-2">
                  <div>
                    <div className="text-xs text-slate-600 mb-1">fork() 返回值:</div>
                    <div className="font-mono bg-green-200 px-2 py-1 rounded inline-block">
                      pid = 1235
                    </div>
                    <span className="text-xs text-slate-600 ml-2">(子进程 PID)</span>
                  </div>
                  <div>
                    <div className="text-xs text-slate-600 mb-1">变量 x:</div>
                    <div className="font-mono bg-green-200 px-2 py-1 rounded inline-block">
                      x = {parentValue}
                    </div>
                  </div>
                  <div className="text-xs text-slate-600 mt-2">
                    ✓ 修改 x 不影响子进程
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Child Process */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-lg shadow-md p-6"
            >
              <h4 className="font-bold text-slate-800 mb-4">子进程</h4>
              <div className="bg-purple-100 border-2 border-purple-400 rounded-lg p-4">
                <div className="font-bold text-purple-700 mb-2">进程 (PID 1235)</div>
                <div className="text-sm text-slate-700 space-y-2">
                  <div>
                    <div className="text-xs text-slate-600 mb-1">fork() 返回值:</div>
                    <div className="font-mono bg-purple-200 px-2 py-1 rounded inline-block">
                      pid = 0
                    </div>
                    <span className="text-xs text-slate-600 ml-2">(标识子进程)</span>
                  </div>
                  <div>
                    <div className="text-xs text-slate-600 mb-1">变量 x:</div>
                    <div className="font-mono bg-purple-200 px-2 py-1 rounded inline-block">
                      x = {childValue}
                    </div>
                  </div>
                  <div className="text-xs text-slate-600 mt-2">
                    ✓ 修改 x 不影响父进程
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </div>

      {/* Code and Output */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Code */}
        <div className="bg-white rounded-lg shadow-md p-4">
          <h4 className="font-bold text-slate-800 mb-3">代码示例</h4>
          <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto">
            {codeExample}
          </pre>
        </div>

        {/* Output */}
        <div className="bg-white rounded-lg shadow-md p-4">
          <h4 className="font-bold text-slate-800 mb-3">运行输出</h4>
          <pre className="bg-slate-900 text-yellow-400 p-4 rounded-lg text-xs overflow-x-auto whitespace-pre-wrap">
            {output}
          </pre>
        </div>
      </div>

      {/* Key Points */}
      <div className="mt-6 bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
        <h5 className="font-bold text-slate-800 mb-2">关键点</h5>
        <ul className="text-sm text-slate-700 space-y-1">
          <li>• <strong>Before fork</strong> 只打印一次（fork 之前）</li>
          <li>• <strong>After fork</strong> 打印两次（父子进程各一次）</li>
          <li>• 父进程返回子进程 PID（1235），子进程返回 0</li>
          <li>• 父子进程的变量 x 互不影响（独立地址空间）</li>
          <li>• 子进程获得父进程的完整副本（代码、数据、堆、栈）</li>
        </ul>
      </div>
    </div>
  );
}
