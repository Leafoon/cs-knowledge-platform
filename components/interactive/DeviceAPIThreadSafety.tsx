"use client";

import { useState } from "react";

export function DeviceAPIThreadSafety() {
  const [threadCount, setThreadCount] = useState(2);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);

  const simulate = () => {
    setRunning(true);
    setLogs([]);
    const newLogs: string[] = [];

    for (let t = 0; t < threadCount; t++) {
      newLogs.push(`[Thread-${t}] 请求 AllocMemory...`);
    }
    newLogs.push("[Mutex] 获取锁 → Thread-0 获得锁");
    newLogs.push("[Thread-0] ✅ AllocMemory 完成");
    newLogs.push("[Mutex] 释放锁 → Thread-1 获得锁");
    newLogs.push("[Thread-1] ✅ AllocMemory 完成");
    newLogs.push("[All] 所有线程同步完成");

    let i = 0;
    const interval = setInterval(() => {
      if (i < newLogs.length) {
        setLogs((prev) => [...prev, newLogs[i]]);
        i++;
      } else {
        clearInterval(interval);
        setRunning(false);
      }
    }, 500);
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        DeviceAPI 线程安全模型
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700 mb-4">
            <h4 className="font-bold text-sm text-slate-700 dark:text-slate-300 mb-3">
              同步机制
            </h4>
            <div className="space-y-2">
              <div className="flex items-center gap-2 p-2 bg-violet-50 dark:bg-violet-900/20 rounded">
                <span className="w-3 h-3 rounded-full bg-violet-500" />
                <span className="text-xs text-slate-700 dark:text-slate-300">
                  全局 Mutex 保护 Alloc/Free
                </span>
              </div>
              <div className="flex items-center gap-2 p-2 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                <span className="w-3 h-3 rounded-full bg-indigo-500" />
                <span className="text-xs text-slate-700 dark:text-slate-300">
                  Stream 级别独立，无需锁
                </span>
              </div>
              <div className="flex items-center gap-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                <span className="w-3 h-3 rounded-full bg-blue-500" />
                <span className="text-xs text-slate-700 dark:text-slate-300">
                  StreamSync 提供线程屏障
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 mb-4">
            <label className="text-sm text-slate-700 dark:text-slate-300">
              线程数:
            </label>
            <input
              type="range"
              min={1}
              max={4}
              value={threadCount}
              onChange={(e) => setThreadCount(Number(e.target.value))}
              className="flex-1"
            />
            <span className="text-sm font-bold text-indigo-600 dark:text-indigo-400">
              {threadCount}
            </span>
          </div>

          <button
            onClick={simulate}
            disabled={running}
            className="w-full px-4 py-2 bg-indigo-500 hover:bg-indigo-600 disabled:bg-slate-400 text-white rounded-lg font-bold text-sm transition-colors"
          >
            {running ? "模拟中..." : "模拟多线程访问"}
          </button>
        </div>

        <div className="bg-slate-900 rounded-xl p-4 font-mono text-xs max-h-64 overflow-y-auto">
          {logs.length === 0 ? (
            <p className="text-slate-500">点击按钮查看线程交互...</p>
          ) : (
            logs.map((log, i) => (
              <div
                key={i}
                className={`py-0.5 ${
                  log.includes("✅")
                    ? "text-green-400"
                    : log.includes("Mutex")
                    ? "text-amber-400"
                    : "text-slate-300"
                }`}
              >
                {log}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
