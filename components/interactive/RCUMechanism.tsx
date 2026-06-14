"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BookOpen, Edit3, Clock, Trash2, Play, RotateCcw, Pause, ArrowRight } from "lucide-react";

type Phase = "idle" | "readers-reading" | "writer-copy" | "writer-modify" | "writer-swap" | "grace-period" | "reclaim";

interface Reader {
  id: number;
  version: "v1" | "v2" | null;
  inCriticalSection: boolean;
}

export default function RCUMechanism() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [isRunning, setIsRunning] = useState(false);
  const [version, setVersion] = useState<"v1" | "v2">("v1");
  const [pointerTarget, setPointerTarget] = useState<"v1" | "v2">("v1");
  const [readers, setReaders] = useState<Reader[]>([
    { id: 0, version: null, inCriticalSection: false },
    { id: 1, version: null, inCriticalSection: false },
    { id: 2, version: null, inCriticalSection: false },
  ]);
  const [gracePeriodProgress, setGracePeriodProgress] = useState(0);
  const [v1Alive, setV1Alive] = useState(true);
  const [v2Alive, setV2Alive] = useState(true);
  const [log, setLog] = useState<string[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const addLog = (msg: string) => setLog(prev => [...prev.slice(-6), msg]);

  const reset = () => {
    setIsRunning(false);
    setPhase("idle");
    setVersion("v1");
    setPointerTarget("v1");
    setReaders([
      { id: 0, version: null, inCriticalSection: false },
      { id: 1, version: null, inCriticalSection: false },
      { id: 2, version: null, inCriticalSection: false },
    ]);
    setGracePeriodProgress(0);
    setV1Alive(true);
    setV2Alive(false);
    setLog([]);
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  useEffect(() => {
    if (!isRunning) return;

    const advance = () => {
      setPhase(current => {
        switch (current) {
          case "idle":
            addLog("📖 读者进入 RCU 临界区");
            setReaders(prev => prev.map(r => ({ ...r, inCriticalSection: true, version: "v1" as const })));
            timerRef.current = setTimeout(advance, 1200);
            return "readers-reading";

          case "readers-reading":
            addLog("📝 写者: 复制数据 v1 → v2");
            setV2Alive(true);
            timerRef.current = setTimeout(advance, 1200);
            return "writer-copy";

          case "writer-copy":
            addLog("✏️ 写者: 修改副本 v2");
            setVersion("v2");
            timerRef.current = setTimeout(advance, 1200);
            return "writer-modify";

          case "writer-modify":
            addLog("🔄 写者: 原子替换指针 → v2");
            setPointerTarget("v2");
            setReaders(prev => {
              const newReaders = [...prev];
              if (!newReaders[0].inCriticalSection) {
                newReaders[0] = { ...newReaders[0], version: "v2" };
              }
              if (!newReaders[1].inCriticalSection) {
                newReaders[1] = { ...newReaders[1], version: "v2" };
              }
              return newReaders;
            });
            timerRef.current = setTimeout(advance, 1000);
            return "writer-swap";

          case "writer-swap":
            addLog("⏳ Grace Period: 等待所有旧读者离开...");
            setGracePeriodProgress(0);
            let prog = 0;
            const gpInterval = setInterval(() => {
              prog += 10;
              setGracePeriodProgress(prog);
              if (prog >= 30) {
                setReaders(prev => {
                  const newReaders = [...prev];
                  newReaders[0] = { ...newReaders[0], inCriticalSection: false, version: null };
                  return newReaders;
                });
                addLog("📖 读者 0 离开临界区");
              }
              if (prog >= 60) {
                setReaders(prev => {
                  const newReaders = [...prev];
                  newReaders[1] = { ...newReaders[1], inCriticalSection: false, version: null };
                  return newReaders;
                });
                addLog("📖 读者 1 离开临界区");
              }
              if (prog >= 100) {
                clearInterval(gpInterval);
                setReaders(prev => {
                  const newReaders = [...prev];
                  newReaders[2] = { ...newReaders[2], inCriticalSection: false, version: null };
                  return newReaders;
                });
                addLog("📖 读者 2 离开临界区");
                addLog("✅ Grace Period 结束");
                timerRef.current = setTimeout(advance, 800);
              }
            }, 300);
            return "grace-period";

          case "grace-period":
            addLog("🗑️ 释放旧版本 v1");
            setV1Alive(false);
            setIsRunning(false);
            return "reclaim";

          case "reclaim":
            return "reclaim";

          default:
            return current;
        }
      });
    };

    timerRef.current = setTimeout(advance, 500);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [isRunning]);

  const phaseLabel: Record<Phase, string> = {
    idle: "初始状态",
    "readers-reading": "读者读取 v1",
    "writer-copy": "写者复制",
    "writer-modify": "写者修改副本",
    "writer-swap": "原子替换指针",
    "grace-period": "Grace Period",
    reclaim: "回收旧版本",
  };

  const phaseColor: Record<Phase, string> = {
    idle: "bg-slate-500",
    "readers-reading": "bg-blue-500",
    "writer-copy": "bg-orange-500",
    "writer-modify": "bg-orange-600",
    "writer-swap": "bg-purple-500",
    "grace-period": "bg-yellow-500",
    reclaim: "bg-green-500",
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-violet-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2 text-center">
        RCU (Read-Copy-Update) 机制
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 text-center mb-6">
        读者无锁访问，写者复制后原子替换，Grace Period 后回收旧版本
      </p>

      <div className="flex justify-center gap-2 mb-6 flex-wrap">
        {(["idle", "readers-reading", "writer-copy", "writer-modify", "writer-swap", "grace-period", "reclaim"] as Phase[]).map(p => (
          <div
            key={p}
            className={`px-3 py-1 rounded-full text-xs font-medium text-white transition-all ${
              phase === p ? `${phaseColor[p]} shadow-lg scale-110` : "bg-slate-300 dark:bg-slate-600"
            }`}
          >
            {phaseLabel[p]}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="col-span-2 bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">数据版本</h4>
          <div className="flex items-center justify-center gap-8">
            <motion.div
              animate={{ opacity: v1Alive ? 1 : 0.3, scale: v1Alive ? 1 : 0.9 }}
              className={`relative p-4 rounded-xl border-2 ${
                pointerTarget === "v1"
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-900/30"
                  : "border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700"
              }`}
            >
              <div className="text-center">
                <div className="text-lg font-bold text-slate-700 dark:text-slate-200">v1</div>
                <div className="text-xs text-slate-500 mt-1">数据 = {version === "v1" ? "旧值" : "旧值"}</div>
                {!v1Alive && <div className="text-xs text-red-500 mt-1">已释放</div>}
              </div>
              {pointerTarget === "v1" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute -top-2 left-1/2 -translate-x-1/2 px-2 py-0.5 bg-blue-500 text-white text-[10px] rounded-full"
                >
                  ptr →
                </motion.div>
              )}
            </motion.div>

            <ArrowRight className="w-6 h-6 text-slate-400" />

            <motion.div
              animate={{ opacity: v2Alive ? 1 : 0.3, scale: v2Alive ? 1 : 0.9 }}
              className={`relative p-4 rounded-xl border-2 ${
                pointerTarget === "v2"
                  ? "border-green-500 bg-green-50 dark:bg-green-900/30"
                  : "border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700"
              }`}
            >
              <div className="text-center">
                <div className="text-lg font-bold text-slate-700 dark:text-slate-200">v2</div>
                <div className="text-xs text-slate-500 mt-1">数据 = {version === "v2" ? "新值" : "—"}</div>
                {!v2Alive && <div className="text-xs text-slate-400 mt-1">未创建</div>}
              </div>
              {pointerTarget === "v2" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute -top-2 left-1/2 -translate-x-1/2 px-2 py-0.5 bg-green-500 text-white text-[10px] rounded-full"
                >
                  ptr →
                </motion.div>
              )}
            </motion.div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">事件日志</h4>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {log.length === 0 ? (
              <p className="text-xs text-slate-400 italic">点击开始演示...</p>
            ) : (
              log.map((entry, i) => (
                <motion.p
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="text-xs text-slate-600 dark:text-slate-300"
                >
                  {entry}
                </motion.p>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-6">
        {readers.map(reader => (
          <motion.div
            key={reader.id}
            animate={reader.inCriticalSection ? { borderColor: ["#60a5fa", "#a78bfa", "#60a5fa"] } : {}}
            transition={{ duration: 1.5, repeat: Infinity }}
            className={`p-3 rounded-lg border-2 text-center ${
              reader.inCriticalSection
                ? "bg-blue-50 dark:bg-blue-900/20 border-blue-400"
                : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700"
            }`}
          >
            <BookOpen className={`w-5 h-5 mx-auto mb-1 ${reader.inCriticalSection ? "text-blue-500" : "text-slate-400"}`} />
            <div className="text-sm font-medium text-slate-700 dark:text-slate-200">读者 {reader.id}</div>
            <div className="text-xs mt-1">
              {reader.inCriticalSection ? (
                <span className="text-blue-600 dark:text-blue-400">
                  临界区中 → {reader.version}
                </span>
              ) : (
                <span className="text-slate-400">空闲</span>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {phase === "grace-period" && (
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm text-slate-600 dark:text-slate-300 mb-2">
            <span className="flex items-center gap-1"><Clock className="w-4 h-4" /> Grace Period 进度</span>
            <span className="font-mono">{gracePeriodProgress}%</span>
          </div>
          <div className="w-full h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-yellow-400 to-green-500 rounded-full"
              animate={{ width: `${gracePeriodProgress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            等待所有在指针替换前进入临界区的读者离开...
          </p>
        </div>
      )}

      <div className="flex justify-center gap-3 mb-6">
        <button
          onClick={() => { reset(); setTimeout(() => setIsRunning(true), 100); }}
          disabled={isRunning}
          className="flex items-center gap-2 px-5 py-2.5 bg-violet-600 hover:bg-violet-700 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-all"
        >
          <Play className="w-4 h-4" /> 开始演示
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-5 py-2.5 bg-slate-600 hover:bg-slate-700 text-white rounded-lg font-semibold transition-all"
        >
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      <div className="bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800 rounded-lg p-4">
        <h4 className="font-semibold text-violet-700 dark:text-violet-300 mb-2">RCU 核心思想</h4>
        <div className="grid grid-cols-2 gap-4 text-xs text-violet-800 dark:text-violet-200">
          <div>
            <p className="font-medium mb-1">读者（Read）</p>
            <ul className="list-disc list-inside space-y-0.5">
              <li>rcu_read_lock() — 禁用抢占</li>
              <li>rcu_dereference() — 安全读取指针</li>
              <li>rcu_read_unlock() — 恢复抢占</li>
              <li>完全无锁！</li>
            </ul>
          </div>
          <div>
            <p className="font-medium mb-1">写者（Copy-Update）</p>
            <ul className="list-disc list-inside space-y-0.5">
              <li>复制数据到新版本</li>
              <li>修改新版本</li>
              <li>rcu_assign_pointer() — 原子替换</li>
              <li>synchronize_rcu() — 等待 Grace Period</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
