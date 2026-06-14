"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, ArrowRight, Play, RotateCcw, Zap } from "lucide-react";

type MesiState = "Modified" | "Exclusive" | "Shared" | "Invalid";

interface CpuCache {
  id: number;
  state: MesiState;
  data: number;
}

interface Transition {
  from: MesiState;
  to: MesiState;
  event: string;
  description: string;
  busAction?: string;
}

const TRANSITIONS: Transition[] = [
  { from: "Invalid", to: "Exclusive", event: "本地读（其他CPU无副本）", description: "从内存读取数据，获得独占权", busAction: "BusRd" },
  { from: "Invalid", to: "Shared", event: "本地读（其他CPU有副本）", description: "从其他CPU或内存读取，共享状态", busAction: "BusRd" },
  { from: "Invalid", to: "Modified", event: "本地写", description: "获取独占权并写入数据", busAction: "BusRdX" },
  { from: "Exclusive", to: "Modified", event: "本地写", description: "直接升级为修改状态", busAction: undefined },
  { from: "Exclusive", to: "Shared", event: "其他CPU读", description: "其他CPU请求共享，转为共享状态", busAction: "BusRd" },
  { from: "Exclusive", to: "Invalid", event: "其他CPU写", description: "其他CPU请求独占，本缓存行失效", busAction: "BusRdX" },
  { from: "Shared", to: "Modified", event: "本地写", description: "升级为修改状态，其他CPU失效", busAction: "BusUpgr" },
  { from: "Shared", to: "Shared", event: "本地读", description: "直接读取，状态不变", busAction: undefined },
  { from: "Shared", to: "Invalid", event: "其他CPU写", description: "其他CPU修改，本缓存行失效", busAction: "BusRdX" },
  { from: "Modified", to: "Shared", event: "其他CPU读", description: "回写到内存，转为共享状态", busAction: "BusRd (回写)" },
  { from: "Modified", to: "Invalid", event: "其他CPU写", description: "回写到内存，转为无效状态", busAction: "BusRdX (回写)" },
  { from: "Modified", to: "Shared", event: "其他CPU读", description: "数据发送给其他CPU，转为共享", busAction: "Flush" },
];

const STATE_COLORS: Record<MesiState, { bg: string; border: string; text: string; label: string }> = {
  Modified: { bg: "bg-red-100 dark:bg-red-900/30", border: "border-red-500", text: "text-red-700 dark:text-red-300", label: "M (Modified)" },
  Exclusive: { bg: "bg-blue-100 dark:bg-blue-900/30", border: "border-blue-500", text: "text-blue-700 dark:text-blue-300", label: "E (Exclusive)" },
  Shared: { bg: "bg-green-100 dark:bg-green-900/30", border: "border-green-500", text: "text-green-700 dark:text-green-300", label: "S (Shared)" },
  Invalid: { bg: "bg-gray-100 dark:bg-gray-800", border: "border-gray-400", text: "text-gray-500 dark:text-gray-400", label: "I (Invalid)" },
};

export default function CacheCoherenceProtocol() {
  const [cpus, setCpus] = useState<CpuCache[]>([
    { id: 0, state: "Exclusive", data: 42 },
    { id: 1, state: "Invalid", data: 0 },
    { id: 2, state: "Invalid", data: 0 },
  ]);
  const [selectedCpu, setSelectedCpu] = useState<number | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [memoryValue, setMemoryValue] = useState(42);
  const [highlightedTransition, setHighlightedTransition] = useState<string | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog(prev => [...prev.slice(-8), `[CPU ${msg}]`]);
  }, []);

  const reset = () => {
    setCpus([
      { id: 0, state: "Exclusive", data: 42 },
      { id: 1, state: "Invalid", data: 0 },
      { id: 2, state: "Invalid", data: 0 },
    ]);
    setMemoryValue(42);
    setLog([]);
    setSelectedCpu(null);
    setHighlightedTransition(null);
  };

  const simulateRead = (cpuId: number) => {
    setCpus(prev => {
      const next = prev.map(c => ({ ...c }));
      const cpu = next[cpuId];
      const others = next.filter(c => c.id !== cpuId);
      const hasOtherCopy = others.some(c => c.state !== "Invalid");

      if (cpu.state === "Invalid") {
        if (hasOtherCopy) {
          const modified = others.find(c => c.state === "Modified");
          if (modified) {
            modified.state = "Shared";
            setMemoryValue(modified.data);
          }
          others.forEach(c => { if (c.state === "Exclusive") c.state = "Shared"; });
          cpu.state = "Shared";
          cpu.data = modified ? modified.data : memoryValue;
          addLog(`${cpuId}: BusRd → 获得共享副本`);
        } else {
          cpu.state = "Exclusive";
          cpu.data = memoryValue;
          addLog(`${cpuId}: BusRd → 独占读取`);
        }
      } else if (cpu.state === "Shared" || cpu.state === "Exclusive" || cpu.state === "Modified") {
        addLog(`${cpuId}: 读命中 (${cpu.state})`);
      }
      return next;
    });
  };

  const simulateWrite = (cpuId: number) => {
    setCpus(prev => {
      const next = prev.map(c => ({ ...c }));
      const cpu = next[cpuId];
      const others = next.filter(c => c.id !== cpuId);
      const newValue = cpu.data + 1;

      if (cpu.state === "Invalid") {
        others.forEach(c => { if (c.state !== "Invalid") c.state = "Invalid"; });
        cpu.state = "Modified";
        cpu.data = newValue;
        addLog(`${cpuId}: BusRdX → 独占写入 (${newValue})`);
      } else if (cpu.state === "Shared") {
        others.forEach(c => { if (c.state === "Shared") c.state = "Invalid"; });
        cpu.state = "Modified";
        cpu.data = newValue;
        addLog(`${cpuId}: BusUpgr → 升级写入 (${newValue})`);
      } else if (cpu.state === "Exclusive") {
        cpu.state = "Modified";
        cpu.data = newValue;
        addLog(`${cpuId}: 直接写入 (${newValue})`);
      } else {
        cpu.data = newValue;
        addLog(`${cpuId}: 写命中 → 修改 (${newValue})`);
      }
      return next;
    });
  };

  const getValidTransitions = (state: MesiState) => {
    return TRANSITIONS.filter(t => t.from === state);
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-indigo-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2 text-center">
        MESI 缓存一致性协议
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 text-center mb-6">
        点击 CPU 选择，然后执行读/写操作观察状态转换
      </p>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {cpus.map(cpu => {
          const colors = STATE_COLORS[cpu.state];
          const isSelected = selectedCpu === cpu.id;
          return (
            <motion.div
              key={cpu.id}
              onClick={() => setSelectedCpu(cpu.id)}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className={`relative p-4 rounded-xl border-2 cursor-pointer transition-all ${colors.bg} ${colors.border} ${isSelected ? "ring-4 ring-indigo-400 dark:ring-indigo-600 shadow-lg" : ""}`}
            >
              <div className="flex items-center gap-2 mb-3">
                <Cpu className={`w-5 h-5 ${colors.text}`} />
                <span className="font-bold text-slate-700 dark:text-slate-200">CPU {cpu.id}</span>
              </div>
              <div className="text-center mb-2">
                <span className={`text-xs font-mono px-2 py-1 rounded ${colors.bg} ${colors.text} border ${colors.border}`}>
                  {colors.label}
                </span>
              </div>
              <div className="text-center">
                <span className="text-sm text-slate-500 dark:text-slate-400">数据: </span>
                <span className={`font-mono font-bold ${cpu.state === "Invalid" ? "text-gray-400 line-through" : "text-slate-800 dark:text-slate-100"}`}>
                  {cpu.state === "Invalid" ? "—" : cpu.data}
                </span>
              </div>
              <div className="mt-2 text-center">
                <span className="text-xs text-slate-400">
                  {cpu.state === "Modified" ? "脏数据，与内存不一致" :
                   cpu.state === "Exclusive" ? "干净数据，独占" :
                   cpu.state === "Shared" ? "干净数据，共享" : "无效"}
                </span>
              </div>
              <AnimatePresence>
                {isSelected && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute -top-1 -right-1 w-4 h-4 bg-indigo-500 rounded-full"
                  />
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      <div className="flex items-center justify-center gap-3 mb-4">
        <div className="px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg text-sm">
          <span className="text-slate-500 dark:text-slate-400">内存值: </span>
          <span className="font-mono font-bold text-slate-800 dark:text-slate-100">{memoryValue}</span>
        </div>
      </div>

      <div className="flex justify-center gap-3 mb-6">
        <button
          onClick={() => selectedCpu !== null && simulateRead(selectedCpu)}
          disabled={selectedCpu === null}
          className="flex items-center gap-2 px-5 py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-all"
        >
          <Play className="w-4 h-4" /> 读取
        </button>
        <button
          onClick={() => selectedCpu !== null && simulateWrite(selectedCpu)}
          disabled={selectedCpu === null}
          className="flex items-center gap-2 px-5 py-2.5 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-all"
        >
          <Zap className="w-4 h-4" /> 写入
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-5 py-2.5 bg-slate-600 hover:bg-slate-700 text-white rounded-lg font-semibold transition-all"
        >
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">状态说明</h4>
          {Object.entries(STATE_COLORS).map(([state, colors]) => (
            <div key={state} className="flex items-center gap-2 mb-2">
              <span className={`text-xs font-mono px-2 py-0.5 rounded ${colors.bg} ${colors.text} border ${colors.border}`}>
                {colors.label}
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {state === "Modified" ? "独占修改，脏数据" :
                 state === "Exclusive" ? "独占，与内存一致" :
                 state === "Shared" ? "共享，与内存一致" : "无效"}
              </span>
            </div>
          ))}
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">事件日志</h4>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {log.length === 0 ? (
              <p className="text-xs text-slate-400 italic">选择 CPU 并执行操作...</p>
            ) : (
              log.map((entry, i) => (
                <motion.p
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="text-xs font-mono text-slate-600 dark:text-slate-300"
                >
                  {entry}
                </motion.p>
              ))
            )}
          </div>
        </div>
      </div>

      {selectedCpu !== null && (
        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">
            CPU {selectedCpu} 的可能转换（当前: {STATE_COLORS[cpus[selectedCpu].state].label}）
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {getValidTransitions(cpus[selectedCpu].state).map((t, i) => (
              <div
                key={i}
                className="flex items-center gap-2 p-2 bg-slate-50 dark:bg-slate-700/50 rounded text-xs"
              >
                <ArrowRight className="w-3 h-3 text-slate-400 flex-shrink-0" />
                <div>
                  <span className="font-medium text-slate-700 dark:text-slate-200">{t.event}</span>
                  <span className="text-slate-400 mx-1">→</span>
                  <span className={STATE_COLORS[t.to].text}>{STATE_COLORS[t.to].label}</span>
                  {t.busAction && <span className="ml-1 text-orange-500">[{t.busAction}]</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
