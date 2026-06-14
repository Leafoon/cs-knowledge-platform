"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Database, ArrowRight } from "lucide-react";

interface CacheEntry {
  dev: number;
  blockno: number;
  valid: boolean;
  refcnt: number;
  data: string;
  isHit?: boolean;
  isEvicted?: boolean;
}

const INITIAL_ENTRIES: CacheEntry[] = [
  { dev: 1, blockno: 1, valid: true, refcnt: 0, data: "superblock" },
  { dev: 1, blockno: 5, valid: true, refcnt: 0, data: "inode #33" },
  { dev: 1, blockno: 12, valid: true, refcnt: 0, data: "data block" },
  { dev: 1, blockno: 30, valid: true, refcnt: 0, data: "bitmap" },
  { dev: 1, blockno: 45, valid: true, refcnt: 0, data: "dir data" },
];

interface AccessEvent {
  blockno: number;
  label: string;
  type: "hit" | "miss";
}

const ACCESS_SEQUENCE: AccessEvent[] = [
  { blockno: 1, label: "bread(1, 1)", type: "hit" },
  { blockno: 100, label: "bread(1, 100)", type: "miss" },
  { blockno: 45, label: "bread(1, 45)", type: "hit" },
  { blockno: 200, label: "bread(1, 200)", type: "miss" },
  { blockno: 12, label: "bread(1, 12)", type: "hit" },
  { blockno: 300, label: "bread(1, 300)", type: "miss" },
  { blockno: 5, label: "bread(1, 5)", type: "hit" },
];

export default function BufferCacheLRU() {
  const [entries, setEntries] = useState<CacheEntry[]>(INITIAL_ENTRIES);
  const [step, setStep] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [highlightBlock, setHighlightBlock] = useState<number | null>(null);

  const reset = useCallback(() => {
    setEntries(INITIAL_ENTRIES.map((e) => ({ ...e, isHit: undefined, isEvicted: undefined })));
    setStep(-1);
    setIsRunning(false);
    setLog([]);
    setHighlightBlock(null);
  }, []);

  const runStep = useCallback(() => {
    setStep((prevStep) => {
      const nextStep = prevStep + 1;
      if (nextStep >= ACCESS_SEQUENCE.length) {
        setIsRunning(false);
        return prevStep;
      }

      const event = ACCESS_SEQUENCE[nextStep];

      setEntries((prev) => {
        const updated = prev.map((e) => ({
          ...e,
          isHit: undefined as boolean | undefined,
          isEvicted: undefined as boolean | undefined,
        }));

        // Check if block is in cache (LRU search from MRU end)
        const mruIdx = updated.findIndex(
          (e) => e.valid && e.dev === 1 && e.blockno === event.blockno
        );

        if (mruIdx !== -1) {
          // Cache hit: increment refcnt, move to MRU
          updated[mruIdx].refcnt++;
          updated[mruIdx].isHit = true;
          setHighlightBlock(event.blockno);
          setLog((prev) => [
            ...prev,
            `✓ 命中: block ${event.blockno} (refcnt=${updated[mruIdx].refcnt})`,
          ]);
        } else {
          // Cache miss: find LRU victim (search from LRU end, refcnt==0)
          const victimIdx = [...updated]
            .reverse()
            .findIndex((e) => e.refcnt === 0);
          const actualIdx = victimIdx === -1 ? -1 : updated.length - 1 - victimIdx;

          if (actualIdx !== -1) {
            const old = updated[actualIdx];
            updated[actualIdx] = {
              dev: 1,
              blockno: event.blockno,
              valid: true,
              refcnt: 1,
              data: `block #${event.blockno}`,
              isHit: false,
              isEvicted: false,
            };
            // Mark the evicted position
            setHighlightBlock(event.blockno);
            setLog((prev) => [
              ...prev,
              `✗ 未命中: block ${event.blockno} → 替换 LRU block ${old.blockno}`,
            ]);
          }
        }

        return updated;
      });

      return nextStep;
    });
  }, []);

  const autoRun = useCallback(() => {
    if (isRunning) return;
    setIsRunning(true);
    reset();
    // Small delay then start
    setTimeout(() => {
      let i = 0;
      const interval = setInterval(() => {
        runStep();
        i++;
        if (i >= ACCESS_SEQUENCE.length) {
          clearInterval(interval);
          setIsRunning(false);
        }
      }, 1200);
    }, 300);
  }, [isRunning, reset, runStep]);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        Buffer Cache — LRU 替换模拟
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        模拟 xv6 缓冲区缓存的 LRU 替换策略。MRU 在左，LRU 在右。
      </p>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={autoRun}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <Play className="w-4 h-4" />
          自动演示
        </button>
        <button
          onClick={() => {
            if (step < ACCESS_SEQUENCE.length - 1) runStep();
          }}
          disabled={isRunning || step >= ACCESS_SEQUENCE.length - 1}
          className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <ArrowRight className="w-4 h-4" />
          单步
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      {/* Current operation */}
      {step >= 0 && step < ACCESS_SEQUENCE.length && (
        <motion.div
          key={step}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-4"
        >
          <span className="inline-block px-4 py-2 bg-white dark:bg-gray-800 rounded-lg border border-slate-200 dark:border-gray-700 font-mono text-sm text-slate-700 dark:text-gray-200">
            {ACCESS_SEQUENCE[step].label}
            <span
              className={`ml-2 text-xs font-bold ${
                ACCESS_SEQUENCE[step].type === "hit"
                  ? "text-emerald-500"
                  : "text-red-500"
              }`}
            >
              {ACCESS_SEQUENCE[step].type === "hit" ? "HIT" : "MISS"}
            </span>
          </span>
        </motion.div>
      )}

      {/* Cache entries as LRU list */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700 mb-6">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-bold text-slate-500 dark:text-gray-400 uppercase tracking-wider">
            MRU 端
          </span>
          <Database className="w-4 h-4 text-slate-400" />
          <span className="text-xs font-bold text-slate-500 dark:text-gray-400 uppercase tracking-wider">
            LRU 端
          </span>
        </div>
        <div className="grid grid-cols-5 gap-3">
          <AnimatePresence mode="popLayout">
            {entries.map((entry, i) => (
              <motion.div
                key={`${entry.blockno}-${i}`}
                layout
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className={`relative p-3 rounded-lg border-2 text-center transition-all ${
                  entry.isHit
                    ? "border-emerald-400 bg-emerald-50 dark:bg-emerald-950/30 shadow-md"
                    : highlightBlock === entry.blockno && !entry.isHit
                    ? "border-red-400 bg-red-50 dark:bg-red-950/30 shadow-md"
                    : "border-slate-200 dark:border-gray-600 bg-slate-50 dark:bg-gray-750"
                }`}
              >
                <div className="text-xs font-mono text-slate-500 dark:text-gray-400 mb-1">
                  slot {i}
                </div>
                <div className="text-sm font-bold font-mono text-slate-800 dark:text-gray-100">
                  block {entry.blockno}
                </div>
                <div className="text-[10px] text-slate-500 dark:text-gray-400 mt-1">
                  {entry.data}
                </div>
                <div className="flex items-center justify-center gap-1 mt-2">
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                      entry.refcnt > 0
                        ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                        : "bg-slate-100 text-slate-500 dark:bg-gray-700 dark:text-gray-400"
                    }`}
                  >
                    ref={entry.refcnt}
                  </span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded ${
                      entry.valid
                        ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                        : "bg-slate-100 text-slate-500 dark:bg-gray-700 dark:text-gray-400"
                    }`}
                  >
                    {entry.valid ? "valid" : "invalid"}
                  </span>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* Event log */}
      {log.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-2">
            操作日志
          </h3>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {log.map((entry, i) => (
              <div
                key={i}
                className={`text-xs font-mono ${
                  entry.startsWith("✓")
                    ? "text-emerald-600 dark:text-emerald-400"
                    : "text-red-600 dark:text-red-400"
                }`}
              >
                {entry}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 justify-center text-xs text-slate-500 dark:text-gray-400">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-emerald-100 border border-emerald-400" />
          缓存命中
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-red-100 border border-red-400" />
          缓存未命中 & 替换
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-blue-100 border border-blue-300" />
          refcnt &gt; 0（不可替换）
        </span>
      </div>
    </div>
  );
}
