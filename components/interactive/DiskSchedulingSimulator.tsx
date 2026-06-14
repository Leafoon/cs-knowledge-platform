"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, HardDrive, ArrowRight, Info, ChevronRight } from "lucide-react";

type Algorithm = "FCFS" | "SSTF" | "SCAN" | "C-SCAN" | "LOOK" | "C-LOOK";

const ALGORITHMS: Algorithm[] = ["FCFS", "SSTF", "SCAN", "C-SCAN", "LOOK", "C-LOOK"];

const ALGO_DESC: Record<Algorithm, string> = {
  FCFS: "先来先服务：按请求顺序依次访问",
  SSTF: "最短寻道时间优先：选择距离最近的请求",
  SCAN: "电梯算法：向一个方向移动到尽头再折返",
  "C-SCAN": "循环扫描：到尽头后跳回起点重新扫描",
  LOOK: "类似SCAN但不到尽头，到达最后请求即折返",
  "C-LOOK": "类似C-SCAN但不到尽头，到达最后请求后跳回",
};

function schedule(algo: Algorithm, requests: number[], start: number, maxTrack: number): number[] {
  const seq: number[] = [start];
  const remaining = [...requests];

  switch (algo) {
    case "FCFS": {
      seq.push(...remaining);
      break;
    }
    case "SSTF": {
      let pos = start;
      while (remaining.length > 0) {
        let minDist = Infinity;
        let minIdx = 0;
        remaining.forEach((r, i) => {
          const d = Math.abs(r - pos);
          if (d < minDist) { minDist = d; minIdx = i; }
        });
        pos = remaining.splice(minIdx, 1)[0];
        seq.push(pos);
      }
      break;
    }
    case "SCAN": {
      const sorted = [...remaining].sort((a, b) => a - b);
      const left = sorted.filter(r => r < start);
      const right = sorted.filter(r => r >= start);
      seq.push(...right, maxTrack, ...left.reverse());
      break;
    }
    case "C-SCAN": {
      const sorted = [...remaining].sort((a, b) => a - b);
      const left = sorted.filter(r => r < start);
      const right = sorted.filter(r => r >= start);
      seq.push(...right, maxTrack, 0, ...left);
      break;
    }
    case "LOOK": {
      const sorted = [...remaining].sort((a, b) => a - b);
      const left = sorted.filter(r => r < start);
      const right = sorted.filter(r => r >= start);
      seq.push(...right, ...left.reverse());
      break;
    }
    case "C-LOOK": {
      const sorted = [...remaining].sort((a, b) => a - b);
      const left = sorted.filter(r => r < start);
      const right = sorted.filter(r => r >= start);
      seq.push(...right, ...left);
      break;
    }
  }
  return seq;
}

export function DiskSchedulingSimulator() {
  const [inputValue, setInputValue] = useState("98,183,37,122,14,124,65,67");
  const [startPos, setStartPos] = useState(53);
  const [maxTrack, setMaxTrack] = useState(200);
  const [algorithm, setAlgorithm] = useState<Algorithm>("FCFS");
  const [sequence, setSequence] = useState<number[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [animating, setAnimating] = useState(false);
  const [totalSeek, setTotalSeek] = useState(0);

  const runSimulation = useCallback(() => {
    const requests = inputValue.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    if (requests.length === 0) return;
    const seq = schedule(algorithm, requests, startPos, maxTrack);
    setSequence(seq);
    setCurrentStep(0);
    setAnimating(true);
    setTotalSeek(0);

    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step >= seq.length) {
        clearInterval(interval);
        setAnimating(false);
        let total = 0;
        for (let i = 1; i < seq.length; i++) total += Math.abs(seq[i] - seq[i - 1]);
        setTotalSeek(total);
        return;
      }
      setCurrentStep(step);
    }, 500);
  }, [algorithm, inputValue, startPos, maxTrack]);

  const reset = () => {
    setSequence([]);
    setCurrentStep(0);
    setAnimating(false);
    setTotalSeek(0);
  };

  const progress = sequence.length > 1 ? currentStep / (sequence.length - 1) : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 mb-2">
        <HardDrive className="w-6 h-6 text-blue-600 dark:text-blue-400" />
        <h3 className="text-lg font-bold text-text-primary">磁盘调度模拟器</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">请求队列（逗号分隔）</label>
          <input
            type="text"
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-text-primary font-mono text-sm"
            disabled={animating}
          />
        </div>
        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-text-secondary mb-1">磁头位置</label>
            <input
              type="number"
              value={startPos}
              onChange={e => setStartPos(parseInt(e.target.value) || 0)}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-text-primary font-mono text-sm"
              disabled={animating}
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium text-text-secondary mb-1">最大磁道</label>
            <input
              type="number"
              value={maxTrack}
              onChange={e => setMaxTrack(parseInt(e.target.value) || 200)}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-text-primary font-mono text-sm"
              disabled={animating}
            />
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {ALGORITHMS.map(algo => (
          <button
            key={algo}
            onClick={() => setAlgorithm(algo)}
            disabled={animating}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              algorithm === algo
                ? "bg-blue-600 text-white"
                : "bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700"
            }`}
          >
            {algo}
          </button>
        ))}
      </div>

      <p className="text-sm text-text-secondary">{ALGO_DESC[algorithm]}</p>

      <div className="flex gap-3">
        <button
          onClick={runSimulation}
          disabled={animating}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors text-sm font-medium"
        >
          <Play className="w-4 h-4" /> 运行
        </button>
        <button
          onClick={reset}
          disabled={animating}
          className="flex items-center gap-2 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-text-primary rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors text-sm font-medium"
        >
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      {sequence.length > 0 && (
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-text-secondary">磁头位置: {sequence[currentStep]}</span>
              <span className="text-sm text-text-secondary">
                步骤 {currentStep}/{sequence.length - 1}
              </span>
            </div>
            <div className="relative h-10 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className="absolute top-0 h-full bg-blue-500/20 rounded-full"
                animate={{ width: `${(currentStep / Math.max(sequence.length - 1, 1)) * 100}%` }}
                transition={{ duration: 0.3 }}
              />
              <motion.div
                className="absolute top-1/2 -translate-y-1/2 w-6 h-6 bg-blue-600 rounded-full shadow-lg border-2 border-white dark:border-gray-900 flex items-center justify-center"
                animate={{ left: `${(sequence[currentStep] / maxTrack) * 100}%` }}
                transition={{ duration: 0.4, type: "spring" }}
              >
                <div className="w-2 h-2 bg-white rounded-full" />
              </motion.div>
            </div>
            <div className="flex justify-between text-xs text-text-secondary mt-1">
              <span>0</span>
              <span>{maxTrack}</span>
            </div>
          </div>

          <div className="flex flex-wrap gap-1 items-center">
            {sequence.map((track, i) => (
              <div key={i} className="flex items-center">
                <motion.span
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className={`px-2 py-1 rounded text-xs font-mono ${
                    i <= currentStep
                      ? i === currentStep
                        ? "bg-blue-600 text-white"
                        : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                      : "bg-gray-100 dark:bg-gray-800 text-text-secondary"
                  }`}
                >
                  {track}
                </motion.span>
                {i < sequence.length - 1 && (
                  <ChevronRight className="w-3 h-3 text-text-secondary mx-0.5" />
                )}
              </div>
            ))}
          </div>

          {totalSeek > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-300 dark:border-green-700"
            >
              <div className="flex items-center gap-2">
                <ArrowRight className="w-5 h-5 text-green-600 dark:text-green-400" />
                <span className="font-semibold text-green-700 dark:text-green-300">
                  总寻道距离: {totalSeek} 磁道
                </span>
              </div>
              <div className="text-sm text-text-secondary mt-1">
                平均寻道距离: {(totalSeek / (sequence.length - 1)).toFixed(1)} 磁道
              </div>
            </motion.div>
          )}
        </div>
      )}

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-text-secondary">
            <p><strong className="text-text-primary">磁盘调度算法</strong>优化磁头移动顺序以减少寻道时间。SSTF 通常最优但可能饥饿，SCAN/LOOK 更公平。</p>
          </div>
        </div>
      </div>
    </div>
  );
}
