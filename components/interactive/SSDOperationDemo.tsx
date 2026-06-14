"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Clock, Trash2, Info, AlertTriangle, RotateCcw } from "lucide-react";

type Operation = "read" | "program" | "erase";

interface Block {
  id: number;
  pages: boolean[];
  eraseCount: number;
}

const BLOCK_SIZE = 4;
const INIT_BLOCKS = 4;

function createBlocks(): Block[] {
  return Array.from({ length: INIT_BLOCKS }, (_, i) => ({
    id: i,
    pages: Array(BLOCK_SIZE).fill(i === 0 ? true : false),
    eraseCount: 0,
  }));
}

export function SSDOperationDemo() {
  const [blocks, setBlocks] = useState<Block[]>(createBlocks);
  const [currentOp, setCurrentOp] = useState<Operation | null>(null);
  const [animating, setAnimating] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [writeAmp, setWriteAmp] = useState({ userWrites: 0, actualWrites: 0 });
  const [wearLevel, setWearLevel] = useState<number[]>(Array(INIT_BLOCKS).fill(0));
  const [highlightPage, setHighlightPage] = useState<{ block: number; page: number } | null>(null);

  const opTimes: Record<Operation, string> = {
    read: "~25 μs",
    program: "~200 μs",
    erase: "~2 ms",
  };

  const opColors: Record<Operation, { bg: string; text: string; border: string }> = {
    read: { bg: "bg-green-50 dark:bg-green-900/20", text: "text-green-700 dark:text-green-300", border: "border-green-400 dark:border-green-600" },
    program: { bg: "bg-blue-50 dark:bg-blue-900/20", text: "text-blue-700 dark:text-blue-300", border: "border-blue-400 dark:border-blue-600" },
    erase: { bg: "bg-red-50 dark:bg-red-900/20", text: "text-red-700 dark:text-red-300", border: "border-red-400 dark:border-red-600" },
  };

  const addLog = (msg: string) => {
    setLog(prev => [`${new Date().toLocaleTimeString()}: ${msg}`, ...prev].slice(0, 20));
  };

  const readPage = (blockIdx: number, pageIdx: number) => {
    if (animating) return;
    setAnimating(true);
    setCurrentOp("read");
    setHighlightPage({ block: blockIdx, page: pageIdx });
    addLog(`读取 Block ${blockIdx} Page ${pageIdx} (${opTimes.read})`);

    setTimeout(() => {
      setCurrentOp(null);
      setAnimating(false);
      setHighlightPage(null);
    }, 600);
  };

  const programPage = (blockIdx: number, pageIdx: number) => {
    if (animating) return;
    if (blocks[blockIdx].pages[pageIdx]) {
      addLog(`Block ${blockIdx} Page ${pageIdx} 已编程，需要先擦除`);
      return;
    }
    setAnimating(true);
    setCurrentOp("program");
    setHighlightPage({ block: blockIdx, page: pageIdx });
    addLog(`编程 Block ${blockIdx} Page ${pageIdx} (${opTimes.program})`);

    setTimeout(() => {
      setBlocks(prev => {
        const next = [...prev];
        next[blockIdx] = { ...next[blockIdx], pages: [...next[blockIdx].pages] };
        next[blockIdx].pages[pageIdx] = true;
        return next;
      });
      setWriteAmp(prev => ({ userWrites: prev.userWrites + 1, actualWrites: prev.actualWrites + 1 }));
      setCurrentOp(null);
      setAnimating(false);
      setHighlightPage(null);
    }, 600);
  };

  const eraseBlock = (blockIdx: number) => {
    if (animating) return;
    setAnimating(true);
    setCurrentOp("erase");
    addLog(`擦除整个 Block ${blockIdx} (${opTimes.erase})`);

    setTimeout(() => {
      setBlocks(prev => {
        const next = [...prev];
        next[blockIdx] = {
          ...next[blockIdx],
          pages: Array(BLOCK_SIZE).fill(false),
          eraseCount: next[blockIdx].eraseCount + 1,
        };
        return next;
      });
      setWearLevel(prev => {
        const next = [...prev];
        next[blockIdx]++;
        return next;
      });
      setCurrentOp(null);
      setAnimating(false);
    }, 1000);
  };

  const demonstrateWriteAmplification = () => {
    if (animating) return;
    const targetBlock = blocks.findIndex(b => b.pages.some(p => p) && b.pages.some(p => !p));
    if (targetBlock === -1) {
      addLog("没有可用的部分填充块来演示写放大");
      return;
    }

    setAnimating(true);
    addLog("=== 写放大演示: 写入 1 页 ===");
    addLog("步骤1: 读取整个块的有效页 (READ)");
    setCurrentOp("read");

    setTimeout(() => {
      addLog("步骤2: 擦除整个块 (ERASE)");
      setCurrentOp("erase");
    }, 800);

    setTimeout(() => {
      addLog("步骤3: 回写所有有效页 + 新页 (PROGRAM)");
      setCurrentOp("program");
      setBlocks(prev => {
        const next = [...prev];
        next[targetBlock] = {
          ...next[targetBlock],
          pages: Array(BLOCK_SIZE).fill(true),
          eraseCount: next[targetBlock].eraseCount + 1,
        };
        return next;
      });
      setWearLevel(prev => {
        const next = [...prev];
        next[targetBlock]++;
        return next;
      });
      setWriteAmp(prev => ({
        userWrites: prev.userWrites + 1,
        actualWrites: prev.actualWrites + BLOCK_SIZE + 1,
      }));
    }, 1600);

    setTimeout(() => {
      setCurrentOp(null);
      setAnimating(false);
      addLog(`完成: 用户写1页 → 实际写了 ${BLOCK_SIZE + 1} 页 (含读+擦除)`);
    }, 2400);
  };

  const reset = () => {
    setBlocks(createBlocks());
    setWriteAmp({ userWrites: 0, actualWrites: 0 });
    setWearLevel(Array(INIT_BLOCKS).fill(0));
    setLog([]);
    setCurrentOp(null);
    setAnimating(false);
  };

  const ampRatio = writeAmp.userWrites > 0 ? (writeAmp.actualWrites / writeAmp.userWrites).toFixed(2) : "N/A";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Zap className="w-6 h-6 text-yellow-500" />
          <h3 className="text-lg font-bold text-text-primary">SSD 操作演示</h3>
        </div>
        <button onClick={reset} className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {(["read", "program", "erase"] as Operation[]).map(op => (
          <div key={op} className={`p-3 rounded-lg border ${currentOp === op ? `${opColors[op].bg} ${opColors[op].border}` : "border-gray-200 dark:border-gray-700"}`}>
            <div className="flex items-center gap-2 mb-1">
              <Clock className="w-4 h-4 text-text-secondary" />
              <span className="font-semibold text-sm text-text-primary">
                {op === "read" ? "页读取" : op === "program" ? "页编程" : "块擦除"}
              </span>
            </div>
            <span className={`font-mono text-sm ${currentOp === op ? opColors[op].text : "text-text-secondary"}`}>{opTimes[op]}</span>
          </div>
        ))}
      </div>

      <div className="space-y-2">
        <div className="text-sm font-medium text-text-secondary mb-1">Flash 存储阵列（点击页/块操作）</div>
        <div className="grid grid-cols-2 gap-4">
          {blocks.map((block, bi) => (
            <div key={bi} className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-text-primary">Block {bi}</span>
                <span className="text-xs text-text-secondary">擦除: {wearLevel[bi]}</span>
              </div>
              <div className="grid grid-cols-4 gap-1 mb-2">
                {block.pages.map((filled, pi) => (
                  <motion.button
                    key={pi}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => filled ? readPage(bi, pi) : programPage(bi, pi)}
                    className={`h-10 rounded text-xs font-mono transition-colors ${
                      highlightPage?.block === bi && highlightPage?.page === pi
                        ? "ring-2 ring-yellow-400"
                        : ""
                    } ${
                      filled
                        ? "bg-blue-500 text-white hover:bg-blue-600"
                        : "bg-gray-200 dark:bg-gray-700 text-text-secondary hover:bg-gray-300 dark:hover:bg-gray-600"
                    }`}
                  >
                    {filled ? "1" : "0"}
                  </motion.button>
                ))}
              </div>
              <button
                onClick={() => eraseBlock(bi)}
                className="w-full flex items-center justify-center gap-1 px-2 py-1 rounded text-xs bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50"
              >
                <Trash2 className="w-3 h-3" /> 擦除块
              </button>
            </div>
          ))}
        </div>
      </div>

      <button
        onClick={demonstrateWriteAmplification}
        disabled={animating}
        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50 transition-colors font-medium"
      >
        <AlertTriangle className="w-4 h-4" /> 演示写放大（写1页 → 擦写整个块）
      </button>

      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <h4 className="font-semibold text-text-primary text-sm mb-2">写放大统计</h4>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-text-secondary">用户写入页数</span>
              <span className="font-mono text-text-primary">{writeAmp.userWrites}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">实际写入页数</span>
              <span className="font-mono text-text-primary">{writeAmp.actualWrites}</span>
            </div>
            <div className="flex justify-between pt-1 border-t border-gray-200 dark:border-gray-700">
              <span className="text-text-secondary">写放大比 (WAF)</span>
              <span className="font-mono font-semibold text-orange-600 dark:text-orange-400">{ampRatio}</span>
            </div>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <h4 className="font-semibold text-text-primary text-sm mb-2">磨损均衡</h4>
          <div className="space-y-1">
            {wearLevel.map((count, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="text-text-secondary w-12">B{i}</span>
                <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${Math.min(count * 10, 100)}%` }} />
                </div>
                <span className="font-mono text-xs text-text-secondary w-6">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <AnimatePresence>
        {log.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="p-3 bg-gray-900 dark:bg-gray-950 rounded-lg overflow-hidden"
          >
            <div className="text-xs font-mono space-y-0.5 max-h-32 overflow-y-auto">
              {log.map((entry, i) => (
                <div key={i} className={`${i === 0 ? "text-green-400" : "text-gray-500"}`}>{entry}</div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-text-secondary">
            <p><strong className="text-text-primary">SSD 特性</strong>：不能原地覆写，必须先擦除再写入。擦除以块为单位（大），读写以页为单位（小），导致写放大。磨损均衡确保各块擦除次数均匀。</p>
          </div>
        </div>
      </div>
    </div>
  );
}
