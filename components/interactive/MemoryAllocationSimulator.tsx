"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, RotateCcw, Zap, Info } from "lucide-react";

interface MemBlock {
  id: number;
  start: number;
  size: number;
  allocated: boolean;
  color: string;
}

type Algorithm = "first" | "best" | "worst";

const COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#F97316"];

function makeInitialBlocks(): MemBlock[] {
  return [
    { id: 0, start: 0, size: 100, allocated: true, color: COLORS[0] },
    { id: 1, start: 100, size: 60, allocated: false, color: "#4B5563" },
    { id: 2, start: 160, size: 120, allocated: true, color: COLORS[1] },
    { id: 3, start: 280, size: 80, allocated: false, color: "#4B5563" },
    { id: 4, start: 360, size: 200, allocated: true, color: COLORS[2] },
    { id: 5, start: 560, size: 40, allocated: false, color: "#4B5563" },
    { id: 6, start: 600, size: 150, allocated: true, color: COLORS[3] },
    { id: 7, start: 750, size: 100, allocated: false, color: "#4B5563" },
  ];
}

export default function MemoryAllocationSimulator() {
  const [blocks, setBlocks] = useState<MemBlock[]>(makeInitialBlocks);
  const [algorithm, setAlgorithm] = useState<Algorithm>("first");
  const [allocSize, setAllocSize] = useState(50);
  const [scanning, setScanning] = useState<number[]>([]);
  const [selectedBlock, setSelectedBlock] = useState<number | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [nextId, setNextId] = useState(8);

  const totalSize = blocks.reduce((s, b) => s + b.size, 0);
  const freeSize = blocks.filter((b) => !b.allocated).reduce((s, b) => s + b.size, 0);
  const freeBlocks = blocks.filter((b) => !b.allocated);
  const externalFrag = freeBlocks.length > 0 ? (1 - Math.max(...freeBlocks.map((b) => b.size)) / freeSize) * 100 : 0;

  const animateScan = useCallback(async (indices: number[]): Promise<number | null> => {
    for (let i = 0; i < indices.length; i++) {
      setScanning([indices[i]]);
      await new Promise((r) => setTimeout(r, 400));
    }
    setScanning([]);
    if (indices.length > 0) return indices[indices.length - 1];
    return null;
  }, []);

  const handleAllocate = useCallback(async () => {
    setLog((prev) => [`Allocating ${allocSize} units using ${algorithm === "first" ? "First Fit" : algorithm === "best" ? "Best Fit" : "Worst Fit"}...`, ...prev]);

    const freeBlocks = blocks
      .map((b, i) => ({ ...b, idx: i }))
      .filter((b) => !b.allocated && b.size >= allocSize);

    if (freeBlocks.length === 0) {
      setLog((prev) => ["ERROR: No suitable free block found!", ...prev]);
      return;
    }

    let chosen: typeof freeBlocks[0];
    if (algorithm === "first") {
      chosen = freeBlocks[0];
    } else if (algorithm === "best") {
      chosen = freeBlocks.reduce((min, b) => (b.size < min.size ? b : min), freeBlocks[0]);
    } else {
      chosen = freeBlocks.reduce((max, b) => (b.size > max.size ? b : max), freeBlocks[0]);
    }

    const indicesToScan = freeBlocks.map((b) => b.idx);
    if (algorithm === "best" || algorithm === "worst") {
      indicesToScan.sort((a, b) => a - b);
    }

    await animateScan(indicesToScan);

    const newBlocks = [...blocks];
    const block = newBlocks[chosen.idx];

    if (block.size === allocSize) {
      block.allocated = true;
      block.color = COLORS[nextId % COLORS.length];
      setLog((prev) => [`Exact fit at block ${chosen.idx} (${block.start})`, ...prev]);
    } else {
      const newBlock: MemBlock = {
        id: nextId,
        start: block.start,
        size: allocSize,
        allocated: true,
        color: COLORS[nextId % COLORS.length],
      };
      block.start += allocSize;
      block.size -= allocSize;
      newBlocks.splice(chosen.idx, 0, newBlock);
      setLog((prev) => [`Split block ${chosen.idx}: allocated ${allocSize} at ${newBlock.start}, remainder ${block.size}`, ...prev]);
    }

    setBlocks(newBlocks);
    setNextId((n) => n + 1);
  }, [allocSize, algorithm, blocks, nextId, animateScan]);

  const handleFree = useCallback((idx: number) => {
    const block = blocks[idx];
    if (!block.allocated) return;

    setLog((prev) => [`Freed block ${idx} (${block.size} units at ${block.start})`, ...prev]);

    const newBlocks = [...blocks];
    newBlocks[idx] = { ...block, allocated: false, color: "#4B5563" };

    // Coalesce with next free block
    if (idx + 1 < newBlocks.length && !newBlocks[idx + 1].allocated) {
      newBlocks[idx].size += newBlocks[idx + 1].size;
      newBlocks.splice(idx + 1, 1);
    }
    // Coalesce with previous free block
    if (idx - 1 >= 0 && !newBlocks[idx - 1].allocated) {
      newBlocks[idx - 1].size += newBlocks[idx].size;
      newBlocks.splice(idx, 1);
    }

    setBlocks(newBlocks);
  }, [blocks]);

  const handleReset = () => {
    setBlocks(makeInitialBlocks());
    setScanning([]);
    setSelectedBlock(null);
    setLog([]);
    setNextId(8);
  };

  const scale = 800; // total pixel width

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Cpu className="w-7 h-7 text-cyan-600 dark:text-cyan-400" />
        <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100">
          Memory Allocation Simulator
        </h2>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6 items-end">
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Algorithm</label>
          <div className="flex bg-slate-200 dark:bg-gray-700 rounded-lg p-1">
            {(["first", "best", "worst"] as const).map((a) => (
              <button key={a} onClick={() => setAlgorithm(a)}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition ${
                  algorithm === a ? "bg-cyan-600 text-white shadow" : "text-slate-600 dark:text-slate-300"
                }`}>
                {a === "first" ? "First Fit" : a === "best" ? "Best Fit" : "Worst Fit"}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            Size: {allocSize} units
          </label>
          <input type="range" min={10} max={120} value={allocSize}
            onChange={(e) => setAllocSize(Number(e.target.value))}
            className="w-40" />
        </div>
        <button onClick={handleAllocate}
          className="px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition flex items-center gap-2 text-sm font-medium">
          <Zap className="w-4 h-4" /> Allocate
        </button>
        <button onClick={handleReset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition flex items-center gap-2 text-sm font-medium">
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
      </div>

      {/* Memory visualization */}
      <div className="mb-4 bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-2 text-xs text-slate-500 dark:text-slate-400 font-mono">
          <span>0</span>
          <span>{totalSize}</span>
        </div>
        <div className="relative h-16 rounded-lg overflow-hidden bg-gray-200 dark:bg-gray-700">
          {blocks.map((block, idx) => {
            const left = (block.start / totalSize) * 100;
            const width = (block.size / totalSize) * 100;
            const isScanning = scanning.includes(idx);
            return (
              <motion.div
                key={block.id}
                className="absolute top-0 h-full flex items-center justify-center cursor-pointer border-r border-white/30 dark:border-black/20"
                style={{
                  left: `${left}%`,
                  width: `${width}%`,
                  backgroundColor: block.allocated ? block.color : "#6B7280",
                }}
                initial={{ opacity: 0, scaleY: 0 }}
                animate={{
                  opacity: 1,
                  scaleY: 1,
                  filter: isScanning ? "brightness(1.4)" : "brightness(1)",
                  boxShadow: isScanning ? "0 0 12px rgba(255,255,255,0.5)" : "none",
                  outline: selectedBlock === idx ? "3px solid white" : "none",
                }}
                transition={{ duration: 0.3 }}
                onClick={() => block.allocated ? handleFree(idx) : null}
                title={block.allocated ? `Click to free (${block.size} units)` : `Free: ${block.size} units`}
              >
                {width > 3 && (
                  <span className="text-white text-xs font-mono truncate px-1">
                    {block.size}
                  </span>
                )}
              </motion.div>
            );
          })}
        </div>
        <div className="flex gap-4 mt-3 text-xs text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-gray-500 inline-block" /> Free</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500 inline-block" /> Allocated</span>
          <span className="text-slate-400 dark:text-slate-500 ml-2">Click an allocated block to free it</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg border border-cyan-200 dark:border-cyan-800 text-center">
          <div className="text-xl font-bold text-cyan-700 dark:text-cyan-300">{blocks.length}</div>
          <div className="text-xs text-slate-500 dark:text-slate-400">Total Blocks</div>
        </div>
        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800 text-center">
          <div className="text-xl font-bold text-green-700 dark:text-green-300">{freeBlocks.length}</div>
          <div className="text-xs text-slate-500 dark:text-slate-400">Free Blocks</div>
        </div>
        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800 text-center">
          <div className="text-xl font-bold text-amber-700 dark:text-amber-300">{freeSize}</div>
          <div className="text-xs text-slate-500 dark:text-slate-400">Free Units</div>
        </div>
        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800 text-center">
          <div className="text-xl font-bold text-red-700 dark:text-red-300">{externalFrag.toFixed(1)}%</div>
          <div className="text-xs text-slate-500 dark:text-slate-400">Ext. Fragmentation</div>
        </div>
      </div>

      {/* Algorithm info & log side by side */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-slate-200 dark:border-gray-700">
          <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
            <Info className="w-4 h-4" /> Algorithm: {algorithm === "first" ? "First Fit" : algorithm === "best" ? "Best Fit" : "Worst Fit"}
          </h4>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            {algorithm === "first" && "Scans the free list from the beginning and allocates the first block that is large enough. Fast but can leave small fragments at the front."}
            {algorithm === "best" && "Searches the entire free list and picks the smallest block that fits. Minimizes wasted space but is slower and can create many tiny fragments."}
            {algorithm === "worst" && "Picks the largest available block. Leaves larger leftover blocks but may exhaust large blocks quickly."}
          </p>
        </div>
        <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-slate-200 dark:border-gray-700">
          <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">Operation Log</h4>
          <div className="max-h-32 overflow-y-auto space-y-1">
            <AnimatePresence>
              {log.slice(0, 8).map((entry, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                  className="text-xs font-mono text-slate-600 dark:text-slate-400">
                  {entry}
                </motion.div>
              ))}
            </AnimatePresence>
            {log.length === 0 && <p className="text-xs text-slate-400 dark:text-slate-500">No operations yet</p>}
          </div>
        </div>
      </div>
    </div>
  );
}
