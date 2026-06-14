"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowRight,
  Play,
  RotateCcw,
  Calculator,
  Layers,
  HardDrive,
  ChevronRight,
} from "lucide-react";

interface TraversalStep {
  level: string;
  description: string;
  index: number;
  result: string;
}

const BLOCK_SIZE = 4096;
const PTR_SIZE = 4;
const PTRS_PER_BLOCK = BLOCK_SIZE / PTR_SIZE; // 1024

const directLimit = 12;
const singleLimit = directLimit + PTRS_PER_BLOCK; // 1036
const doubleLimit = singleLimit + PTRS_PER_BLOCK * PTRS_PER_BLOCK; // 1049612
const tripleLimit = doubleLimit + PTRS_PER_BLOCK * PTRS_PER_BLOCK * PTRS_PER_BLOCK; // ~1 billion

export default function MultiLevelIndexing() {
  const [blockNumber, setBlockNumber] = useState(15);
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [steps, setSteps] = useState<TraversalStep[]>([]);
  const [physicalBlock, setPhysicalBlock] = useState<number | null>(null);
  const [resolved, setResolved] = useState(false);

  // Simulated block pointers
  const directBlocks = [1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035];
  const singleIndirectBlock = 2048;
  const doubleIndirectBlock = 3072;
  const tripleIndirectBlock = 4096;

  // Simulated indirect block contents
  const singleIndirectContents: Record<number, number> = {};
  for (let i = 0; i < PTRS_PER_BLOCK; i++) {
    singleIndirectContents[i] = 5000 + i;
  }
  const doubleIndirectL1: Record<number, number> = {};
  for (let i = 0; i < 5; i++) {
    doubleIndirectL1[i] = 8000 + i;
  }
  const doubleIndirectL2: Record<number, Record<number, number>> = {};
  for (let i = 0; i < 5; i++) {
    doubleIndirectL2[i] = {};
    for (let j = 0; j < 10; j++) {
      doubleIndirectL2[i][j] = 10000 + i * 1000 + j;
    }
  }

  const resolveBlock = useCallback(() => {
    const n = blockNumber;
    const result: TraversalStep[] = [];
    let physBlock = 0;

    if (n < 12) {
      // Direct
      result.push({
        level: "Direct",
        description: `Block ${n} is within the 12 direct pointers (0-11).`,
        index: n,
        result: `i_block[${n}] -> physical block ${directBlocks[n]}`,
      });
      physBlock = directBlocks[n];
    } else if (n < singleLimit) {
      // Single indirect
      const indirectIndex = n - 12;
      result.push({
        level: "Direct Check",
        description: `Block ${n} > 11, not in direct range. Use single indirect pointer.`,
        index: 12,
        result: `i_block[12] -> indirect block ${singleIndirectBlock}`,
      });
      result.push({
        level: "Single Indirect",
        description: `Index into indirect block: ${n} - 12 = ${indirectIndex}`,
        index: indirectIndex,
        result: `indirect[${indirectIndex}] -> physical block ${singleIndirectContents[indirectIndex] || 5000 + indirectIndex}`,
      });
      physBlock = singleIndirectContents[indirectIndex] || 5000 + indirectIndex;
    } else if (n < doubleLimit) {
      // Double indirect
      const offset = n - singleLimit;
      const l1Index = Math.floor(offset / PTRS_PER_BLOCK);
      const l2Index = offset % PTRS_PER_BLOCK;
      result.push({
        level: "Direct Check",
        description: `Block ${n} > ${singleLimit - 1}. Use double indirect pointer.`,
        index: 13,
        result: `i_block[13] -> L1 block ${doubleIndirectBlock}`,
      });
      const l1Block = doubleIndirectL1[l1Index] || 8000 + l1Index;
      result.push({
        level: "Double Indirect L1",
        description: `L1 index: (${n} - ${singleLimit}) / ${PTRS_PER_BLOCK} = ${l1Index}`,
        index: l1Index,
        result: `L1[${l1Index}] -> L2 block ${l1Block}`,
      });
      const l2Block = (doubleIndirectL2[l1Index] && doubleIndirectL2[l1Index][l2Index]) || 10000 + l1Index * 1000 + l2Index;
      result.push({
        level: "Double Indirect L2",
        description: `L2 index: (${n} - ${singleLimit}) % ${PTRS_PER_BLOCK} = ${l2Index}`,
        index: l2Index,
        result: `L2[${l2Index}] -> physical block ${l2Block}`,
      });
      physBlock = l2Block;
    } else if (n < tripleLimit) {
      // Triple indirect
      const offset = n - doubleLimit;
      const l1Index = Math.floor(offset / (PTRS_PER_BLOCK * PTRS_PER_BLOCK));
      const l2Index = Math.floor((offset % (PTRS_PER_BLOCK * PTRS_PER_BLOCK)) / PTRS_PER_BLOCK);
      const l3Index = offset % PTRS_PER_BLOCK;
      result.push({
        level: "Direct Check",
        description: `Block ${n} > ${doubleLimit - 1}. Use triple indirect pointer.`,
        index: 14,
        result: `i_block[14] -> L1 block ${tripleIndirectBlock}`,
      });
      result.push({
        level: "Triple Indirect L1",
        description: `L1 index: ${l1Index}`,
        index: l1Index,
        result: `L1[${l1Index}] -> L2 block ${6000 + l1Index}`,
      });
      result.push({
        level: "Triple Indirect L2",
        description: `L2 index: ${l2Index}`,
        index: l2Index,
        result: `L2[${l2Index}] -> L3 block ${7000 + l1Index * 100 + l2Index}`,
      });
      result.push({
        level: "Triple Indirect L3",
        description: `L3 index: ${l3Index}`,
        index: l3Index,
        result: `L3[${l3Index}] -> physical block ${90000 + offset}`,
      });
      physBlock = 90000 + offset;
    } else {
      result.push({
        level: "Error",
        description: `Block number ${n} exceeds maximum file size.`,
        index: -1,
        result: "Out of range",
      });
    }

    setSteps(result);
    setPhysicalBlock(physBlock);
    setCurrentStep(0);
    setIsAnimating(true);
    setResolved(false);
  }, [blockNumber]);

  useEffect(() => {
    if (!isAnimating || currentStep < 0) return;
    if (currentStep >= steps.length) {
      setIsAnimating(false);
      setResolved(true);
      return;
    }
    const timer = setTimeout(() => {
      setCurrentStep((prev) => prev + 1);
    }, 1000);
    return () => clearTimeout(timer);
  }, [isAnimating, currentStep, steps.length]);

  const reset = () => {
    setIsAnimating(false);
    setCurrentStep(-1);
    setSteps([]);
    setPhysicalBlock(null);
    setResolved(false);
  };

  const levelDescriptions = [
    { name: "Direct [0-11]", max: "12 blocks = 48 KB", color: "bg-red-500" },
    { name: "Single Indirect", max: "1024 blocks = 4 MB", color: "bg-orange-500" },
    { name: "Double Indirect", max: "1024^2 = 4 GB", color: "bg-yellow-500" },
    { name: "Triple Indirect", max: "1024^3 = 4 TB", color: "bg-green-500" },
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Multi-Level Indexing
      </h2>

      {/* Input */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6 items-center justify-center">
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600 dark:text-gray-300 font-medium">
            Logical Block #:
          </label>
          <input
            type="number"
            min={0}
            max={tripleLimit - 1}
            value={blockNumber}
            onChange={(e) => {
              setBlockNumber(Math.max(0, Math.min(tripleLimit - 1, Number(e.target.value))));
              reset();
            }}
            className="w-32 px-3 py-2 text-sm font-mono bg-white dark:bg-gray-800 border border-slate-300 dark:border-gray-600 rounded-lg text-slate-800 dark:text-gray-100"
          />
        </div>
        <div className="flex gap-2">
          <button
            onClick={resolveBlock}
            disabled={isAnimating}
            className="flex items-center gap-2 px-4 py-2 bg-rose-500 text-white rounded-lg hover:bg-rose-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Play className="w-4 h-4" />
            Resolve
          </button>
          <button
            onClick={reset}
            className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>

      {/* Quick block number presets */}
      <div className="flex gap-2 mb-6 justify-center flex-wrap">
        {[
          { label: "Direct (5)", value: 5 },
          { label: "Single Indirect (15)", value: 15 },
          { label: "Single Indirect (500)", value: 500 },
          { label: "Double Indirect (2000)", value: 2000 },
          { label: "Large (100000)", value: 100000 },
        ].map((preset) => (
          <button
            key={preset.value}
            onClick={() => {
              setBlockNumber(preset.value);
              reset();
            }}
            className="px-3 py-1.5 text-xs bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-300 rounded-full hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Inode visualization */}
        <div className="lg:col-span-2">
          {/* Inode block pointers */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700 mb-4">
            <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
              <Layers className="w-4 h-4 text-rose-500" />
              Inode Block Pointers (i_block[0..14])
            </h3>

            <div className="grid grid-cols-6 sm:grid-cols-15 gap-1 mb-4">
              {Array.from({ length: 15 }, (_, i) => {
                const isDirect = i < 12;
                const isSingle = i === 12;
                const isDouble = i === 13;
                const isTriple = i === 14;
                const isHighlighted =
                  steps.some(
                    (s) =>
                      (s.level === "Direct" && s.index === i) ||
                      (s.level === "Direct Check" && s.index === i)
                  );

                return (
                  <motion.div
                    key={i}
                    className={`p-1.5 rounded text-center text-[10px] font-mono border transition-all ${
                      isHighlighted
                        ? "bg-yellow-200 dark:bg-yellow-800 border-yellow-400 font-bold"
                        : isDirect
                        ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                        : isSingle
                        ? "bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800"
                        : isDouble
                        ? "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800"
                        : "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                    }`}
                  >
                    <div className="text-slate-500 dark:text-gray-400">[{i}]</div>
                    <div className="text-slate-700 dark:text-gray-200 text-[9px]">
                      {isDirect
                        ? directBlocks[i]
                        : isSingle
                        ? singleIndirectBlock
                        : isDouble
                        ? doubleIndirectBlock
                        : tripleIndirectBlock}
                    </div>
                  </motion.div>
                );
              })}
            </div>

            <div className="flex flex-wrap gap-3">
              {levelDescriptions.map((ld) => (
                <div key={ld.name} className="flex items-center gap-1.5">
                  <div className={`w-2.5 h-2.5 rounded ${ld.color}`} />
                  <span className="text-[10px] text-slate-600 dark:text-gray-300">
                    {ld.name}: {ld.max}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Traversal steps */}
          <AnimatePresence>
            {steps.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700"
              >
                <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3">
                  Traversal Steps
                </h3>
                <div className="space-y-2">
                  {steps.map((step, i) => {
                    const isActive = currentStep === i;
                    const isDone = currentStep > i;

                    return (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: i <= currentStep ? 1 : 0.3, x: 0 }}
                        className={`p-3 rounded-lg border-l-4 transition-all ${
                          isActive
                            ? "border-yellow-400 bg-yellow-50 dark:bg-yellow-900/20"
                            : isDone
                            ? "border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20"
                            : "border-slate-200 dark:border-gray-700 bg-slate-50 dark:bg-gray-750"
                        }`}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          {isActive && (
                            <motion.div
                              animate={{ scale: [1, 1.2, 1] }}
                              transition={{ repeat: Infinity, duration: 1 }}
                              className="w-2 h-2 rounded-full bg-yellow-400"
                            />
                          )}
                          {isDone && (
                            <div className="w-2 h-2 rounded-full bg-emerald-400" />
                          )}
                          <span className="text-xs font-bold text-slate-700 dark:text-gray-200">
                            {step.level}
                          </span>
                        </div>
                        <p className="text-xs text-slate-600 dark:text-gray-300 mb-1">
                          {step.description}
                        </p>
                        {isDone && (
                          <p className="text-xs font-mono text-emerald-600 dark:text-emerald-400">
                            {step.result}
                          </p>
                        )}
                      </motion.div>
                    );
                  })}
                </div>

                {/* Result */}
                {resolved && physicalBlock !== null && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="mt-4 p-4 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg border border-emerald-300 dark:border-emerald-700 flex items-center gap-3"
                  >
                    <HardDrive className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
                    <div>
                      <p className="text-sm font-bold text-emerald-700 dark:text-emerald-300">
                        Physical Block Found
                      </p>
                      <p className="text-lg font-mono text-emerald-800 dark:text-emerald-200">
                        Block #{physicalBlock}
                      </p>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Calculator / info panel */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700 mb-4">
            <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
              <Calculator className="w-4 h-4 text-rose-500" />
              Max File Size Calculator
            </h3>
            <div className="space-y-3">
              {[
                {
                  level: "Direct only",
                  blocks: 12,
                  size: "48 KB",
                  color: "text-red-600 dark:text-red-400",
                },
                {
                  level: "Up to Single Indirect",
                  blocks: singleLimit,
                  size: "~4 MB",
                  color: "text-orange-600 dark:text-orange-400",
                },
                {
                  level: "Up to Double Indirect",
                  blocks: doubleLimit,
                  size: "~4 GB",
                  color: "text-yellow-600 dark:text-yellow-400",
                },
                {
                  level: "Up to Triple Indirect",
                  blocks: tripleLimit,
                  size: "~4 TB",
                  color: "text-green-600 dark:text-green-400",
                },
              ].map((item) => (
                <div
                  key={item.level}
                  className="flex justify-between items-center text-xs border-b border-slate-100 dark:border-gray-700 pb-2"
                >
                  <div>
                    <p className="text-slate-700 dark:text-gray-200 font-medium">{item.level}</p>
                    <p className="text-slate-500 dark:text-gray-400">
                      {item.blocks.toLocaleString()} blocks
                    </p>
                  </div>
                  <span className={`font-mono font-bold ${item.color}`}>{item.size}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700">
            <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-2">
              How It Works
            </h3>
            <div className="text-xs text-slate-600 dark:text-gray-300 space-y-2">
              <p>
                The inode contains 15 block pointers. The first 12 are <strong>direct</strong> -- they
                point straight to data blocks.
              </p>
              <p>
                Pointer 12 (<strong>single indirect</strong>) points to a block containing 1024 more
                pointers.
              </p>
              <p>
                Pointer 13 (<strong>double indirect</strong>) adds another level: a block of pointers to
                blocks of pointers.
              </p>
              <p>
                Pointer 14 (<strong>triple indirect</strong>) adds a third level, enabling files up to ~4
                TB.
              </p>
            </div>
            <div className="mt-3 text-xs text-slate-500 dark:text-gray-400 font-mono">
              Block size: {BLOCK_SIZE} bytes | Ptr size: {PTR_SIZE} bytes | Ptrs/block: {PTRS_PER_BLOCK}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
