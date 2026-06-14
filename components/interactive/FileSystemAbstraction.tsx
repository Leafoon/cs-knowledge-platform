"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowDown, ArrowUp, Play, RotateCcw, Layers, Info } from "lucide-react";

interface Layer {
  id: string;
  name: string;
  color: string;
  darkColor: string;
  description: string;
  functions: string[];
}

const layers: Layer[] = [
  {
    id: "app",
    name: "Application",
    color: "bg-purple-500",
    darkColor: "dark:bg-purple-600",
    description: "User programs that perform file operations through standard library calls.",
    functions: ["fopen()", "fread()", "fwrite()", "fprintf()", "fclose()"],
  },
  {
    id: "fileapi",
    name: "File API / VFS Interface",
    color: "bg-blue-500",
    darkColor: "dark:bg-blue-600",
    description: "Standard file descriptor API layer providing uniform system call interface.",
    functions: ["open()", "read()", "write()", "lseek()", "close()", "stat()"],
  },
  {
    id: "vfs",
    name: "Virtual File System (VFS)",
    color: "bg-cyan-500",
    darkColor: "dark:bg-cyan-600",
    description: "Abstraction layer that provides a common interface for different file system implementations. Uses superblock, inode, dentry, and file objects.",
    functions: ["mount()", "lookup()", "readinode()", "create()", "unlink()"],
  },
  {
    id: "fs",
    name: "File System (ext4/xfs/...)",
    color: "bg-emerald-500",
    darkColor: "dark:bg-emerald-600",
    description: "Concrete file system implementation that manages inodes, directories, and data blocks on a logical volume.",
    functions: ["ext4_read()", "ext4_write()", "allocate_block()", "free_block()"],
  },
  {
    id: "block",
    name: "Block Layer / I/O Scheduler",
    color: "bg-amber-500",
    darkColor: "dark:bg-amber-600",
    description: "Manages block I/O requests, performs scheduling, merging, and dispatching of I/O operations to device drivers.",
    functions: ["submit_bio()", "blk_mq_make_request()", "elevator_schedule()"],
  },
  {
    id: "disk",
    name: "Device Driver / Disk",
    color: "bg-red-500",
    darkColor: "dark:bg-red-600",
    description: "Hardware device driver that communicates with the physical storage device (HDD/SSD) via DMA and interrupts.",
    functions: ["read_sector()", "write_sector()", "DMA_transfer()", "interrupt_handler()"],
  },
];

export default function FileSystemAbstraction() {
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [animatingDown, setAnimatingDown] = useState(false);
  const [animatingUp, setAnimatingUp] = useState(false);
  const [currentFlowIndex, setCurrentFlowIndex] = useState(-1);
  const [flowDirection, setFlowDirection] = useState<"down" | "up">("down");
  const [isRunning, setIsRunning] = useState(false);

  const startAnimation = useCallback(() => {
    if (isRunning) return;
    setIsRunning(true);
    setAnimatingDown(true);
    setAnimatingUp(false);
    setCurrentFlowIndex(0);
    setFlowDirection("down");
  }, [isRunning]);

  useEffect(() => {
    if (!animatingDown && !animatingUp) return;

    const timer = setInterval(() => {
      setCurrentFlowIndex((prev) => {
        if (flowDirection === "down") {
          if (prev >= layers.length - 1) {
            setAnimatingDown(false);
            setAnimatingUp(true);
            setFlowDirection("up");
            return prev;
          }
          return prev + 1;
        } else {
          if (prev <= 0) {
            setAnimatingUp(false);
            setIsRunning(false);
            return -1;
          }
          return prev - 1;
        }
      });
    }, 700);

    return () => clearInterval(timer);
  }, [animatingDown, animatingUp, flowDirection]);

  const resetAnimation = () => {
    setIsRunning(false);
    setAnimatingDown(false);
    setAnimatingUp(false);
    setCurrentFlowIndex(-1);
    setSelectedLayer(null);
  };

  const selectedLayerData = layers.find((l) => l.id === selectedLayer);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        File System Abstraction Layers
      </h2>

      <div className="flex gap-4 mb-6 justify-center">
        <button
          onClick={startAnimation}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Play className="w-4 h-4" />
          Animate Read Request
        </button>
        <button
          onClick={resetAnimation}
          className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Layer stack */}
        <div className="lg:col-span-2 flex flex-col items-center gap-2">
          {/* Direction indicator */}
          {isRunning && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-2 text-sm font-medium text-indigo-600 dark:text-indigo-400 mb-2"
            >
              {flowDirection === "down" ? (
                <>
                  <ArrowDown className="w-4 h-4 animate-bounce" />
                  <span>Request flowing down...</span>
                </>
              ) : (
                <>
                  <ArrowUp className="w-4 h-4 animate-bounce" />
                  <span>Response flowing up...</span>
                </>
              )}
            </motion.div>
          )}

          {layers.map((layer, index) => {
            const isActive = currentFlowIndex === index;
            const isHighlighted = index <= currentFlowIndex && isRunning;
            const isPassed = flowDirection === "up" && index > currentFlowIndex && isRunning;

            return (
              <motion.div
                key={layer.id}
                className="w-full max-w-xl cursor-pointer relative"
                onClick={() => setSelectedLayer(selectedLayer === layer.id ? null : layer.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {/* Flow arrow indicator */}
                {isActive && isRunning && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="absolute -left-10 top-1/2 -translate-y-1/2 z-10"
                  >
                    {flowDirection === "down" ? (
                      <ArrowDown className="w-6 h-6 text-indigo-500" />
                    ) : (
                      <ArrowUp className="w-6 h-6 text-emerald-500" />
                    )}
                  </motion.div>
                )}

                <div
                  className={`
                    ${layer.color} ${layer.darkColor} rounded-lg px-6 py-4 text-white font-semibold text-center
                    transition-all duration-300 relative overflow-hidden
                    ${selectedLayer === layer.id ? "ring-2 ring-offset-2 ring-indigo-400 dark:ring-offset-gray-900" : ""}
                    ${isActive ? "ring-2 ring-yellow-400 shadow-lg shadow-yellow-400/30" : ""}
                    ${isPassed ? "opacity-70" : ""}
                  `}
                >
                  {/* Scan effect during animation */}
                  {isActive && (
                    <motion.div
                      className="absolute inset-0 bg-white/20"
                      initial={{ x: "-100%" }}
                      animate={{ x: "100%" }}
                      transition={{ duration: 0.6, ease: "easeInOut" }}
                    />
                  )}

                  <div className="flex items-center justify-center gap-2 relative z-10">
                    <Layers className="w-5 h-5" />
                    <span className="text-lg">{layer.name}</span>
                  </div>

                  {/* Status text during animation */}
                  {isActive && isRunning && (
                    <motion.div
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-xs mt-1 opacity-80 relative z-10"
                    >
                      {flowDirection === "down"
                        ? `Processing: ${layer.functions[0]}`
                        : `Returning result from ${layer.name}`}
                    </motion.div>
                  )}
                </div>

                {/* Connector line between layers */}
                {index < layers.length - 1 && (
                  <div className="flex justify-center my-1">
                    <div className={`w-0.5 h-3 transition-colors duration-300 ${
                      isRunning && index < currentFlowIndex
                        ? "bg-indigo-400"
                        : "bg-slate-300 dark:bg-slate-600"
                    }`} />
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Details panel */}
        <div className="lg:col-span-1">
          <AnimatePresence mode="wait">
            {selectedLayerData ? (
              <motion.div
                key={selectedLayerData.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700"
              >
                <h3 className="text-lg font-bold text-slate-800 dark:text-gray-100 mb-3">
                  {selectedLayerData.name}
                </h3>
                <p className="text-sm text-slate-600 dark:text-gray-300 mb-4 leading-relaxed">
                  {selectedLayerData.description}
                </p>
                <div>
                  <h4 className="text-sm font-semibold text-slate-700 dark:text-gray-200 mb-2">
                    Key Functions:
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedLayerData.functions.map((fn, i) => (
                      <motion.span
                        key={fn}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.05 }}
                        className="px-2 py-1 bg-slate-100 dark:bg-gray-700 text-slate-700 dark:text-gray-200 rounded text-xs font-mono"
                      >
                        {fn}
                      </motion.span>
                    ))}
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700 text-center"
              >
                <Info className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                <p className="text-sm text-slate-500 dark:text-gray-400">
                  Click on any layer to see its description and functions.
                </p>
                <p className="text-xs text-slate-400 dark:text-gray-500 mt-2">
                  Use &quot;Animate Read Request&quot; to see a request flow through all layers.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Legend */}
          <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4 shadow-md border border-slate-200 dark:border-gray-700">
            <h4 className="text-sm font-semibold text-slate-700 dark:text-gray-200 mb-2">
              Legend
            </h4>
            <div className="space-y-2 text-xs text-slate-600 dark:text-gray-300">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-yellow-400" />
                <span>Currently processing</span>
              </div>
              <div className="flex items-center gap-2">
                <ArrowDown className="w-3 h-3 text-indigo-500" />
                <span>Request direction</span>
              </div>
              <div className="flex items-center gap-2">
                <ArrowUp className="w-3 h-3 text-emerald-500" />
                <span>Response direction</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
