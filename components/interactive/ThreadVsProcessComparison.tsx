"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Cpu, HardDrive, ArrowRight, RotateCcw } from "lucide-react";

const MEMORY_REGIONS = {
  process: [
    { name: "Code (Text)", color: "bg-blue-500", borderColor: "border-blue-600", shared: false, desc: "Executable instructions - shared between threads" },
    { name: "Data", color: "bg-emerald-500", borderColor: "border-emerald-600", shared: true, desc: "Global/static variables - shared between threads" },
    { name: "Heap", color: "bg-amber-500", borderColor: "border-amber-600", shared: true, desc: "Dynamically allocated memory - shared between threads" },
    { name: "Stack", color: "bg-red-500", borderColor: "border-red-600", shared: false, desc: "Function call frames, local variables - per-thread" },
    { name: "File Descriptors", color: "bg-purple-500", borderColor: "border-purple-600", shared: true, desc: "Open files, sockets - shared between threads" },
    { name: "Page Table", color: "bg-pink-500", borderColor: "border-pink-600", shared: true, desc: "Virtual-to-physical address mapping - shared" },
  ],
  threads: {
    shared: [
      { name: "Code (Text)", color: "bg-blue-500", borderColor: "border-blue-600", desc: "Shared executable instructions" },
      { name: "Data", color: "bg-emerald-500", borderColor: "border-emerald-600", desc: "Shared global/static variables" },
      { name: "Heap", color: "bg-amber-500", borderColor: "border-amber-600", desc: "Shared dynamically allocated memory" },
      { name: "File Descriptors", color: "bg-purple-500", borderColor: "border-purple-600", desc: "Shared open files and sockets" },
      { name: "Page Table", color: "bg-pink-500", borderColor: "border-pink-600", desc: "Shared address space mapping" },
    ],
    private: [
      { name: "Stack (Thread 1)", color: "bg-red-400", borderColor: "border-red-500", desc: "Thread 1's private stack" },
      { name: "Stack (Thread 2)", color: "bg-orange-400", borderColor: "border-orange-500", desc: "Thread 2's private stack" },
      { name: "Registers (Thread 1)", color: "bg-teal-400", borderColor: "border-teal-500", desc: "Thread 1's private register set" },
      { name: "Registers (Thread 2)", color: "bg-cyan-400", borderColor: "border-cyan-500", desc: "Thread 2's private register set" },
    ],
  },
};

export default function ThreadVsProcessComparison() {
  const [phase, setPhase] = useState<"idle" | "process" | "threads">("idle");
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [autoPlay, setAutoPlay] = useState(false);

  useEffect(() => {
    if (!autoPlay) return;
    const steps: Array<"idle" | "process" | "threads"> = ["idle", "process", "threads"];
    let idx = 0;
    const timer = setInterval(() => {
      idx = (idx + 1) % steps.length;
      setPhase(steps[idx]);
    }, 2500);
    return () => clearInterval(timer);
  }, [autoPlay]);

  const handleRegionHover = (name: string, e: React.MouseEvent) => {
    setHoveredRegion(name);
    setTooltipPos({ x: e.clientX, y: e.clientY });
  };

  const getRegionInfo = (name: string) => {
    const proc = MEMORY_REGIONS.process.find((r) => r.name === name);
    if (proc) return { shared: proc.shared, desc: proc.desc };
    const shared = MEMORY_REGIONS.threads.shared.find((r) => r.name === name);
    if (shared) return { shared: true, desc: shared.desc };
    const priv = MEMORY_REGIONS.threads.private.find((r) => r.name === name);
    if (priv) return { shared: false, desc: priv.desc };
    return { shared: false, desc: "" };
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center flex items-center justify-center gap-2">
        <Layers className="w-7 h-7 text-indigo-600" />
        Process vs Thread Memory Model
      </h2>

      {/* Controls */}
      <div className="flex justify-center gap-3 mb-6 flex-wrap">
        {(["idle", "process", "threads"] as const).map((p) => (
          <button
            key={p}
            onClick={() => { setPhase(p); setAutoPlay(false); }}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              phase === p
                ? "bg-indigo-600 text-white shadow-md"
                : "bg-white text-slate-700 border border-slate-300 hover:bg-indigo-50 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600"
            }`}
          >
            {p === "idle" ? "Initial" : p === "process" ? "Single Process" : "Multi-Threaded"}
          </button>
        ))}
        <button
          onClick={() => setAutoPlay(!autoPlay)}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            autoPlay ? "bg-amber-500 text-white" : "bg-white text-slate-700 border border-slate-300 hover:bg-amber-50 dark:bg-gray-700 dark:text-gray-200"
          }`}
        >
          {autoPlay ? "Stop Auto" : "Auto Play"}
        </button>
        <button
          onClick={() => { setPhase("idle"); setAutoPlay(false); }}
          className="px-4 py-2 bg-white text-slate-700 border border-slate-300 rounded-lg hover:bg-red-50 dark:bg-gray-700 dark:text-gray-200"
        >
          <RotateCcw className="w-4 h-4 inline" />
        </button>
      </div>

      {/* Visual Area */}
      <div className="flex gap-8 justify-center items-start flex-wrap">
        {/* Left: Process View */}
        <div className="flex-1 min-w-[300px] max-w-[450px]">
          <div className="text-center mb-3">
            <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200">
              <Cpu className="w-5 h-5 inline mr-1 text-indigo-500" />
              Process Address Space
            </h3>
          </div>
          <div className="relative bg-white dark:bg-gray-800 border-2 border-slate-300 dark:border-gray-600 rounded-lg p-4 min-h-[400px]">
            <div className="absolute top-2 left-3 text-xs text-slate-400 dark:text-gray-500 font-mono">
              Virtual Memory
            </div>

            <AnimatePresence>
              {phase === "process" && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="space-y-2 mt-6"
                >
                  {MEMORY_REGIONS.process.map((region, i) => (
                    <motion.div
                      key={region.name}
                      initial={{ opacity: 0, x: -30 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.15 }}
                      onMouseEnter={(e) => handleRegionHover(region.name, e)}
                      onMouseLeave={() => setHoveredRegion(null)}
                      className={`${region.color} ${region.borderColor} border-2 rounded-md px-3 py-3 text-white text-sm font-medium cursor-pointer hover:brightness-110 transition-all hover:scale-[1.02]`}
                    >
                      <div className="flex items-center justify-between">
                        <span>{region.name}</span>
                        <span className="text-xs opacity-80 bg-white/20 px-2 py-0.5 rounded">
                          {region.shared ? "Shared" : "Private"}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                  <div className="flex items-center justify-center gap-2 mt-3 text-xs text-slate-500 dark:text-gray-400">
                    <HardDrive className="w-4 h-4" />
                    <span>Single process — all resources in one address space</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {phase === "idle" && (
              <div className="flex items-center justify-center h-full text-slate-400 dark:text-gray-500 text-sm">
                Click &quot;Single Process&quot; or &quot;Multi-Threaded&quot; to begin
              </div>
            )}

            {phase === "threads" && (
              <div className="flex gap-3 mt-6">
                {/* Shared Section */}
                <div className="flex-1">
                  <div className="text-xs text-center font-semibold text-slate-600 dark:text-gray-300 mb-2">
                    Shared Resources
                  </div>
                  {MEMORY_REGIONS.threads.shared.map((region, i) => (
                    <motion.div
                      key={region.name}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                      onMouseEnter={(e) => handleRegionHover(region.name, e)}
                      onMouseLeave={() => setHoveredRegion(null)}
                      className={`${region.color} ${region.borderColor} border-2 rounded-md px-2 py-2 text-white text-xs font-medium mb-1.5 cursor-pointer hover:brightness-110 transition-all`}
                    >
                      {region.name}
                    </motion.div>
                  ))}
                </div>
                {/* Thread 1 Private */}
                <div className="flex-1">
                  <div className="text-xs text-center font-semibold text-red-600 dark:text-red-400 mb-2">
                    Thread 1 Private
                  </div>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    onMouseEnter={(e) => handleRegionHover("Stack (Thread 1)", e)}
                    onMouseLeave={() => setHoveredRegion(null)}
                    className="bg-red-400 border-2 border-red-500 rounded-md px-2 py-2 text-white text-xs font-medium mb-1.5 cursor-pointer hover:brightness-110"
                  >
                    Stack (T1)
                  </motion.div>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                    onMouseEnter={(e) => handleRegionHover("Registers (Thread 1)", e)}
                    onMouseLeave={() => setHoveredRegion(null)}
                    className="bg-teal-400 border-2 border-teal-500 rounded-md px-2 py-2 text-white text-xs font-medium cursor-pointer hover:brightness-110"
                  >
                    Registers (T1)
                  </motion.div>
                </div>
                {/* Thread 2 Private */}
                <div className="flex-1">
                  <div className="text-xs text-center font-semibold text-orange-600 dark:text-orange-400 mb-2">
                    Thread 2 Private
                  </div>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                    onMouseEnter={(e) => handleRegionHover("Stack (Thread 2)", e)}
                    onMouseLeave={() => setHoveredRegion(null)}
                    className="bg-orange-400 border-2 border-orange-500 rounded-md px-2 py-2 text-white text-xs font-medium mb-1.5 cursor-pointer hover:brightness-110"
                  >
                    Stack (T2)
                  </motion.div>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8 }}
                    onMouseEnter={(e) => handleRegionHover("Registers (Thread 2)", e)}
                    onMouseLeave={() => setHoveredRegion(null)}
                    className="bg-cyan-400 border-2 border-cyan-500 rounded-md px-2 py-2 text-white text-xs font-medium cursor-pointer hover:brightness-110"
                  >
                    Registers (T2)
                  </motion.div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Arrow */}
        <div className="flex items-center justify-center pt-20">
          <AnimatePresence>
            {phase === "threads" && (
              <motion.div
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0 }}
              >
                <ArrowRight className="w-10 h-10 text-indigo-400" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right: Comparison Table */}
        <div className="flex-1 min-w-[280px] max-w-[400px]">
          <div className="text-center mb-3">
            <h3 className="text-lg font-semibold text-slate-700 dark:text-gray-200">Comparison</h3>
          </div>
          <div className="bg-white dark:bg-gray-800 border-2 border-slate-300 dark:border-gray-600 rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-100 dark:bg-gray-700">
                  <th className="px-3 py-2 text-left text-slate-700 dark:text-gray-200">Aspect</th>
                  <th className="px-3 py-2 text-left text-slate-700 dark:text-gray-200">Process</th>
                  <th className="px-3 py-2 text-left text-slate-700 dark:text-gray-200">Thread</th>
                </tr>
              </thead>
              <tbody className="text-slate-600 dark:text-gray-300">
                {[
                  ["Memory", "Separate address space", "Shared address space"],
                  ["Code/Data", "Own copy", "Shared"],
                  ["Stack", "Own stack", "Own stack per thread"],
                  ["File Descriptors", "Own table", "Shared table"],
                  ["Page Table", "Own page table", "Shared page table"],
                  ["Creation Cost", "High (fork/exec)", "Low (clone)"],
                  ["IPC", "Explicit needed", "Direct (shared memory)"],
                  ["Isolation", "Fully isolated", "No memory isolation"],
                  ["Context Switch", "Expensive", "Lightweight"],
                ].map(([aspect, proc, thread], i) => (
                  <motion.tr
                    key={aspect}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: phase !== "idle" ? i * 0.05 : 0 }}
                    className="border-t border-slate-200 dark:border-gray-600 hover:bg-indigo-50 dark:hover:bg-gray-700/50 transition-colors"
                  >
                    <td className="px-3 py-1.5 font-medium">{aspect}</td>
                    <td className="px-3 py-1.5">{proc}</td>
                    <td className="px-3 py-1.5">{thread}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Key Insight */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: phase === "threads" ? 1 : 0.5 }}
            className="mt-4 p-3 bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-200 dark:border-indigo-700 rounded-lg text-sm text-indigo-800 dark:text-indigo-300"
          >
            <strong>Key Insight:</strong> Threads share code, data, heap, and file descriptors
            but each has its own stack and register context. This makes communication faster
            but requires synchronization for shared data.
          </motion.div>
        </div>
      </div>

      {/* Tooltip */}
      <AnimatePresence>
        {hoveredRegion && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
            className="fixed z-50 bg-slate-800 text-white text-xs rounded-lg px-3 py-2 shadow-xl pointer-events-none max-w-[250px]"
            style={{ left: tooltipPos.x + 12, top: tooltipPos.y - 10 }}
          >
            <div className="font-semibold mb-1">{hoveredRegion}</div>
            <div className="text-slate-300">{getRegionInfo(hoveredRegion).desc}</div>
            <div className={`mt-1 font-medium ${getRegionInfo(hoveredRegion).shared ? "text-emerald-400" : "text-amber-400"}`}>
              {getRegionInfo(hoveredRegion).shared ? "Shared between threads" : "Private to thread"}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
