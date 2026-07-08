"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, Zap, Clock, CheckCircle } from "lucide-react";

export default function ToThreadDemo() {
  const [mode, setMode] = useState<"idle" | "direct" | "threaded">("idle");
  const [directProgress, setDirectProgress] = useState(0);
  const [threadProgress, setThreadProgress] = useState(0);
  const [eventLoopFree, setEventLoopFree] = useState(true);

  const simulateBlocking = useCallback(async () => {
    setMode("direct");
    setDirectProgress(0);
    setEventLoopFree(false);
    for (let i = 0; i <= 100; i += 5) {
      await new Promise((r) => setTimeout(r, 80));
      setDirectProgress(i);
    }
    setEventLoopFree(true);
    setMode("idle");
  }, []);

  const simulateToThread = useCallback(async () => {
    setMode("threaded");
    setThreadProgress(0);
    setDirectProgress(0);
    setEventLoopFree(true);
    const threadPromise = (async () => {
      for (let i = 0; i <= 100; i += 5) {
        await new Promise((r) => setTimeout(r, 80));
        setThreadProgress(i);
      }
    })();
    for (let i = 0; i <= 100; i += 10) {
      await new Promise((r) => setTimeout(r, 100));
      setDirectProgress(i);
    }
    await threadPromise;
    setMode("idle");
  }, []);

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white">
        asyncio.to_thread vs Direct Call
      </h2>
      <div className="flex gap-4">
        <button
          onClick={simulateBlocking}
          disabled={mode !== "idle"}
          className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg disabled:opacity-50 hover:bg-red-600"
        >
          <Play className="w-4 h-4" /> Run Directly (Blocks)
        </button>
        <button
          onClick={simulateToThread}
          disabled={mode !== "idle"}
          className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg disabled:opacity-50 hover:bg-green-600"
        >
          <Zap className="w-4 h-4" /> Use to_thread
        </button>
      </div>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="p-4 rounded-xl bg-gray-100 dark:bg-gray-800">
          <h3 className="font-semibold mb-3 dark:text-white flex items-center gap-2">
            <Clock className="w-5 h-5" /> Blocking Function
          </h3>
          <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-red-500"
              animate={{ width: `${directProgress}%` }}
            />
          </div>
          <p className="text-sm mt-2 dark:text-gray-300">
            Progress: {directProgress}%
          </p>
        </div>
        <div className="p-4 rounded-xl bg-gray-100 dark:bg-gray-800">
          <h3 className="font-semibold mb-3 dark:text-white flex items-center gap-2">
            <Zap className="w-5 h-5" /> Event Loop (Other Tasks)
          </h3>
          <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-blue-500"
              animate={{ width: `${threadProgress}%` }}
            />
          </div>
          <p className="text-sm mt-2 dark:text-gray-300">
            {eventLoopFree ? "Free to handle tasks" : "Blocked!"}
          </p>
        </div>
      </div>
      <AnimatePresence>
        {mode === "direct" && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-4 bg-red-100 dark:bg-red-900/30 rounded-lg text-red-700 dark:text-red-300"
          >
            Event loop is blocked - no other coroutines can run!
          </motion.div>
        )}
        {mode === "threaded" && directProgress === 100 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-4 bg-green-100 dark:bg-green-900/30 rounded-lg text-green-700 dark:text-green-300 flex items-center gap-2"
          >
            <CheckCircle className="w-5 h-5" /> to_thread runs blocking code
            in a thread - event loop stays free!
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
