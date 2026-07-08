"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, CheckCircle, MessageSquare, Play, RefreshCw } from "lucide-react";

export default function CrossThreadCallDemo() {
  const [messages, setMessages] = useState<string[]>([]);
  const [phase, setPhase] = useState<"idle" | "sending" | "received">("idle");
  const [loopBusy, setLoopBusy] = useState(false);

  const simulate = useCallback(async () => {
    setPhase("sending");
    setMessages([]);
    setLoopBusy(true);
    await new Promise((r) => setTimeout(r, 600));
    setMessages((prev) => [...prev, "[Thread] Computing result..."]);
    await new Promise((r) => setTimeout(r, 800));
    setMessages((prev) => [...prev, "[Thread] Result ready: 42"]);
    await new Promise((r) => setTimeout(r, 400));
    setMessages((prev) => [
      ...prev,
      "[Thread] Calling loop.call_soon_threadsafe()",
    ]);
    await new Promise((r) => setTimeout(r, 600));
    setMessages((prev) => [...prev, "[Loop] Callback scheduled in event loop"]);
    await new Promise((r) => setTimeout(r, 500));
    setMessages((prev) => [...prev, "[Loop] Processing result: 42"]);
    setLoopBusy(false);
    setPhase("received");
  }, []);

  const reset = () => {
    setMessages([]);
    setPhase("idle");
    setLoopBusy(false);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
        <MessageSquare className="w-6 h-6" /> Cross-Thread Communication
      </h2>
      <p className="text-gray-600 dark:text-gray-400">
        Demonstrate <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">call_soon_threadsafe()</code> — safely schedule callbacks from worker threads.
      </p>
      <div className="flex gap-4">
        <button
          onClick={simulate}
          disabled={phase !== "idle"}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 hover:bg-blue-600"
        >
          <Play className="w-4 h-4" /> Run Demo
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
        >
          <RefreshCw className="w-4 h-4" /> Reset
        </button>
      </div>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="p-4 rounded-xl bg-orange-50 dark:bg-orange-900/20">
          <h3 className="font-semibold mb-3 dark:text-white">Worker Thread</h3>
          <div className="space-y-2">
            {messages
              .filter((m) => m.startsWith("[Thread]"))
              .map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="p-2 bg-orange-100 dark:bg-orange-900/40 rounded text-sm dark:text-orange-200"
                >
                  {msg}
                </motion.div>
              ))}
          </div>
        </div>
        <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-900/20">
          <h3 className="font-semibold mb-3 dark:text-white flex items-center gap-2">
            Event Loop
            {loopBusy && (
              <motion.span
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ repeat: Infinity }}
                className="text-xs bg-green-500 text-white px-2 py-0.5 rounded-full"
              >
                Free
              </motion.span>
            )}
          </h3>
          <div className="space-y-2">
            {messages
              .filter((m) => m.startsWith("[Loop]"))
              .map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="p-2 bg-blue-100 dark:bg-blue-900/40 rounded text-sm dark:text-blue-200"
                >
                  {msg}
                </motion.div>
              ))}
          </div>
        </div>
      </div>
      <AnimatePresence>
        {phase === "sending" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-center gap-2 text-orange-500"
          >
            <ArrowRight className="w-5 h-5 animate-pulse" />
            <span>Thread → Event Loop communication in progress...</span>
          </motion.div>
        )}
        {phase === "received" && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-green-100 dark:bg-green-900/30 rounded-lg text-green-700 dark:text-green-300 flex items-center gap-2"
          >
            <CheckCircle className="w-5 h-5" /> Result safely delivered to
            event loop via call_soon_threadsafe!
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
