"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HelpCircle, ArrowRight, Check, RotateCcw, Zap, Cpu, Users } from "lucide-react";

interface Step {
  id: string;
  question: string;
  options?: { label: string; next: string }[];
  result?: string;
  recommendation?: string;
  icon?: typeof Zap;
}

const steps: Record<string, Step> = {
  start: {
    id: "start",
    question: "What type of work is your task performing?",
    options: [
      { label: "Network / API calls", next: "io_type" },
      { label: "Math / Data processing", next: "cpu_type" },
      { label: "File system operations", next: "file_type" },
    ],
  },
  io_type: {
    id: "io_type",
    question: "How many concurrent requests?",
    options: [
      { label: "Many (100+)", next: "io_result_many" },
      { label: "Few (< 20)", next: "io_result_few" },
    ],
  },
  io_result_many: {
    id: "io_result_many",
    question: "Recommendation:",
    result: "Pure async with aiohttp/httpx",
    recommendation:
      "Use native async I/O for maximum concurrency. No threads needed for network calls.",
    icon: Zap,
  },
  io_result_few: {
    id: "io_result_few",
    question: "Recommendation:",
    result: "asyncio + to_thread for blocking libs",
    recommendation:
      "Use async with to_thread() wrap for libraries that don't support async natively.",
    icon: Zap,
  },
  cpu_type: {
    id: "cpu_type",
    question: "Is the task embarrassingly parallel?",
    options: [
      { label: "Yes (independent chunks)", next: "cpu_result_parallel" },
      { label: "No (sequential dependencies)", next: "cpu_result_sequential" },
    ],
  },
  cpu_result_parallel: {
    id: "cpu_result_parallel",
    question: "Recommendation:",
    result: "ProcessPoolExecutor",
    recommendation:
      "Use multiple processes for true CPU parallelism. Bypasses GIL completely.",
    icon: Cpu,
  },
  cpu_result_sequential: {
    id: "cpu_result_sequential",
    question: "Recommendation:",
    result: "Single async task + run_in_executor",
    recommendation:
      "Run CPU work in executor to avoid blocking the event loop. Use process pool.",
    icon: Cpu,
  },
  file_type: {
    id: "file_type",
    question: "Can you use async file libraries?",
    options: [
      { label: "Yes (aiofiles)", next: "file_result_async" },
      { label: "No (sync libraries only)", next: "file_result_sync" },
    ],
  },
  file_result_async: {
    id: "file_result_async",
    question: "Recommendation:",
    result: "aiofiles + native async",
    recommendation:
      "Use aiofiles for non-blocking file I/O directly in async context.",
    icon: Zap,
  },
  file_result_sync: {
    id: "file_result_sync",
    question: "Recommendation:",
    result: "asyncio.to_thread()",
    recommendation:
      "Wrap blocking file calls with to_thread() to keep event loop responsive.",
    icon: Users,
  },
};

export default function DecisionGuideFlowchart() {
  const [currentStep, setCurrentStep] = useState("start");
  const [history, setHistory] = useState<string[]>([]);

  const step = steps[currentStep];

  const handleChoice = (next: string) => {
    setHistory((prev) => [...prev, currentStep]);
    setCurrentStep(next);
  };

  const reset = () => {
    setCurrentStep("start");
    setHistory([]);
  };

  const Icon = step.icon || HelpCircle;

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
        <HelpCircle className="w-6 h-6" /> Async Decision Guide
      </h2>
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="p-6 rounded-xl bg-white dark:bg-gray-800 border dark:border-gray-700"
        >
          <div className="flex items-center gap-3 mb-4">
            <Icon className="w-8 h-8 text-blue-500" />
            <h3 className="text-lg font-semibold dark:text-white">
              {step.question}
            </h3>
          </div>
          {step.result ? (
            <div className="space-y-3">
              <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
                <p className="font-semibold text-green-700 dark:text-green-300">
                  {step.result}
                </p>
                <p className="text-sm text-green-600 dark:text-green-400 mt-1">
                  {step.recommendation}
                </p>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                <Check className="w-4 h-4" /> Steps taken: {history.length}
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {(step.options || []).map((opt, i) => (
                <motion.button
                  key={i}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handleChoice(opt.next)}
                  className="w-full flex items-center justify-between p-3 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-left dark:text-white"
                >
                  <span>{opt.label}</span>
                  <ArrowRight className="w-4 h-4" />
                </motion.button>
              ))}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
      {history.length > 0 && (
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
        >
          <RotateCcw className="w-4 h-4" /> Start Over
        </button>
      )}
    </div>
  );
}
