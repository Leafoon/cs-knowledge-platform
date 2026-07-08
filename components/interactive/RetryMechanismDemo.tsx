"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RefreshCw, AlertTriangle, CheckCircle, Clock, Play, RotateCcw } from "lucide-react";

interface RetryAttempt {
  attempt: number;
  delay: number;
  status: "pending" | "retrying" | "failed" | "success";
}

export default function RetryMechanismDemo() {
  const [maxRetries, setMaxRetries] = useState(4);
  const [failChance, setFailChance] = useState(0.7);
  const [attempts, setAttempts] = useState<RetryAttempt[]>([]);
  const [running, setRunning] = useState(false);
  const [finalStatus, setFinalStatus] = useState<"success" | "exhausted" | null>(null);

  const getDelay = (attempt: number) => Math.min(1000 * Math.pow(2, attempt), 8000);

  const runDemo = useCallback(async () => {
    setRunning(true);
    setFinalStatus(null);
    const newAttempts: RetryAttempt[] = [];
    setAttempts(newAttempts);

    for (let i = 0; i <= maxRetries; i++) {
      const delay = i === 0 ? 0 : getDelay(i - 1);
      newAttempts.push({ attempt: i, delay, status: "pending" });
      setAttempts([...newAttempts]);

      if (delay > 0) {
        newAttempts[i].status = "retrying";
        setAttempts([...newAttempts]);
        await new Promise((r) => setTimeout(r, Math.min(delay / 10, 1500)));
      }

      const fails = Math.random() < failChance;
      if (fails && i < maxRetries) {
        newAttempts[i].status = "failed";
        setAttempts([...newAttempts]);
        await new Promise((r) => setTimeout(r, 400));
      } else {
        newAttempts[i].status = fails ? "failed" : "success";
        setAttempts([...newAttempts]);
        setFinalStatus(fails ? "exhausted" : "success");
        break;
      }
    }
    setRunning(false);
  }, [maxRetries, failChance]);

  const reset = () => {
    setAttempts([]);
    setRunning(false);
    setFinalStatus(null);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
        <RefreshCw className="w-6 h-6" /> Retry Mechanism with Backoff
      </h2>
      <div className="grid md:grid-cols-2 gap-4">
        <div className="flex items-center gap-2">
          <label className="dark:text-gray-300">Max Retries:</label>
          <input
            type="range"
            min={1}
            max={6}
            value={maxRetries}
            onChange={(e) => setMaxRetries(+e.target.value)}
            disabled={running}
            className="w-32"
          />
          <span className="font-mono dark:text-white">{maxRetries}</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="dark:text-gray-300">Fail Chance:</label>
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.1}
            value={failChance}
            onChange={(e) => setFailChance(+e.target.value)}
            disabled={running}
            className="w-32"
          />
          <span className="font-mono dark:text-white">{(failChance * 100).toFixed(0)}%</span>
        </div>
      </div>
      <div className="flex gap-4">
        <button
          onClick={runDemo}
          disabled={running}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 hover:bg-blue-600"
        >
          <Play className="w-4 h-4" /> Run Task
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
        >
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
      </div>
      <div className="p-4 rounded-xl bg-gray-100 dark:bg-gray-800 min-h-[200px]">
        <h3 className="font-semibold mb-3 dark:text-white">Retry Timeline</h3>
        <div className="space-y-3">
          <AnimatePresence>
            {attempts.map((attempt, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-3"
              >
                <div className="w-20 text-sm font-mono dark:text-gray-300">
                  Attempt {attempt.attempt}
                </div>
                {attempt.delay > 0 && (
                  <div className="flex items-center gap-1 text-sm text-orange-500">
                    <Clock className="w-4 h-4" />
                    <span>{attempt.delay}ms delay</span>
                  </div>
                )}
                <motion.div
                  animate={
                    attempt.status === "retrying"
                      ? { opacity: [0.5, 1, 0.5] }
                      : {}
                  }
                  transition={{ repeat: attempt.status === "retrying" ? Infinity : 0 }}
                  className={`px-3 py-1 rounded-full text-sm ${
                    attempt.status === "success"
                      ? "bg-green-500 text-white"
                      : attempt.status === "failed"
                      ? "bg-red-500 text-white"
                      : "bg-orange-500 text-white"
                  }`}
                >
                  {attempt.status === "success" && <CheckCircle className="w-4 h-4 inline mr-1" />}
                  {attempt.status === "failed" && <AlertTriangle className="w-4 h-4 inline mr-1" />}
                  {attempt.status === "retrying" && <RefreshCw className="w-4 h-4 inline mr-1 animate-spin" />}
                  {attempt.status}
                </motion.div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
      <AnimatePresence>
        {finalStatus && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`p-4 rounded-lg flex items-center gap-2 ${
              finalStatus === "success"
                ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                : "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
            }`}
          >
            {finalStatus === "success" ? (
              <CheckCircle className="w-5 h-5" />
            ) : (
              <AlertTriangle className="w-5 h-5" />
            )}
            {finalStatus === "success"
              ? `Task succeeded after ${attempts.length} attempt(s)!`
              : `Task failed after ${maxRetries + 1} attempts (max retries exhausted).`}
          </motion.div>
        )}
      </AnimatePresence>
      <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
        <h4 className="font-semibold dark:text-white mb-2">Backoff Formula:</h4>
        <p className="text-sm font-mono dark:text-gray-300">
          delay = min(1000 × 2^attempt, 8000) ms
        </p>
      </div>
    </div>
  );
}
