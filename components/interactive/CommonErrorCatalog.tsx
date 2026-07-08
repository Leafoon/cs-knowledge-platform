"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, AlertCircle, XCircle, ChevronDown, ChevronUp, Bug } from "lucide-react";

interface ErrorEntry {
  id: number;
  name: string;
  severity: "low" | "medium" | "high";
  wrong: string;
  correct: string;
  explanation: string;
}

const ERRORS: ErrorEntry[] = [
  { id: 1, name: "Unawaited coroutine", severity: "high", wrong: "asyncio.sleep(1)  # not awaited!", correct: "await asyncio.sleep(1)", explanation: "Calling a coroutine without await creates a coroutine object that is never executed, causing a RuntimeWarning." },
  { id: 2, name: "Blocking call in async", severity: "high", wrong: "time.sleep(5)  # blocks event loop!", correct: "await asyncio.sleep(5)", explanation: "time.sleep() blocks the entire event loop. Use asyncio.sleep() to yield control." },
  { id: 3, name: "Missing asyncio.run()", severity: "medium", wrong: "asyncio.get_event_loop().run_until_complete(main())", correct: "asyncio.run(main())", explanation: "asyncio.run() handles setup and teardown properly. Low-level APIs are error-prone." },
  { id: 4, name: "Fire-and-forget task", severity: "high", wrong: "asyncio.create_task(do_work())  # not saved!", correct: "task = asyncio.create_task(do_work())\nawait task", explanation: "Unsaved tasks may be garbage collected. Always keep a reference." },
  { id: 5, name: "Shared mutable state", severity: "medium", wrong: "counter += 1  # not thread-safe!", correct: "async with lock:\n    counter += 1", explanation: "Between await points, tasks can interleave. Use a Lock for shared state." },
  { id: 6, name: "Exception swallowed", severity: "medium", wrong: "asyncio.gather(*tasks)  # errors hidden", correct: "results = await asyncio.gather(*tasks, return_exceptions=True)", explanation: "By default gather() raises the first exception and cancels others. Use return_exceptions to collect all." },
  { id: 7, name: "Deadlock with semaphores", severity: "high", wrong: "await mutex.acquire()\nawait empty.acquire()  # deadlock!", correct: "await empty.acquire()\nawait mutex.acquire()", explanation: "Always acquire semaphores in the same order to prevent deadlock." },
  { id: 8, name: "Creating tasks in loop", severity: "low", wrong: "for i in range(100000):\n    asyncio.create_task(work(i))", correct: "sem = asyncio.Semaphore(100)\nasync def limited(i):\n    async with sem:\n        await work(i)", explanation: "Unbounded task creation exhausts memory. Use a semaphore to limit concurrency." },
];

const severityConfig = {
  low: { color: "bg-yellow-100 text-yellow-700 border-yellow-300 dark:bg-yellow-900/30 dark:text-yellow-400 dark:border-yellow-700", icon: AlertCircle, label: "Low" },
  medium: { color: "bg-orange-100 text-orange-700 border-orange-300 dark:bg-orange-900/30 dark:text-orange-400 dark:border-orange-700", icon: AlertTriangle, label: "Medium" },
  high: { color: "bg-red-100 text-red-700 border-red-300 dark:bg-red-900/30 dark:text-red-400 dark:border-red-700", icon: XCircle, label: "High" },
};

export function CommonErrorCatalog() {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center flex items-center justify-center gap-2">
        <Bug className="w-5 h-5 text-red-500" /> Common Async Error Catalog
      </h3>

      <div className="space-y-2">
        {ERRORS.map((err) => {
          const isExpanded = expanded === err.id;
          const cfg = severityConfig[err.severity];
          const Icon = cfg.icon;
          return (
            <div key={err.id} className={`rounded-lg border ${isExpanded ? "border-red-300 dark:border-red-700" : "border-slate-200 dark:border-gray-700"} bg-white dark:bg-gray-800 overflow-hidden`}>
              <button onClick={() => setExpanded(isExpanded ? null : err.id)} className="w-full flex items-center gap-3 p-3 text-left hover:bg-slate-50 dark:hover:bg-gray-750">
                <Icon className="w-4 h-4 flex-shrink-0 text-red-500" />
                <span className="text-sm font-medium text-slate-700 dark:text-gray-200 flex-1">{err.name}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full border ${cfg.color}`}>{cfg.label}</span>
                {isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>
              <AnimatePresence>
                {isExpanded && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                    <div className="px-3 pb-3 space-y-2">
                      <p className="text-xs text-slate-600 dark:text-gray-300">{err.explanation}</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-red-50 dark:bg-red-900/20 rounded p-2 border border-red-200 dark:border-red-800">
                          <div className="text-xs font-bold text-red-600 dark:text-red-400 mb-1">Wrong</div>
                          <pre className="text-xs font-mono text-red-700 dark:text-red-300 whitespace-pre-wrap">{err.wrong}</pre>
                        </div>
                        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded p-2 border border-emerald-200 dark:border-emerald-800">
                          <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400 mb-1">Correct</div>
                          <pre className="text-xs font-mono text-emerald-700 dark:text-emerald-300 whitespace-pre-wrap">{err.correct}</pre>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </div>
  );
}
