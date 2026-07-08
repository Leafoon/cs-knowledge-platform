"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, Circle, ChevronDown, ChevronUp, BookOpen, Star } from "lucide-react";

interface Practice {
  id: number;
  title: string;
  good: string;
  bad: string;
  tip: string;
}

const PRACTICES: Practice[] = [
  { id: 1, title: "Always await coroutines", good: "await fetch_data()", bad: "fetch_data()  # coroutine never runs!", tip: "An unawaited coroutine is silently discarded." },
  { id: 2, title: "Use asyncio.run() as entry point", good: "asyncio.run(main())", bad: "loop.run_until_complete(main())", tip: "asyncio.run() handles cleanup automatically." },
  { id: 3, title: "Keep references to tasks", good: "task = asyncio.create_task(work())\nawait task", bad: "asyncio.create_task(work())  # lost!", tip: "Unreferenced tasks may be garbage collected." },
  { id: 4, title: "Avoid blocking calls", good: "await asyncio.sleep(1)", bad: "time.sleep(1)  # blocks loop!", tip: "Use async versions of I/O and sleep." },
  { id: 5, title: "Use Locks for shared state", good: "async with lock:\n    counter += 1", bad: "counter += 1  # race condition!", tip: "Between awaits, other tasks can run." },
  { id: 6, title: "Use gather() with return_exceptions", good: "await gather(*tasks,\n    return_exceptions=True)", bad: "await gather(*tasks)\n# first error cancels rest", tip: "Collect all results, handle errors after." },
  { id: 7, title: "Limit concurrency with Semaphore", good: "sem = Semaphore(100)\nasync with sem:\n    await work()", bad: "for i in range(100000):\n    create_task(work(i))", tip: "Prevent memory exhaustion from unbounded tasks." },
  { id: 8, title: "Use structured concurrency", good: "async with TaskGroup() as tg:\n    tg.create_task(a())\n    tg.create_task(b())", bad: "t1 = create_task(a())\nt2 = create_task(b())\nawait t1; await t2", tip: "TaskGroup ensures all tasks complete or cancel." },
  { id: 9, title: "Handle cancellation gracefully", good: "try:\n    await work()\nexcept CancelledError:\n    cleanup()", bad: "await work()\n# no cleanup on cancel!", tip: "Always clean up resources on cancellation." },
  { id: 10, title: "Enable debug mode during development", good: "asyncio.run(main(), debug=True)", bad: "# never check for warnings", tip: "Catches unawaited coroutines and slow callbacks." },
];

export function BestPracticesChecklist() {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [learned, setLearned] = useState<Set<number>>(new Set());

  const toggleLearned = (id: number) => {
    setLearned((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const progress = (learned.size / PRACTICES.length) * 100;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center flex items-center justify-center gap-2">
        <Star className="w-5 h-5 text-emerald-500" /> Async Best Practices Checklist
      </h3>

      {/* Progress */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-slate-500 dark:text-gray-400 mb-1">
          <span>Progress</span>
          <span>{learned.size}/{PRACTICES.length} learned</span>
        </div>
        <div className="h-3 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div animate={{ width: `${progress}%` }} className="h-full bg-emerald-500 rounded-full" />
        </div>
      </div>

      {/* Checklist */}
      <div className="space-y-2">
        {PRACTICES.map((p) => {
          const isExpanded = expanded === p.id;
          const isLearned = learned.has(p.id);
          return (
            <div key={p.id} className={`rounded-lg border overflow-hidden transition-colors ${isLearned ? "border-emerald-300 dark:border-emerald-700 bg-emerald-50/50 dark:bg-emerald-900/10" : "border-slate-200 dark:border-gray-700 bg-white dark:bg-gray-800"}`}>
              <div className="flex items-center gap-3 p-3">
                <button onClick={() => toggleLearned(p.id)} className="flex-shrink-0">
                  {isLearned ? <CheckCircle className="w-5 h-5 text-emerald-500" /> : <Circle className="w-5 h-5 text-slate-300 dark:text-gray-600" />}
                </button>
                <button onClick={() => setExpanded(isExpanded ? null : p.id)} className="flex-1 flex items-center justify-between text-left">
                  <span className={`text-sm ${isLearned ? "text-emerald-700 dark:text-emerald-300 line-through" : "text-slate-700 dark:text-gray-200"}`}>{p.title}</span>
                  {isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
                </button>
              </div>
              <AnimatePresence>
                {isExpanded && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                    <div className="px-3 pb-3 space-y-2">
                      <div className="flex items-start gap-1.5 bg-amber-50 dark:bg-amber-900/20 rounded p-2 border border-amber-200 dark:border-amber-800">
                        <BookOpen className="w-3 h-3 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                        <span className="text-xs text-amber-700 dark:text-amber-300">{p.tip}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-red-50 dark:bg-red-900/20 rounded p-2 border border-red-200 dark:border-red-800">
                          <div className="text-xs font-bold text-red-600 dark:text-red-400 mb-1">Bad</div>
                          <pre className="text-xs font-mono text-red-700 dark:text-red-300 whitespace-pre-wrap">{p.bad}</pre>
                        </div>
                        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded p-2 border border-emerald-200 dark:border-emerald-800">
                          <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400 mb-1">Good</div>
                          <pre className="text-xs font-mono text-emerald-700 dark:text-emerald-300 whitespace-pre-wrap">{p.good}</pre>
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

      {learned.size === PRACTICES.length && (
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="mt-4 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg p-3 border border-emerald-300 dark:border-emerald-700 text-center">
          <span className="text-sm font-bold text-emerald-700 dark:text-emerald-300">All practices learned!</span>
        </motion.div>
      )}
    </div>
  );
}
