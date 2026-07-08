"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, ChevronRight, ChevronLeft, RotateCcw, Cpu, ListTodo, Rocket, CheckCircle, XCircle } from "lucide-react";

interface Step {
  title: string;
  icon: React.ReactNode;
  description: string;
  code: string;
  color: string;
}

const steps: Step[] = [
  {
    title: "Create Event Loop",
    icon: <Cpu className="w-5 h-5" />,
    description: "Create or get a reference to the event loop. In Python 3.10+, asyncio.run() handles this automatically.",
    code: "loop = asyncio.new_event_loop()\n# or simply: asyncio.run(main())",
    color: "bg-blue-500",
  },
  {
    title: "Add Tasks",
    icon: <ListTodo className="w-5 h-5" />,
    description: "Schedule coroutines as tasks using create_task(). They are added to the ready queue but don't execute yet.",
    code: "task1 = asyncio.create_task(fetch_data())\ntask2 = asyncio.create_task(process_data())",
    color: "bg-purple-500",
  },
  {
    title: "Run the Loop",
    icon: <Rocket className="w-5 h-5" />,
    description: "The event loop starts processing. It picks ready tasks, runs them until they await, then switches to the next ready task.",
    code: "await asyncio.gather(task1, task2)\n# Loop runs until all tasks complete",
    color: "bg-amber-500",
  },
  {
    title: "Tasks Execute",
    icon: <Play className="w-5 h-5" />,
    description: "Tasks run concurrently via cooperative multitasking. When one awaits I/O, the loop switches to another ready task.",
    code: "# Task 1: await response\n# Loop switches to Task 2\n# Task 2: await db_query\n# Loop switches back when I/O ready",
    color: "bg-green-500",
  },
  {
    title: "All Complete",
    icon: <CheckCircle className="w-5 h-5" />,
    description: "When all tasks are done, gather() returns. Results are collected and exceptions (if any) are re-raised.",
    code: "results = await asyncio.gather(task1, task2)\nprint(results)  # [data1, data2]",
    color: "bg-teal-500",
  },
  {
    title: "Close Loop",
    icon: <XCircle className="w-5 h-5" />,
    description: "The event loop is closed. asyncio.run() handles cleanup automatically. No more tasks can be scheduled.",
    code: "# asyncio.run() closes automatically\nloop.close()  # if manual",
    color: "bg-gray-500",
  },
];

export function EventLoopLifecycle() {
  const [step, setStep] = useState(0);

  const next = () => setStep((s) => Math.min(s + 1, steps.length - 1));
  const prev = () => setStep((s) => Math.max(s - 1, 0));
  const reset = () => setStep(0);
  const current = steps[step];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100">
        Event Loop Lifecycle
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Step through the complete lifecycle of an asyncio event loop, from creation to shutdown.
      </p>

      <div className="flex items-center justify-center gap-1 mb-6 overflow-x-auto pb-2">
        {steps.map((s, i) => (
          <React.Fragment key={i}>
            <motion.button
              whileHover={{ scale: 1.05 }}
              onClick={() => setStep(i)}
              className={`flex items-center gap-1 px-3 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                i === step
                  ? `${s.color} text-white shadow-lg ring-2 ring-offset-1 ring-gray-300 dark:ring-offset-gray-900`
                  : i < step
                  ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500"
              }`}
            >
              {s.icon}
              <span className="hidden sm:inline">{s.title}</span>
            </motion.button>
            {i < steps.length - 1 && (
              <ChevronRight className={`w-4 h-4 flex-shrink-0 ${i < step ? "text-green-400" : "text-gray-300 dark:text-gray-600"}`} />
            )}
          </React.Fragment>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm mb-6"
        >
          <div className="flex items-center gap-3 mb-3">
            <div className={`p-2 rounded-lg ${current.color} text-white`}>{current.icon}</div>
            <div>
              <span className="text-xs text-gray-400 dark:text-gray-500">Step {step + 1} of {steps.length}</span>
              <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100">{current.title}</h4>
            </div>
          </div>
          <p className="text-gray-700 dark:text-gray-300 mb-4">{current.description}</p>
          <pre className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 text-sm font-mono overflow-x-auto text-gray-800 dark:text-gray-200">
            {current.code}
          </pre>
        </motion.div>
      </AnimatePresence>

      <div className="flex gap-3 justify-center">
        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={prev} disabled={step === 0}
          className="flex items-center gap-1 px-5 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 disabled:opacity-40 text-white font-medium shadow transition-colors">
          <ChevronLeft className="w-4 h-4" /> Prev
        </motion.button>
        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={reset}
          className="flex items-center gap-1 px-5 py-2 rounded-lg bg-gray-500 hover:bg-gray-600 text-white font-medium shadow transition-colors">
          <RotateCcw className="w-4 h-4" />
        </motion.button>
        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={next} disabled={step === steps.length - 1}
          className="flex items-center gap-1 px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-medium shadow transition-colors">
          Next <ChevronRight className="w-4 h-4" />
        </motion.button>
      </div>
    </div>
  );
}
