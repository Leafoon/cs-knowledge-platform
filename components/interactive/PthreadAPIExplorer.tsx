"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Code, Terminal, AlertTriangle, ChevronRight, BookOpen, Search } from "lucide-react";

type Category = "create" | "join" | "exit" | "mutex" | "cond" | "misc";

const CATEGORIES: { id: Category; label: string; icon: string }[] = [
  { id: "create", label: "Thread Creation", icon: " +" },
  { id: "join", label: "Thread Joining", icon: ">>" },
  { id: "exit", label: "Thread Exit/Cancel", icon: "x" },
  { id: "mutex", label: "Mutex", icon: "L" },
  { id: "cond", label: "Condition Variables", icon: "?" },
  { id: "misc", label: "Miscellaneous", icon: "~" },
];

type APIFunction = {
  name: string;
  signature: string;
  params: { name: string; type: string; desc: string }[];
  returns: string;
  code: string;
  pitfalls: string[];
};

const API_DATA: Record<Category, APIFunction[]> = {
  create: [
    {
      name: "pthread_create",
      signature: "int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg);",
      params: [
        { name: "thread", type: "pthread_t *", desc: "Pointer to thread ID (output)" },
        { name: "attr", type: "const pthread_attr_t *", desc: "Thread attributes (NULL for defaults)" },
        { name: "start_routine", type: "void *(*)(void *)", desc: "Function the thread will execute" },
        { name: "arg", type: "void *", desc: "Argument passed to start_routine" },
      ],
      returns: "0 on success, error number on failure",
      code: `#include <pthread.h>
#include <stdio.h>

void *worker(void *arg) {
    int id = *(int *)arg;
    printf("Thread %d running\\n", id);
    return NULL;
}

int main() {
    pthread_t tid;
    int id = 1;
    pthread_create(&tid, NULL, worker, &id);
    pthread_join(tid, NULL);
    return 0;
}`,
      pitfalls: [
        "Passing address of a local variable that goes out of scope before the thread reads it",
        "Not checking the return value for errors",
        "Passing a stack-allocated variable that is modified before thread reads it",
      ],
    },
  ],
  join: [
    {
      name: "pthread_join",
      signature: "int pthread_join(pthread_t thread, void **retval);",
      params: [
        { name: "thread", type: "pthread_t", desc: "Thread ID to wait for" },
        { name: "retval", type: "void **", desc: "Pointer to store thread's return value (NULL to ignore)" },
      ],
      returns: "0 on success, error number on failure",
      code: `void *result;
pthread_join(tid, &result);
printf("Thread returned: %p\\n", result);`,
      pitfalls: [
        "Joining a thread that has already been joined (undefined behavior)",
        "Joining a detached thread (returns EINVAL)",
        "Forgetting to join causes resource leak (zombie thread)",
      ],
    },
  ],
  exit: [
    {
      name: "pthread_exit",
      signature: "void pthread_exit(void *retval);",
      params: [
        { name: "retval", type: "void *", desc: "Return value available to pthread_join" },
      ],
      returns: "Does not return (terminates calling thread)",
      code: `void *worker(void *arg) {
    // Do work...
    int *result = malloc(sizeof(int));
    *result = 42;
    pthread_exit(result);  // Return 42 to joiner
}`,
      pitfalls: [
        "Returning pointer to a local variable — it will be invalid after thread exits",
        "Using pthread_exit in main() before all threads finish causes issues",
        "exit() terminates the entire process; use pthread_exit() for just the thread",
      ],
    },
    {
      name: "pthread_cancel",
      signature: "int pthread_cancel(pthread_t thread);",
      params: [
        { name: "thread", type: "pthread_t", desc: "Thread ID to cancel" },
      ],
      returns: "0 on success, error number on failure",
      code: `pthread_cancel(tid);
// Thread will terminate at next cancellation point
// (e.g., pthread_join, read, write, sleep)`,
      pitfalls: [
        "Cancellation only happens at cancellation points",
        "Resources (locks, memory) may not be cleaned up without cleanup handlers",
        "pthread_setcancelstate can disable cancellation",
      ],
    },
  ],
  mutex: [
    {
      name: "pthread_mutex_init",
      signature: "int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr);",
      params: [
        { name: "mutex", type: "pthread_mutex_t *", desc: "Pointer to mutex to initialize" },
        { name: "attr", type: "const pthread_mutexattr_t *", desc: "Mutex attributes (NULL for defaults)" },
      ],
      returns: "0 on success, error number on failure",
      code: `pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
// Or dynamically:
pthread_mutex_t lock;
pthread_mutex_init(&lock, NULL);`,
      pitfalls: [
        "Forgetting to destroy dynamically initialized mutexes (memory leak)",
        "Using PTHREAD_MUTEX_INITIALIZER on a dynamically allocated mutex",
        "Not using error checking type for debugging deadlocks",
      ],
    },
    {
      name: "pthread_mutex_lock",
      signature: "int pthread_mutex_lock(pthread_mutex_t *mutex);",
      params: [
        { name: "mutex", type: "pthread_mutex_t *", desc: "Pointer to mutex to lock" },
      ],
      returns: "0 on success, error number on failure",
      code: `pthread_mutex_lock(&shared_lock);
// Critical section — only one thread here
shared_counter++;
pthread_mutex_unlock(&shared_lock);`,
      pitfalls: [
        "Forgetting to unlock causes deadlock",
        "Locking a mutex you already hold (deadlock with default type)",
        "Not handling lock failures",
      ],
    },
    {
      name: "pthread_mutex_unlock",
      signature: "int pthread_mutex_unlock(pthread_mutex_t *mutex);",
      params: [
        { name: "mutex", type: "pthread_mutex_t *", desc: "Pointer to mutex to unlock" },
      ],
      returns: "0 on success, error number on failure",
      code: `// Always unlock in all code paths
pthread_mutex_lock(&lock);
if (error_condition) {
    pthread_mutex_unlock(&lock);
    return -1;
}
// ... normal work ...
pthread_mutex_unlock(&lock);`,
      pitfalls: [
        "Unlocking a mutex you don't own (undefined behavior)",
        "Missing unlock on error paths — use goto cleanup pattern",
        "Not unlocking before thread exit",
      ],
    },
  ],
  cond: [
    {
      name: "pthread_cond_init",
      signature: "int pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *attr);",
      params: [
        { name: "cond", type: "pthread_cond_t *", desc: "Pointer to condition variable to initialize" },
        { name: "attr", type: "const pthread_condattr_t *", desc: "Attributes (NULL for defaults)" },
      ],
      returns: "0 on success, error number on failure",
      code: `pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
// Or dynamically:
pthread_cond_t cond;
pthread_cond_init(&cond, NULL);`,
      pitfalls: [
        "Forgetting to destroy dynamically initialized cond vars",
        "Using condition variable without associated mutex",
      ],
    },
    {
      name: "pthread_cond_wait",
      signature: "int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);",
      params: [
        { name: "cond", type: "pthread_cond_t *", desc: "Condition variable to wait on" },
        { name: "mutex", type: "pthread_mutex_t *", desc: "Associated mutex (atomically released)" },
      ],
      returns: "0 on success, error number on failure",
      code: `pthread_mutex_lock(&lock);
while (!condition_met) {          // ALWAYS use while loop
    pthread_cond_wait(&cond, &lock);  // Atomically: unlock + sleep
}
// Process data (lock is re-acquired)
pthread_mutex_unlock(&lock);`,
      pitfalls: [
        "Using if() instead of while() — spurious wakeups can cause bugs",
        "Not holding the mutex before calling wait",
        "Signal can be lost if sent before wait — always check predicate in loop",
      ],
    },
    {
      name: "pthread_cond_signal",
      signature: "int pthread_cond_signal(pthread_cond_t *cond);",
      params: [
        { name: "cond", type: "pthread_cond_t *", desc: "Condition variable to signal" },
      ],
      returns: "0 on success, error number on failure",
      code: `pthread_mutex_lock(&lock);
data_ready = 1;
pthread_cond_signal(&cond);  // Wake one waiting thread
pthread_mutex_unlock(&lock);`,
      pitfalls: [
        "Signaling without holding the mutex can cause lost wakeups",
        "Signal wakes only one thread; use broadcast for all",
        "Signaling when no thread is waiting is a no-op (not an error)",
      ],
    },
    {
      name: "pthread_cond_broadcast",
      signature: "int pthread_cond_broadcast(pthread_cond_t *cond);",
      params: [
        { name: "cond", type: "pthread_cond_t *", desc: "Condition variable to broadcast on" },
      ],
      returns: "0 on success, error number on failure",
      code: `// Wake ALL waiting threads
pthread_mutex_lock(&lock);
shutdown = 1;
pthread_cond_broadcast(&cond);
pthread_mutex_unlock(&lock);`,
      pitfalls: [
        "All awakened threads will compete for the mutex — thundering herd",
        "Use when all waiters need to recheck state, not just one",
      ],
    },
  ],
  misc: [
    {
      name: "pthread_self",
      signature: "pthread_t pthread_self(void);",
      params: [],
      returns: "Thread ID of the calling thread",
      code: `printf("My thread ID: %lu\\n", pthread_self());`,
      pitfalls: [
        "Return value is opaque — don't compare with ==, use pthread_equal",
        "Thread ID may be a struct on some platforms, not just an integer",
      ],
    },
    {
      name: "pthread_detach",
      signature: "int pthread_detach(pthread_t thread);",
      params: [
        { name: "thread", type: "pthread_t", desc: "Thread ID to detach" },
      ],
      returns: "0 on success, error number on failure",
      code: `pthread_t tid;
pthread_create(&tid, NULL, worker, NULL);
pthread_detach(tid);  // Thread resources auto-cleaned on exit
// Cannot join this thread anymore!`,
      pitfalls: [
        "Cannot join a detached thread — resources are automatically released",
        "Detaching an already-detached thread is undefined behavior",
        "Main thread exiting may kill detached threads abruptly",
      ],
    },
    {
      name: "pthread_equal",
      signature: "int pthread_equal(pthread_t t1, pthread_t t2);",
      params: [
        { name: "t1", type: "pthread_t", desc: "First thread ID" },
        { name: "t2", type: "pthread_t", desc: "Second thread ID" },
      ],
      returns: "Non-zero if equal, 0 if different",
      code: `if (pthread_equal(tid1, tid2)) {
    printf("Same thread!\\n");
}`,
      pitfalls: [
        "Must use this function, not == operator (pthread_t may be a struct)",
        "Comparing with uninitialized thread ID is undefined",
      ],
    },
  ],
};

function CodeBlock({ code }: { code: string }) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
      <div className="flex items-center gap-2 mb-2">
        <Terminal className="w-4 h-4 text-green-400" />
        <span className="text-xs text-green-400 font-mono">C Code Example</span>
      </div>
      <pre className="text-sm font-mono text-green-300 whitespace-pre-wrap leading-relaxed">
        {code}
      </pre>
    </div>
  );
}

export default function PthreadAPIExplorer() {
  const [activeCategory, setActiveCategory] = useState<Category>("create");
  const [activeFunc, setActiveFunc] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");

  const allFunctions = Object.entries(API_DATA).flatMap(([cat, fns]) =>
    fns.map((fn) => ({ ...fn, category: cat as Category }))
  );

  const filteredFunctions = searchTerm
    ? allFunctions.filter((fn) => fn.name.toLowerCase().includes(searchTerm.toLowerCase()))
    : null;

  const currentFunctions = filteredFunctions || API_DATA[activeCategory];
  const currentFunc = currentFunctions[activeFunc] || currentFunctions[0];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center flex items-center justify-center gap-2">
        <Code className="w-7 h-7 text-cyan-600" />
        POSIX Thread (Pthread) API Explorer
      </h2>
      <p className="text-center text-slate-500 dark:text-gray-400 text-sm mb-6">
        Interactive reference for the POSIX threads programming interface
      </p>

      {/* Search */}
      <div className="relative max-w-md mx-auto mb-6">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
        <input
          type="text"
          placeholder="Search API functions..."
          value={searchTerm}
          onChange={(e) => { setSearchTerm(e.target.value); setActiveFunc(0); }}
          className="w-full pl-10 pr-4 py-2 rounded-lg border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-700 dark:text-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
        />
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left Panel - Categories & Functions */}
        <div className="w-full lg:w-64 shrink-0">
          {!searchTerm && (
            <div className="space-y-1 mb-4">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => { setActiveCategory(cat.id); setActiveFunc(0); }}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
                    activeCategory === cat.id
                      ? "bg-cyan-600 text-white shadow-md"
                      : "bg-white dark:bg-gray-700 text-slate-700 dark:text-gray-200 hover:bg-cyan-50 dark:hover:bg-gray-600 border border-slate-200 dark:border-gray-600"
                  }`}
                >
                  <span className="font-mono text-xs opacity-60">{cat.icon}</span>
                  {cat.label}
                </button>
              ))}
            </div>
          )}

          {/* Function List */}
          <div className="space-y-1">
            {currentFunctions.map((fn, i) => (
              <button
                key={fn.name}
                onClick={() => setActiveFunc(i)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all flex items-center justify-between ${
                  activeFunc === i
                    ? "bg-cyan-100 dark:bg-cyan-900/40 text-cyan-800 dark:text-cyan-300 font-semibold"
                    : "bg-white dark:bg-gray-700 text-slate-600 dark:text-gray-300 hover:bg-slate-50 dark:hover:bg-gray-600"
                }`}
              >
                <span className="font-mono text-xs">{fn.name}</span>
                <ChevronRight className="w-3 h-3 opacity-40" />
              </button>
            ))}
          </div>
        </div>

        {/* Right Panel - Detail */}
        <div className="flex-1 min-w-0">
          <AnimatePresence mode="wait">
            {currentFunc && (
              <motion.div
                key={currentFunc.name}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className="space-y-5"
              >
                {/* Function Name */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-slate-200 dark:border-gray-700">
                  <h3 className="text-xl font-bold text-cyan-700 dark:text-cyan-400 font-mono mb-3">
                    {currentFunc.name}
                  </h3>
                  <div className="bg-slate-100 dark:bg-gray-900 rounded-lg p-3 overflow-x-auto">
                    <code className="text-sm font-mono text-slate-800 dark:text-gray-200 whitespace-pre-wrap break-all">
                      {currentFunc.signature}
                    </code>
                  </div>
                </div>

                {/* Parameters */}
                {currentFunc.params.length > 0 && (
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-slate-200 dark:border-gray-700">
                    <h4 className="font-semibold text-slate-700 dark:text-gray-200 mb-3">Parameters</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="bg-slate-100 dark:bg-gray-700">
                            <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Name</th>
                            <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Type</th>
                            <th className="px-3 py-2 text-left text-slate-600 dark:text-gray-300">Description</th>
                          </tr>
                        </thead>
                        <tbody>
                          {currentFunc.params.map((p, i) => (
                            <tr key={p.name} className="border-t border-slate-100 dark:border-gray-700">
                              <td className="px-3 py-2 font-mono text-cyan-700 dark:text-cyan-400 font-medium">{p.name}</td>
                              <td className="px-3 py-2 font-mono text-xs text-purple-600 dark:text-purple-400">{p.type}</td>
                              <td className="px-3 py-2 text-slate-600 dark:text-gray-300">{p.desc}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Return Value */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-slate-200 dark:border-gray-700">
                  <h4 className="font-semibold text-slate-700 dark:text-gray-200 mb-2">Return Value</h4>
                  <p className="text-sm text-slate-600 dark:text-gray-300 font-mono bg-emerald-50 dark:bg-emerald-900/30 px-3 py-2 rounded-lg">
                    {currentFunc.returns}
                  </p>
                </div>

                {/* Code Example */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-slate-200 dark:border-gray-700">
                  <h4 className="font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
                    <BookOpen className="w-4 h-4 text-cyan-600" />
                    Code Example
                  </h4>
                  <CodeBlock code={currentFunc.code} />
                </div>

                {/* Pitfalls */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-slate-200 dark:border-gray-700">
                  <h4 className="font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                    Common Pitfalls
                  </h4>
                  <ul className="space-y-2">
                    {currentFunc.pitfalls.map((p, i) => (
                      <motion.li
                        key={i}
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="flex items-start gap-2 text-sm text-slate-600 dark:text-gray-300"
                      >
                        <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
                        {p}
                      </motion.li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
