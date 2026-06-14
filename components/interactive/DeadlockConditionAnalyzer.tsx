"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Lock, Clock, Ban, RefreshCw, AlertTriangle, CheckCircle, Zap } from "lucide-react";

interface Condition {
  id: string;
  name: string;
  icon: React.ReactNode;
  color: string;
  activeColor: string;
  definition: string;
  example: string;
  solution: string;
  animationKey: string;
}

const CONDITIONS: Condition[] = [
  {
    id: "mutex",
    name: "Mutual Exclusion",
    icon: <Lock className="w-5 h-5" />,
    color: "from-rose-400 to-rose-500",
    activeColor: "border-rose-400 dark:border-rose-500",
    definition: "At least one resource must be held in a non-sharable mode. Only one process can use the resource at a time.",
    example: "A printer can only serve one print job at a time. Process A holds the printer; Process B cannot use it simultaneously.",
    solution: "Use sharable resources where possible (e.g., read-only files). Spooling can convert non-sharable devices to sharable ones.",
    animationKey: "mutex",
  },
  {
    id: "holdwait",
    name: "Hold and Wait",
    icon: <Clock className="w-5 h-5" />,
    color: "from-amber-400 to-amber-500",
    activeColor: "border-amber-400 dark:border-amber-500",
    definition: "A process holds at least one resource while waiting to acquire additional resources held by other processes.",
    example: "Process A holds the scanner and waits for the printer. Process B holds the printer and waits for the scanner.",
    solution: "Require processes to request all resources at once before execution begins. Or require processes to release all resources before requesting new ones.",
    animationKey: "holdwait",
  },
  {
    id: "nopreempt",
    name: "No Preemption",
    icon: <Ban className="w-5 h-5" />,
    color: "from-violet-400 to-violet-500",
    activeColor: "border-violet-400 dark:border-violet-500",
    definition: "Resources cannot be forcibly taken from a process. A resource can only be released voluntarily by the process holding it.",
    example: "Process A holds the CPU and won't release it until it finishes. Process B needs the CPU but cannot force A to give it up.",
    solution: "Allow the OS to preempt resources. If a process holding resources requests one that cannot be immediately allocated, all its resources are released.",
    animationKey: "nopreempt",
  },
  {
    id: "circular",
    name: "Circular Wait",
    icon: <RefreshCw className="w-5 h-5" />,
    color: "from-cyan-400 to-cyan-500",
    activeColor: "border-cyan-400 dark:border-cyan-500",
    definition: "A circular chain of two or more processes exists, where each process waits for a resource held by the next process in the chain.",
    example: "P1 waits for resource held by P2, P2 waits for resource held by P3, P3 waits for resource held by P1.",
    solution: "Impose a total ordering on resource types. Require each process to request resources in increasing order of enumeration.",
    animationKey: "circular",
  },
];

function MutexAnimation() {
  return (
    <div className="flex items-center justify-center gap-4 py-4">
      <motion.div
        animate={{ scale: [1, 1.05, 1] }}
        transition={{ repeat: Infinity, duration: 1.5 }}
        className="w-16 h-16 rounded-xl bg-gradient-to-br from-rose-400 to-rose-500 flex items-center justify-center shadow-md"
      >
        <Lock className="w-7 h-7 text-white" />
      </motion.div>
      <div className="flex flex-col gap-2">
        {["P1", "P2"].map((p, i) => (
          <div key={p} className="flex items-center gap-2">
            <motion.div
              animate={i === 0 ? { x: [0, -5, 0] } : { x: [0, 5, 0] }}
              transition={{ repeat: Infinity, duration: 1 }}
              className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                i === 0 ? "bg-blue-500" : "bg-slate-400"
              }`}
            >
              {p}
            </motion.div>
            <span className="text-xs text-slate-500 dark:text-gray-400">
              {i === 0 ? "Holds resource" : "Blocked! Cannot access"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HoldWaitAnimation() {
  return (
    <div className="flex items-center justify-center gap-6 py-4">
      {[
        { p: "P1", holds: "A", wants: "B", color: "bg-blue-500" },
        { p: "P2", holds: "B", wants: "A", color: "bg-emerald-500" },
      ].map((item) => (
        <div key={item.p} className="flex flex-col items-center gap-1">
          <motion.div
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ repeat: Infinity, duration: 1.2 }}
            className={`w-12 h-12 rounded-xl ${item.color} flex items-center justify-center text-white font-bold text-sm shadow-md`}
          >
            {item.p}
          </motion.div>
          <span className="text-xs text-amber-600 dark:text-amber-400 font-medium">Holds: {item.holds}</span>
          <motion.span
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ repeat: Infinity, duration: 1 }}
            className="text-xs text-red-500 dark:text-red-400 font-medium"
          >
            Wants: {item.wants}
          </motion.span>
        </div>
      ))}
    </div>
  );
}

function NoPreemptAnimation() {
  return (
    <div className="flex items-center justify-center gap-6 py-4">
      <div className="flex flex-col items-center gap-1">
        <motion.div
          animate={{ scale: [1, 1.08, 1] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="w-12 h-12 rounded-xl bg-violet-500 flex items-center justify-center text-white font-bold text-sm shadow-md"
        >
          P1
        </motion.div>
        <span className="text-xs text-violet-600 dark:text-violet-400 font-medium">Holds CPU</span>
      </div>
      <div className="flex flex-col items-center">
        <motion.div
          animate={{ x: [0, 8, 0] }}
          transition={{ repeat: Infinity, duration: 1.2 }}
        >
          <Ban className="w-8 h-8 text-red-400" />
        </motion.div>
        <span className="text-[10px] text-red-500 dark:text-red-400">Cannot force</span>
      </div>
      <div className="flex flex-col items-center gap-1">
        <motion.div
          animate={{ y: [0, -3, 0] }}
          transition={{ repeat: Infinity, duration: 0.8 }}
          className="w-12 h-12 rounded-xl bg-slate-400 flex items-center justify-center text-white font-bold text-sm shadow-md"
        >
          P2
        </motion.div>
        <span className="text-xs text-slate-500 dark:text-gray-400 font-medium">Wants CPU</span>
      </div>
    </div>
  );
}

function CircularWaitAnimation() {
  const nodes = [
    { id: "P1", x: 80, y: 30, color: "bg-blue-500" },
    { id: "P2", x: 200, y: 30, color: "bg-emerald-500" },
    { id: "P3", x: 140, y: 100, color: "bg-amber-500" },
  ];

  return (
    <div className="flex justify-center py-2">
      <svg width="280" height="130" viewBox="0 0 280 130">
        <defs>
          <marker id="circ-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444" />
          </marker>
        </defs>
        {/* Circular arrows */}
        <motion.path
          d="M 110 30 Q 160 10 195 30"
          fill="none"
          stroke="#ef4444"
          strokeWidth={2}
          markerEnd="url(#circ-arrow)"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        />
        <motion.path
          d="M 200 60 Q 210 90 170 100"
          fill="none"
          stroke="#ef4444"
          strokeWidth={2}
          markerEnd="url(#circ-arrow)"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ repeat: Infinity, duration: 1.5, delay: 0.5 }}
        />
        <motion.path
          d="M 110 100 Q 60 90 70 40"
          fill="none"
          stroke="#ef4444"
          strokeWidth={2}
          markerEnd="url(#circ-arrow)"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ repeat: Infinity, duration: 1.5, delay: 1 }}
        />
        {/* Nodes */}
        {nodes.map((n) => (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r={18} className={`${n.color} opacity-80`} />
            <text x={n.x} y={n.y + 1} textAnchor="middle" dominantBaseline="middle" className="text-xs font-bold fill-white">
              {n.id}
            </text>
          </g>
        ))}
        <text x="155" y="15" textAnchor="middle" className="text-[10px] fill-red-500 font-medium">waits for</text>
        <text x="220" y="85" textAnchor="middle" className="text-[10px] fill-red-500 font-medium">waits for</text>
        <text x="50" y="85" textAnchor="middle" className="text-[10px] fill-red-500 font-medium">waits for</text>
      </svg>
    </div>
  );
}

const ANIMATIONS: Record<string, React.FC> = {
  mutex: MutexAnimation,
  holdwait: HoldWaitAnimation,
  nopreempt: NoPreemptAnimation,
  circular: CircularWaitAnimation,
};

export default function DeadlockConditionAnalyzer() {
  const [activeConditions, setActiveConditions] = useState<Set<string>>(new Set());
  const [expandedCondition, setExpandedCondition] = useState<string | null>(null);

  const toggleCondition = (id: string) => {
    setActiveConditions((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const allSatisfied = activeConditions.size === 4;
  const count = activeConditions.size;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        Deadlock Condition Analyzer
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        The Four Coffman Conditions: All four must hold simultaneously for deadlock to occur
      </p>

      {/* Summary Bar */}
      <div className="mb-6 p-4 bg-white dark:bg-gray-800 rounded-xl border border-slate-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-semibold text-slate-700 dark:text-gray-300">
            Conditions Satisfied: {count} / 4
          </span>
          <AnimatePresence mode="wait">
            {allSatisfied ? (
              <motion.div
                key="deadlock"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-2 px-3 py-1 rounded-full bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
              >
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm font-bold">DEADLOCK POSSIBLE</span>
              </motion.div>
            ) : (
              <motion.div
                key="safe"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
              >
                <CheckCircle className="w-4 h-4" />
                <span className="text-sm font-bold">NO DEADLOCK</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        {/* Progress bar */}
        <div className="flex gap-1.5">
          {CONDITIONS.map((c) => (
            <motion.div
              key={c.id}
              animate={{ scaleX: activeConditions.has(c.id) ? 1 : 0.3 }}
              className={`flex-1 h-2 rounded-full transition-colors ${
                activeConditions.has(c.id)
                  ? `bg-gradient-to-r ${c.color}`
                  : "bg-slate-200 dark:bg-gray-700"
              }`}
              style={{ transformOrigin: "left" }}
            />
          ))}
        </div>
      </div>

      {/* Condition Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {CONDITIONS.map((condition) => {
          const isActive = activeConditions.has(condition.id);
          const isExpanded = expandedCondition === condition.id;
          const AnimComp = ANIMATIONS[condition.animationKey];

          return (
            <motion.div
              key={condition.id}
              layout
              className={`bg-white dark:bg-gray-800 rounded-xl border-2 transition-all ${
                isActive
                  ? `${condition.activeColor} shadow-lg`
                  : "border-slate-200 dark:border-gray-700"
              }`}
            >
              {/* Header */}
              <button
                onClick={() => setExpandedCondition(isExpanded ? null : condition.id)}
                className="w-full p-4 flex items-center gap-3 text-left"
              >
                <motion.div
                  animate={{
                    scale: isActive ? [1, 1.1, 1] : 1,
                    rotate: isActive ? [0, 5, -5, 0] : 0,
                  }}
                  transition={{ repeat: isActive ? Infinity : 0, duration: 2 }}
                  className={`w-10 h-10 rounded-xl bg-gradient-to-br ${condition.color} flex items-center justify-center text-white shadow-md flex-shrink-0`}
                >
                  {condition.icon}
                </motion.div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-bold text-slate-800 dark:text-gray-100">{condition.name}</h3>
                  <p className="text-xs text-slate-500 dark:text-gray-400 truncate">
                    {condition.definition.slice(0, 60)}...
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleCondition(condition.id);
                  }}
                  className={`flex-shrink-0 w-12 h-7 rounded-full transition-all relative ${
                    isActive
                      ? `bg-gradient-to-r ${condition.color}`
                      : "bg-slate-300 dark:bg-gray-600"
                  }`}
                >
                  <motion.div
                    animate={{ x: isActive ? 22 : 2 }}
                    className="absolute top-1 w-5 h-5 rounded-full bg-white shadow-sm"
                  />
                </button>
              </button>

              {/* Expanded Content */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 space-y-3">
                      {/* Animation */}
                      <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-2">
                        {AnimComp && <AnimComp />}
                      </div>

                      {/* Definition */}
                      <div>
                        <h4 className="text-xs font-semibold text-slate-600 dark:text-gray-400 mb-1 flex items-center gap-1">
                          <Zap className="w-3 h-3" /> Definition
                        </h4>
                        <p className="text-xs text-slate-700 dark:text-gray-300 leading-relaxed">
                          {condition.definition}
                        </p>
                      </div>

                      {/* Example */}
                      <div>
                        <h4 className="text-xs font-semibold text-slate-600 dark:text-gray-400 mb-1">Example</h4>
                        <p className="text-xs text-slate-600 dark:text-gray-400 leading-relaxed bg-amber-50 dark:bg-amber-900/20 p-2 rounded-lg border border-amber-200 dark:border-amber-800">
                          {condition.example}
                        </p>
                      </div>

                      {/* How to Break */}
                      <div>
                        <h4 className="text-xs font-semibold text-slate-600 dark:text-gray-400 mb-1">How to Break It</h4>
                        <p className="text-xs text-green-700 dark:text-green-300 leading-relaxed bg-green-50 dark:bg-green-900/20 p-2 rounded-lg border border-green-200 dark:border-green-800">
                          {condition.solution}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* Deadlock Explanation */}
      <AnimatePresence>
        {allSatisfied && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="mt-6 p-5 bg-red-50 dark:bg-red-900/20 rounded-xl border-2 border-red-300 dark:border-red-700"
          >
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-base font-bold text-red-700 dark:text-red-300 mb-1">Deadlock Analysis</h3>
                <p className="text-sm text-red-600 dark:text-red-400 leading-relaxed">
                  All four Coffman conditions are satisfied simultaneously. This means a deadlock <strong>can</strong> occur.
                  To prevent deadlock, break any one of the four conditions. The most common strategies are:
                </p>
                <ul className="text-sm text-red-600 dark:text-red-400 mt-2 space-y-1 list-disc list-inside">
                  <li><strong>Break Mutual Exclusion:</strong> Use sharable resources (not always possible)</li>
                  <li><strong>Break Hold and Wait:</strong> Request all resources atomically before execution</li>
                  <li><strong>Break No Preemption:</strong> Allow OS to forcibly reclaim resources</li>
                  <li><strong>Break Circular Wait:</strong> Impose a total order on resource acquisition</li>
                </ul>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Toggle All */}
      <div className="flex justify-center gap-3 mt-6">
        <button
          onClick={() => setActiveConditions(new Set(CONDITIONS.map((c) => c.id)))}
          className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-gray-700 text-slate-700 dark:text-gray-300 text-sm font-medium hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
        >
          Enable All
        </button>
        <button
          onClick={() => setActiveConditions(new Set())}
          className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-gray-700 text-slate-700 dark:text-gray-300 text-sm font-medium hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
        >
          Disable All
        </button>
      </div>
    </div>
  );
}
