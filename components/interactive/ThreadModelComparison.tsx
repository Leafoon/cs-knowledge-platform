"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Cpu, Zap, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

type Model = "ult" | "klt" | "mn";

const MODEL_DATA: Record<Model, {
  title: string;
  desc: string;
  icon: typeof Layers;
  color: string;
  bgColor: string;
  pros: string[];
  cons: string[];
}> = {
  ult: {
    title: "User-Level Threads (ULT)",
    desc: "Threads managed entirely by a thread library in user space. Kernel sees only one process.",
    icon: Layers,
    color: "blue",
    bgColor: "from-blue-50 to-blue-100",
    pros: ["Fast creation & switching", "Portable across OSes", "No kernel mode switch needed"],
    cons: ["One blocks → all block", "No true parallelism on multi-core", "Kernel can't schedule individually"],
  },
  klt: {
    title: "Kernel-Level Threads (KLT)",
    desc: "Each user thread maps 1:1 to a kernel thread. OS manages scheduling directly.",
    icon: Cpu,
    color: "emerald",
    bgColor: "from-emerald-50 to-emerald-100",
    pros: ["True parallelism", "One blocks, others continue", "Kernel can schedule each thread"],
    cons: ["Slower creation (syscall)", "Higher overhead per thread", "OS-dependent implementation"],
  },
  mn: {
    title: "M:N Hybrid Model",
    desc: "Multiple user threads mapped to fewer kernel threads. Combines benefits of both models.",
    icon: Zap,
    color: "purple",
    bgColor: "from-purple-50 to-purple-100",
    pros: ["Flexible scheduling", "Good parallelism", "Lower overhead than pure KLT"],
    cons: ["Complex implementation", "Hard to get right", "Scheduling conflicts possible"],
  },
};

function ThreadNode({ label, y, color, blocked, delay }: { label: string; y: number; color: string; blocked?: boolean; delay: number }) {
  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.5 }}
      transition={{ delay, type: "spring", stiffness: 200 }}
    >
      <rect x={0} y={y} width={90} height={30} rx={6} fill={blocked ? "#ef4444" : color} stroke={blocked ? "#b91c1c" : "transparent"} strokeWidth={2} />
      <text x={45} y={y + 19} textAnchor="middle" fill="white" fontSize={11} fontWeight={600}>
        {label}
      </text>
      {blocked && (
        <text x={45} y={y + 19} textAnchor="middle" fill="white" fontSize={9} fontWeight={700} opacity={0.8}>
          BLOCKED
        </text>
      )}
    </motion.g>
  );
}

function MappingLines({ mappings }: { mappings: Array<{ x1: number; y1: number; x2: number; y2: number; color: string }>; }) {
  return (
    <g>
      {mappings.map((m, i) => (
        <motion.line
          key={i}
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ delay: 0.3 + i * 0.1, duration: 0.4 }}
          x1={m.x1} y1={m.y1} x2={m.x2} y2={m.y2}
          stroke={m.color} strokeWidth={2} strokeDasharray="5,3"
        />
      ))}
    </g>
  );
}

function ULTModel({ blocking }: { blocking: boolean }) {
  return (
    <svg width={220} height={200} viewBox="0 0 220 200">
      {/* User Space Label */}
      <text x={110} y={15} textAnchor="middle" fontSize={11} fill="#64748b" fontWeight={600}>
        User Space
      </text>
      <line x1={10} y1={22} x2={210} y2={22} stroke="#cbd5e1" strokeWidth={1} />

      {/* ULTs */}
      <g transform="translate(20, 30)">
        <ThreadNode label="ULT 0" y={0} color="#3b82f6" blocked={blocking} delay={0} />
        <ThreadNode label="ULT 1" y={40} color="#3b82f6" blocked={blocking} delay={0.1} />
        <ThreadNode label="ULT 2" y={80} color="#3b82f6" blocked={blocking} delay={0.2} />
      </g>

      {/* Thread Library Box */}
      <rect x={20} y={130} width={180} height={20} rx={4} fill="#94a3b8" />
      <text x={110} y={144} textAnchor="middle" fontSize={10} fill="white" fontWeight={600}>Thread Library</text>

      {/* Kernel Space Label */}
      <text x={110} y={165} textAnchor="middle" fontSize={11} fill="#64748b" fontWeight={600}>
        Kernel Space
      </text>
      <line x1={10} y1={172} x2={210} y2={172} stroke="#cbd5e1" strokeWidth={1} />

      {/* Single KLT */}
      <g transform="translate(65, 178)">
        <ThreadNode label="KLT 0" y={0} color="#1e293b" blocked={false} delay={0.3} />
      </g>
    </svg>
  );
}

function KLTModel({ blocking }: { blocking: boolean }) {
  return (
    <svg width={220} height={200} viewBox="0 0 220 200">
      <text x={110} y={15} textAnchor="middle" fontSize={11} fill="#64748b" fontWeight={600}>
        User Space
      </text>
      <line x1={10} y1={22} x2={210} y2={22} stroke="#cbd5e1" strokeWidth={1} />

      {/* ULTs */}
      <g transform="translate(20, 30)">
        <ThreadNode label="ULT 0" y={0} color="#10b981" blocked={blocking} delay={0} />
        <ThreadNode label="ULT 1" y={40} color="#10b981" blocked={false} delay={0.1} />
        <ThreadNode label="ULT 2" y={80} color="#10b981" blocked={false} delay={0.2} />
      </g>

      {/* Mapping lines */}
      <MappingLines mappings={[
        { x1: 65, y1: 60, x2: 65, y2: 178, color: "#10b981" },
        { x1: 65, y1: 100, x2: 110, y2: 178, color: "#10b981" },
        { x1: 65, y1: 140, x2: 155, y2: 178, color: "#10b981" },
      ]} />

      <text x={110} y={160} textAnchor="middle" fontSize={11} fill="#64748b" fontWeight={600}>
        Kernel Space
      </text>
      <line x1={10} y1={167} x2={210} y2={167} stroke="#cbd5e1" strokeWidth={1} />

      {/* KLTs */}
      <g transform="translate(20, 173)">
        <ThreadNode label="KLT 0" y={0} color="#1e293b" blocked={blocking} delay={0.3} />
        <ThreadNode label="KLT 1" y={0} color="#1e293b" blocked={false} delay={0.4} />
        <ThreadNode label="KLT 2" y={0} color="#1e293b" blocked={false} delay={0.5} />
      </g>

      {/* Correctly position KLTs */}
      <g>
        <rect x={20} y={173} width={90} height={26} rx={6} fill="#1e293b" />
        <text x={65} y={190} textAnchor="middle" fill="white" fontSize={10} fontWeight={600}>KLT 0 {blocking ? "BLOCKED" : ""}</text>
        <rect x={115} y={173} width={90} height={26} rx={6} fill="#1e293b" />
        <text x={160} y={190} textAnchor="middle" fill="white" fontSize={10} fontWeight={600}>KLT 1</text>
      </g>
    </svg>
  );
}

function MNModel({ blocking }: { blocking: boolean }) {
  return (
    <svg width={220} height={200} viewBox="0 0 220 200">
      <text x={110} y={15} textAnchor="middle" fontSize={11} fill="#64748b" fontWeight={600}>
        User Space
      </text>
      <line x1={10} y1={22} x2={210} y2={22} stroke="#cbd5e1" strokeWidth={1} />

      {/* ULTs */}
      <g transform="translate(15, 30)">
        <rect x={0} y={0} width={60} height={24} rx={5} fill={blocking ? "#ef4444" : "#8b5cf6"} />
        <text x={30} y={16} textAnchor="middle" fill="white" fontSize={9} fontWeight={600}>ULT 0</text>
        <rect x={0} y={30} width={60} height={24} rx={5} fill={blocking ? "#ef4444" : "#8b5cf6"} />
        <text x={30} y={46} textAnchor="middle" fill="white" fontSize={9} fontWeight={600}>ULT 1</text>
        <rect x={70} y={0} width={60} height={24} rx={5} fill="#8b5cf6" />
        <text x={100} y={16} textAnchor="middle" fill="white" fontSize={9} fontWeight={600}>ULT 2</text>
        <rect x={70} y={30} width={60} height={24} rx={5} fill="#8b5cf6" />
        <text x={100} y={46} textAnchor="middle" fill="white" fontSize={9} fontWeight={600}>ULT 3</text>
        <rect x={140} y={0} width={60} height={24} rx={5} fill="#8b5cf6" />
        <text x={170} y={16} textAnchor="middle" fill="white" fontSize={9} fontWeight={600}>ULT 4</text>
      </g>

      {/* Mapping lines */}
      <g>
        {[30, 75].map((x, i) => (
          <motion.line key={`m0-${i}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
            x1={x} y1={80} x2={55} y2={140} stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4,3" />
        ))}
        {[100, 170].map((x, i) => (
          <motion.line key={`m1-${i}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
            x1={x} y1={50} x2={165} y2={140} stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4,3" />
        ))}
      </g>

      <text x={110} y={120} textAnchor="middle" fontSize={10} fill="#94a3b8" fontWeight={500}>
        Thread Scheduler
      </text>

      <text x={110} y={138} textAnchor="middle" fontSize={11} fill="#64748b" fontWeight={600}>
        Kernel Space
      </text>
      <line x1={10} y1={145} x2={210} y2={145} stroke="#cbd5e1" strokeWidth={1} />

      {/* KLTs */}
      <rect x={20} y={152} width={70} height={26} rx={6} fill="#1e293b" />
      <text x={55} y={169} textAnchor="middle" fill="white" fontSize={10} fontWeight={600}>
        KLT 0 {blocking ? "BLOCKED" : ""}
      </text>
      <rect x={130} y={152} width={70} height={26} rx={6} fill="#1e293b" />
      <text x={165} y={169} textAnchor="middle" fill="white" fontSize={10} fontWeight={600}>KLT 1</text>
    </svg>
  );
}

export default function ThreadModelComparison() {
  const [activeModel, setActiveModel] = useState<Model>("ult");
  const [blocking, setBlocking] = useState(false);
  const [autoBlock, setAutoBlock] = useState(false);

  useEffect(() => {
    if (!autoBlock) return;
    const timer = setInterval(() => {
      setBlocking((prev) => !prev);
    }, 2000);
    return () => clearInterval(timer);
  }, [autoBlock]);

  const triggerBlock = useCallback(() => {
    setBlocking(true);
    setTimeout(() => setBlocking(false), 2500);
  }, []);

  const current = MODEL_DATA[activeModel];
  const Icon = current.icon;

  return (
    <div className={`w-full max-w-6xl mx-auto p-6 bg-gradient-to-br ${current.bgColor} rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800 transition-colors duration-500`}>
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center flex items-center justify-center gap-2">
        <Icon className="w-7 h-7" />
        {current.title}
      </h2>
      <p className="text-center text-slate-600 dark:text-gray-400 text-sm mb-6">{current.desc}</p>

      {/* Model Selector */}
      <div className="flex justify-center gap-3 mb-6 flex-wrap">
        {(["ult", "klt", "mn"] as const).map((m) => {
          const data = MODEL_DATA[m];
          const MIcon = data.icon;
          return (
            <button
              key={m}
              onClick={() => { setActiveModel(m); setBlocking(false); }}
              className={`px-5 py-2.5 rounded-lg font-medium transition-all flex items-center gap-2 ${
                activeModel === m
                  ? `bg-${data.color}-600 text-white shadow-lg scale-105`
                  : "bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 dark:bg-gray-700 dark:text-gray-200"
              }`}
            >
              <MIcon className="w-4 h-4" />
              {m === "ult" ? "User-Level" : m === "klt" ? "Kernel-Level" : "M:N Hybrid"}
            </button>
          );
        })}
      </div>

      {/* Visualization */}
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        <div className="flex-1 flex flex-col items-center">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeModel}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-md border border-slate-200 dark:border-gray-700"
            >
              {activeModel === "ult" && <ULTModel blocking={blocking} />}
              {activeModel === "klt" && <KLTModel blocking={blocking} />}
              {activeModel === "mn" && <MNModel blocking={blocking} />}
            </motion.div>
          </AnimatePresence>

          {/* Blocking Controls */}
          <div className="flex gap-3 mt-4">
            <button
              onClick={triggerBlock}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 flex items-center gap-2"
            >
              <AlertTriangle className="w-4 h-4" />
              Simulate Block
            </button>
            <button
              onClick={() => setAutoBlock(!autoBlock)}
              className={`px-4 py-2 rounded-lg font-medium ${
                autoBlock ? "bg-amber-500 text-white" : "bg-white text-slate-700 border border-slate-300 dark:bg-gray-700 dark:text-gray-200"
              }`}
            >
              {autoBlock ? "Stop Auto" : "Auto Block"}
            </button>
          </div>

          {/* Blocking Explanation */}
          <AnimatePresence>
            {blocking && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-lg text-sm text-red-800 dark:text-red-300 max-w-md"
              >
                {activeModel === "ult" && (
                  <span><strong>ULT blocks all!</strong> Since kernel sees one thread, when any ULT blocks (e.g., I/O), the entire process blocks — all ULTs stop.</span>
                )}
                {activeModel === "klt" && (
                  <span><strong>Only one KLT blocks.</strong> The blocked KLT yields the CPU. Other KLTs (and their ULTs) continue executing normally.</span>
                )}
                {activeModel === "mn" && (
                  <span><strong>Partial impact.</strong> The blocked ULT stops, but other ULTs on the same KLT can be rescheduled by the user-space scheduler to another KLT.</span>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Pros/Cons */}
        <div className="flex-1 min-w-[280px]">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-md border border-slate-200 dark:border-gray-700">
            <h4 className="font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-500" />
              Advantages
            </h4>
            <ul className="space-y-2 mb-5">
              {current.pros.map((pro, i) => (
                <motion.li
                  key={pro}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-start gap-2 text-sm text-slate-600 dark:text-gray-300"
                >
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                  {pro}
                </motion.li>
              ))}
            </ul>

            <h4 className="font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
              <XCircle className="w-5 h-5 text-red-500" />
              Disadvantages
            </h4>
            <ul className="space-y-2">
              {current.cons.map((con, i) => (
                <motion.li
                  key={con}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + i * 0.1 }}
                  className="flex items-start gap-2 text-sm text-slate-600 dark:text-gray-300"
                >
                  <XCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
                  {con}
                </motion.li>
              ))}
            </ul>
          </div>

          {/* Summary Table */}
          <div className="mt-4 bg-white dark:bg-gray-800 rounded-xl p-4 shadow-md border border-slate-200 dark:border-gray-700">
            <h4 className="font-semibold text-slate-700 dark:text-gray-200 mb-3">Quick Comparison</h4>
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-100 dark:bg-gray-700">
                  <th className="px-2 py-1.5 text-left text-slate-600 dark:text-gray-300">Feature</th>
                  <th className="px-2 py-1.5 text-center text-blue-600">ULT</th>
                  <th className="px-2 py-1.5 text-center text-emerald-600">KLT</th>
                  <th className="px-2 py-1.5 text-center text-purple-600">M:N</th>
                </tr>
              </thead>
              <tbody className="text-slate-600 dark:text-gray-300">
                {[
                  ["Parallelism", "No", "Yes", "Yes"],
                  ["Blocking", "All block", "One blocks", "Partial"],
                  ["Creation cost", "Low", "High", "Medium"],
                  ["Scheduling", "User-space", "Kernel", "Both"],
                  ["Complexity", "Simple", "Simple", "Complex"],
                ].map(([feat, ult, klt, mn], i) => (
                  <tr key={feat} className="border-t border-slate-100 dark:border-gray-700">
                    <td className="px-2 py-1 font-medium">{feat}</td>
                    <td className="px-2 py-1 text-center">{ult}</td>
                    <td className="px-2 py-1 text-center">{klt}</td>
                    <td className="px-2 py-1 text-center">{mn}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
