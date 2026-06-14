"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, ChevronRight, CheckCircle, AlertTriangle, Info, BookOpen } from "lucide-react";

interface Preset {
  name: string;
  n: number;
  m: number;
  available: number[];
  max: number[][];
  allocation: number[][];
}

const PRESETS: Preset[] = [
  {
    name: "Classic Safe",
    n: 5,
    m: 3,
    available: [3, 3, 2],
    max: [
      [7, 5, 3],
      [3, 2, 2],
      [9, 0, 2],
      [2, 2, 2],
      [4, 3, 3],
    ],
    allocation: [
      [0, 1, 0],
      [2, 0, 0],
      [3, 0, 2],
      [2, 1, 1],
      [0, 0, 2],
    ],
  },
  {
    name: "Unsafe (Deadlock Risk)",
    n: 3,
    m: 2,
    available: [0, 0],
    max: [
      [2, 2],
      [1, 1],
      [2, 2],
    ],
    allocation: [
      [1, 1],
      [1, 0],
      [0, 1],
    ],
  },
  {
    name: "2 Resources, 4 Processes",
    n: 4,
    m: 2,
    available: [1, 1],
    max: [
      [3, 3],
      [2, 2],
      [2, 1],
      [1, 1],
    ],
    allocation: [
      [1, 1],
      [1, 0],
      [1, 0],
      [0, 0],
    ],
  },
];

export default function BankerAlgorithmSimulator() {
  const [n, setN] = useState(5);
  const [m, setM] = useState(3);
  const [available, setAvailable] = useState<number[]>([3, 3, 2]);
  const [max, setMax] = useState<number[][]>(PRESETS[0].max);
  const [allocation, setAllocation] = useState<number[][]>(PRESETS[0].allocation);
  const [safeSequence, setSafeSequence] = useState<number[]>([]);
  const [isSafe, setIsSafe] = useState<boolean | null>(null);
  const [steps, setSteps] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);
  const [requestPid, setRequestPid] = useState(0);
  const [requestVec, setRequestVec] = useState<number[]>([0, 0, 0]);
  const [requestResult, setRequestResult] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const need = allocation.map((alloc, i) => alloc.map((a, j) => max[i][j] - a));

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [msg, ...prev].slice(0, 12));
  }, []);

  const loadPreset = useCallback((preset: Preset) => {
    setN(preset.n);
    setM(preset.m);
    setAvailable([...preset.available]);
    setMax(preset.max.map((r) => [...r]));
    setAllocation(preset.allocation.map((r) => [...r]));
    setSafeSequence([]);
    setIsSafe(null);
    setSteps([]);
    setCurrentStep(-1);
    setRequestResult(null);
    setRequestVec(new Array(preset.m).fill(0));
    setLog([]);
    addLog(`Loaded preset: ${preset.name}`);
  }, [addLog]);

  const runSafetyCheck = useCallback(() => {
    const work = [...available];
    const finish = new Array(n).fill(false);
    const seq: number[] = [];
    const stepLog: string[] = [];

    stepLog.push(`Initial: Work = [${work.join(", ")}], Finish = [${finish.map((f) => f ? "T" : "F").join(", ")}]`);

    let found = true;
    let iter = 0;
    while (found && iter < n) {
      found = false;
      for (let i = 0; i < n; i++) {
        if (finish[i]) continue;
        const needI = need[i];
        const canRun = needI.every((nj, j) => nj <= work[j]);
        if (canRun) {
          stepLog.push(`P${i}: Need=[${needI.join(",")}] <= Work=[${work.join(",")}] -- PASS. Allocate & release.`);
          for (let j = 0; j < m; j++) work[j] += allocation[i][j];
          finish[i] = true;
          seq.push(i);
          stepLog.push(`  Work updated: [${work.join(", ")}], Finish[${i}] = T`);
          found = true;
          break;
        }
      }
      if (!found && finish.some((f) => !f)) {
        const unf = finish.map((f, i) => !f ? `P${i}` : "").filter(Boolean);
        stepLog.push(`No process in {${unf.join(",")}} can have Need <= Work. UNSAFE!`);
      }
      iter++;
    }

    const allFinish = finish.every((f) => f);
    if (allFinish) {
      stepLog.push(`All processes finished. Safe sequence: <${seq.map((i) => `P${i}`).join(", ")}>`);
    }

    setSteps(stepLog);
    setSafeSequence(seq);
    setIsSafe(allFinish);
    setCurrentStep(0);
    setIsRunning(true);
    addLog(allFinish ? `Safe! Sequence: ${seq.map((i) => `P${i}`).join(" -> ")}` : "Unsafe state detected!");
  }, [available, n, m, need, allocation, addLog]);

  const stepForward = useCallback(() => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((s) => s + 1);
    } else {
      setIsRunning(false);
    }
  }, [currentStep, steps]);

  const runAllSteps = useCallback(() => {
    let idx = 0;
    const interval = setInterval(() => {
      idx++;
      setCurrentStep(idx);
      if (idx >= steps.length - 1) {
        clearInterval(interval);
        setIsRunning(false);
      }
    }, 600);
  }, [steps]);

  const handleRequest = useCallback(() => {
    const rv = requestVec;
    // Step 1: Check Request <= Need
    const needI = need[requestPid];
    if (!rv.every((r, j) => r <= needI[j])) {
      setRequestResult("ERROR: Request exceeds maximum claim (Need).");
      addLog(`P${requestPid} request denied: exceeds Need`);
      return;
    }
    // Step 2: Check Request <= Available
    if (!rv.every((r, j) => r <= available[j])) {
      setRequestResult("DENIED: Resources not available. P must wait.");
      addLog(`P${requestPid} request denied: insufficient resources`);
      return;
    }
    // Step 3: Pretend to allocate
    const newAvail = available.map((a, j) => a - rv[j]);
    const newAlloc = allocation.map((r, i) => r.map((a, j) => a + (i === requestPid ? rv[j] : 0)));
    const newNeed = newAlloc.map((alloc, i) => max[i].map((mx, j) => mx - alloc[j]));

    // Step 4: Run safety check on pretend state
    const work = [...newAvail];
    const finish = new Array(n).fill(false);
    const seq: number[] = [];
    let found = true;
    let iter = 0;
    while (found && iter < n) {
      found = false;
      for (let i = 0; i < n; i++) {
        if (finish[i]) continue;
        const canRun = newNeed[i].every((nj, j) => nj <= work[j]);
        if (canRun) {
          for (let j = 0; j < m; j++) work[j] += newAlloc[i][j];
          finish[i] = true;
          seq.push(i);
          found = true;
          break;
        }
      }
      iter++;
    }
    const allFinish = finish.every((f) => f);
    if (allFinish) {
      setRequestResult(`GRANTED! Safe sequence: <${seq.map((i) => `P${i}`).join(", ")}>`);
      addLog(`P${requestPid} request [${rv.join(",")}] granted (safe)`);
    } else {
      setRequestResult("DENIED: Would lead to unsafe state.");
      addLog(`P${requestPid} request [${rv.join(",")}] denied (unsafe)`);
    }
  }, [requestVec, requestPid, need, available, allocation, max, n, m, addLog]);

  const updateMatrix = (matrix: number[][], row: number, col: number, val: number) => {
    const copy = matrix.map((r) => [...r]);
    copy[row][col] = val;
    return copy;
  };

  const resourceLabels = Array.from({ length: m }, (_, i) => String.fromCharCode(65 + i));

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        Banker&apos;s Algorithm Simulator
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        Interactive simulation of the Banker&apos;s Deadlock Avoidance Algorithm
      </p>

      {/* Presets */}
      <div className="flex justify-center gap-3 mb-6 flex-wrap">
        {PRESETS.map((p, i) => (
          <button
            key={i}
            onClick={() => loadPreset(p)}
            className="px-4 py-2 rounded-lg bg-white dark:bg-gray-700 text-sm font-medium text-slate-700 dark:text-gray-300 border border-slate-200 dark:border-gray-600 hover:bg-cyan-50 dark:hover:bg-gray-600 transition-colors"
          >
            <BookOpen className="w-3.5 h-3.5 inline-block mr-1.5" />
            {p.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Matrices */}
        <div className="xl:col-span-2 space-y-4">
          {/* Resource Count Controls */}
          <div className="flex items-center gap-4 bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-slate-700 dark:text-gray-300">Processes:</span>
              <div className="flex gap-1">
                {[3, 4, 5].map((v) => (
                  <button
                    key={v}
                    onClick={() => {
                      setN(v);
                      setAllocation(Array.from({ length: v }, (_, i) => allocation[i] || new Array(m).fill(0)));
                      setMax(Array.from({ length: v }, (_, i) => max[i] || new Array(m).fill(0)));
                      setSafeSequence([]);
                      setIsSafe(null);
                      setSteps([]);
                      setCurrentStep(-1);
                    }}
                    className={`w-8 h-8 rounded text-sm font-bold ${
                      n === v ? "bg-cyan-500 text-white" : "bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-300"
                    }`}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-slate-700 dark:text-gray-300">Resources:</span>
              <div className="flex gap-1">
                {[2, 3, 4].map((v) => (
                  <button
                    key={v}
                    onClick={() => {
                      setM(v);
                      setAvailable(new Array(v).fill(0));
                      setAllocation(allocation.map((r) => r.length < v ? [...r, ...new Array(v - r.length).fill(0)] : r.slice(0, v)));
                      setMax(max.map((r) => r.length < v ? [...r, ...new Array(v - r.length).fill(0)] : r.slice(0, v)));
                      setRequestVec(new Array(v).fill(0));
                      setSafeSequence([]);
                      setIsSafe(null);
                      setSteps([]);
                      setCurrentStep(-1);
                    }}
                    className={`w-8 h-8 rounded text-sm font-bold ${
                      m === v ? "bg-cyan-500 text-white" : "bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-300"
                    }`}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Available Vector */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-2">Available</h3>
            <div className="flex gap-2">
              {available.map((v, j) => (
                <div key={j} className="flex flex-col items-center gap-1">
                  <span className="text-xs text-slate-500 dark:text-gray-400 font-bold">{resourceLabels[j]}</span>
                  <input
                    type="number"
                    min={0}
                    max={20}
                    value={v}
                    onChange={(e) => {
                      const copy = [...available];
                      copy[j] = parseInt(e.target.value) || 0;
                      setAvailable(copy);
                    }}
                    className="w-14 h-9 rounded-lg border border-slate-300 dark:border-gray-600 bg-slate-50 dark:bg-gray-900 text-center text-sm font-medium text-slate-800 dark:text-gray-200 focus:ring-2 focus:ring-cyan-400 outline-none"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Max and Allocation tables side by side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Max */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
              <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-2">Max</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th className="px-2 py-1 text-xs text-slate-500 dark:text-gray-400"></th>
                      {resourceLabels.map((r) => (
                        <th key={r} className="px-2 py-1 text-xs font-bold text-slate-600 dark:text-gray-300">{r}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: n }, (_, i) => (
                      <tr key={i}>
                        <td className="px-2 py-1 text-xs font-bold text-blue-600 dark:text-blue-400">P{i}</td>
                        {Array.from({ length: m }, (_, j) => (
                          <td key={j} className="px-1 py-1">
                            <input
                              type="number"
                              min={0}
                              max={20}
                              value={max[i]?.[j] ?? 0}
                              onChange={(e) => setMax(updateMatrix(max, i, j, parseInt(e.target.value) || 0))}
                              className="w-12 h-7 rounded border border-slate-300 dark:border-gray-600 bg-slate-50 dark:bg-gray-900 text-center text-xs font-medium text-slate-800 dark:text-gray-200 focus:ring-2 focus:ring-cyan-400 outline-none"
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Allocation */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
              <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-2">Allocation</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th className="px-2 py-1 text-xs text-slate-500 dark:text-gray-400"></th>
                      {resourceLabels.map((r) => (
                        <th key={r} className="px-2 py-1 text-xs font-bold text-slate-600 dark:text-gray-300">{r}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: n }, (_, i) => (
                      <tr key={i}>
                        <td className="px-2 py-1 text-xs font-bold text-blue-600 dark:text-blue-400">P{i}</td>
                        {Array.from({ length: m }, (_, j) => (
                          <td key={j} className="px-1 py-1">
                            <input
                              type="number"
                              min={0}
                              max={max[i]?.[j] ?? 0}
                              value={allocation[i]?.[j] ?? 0}
                              onChange={(e) => setAllocation(updateMatrix(allocation, i, j, parseInt(e.target.value) || 0))}
                              className="w-12 h-7 rounded border border-slate-300 dark:border-gray-600 bg-slate-50 dark:bg-gray-900 text-center text-xs font-medium text-slate-800 dark:text-gray-200 focus:ring-2 focus:ring-cyan-400 outline-none"
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Need (computed) */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-2">
              Need <span className="text-xs font-normal text-slate-400 dark:text-gray-500">(Max - Allocation)</span>
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    <th className="px-2 py-1 text-xs text-slate-500 dark:text-gray-400"></th>
                    {resourceLabels.map((r) => (
                      <th key={r} className="px-2 py-1 text-xs font-bold text-slate-600 dark:text-gray-300">{r}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {need.map((row, i) => (
                    <tr key={i}>
                      <td className="px-2 py-1 text-xs font-bold text-blue-600 dark:text-blue-400">P{i}</td>
                      {row.map((v, j) => (
                        <td key={j} className="px-2 py-1 text-center">
                          <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                            v < 0 ? "bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400" : "bg-cyan-50 dark:bg-cyan-900/20 text-cyan-700 dark:text-cyan-300"
                          }`}>
                            {v}
                          </span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={runSafetyCheck}
              className="flex-1 px-4 py-3 rounded-xl bg-cyan-500 text-white font-semibold hover:bg-cyan-600 transition-colors flex items-center justify-center gap-2 shadow-md"
            >
              <Play className="w-5 h-5" /> Run Safety Check
            </button>
            <button
              onClick={() => {
                setSafeSequence([]);
                setIsSafe(null);
                setSteps([]);
                setCurrentStep(-1);
                setRequestResult(null);
                setLog([]);
              }}
              className="px-4 py-3 rounded-xl bg-slate-400 text-white font-semibold hover:bg-slate-500 transition-colors"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>

          {/* Resource Request */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-3">Resource Request</h3>
            <div className="flex items-end gap-3 flex-wrap">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-500 dark:text-gray-400">Process</label>
                <select
                  value={requestPid}
                  onChange={(e) => setRequestPid(parseInt(e.target.value))}
                  className="h-9 px-3 rounded-lg border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-slate-800 dark:text-gray-200"
                >
                  {Array.from({ length: n }, (_, i) => (
                    <option key={i} value={i}>P{i}</option>
                  ))}
                </select>
              </div>
              {Array.from({ length: m }, (_, j) => (
                <div key={j} className="flex flex-col gap-1">
                  <label className="text-xs text-slate-500 dark:text-gray-400 font-bold">{resourceLabels[j]}</label>
                  <input
                    type="number"
                    min={0}
                    max={need[requestPid]?.[j] ?? 0}
                    value={requestVec[j] ?? 0}
                    onChange={(e) => {
                      const copy = [...requestVec];
                      copy[j] = parseInt(e.target.value) || 0;
                      setRequestVec(copy);
                    }}
                    className="w-14 h-9 rounded-lg border border-slate-300 dark:border-gray-600 bg-slate-50 dark:bg-gray-900 text-center text-sm font-medium text-slate-800 dark:text-gray-200 focus:ring-2 focus:ring-cyan-400 outline-none"
                  />
                </div>
              ))}
              <button
                onClick={handleRequest}
                className="h-9 px-4 rounded-lg bg-emerald-500 text-white text-sm font-medium hover:bg-emerald-600 transition-colors"
              >
                Request
              </button>
            </div>
            <AnimatePresence>
              {requestResult && (
                <motion.div
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className={`mt-3 p-3 rounded-lg text-sm font-medium ${
                    requestResult.startsWith("GRANTED")
                      ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 border border-green-300 dark:border-green-700"
                      : "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-700"
                  }`}
                >
                  {requestResult}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Right Panel: Steps & Log */}
        <div className="space-y-4">
          {/* Safety Check Steps */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300">Safety Algorithm Steps</h3>
              {steps.length > 0 && currentStep < steps.length - 1 && (
                <button
                  onClick={stepForward}
                  className="px-2 py-1 rounded bg-cyan-500 text-white text-xs font-medium hover:bg-cyan-600 flex items-center gap-1"
                >
                  Next <ChevronRight className="w-3 h-3" />
                </button>
              )}
            </div>
            <div className="space-y-1.5 max-h-64 overflow-y-auto">
              {steps.length === 0 && (
                <p className="text-xs text-slate-400 dark:text-gray-500 italic">Run safety check to see steps...</p>
              )}
              {steps.map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: i <= currentStep ? 1 : 0.3 }}
                  className={`text-xs p-2 rounded font-mono ${
                    i === currentStep
                      ? "bg-cyan-50 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-300 border border-cyan-300 dark:border-cyan-700"
                      : i < currentStep
                      ? "text-slate-500 dark:text-gray-400"
                      : "text-slate-300 dark:text-gray-600"
                  }`}
                >
                  {step}
                </motion.div>
              ))}
            </div>

            {/* Result */}
            <AnimatePresence>
              {isSafe !== null && currentStep >= steps.length - 1 && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className={`mt-3 p-3 rounded-xl border-2 ${
                    isSafe
                      ? "bg-green-50 dark:bg-green-900/20 border-green-400 dark:border-green-600"
                      : "bg-red-50 dark:bg-red-900/20 border-red-400 dark:border-red-600"
                  }`}
                >
                  {isSafe ? (
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                      <div>
                        <p className="text-sm font-bold text-green-700 dark:text-green-300">SAFE STATE</p>
                        <p className="text-xs text-green-600 dark:text-green-400">
                          Sequence: &lt;{safeSequence.map((i) => `P${i}`).join(", ")}&gt;
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
                      <div>
                        <p className="text-sm font-bold text-red-700 dark:text-red-300">UNSAFE STATE</p>
                        <p className="text-xs text-red-600 dark:text-red-400">No safe sequence exists</p>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Activity Log */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-3 flex items-center gap-2">
              <Info className="w-4 h-4" /> Activity Log
            </h3>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {log.length === 0 && (
                <p className="text-xs text-slate-400 dark:text-gray-500 italic">No activity yet...</p>
              )}
              {log.map((entry, i) => (
                <div key={i} className={`text-xs p-1.5 rounded ${i === 0 ? "bg-cyan-50 dark:bg-cyan-900/20 text-cyan-700 dark:text-cyan-300" : "text-slate-500 dark:text-gray-400"}`}>
                  {entry}
                </div>
              ))}
            </div>
          </div>

          {/* Info */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
            <h3 className="text-xs font-semibold text-blue-700 dark:text-blue-300 mb-2">How it works</h3>
            <ol className="text-xs text-blue-600 dark:text-blue-400 space-y-1 list-decimal list-inside">
              <li>Find a process whose Need &le; Work</li>
              <li>Assume it runs and releases resources</li>
              <li>Add its Allocation back to Work</li>
              <li>Repeat until all finish (safe) or none can proceed (unsafe)</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
