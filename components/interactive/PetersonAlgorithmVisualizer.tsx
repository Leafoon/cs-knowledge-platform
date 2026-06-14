"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, StepForward, Shield, Lock, Unlock } from "lucide-react";

const P0_LINES = [
  "flag[0] = true;",
  "turn = 1;",
  "while (flag[1] && turn == 1)",
  "  ; // busy wait",
  "// critical section",
  "flag[0] = false;",
];

const P1_LINES = [
  "flag[1] = true;",
  "turn = 0;",
  "while (flag[0] && turn == 0)",
  "  ; // busy wait",
  "// critical section",
  "flag[1] = false;",
];

interface StepState {
  p0Line: number;
  p1Line: number;
  flag: [boolean, boolean];
  turn: number | null;
  p0InCS: boolean;
  p1InCS: boolean;
  description: string;
  executing: number; // which process is executing this step
}

function buildSteps(): StepState[] {
  const steps: StepState[] = [];
  let flag: [boolean, boolean] = [false, false];
  let turn: number | null = null;
  let p0InCS = false;
  let p1InCS = false;

  // P0 enters
  steps.push({ p0Line: 0, p1Line: -1, flag: [false, false], turn: null, p0InCS: false, p1InCS: false, description: "P0: Set flag[0] = true", executing: 0 });
  flag = [true, false];
  steps.push({ p0Line: 0, p1Line: -1, flag: [...flag] as [boolean, boolean], turn: null, p0InCS: false, p1InCS: false, description: "P0: flag[0] is now true", executing: 0 });

  steps.push({ p0Line: 1, p1Line: -1, flag: [...flag] as [boolean, boolean], turn: null, p0InCS: false, p1InCS: false, description: "P0: Set turn = 1 (yield to P1)", executing: 0 });
  turn = 1;

  // P1 enters while P0 is checking
  steps.push({ p0Line: 2, p1Line: 0, flag: [...flag] as [boolean, boolean], turn: 1, p0InCS: false, p1InCS: false, description: "P0 checks while: flag[1]=false -> exit loop. P1: Set flag[1] = true", executing: 0 });
  flag = [true, true];

  steps.push({ p0Line: 2, p1Line: 1, flag: [...flag] as [boolean, boolean], turn: 1, p0InCS: false, p1InCS: false, description: "P1: Set turn = 0 (yield to P0)", executing: 1 });
  turn = 0;

  // P0 enters critical section
  steps.push({ p0Line: 4, p1Line: 2, flag: [...flag] as [boolean, boolean], turn: 0, p0InCS: false, p1InCS: false, description: "P0 enters critical section. P1 checks: flag[0]=true && turn==0 -> busy wait", executing: 0 });
  p0InCS = true;
  steps.push({ p0Line: 4, p1Line: 3, flag: [...flag] as [boolean, boolean], turn: 0, p0InCS: true, p1InCS: false, description: "P0 is in critical section. P1 is busy waiting...", executing: 0 });

  // P1 continues waiting
  steps.push({ p0Line: 4, p1Line: 2, flag: [...flag] as [boolean, boolean], turn: 0, p0InCS: true, p1InCS: false, description: "P1 re-checks: flag[0]=true && turn==0 -> still waiting", executing: 1 });

  // P0 exits
  steps.push({ p0Line: 5, p1Line: 2, flag: [...flag] as [boolean, boolean], turn: 0, p0InCS: true, p1InCS: false, description: "P0: Set flag[0] = false (exiting)", executing: 0 });
  flag = [false, true];
  p0InCS = false;

  // P1 enters critical section
  steps.push({ p0Line: -1, p1Line: 2, flag: [...flag] as [boolean, boolean], turn: 0, p0InCS: false, p1InCS: false, description: "P1 checks: flag[0]=false -> exit loop, enter critical section", executing: 1 });
  p1InCS = true;
  steps.push({ p0Line: -1, p1Line: 4, flag: [...flag] as [boolean, boolean], turn: 0, p0InCS: false, p1InCS: true, description: "P1 is in critical section!", executing: 1 });

  // P1 exits
  steps.push({ p0Line: -1, p1Line: 5, flag: [...flag] as [boolean, boolean], turn: 0, p0InCS: false, p1InCS: true, description: "P1: Set flag[1] = false (exiting)", executing: 1 });
  flag = [false, false];
  p1InCS = false;

  steps.push({ p0Line: -1, p1Line: -1, flag: [false, false], turn: 0, p0InCS: false, p1InCS: false, description: "Both processes have exited. Mutual exclusion maintained!", executing: -1 });

  return steps;
}

const ALL_STEPS = buildSteps();

export default function PetersonAlgorithmVisualizer() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1200);

  const step = ALL_STEPS[Math.min(currentStep, ALL_STEPS.length - 1)];

  const goNext = useCallback(() => {
    setCurrentStep((s) => Math.min(s + 1, ALL_STEPS.length - 1));
  }, []);

  const goPrev = useCallback(() => {
    setCurrentStep((s) => Math.max(s - 1, 0));
  }, []);

  const reset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  // Auto-play
  useState(() => {
    let timer: NodeJS.Timeout;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((s) => {
          if (s >= ALL_STEPS.length - 1) {
            setIsPlaying(false);
            return s;
          }
          return s + 1;
        });
      }, speed);
    }
    return () => clearInterval(timer);
  });

  const renderCodePanel = (lines: string[], processId: number, highlightLine: number) => {
    const isExecutingProcess = step.executing === processId;
    const inCS = processId === 0 ? step.p0InCS : step.p1InCS;

    return (
      <div className={`rounded-lg border-2 p-4 transition-all duration-300 ${
        isExecutingProcess
          ? "border-blue-400 bg-blue-50 dark:border-blue-500 dark:bg-blue-900/30"
          : "border-slate-200 bg-white dark:border-gray-600 dark:bg-gray-800"
      }`}>
        <div className="flex items-center gap-2 mb-3">
          <div className={`w-3 h-3 rounded-full ${processId === 0 ? "bg-blue-500" : "bg-purple-500"}`} />
          <span className="font-bold text-slate-700 dark:text-gray-200">Process {processId}</span>
          {isExecutingProcess && (
            <motion.span
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-800 dark:text-blue-200 rounded-full"
            >
              Executing
            </motion.span>
          )}
          {inCS && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="flex items-center gap-1 text-xs px-2 py-0.5 bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200 rounded-full"
            >
              <Lock className="w-3 h-3" /> In CS
            </motion.div>
          )}
        </div>
        <pre className="text-sm font-mono leading-relaxed">
          {lines.map((line, i) => {
            const isHighlight = i === highlightLine;
            return (
              <div
                key={i}
                className={`px-2 py-1 rounded transition-all duration-200 ${
                  isHighlight
                    ? isExecutingProcess
                      ? "bg-yellow-200 dark:bg-yellow-700/50 font-semibold"
                      : "bg-slate-100 dark:bg-gray-700"
                    : "hover:bg-slate-50 dark:hover:bg-gray-800"
                }`}
              >
                <span className="text-slate-400 dark:text-gray-500 select-none mr-2 text-xs">{i + 1}</span>
                <span className="text-slate-800 dark:text-gray-200">{line}</span>
                {isHighlight && isExecutingProcess && (
                  <motion.span
                    animate={{ opacity: [1, 0] }}
                    transition={{ repeat: Infinity, duration: 0.8 }}
                    className="ml-1 text-blue-500"
                  >
                    {"<--"}
                  </motion.span>
                )}
              </div>
            );
          })}
        </pre>
      </div>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Peterson&apos;s Algorithm Visualizer
      </h2>

      {/* Shared State Display */}
      <div className="flex justify-center gap-8 mb-6">
        <div className="flex items-center gap-3 bg-white dark:bg-gray-800 px-5 py-3 rounded-lg shadow-sm border border-slate-200 dark:border-gray-700">
          <span className="text-sm font-medium text-slate-600 dark:text-gray-300">flag[]:</span>
          <div className="flex gap-2">
            {[0, 1].map((i) => (
              <div key={i} className="flex items-center gap-1">
                <span className="text-xs text-slate-500 dark:text-gray-400">[{i}]</span>
                <motion.div
                  animate={{
                    backgroundColor: step.flag[i] ? "#3b82f6" : "#e2e8f0",
                    scale: step.flag[i] ? 1.1 : 1,
                  }}
                  className="w-8 h-8 rounded-md flex items-center justify-center text-xs font-bold"
                >
                  <span className={step.flag[i] ? "text-white" : "text-slate-400"}>
                    {step.flag[i] ? "T" : "F"}
                  </span>
                </motion.div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3 bg-white dark:bg-gray-800 px-5 py-3 rounded-lg shadow-sm border border-slate-200 dark:border-gray-700">
          <span className="text-sm font-medium text-slate-600 dark:text-gray-300">turn:</span>
          <motion.div
            key={step.turn}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            className="w-10 h-8 rounded-md bg-amber-100 dark:bg-amber-900/40 flex items-center justify-center text-sm font-bold text-amber-700 dark:text-amber-300"
          >
            {step.turn ?? "-"}
          </motion.div>
        </div>

        <div className="flex items-center gap-3 bg-white dark:bg-gray-800 px-5 py-3 rounded-lg shadow-sm border border-slate-200 dark:border-gray-700">
          <Shield className="w-5 h-5 text-green-500" />
          <span className="text-sm font-medium text-slate-600 dark:text-gray-300">Critical Section:</span>
          <div className="flex gap-2">
            {step.p0InCS && (
              <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} className="text-xs px-2 py-1 bg-blue-100 text-blue-700 dark:bg-blue-800 dark:text-blue-200 rounded-full">P0</motion.span>
            )}
            {step.p1InCS && (
              <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} className="text-xs px-2 py-1 bg-purple-100 text-purple-700 dark:bg-purple-800 dark:text-purple-200 rounded-full">P1</motion.span>
            )}
            {!step.p0InCS && !step.p1InCS && (
              <span className="text-xs text-slate-400 dark:text-gray-500">Empty</span>
            )}
          </div>
        </div>
      </div>

      {/* Code Panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {renderCodePanel(P0_LINES, 0, step.p0Line)}
        {renderCodePanel(P1_LINES, 1, step.p1Line)}
      </div>

      {/* Description */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="text-center mb-4 p-3 bg-white dark:bg-gray-800 rounded-lg border border-slate-200 dark:border-gray-700"
        >
          <span className="text-sm text-slate-700 dark:text-gray-200">{step.description}</span>
        </motion.div>
      </AnimatePresence>

      {/* Progress */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-slate-500 dark:text-gray-400 mb-1">
          <span>Step {currentStep + 1} / {ALL_STEPS.length}</span>
          <span>{Math.round(((currentStep + 1) / ALL_STEPS.length) * 100)}%</span>
        </div>
        <div className="w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-blue-500 rounded-full"
            animate={{ width: `${((currentStep + 1) / ALL_STEPS.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={reset}
          className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 transition-colors"
        >
          <RotateCcw className="w-5 h-5" />
        </button>
        <button
          onClick={goPrev}
          disabled={currentStep === 0}
          className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 transition-colors disabled:opacity-40"
        >
          <StepForward className="w-5 h-5 rotate-180" />
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-5 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white font-medium transition-colors flex items-center gap-2"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button
          onClick={goNext}
          disabled={currentStep >= ALL_STEPS.length - 1}
          className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 transition-colors disabled:opacity-40"
        >
          <StepForward className="w-5 h-5" />
        </button>

        <div className="ml-4 flex items-center gap-2">
          <span className="text-xs text-slate-500 dark:text-gray-400">Speed:</span>
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="text-xs bg-white dark:bg-gray-700 border border-slate-300 dark:border-gray-600 rounded px-2 py-1 text-slate-700 dark:text-gray-200"
          >
            <option value={2000}>Slow</option>
            <option value={1200}>Normal</option>
            <option value={600}>Fast</option>
          </select>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex justify-center gap-6 text-xs text-slate-500 dark:text-gray-400">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-blue-500" /> P0 executing
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-purple-500" /> P1 executing
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-yellow-200 dark:bg-yellow-700" /> Highlighted line
        </div>
      </div>
    </div>
  );
}
