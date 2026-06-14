"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Users, AlertTriangle } from "lucide-react";

type PhilosopherState = "thinking" | "hungry" | "eating";
type Solution = "naive" | "asymmetric" | "room" | "mutex";

interface Philosopher {
  id: number;
  state: PhilosopherState;
  leftFork: number;
  rightFork: number;
}

interface Fork {
  id: number;
  owner: number | null; // philosopher id or null
}

const STATE_COLORS: Record<PhilosopherState, { bg: string; text: string; label: string }> = {
  thinking: { bg: "bg-blue-400", text: "text-white", label: "Thinking" },
  hungry: { bg: "bg-yellow-400", text: "text-yellow-900", label: "Hungry" },
  eating: { bg: "bg-green-400", text: "text-white", label: "Eating" },
};

const PHILOSOPHER_NAMES = ["Kant", "Plato", "Descartes", "Hume", "Aristotle"];

function createInitialPhilosophers(): Philosopher[] {
  return Array.from({ length: 5 }, (_, i) => ({
    id: i,
    state: "thinking" as PhilosopherState,
    leftFork: i,
    rightFork: (i + 1) % 5,
  }));
}

function createInitialForks(): Fork[] {
  return Array.from({ length: 5 }, (_, i) => ({ id: i, owner: null }));
}

export default function DiningPhilosophersAnimation() {
  const [philosophers, setPhilosophers] = useState<Philosopher[]>(createInitialPhilosophers);
  const [forks, setForks] = useState<Fork[]>(createInitialForks);
  const [solution, setSolution] = useState<Solution>("naive");
  const [isRunning, setIsRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [deadlocked, setDeadlocked] = useState(false);
  const [roomSeats, setRoomSeats] = useState(4); // for room limit solution
  const [eatCounts, setEatCounts] = useState<number[]>([0, 0, 0, 0, 0]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [`${msg}`, ...prev].slice(0, 12));
  }, []);

  const reset = useCallback(() => {
    setIsRunning(false);
    setPhilosophers(createInitialPhilosophers());
    setForks(createInitialForks());
    setDeadlocked(false);
    setLog([]);
    setEatCounts([0, 0, 0, 0, 0]);
  }, []);

  useEffect(() => {
    reset();
  }, [solution, reset]);

  const simulateStep = useCallback(() => {
    const phil = philosophers.map((p) => ({ ...p }));
    const fk = forks.map((f) => ({ ...f }));
    const counts = [...eatCounts];
    let isDeadlocked = false;

    // Check for deadlock: all philosophers hungry and no forks available
    const allHungry = phil.every((p) => p.state === "hungry");
    const allForksTaken = fk.every((f) => f.owner !== null);
    if (allHungry && allForksTaken) {
      setDeadlocked(true);
      setIsRunning(false);
      addLog("DEADLOCK! All philosophers holding one fork, waiting for another.");
      return;
    }

    // Each philosopher acts based on their state
    for (let i = 0; i < 5; i++) {
      const p = phil[i];

      if (p.state === "thinking") {
        // Randomly become hungry
        if (Math.random() < 0.4) {
          p.state = "hungry";
          addLog(`${PHILOSOPHER_NAMES[i]} becomes hungry`);
        }
      } else if (p.state === "hungry") {
        const leftFork = fk[p.leftFork];
        const rightFork = fk[p.rightFork];

        let pickLeftFirst = true;
        if (solution === "asymmetric") {
          // Even philosophers pick left first, odd pick right first
          pickLeftFirst = i % 2 === 0;
        }

        if (solution === "room") {
          // Only allow roomSeats philosophers to try
          const hungryCount = phil.filter((pp) => pp.state === "hungry" || pp.state === "eating").length;
          if (hungryCount > roomSeats) {
            continue; // wait outside
          }
        }

        if (pickLeftFirst) {
          // Try left then right
          if (leftFork.owner === null) {
            leftFork.owner = i;
            addLog(`${PHILOSOPHER_NAMES[i]} picks up left fork ${p.leftFork}`);
          }
          if (leftFork.owner === i && rightFork.owner === null) {
            rightFork.owner = i;
            p.state = "eating";
            addLog(`${PHILOSOPHER_NAMES[i]} picks up right fork ${p.rightFork} -> EATING`);
          }
        } else {
          // Try right then left (asymmetric)
          if (rightFork.owner === null) {
            rightFork.owner = i;
            addLog(`${PHILOSOPHER_NAMES[i]} picks up right fork ${p.rightFork}`);
          }
          if (rightFork.owner === i && leftFork.owner === null) {
            leftFork.owner = i;
            p.state = "eating";
            addLog(`${PHILOSOPHER_NAMES[i]} picks up left fork ${p.leftFork} -> EATING`);
          }
        }
      } else if (p.state === "eating") {
        // Randomly finish eating
        if (Math.random() < 0.3) {
          p.state = "thinking";
          fk[p.leftFork].owner = null;
          fk[p.rightFork].owner = null;
          counts[i]++;
          addLog(`${PHILOSOPHER_NAMES[i]} puts down forks, starts thinking (ate ${counts[i]}x)`);
        }
      }
    }

    setPhilosophers(phil);
    setForks(fk);
    setEatCounts(counts);
  }, [philosophers, forks, solution, roomSeats, eatCounts, addLog]);

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(simulateStep, 1200);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, simulateStep]);

  // Calculate philosopher positions on circle
  const getPhilosopherPos = (i: number, radius: number, cx: number, cy: number) => {
    const angle = (i / 5) * 2 * Math.PI - Math.PI / 2;
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  };

  const getForkPos = (i: number, radius: number, cx: number, cy: number) => {
    const angle = ((i + 0.5) / 5) * 2 * Math.PI - Math.PI / 2;
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  };

  const cx = 200, cy = 200, philRadius = 140, forkRadius = 85;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Dining Philosophers Problem
      </h2>

      {/* Solution Selector */}
      <div className="flex justify-center gap-2 mb-6 flex-wrap">
        {([
          { key: "naive", label: "Naive (Deadlock)", desc: "All pick left first" },
          { key: "asymmetric", label: "Asymmetric", desc: "Odd picks right first" },
          { key: "room", label: "Room Limit", desc: "Max 4 at table" },
          { key: "mutex", label: "Resource Hierarchy", desc: "Ordered fork pickup" },
        ] as { key: Solution; label: string; desc: string }[]).map((s) => (
          <button
            key={s.key}
            onClick={() => setSolution(s.key)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              solution === s.key
                ? "bg-rose-500 text-white shadow-md"
                : "bg-white dark:bg-gray-800 text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700 hover:bg-slate-50"
            }`}
          >
            <div>{s.label}</div>
            <div className="text-xs opacity-75">{s.desc}</div>
          </button>
        ))}
      </div>

      {/* Deadlock Warning */}
      <AnimatePresence>
        {deadlocked && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="mb-4 p-4 bg-red-50 dark:bg-red-900/30 border-2 border-red-400 rounded-lg flex items-center gap-3"
          >
            <AlertTriangle className="w-6 h-6 text-red-500 flex-shrink-0" />
            <div>
              <div className="font-bold text-red-700 dark:text-red-300">Deadlock Detected!</div>
              <div className="text-sm text-red-600 dark:text-red-400">
                All 5 philosophers picked up their left fork and are waiting for the right fork forever.
                Try the &quot;Asymmetric&quot; or &quot;Room Limit&quot; solution.
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Table Visualization */}
      <div className="flex justify-center mb-6">
        <div className="relative" style={{ width: 400, height: 400 }}>
          {/* Table */}
          <div
            className="absolute rounded-full bg-gradient-to-br from-amber-100 to-amber-200 dark:from-amber-900/40 dark:to-amber-800/40 border-4 border-amber-300 dark:border-amber-700"
            style={{ left: 50, top: 50, width: 300, height: 300 }}
          />
          <div
            className="absolute flex items-center justify-center"
            style={{ left: 120, top: 120, width: 160, height: 160 }}
          >
            <div className="text-center">
              <Users className="w-8 h-8 text-amber-500 mx-auto mb-1" />
              <span className="text-xs text-amber-600 dark:text-amber-400 font-medium">Dining Table</span>
            </div>
          </div>

          {/* Forks */}
          {forks.map((fork) => {
            const pos = getForkPos(fork.id, forkRadius, cx, cy);
            const owner = fork.owner !== null ? philosophers[fork.owner] : null;
            return (
              <motion.div
                key={`fork-${fork.id}`}
                animate={{
                  x: pos.x - 16,
                  y: pos.y - 16,
                  scale: fork.owner !== null ? 0.8 : 1,
                }}
                className="absolute w-8 h-8 flex items-center justify-center"
              >
                <div
                  className={`w-6 h-6 rounded border-2 flex items-center justify-center text-xs font-bold ${
                    fork.owner !== null
                      ? "bg-gray-300 border-gray-400 text-gray-500 dark:bg-gray-600 dark:border-gray-500"
                      : "bg-amber-200 border-amber-400 text-amber-700 dark:bg-amber-800 dark:border-amber-600 dark:text-amber-200"
                  }`}
                >
                  F{fork.id}
                </div>
              </motion.div>
            );
          })}

          {/* Philosophers */}
          {philosophers.map((phil) => {
            const pos = getPhilosopherPos(phil.id, philRadius, cx, cy);
            const sc = STATE_COLORS[phil.state];
            return (
              <motion.div
                key={`phil-${phil.id}`}
                animate={{ x: pos.x - 32, y: pos.y - 32 }}
                className="absolute w-16 flex flex-col items-center"
              >
                <motion.div
                  animate={{
                    scale: phil.state === "eating" ? [1, 1.1, 1] : 1,
                    backgroundColor: phil.state === "thinking" ? "#60a5fa" : phil.state === "hungry" ? "#facc15" : "#4ade80",
                  }}
                  transition={{ repeat: phil.state === "eating" ? Infinity : 0, duration: 0.8 }}
                  className="w-14 h-14 rounded-full flex items-center justify-center shadow-lg border-2 border-white dark:border-gray-800"
                >
                  <span className={`text-lg font-bold ${sc.text}`}>{phil.id}</span>
                </motion.div>
                <div className="mt-1 text-center">
                  <div className="text-xs font-bold text-slate-700 dark:text-gray-200">{PHILOSOPHER_NAMES[phil.id]}</div>
                  <div className={`text-xs ${phil.state === "thinking" ? "text-blue-500" : phil.state === "hungry" ? "text-yellow-600" : "text-green-500"}`}>
                    {sc.label}
                  </div>
                  <div className="text-xs text-slate-400 dark:text-gray-500">Ate: {eatCounts[phil.id]}</div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-6 mb-4">
        {(["thinking", "hungry", "eating"] as PhilosopherState[]).map((state) => {
          const sc = STATE_COLORS[state];
          return (
            <div key={state} className="flex items-center gap-2">
              <div className={`w-4 h-4 rounded-full ${sc.bg}`} />
              <span className="text-xs text-slate-600 dark:text-gray-300">{sc.label}</span>
            </div>
          );
        })}
      </div>

      {/* Room Limit Control */}
      {solution === "room" && (
        <div className="flex items-center justify-center gap-3 mb-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <span className="text-sm text-purple-700 dark:text-purple-300">Room capacity:</span>
          <button
            onClick={() => setRoomSeats((s) => Math.max(2, s - 1))}
            className="px-3 py-1 rounded bg-purple-200 dark:bg-purple-800 text-purple-700 dark:text-purple-200 font-bold"
          >
            -
          </button>
          <span className="text-lg font-bold text-purple-700 dark:text-purple-300 w-8 text-center">{roomSeats}</span>
          <button
            onClick={() => setRoomSeats((s) => Math.min(5, s + 1))}
            className="px-3 py-1 rounded bg-purple-200 dark:bg-purple-800 text-purple-700 dark:text-purple-200 font-bold"
          >
            +
          </button>
          <span className="text-xs text-purple-500 dark:text-purple-400">(max philosophers at table)</span>
        </div>
      )}

      {/* Event Log */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-slate-200 dark:border-gray-700 mb-4 max-h-28 overflow-y-auto">
        {log.length === 0 ? (
          <p className="text-xs text-slate-400 dark:text-gray-500 text-center">Click Run to start simulation...</p>
        ) : (
          log.map((entry, i) => (
            <div key={i} className="text-xs font-mono text-slate-600 dark:text-gray-300 py-0.5 border-b border-slate-100 dark:border-gray-700 last:border-0">
              {entry}
            </div>
          ))
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300">
          <RotateCcw className="w-5 h-5" />
        </button>
        <button
          onClick={() => setIsRunning(!isRunning)}
          disabled={deadlocked}
          className="px-5 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white font-medium flex items-center gap-2 disabled:opacity-40"
        >
          {isRunning ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Run</>}
        </button>
        <button
          onClick={simulateStep}
          disabled={isRunning || deadlocked}
          className="px-4 py-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 text-sm disabled:opacity-40"
        >
          Step
        </button>
      </div>
    </div>
  );
}
