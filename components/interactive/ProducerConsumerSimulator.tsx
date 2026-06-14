"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Plus, Minus, Package, ShoppingCart, Lock, Unlock } from "lucide-react";

const BUFFER_SIZE = 8;

interface BufferSlot {
  value: number | null;
  color: string;
}

interface SemaphoreState {
  empty: number;
  full: number;
  mutex: number;
}

interface ActorState {
  id: number;
  type: "producer" | "consumer";
  status: "idle" | "waiting_empty" | "waiting_full" | "waiting_mutex" | "in_cs" | "producing" | "consuming";
  color: string;
}

const COLORS = ["bg-blue-400", "bg-emerald-400", "bg-purple-400", "bg-rose-400", "bg-amber-400", "bg-cyan-400"];

export default function ProducerConsumerSimulator() {
  const [buffer, setBuffer] = useState<BufferSlot[]>(
    Array.from({ length: BUFFER_SIZE }, () => ({ value: null, color: "" }))
  );
  const [inPtr, setInPtr] = useState(0);
  const [outPtr, setOutPtr] = useState(0);
  const [semaphores, setSemaphores] = useState<SemaphoreState>({ empty: BUFFER_SIZE, full: 0, mutex: 1 });
  const [producers, setProducers] = useState<ActorState[]>([
    { id: 0, type: "producer", status: "idle", color: COLORS[0] },
  ]);
  const [consumers, setConsumers] = useState<ActorState[]>([
    { id: 0, type: "consumer", status: "idle", color: COLORS[1] },
  ]);
  const [log, setLog] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(800);
  const nextItem = useRef(1);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 15));
  }, []);

  const addProducer = () => {
    if (producers.length >= 3) return;
    const id = producers.length;
    setProducers((prev) => [...prev, { id, type: "producer", status: "idle", color: COLORS[id % COLORS.length] }]);
    addLog(`Producer ${id} added`);
  };

  const removeProducer = () => {
    if (producers.length <= 1) return;
    setProducers((prev) => prev.slice(0, -1));
    addLog(`Producer removed`);
  };

  const addConsumer = () => {
    if (consumers.length >= 3) return;
    const id = consumers.length;
    setConsumers((prev) => [...prev, { id, type: "consumer", status: "idle", color: COLORS[(id + 3) % COLORS.length] }]);
    addLog(`Consumer ${id} added`);
  };

  const removeConsumer = () => {
    if (consumers.length <= 1) return;
    setConsumers((prev) => prev.slice(0, -1));
    addLog(`Consumer removed`);
  };

  const simulateStep = useCallback(() => {
    const s = { ...semaphores };
    const b = [...buffer.map((s) => ({ ...s }))];
    let ip = inPtr;
    let op = outPtr;

    // Randomly pick producer or consumer
    const pickProducer = Math.random() < 0.5;

    if (pickProducer && producers.length > 0) {
      const pid = Math.floor(Math.random() * producers.length);
      // Producer: P(empty), P(mutex), produce, V(mutex), V(full)
      if (s.empty > 0 && s.mutex === 1) {
        s.empty--;
        s.mutex--;
        const item = nextItem.current++;
        b[ip] = { value: item, color: COLORS[pid % COLORS.length] };
        addLog(`P${pid}: Produced item ${item} at slot ${ip}`);
        ip = (ip + 1) % BUFFER_SIZE;
        s.mutex++;
        s.full++;

        setProducers((prev) =>
          prev.map((p, i) => (i === pid ? { ...p, status: "producing" as const } : { ...p, status: "idle" as const }))
        );
      } else if (s.empty <= 0) {
        addLog(`P${pid}: Buffer full! Waiting...`);
        setProducers((prev) =>
          prev.map((p, i) => (i === pid ? { ...p, status: "waiting_full" as const } : p))
        );
      } else {
        addLog(`P${pid}: Mutex held, waiting...`);
        setProducers((prev) =>
          prev.map((p, i) => (i === pid ? { ...p, status: "waiting_mutex" as const } : p))
        );
      }
    } else if (consumers.length > 0) {
      const cid = Math.floor(Math.random() * consumers.length);
      // Consumer: P(full), P(mutex), consume, V(mutex), V(empty)
      if (s.full > 0 && s.mutex === 1) {
        s.full--;
        s.mutex--;
        const item = b[op].value;
        b[op] = { value: null, color: "" };
        addLog(`C${cid}: Consumed item ${item} from slot ${op}`);
        op = (op + 1) % BUFFER_SIZE;
        s.mutex++;
        s.empty++;

        setConsumers((prev) =>
          prev.map((c, i) => (i === cid ? { ...c, status: "consuming" as const } : { ...c, status: "idle" as const }))
        );
      } else if (s.full <= 0) {
        addLog(`C${cid}: Buffer empty! Waiting...`);
        setConsumers((prev) =>
          prev.map((c, i) => (i === cid ? { ...c, status: "waiting_empty" as const } : c))
        );
      } else {
        addLog(`C${cid}: Mutex held, waiting...`);
        setConsumers((prev) =>
          prev.map((c, i) => (i === cid ? { ...c, status: "waiting_mutex" as const } : c))
        );
      }
    }

    setBuffer(b);
    setInPtr(ip);
    setOutPtr(op);
    setSemaphores(s);
  }, [semaphores, buffer, inPtr, outPtr, producers.length, consumers.length, addLog]);

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(simulateStep, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, speed, simulateStep]);

  const reset = useCallback(() => {
    setIsRunning(false);
    setBuffer(Array.from({ length: BUFFER_SIZE }, () => ({ value: null, color: "" })));
    setInPtr(0);
    setOutPtr(0);
    setSemaphores({ empty: BUFFER_SIZE, full: 0, mutex: 1 });
    setProducers([{ id: 0, type: "producer", status: "idle", color: COLORS[0] }]);
    setConsumers([{ id: 0, type: "consumer", status: "idle", color: COLORS[1] }]);
    setLog([]);
    nextItem.current = 1;
  }, []);

  const getSemColor = (val: number, max: number) => {
    if (val <= 0) return "bg-red-100 border-red-300 text-red-700 dark:bg-red-900/30 dark:border-red-600";
    if (val < max * 0.3) return "bg-amber-100 border-amber-300 text-amber-700 dark:bg-amber-900/30 dark:border-amber-600";
    return "bg-green-100 border-green-300 text-green-700 dark:bg-green-900/30 dark:border-green-600";
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Producer-Consumer Simulator
      </h2>

      {/* Semaphore State */}
      <div className="flex justify-center gap-4 mb-6">
        {[
          { name: "empty", value: semaphores.empty, max: BUFFER_SIZE, icon: Package },
          { name: "full", value: semaphores.full, max: BUFFER_SIZE, icon: ShoppingCart },
          { name: "mutex", value: semaphores.mutex, max: 1, icon: semaphores.mutex === 1 ? Unlock : Lock },
        ].map((sem) => (
          <div key={sem.name} className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 ${getSemColor(sem.value, sem.max)}`}>
            <sem.icon className="w-4 h-4" />
            <span className="text-sm font-bold">{sem.name}:</span>
            <span className="text-lg font-mono">{sem.value}</span>
          </div>
        ))}
      </div>

      {/* Circular Buffer */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-slate-200 dark:border-gray-700 mb-6">
        <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-4 text-center">Circular Buffer (Size: {BUFFER_SIZE})</h4>
        <div className="flex justify-center">
          <div className="relative w-80 h-80">
            {/* Buffer slots in a circle */}
            {buffer.map((slot, i) => {
              const angle = (i / BUFFER_SIZE) * 2 * Math.PI - Math.PI / 2;
              const radius = 120;
              const x = 160 + radius * Math.cos(angle) - 30;
              const y = 160 + radius * Math.sin(angle) - 30;
              const isInPtr = i === inPtr;
              const isOutPtr = i === outPtr;

              return (
                <div key={i} className="absolute" style={{ left: x, top: y }}>
                  <motion.div
                    animate={{
                      scale: slot.value !== null ? 1 : 0.9,
                      borderColor: isInPtr ? "#3b82f6" : isOutPtr ? "#f59e0b" : "#e2e8f0",
                    }}
                    className={`w-16 h-16 rounded-lg border-2 flex flex-col items-center justify-center ${
                      slot.value !== null
                        ? "bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-700 dark:to-gray-600"
                        : "bg-slate-50 dark:bg-gray-800"
                    }`}
                  >
                    <span className="text-xs text-slate-400 dark:text-gray-500">[{i}]</span>
                    {slot.value !== null && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className={`w-8 h-8 rounded-md ${slot.color} flex items-center justify-center text-white text-xs font-bold shadow-sm`}
                      >
                        {slot.value}
                      </motion.div>
                    )}
                  </motion.div>
                  {/* Pointer labels */}
                  {isInPtr && (
                    <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-xs font-bold text-blue-500">IN</div>
                  )}
                  {isOutPtr && (
                    <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-xs font-bold text-amber-500">OUT</div>
                  )}
                </div>
              );
            })}
            {/* Center */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center">
              <div className="text-sm font-bold text-slate-600 dark:text-gray-300">Buffer</div>
              <div className="text-xs text-slate-400 dark:text-gray-500">
                {BUFFER_SIZE - semaphores.empty}/{BUFFER_SIZE} used
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Producers & Consumers */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Producers */}
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-bold text-blue-700 dark:text-blue-300">Producers</h4>
            <div className="flex gap-1">
              <button onClick={removeProducer} className="p-1 rounded bg-blue-200 hover:bg-blue-300 dark:bg-blue-800 dark:hover:bg-blue-700">
                <Minus className="w-3 h-3" />
              </button>
              <button onClick={addProducer} className="p-1 rounded bg-blue-200 hover:bg-blue-300 dark:bg-blue-800 dark:hover:bg-blue-700">
                <Plus className="w-3 h-3" />
              </button>
            </div>
          </div>
          <div className="space-y-2">
            {producers.map((p) => (
              <div key={p.id} className="flex items-center gap-2 p-2 bg-white dark:bg-gray-800 rounded border border-slate-200 dark:border-gray-700">
                <div className={`w-6 h-6 rounded-full ${p.color} flex items-center justify-center text-white text-xs font-bold`}>P{p.id}</div>
                <span className="text-xs text-slate-600 dark:text-gray-300 capitalize">{p.status.replace("_", " ")}</span>
                {p.status === "producing" && (
                  <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }} className="ml-auto">
                    <Package className="w-4 h-4 text-blue-500" />
                  </motion.div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Consumers */}
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-bold text-emerald-700 dark:text-emerald-300">Consumers</h4>
            <div className="flex gap-1">
              <button onClick={removeConsumer} className="p-1 rounded bg-emerald-200 hover:bg-emerald-300 dark:bg-emerald-800 dark:hover:bg-emerald-700">
                <Minus className="w-3 h-3" />
              </button>
              <button onClick={addConsumer} className="p-1 rounded bg-emerald-200 hover:bg-emerald-300 dark:bg-emerald-800 dark:hover:bg-emerald-700">
                <Plus className="w-3 h-3" />
              </button>
            </div>
          </div>
          <div className="space-y-2">
            {consumers.map((c) => (
              <div key={c.id} className="flex items-center gap-2 p-2 bg-white dark:bg-gray-800 rounded border border-slate-200 dark:border-gray-700">
                <div className={`w-6 h-6 rounded-full ${c.color} flex items-center justify-center text-white text-xs font-bold`}>C{c.id}</div>
                <span className="text-xs text-slate-600 dark:text-gray-300 capitalize">{c.status.replace("_", " ")}</span>
                {c.status === "consuming" && (
                  <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }} className="ml-auto">
                    <ShoppingCart className="w-4 h-4 text-emerald-500" />
                  </motion.div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Correct P() Order */}
      <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-4 border border-amber-200 dark:border-amber-800 mb-6">
        <h4 className="text-sm font-bold text-amber-700 dark:text-amber-300 mb-2">Correct Semaphore Ordering</h4>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div className="font-mono text-slate-700 dark:text-gray-200">
            <div className="text-blue-600 dark:text-blue-400 font-bold mb-1">Producer:</div>
            <div>P(empty)  // wait for space</div>
            <div>P(mutex)  // acquire lock</div>
            <div>produce item</div>
            <div>V(mutex)  // release lock</div>
            <div>V(full)   // signal item ready</div>
          </div>
          <div className="font-mono text-slate-700 dark:text-gray-200">
            <div className="text-emerald-600 dark:text-emerald-400 font-bold mb-1">Consumer:</div>
            <div>P(full)   // wait for item</div>
            <div>P(mutex)  // acquire lock</div>
            <div>consume item</div>
            <div>V(mutex)  // release lock</div>
            <div>V(empty)  // signal space free</div>
          </div>
        </div>
        <p className="text-xs text-amber-600 dark:text-amber-400 mt-2">
          Note: P(empty/full) BEFORE P(mutex) to avoid deadlock!
        </p>
      </div>

      {/* Event Log */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-slate-200 dark:border-gray-700 mb-4 max-h-32 overflow-y-auto">
        {log.length === 0 ? (
          <p className="text-xs text-slate-400 dark:text-gray-500 text-center">Events will appear here...</p>
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
          className="px-5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-white font-medium flex items-center gap-2"
        >
          {isRunning ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Run</>}
        </button>
        <button
          onClick={simulateStep}
          disabled={isRunning}
          className="px-4 py-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 text-sm disabled:opacity-40"
        >
          Step
        </button>
        <div className="ml-4 flex items-center gap-2">
          <span className="text-xs text-slate-500 dark:text-gray-400">Speed:</span>
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="text-xs bg-white dark:bg-gray-700 border border-slate-300 dark:border-gray-600 rounded px-2 py-1"
          >
            <option value={1500}>Slow</option>
            <option value={800}>Normal</option>
            <option value={400}>Fast</option>
          </select>
        </div>
      </div>
    </div>
  );
}
