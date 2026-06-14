"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Moon, BarChart3, Sliders } from "lucide-react";

const C_STATES = [
  { id: "C0", name: "C0 – Active", power: 100, latency: 0, savings: 0, color: "#10b981" },
  { id: "C1", name: "C1 – Halt", power: 60, latency: 1, savings: 40, color: "#f59e0b" },
  { id: "C3", name: "C3 – Sleep", power: 20, latency: 50, savings: 80, color: "#f97316" },
  { id: "C6", name: "C6 – Deep Sleep", power: 5, latency: 200, savings: 95, color: "#ef4444" },
];

const FREQ_POINTS = [
  { freq: 0.8, voltage: 0.85 },
  { freq: 1.2, voltage: 1.00 },
  { freq: 1.6, voltage: 1.10 },
  { freq: 2.0, voltage: 1.20 },
  { freq: 2.4, voltage: 1.30 },
  { freq: 2.8, voltage: 1.40 },
  { freq: 3.2, voltage: 1.50 },
];

const CAPACITANCE = 30;

function calcPower(voltage: number, freq: number): number {
  return CAPACITANCE * voltage * voltage * freq;
}

function interpVoltage(targetFreq: number): number {
  if (targetFreq <= FREQ_POINTS[0].freq) return FREQ_POINTS[0].voltage;
  if (targetFreq >= FREQ_POINTS[FREQ_POINTS.length - 1].freq)
    return FREQ_POINTS[FREQ_POINTS.length - 1].voltage;
  for (let i = 0; i < FREQ_POINTS.length - 1; i++) {
    const a = FREQ_POINTS[i];
    const b = FREQ_POINTS[i + 1];
    if (targetFreq >= a.freq && targetFreq <= b.freq) {
      const t = (targetFreq - a.freq) / (b.freq - a.freq);
      return a.voltage + t * (b.voltage - a.voltage);
    }
  }
  return 1.0;
}

export default function EnergyAwareScheduling() {
  const [tab, setTab] = useState<"dvfs" | "cstates">("dvfs");
  const [freqIdx, setFreqIdx] = useState(3);

  const freq = FREQ_POINTS[freqIdx].freq;
  const voltage = FREQ_POINTS[freqIdx].voltage;
  const power = useMemo(() => calcPower(voltage, freq), [voltage, freq]);
  const maxPower = useMemo(
    () => calcPower(FREQ_POINTS[FREQ_POINTS.length - 1].voltage, FREQ_POINTS[FREQ_POINTS.length - 1].freq),
    [],
  );
  const powerRatio = power / maxPower;

  const [activeState, setActiveState] = useState(0);

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 dark:from-slate-900 dark:to-amber-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2 text-center flex items-center justify-center gap-2">
        <Zap className="w-6 h-6 text-amber-500" />
        Energy-Aware Scheduling
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 text-center mb-4">
        DVFS &amp; C-State Power Management Visualization
      </p>

      <div className="flex justify-center gap-2 mb-5">
        {(["dvfs", "cstates"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              tab === t
                ? "bg-amber-500 text-white shadow"
                : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600"
            }`}
          >
            {t === "dvfs" ? (
              <span className="flex items-center gap-1"><Sliders className="w-4 h-4" /> DVFS</span>
            ) : (
              <span className="flex items-center gap-1"><Moon className="w-4 h-4" /> C-States</span>
            )}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {tab === "dvfs" ? (
          <motion.div
            key="dvfs"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.25 }}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
                <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-3 flex items-center gap-1">
                  <Sliders className="w-4 h-4 text-amber-500" /> Frequency Control
                </h4>
                <label className="text-xs text-slate-500 dark:text-slate-400">
                  Frequency: <span className="font-mono text-amber-600 dark:text-amber-400">{freq.toFixed(1)} GHz</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={FREQ_POINTS.length - 1}
                  step={1}
                  value={freqIdx}
                  onChange={(e) => setFreqIdx(Number(e.target.value))}
                  className="w-full mt-1 accent-amber-500"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  {FREQ_POINTS.map((p) => (
                    <span key={p.freq}>{p.freq}</span>
                  ))}
                </div>

                <div className="mt-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Voltage (V)</span>
                    <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">{voltage.toFixed(2)} V</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Power (P = C·V²·f)</span>
                    <span className="font-mono font-semibold text-red-600 dark:text-red-400">{power.toFixed(1)} W</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Capacitance (C)</span>
                    <span className="font-mono text-slate-600 dark:text-slate-300">{CAPACITANCE} nF</span>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
                <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-3 flex items-center gap-1">
                  <BarChart3 className="w-4 h-4 text-amber-500" /> Power Breakdown
                </h4>

                <svg viewBox="0 0 300 200" className="w-full">
                  <rect x="0" y="0" width="300" height="200" rx="8" className="fill-slate-50 dark:fill-slate-900" />

                  {FREQ_POINTS.map((p, i) => {
                    const pw = calcPower(p.voltage, p.freq);
                    const barH = (pw / maxPower) * 140;
                    const x = 20 + i * 38;
                    const isActive = i === freqIdx;
                    return (
                      <g key={p.freq}>
                        <motion.rect
                          x={x}
                          y={170 - barH}
                          width={28}
                          height={barH}
                          rx={4}
                          fill={isActive ? "#f59e0b" : "#d4d4d8"}
                          className={isActive ? "" : "dark:fill-slate-600"}
                          animate={{ height: barH, y: 170 - barH }}
                          transition={{ type: "spring", stiffness: 200, damping: 20 }}
                        />
                        <text
                          x={x + 14}
                          y={186}
                          textAnchor="middle"
                          className="fill-slate-500 dark:fill-slate-400"
                          fontSize="9"
                        >
                          {p.freq}
                        </text>
                        {isActive && (
                          <motion.text
                            x={x + 14}
                            y={165 - barH}
                            textAnchor="middle"
                            className="fill-amber-600 dark:fill-amber-400"
                            fontSize="10"
                            fontWeight="bold"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                          >
                            {pw.toFixed(0)}W
                          </motion.text>
                        )}
                      </g>
                    );
                  })}
                  <text x="150" y="198" textAnchor="middle" className="fill-slate-400" fontSize="9">
                    Frequency (GHz)
                  </text>
                </svg>

                <div className="mt-3 space-y-1.5">
                  <div className="text-xs text-slate-500 dark:text-slate-400">Power Ratio</div>
                  <div className="w-full h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: `linear-gradient(90deg, #10b981, #f59e0b, #ef4444)` }}
                      animate={{ width: `${powerRatio * 100}%` }}
                      transition={{ type: "spring", stiffness: 120 }}
                    />
                  </div>
                  <div className="text-xs text-slate-400 text-right">{(powerRatio * 100).toFixed(1)}% of max</div>
                </div>
              </div>
            </div>

            <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
              <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-2">
                Formula: P = C × V² × f
              </h4>
              <div className="grid grid-cols-3 gap-3 text-center">
                {[
                  { label: "C (Capacitance)", value: `${CAPACITANCE} nF`, color: "text-emerald-600 dark:text-emerald-400" },
                  { label: "V (Voltage)", value: `${voltage.toFixed(2)} V`, color: "text-blue-600 dark:text-blue-400" },
                  { label: "f (Frequency)", value: `${freq.toFixed(1)} GHz`, color: "text-amber-600 dark:text-amber-400" },
                ].map((item) => (
                  <div key={item.label} className="bg-slate-50 dark:bg-slate-900 rounded-lg p-2">
                    <div className="text-xs text-slate-500 dark:text-slate-400">{item.label}</div>
                    <div className={`text-lg font-mono font-bold ${item.color}`}>{item.value}</div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-2 text-center">
                Power scales linearly with frequency but quadratically with voltage — reducing V is the most effective way to save energy.
              </p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="cstates"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.25 }}
          >
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow">
              <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-1">
                <Moon className="w-4 h-4 text-amber-500" /> C-State Transition Diagram
              </h4>

              <div className="flex items-center justify-center gap-2 mb-5 flex-wrap">
                {C_STATES.map((s, i) => (
                  <React.Fragment key={s.id}>
                    <motion.button
                      onClick={() => setActiveState(i)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`relative px-4 py-3 rounded-lg border-2 text-sm font-semibold transition-colors ${
                        activeState === i
                          ? "border-amber-500 bg-amber-50 dark:bg-amber-900/30 shadow-lg"
                          : "border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-800"
                      }`}
                    >
                      <motion.div
                        className="w-3 h-3 rounded-full mx-auto mb-1"
                        style={{ backgroundColor: s.color }}
                        animate={
                          activeState === i
                            ? { scale: [1, 1.3, 1], opacity: [1, 0.7, 1] }
                            : {}
                        }
                        transition={{ repeat: Infinity, duration: 1.5 }}
                      />
                      <span className="text-slate-800 dark:text-slate-100">{s.id}</span>
                    </motion.button>
                    {i < C_STATES.length - 1 && (
                      <motion.div
                        className="text-slate-300 dark:text-slate-600 text-lg"
                        animate={{ x: [0, 4, 0] }}
                        transition={{ repeat: Infinity, duration: 1, delay: i * 0.2 }}
                      >
                        →
                      </motion.div>
                    )}
                  </React.Fragment>
                ))}
              </div>

              <AnimatePresence mode="wait">
                <motion.div
                  key={activeState}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                  className="grid grid-cols-1 sm:grid-cols-3 gap-4"
                >
                  <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 text-center">
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Power</div>
                    <div className="text-2xl font-mono font-bold" style={{ color: C_STATES[activeState].color }}>
                      {C_STATES[activeState].power}%
                    </div>
                    <div className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full mt-2 overflow-hidden">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ backgroundColor: C_STATES[activeState].color }}
                        animate={{ width: `${C_STATES[activeState].power}%` }}
                        transition={{ type: "spring", stiffness: 120 }}
                      />
                    </div>
                  </div>

                  <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 text-center">
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Wake-up Latency</div>
                    <div className="text-2xl font-mono font-bold text-blue-600 dark:text-blue-400">
                      {C_STATES[activeState].latency === 0 ? "—" : `${C_STATES[activeState].latency} μs`}
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      {C_STATES[activeState].latency === 0 ? "Already active" : C_STATES[activeState].latency > 100 ? "High penalty" : "Low penalty"}
                    </div>
                  </div>

                  <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 text-center">
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Energy Savings</div>
                    <div className="text-2xl font-mono font-bold text-emerald-600 dark:text-emerald-400">
                      {C_STATES[activeState].savings}%
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      vs. C0 active state
                    </div>
                  </div>
                </motion.div>
              </AnimatePresence>
            </div>

            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
              {C_STATES.map((s, i) => (
                <motion.div
                  key={s.id}
                  className={`bg-white dark:bg-gray-800 rounded-lg p-3 shadow cursor-pointer border-2 transition-colors ${
                    activeState === i ? "border-amber-400" : "border-transparent"
                  }`}
                  whileHover={{ y: -2 }}
                  onClick={() => setActiveState(i)}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: s.color }} />
                    <span className="text-xs font-bold text-slate-700 dark:text-slate-200">{s.id}</span>
                  </div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug">{s.name}</div>
                  <div className="text-xs font-mono mt-1 text-slate-600 dark:text-slate-300">
                    {s.power}% power · {s.latency}μs wake
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
              <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-3">Trade-off Summary</h4>
              <div className="flex items-end gap-1 h-24">
                {C_STATES.map((s, i) => (
                  <div key={s.id} className="flex-1 flex flex-col items-center">
                    <motion.div
                      className="w-full rounded-t"
                      style={{ backgroundColor: s.color }}
                      initial={{ height: 0 }}
                      animate={{ height: `${(s.savings / 100) * 80}px` }}
                      transition={{ delay: i * 0.1, type: "spring" }}
                    />
                    <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">{s.id}</div>
                    <div className="text-[9px] text-slate-400">{s.latency}μs</div>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-2 text-center">
                Deeper states save more energy but incur higher wake-up latency — the scheduler must balance these tradeoffs.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
