"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function EmbodiedAIDemo() {
    const [mode, setMode] = useState<"sim" | "real">("sim");
    const [noiseLevel, setNoiseLevel] = useState(0.5);
    const [successCount, setSuccessCount] = useState({ sim: 0, real: 0 });
    const [attempts, setAttempts] = useState({ sim: 0, real: 0 });

    // Robot state
    const [robotAngle, setRobotAngle] = useState(0);
    const [targetPos, setTargetPos] = useState({ x: 150, y: 100 });
    const [isMoving, setIsMoving] = useState(false);

    // Run a trial
    useEffect(() => {
        const runTrial = () => {
            setIsMoving(true);
            setAttempts(prev => ({ ...prev, [mode]: prev[mode] + 1 }));

            // 1. Generate new target
            const newTarget = {
                x: 100 + Math.random() * 100,
                y: 50 + Math.random() * 100
            };
            setTargetPos(newTarget);

            // 2. Compute "ideal" angle (inverse kinematics simplified)
            const idealAngle = Math.atan2(newTarget.y, newTarget.x) * (180 / Math.PI);

            // 3. Add noise (domain gap)
            // Sim has small noise, Real has large noise (unless Domain Randomization used)
            let noise = 0;
            if (mode === "sim") {
                noise = (Math.random() - 0.5) * 5; // +/- 2.5 deg
            } else {
                noise = (Math.random() - 0.5) * (noiseLevel * 60); // Up to +/- 30 deg depending on level
            }

            const actualAngle = idealAngle + noise;

            // Check success (within 10 degrees)
            const success = Math.abs(actualAngle - idealAngle) < 10;

            // Animate
            setRobotAngle(actualAngle);

            setTimeout(() => {
                if (success) {
                    setSuccessCount(prev => ({ ...prev, [mode]: prev[mode] + 1 }));
                }
                setIsMoving(false);
            }, 800);
        };

        if (isMoving) return;

        const interval = setInterval(() => {
            runTrial();
        }, 2000);

        return () => clearInterval(interval);
    }, [mode, noiseLevel, isMoving]);

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-slate-100 dark:bg-slate-900 rounded-2xl shadow-lg flex flex-col md:flex-row gap-8">
            {/* Control Panel */}
            <div className="w-full md:w-1/3 flex flex-col gap-6">
                <div>
                    <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">Sim-to-Real Transfer</h3>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                        Test the policy in Simulation vs. Real World.
                        In "Real" mode, physics noise (friction, sensor error) increases, causing failure unless the policy is robust.
                    </p>
                </div>

                <div className="flex bg-white dark:bg-slate-800 p-1 rounded-lg border border-slate-200 dark:border-slate-700">
                    <button
                        onClick={() => setMode("sim")}
                        className={`flex-1 py-2 rounded-md font-bold text-sm transition ${mode === "sim"
                            ? "bg-blue-500 text-white shadow"
                            : "text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700"
                            }`}
                    >
                        Simulation
                    </button>
                    <button
                        onClick={() => setMode("real")}
                        className={`flex-1 py-2 rounded-md font-bold text-sm transition ${mode === "real"
                            ? "bg-orange-500 text-white shadow"
                            : "text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700"
                            }`}
                    >
                        Real World
                    </button>
                </div>

                {mode === "real" && (
                    <div className="space-y-2">
                        <div className="flex justify-between text-xs font-bold text-slate-500">
                            <span>Reality Gap (Noise)</span>
                            <span>{(noiseLevel * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={noiseLevel}
                            onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                            className="w-full accent-orange-500"
                        />
                        <div className="text-xs text-slate-400">
                            High noise simulates friction, mass variance, and light changes.
                        </div>
                    </div>
                )}

                <div className="bg-white dark:bg-slate-800 p-4 rounded-xl border border-slate-200 dark:border-slate-700">
                    <div className="text-sm font-bold text-slate-500 mb-2">Success Rate ({mode.toUpperCase()})</div>
                    <div className="flex items-baseline gap-2">
                        <span className={`text-4xl font-bold ${(successCount[mode] / Math.max(1, attempts[mode])) > 0.8 ? "text-green-500" :
                            (successCount[mode] / Math.max(1, attempts[mode])) > 0.5 ? "text-yellow-500" : "text-red-500"
                            }`}>
                            {attempts[mode] === 0 ? "0" : ((successCount[mode] / attempts[mode]) * 100).toFixed(0)}%
                        </span>
                        <span className="text-sm text-slate-400">
                            ({successCount[mode]}/{attempts[mode]})
                        </span>
                    </div>
                </div>
            </div>

            {/* Visualization */}
            <div className="w-full md:w-2/3 relative h-[300px] bg-slate-200 dark:bg-slate-800/50 rounded-xl overflow-hidden border-2 border-slate-300 dark:border-slate-700">
                {/* Background Grid */}
                <div className="absolute inset-0 opacity-20"
                    style={{
                        backgroundImage: "linear-gradient(#94a3b8 1px, transparent 1px), linear-gradient(90deg, #94a3b8 1px, transparent 1px)",
                        backgroundSize: "20px 20px"
                    }}
                />

                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <span className="text-9xl font-black text-slate-300 dark:text-slate-700 opacity-20 uppercase">
                        {mode}
                    </span>
                </div>

                {/* Robot Base */}
                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 w-16 h-8 bg-slate-700 rounded-t-lg z-10" />

                {/* Robot Arm */}
                <motion.div
                    className="absolute bottom-12 left-1/2 w-4 h-40 bg-blue-500 rounded-full origin-bottom"
                    style={{ left: "calc(50% - 8px)" }}
                    animate={{ rotate: robotAngle - 90 }} // -90 to align 0 deg with right -> up
                    transition={{ type: "spring", stiffness: 100, damping: 10 }}
                >
                    {/* End Effector */}
                    <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full border-4 border-blue-600 bg-white dark:bg-slate-900" />
                </motion.div>

                {/* Target */}
                <motion.div
                    className="absolute w-8 h-8 rounded-full bg-green-500 shadow-lg shadow-green-500/50 flex items-center justify-center"
                    animate={{
                        left: `50%`,
                        bottom: `20px`,
                        x: targetPos.x - 150, // simple mapping relative to center
                        y: -targetPos.y + 100 // up is negative y in CSS usually, but here using bottom
                    }}
                    style={{
                        bottom: 20 + targetPos.y,
                        left: `50%`,
                        marginLeft: targetPos.x - 150
                    }}
                >
                    <div className="w-2 h-2 bg-white rounded-full animate-ping" />
                </motion.div>

            </div>
        </div>
    );
}
