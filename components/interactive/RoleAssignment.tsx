"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function RoleAssignment() {
    const [taskLoad, setTaskLoad] = useState([30, 25, 45]);  // Load percentage for 3 tasks
    const agents = 4;

    const roles = ["Scout", "Defender", "Attacker", "Support"];
    const roleColors = ["bg-blue-500", "bg-green-500", "bg-red-500", "bg-purple-500"];

    // Assign agents to roles based on task load
    const assignments = taskLoad.map((load, idx) => {
        const count = Math.max(1, Math.round((load / 100) * agents));
        return { task: `Task ${idx + 1}`, role: roles[idx % roles.length], count, load };
    });

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Dynamic Role Assignment
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Adjust task loads to see agent reallocation
                </p>
            </div>

            {/* Task Load Controls */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Task Load Distribution</h4>
                <div className="space-y-4">
                    {taskLoad.map((load, idx) => (
                        <div key={idx}>
                            <div className="flex justify-between text-sm mb-2">
                                <span className="font-semibold">{roles[idx]} Task</span>
                                <span className="text-slate-600 dark:text-slate-400">
                                    {load}% load → {assignments[idx].count} agents
                                </span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="100"
                                value={load}
                                onChange={(e) => {
                                    const newLoads = [...taskLoad];
                                    newLoads[idx] = parseInt(e.target.value);
                                    setTaskLoad(newLoads);
                                }}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                            />
                            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mt-2">
                                <div
                                    className={`h-full ${roleColors[idx]}`}
                                    style={{ width: `${load}%` }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Agent Assignments Visualization */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Agent Role Assignments ({agents} total)</h4>
                <div className="grid grid-cols-4 gap-4">
                    {Array.from({ length: agents }).map((_, agentIdx) => {
                        // Determine which role this agent has
                        let assignedRole = "Support";
                        let assignedColor = "bg-gray-400";
                        let cumulativeCount = 0;

                        for (const assignment of assignments) {
                            if (agentIdx < cumulativeCount + assignment.count) {
                                assignedRole = assignment.role;
                                assignedColor = roleColors[roles.indexOf(assignment.role)];
                                break;
                            }
                            cumulativeCount += assignment.count;
                        }

                        return (
                            <motion.div
                                key={agentIdx}
                                className={`${assignedColor} text-white p-4 rounded-lg shadow-md`}
                                layout
                                transition={{ duration: 0.3 }}
                            >
                                <div className="text-center">
                                    <div className="text-2xl mb-1">🤖</div>
                                    <div className="font-bold text-sm">Agent {agentIdx + 1}</div>
                                    <div className="text-xs mt-1 bg-black/20 px-2 py-1 rounded">
                                        {assignedRole}
                                    </div>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Load Balance Metrics */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Load Balance Metrics</h4>
                <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Total Load</div>
                        <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                            {taskLoad.reduce((a, b) => a + b, 0)}%
                        </div>
                    </div>
                    <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Avg per Agent</div>
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                            {(taskLoad.reduce((a, b) => a + b, 0) / agents).toFixed(1)}%
                        </div>
                    </div>
                    <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Utilization</div>
                        <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                            {Math.min(100, taskLoad.reduce((a, b) => a + b, 0) / agents).toFixed(0)}%
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 Dynamic role assignment optimizes resource utilization in cooperative tasks
            </div>
        </div>
    );
}
