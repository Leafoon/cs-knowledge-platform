"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function RLMonitoringDashboard() {
    // Simulated metrics state
    const [metrics, setMetrics] = useState({
        requests: 12450,
        latency: 45,
        avgReward: 12.5,
        errorRate: 0.2
    });

    // Simulate real-time updates
    useEffect(() => {
        const interval = setInterval(() => {
            setMetrics(prev => ({
                requests: prev.requests + Math.floor(Math.random() * 10),
                latency: Math.max(20, Math.min(100, prev.latency + (Math.random() - 0.5) * 10)),
                avgReward: Math.max(0, Math.min(20, prev.avgReward + (Math.random() - 0.5) * 1)),
                errorRate: Math.max(0, Math.min(2, prev.errorRate + (Math.random() - 0.5) * 0.1))
            }));
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    // Alert thresholds
    const alerts = [
        { metric: "Latency", threshold: 80, current: metrics.latency, status: metrics.latency > 80 ? "critical" : "normal" },
        { metric: "Error Rate", threshold: 1.0, current: metrics.errorRate, status: metrics.errorRate > 1.0 ? "warning" : "normal" }
    ];

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-slate-900 text-slate-100 rounded-2xl shadow-xl font-mono">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h3 className="text-xl font-bold text-green-400 flex items-center gap-2">
                        <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                        RL Agent Monitoring: Prod-v1
                    </h3>
                </div>
                <div className="text-xs text-slate-400">
                    Last updated: {new Date().toLocaleTimeString()}
                </div>
            </div>

            {/* Top Metrics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <MetricCard
                    label="Total Steps"
                    value={metrics.requests.toLocaleString()}
                    trend="+1.2%"
                    color="blue"
                />
                <MetricCard
                    label="Avg Latency"
                    value={`${metrics.latency.toFixed(1)} ms`}
                    trend={metrics.latency > 50 ? "↗" : "↘"}
                    color={metrics.latency > 80 ? "red" : "green"}
                />
                <MetricCard
                    label="Episode Reward"
                    value={metrics.avgReward.toFixed(2)}
                    trend={metrics.avgReward > 10 ? "↑" : "↓"}
                    color="purple"
                />
                <MetricCard
                    label="Error Rate"
                    value={`${metrics.errorRate.toFixed(2)}%`}
                    trend={metrics.errorRate > 0.5 ? "↗" : "→"}
                    color={metrics.errorRate > 1.0 ? "red" : "gray"}
                />
            </div>

            {/* Main Charts Area */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                {/* Reward Chart */}
                <div className="col-span-2 bg-slate-800 p-4 rounded-xl border border-slate-700">
                    <div className="text-sm text-slate-400 mb-2 font-bold">Reward History (1h)</div>
                    <div className="h-40 flex items-end gap-1">
                        {Array.from({ length: 30 }).map((_, i) => {
                            const h = 20 + Math.random() * 60 + (i > 20 ? 10 : 0); // Simulated drift
                            return (
                                <motion.div
                                    key={i}
                                    className="flex-1 bg-green-500/50 hover:bg-green-400"
                                    initial={{ height: 0 }}
                                    animate={{ height: `${h}%` }}
                                    transition={{ duration: 0.5, delay: i * 0.02 }}
                                />
                            );
                        })}
                    </div>
                </div>

                {/* Alert Panel */}
                <div className="col-span-1 bg-slate-800 p-4 rounded-xl border border-slate-700">
                    <div className="text-sm text-slate-400 mb-4 font-bold">System Status</div>
                    <div className="space-y-3">
                        {alerts.map(alert => (
                            <div key={alert.metric} className={`p-3 rounded border ${alert.status === "critical" ? "bg-red-900/20 border-red-500/50" :
                                    alert.status === "warning" ? "bg-yellow-900/20 border-yellow-500/50" :
                                        "bg-slate-700/50 border-slate-600"
                                }`}>
                                <div className="flex justify-between items-center mb-1">
                                    <span className="text-xs font-bold">{alert.metric}</span>
                                    <span className={`text-xs px-1.5 rounded ${alert.status === "critical" ? "bg-red-500 text-white" :
                                            alert.status === "warning" ? "bg-yellow-500 text-black" :
                                                "bg-green-500/20 text-green-400"
                                        }`}>
                                        {alert.status.toUpperCase()}
                                    </span>
                                </div>
                                <div className="text-xs text-slate-400">
                                    Current: <span className="text-slate-200">{alert.current.toFixed(1)}</span>
                                    <span className="mx-1">/</span>
                                    Limit: {alert.threshold}
                                </div>
                            </div>
                        ))}

                        <div className="p-3 rounded border bg-slate-700/50 border-slate-600">
                            <div className="flex justify-between items-center mb-1">
                                <span className="text-xs font-bold">Drift Check</span>
                                <span className="text-xs px-1.5 rounded bg-green-500/20 text-green-400">OK</span>
                            </div>
                            <div className="text-xs text-slate-400">KL Div: 0.02 (Low)</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Logs Preview */}
            <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 font-mono text-xs">
                <div className="text-sm text-slate-400 mb-2 font-bold">Recent Logs</div>
                <div className="space-y-1 text-slate-300 max-h-32 overflow-y-auto">
                    <LogLine time="14:24:01" level="INFO" msg="Model update completed (ver: 1.2.4)" />
                    <LogLine time="14:23:45" level="WARN" msg="Latency spike detected (92ms) in region us-east" color="text-yellow-400" />
                    <LogLine time="14:23:44" level="INFO" msg="Batch processed: 64 samples" />
                    <LogLine time="14:23:12" level="INFO" msg="Health check passed" color="text-green-400" />
                    <LogLine time="14:22:55" level="INFO" msg="Inference worker scaled up to 4 replicas" />
                </div>
            </div>
        </div>
    );
}

function MetricCard({ label, value, trend, color }: { label: string, value: string, trend: string, color: string }) {
    const colorClass = {
        blue: "text-blue-400",
        green: "text-green-400",
        red: "text-red-400",
        purple: "text-purple-400",
        gray: "text-slate-400"
    }[color] || "text-slate-100";

    return (
        <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
            <div className="text-xs text-slate-400 mb-1">{label}</div>
            <div className={`text-2xl font-bold ${colorClass.replace("400", "300")}`}>{value}</div>
            <div className={`text-xs ${colorClass} mt-1`}>
                Trend: {trend}
            </div>
        </div>
    );
}

function LogLine({ time, level, msg, color }: { time: string, level: string, msg: string, color?: string }) {
    return (
        <div className="flex gap-2">
            <span className="text-slate-500">[{time}]</span>
            <span className={color || "text-slate-300"}>
                [{level}] {msg}
            </span>
        </div>
    );
}
