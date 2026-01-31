"use client";

import { motion } from "framer-motion";

export function SMACEnvironment() {
    const gridSize = 8;
    const units = [
        { id: 1, type: "Marine", pos: [2, 2], team: "ally", hp: 100, color: "bg-blue-500" },
        { id: 2, type: "Marine", pos: [3, 2], team: "ally", hp: 80, color: "bg-blue-500" },
        { id: 3, type: "Marine", pos: [2, 3], team: "ally", hp: 60, color: "bg-blue-500" },
        { id: 4, type: "Marine", pos: [5, 5], team: "enemy", hp: 100, color: "bg-red-500" },
        { id: 5, type: "Marine", pos: [6, 5], team: "enemy", hp: 100, color: "bg-red-500" },
        { id: 6, type: "Marine", pos: [5, 6], team: "enemy", hp: 100, color: "bg-red-500" },
    ];

    const actions = ["Move North", "Move South", "Move East", "Move West", "Attack", "Stop", "Hold"];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-800 to-gray-900 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-white mb-2">
                    SMAC Environment (StarCraft II)
                </h3>
                <p className="text-sm text-gray-300">
                    3m scenario: 3 Marines vs 3 Marines micromanagement
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Battlefield Grid */}
                <div className="lg:col-span-2 bg-gray-900 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-white mb-4">Battlefield</h4>
                    <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))` }}>
                        {Array.from({ length: gridSize * gridSize }).map((_, i) => {
                            const row = Math.floor(i / gridSize);
                            const col = i % gridSize;

                            // Check if a unit occupies this cell
                            const unit = units.find(u => u.pos[0] === row && u.pos[1] === col);

                            return (
                                <div
                                    key={i}
                                    className={`aspect-square border border-gray-700 rounded flex items-center justify-center text-xs font-bold ${unit ? unit.color : "bg-gray-800"
                                        }`}
                                >
                                    {unit && (
                                        <motion.div
                                            className="text-white"
                                            animate={{ scale: [1, 1.1, 1] }}
                                            transition={{ duration: 1, repeat: Infinity }}
                                        >
                                            {unit.team === "ally" ? "🔵" : "🔴"}
                                        </motion.div>
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    <div className="flex gap-4 justify-center mt-4 text-white text-sm">
                        <span className="flex items-center gap-1">
                            <span className="w-4 h-4 bg-blue-500 rounded"></span> Ally Units
                        </span>
                        <span className="flex items-center gap-1">
                            <span className="w-4 h-4 bg-red-500 rounded"></span> Enemy Units
                        </span>
                    </div>
                </div>

                {/* Unit Stats & Actions */}
                <div className="space-y-4">
                    <div className="bg-gray-800 rounded-xl p-4 shadow-lg">
                        <h4 className="text-lg font-bold text-white mb-3">Ally Units (Controllable)</h4>
                        <div className="space-y-2">
                            {units.filter(u => u.team === "ally").map(unit => (
                                <div key={unit.id} className="bg-blue-900/40 p-3 rounded border border-blue-700">
                                    <div className="flex justify-between text-white text-sm mb-1">
                                        <span className="font-bold">{unit.type} #{unit.id}</span>
                                        <span className="text-xs">Pos: ({unit.pos[0]}, {unit.pos[1]})</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="text-xs text-gray-300">HP:</div>
                                        <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                                            <div
                                                className={`h-full ${unit.hp > 70 ? "bg-green-500" :
                                                        unit.hp > 30 ? "bg-yellow-500" : "bg-red-500"
                                                    }`}
                                                style={{ width: `${unit.hp}%` }}
                                            />
                                        </div>
                                        <div className="text-xs text-white font-mono">{unit.hp}%</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-gray-800 rounded-xl p-4 shadow-lg">
                        <h4 className="text-lg font-bold text-white mb-3">Enemy Units</h4>
                        <div className="space-y-2">
                            {units.filter(u => u.team === "enemy").map(unit => (
                                <div key={unit.id} className="bg-red-900/40 p-3 rounded border border-red-700">
                                    <div className="flex justify-between text-white text-sm mb-1">
                                        <span className="font-bold">{unit.type} #{unit.id}</span>
                                        <span className="text-xs">HP: {unit.hp}%</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Action Space */}
            <div className="mt-6 bg-gray-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold text-white mb-4">Available Actions (per unit)</h4>
                <div className="grid grid-cols-4 md:grid-cols-7 gap-2">
                    {actions.map((action, idx) => (
                        <div
                            key={idx}
                            className="bg-gray-700 hover:bg-gray-600 text-white text-xs p-2 rounded text-center cursor-pointer transition"
                        >
                            {action}
                        </div>
                    ))}
                </div>
            </div>

            <div className="mt-6 bg-blue-900/30 rounded-xl p-4 border border-blue-700">
                <h5 className="font-bold text-white text-sm mb-2">Micromanagement Challenge</h5>
                <div className="text-xs text-gray-300 space-y-1">
                    <div>• <strong>Partial Observability</strong>: Each unit sees only nearby enemies</div>
                    <div>• <strong>Coordination</strong>: Focus fire on single target for efficiency</div>
                    <div>• <strong>Positioning</strong>: Maintain formation, kite enemies</div>
                    <div>• <strong>Reward</strong>: +damage dealt +10 per enemy kill -1 per ally death</div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-gray-400">
                💡 SMAC requires precise coordination for efficient combat micromanagement
            </div>
        </div>
    );
}
