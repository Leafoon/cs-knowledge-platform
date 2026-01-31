"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function ZeroSumGameVisualization() {
    const [payoffMatrix, setPayoffMatrix] = useState([
        [3, 0, 5],
        [2, 1, 3],
        [0, 4, 2]
    ]);

    const [selectedRow, setSelectedRow] = useState<number | null>(null);
    const [selectedCol, setSelectedCol] = useState<number | null>(null);

    // Compute minimax
    const rowMins = payoffMatrix.map(row => Math.min(...row));
    const maximin = Math.max(...rowMins);
    const maximinRow = rowMins.indexOf(maximin);

    const colMaxs = payoffMatrix[0].map((_, colIdx) =>
        Math.max(...payoffMatrix.map(row => row[colIdx]))
    );
    const minimax = Math.min(...colMaxs);
    const minimaxCol = colMaxs.indexOf(minimax);

    const gameValue = (maximin === minimax) ? maximin : null;

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-red-50 to-orange-50 dark:from-slate-900 dark:to-red-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Zero-Sum Game: Minimax/Maximin
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Click cells to explore strategies
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Payoff Matrix (Player 1)</h4>

                <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                        <thead>
                            <tr>
                                <th className="p-2 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800"></th>
                                {[0, 1, 2].map(col => (
                                    <th
                                        key={col}
                                        className={`p-2 border border-gray-300 dark:border-gray-700 font-semibold ${col === minimaxCol ? "bg-blue-200 dark:bg-blue-900" : "bg-gray-100 dark:bg-gray-800"
                                            }`}
                                    >
                                        Col {col + 1}<br />
                                        <span className="text-xs text-slate-500">(P2)</span>
                                    </th>
                                ))}
                                <th className="p-2 border border-gray-300 dark:border-gray-700 bg-green-100 dark:bg-green-900 text-xs">
                                    Row Min
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {payoffMatrix.map((row, rowIdx) => (
                                <tr key={rowIdx}>
                                    <td className={`p-2 border border-gray-300 dark:border-gray-700 font-semibold ${rowIdx === maximinRow ? "bg-red-200 dark:bg-red-900" : "bg-gray-100 dark:bg-gray-800"
                                        }`}>
                                        Row {rowIdx + 1}<br />
                                        <span className="text-xs text-slate-500">(P1)</span>
                                    </td>
                                    {row.map((value, colIdx) => (
                                        <td
                                            key={colIdx}
                                            onClick={() => {
                                                setSelectedRow(rowIdx);
                                                setSelectedCol(colIdx);
                                            }}
                                            className={`p-4 border border-gray-300 dark:border-gray-700 text-center cursor-pointer transition ${selectedRow === rowIdx && selectedCol === colIdx
                                                    ? "bg-yellow-200 dark:bg-yellow-700 font-bold text-lg scale-110"
                                                    : rowIdx === maximinRow && colIdx === minimaxCol && gameValue
                                                        ? "bg-green-300 dark:bg-green-700 font-bold"
                                                        : "bg-white dark:bg-slate-700 hover:bg-gray-100 dark:hover:bg-gray-600"
                                                }`}
                                        >
                                            {value}
                                        </td>
                                    ))}
                                    <td className="p-2 border border-gray-300 dark:border-gray-700 text-center font-mono bg-green-50 dark:bg-green-900/30">
                                        {rowMins[rowIdx]}
                                    </td>
                                </tr>
                            ))}
                            <tr>
                                <td className="p-2 border border-gray-300 dark:border-gray-700 bg-blue-100 dark:bg-blue-900 text-xs font-semibold">
                                    Col Max
                                </td>
                                {colMaxs.map((max, colIdx) => (
                                    <td key={colIdx} className="p-2 border border-gray-300 dark:border-gray-700 text-center font-mono bg-blue-50 dark:bg-blue-900/30">
                                        {max}
                                    </td>
                                ))}
                                <td className="p-2 border border-gray-300 dark:border-gray-700"></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-300 dark:border-red-700">
                    <h5 className="font-bold text-red-700 dark:text-red-400 mb-2">Player 1 (Row): Maximin</h5>
                    <div className="text-sm space-y-1">
                        <div>Strategy: Row {maximinRow + 1}</div>
                        <div className="font-mono text-lg">Value = {maximin}</div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            Maximize the minimum payoff
                        </div>
                    </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-2 border-blue-300 dark:border-blue-700">
                    <h5 className="font-bold text-blue-700 dark:text-blue-400 mb-2">Player 2 (Col): Minimax</h5>
                    <div className="text-sm space-y-1">
                        <div>Strategy: Col {minimaxCol + 1}</div>
                        <div className="font-mono text-lg">Value = {minimax}</div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            Minimize the maximum payoff (for P1)
                        </div>
                    </div>
                </div>
            </div>

            {gameValue !== null ? (
                <motion.div
                    className="bg-green-100 dark:bg-green-900/30 p-4 rounded-lg border-2 border-green-500 text-center"
                    animate={{ scale: [1, 1.02, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                >
                    <div className="font-bold text-green-700 dark:text-green-400 text-lg mb-1">
                        ✓ Pure Nash Equilibrium Found!
                    </div>
                    <div className="text-sm">
                        Game Value: <span className="font-mono font-bold text-xl">{gameValue}</span>
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                        Saddle point at ({maximinRow + 1}, {minimaxCol + 1})
                    </div>
                </motion.div>
            ) : (
                <div className="bg-orange-100 dark:bg-orange-900/30 p-4 rounded-lg border-2 border-orange-500 text-center">
                    <div className="font-bold text-orange-700 dark:text-orange-400">
                        No Pure Nash Equilibrium
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                        Players need mixed strategies
                    </div>
                </div>
            )}

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 In zero-sum games, maximin = minimax at Nash equilibrium (von Neumann's theorem)
            </div>
        </div>
    );
}
