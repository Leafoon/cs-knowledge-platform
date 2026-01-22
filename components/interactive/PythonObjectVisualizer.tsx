"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type PyObject = {
    id: number;
    type: string;
    value: string;
    refCount: number;
    color: string;
    isMutable: boolean;
};

type Variable = {
    name: string;
    targetId: number | null;
};

const INITIAL_OBJECTS: PyObject[] = [
    { id: 4301, type: "int", value: "42", refCount: 0, color: "bg-blue-500", isMutable: false },
    { id: 4302, type: "list", value: "[1, 2]", refCount: 0, color: "bg-green-500", isMutable: true },
    { id: 4303, type: "str", value: "'Python'", refCount: 0, color: "bg-yellow-500", isMutable: false },
];

export function PythonObjectVisualizer() {
    const [objects, setObjects] = useState<PyObject[]>(INITIAL_OBJECTS);
    const [variables, setVariables] = useState<Variable[]>([
        { name: "a", targetId: null },
        { name: "b", targetId: null },
    ]);
    const [consoleOutput, setConsoleOutput] = useState<string[]>([]);

    const log = (msg: string) => setConsoleOutput(prev => [...prev.slice(-4), msg]);

    const assignVariable = (varName: string, objId: number) => {
        setVariables(prev => prev.map(v =>
            v.name === varName ? { ...v, targetId: objId } : v
        ));

        // Update ref counts
        setObjects(prev => prev.map(obj => {
            // Simple recalculation based on new state would be complex in this simplified view 
            // without tracking previous state accurately in this specific function scope.
            // So we'll calculate ref counts derived from variables state in a separate effect or just update manually here.
            // For visualization, let's just count how many vars point to it in the NEXT state.
            let count = 0;
            // Iterate vars to count, including the change we just made
            const currentVars = variables.map(v => v.name === varName ? { ...v, targetId: objId } : v);
            currentVars.forEach(v => {
                if (v.targetId === obj.id) count++;
            });
            return { ...obj, refCount: count };
        }));

        const targetObj = objects.find(o => o.id === objId);
        log(`>>> ${varName} = ${targetObj?.value}`);
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur-sm rounded-xl border border-border-subtle shadow-lg my-8 font-mono">
            <h3 className="text-xl font-bold text-center mb-6 text-text-primary font-sans">
                Python Object Model: Variables are Labels
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                {/* Left: Stack / Names */}
                <div className="space-y-6">
                    <h4 className="text-sm font-semibold text-text-secondary uppercase tracking-wider border-b border-border-subtle pb-2">
                        Frames / Variables (Stack)
                    </h4>
                    <div className="space-y-4">
                        {variables.map((v) => (
                            <div key={v.name} className="flex items-center justify-between p-3 bg-bg-base border border-border-subtle rounded-lg">
                                <span className="font-bold text-lg text-accent-primary">{v.name}</span>
                                <div className="flex gap-2">
                                    {objects.map((obj) => (
                                        <button
                                            key={obj.id}
                                            onClick={() => assignVariable(v.name, obj.id)}
                                            className={`px-3 py-1 text-xs rounded-full transition-all ${v.targetId === obj.id
                                                    ? `${obj.color} text-white shadow-md scale-105`
                                                    : "bg-bg-elevated text-text-tertiary hover:bg-bg-elevated/80"
                                                }`}
                                        >
                                            {v.targetId === obj.id ? "●" : "○"} {obj.value}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-8 p-4 bg-black/80 rounded-lg text-xs text-green-400 font-mono min-h-[100px]">
                        <div className="text-gray-500 mb-2 border-b border-gray-700 pb-1">Interactive Console</div>
                        {consoleOutput.map((line, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                            >
                                {line}
                            </motion.div>
                        ))}
                        <motion.div
                            animate={{ opacity: [0, 1, 0] }}
                            transition={{ repeat: Infinity, duration: 0.8 }}
                            className="inline-block w-2 h-4 bg-green-400 align-middle ml-1"
                        />
                    </div>
                </div>

                {/* Right: Heap / Objects */}
                <div className="space-y-6">
                    <h4 className="text-sm font-semibold text-text-secondary uppercase tracking-wider border-b border-border-subtle pb-2">
                        Object Heap (Memory)
                    </h4>
                    <div className="space-y-4 relative">
                        {objects.map((obj) => {
                            const isReferenced = obj.refCount > 0;
                            return (
                                <motion.div
                                    key={obj.id}
                                    layout
                                    className={`relative p-4 rounded-xl border-2 transition-colors duration-300 ${isReferenced ? "border-border-strong bg-bg-base shadow-sm" : "border-dashed border-border-subtle bg-transparent opacity-60"
                                        }`}
                                >
                                    {/* Object Header */}
                                    <div className="flex justify-between items-start mb-2">
                                        <div className="text-xs text-text-tertiary">
                                            <div>ID: <span className="text-text-secondary">0x{obj.id}</span></div>
                                            <div>Type: <span className="text-accent-secondary">{obj.type}</span></div>
                                        </div>
                                        <div className={`px-2 py-0.5 rounded text-xs font-bold ${obj.refCount > 0 ? "bg-green-500/20 text-green-500" : "bg-red-500/20 text-red-500"
                                            }`}>
                                            Ref Cnt: {obj.refCount}
                                        </div>
                                    </div>

                                    {/* Object Value */}
                                    <div className="text-center py-2 relative">
                                        <motion.div
                                            className={`inline-block px-4 py-2 rounded-lg text-white font-bold text-lg ${obj.color} shadow-lg`}
                                            animate={isReferenced ? { scale: 1 } : { scale: 0.9, filter: "grayscale(100%)" }}
                                        >
                                            {obj.value}
                                        </motion.div>
                                    </div>

                                    {/* Connections lines would replace this textual representation in a Canvas impl, 
                                        but here we use proximity and color matching for simplicity */}
                                </motion.div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
}
