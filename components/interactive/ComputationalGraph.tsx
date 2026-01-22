"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Node {
    id: string;
    label: string;
    value: number;
    grad: number;
    type: 'leaf' | 'op' | 'output';
    x: number;
    y: number;
    inputs?: string[];
    op?: '+' | '*' | 'sin' | 'pow';
}

const ComputationalGraph = () => {
    const [step, setStep] = useState(0); // 0: input, 1: forward, 2: backward start, 3...: backward steps

    // Initial state of our simple graph: y = (x + w) * 2
    // Let x=3, w=1 -> a=(3+1)=4 -> y=4*2=8
    // Backprop: dy/dy=1 -> dy/da=2 -> dy/dx=2, dy/dw=2
    const nodesInitial: Record<string, Node> = {
        'x': { id: 'x', label: 'x', value: 3, grad: 0, type: 'leaf', x: 50, y: 50 },
        'w': { id: 'w', label: 'w', value: 1, grad: 0, type: 'leaf', x: 50, y: 150 },
        'add': { id: 'add', label: '+', value: 0, grad: 0, type: 'op', x: 150, y: 100, inputs: ['x', 'w'], op: '+' },
        'mul': { id: 'mul', label: '*', value: 0, grad: 0, type: 'op', x: 250, y: 100, inputs: ['add', 'scalar'], op: '*' },
        'scalar': { id: 'scalar', label: '2', value: 2, grad: 0, type: 'leaf', x: 250, y: 40 },
        'y': { id: 'y', label: 'y', value: 0, grad: 0, type: 'output', x: 350, y: 100, inputs: ['mul'] }
    };

    const [nodes, setNodes] = useState(nodesInitial);

    // Animation Steps
    useEffect(() => {
        let newNodes = { ...nodesInitial };

        // Step 1: Forward Pass Calculation
        if (step >= 1) {
            newNodes['add'].value = newNodes['x'].value + newNodes['w'].value; // 3+1=4
            newNodes['mul'].value = newNodes['add'].value * newNodes['scalar'].value; // 4*2=8
            newNodes['y'].value = newNodes['mul'].value;
        }

        // Step 2: Backward Pass (Gradient Initialization)
        if (step >= 2) {
            newNodes['y'].grad = 1.0;
            newNodes['mul'].grad = 1.0; // Identity for output
        }

        // Step 3: Backward to 'add' source
        if (step >= 3) {
            // dy/d(add) = dy/d(mul) * d(mul)/d(add) = 1 * 2 = 2
            newNodes['add'].grad = newNodes['mul'].grad * newNodes['scalar'].value;
            // dy/d(scalar) = dy/d(mul) * d(mul)/d(scalar) = 1 * 4 = 4
            newNodes['scalar'].grad = newNodes['mul'].grad * newNodes['add'].value;
        }

        // Step 4: Backward to leaves
        if (step >= 4) {
            // dy/dx = dy/d(add) * d(add)/dx = 2 * 1 = 2
            newNodes['x'].grad = newNodes['add'].grad * 1;
            newNodes['w'].grad = newNodes['add'].grad * 1;
        }

        setNodes(newNodes);
    }, [step]);

    // Edges for rendering
    const edges = [
        { from: 'x', to: 'add' },
        { from: 'w', to: 'add' },
        { from: 'scalar', to: 'mul' },
        { from: 'add', to: 'mul' },
        { from: 'mul', to: 'y' }
    ];

    return (
        <div className="w-full max-w-2xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">Autograd 可视化演示: $y = (x + w) \times 2$</h3>

            <div className="flex gap-4 mb-6">
                <button
                    onClick={() => setStep(Math.max(0, step - 1))}
                    className="px-4 py-2 text-sm font-medium rounded-lg bg-bg-surface border border-border-subtle hover:bg-bg-elevated transition-colors"
                    disabled={step === 0}
                >
                    上一步
                </button>
                <button
                    onClick={() => setStep(Math.min(4, step + 1))}
                    className="px-4 py-2 text-sm font-medium rounded-lg bg-accent-primary text-white hover:bg-accent-primary/90 transition-colors"
                    disabled={step === 4}
                >
                    {step === 4 ? '演示完成' : '下一步 (Forward/Backward)'}
                </button>
                <div className="flex items-center text-sm text-text-secondary ml-auto">
                    当前阶段:
                    <span className="ml-2 font-mono text-accent-primary">
                        {step === 0 && "1. 初始化输入 (x=3, w=1)"}
                        {step === 1 && "2. 前向传播 (Forward Pass)"}
                        {step === 2 && "3. 反向传播开始 (y.backward())"}
                        {step === 3 && "4. 梯度反传至中间层"}
                        {step === 4 && "5. 梯度到达叶子节点"}
                    </span>
                </div>
            </div>

            <div className="relative h-[250px] w-full bg-grid-slate-100/50 rounded-lg overflow-hidden border border-border-subtle/50">
                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                    {edges.map((edge, i) => {
                        const start = nodes[edge.from];
                        const end = nodes[edge.to];
                        // Simple straight lines for now
                        return (
                            <motion.line
                                key={i}
                                x1={start.x} y1={start.y}
                                x2={end.x} y2={end.y}
                                stroke="currentColor"
                                strokeWidth="2"
                                className={`text-border-active transition-colors duration-500 ${step >= 2 ? 'text-accent-primary/60' : ''}`}
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                            />
                        );
                    })}
                </svg>

                {Object.values(nodes).map((node) => (
                    <motion.div
                        key={node.id}
                        className={`absolute flex flex-col items-center justify-center w-16 h-16 -ml-8 -mt-8 rounded-full border-2 shadow-sm text-sm font-bold z-10 transition-colors duration-500
              ${node.type === 'leaf' ? 'bg-blue-50/80 border-blue-200 text-blue-700' : ''}
              ${node.type === 'op' ? 'bg-orange-50/80 border-orange-200 text-orange-700' : ''}
              ${node.type === 'output' ? 'bg-green-50/80 border-green-200 text-green-700' : ''}
            `}
                        style={{ left: node.x, top: node.y }}
                        layout
                    >
                        <div>{node.label}</div>

                        {/* Value Display */}
                        {step >= 1 && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.5 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="text-[10px] font-mono text-text-secondary"
                            >
                                v:{node.value.toFixed(0)}
                            </motion.div>
                        )}

                        {/* Gradient Display (The Badge) */}
                        <AnimatePresence>
                            {step >= 2 && (node.type === 'output' || step >= 3) && (node.grad !== 0 || node.type === 'output') && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 22 }}
                                    exit={{ opacity: 0 }}
                                    className="absolute bg-accent-primary text-white text-[10px] px-1.5 py-0.5 rounded shadow-sm whitespace-nowrap"
                                >
                                    grad:{node.grad.toFixed(1)}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                ))}
            </div>

            <div className="mt-4 text-xs text-text-tertiary">
                <p>* 绿色: 输出节点 (Root) | 橙色: 算子节点 (Op) | 蓝色: 叶子节点 (Leaf, requires_grad=True)</p>
            </div>
        </div>
    );
};

export default ComputationalGraph;
