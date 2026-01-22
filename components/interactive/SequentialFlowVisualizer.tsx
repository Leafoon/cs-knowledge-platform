"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type LayerType = 'Linear' | 'Conv2d' | 'Flatten' | 'ReLU' | 'MaxPool2d';

interface Layer {
    id: string;
    type: LayerType;
    params: Record<string, number>;
}

interface TensorShape {
    dims: number[];
    valid: boolean;
    error?: string;
}

const SequentialFlowVisualizer = () => {
    const [inputShape, setInputShape] = useState<number[]>([1, 1, 28, 28]); // B, C, H, W (MNIST style)
    const [layers, setLayers] = useState<Layer[]>([
        { id: '1', type: 'Conv2d', params: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 1 } },
        { id: '2', type: 'ReLU', params: {} },
        { id: '3', type: 'MaxPool2d', params: { kernel_size: 2, stride: 2 } },
        { id: '4', type: 'Flatten', params: {} },
        { id: '5', type: 'Linear', params: { in_features: 0, out_features: 128 } }, // in_features auto-calc usually, but here manual for simplicity or auto logic?
    ]);

    // Shapes calculation logic
    const calculateShapes = () => {
        let currentShape = [...inputShape];
        const shapes: TensorShape[] = [{ dims: [...currentShape], valid: true }];

        for (const layer of layers) {
            if (!shapes[shapes.length - 1].valid) {
                shapes.push({ dims: [], valid: false, error: "上一层形状无效" });
                continue;
            }

            try {
                let nextShape = [...currentShape];

                switch (layer.type) {
                    case 'Conv2d': {
                        // (N, Cin, Hin, Win) -> (N, Cout, Hout, Wout)
                        const [N, C, H, W] = currentShape;
                        const { out_channels, kernel_size, stride, padding } = layer.params;

                        if (C !== layer.params.in_channels) throw new Error(`通道数不匹配: 输入${C} vs 层${layer.params.in_channels}`);

                        const H_out = Math.floor((H + 2 * padding - kernel_size) / stride + 1);
                        const W_out = Math.floor((W + 2 * padding - kernel_size) / stride + 1);
                        nextShape = [N, out_channels, H_out, W_out];
                        break;
                    }
                    case 'MaxPool2d': {
                        const [N, C, H, W] = currentShape;
                        const { kernel_size, stride } = layer.params;
                        const H_out = Math.floor((H - kernel_size) / stride + 1);
                        const W_out = Math.floor((W - kernel_size) / stride + 1);
                        nextShape = [N, C, H_out, W_out];
                        break;
                    }
                    case 'Flatten': {
                        // (N, ...) -> (N, Product(...))
                        const N = currentShape[0];
                        const features = currentShape.slice(1).reduce((a, b) => a * b, 1);
                        nextShape = [N, features];
                        break;
                    }
                    case 'Linear': {
                        // (N, in_features) -> (N, out_features)
                        const [N, in_f] = currentShape;
                        if (in_f !== layer.params.in_features && layer.params.in_features !== 0) {
                            // Auto-fix for demo purposes if 0, else error
                            if (layer.params.in_features === 0) {
                                // This is dynamic mode
                                // Don't update state during render, just calculate
                            } else {
                                throw new Error(`输入特征数不匹配: ${in_f} vs ${layer.params.in_features}`);
                            }
                        }
                        nextShape = [N, layer.params.out_features];
                        break;
                    }
                    case 'ReLU':
                        // Shape doesn't change
                        break;
                }
                currentShape = nextShape;
                shapes.push({ dims: nextShape, valid: true });
            } catch (e: any) {
                shapes.push({ dims: [], valid: false, error: e.message });
            }
        }
        return shapes;
    };

    const calculatedShapes = calculateShapes();

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">nn.Sequential 形状推导演示</h3>
            <div className="text-sm text-text-secondary mb-6">
                输入 Tensor 形状 (N, C, H, W):
                <span className="font-mono bg-bg-surface px-2 py-1 rounded ml-2 border border-border-subtle">
                    [{inputShape.join(', ')}]
                </span>
            </div>

            <div className="space-y-4">
                {/* Input Node */}
                <div className="flex items-center gap-4">
                    <div className="w-24 text-right text-xs text-text-tertiary">Input</div>
                    <div className="px-4 py-2 bg-blue-500/10 border border-blue-500/30 text-blue-600 rounded font-mono text-sm">
                        Tensor shape: [{inputShape.join(', ')}]
                    </div>
                </div>

                {/* Arrow */}
                <div className="flex justify-start ml-28">
                    <svg className="w-4 h-6 text-text-quaternary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
                </div>

                <AnimatePresence>
                    {layers.map((layer, idx) => {
                        const outShape = calculatedShapes[idx + 1];
                        return (
                            <motion.div key={layer.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="group relative">
                                <div className="flex items-center gap-4">
                                    <div className="w-24 text-right text-sm font-medium text-text-secondary">{layer.type}</div>

                                    <div className={`flex-1 p-3 rounded-lg border flex items-center justify-between
                        ${outShape.valid ? 'bg-bg-surface border-border-subtle' : 'bg-red-50 border-red-200'}
                     `}>
                                        <div className="text-xs text-text-tertiary font-mono">
                                            {Object.entries(layer.params).map(([k, v]) => (
                                                <span key={k} className="mr-3">{k}={v}</span>
                                            ))}
                                        </div>

                                        <div className={`font-mono text-sm font-bold ${outShape.valid ? 'text-green-600' : 'text-red-600'}`}>
                                            {outShape.valid ? `[${outShape.dims.join(', ')}]` : 'ERROR'}
                                        </div>
                                    </div>
                                </div>

                                {!outShape.valid && (
                                    <div className="ml-28 mt-1 text-xs text-red-500">{outShape.error}</div>
                                )}

                                {/* Connector Arrow */}
                                {idx < layers.length - 1 && (
                                    <div className="flex justify-start ml-28 mt-2 mb-2">
                                        <svg className="w-4 h-4 text-text-quaternary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
                                    </div>
                                )}
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>

            <div className="mt-8 p-4 bg-orange-50/50 rounded text-xs text-orange-800 border border-orange-100">
                <strong>TIP:</strong> 在设计 CNN 时，最常遇到的错误就是全连接层（Linear）前的 <code>in_features</code> 计算错误。使用 <code>torch.flatten</code> 后，特征维度变成了 $C \times H \times W$。
            </div>
        </div>
    );
};

export default SequentialFlowVisualizer;
