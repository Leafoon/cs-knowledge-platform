"use client";

import React, { useState, useEffect, useRef } from 'react';

const OptimizerPathVisualizer = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [running, setRunning] = useState(false);

    // Initial Params
    const startX = -3.5;
    const startY = 2.5;

    // Optimizers State
    // We compare: SGD, Momentum, Adam
    const [paths, setPaths] = useState<any>({
        sgd: [],
        momentum: [],
        adam: []
    });

    // Hyperparams
    const lr = 0.1;

    // Loss Function: Beale Function-like but simpler valley
    // f(x, y) = x^2 + 10y^2 (Ellipsoid, narrow valley along X axis if we rotate? No, lets stick to simple)
    // Let's use: f(x, y) = 0.5*x^2 + 5*y^2 (High curvature in Y, low in X)
    // Local minimum at (0, 0)
    const f = (x: number, y: number) => 0.1 * x * x + 2 * y * y;
    const df = (x: number, y: number) => [0.2 * x, 4 * y]; // Gradients

    // Animation Loop
    useEffect(() => {
        if (!running) return;

        let iter = 0;
        const maxIter = 100;

        let sgd = { x: startX, y: startY };
        let mom = { x: startX, y: startY, vx: 0, vy: 0 };
        let adam = { x: startX, y: startY, m: [0, 0], v: [0, 0], t: 0 };

        const tempPaths = { sgd: [{ x: startX, y: startY }], momentum: [{ x: startX, y: startY }], adam: [{ x: startX, y: startY }] };

        const interval = setInterval(() => {
            if (iter >= maxIter) {
                setRunning(false);
                clearInterval(interval);
                return;
            }

            // 1. SGD Update
            const g_sgd = df(sgd.x, sgd.y);
            sgd.x -= lr * g_sgd[0];
            sgd.y -= lr * g_sgd[1];
            tempPaths.sgd.push({ ...sgd });

            // 2. Momentum Update (beta=0.9)
            const beta = 0.9;
            const g_mom = df(mom.x, mom.y);
            mom.vx = beta * mom.vx + lr * g_mom[0];
            mom.vy = beta * mom.vy + lr * g_mom[1];
            mom.x -= mom.vx;
            mom.y -= mom.vy;
            tempPaths.momentum.push({ ...mom });

            // 3. Adam Update (beta1=0.9, beta2=0.999)
            adam.t += 1;
            const g_adam = df(adam.x, adam.y);
            // m = beta1*m + (1-beta1)*g
            adam.m[0] = 0.9 * adam.m[0] + 0.1 * g_adam[0];
            adam.m[1] = 0.9 * adam.m[1] + 0.1 * g_adam[1];
            // v = beta2*v + (1-beta2)*g^2
            adam.v[0] = 0.999 * adam.v[0] + 0.001 * (g_adam[0] ** 2);
            adam.v[1] = 0.999 * adam.v[1] + 0.001 * (g_adam[1] ** 2);
            // Bias correction
            const m_hat_x = adam.m[0] / (1 - 0.9 ** adam.t);
            const m_hat_y = adam.m[1] / (1 - 0.9 ** adam.t);
            const v_hat_x = adam.v[0] / (1 - 0.999 ** adam.t);
            const v_hat_y = adam.v[1] / (1 - 0.999 ** adam.t);

            // Update
            const epsilon = 1e-8;
            adam.x -= lr * m_hat_x / (Math.sqrt(v_hat_x) + epsilon);
            adam.y -= lr * m_hat_y / (Math.sqrt(v_hat_y) + epsilon);
            tempPaths.adam.push({ ...adam });

            setPaths({ ...tempPaths }); // Trigger render
            iter++;
        }, 50);

        return () => clearInterval(interval);
    }, [running]);

    // Canvas Rendering
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Coordinate Mapping
        // World: x[-4, 4], y[-3, 3] -> Canvas 400x300
        const mapX = (x: number) => (x + 4) / 8 * canvas.width;
        const mapY = (y: number) => (-y + 3) / 6 * canvas.height;

        // Draw Contours
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        for (let l = 0.5; l < 20; l += 1.5) {
            ctx.beginPath();
            // f(x,y) = c => 0.1x^2 + 2y^2 = c
            // Ellipse
            for (let angle = 0; angle <= Math.PI * 2; angle += 0.1) {
                // Not exact contour drawing, but concentric ellipses hint
                // x^2/(10c) + y^2/(0.5c) = 1
                // a = sqrt(10c), b = sqrt(0.5c)
                const a = Math.sqrt(l / 0.1);
                const b = Math.sqrt(l / 2);
                const px = mapX(a * Math.cos(angle));
                const py = mapY(b * Math.sin(angle));
                if (angle === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        // Helper to draw path
        const drawPath = (pts: any[], color: string) => {
            if (pts.length < 2) return;
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.moveTo(mapX(pts[0].x), mapY(pts[0].y));
            for (let i = 1; i < pts.length; i++) {
                ctx.lineTo(mapX(pts[i].x), mapY(pts[i].y));
            }
            ctx.stroke();
            // Head
            const last = pts[pts.length - 1];
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(mapX(last.x), mapY(last.y), 4, 0, Math.PI * 2);
            ctx.fill();
        };

        drawPath(paths.sgd, '#ef4444'); // Red
        drawPath(paths.momentum, '#f59e0b'); // Orange
        drawPath(paths.adam, '#22c55e'); // Green

    }, [paths]);

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">优化器寻路对比 (Optimizer Trajectory)</h3>

            <div className="flex flex-col md:flex-row gap-8">
                <div className="relative border border-border-subtle rounded-xl bg-white dark:bg-slate-900 overflow-hidden shadow-inner">
                    <canvas
                        ref={canvasRef}
                        width={400}
                        height={300}
                        className="w-[400px] h-[300px]"
                    />
                    <div className="absolute top-2 left-2 text-xs text-text-tertiary">Loss Landscape (Ellipsoid)</div>
                </div>

                <div className="flex-1 space-y-6">
                    <div className="space-y-3">
                        <div className="flex items-center gap-3">
                            <div className="w-4 h-4 rounded-full bg-red-500"></div>
                            <div className="text-sm font-bold">SGD</div>
                            <div className="text-xs text-text-tertiary">Only Gradients. Oscillation.</div>
                        </div>
                        <div className="flex items-center gap-3">
                            <div className="w-4 h-4 rounded-full bg-orange-500"></div>
                            <div className="text-sm font-bold">SGD + Momentum</div>
                            <div className="text-xs text-text-tertiary">Accumulates velocity. Faster.</div>
                        </div>
                        <div className="flex items-center gap-3">
                            <div className="w-4 h-4 rounded-full bg-green-500"></div>
                            <div className="text-sm font-bold">Adam</div>
                            <div className="text-xs text-text-tertiary">Adaptive Learning Rate. Direct path.</div>
                        </div>
                    </div>

                    <button
                        onClick={() => {
                            setRunning(false);
                            setPaths({ sgd: [], momentum: [], adam: [] });
                            setTimeout(() => setRunning(true), 100);
                        }}
                        className="w-full py-3 bg-accent-primary text-white rounded-xl font-bold hover:shadow-lg transition-all active:scale-95"
                    >
                        Re-run Simulation
                    </button>

                    <div className="p-3 bg-bg-surface rounded border border-border-subtle text-xs text-text-secondary leading-relaxed">
                        <p className="mb-2"><strong>观察重点：</strong></p>
                        <ul className="list-disc pl-4 space-y-1">
                            <li>SGD 沿着梯度最陡峭的方向走（垂直于等高线），在狭长山谷中会剧烈震荡（所谓的 "Zig-Zag" 现象）。</li>
                            <li>Momentum 累积了惯性，能冲过震荡，但可能会冲过头（Overshoot）。</li>
                            <li>Adam 自适应调整每个维度的步长。在平坦方向（X轴）步子大，在陡峭方向（Y轴）步子小，因此能走出一条近乎直线的完美路径。</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default OptimizerPathVisualizer;
