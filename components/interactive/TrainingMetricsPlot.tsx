"use client";

import React, { useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { TrendingDown, Award, Zap, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';

// Metric Data Interface
interface MetricData {
  epoch: number;
  trainLoss: number;
  evalLoss: number;
  accuracy: number;
  learningRate: number;
}

// Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white/90 dark:bg-slate-800/90 p-4 border border-slate-200 dark:border-slate-700 rounded-lg shadow-xl backdrop-blur-sm">
        <p className="font-bold text-slate-700 dark:text-slate-200 mb-2">Epoch {label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2 text-sm mb-1">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-slate-600 dark:text-slate-400 capitalize">
              {entry.name === 'trainLoss' ? 'Train Loss' :
                entry.name === 'evalLoss' ? 'Eval Loss' :
                  entry.name === 'learningRate' ? 'Learning Rate' : entry.name}:
            </span>
            <span className="font-mono font-semibold text-slate-900 dark:text-slate-100">
              {entry.value.toFixed(4)}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function TrainingMetricsPlot() {
  const [viewMode, setViewMode] = useState<'loss' | 'performance'>('loss');

  // Simulated robust training data
  const data: MetricData[] = [
    { epoch: 1, trainLoss: 2.30, evalLoss: 2.28, accuracy: 0.52, learningRate: 5e-5 },
    { epoch: 2, trainLoss: 1.85, evalLoss: 1.92, accuracy: 0.61, learningRate: 4.8e-5 },
    { epoch: 3, trainLoss: 1.45, evalLoss: 1.55, accuracy: 0.68, learningRate: 4.5e-5 },
    { epoch: 4, trainLoss: 1.15, evalLoss: 1.25, accuracy: 0.74, learningRate: 4.0e-5 },
    { epoch: 5, trainLoss: 0.88, evalLoss: 0.98, accuracy: 0.79, learningRate: 3.5e-5 },
    { epoch: 6, trainLoss: 0.65, evalLoss: 0.82, accuracy: 0.83, learningRate: 3.0e-5 },
    { epoch: 7, trainLoss: 0.48, evalLoss: 0.75, accuracy: 0.86, learningRate: 2.5e-5 },
    { epoch: 8, trainLoss: 0.35, evalLoss: 0.68, accuracy: 0.88, learningRate: 2.0e-5 },
    { epoch: 9, trainLoss: 0.28, evalLoss: 0.65, accuracy: 0.89, learningRate: 1.5e-5 },
    { epoch: 10, trainLoss: 0.22, evalLoss: 0.64, accuracy: 0.90, learningRate: 1.0e-5 }, // Best Eval Loss around here
    { epoch: 11, trainLoss: 0.18, evalLoss: 0.66, accuracy: 0.91, learningRate: 0.5e-5 }, // Overfitting starts
    { epoch: 12, trainLoss: 0.15, evalLoss: 0.69, accuracy: 0.91, learningRate: 0.0e-5 },
  ];

  const bestEpoch = data.reduce((prev, current) =>
    (prev.evalLoss < current.evalLoss) ? prev : current
  );

  return (
    <div className="my-8 p-6 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-500" />
            Training Metrics
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Real-time visualization of model convergence
          </p>
        </div>

        <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-lg">
          <button
            onClick={() => setViewMode('loss')}
            className={cn(
              "px-4 py-1.5 text-sm font-medium rounded-md transition-all",
              viewMode === 'loss'
                ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm"
                : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
            )}
          >
            Loss Curves
          </button>
          <button
            onClick={() => setViewMode('performance')}
            className={cn(
              "px-4 py-1.5 text-sm font-medium rounded-md transition-all",
              viewMode === 'performance'
                ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm"
                : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
            )}
          >
            Performance
          </button>
        </div>
      </div>

      {/* Chart Area */}
      <div className="h-[350px] w-full mb-8">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorTrainLoss" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorEvalLoss" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#94a3b8" opacity={0.1} />
            <XAxis
              dataKey="epoch"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#64748b', fontSize: 12 }}
              dy={10}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#64748b', fontSize: 12 }}
              dx={-10}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend iconType="circle" />

            {viewMode === 'loss' ? (
              <>
                <Area
                  type="monotone"
                  dataKey="trainLoss"
                  stroke="#6366f1"
                  strokeWidth={3}
                  fillOpacity={1}
                  fill="url(#colorTrainLoss)"
                  name="Train Loss"
                  animationDuration={1500}
                />
                <Area
                  type="monotone"
                  dataKey="evalLoss"
                  stroke="#f43f5e"
                  strokeWidth={3}
                  strokeDasharray="5 5"
                  fillOpacity={1}
                  fill="url(#colorEvalLoss)"
                  name="Eval Loss"
                  animationDuration={1500}
                />
              </>
            ) : (
              <>
                <Area
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#10b981"
                  strokeWidth={3}
                  fillOpacity={1}
                  fill="url(#colorAccuracy)"
                  name="Accuracy"
                  animationDuration={1500}
                />
              </>
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Key Metrics Dashboard */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-800 transition-hover hover:border-indigo-200 dark:hover:border-indigo-900/50">
          <div className="flex items-center gap-2 mb-2 text-slate-500 dark:text-slate-400">
            <Award className="w-4 h-4 text-emerald-500" />
            <span className="text-xs font-semibold uppercase tracking-wider">Best Epoch</span>
          </div>
          <div className="text-2xl font-black text-slate-800 dark:text-slate-100">
            {bestEpoch.epoch}
          </div>
        </div>

        <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-800 transition-hover hover:border-indigo-200 dark:hover:border-indigo-900/50">
          <div className="flex items-center gap-2 mb-2 text-slate-500 dark:text-slate-400">
            <TrendingDown className="w-4 h-4 text-rose-500" />
            <span className="text-xs font-semibold uppercase tracking-wider">Min Loss</span>
          </div>
          <div className="text-2xl font-black text-slate-800 dark:text-slate-100">
            {bestEpoch.evalLoss.toFixed(4)}
          </div>
        </div>

        <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-800 transition-hover hover:border-indigo-200 dark:hover:border-indigo-900/50">
          <div className="flex items-center gap-2 mb-2 text-slate-500 dark:text-slate-400">
            <Zap className="w-4 h-4 text-amber-500" />
            <span className="text-xs font-semibold uppercase tracking-wider">Top Accuracy</span>
          </div>
          <div className="text-2xl font-black text-slate-800 dark:text-slate-100">
            {(Math.max(...data.map(d => d.accuracy)) * 100).toFixed(1)}%
          </div>
        </div>

        <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-800 transition-hover hover:border-indigo-200 dark:hover:border-indigo-900/50">
          <div className="flex items-center gap-2 mb-2 text-slate-500 dark:text-slate-400">
            <Activity className="w-4 h-4 text-indigo-500" />
            <span className="text-xs font-semibold uppercase tracking-wider">Samples/Sec</span>
          </div>
          <div className="text-2xl font-black text-slate-800 dark:text-slate-100">
            1,240
          </div>
        </div>
      </div>
    </div>
  );
}
