'use client'
import React from 'react'
import { Info } from 'lucide-react'

const PlaceholderComponent = ({ name, title }: { name: string, title: string }) => (
  <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
    <div className="flex items-center gap-3 mb-4">
      <div className="p-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg">
        <Info className="w-6 h-6 text-white" />
      </div>
      <h3 className="text-2xl font-bold text-slate-800">{title}</h3>
    </div>
    <div className="bg-white rounded-lg p-6 border border-blue-200">
      <p className="text-slate-700">此组件用于演示 <strong>{name}</strong> 相关概念。</p>
    </div>
  </div>
)

export function ExtremeLowMemoryTraining() { return <PlaceholderComponent name="ExtremeLowMemoryTraining" title="极低显存训练" /> }
export function FloatPrecisionRangeTradeoff() { return <PlaceholderComponent name="FloatPrecisionRangeTradeoff" title="浮点精度范围权衡" /> }
export function PrecisionLossComparison() { return <PlaceholderComponent name="PrecisionLossComparison" title="精度损失对比" /> }
export function QuantizationMethodComparison() { return <PlaceholderComponent name="QuantizationMethodComparison" title="量化方法对比" /> }
export function QuantizationMethodsComprehensiveComparison() { return <PlaceholderComponent name="QuantizationMethodsComprehensiveComparison" title="量化方法综合对比" /> }
export function PerTensorVsPerChannelQuant() { return <PlaceholderComponent name="PerTensorVsPerChannelQuant" title="Per-Tensor vs Per-Channel 量化" /> }
export function NF4vsINT4Comparison() { return <PlaceholderComponent name="NF4vsINT4Comparison" title="NF4 vs INT4 对比" /> }
export function DistributedMixedPrecision() { return <PlaceholderComponent name="DistributedMixedPrecision" title="分布式混合精度训练" /> }
export function AccelerateWorkflowVisualizer() { return <PlaceholderComponent name="AccelerateWorkflowVisualizer" title="Accelerate 工作流程" /> }
export function AcceleratorAPIDemo() { return <PlaceholderComponent name="AcceleratorAPIDemo" title="Accelerator API 演示" /> }
export function ThreeDParallelismVisualizer() { return <PlaceholderComponent name="ThreeDParallelismVisualizer" title="3D 并行可视化" /> }
export function CollectiveCommunicationPrimitives() { return <PlaceholderComponent name="CollectiveCommunicationPrimitives" title="集合通信原语" /> }
export function TGIArchitectureDiagram() { return <PlaceholderComponent name="TGIArchitectureDiagram" title="TGI 架构图" /> }
export function ModelExportDecisionTree() { return <PlaceholderComponent name="ModelExportDecisionTree" title="模型导出决策树" /> }
export function BackendAutoSelector() { return <PlaceholderComponent name="BackendAutoSelector" title="后端自动选择器" /> }
export function OptimizationEffectComparison() { return <PlaceholderComponent name="OptimizationEffectComparison" title="优化效果对比" /> }
export function ProfilerVisualizationDemo() { return <PlaceholderComponent name="ProfilerVisualizationDemo" title="Profiler 可视化演示" /> }
export function PEFTTrainingSpeedComparison() { return <PlaceholderComponent name="PEFTTrainingSpeedComparison" title="PEFT 训练速度对比" /> }
