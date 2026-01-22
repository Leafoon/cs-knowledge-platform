"use client";

import { useEffect, useRef, useMemo } from "react";
import { createRoot } from "react-dom/client";
import { 
    InstructionCycleSimulator, VonNeumannArchitecture, ComputerEvolutionTimeline, SystemLayersVisualization, 
    PythonInterpreterFlow, PythonObjectVisualizer, UnicodeEncodingVisualizer, ListResizingVisualizer, 
    IntegerMemoryLayout, HashTableVisualizer, FunctionCallStackVisualizer, DecoratorExecutionFlow, 
    GeneratorStateVisualizer, ExceptionHierarchyTree, ComputationalGraph, SequentialFlowVisualizer, 
    BatchProcessor, TrainingSimulator, CheckpointSimulator, ProfilerVisualizer, AttentionMatrixVisualizer, 
    ParallelVisualizer, QuantizationVisualizer, DistributedVisualizer, KernelFusionVisualizer, 
    DispatcherVisualizer, TensorBroadcastingVisualizer, TensorStorageVisualizer, ActivationVisualizer, 
    SamplerVisualizer, OptimizerPathVisualizer, TransferLearningVisualizer, TrainingDynamicsVisualizer, 
    ConvolutionVisualizer, HookVisualizer, TorchScriptVisualizer, StridedMemoryVisualizer, CUDAStreamVisualizer,
    // Transformers Components
    TransformersEcosystemComparison, HuggingFaceEcosystemMap, VersionCompatibilityMatrix,
    PipelineFlowVisualizer, GenerationParametersExplorer, TopKTopPVisualizer,
    NERVisualizer, QuestionAnsweringVisualizer, PipelinePerformanceAnalyzer,
    TokenizationVisualizer, TokenAlgorithmComparison, AttentionMaskBuilder,
    ArchitectureExplorer, ConfigEditor, ModelOutputInspector,
    ModelRepoStructureExplorer, CacheManagementVisualizer, PipelineInternalFlow,
    DatasetPipeline, DataCollatorDemo, TrainingLoopVisualizer,
    TrainingArgumentsExplorer, MixedPrecisionComparison, TrainingStepBreakdown,
    CallbackFlow, LearningRateScheduler, TrainingMetricsPlot
} from "@/components/interactive";















interface ContentRendererProps {
    html: string;
    moduleId?: string;
}

const componentMap: Record<string, React.ComponentType> = {
    "InstructionCycleSimulator": InstructionCycleSimulator,
    "VonNeumannArchitecture": VonNeumannArchitecture,
    "ComputerEvolutionTimeline": ComputerEvolutionTimeline,
    "SystemLayersVisualization": SystemLayersVisualization,
    "PythonInterpreterFlow": PythonInterpreterFlow,
    "PythonObjectVisualizer": PythonObjectVisualizer,
    "UnicodeEncodingVisualizer": UnicodeEncodingVisualizer,
    "ListResizingVisualizer": ListResizingVisualizer,
    "IntegerMemoryLayout": IntegerMemoryLayout,
    "HashTableVisualizer": HashTableVisualizer,
    "FunctionCallStackVisualizer": FunctionCallStackVisualizer,
    "DecoratorExecutionFlow": DecoratorExecutionFlow,
    "GeneratorStateVisualizer": GeneratorStateVisualizer,
    "ExceptionHierarchyTree": ExceptionHierarchyTree,
    "ComputationalGraph": ComputationalGraph,
    "SequentialFlowVisualizer": SequentialFlowVisualizer,
    "BatchProcessor": BatchProcessor,
    "TrainingSimulator": TrainingSimulator,
    "CheckpointSimulator": CheckpointSimulator,
    "ProfilerVisualizer": ProfilerVisualizer,
    "AttentionMatrixVisualizer": AttentionMatrixVisualizer,
    "ParallelVisualizer": ParallelVisualizer,
    "QuantizationVisualizer": QuantizationVisualizer,
    "DistributedVisualizer": DistributedVisualizer,
    "KernelFusionVisualizer": KernelFusionVisualizer,
    "DispatcherVisualizer": DispatcherVisualizer,
    "TensorBroadcastingVisualizer": TensorBroadcastingVisualizer,
    "TensorStorageVisualizer": TensorStorageVisualizer,
    "ActivationVisualizer": ActivationVisualizer,
    "SamplerVisualizer": SamplerVisualizer,
    "OptimizerPathVisualizer": OptimizerPathVisualizer,
    "TransferLearningVisualizer": TransferLearningVisualizer,
    "TrainingDynamicsVisualizer": TrainingDynamicsVisualizer,
    "ConvolutionVisualizer": ConvolutionVisualizer,
    "HookVisualizer": HookVisualizer,
    "TorchScriptVisualizer": TorchScriptVisualizer,
    "StridedMemoryVisualizer": StridedMemoryVisualizer,
    "CUDAStreamVisualizer": CUDAStreamVisualizer,
    // Transformers Components
    "TransformersEcosystemComparison": TransformersEcosystemComparison,
    "HuggingFaceEcosystemMap": HuggingFaceEcosystemMap,
    "VersionCompatibilityMatrix": VersionCompatibilityMatrix,
    "PipelineFlowVisualizer": PipelineFlowVisualizer,
    "GenerationParametersExplorer": GenerationParametersExplorer,
    "TopKTopPVisualizer": TopKTopPVisualizer,
    "NERVisualizer": NERVisualizer,
    "QuestionAnsweringVisualizer": QuestionAnsweringVisualizer,
    "PipelinePerformanceAnalyzer": PipelinePerformanceAnalyzer,
    "TokenizationVisualizer": TokenizationVisualizer,
    "TokenAlgorithmComparison": TokenAlgorithmComparison,
    "AttentionMaskBuilder": AttentionMaskBuilder,
    "ArchitectureExplorer": ArchitectureExplorer,
    "ConfigEditor": ConfigEditor,
    "ModelOutputInspector": ModelOutputInspector,
    // Chapter 0 missing components
    "ModelRepoStructureExplorer": ModelRepoStructureExplorer,
    "CacheManagementVisualizer": CacheManagementVisualizer,
    "PipelineInternalFlow": PipelineInternalFlow,
    // Chapter 4-5 components
    "DatasetPipeline": DatasetPipeline,
    "DataCollatorDemo": DataCollatorDemo,
    "TrainingLoopVisualizer": TrainingLoopVisualizer,
    "TrainingArgumentsExplorer": TrainingArgumentsExplorer,
    "MixedPrecisionComparison": MixedPrecisionComparison,
    "TrainingStepBreakdown": TrainingStepBreakdown,
    "CallbackFlow": CallbackFlow,
    "LearningRateScheduler": LearningRateScheduler,
    "TrainingMetricsPlot": TrainingMetricsPlot,
};

export function ContentRenderer({ html, moduleId }: ContentRendererProps) {
    const contentRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!contentRef.current) return;

        // Add IDs to headings for scroll spy
        const headings = contentRef.current.querySelectorAll('h1, h2, h3');
        headings.forEach((heading) => {
            const text = heading.textContent || '';
            const id = text
                .toLowerCase()
                .trim()
                .replace(/\s+/g, '-')
                .replace(/[^\w\-\u4e00-\u9fa5]+/g, '')
                .replace(/\-\-+/g, '-');
            heading.id = id;
        });

        // Inject interactive components
        // Find and replace component markers (div with data-component attribute)
        const markers = contentRef.current.querySelectorAll('div[data-component]');
        markers.forEach((marker) => {
            const componentName = marker.getAttribute('data-component');

            if (componentName) {
                const Component = componentMap[componentName];

                if (Component) {
                    const container = document.createElement('div');
                    container.className = 'interactive-component-container my-8';
                    marker.replaceWith(container);

                    const root = createRoot(container);
                    root.render(<Component />);
                } else {
                    console.error(`Component "${componentName}" not found in componentMap`);
                    (marker as HTMLElement).innerText = `[Error: Component "${componentName}" not found]`;
                    (marker as HTMLElement).style.color = 'red';
                    (marker as HTMLElement).style.border = '1px solid red';
                    (marker as HTMLElement).style.padding = '8px';
                }
            }
        });
    }, [html, moduleId]);

    return (
        <div
            ref={contentRef}
            className="prose-content max-w-none"
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
}
