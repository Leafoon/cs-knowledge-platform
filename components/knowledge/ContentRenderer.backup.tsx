"use client";

import { useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { lazy, Suspense } from "react";

// Loading fallback
const ComponentLoading = () => (
    <div className="flex items-center justify-center p-8 bg-bg-elevated/50 rounded-lg border border-border-subtle my-6">
        <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-4 border-accent-primary/30 border-t-accent-primary rounded-full animate-spin" />
            <p className="text-sm text-text-secondary">加载交互组件中...</p>
        </div>
    </div>
);

// 懒加载所有组件
const lazyComponents = {
    // Computer Organization
    InstructionCycleSimulator: lazy(() => import('@/components/interactive').then(m => ({ default: m.InstructionCycleSimulator }))),
    VonNeumannArchitecture: lazy(() => import('@/components/interactive').then(m => ({ default: m.VonNeumannArchitecture }))),
    ComputerEvolutionTimeline: lazy(() => import('@/components/interactive').then(m => ({ default: m.ComputerEvolutionTimeline }))),
    SystemLayersVisualization: lazy(() => import('@/components/interactive').then(m => ({ default: m.SystemLayersVisualization }))),
    
    // Python
    PythonInterpreterFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.PythonInterpreterFlow }))),
    PythonObjectVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.PythonObjectVisualizer }))),
    UnicodeEncodingVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.UnicodeEncodingVisualizer }))),
    ListResizingVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.ListResizingVisualizer }))),
    IntegerMemoryLayout: lazy(() => import('@/components/interactive').then(m => ({ default: m.IntegerMemoryLayout }))),
    HashTableVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.HashTableVisualizer }))),
    FunctionCallStackVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.FunctionCallStackVisualizer }))),
    DecoratorExecutionFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.DecoratorExecutionFlow }))),
    GeneratorStateVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.GeneratorStateVisualizer }))),
    ExceptionHierarchyTree: lazy(() => import('@/components/interactive').then(m => ({ default: m.ExceptionHierarchyTree }))),
    
    // PyTorch
    ComputationalGraph: lazy(() => import('@/components/interactive').then(m => ({ default: m.ComputationalGraph }))),
    SequentialFlowVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.SequentialFlowVisualizer }))),
    BatchProcessor: lazy(() => import('@/components/interactive').then(m => ({ default: m.BatchProcessor }))),
    TrainingSimulator: lazy(() => import('@/components/interactive').then(m => ({ default: m.TrainingSimulator }))),
    CheckpointSimulator: lazy(() => import('@/components/interactive').then(m => ({ default: m.CheckpointSimulator }))),
    ProfilerVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.ProfilerVisualizer }))),
    AttentionMatrixVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.AttentionMatrixVisualizer }))),
    ParallelVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.ParallelVisualizer }))),
    QuantizationVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.QuantizationVisualizer }))),
    DistributedVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.DistributedVisualizer }))),
    KernelFusionVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.KernelFusionVisualizer }))),
    DispatcherVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.DispatcherVisualizer }))),
    TensorBroadcastingVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TensorBroadcastingVisualizer }))),
    TensorStorageVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TensorStorageVisualizer }))),
    ActivationVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.ActivationVisualizer }))),
    SamplerVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.SamplerVisualizer }))),
    OptimizerPathVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.OptimizerPathVisualizer }))),
    TransferLearningVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TransferLearningVisualizer }))),
    TrainingDynamicsVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TrainingDynamicsVisualizer }))),
    ConvolutionVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.ConvolutionVisualizer }))),
    HookVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.HookVisualizer }))),
    TorchScriptVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TorchScriptVisualizer }))),
    StridedMemoryVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.StridedMemoryVisualizer }))),
    CUDAStreamVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.CUDAStreamVisualizer }))),
    
    // Transformers Chapter 0-5
    TransformersEcosystemComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.TransformersEcosystemComparison }))),
    HuggingFaceEcosystemMap: lazy(() => import('@/components/interactive').then(m => ({ default: m.HuggingFaceEcosystemMap }))),
    VersionCompatibilityMatrix: lazy(() => import('@/components/interactive').then(m => ({ default: m.VersionCompatibilityMatrix }))),
    PipelineFlowVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.PipelineFlowVisualizer }))),
    GenerationParametersExplorer: lazy(() => import('@/components/interactive').then(m => ({ default: m.GenerationParametersExplorer }))),
    TopKTopPVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TopKTopPVisualizer }))),
    NERVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.NERVisualizer }))),
    QuestionAnsweringVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.QuestionAnsweringVisualizer }))),
    PipelinePerformanceAnalyzer: lazy(() => import('@/components/interactive').then(m => ({ default: m.PipelinePerformanceAnalyzer }))),
    TokenizationVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TokenizationVisualizer }))),
    TokenAlgorithmComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.TokenAlgorithmComparison }))),
    AttentionMaskBuilder: lazy(() => import('@/components/interactive').then(m => ({ default: m.AttentionMaskBuilder }))),
    ArchitectureExplorer: lazy(() => import('@/components/interactive').then(m => ({ default: m.ArchitectureExplorer }))),
    ConfigEditor: lazy(() => import('@/components/interactive').then(m => ({ default: m.ConfigEditor }))),
    ModelOutputInspector: lazy(() => import('@/components/interactive').then(m => ({ default: m.ModelOutputInspector }))),
    ModelRepoStructureExplorer: lazy(() => import('@/components/interactive').then(m => ({ default: m.ModelRepoStructureExplorer }))),
    CacheManagementVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.CacheManagementVisualizer }))),
    PipelineInternalFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.PipelineInternalFlow }))),
    DatasetPipeline: lazy(() => import('@/components/interactive').then(m => ({ default: m.DatasetPipeline }))),
    DataCollatorDemo: lazy(() => import('@/components/interactive').then(m => ({ default: m.DataCollatorDemo }))),
    TrainingLoopVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TrainingLoopVisualizer }))),
    TrainingArgumentsExplorer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TrainingArgumentsExplorer }))),
    MixedPrecisionComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.MixedPrecisionComparison }))),
    TrainingStepBreakdown: lazy(() => import('@/components/interactive').then(m => ({ default: m.TrainingStepBreakdown }))),
    CallbackFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.CallbackFlow }))),
    LearningRateScheduler: lazy(() => import('@/components/interactive').then(m => ({ default: m.LearningRateScheduler }))),
    TrainingMetricsPlot: lazy(() => import('@/components/interactive').then(m => ({ default: m.TrainingMetricsPlot }))),
    
    // Transformers Chapter 6-10
    PEFTMethodComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.PEFTMethodComparison }))),
    LoRAMatrixDecomposition: lazy(() => import('@/components/interactive').then(m => ({ default: m.LoRAMatrixDecomposition }))),
    LoRARankSelector: lazy(() => import('@/components/interactive').then(m => ({ default: m.LoRARankSelector }))),
    QLoRAQuantizationFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.QLoRAQuantizationFlow }))),
    NF4DataTypeVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.NF4DataTypeVisualizer }))),
    AdaLoraRankEvolution: lazy(() => import('@/components/interactive').then(m => ({ default: m.AdaLoraRankEvolution }))),
    MemoryOptimizationComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.MemoryOptimizationComparison }))),
    PrecisionFormatComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.PrecisionFormatComparison }))),
    FloatFormatBitLayout: lazy(() => import('@/components/interactive').then(m => ({ default: m.FloatFormatBitLayout }))),
    MixedPrecisionTrainingFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.MixedPrecisionTrainingFlow }))),
    QuantizationProcessVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.QuantizationProcessVisualizer }))),
    GradientAccumulationVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.GradientAccumulationVisualizer }))),
    PTQMethodComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.PTQMethodComparison }))),
    DistributedTrainingNeedVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.DistributedTrainingNeedVisualizer }))),
    ParallelismStrategyComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.ParallelismStrategyComparison }))),
    DDPCommunicationFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.DDPCommunicationFlow }))),
    FSDPShardingVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.FSDPShardingVisualizer }))),
    DeepSpeedZeROStages: lazy(() => import('@/components/interactive').then(m => ({ default: m.DeepSpeedZeROStages }))),
    PipelineParallelismVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.PipelineParallelismVisualizer }))),
    TensorParallelismVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.TensorParallelismVisualizer }))),
    InferenceMetricsVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.InferenceMetricsVisualizer }))),
    KVCacheMechanismVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.KVCacheMechanismVisualizer }))),
    FlashAttentionIOComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.FlashAttentionIOComparison }))),
    TorchCompileSpeedupChart: lazy(() => import('@/components/interactive').then(m => ({ default: m.TorchCompileSpeedupChart }))),
    PagedAttentionVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.PagedAttentionVisualizer }))),
    SpeculativeDecodingFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.SpeculativeDecodingFlow }))),
    DeploymentStackComparison: lazy(() => import('@/components/interactive').then(m => ({ default: m.DeploymentStackComparison }))),
    QLoRAInnovationTimeline: lazy(() => import('@/components/interactive').then(m => ({ default: m.QLoRAInnovationTimeline }))),
    NF4EncodingVisualizer: lazy(() => import('@/components/interactive').then(m => ({ default: m.NF4EncodingVisualizer }))),
    DoubleQuantizationFlow: lazy(() => import('@/components/interactive').then(m => ({ default: m.DoubleQuantizationFlow }))),
    PagedOptimizerVisualizer, LoRATargetModulesSelector, QLoRAMemoryOptimizationComparison,
    PEFTMethodComparisonTable,
    // Chapter 11 Mixed Precision Training
    FloatFormatComparison, AMPWorkflow, GradScalerVisualizer,
    FP16vsBF16Comparison, TensorCorePerformance, MixedPrecisionBenchmark,
    TrainingStabilityComparison,
    // Chapter 12 Post-Training Quantization
    PTQvsQATComparison, QuantizationGranularityVisualizer, GPTQAlgorithmFlow,
    GPTQvsBitsAndBytesComparison, AWQChannelProtection, PerplexityComparisonChart,
    QuantizationThroughputComparison, QuantizationMethodSummaryTable,
    // Chapter 13 Gradient Checkpointing
    MemoryBreakdownInteractive, GradientCheckpointingFlow, OptimizationCombinator,
    // Chapter 14 Accelerate
    AccelerateWorkflow, DistributedCommunicationVisualizer,
    // Chapter 15 FSDP
    ZeROStagesComparison, AllGatherReduceScatter,
    // Chapter 16 DeepSpeed
    ZeROStageDecisionTree, DeepSpeedOffloadFlow, ThreeDParallelismDiagram,
    // Chapter 17 Inference Optimization
    InferenceLatencyBreakdown, KVCacheComparisonVisualizer,
    // Chapter 18 vLLM & TGI
    PagedAttentionMemoryVisualizer, ContinuousBatchingDemo, InferenceFrameworkComparison,
    // Chapter 19 Speculative Decoding
    SpeculativeDecodingFlowVisualizer, MQAvsGQAComparison,
    // Chapter 20 Model Export
    SafetensorsVsPickleComparison, TorchScriptModeComparison, OperatorFusionVisualizer,
    // Chapter 21 Optimum
    OptimumBackendEcosystem, QuantizationWorkflowVisualizer,
    // Chapter 22 API & Docker Deployment
    RequestQueueVisualizer, K8sDeploymentVisualizer,
    // Chapter 23 Attention Mechanisms
    AttentionWeightHeatmap, MaskBuilder, PositionEncodingVisualizer, KVCacheDynamics,
    // Chapter 24 Custom Model Development
    ModelBuilderTool, CustomAttentionComparator,
    // Chapter 25 Custom Trainer
    TrainerHookFlow, LossFunctionExplorer,
    // Chapter 26 Multimodal Models
    MultimodalArchitecture, VisionEncoderVisualizer,
    // Chapter 27 RLHF
    RLHFPipeline, DPOvsRLHF,
    // Chapter 28 Frontier Research
    LongContextStrategies, MoERouting, AttentionPatternAnalyzer
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
    // Chapter 6 PEFT
    "PEFTMethodComparison": PEFTMethodComparison,
    "LoRAMatrixDecomposition": LoRAMatrixDecomposition,
    "LoRARankSelector": LoRARankSelector,
    "QLoRAQuantizationFlow": QLoRAQuantizationFlow,
    "NF4DataTypeVisualizer": NF4DataTypeVisualizer,
    "AdaLoraRankEvolution": AdaLoraRankEvolution,
    "MemoryOptimizationComparison": MemoryOptimizationComparison,
    // Chapter 7 Quantization
    "PrecisionFormatComparison": PrecisionFormatComparison,
    "FloatFormatBitLayout": FloatFormatBitLayout,
    "MixedPrecisionTrainingFlow": MixedPrecisionTrainingFlow,
    "QuantizationProcessVisualizer": QuantizationProcessVisualizer,
    "GradientAccumulationVisualizer": GradientAccumulationVisualizer,
    "PTQMethodComparison": PTQMethodComparison,
    // Chapter 8 Distributed Training
    "DistributedTrainingNeedVisualizer": DistributedTrainingNeedVisualizer,
    "ParallelismStrategyComparison": ParallelismStrategyComparison,
    "DDPCommunicationFlow": DDPCommunicationFlow,
    "FSDPShardingVisualizer": FSDPShardingVisualizer,
    "DeepSpeedZeROStages": DeepSpeedZeROStages,
    "PipelineParallelismVisualizer": PipelineParallelismVisualizer,
    "TensorParallelismVisualizer": TensorParallelismVisualizer,
    // Chapter 9 Inference Optimization
    "InferenceMetricsVisualizer": InferenceMetricsVisualizer,
    "KVCacheMechanismVisualizer": KVCacheMechanismVisualizer,
    "FlashAttentionIOComparison": FlashAttentionIOComparison,
    "TorchCompileSpeedupChart": TorchCompileSpeedupChart,
    "PagedAttentionVisualizer": PagedAttentionVisualizer,
    "SpeculativeDecodingFlow": SpeculativeDecodingFlow,
    "DeploymentStackComparison": DeploymentStackComparison,
    // Chapter 10 QLoRA
    "QLoRAInnovationTimeline": QLoRAInnovationTimeline,
    "NF4EncodingVisualizer": NF4EncodingVisualizer,
    "DoubleQuantizationFlow": DoubleQuantizationFlow,
    "PagedOptimizerVisualizer": PagedOptimizerVisualizer,
    "LoRATargetModulesSelector": LoRATargetModulesSelector,
    "QLoRAMemoryOptimizationComparison": QLoRAMemoryOptimizationComparison,
    "PEFTMethodComparisonTable": PEFTMethodComparisonTable,
    // Chapter 11 Mixed Precision Training
    "FloatFormatComparison": FloatFormatComparison,
    "AMPWorkflow": AMPWorkflow,
    "GradScalerVisualizer": GradScalerVisualizer,
    "FP16vsBF16Comparison": FP16vsBF16Comparison,
    "TensorCorePerformance": TensorCorePerformance,
    "MixedPrecisionBenchmark": MixedPrecisionBenchmark,
    "TrainingStabilityComparison": TrainingStabilityComparison,
    // Chapter 12 Post-Training Quantization
    "PTQvsQATComparison": PTQvsQATComparison,
    "QuantizationGranularityVisualizer": QuantizationGranularityVisualizer,
    "GPTQAlgorithmFlow": GPTQAlgorithmFlow,
    "GPTQvsBitsAndBytesComparison": GPTQvsBitsAndBytesComparison,
    "AWQChannelProtection": AWQChannelProtection,
    "PerplexityComparisonChart": PerplexityComparisonChart,
    "QuantizationThroughputComparison": QuantizationThroughputComparison,
    "QuantizationMethodSummaryTable": QuantizationMethodSummaryTable,
    // Chapter 13 Gradient Checkpointing
    "MemoryBreakdownInteractive": MemoryBreakdownInteractive,
    "GradientCheckpointingFlow": GradientCheckpointingFlow,
    "OptimizationCombinator": OptimizationCombinator,
    // Chapter 14 Accelerate
    "AccelerateWorkflow": AccelerateWorkflow,
    "DistributedCommunicationVisualizer": DistributedCommunicationVisualizer,
    // Chapter 15 FSDP
    "ZeROStagesComparison": ZeROStagesComparison,
    "AllGatherReduceScatter": AllGatherReduceScatter,
    // Chapter 16 DeepSpeed
    "ZeROStageDecisionTree": ZeROStageDecisionTree,
    "DeepSpeedOffloadFlow": DeepSpeedOffloadFlow,
    "ThreeDParallelismDiagram": ThreeDParallelismDiagram,
    // Chapter 17 Inference Optimization
    "InferenceLatencyBreakdown": InferenceLatencyBreakdown,
    "KVCacheComparisonVisualizer": KVCacheComparisonVisualizer,
    // Chapter 18 vLLM & TGI
    "PagedAttentionMemoryVisualizer": PagedAttentionMemoryVisualizer,
    "ContinuousBatchingDemo": ContinuousBatchingDemo,
    "InferenceFrameworkComparison": InferenceFrameworkComparison,
    // Chapter 19 Speculative Decoding
    "SpeculativeDecodingFlowVisualizer": SpeculativeDecodingFlowVisualizer,
    "MQAvsGQAComparison": MQAvsGQAComparison,
    // Chapter 20 Model Export
    "SafetensorsVsPickleComparison": SafetensorsVsPickleComparison,
    "TorchScriptModeComparison": TorchScriptModeComparison,
    "OperatorFusionVisualizer": OperatorFusionVisualizer,
    // Chapter 21 Optimum
    "OptimumBackendEcosystem": OptimumBackendEcosystem,
    "QuantizationWorkflowVisualizer": QuantizationWorkflowVisualizer,
    // Chapter 22 API & Docker Deployment
    "RequestQueueVisualizer": RequestQueueVisualizer,
    "K8sDeploymentVisualizer": K8sDeploymentVisualizer,
    // Chapter 23 Attention Mechanisms
    "AttentionWeightHeatmap": AttentionWeightHeatmap,
    "MaskBuilder": MaskBuilder,
    "PositionEncodingVisualizer": PositionEncodingVisualizer,
    "KVCacheDynamics": KVCacheDynamics,
    // Chapter 24 Custom Model Development
    "ModelBuilderTool": ModelBuilderTool,
    "CustomAttentionComparator": CustomAttentionComparator,
    // Chapter 25 Custom Trainer
    "TrainerHookFlow": TrainerHookFlow,
    "LossFunctionExplorer": LossFunctionExplorer,
    // Chapter 26 Multimodal Models
    "MultimodalArchitecture": MultimodalArchitecture,
    "VisionEncoderVisualizer": VisionEncoderVisualizer,
    // Chapter 27 RLHF
    "RLHFPipeline": RLHFPipeline,
    "DPOvsRLHF": DPOvsRLHF,
    // Chapter 28 Frontier Research
    "LongContextStrategies": LongContextStrategies,
    "MoERouting": MoERouting,
    "AttentionPatternAnalyzer": AttentionPatternAnalyzer,
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
