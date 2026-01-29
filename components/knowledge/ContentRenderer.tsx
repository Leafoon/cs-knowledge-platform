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
    CallbackFlow, LearningRateScheduler, TrainingMetricsPlot,
    // Chapter 6-7
    PEFTMethodComparison, LoRAMatrixDecomposition, LoRARankSelector,
    QLoRAQuantizationFlow, NF4DataTypeVisualizer, AdaLoraRankEvolution,
    MemoryOptimizationComparison, PrecisionFormatComparison, FloatFormatBitLayout,
    MixedPrecisionTrainingFlow, QuantizationProcessVisualizer, GradientAccumulationVisualizer,
    PTQMethodComparison,
    // Chapter 8 Distributed Training
    DistributedTrainingNeedVisualizer, ParallelismStrategyComparison, DDPCommunicationFlow,
    FSDPShardingVisualizer, DeepSpeedZeROStages, PipelineParallelismVisualizer,
    TensorParallelismVisualizer,
    // Chapter 9 Inference Optimization
    InferenceMetricsVisualizer, KVCacheMechanismVisualizer, FlashAttentionIOComparison,
    TorchCompileSpeedupChart, PagedAttentionVisualizer, SpeculativeDecodingFlow,
    DeploymentStackComparison,
    // Chapter 10 QLoRA
    QLoRAInnovationTimeline, NF4EncodingVisualizer, DoubleQuantizationFlow,
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
    LongContextStrategies, MoERouting, AttentionPatternAnalyzer,
    // Missing components - Chapter 0-1
    TaskTypeGallery, ZeroShotClassificationDemo, TaskInferenceFlowchart,
    // Missing components - PEFT & Quantization  
    LoRAMemoryAccuracyTradeoff, FSDPScalingChart,
    // Placeholder components
    ExtremeLowMemoryTraining, FloatPrecisionRangeTradeoff, PrecisionLossComparison,
    QuantizationMethodComparison, QuantizationMethodsComprehensiveComparison,
    PerTensorVsPerChannelQuant, NF4vsINT4Comparison, DistributedMixedPrecision,
    AccelerateWorkflowVisualizer, AcceleratorAPIDemo, ThreeDParallelismVisualizer,
    CollectiveCommunicationPrimitives, TGIArchitectureDiagram, ModelExportDecisionTree,
    BackendAutoSelector, OptimizationEffectComparison, ProfilerVisualizationDemo,
    PEFTTrainingSpeedComparison,
    // LangChain Components
    LangChainEcosystemMap, LangChainArchitectureFlow, RunnableProtocolVisualizer, PromptTemplateBuilder,
    MessageFlowDiagram, LegacyVsLCELComparison, ErrorHandlingFlow, ChainGraphVisualizer,
    RunnableCompositionFlow, ParallelExecutionDemo, StreamingVisualizer, AsyncPerformanceComparison,
    ChainOrchestrationDiagram, MapReduceVisualizer, RouterDecisionTree,
    FewShotExampleSelector, PromptComposer, HubBrowser,
    OutputParserFlow, StructuredOutputBuilder, ParsingErrorDemo,
    ToolCallingFlow, FunctionSchemaBuilder, ToolExecutionTimeline,
    MemoryEvolutionTimeline, MemoryTypeComparison, EntityMemoryGraph,
    // Chapter 10-11: Persistence & Optimization
    PersistenceBackendComparison, SessionLifecycleFlow, StateCheckpointVisualizer,
    TokenManagementDashboard, MemoryRetrievalPerformance, PrivacyComplianceFlow,
    // Chapter 12-13: RAG & Vector Stores
    RAGArchitectureDiagram, TextSplittingVisualizer, EmbeddingSpaceVisualization,
    VectorStoreComparison, SimilaritySearchDemo, HybridRetrievalFlow,
    // Chapter 14-15: LangGraph & Agent Design
    LangGraphArchitectureDiagram, StateGraphExecution, CheckpointTimeline,
    AgentArchitectureComparison, ToolCallFlow, MultiAgentArchitecture,
    // Chapter 16: LangSmith Observability
    LangSmithTraceVisualization, EvaluationDashboard, MonitoringDashboard,
    // Chapter 17: LangServe Deployment
    LangServePlayground, DeploymentArchitecture, KubernetesArchitecture,
    // Chapter 18: Advanced RAG & Optimization
    AdvancedRAGComparison, QueryTransformationFlow, HybridSearchArchitecture,
    // Chapter 19: Production Engineering & Best Practices
    RetryFallbackFlow, PerformanceOptimizationDashboard, ProductionArchitectureDiagram,
    // Chapter 20: Multi-Agent Systems
    MultiAgentArchitectureComparison, SupervisorRoutingFlow, CollaborativeDebateFlow, MultiAgentCodeGenFlow,
    // Chapter 21: Planning & Reflection Agents
    PlanExecuteFlowDiagram, ReflectionLoopVisualizer, ErrorRecoveryFlowDiagram,
    // Chapter 22: LangSmith Tracing
    TraceTreeVisualizer, SpanTimelineChart, TokenUsageBreakdown,
    // Chapter 23: LangSmith Evaluation
    EvaluationPipeline, ABTestComparison, FeedbackDashboard,
    // Chapter 24: LangSmith Production Monitoring
    MonitoringDashboardReal, AlertRuleBuilder, CostAnalysisDashboard,
    // Chapter 25: LangServe Basics
    LangServeArchitecture, EndpointExplorer, RemoteRunnableDemo,
    // Chapter 26: LangServe Advanced Features
    AuthenticationFlow, RateLimitingVisualizer, MetricsDashboard,
    // Chapter 27: Deployment & Containerization
    DockerBuildFlow, CloudPlatformComparison, K8sArchitectureDiagram,
    // Chapter 28: Advanced Agent Patterns
    HumanInLoopFlow, LongTermMemoryArchitecture, ToolOrchestrationVisualizer,
    // Chapter 29: Ecosystem Integration
    FrameworkComparisonMatrix, MigrationPathGuide, APIMappingTable,
    // Chapter 30: Performance Optimization
    CachingStrategyComparison, CostOptimizationDashboard, ReliabilityDecisionTree,
    // Chapter 31: Security & Privacy
    PromptInjectionDefense, PIIDetectionFlow, SecurityAuditDashboard,
    // Chapter 32: Large-Scale Architecture
    MicroserviceArchitecture, ModelRoutingFlow, ABTestDashboard,
    // Chapter 33: Future Research
    SpeculativeDecodingFlow as SpeculativeDecodingFlowLangChain, MultimodalMemoryGraph, PluginEcosystemMap,
    // Reinforcement Learning Components
    RLEcosystemMap, AgentEnvironmentLoop, RLTimelineEvolution,
    MDPGraphVisualizer, BellmanEquationDerivation, ValueFunctionEvolution
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
    // Missing components - Chapter 0-1
    "TaskTypeGallery": TaskTypeGallery,
    "ZeroShotClassificationDemo": ZeroShotClassificationDemo,
    "TaskInferenceFlowchart": TaskInferenceFlowchart,
    // Missing components - PEFT & Quantization
    "LoRAMemoryAccuracyTradeoff": LoRAMemoryAccuracyTradeoff,
    "FSDPScalingChart": FSDPScalingChart,
    // Placeholder components
    "ExtremeLowMemoryTraining": ExtremeLowMemoryTraining,
    "FloatPrecisionRangeTradeoff": FloatPrecisionRangeTradeoff,
    "PrecisionLossComparison": PrecisionLossComparison,
    "QuantizationMethodComparison": QuantizationMethodComparison,
    "QuantizationMethodsComprehensiveComparison": QuantizationMethodsComprehensiveComparison,
    "PerTensorVsPerChannelQuant": PerTensorVsPerChannelQuant,
    "NF4vsINT4Comparison": NF4vsINT4Comparison,
    "DistributedMixedPrecision": DistributedMixedPrecision,
    "AccelerateWorkflowVisualizer": AccelerateWorkflowVisualizer,
    "AcceleratorAPIDemo": AcceleratorAPIDemo,
    "ThreeDParallelismVisualizer": ThreeDParallelismVisualizer,
    "CollectiveCommunicationPrimitives": CollectiveCommunicationPrimitives,
    "TGIArchitectureDiagram": TGIArchitectureDiagram,
    "ModelExportDecisionTree": ModelExportDecisionTree,
    "BackendAutoSelector": BackendAutoSelector,
    "OptimizationEffectComparison": OptimizationEffectComparison,
    "ProfilerVisualizationDemo": ProfilerVisualizationDemo,
    "PEFTTrainingSpeedComparison": PEFTTrainingSpeedComparison,
    // LangChain Components
    "LangChainEcosystemMap": LangChainEcosystemMap,
    "LangChainArchitectureFlow": LangChainArchitectureFlow,
    "RunnableProtocolVisualizer": RunnableProtocolVisualizer,
    "PromptTemplateBuilder": PromptTemplateBuilder,
    "MessageFlowDiagram": MessageFlowDiagram,
    "LegacyVsLCELComparison": LegacyVsLCELComparison,
    "ErrorHandlingFlow": ErrorHandlingFlow,
    "ChainGraphVisualizer": ChainGraphVisualizer,
    "RunnableCompositionFlow": RunnableCompositionFlow,
    "ParallelExecutionDemo": ParallelExecutionDemo,
    "StreamingVisualizer": StreamingVisualizer,
    "AsyncPerformanceComparison": AsyncPerformanceComparison,
    "ChainOrchestrationDiagram": ChainOrchestrationDiagram,
    "MapReduceVisualizer": MapReduceVisualizer,
    "RouterDecisionTree": RouterDecisionTree,
    "FewShotExampleSelector": FewShotExampleSelector,
    "PromptComposer": PromptComposer,
    "HubBrowser": HubBrowser,
    "OutputParserFlow": OutputParserFlow,
    "StructuredOutputBuilder": StructuredOutputBuilder,
    "ParsingErrorDemo": ParsingErrorDemo,
    "ToolCallingFlow": ToolCallingFlow,
    "FunctionSchemaBuilder": FunctionSchemaBuilder,
    "ToolExecutionTimeline": ToolExecutionTimeline,
    "MemoryEvolutionTimeline": MemoryEvolutionTimeline,
    "MemoryTypeComparison": MemoryTypeComparison,
    "EntityMemoryGraph": EntityMemoryGraph,
    // Chapter 10-11
    "PersistenceBackendComparison": PersistenceBackendComparison,
    "SessionLifecycleFlow": SessionLifecycleFlow,
    "StateCheckpointVisualizer": StateCheckpointVisualizer,
    "TokenManagementDashboard": TokenManagementDashboard,
    "MemoryRetrievalPerformance": MemoryRetrievalPerformance,
    "PrivacyComplianceFlow": PrivacyComplianceFlow,
    // Chapter 12-13: RAG & Vector Stores
    "RAGArchitectureDiagram": RAGArchitectureDiagram,
    "TextSplittingVisualizer": TextSplittingVisualizer,
    "EmbeddingSpaceVisualization": EmbeddingSpaceVisualization,
    "VectorStoreComparison": VectorStoreComparison,
    "SimilaritySearchDemo": SimilaritySearchDemo,
    "HybridRetrievalFlow": HybridRetrievalFlow,
    // Chapter 14-15: LangGraph & Agent Design
    "LangGraphArchitectureDiagram": LangGraphArchitectureDiagram,
    "StateGraphExecution": StateGraphExecution,
    "CheckpointTimeline": CheckpointTimeline,
    "AgentArchitectureComparison": AgentArchitectureComparison,
    "ToolCallFlow": ToolCallFlow,
    "MultiAgentArchitecture": MultiAgentArchitecture,
    // Chapter 16: LangSmith Observability
    "LangSmithTraceVisualization": LangSmithTraceVisualization,
    "EvaluationDashboard": EvaluationDashboard,
    "MonitoringDashboard": MonitoringDashboard,
    // Chapter 17: LangServe Deployment
    "LangServePlayground": LangServePlayground,
    "DeploymentArchitecture": DeploymentArchitecture,
    "KubernetesArchitecture": KubernetesArchitecture,
    // Chapter 18: Advanced RAG & Optimization
    "AdvancedRAGComparison": AdvancedRAGComparison,
    "QueryTransformationFlow": QueryTransformationFlow,
    "HybridSearchArchitecture": HybridSearchArchitecture,
    // Chapter 19: Production Engineering & Best Practices
    "RetryFallbackFlow": RetryFallbackFlow,
    "PerformanceOptimizationDashboard": PerformanceOptimizationDashboard,
    "ProductionArchitectureDiagram": ProductionArchitectureDiagram,
    // Chapter 20: Multi-Agent Systems
    "MultiAgentArchitectureComparison": MultiAgentArchitectureComparison,
    "SupervisorRoutingFlow": SupervisorRoutingFlow,
    "CollaborativeDebateFlow": CollaborativeDebateFlow,
    "MultiAgentCodeGenFlow": MultiAgentCodeGenFlow,
    // Chapter 21: Planning & Reflection Agents
    "PlanExecuteFlowDiagram": PlanExecuteFlowDiagram,
    "ReflectionLoopVisualizer": ReflectionLoopVisualizer,
    "ErrorRecoveryFlowDiagram": ErrorRecoveryFlowDiagram,
    // Chapter 22: LangSmith Tracing
    "TraceTreeVisualizer": TraceTreeVisualizer,
    "SpanTimelineChart": SpanTimelineChart,
    "TokenUsageBreakdown": TokenUsageBreakdown,
    // Chapter 23: LangSmith Evaluation
    "EvaluationPipeline": EvaluationPipeline,
    "ABTestComparison": ABTestComparison,
    "FeedbackDashboard": FeedbackDashboard,
    // Chapter 24: LangSmith Production Monitoring
    "MonitoringDashboardReal": MonitoringDashboardReal,
    "AlertRuleBuilder": AlertRuleBuilder,
    "CostAnalysisDashboard": CostAnalysisDashboard,
    // Chapter 25: LangServe Basics
    "LangServeArchitecture": LangServeArchitecture,
    "EndpointExplorer": EndpointExplorer,
    "RemoteRunnableDemo": RemoteRunnableDemo,
    // Chapter 26: LangServe Advanced Features
    "AuthenticationFlow": AuthenticationFlow,
    "RateLimitingVisualizer": RateLimitingVisualizer,
    "MetricsDashboard": MetricsDashboard,
    // Chapter 27: Deployment & Containerization
    "DockerBuildFlow": DockerBuildFlow,
    "CloudPlatformComparison": CloudPlatformComparison,
    "K8sArchitectureDiagram": K8sArchitectureDiagram,
    // Chapter 28: Advanced Agent Patterns
    "HumanInLoopFlow": HumanInLoopFlow,
    "LongTermMemoryArchitecture": LongTermMemoryArchitecture,
    "ToolOrchestrationVisualizer": ToolOrchestrationVisualizer,
    // Chapter 29: Ecosystem Integration
    "FrameworkComparisonMatrix": FrameworkComparisonMatrix,
    "MigrationPathGuide": MigrationPathGuide,
    "APIMappingTable": APIMappingTable,
    // Chapter 30: Performance Optimization
    "CachingStrategyComparison": CachingStrategyComparison,
    "CostOptimizationDashboard": CostOptimizationDashboard,
    "ReliabilityDecisionTree": ReliabilityDecisionTree,
    // Chapter 31: Security & Privacy
    "PromptInjectionDefense": PromptInjectionDefense,
    "PIIDetectionFlow": PIIDetectionFlow,
    "SecurityAuditDashboard": SecurityAuditDashboard,
    // Chapter 32: Large-Scale Architecture
    "MicroserviceArchitecture": MicroserviceArchitecture,
    "ModelRoutingFlow": ModelRoutingFlow,
    "ABTestDashboard": ABTestDashboard,
    // Chapter 33: Future Research
    "SpeculativeDecodingFlowLangChain": SpeculativeDecodingFlowLangChain,
    "MultimodalMemoryGraph": MultimodalMemoryGraph,
    "PluginEcosystemMap": PluginEcosystemMap,
    // Reinforcement Learning Components
    "RLEcosystemMap": RLEcosystemMap,
    "AgentEnvironmentLoop": AgentEnvironmentLoop,
    "RLTimelineEvolution": RLTimelineEvolution,
    "MDPGraphVisualizer": MDPGraphVisualizer,
    "BellmanEquationDerivation": BellmanEquationDerivation,
    "ValueFunctionEvolution": ValueFunctionEvolution,
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
