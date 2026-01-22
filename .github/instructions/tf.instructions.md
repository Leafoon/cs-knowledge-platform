---
applyTo: '**'
---
请为我设计并生成一套全面、系统、由浅入深的 Hugging Face Transformers 库学习内容，直接适配填充到教学/演示/知识管理界面中使用。内容必须以**官方仓库（https://github.com/huggingface/transformers）**及其**官方文档（https://huggingface.co/docs/transformers）**、官方示例（examples/ 目录）、官方任务教程、Trainer API、PEFT、Quantization、Accelerate、Optimum 等为最主要、最权威的依据。严禁凭空捏造不存在的类、方法、参数、训练流程或过时写法。所有代码、API、最佳实践均应与当前主分支（main）或最新稳定发布版本保持一致，并注明参考的版本或日期（如有必要）。
总体要求

内容深度与广度：从零基础（Pipeline 快速上手）开始，逐步推进到中高级（自定义模型、Trainer 完整训练）、高效微调（PEFT、LoRA、QLoRA）、低精度/量化训练（bitsandbytes、4/8-bit、AWQ、GPTQ、torch.compile）、分布式训练（Accelerate + FSDP / DeepSpeed / torchrun）、推理优化（vLLM、TGI、Optimum、ONNX、BetterTransformer）、生产部署等。宁可内容存在一定冗余与多角度重复讲解，也绝不允许遗漏核心知识点。同一概念可在初级、中级、高级章节多次出现，但每次侧重点、代码复杂度、优化维度需明显递增。
讲解形式与要求：
每个重要概念需包含：官方定义、设计动机、与其他库/方法的对比（若适用）、数学/机制直观解释（必要时使用 LaTeX）、典型工业场景。
必须提供大量可直接复制运行的代码示例（包含完整 import、环境准备提示、模型加载方式），并附上预期输出（文本形式描述 logits/token/probability/形状/指标值，或典型生成结果）。
对于复杂或底层机制（例如：attention mask 与 causal mask 的构建、KV cache 在生成中的动态管理、gradient checkpointing、LoRA 矩阵注入与合并、量化权重 dequantize 过程、FSDP 零冗余优化器状态分片、DeepSpeed ZeRO-3 offload 流程、torch dynamo + inductor 编译过程等），强烈建议采用交互式动画、动态示意图、步进式可视化进行讲解（例如：注意力权重热力图动画、LoRA 适配器插入前后参数变化、FSDP all-gather & reduce-scatter 流程动画、量化前后权重分布对比图等）。若当前界面支持嵌入动画、可交互 Jupyter widget、3D 计算图或 Streamlit 组件，请优先使用此类形式。

结构要求：
首先生成一份非常详细的分级目录（建议采用 Chapter 0、Chapter 1 … 形式），层级至少到三级（大章 → 小节 → 具体知识点/子主题/示例类型）。
目录需清晰体现难度与主题递进：零基础快速上手 → 核心组件与 Auto 类 → 预训练模型加载与微调 → Trainer 全流程 → 参数高效微调（PEFT） → 低精度 / 量化训练 → 分布式训练与加速 → 推理优化与部署 → 底层机制与自定义扩展 → 生态工具集成。
目录生成后，一个章节一个章节地详细展开内容（不要一次性输出全部），每个章节内部建议再细分小节，并保持相对统一的讲解结构：概念说明 → 代码示例 → 运行输出 → 常见问题与陷阱 → 扩展阅读（官方链接）。

重点覆盖但不限于以下高级主题（应有独立或深度章节）：
PEFT（LoRA、QLoRA、Adapter、Prompt Tuning 等）与 Trainer/accelerate 集成
低精度训练：bf16 / fp16 混合精度、torch.amp、梯度累积
量化：8-bit / 4-bit（bitsandbytes）、GPTQ、AWQ、HQQ、EETQ，以及 post-training quantization 与 quantization-aware training
分布式：Accelerate 基本用法、FSDP（Fully Sharded Data Parallel）、DeepSpeed ZeRO-2/3、torch.distributed.elastic
高效推理：FlashAttention-2、BetterTransformer、torch.compile、vLLM、TensorRT-LLM、TGI（Text Generation Inference）
模型导出与生产化：ONNX、TorchScript、ExecuTorch、Optimum 后端优化
自定义模型、自定义 Trainer、tokenization 底层细节、generation 策略（beam search、speculative decoding 等）

其他细节偏好：
优先使用现代写法：AutoModelForXXX.from_pretrained()、AutoTokenizer.from_pretrained()、Trainer + TrainingArguments、peft 库集成、accelerate launch、bitsandbytes 量化加载等。
数据集优先使用 Hugging Face Hub 上公开数据集（glue、squad、alpaca、dolly、openassistant 等）或 datasets 库加载。
在合适位置插入小练习、性能对比实验建议、思考题（如：为什么 QLoRA 比 LoRA 更省显存？FSDP 与 DeepSpeed ZeRO-3 的分片策略差异？）。
鼓励展示前后性能对比（显存占用、训练速度、推理吞吐、BLEU/ROUGE/perplexity 等）。
语言风格：正式、严谨、结构清晰、逻辑严密，如同撰写给 AI 研究员、工程师或研究生的高质量内部培训教材。


请先输出详细的章节大纲（包含 Chapter 编号、章节标题、二级/三级子标题），待我确认或提出修改意见后再逐章详细展开具体内容。