"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Tip {
    title: string;
    content: string;
}

interface Quiz {
    question: string;
    answer: boolean;
    explanation: string;
}

interface SectionContent {
    tips: Tip[];
    quizzes: Quiz[];
}

const DEFAULT_CONTENT: SectionContent = {
    tips: [
        { title: "学习建议", content: "多动手写代码，理论结合实践才能真正理解。" },
        { title: "官方文档", content: "遇到问题先查官方文档，Hugging Face 文档非常详细。" }
    ],
    quizzes: [
        { question: "学习 AI 需要大量 GPU 资源吗？", answer: false, explanation: "不一定！现在有很多免费资源（Google Colab、Kaggle）和预训练模型可用。" }
    ]
};

// Map URL hash (section IDs) to content
// IDs come from the headings in markdown, e.g. "chapter-1-tensor" -> "1.1 什么是 Tensor？"
const CONTENT_DB: Record<string, SectionContent> = {
    // ============ PyTorch Chapters ============
    "chapter-0": {
        tips: [
            { title: "环境配置", content: "Conda 是管理 Python 环境的神器，强烈建议为每个项目创建独立的 environment。" },
            { title: "CUDA 版本", content: "安装 PyTorch 时，CUDA 版本必须小于等于你显卡驱动支持的最高版本 (nvidia-smi)。" }
        ],
        quizzes: [
            { question: "Mac M1/M2 可以加速 PyTorch 吗？", answer: true, explanation: "可以！使用 MPS (Metal Performance Shaders) 后端即可加速。" }
        ]
    },
    "chapter-1": {
        tips: [
            { title: "View vs Reshape", content: "tensor.view() 要求内存连续，而 reshape() 则没有此限制。不确定时用 reshape() 更安全。" },
            { title: "广播机制", content: "维度为 1 的轴会自动扩展。小心隐式广播导致的维度错误！" },
            { title: "In-place 操作", content: "像 x.add_() 这样带下划线的方法会直接修改原数据，慎用！Autograd 可能会报错。" }
        ],
        quizzes: [
            { question: "tensor.view() 会发生内存拷贝吗？", answer: false, explanation: "通常不会。它是原存储的'视图'。除非数据不连续强制 contiguous()。" },
            { question: "x * y 是矩阵乘法吗？", answer: false, explanation: "不是！* 是元素级乘法 (Hadamard product)。矩阵乘法用 @ 或 torch.matmul。" }
        ]
    },
    "chapter-2": {
        tips: [
            { title: "梯度累加", content: "默认情况下 .backward() 会累加梯度。常用于变相增大 Batch Size。" },
            { title: "叶子节点", content: "只有 requires_grad=True 的叶子节点 (Leaf Node) 才会保留 .grad 属性。" }
        ],
        quizzes: [
            { question: "optimizer.step() 会清零梯度吗？", answer: false, explanation: "不会！必须手动调用 optimizer.zero_grad()。" },
            { question: "推理时应该用 no_grad 吗？", answer: true, explanation: "是的，这能显著减少显存占用并加速计算。" }
        ]
    },
    "chapter-3": {
        tips: [
            { title: "Module 模式", content: "记得调用 model.eval()！不然 Dropout 和 BatchNorm 会继续更新状态，导致推理结果错误。" },
            { title: "Shape Mismatch", content: "Linear 层的输入特征数必须精确匹配。不知多少层合适？先 print(x.shape) 看看。" }
        ],
        quizzes: [
            { question: "nn.ReLU() 有需要学习的参数吗？", answer: false, explanation: "没有。激活函数通常是无参的。" },
            { question: "forward() 函数能直接调用吗？", answer: false, explanation: "永远不要直接调用 model.forward(x)，请使用 model(x) 以确保护钩子 (Hooks) 正常工作。" }
        ]
    },
    "chapter-4": {
        tips: [
            { title: "Num Workers", content: "Windows 上多进程 DataLoader 经常报错？先把 num_workers 设为 0 试试。" },
            { title: "Collate Fn", content: "处理变长文本或特殊数据结构时，必须重写 collate_fn。" }
        ],
        quizzes: [
            { question: "Dataset 必须把所有图片读到内存吗？", answer: false, explanation: "不需要。通常只存储路径，在 __getitem__ 时才实时读取。" }
        ]
    },
    "chapter-5": {
        tips: [
            { title: "Adam vs SGD", content: "Adam 收敛快但可能掉入局部最优；SGD+Momentum 收敛慢但泛化通常更好。" },
            { title: "NaN Loss", content: "Loss 变成 NaN 了？检查一下是否忘记 zero_grad，或者是学习率太大爆炸了。" }
        ],
        quizzes: [
            { question: "CrossEntropyLoss 需要先手动 Softmax 吗？", answer: false, explanation: "不需要！它内部集成了 LogSoftmax，直接传 Logits 即可。" }
        ]
    },
    
    // ============ Transformers Chapters ============
    "0-introduction": {
        tips: [
            { title: "从 Pipeline 开始", content: "不要直接啃论文！先用 pipeline() 快速上手，理解输入输出，再深入原理。" },
            { title: "Hugging Face Hub", content: "超过 30 万个预训练模型免费使用，不需要从零训练。" }
        ],
        quizzes: [
            { question: "Transformers 库只支持 PyTorch 吗？", answer: false, explanation: "不！同时支持 PyTorch、TensorFlow 和 JAX 三大框架。" }
        ]
    },
    "1-pipeline": {
        tips: [
            { title: "Pipeline 是最佳实践", content: "Pipeline 封装了 Tokenizer、Model、后处理全流程，生产环境也推荐用。" },
            { title: "任务类型", content: "pipeline() 支持 30+ 种任务，文本分类、NER、问答、翻译、图像分类等开箱即用。" }
        ],
        quizzes: [
            { question: "Pipeline 会自动下载模型吗？", answer: true, explanation: "是的！首次使用会自动从 Hub 下载并缓存到 ~/.cache/huggingface。" }
        ]
    },
    "2-auto-classes": {
        tips: [
            { title: "Auto* 类自动识别", content: "AutoModel.from_pretrained() 会根据配置文件自动加载正确的模型类，无需手动指定。" },
            { title: "任务特定模型", content: "AutoModelForSequenceClassification、AutoModelForCausalLM 等比通用 AutoModel 更方便。" }
        ],
        quizzes: [
            { question: "AutoTokenizer 和 AutoModel 必须配对使用吗？", answer: true, explanation: "是的！必须用同一个 checkpoint 的 tokenizer 和 model，否则词表不匹配。" }
        ]
    },
    "3-tokenization": {
        tips: [
            { title: "Tokenization 是关键", content: "90% 的模型错误来自 tokenizer 配置不当。务必理解 padding、truncation、special tokens。" },
            { title: "Batch 处理", content: "tokenizer(..., padding=True, truncation=True) 是标配，避免长度不一致错误。" }
        ],
        quizzes: [
            { question: "BERT 的 [CLS] token 在开头还是结尾？", answer: true, explanation: "在开头！[CLS] text [SEP]。而 GPT 系列是 text <EOS>。" }
        ]
    },
    "4-model-basics": {
        tips: [
            { title: "Logits vs Probabilities", content: "模型输出的是 logits（未归一化），需要 softmax/sigmoid 转为概率。" },
            { title: "Attention Mask", content: "padding token 必须用 attention_mask=0 屏蔽，否则会污染注意力计算。" }
        ],
        quizzes: [
            { question: "所有 Transformer 模型都是双向的吗？", answer: false, explanation: "不！BERT 是双向（Masked LM），GPT 是单向（Causal LM）。" }
        ]
    },
    "5-fine-tuning": {
        tips: [
            { title: "Trainer 是官方推荐", content: "Trainer API 封装了训练循环、评估、保存、日志，比手写 for 循环靠谱得多。" },
            { title: "TrainingArguments", content: "超过 100 个参数可配置，重点关注 learning_rate、batch_size、num_epochs、evaluation_strategy。" }
        ],
        quizzes: [
            { question: "微调时需要冻结部分层吗？", answer: false, explanation: "通常不需要！全参数微调效果最好。显存不足时再考虑冻结底层。" }
        ]
    },
    "6-peft": {
        tips: [
            { title: "PEFT 是趋势", content: "LoRA 只训练 0.1% 参数就能达到全参数微调 99% 的效果，显存降低 3-10 倍！" },
            { title: "LoRA 秩的选择", content: "r=8 适合大多数任务，r=16-32 用于复杂任务。别盲目加大，过拟合风险高。" }
        ],
        quizzes: [
            { question: "LoRA 需要修改原模型权重吗？", answer: false, explanation: "不需要！LoRA 是旁路结构，原模型权重保持冻结，只训练低秩矩阵 A、B。" }
        ]
    },
    "7-quantization": {
        tips: [
            { title: "量化是显存杀手", content: "8-bit 量化显存减半，4-bit（QLoRA）减少 75%，准确率损失 <1%。" },
            { title: "推理 vs 训练量化", content: "推理量化最简单（load_in_8bit=True），训练量化需要 QLoRA + bitsandbytes。" }
        ],
        quizzes: [
            { question: "量化后的模型推理速度一定更快吗？", answer: false, explanation: "不一定！INT8 在 CPU 上快，但部分 GPU（如消费级显卡）反而可能变慢，需实测。" }
        ]
    },
    "8-datasets": {
        tips: [
            { title: "Datasets 库是好帮手", content: "load_dataset('glue', 'mrpc') 一行搞定数据加载，支持流式处理超大数据集。" },
            { title: "数据映射技巧", content: "dataset.map() + batched=True 可以并行处理，速度提升 10 倍+。" }
        ],
        quizzes: [
            { question: "Datasets 会把所有数据加载到内存吗？", answer: false, explanation: "不会！默认用内存映射（Memory-Mapped），只在访问时加载，支持 TB 级数据。" }
        ]
    },
    "9-evaluation": {
        tips: [
            { title: "Metrics 要选对", content: "分类用 Accuracy/F1，生成用 BLEU/ROUGE，别混用！" },
            { title: "Evaluate 库", content: "from evaluate import load; metric = load('accuracy') 比手写靠谱。" }
        ],
        quizzes: [
            { question: "BLEU 分数越高越好吗？", answer: true, explanation: "是的！BLEU 衡量生成文本与参考文本的 n-gram 重叠度，范围 0-100，越高越好。" }
        ]
    },
    "10-generation": {
        tips: [
            { title: "Temperature 控制随机性", content: "temperature=0.7 平衡创造性与连贯性，1.0 更随机，0.1 更确定（近似贪心）。" },
            { title: "Top-k vs Top-p", content: "top_k=50 限制候选词数量，top_p=0.9（nucleus sampling）动态筛选，后者更自然。" }
        ],
        quizzes: [
            { question: "Beam Search 一定比 Greedy 好吗？", answer: false, explanation: "不一定！Beam Search 适合翻译等确定性任务，但创意生成（故事、对话）反而会重复、枯燥。" }
        ]
    },
    "11-mixed-precision": {
        tips: [
            { title: "FP16 是标配", content: "fp16=True 训练速度提升 2-3 倍，显存减半，几乎无损精度（注意 loss scaling）。" },
            { title: "BF16 更稳定", content: "A100/H100 优先用 bf16=True，动态范围更大，无需 loss scaling，更不容易 NaN。" }
        ],
        quizzes: [
            { question: "混合精度训练会降低模型精度吗？", answer: false, explanation: "几乎不会！关键操作（如 LayerNorm、Softmax）仍用 FP32，只有矩阵乘法用 FP16/BF16。" }
        ]
    },
    "12-distributed": {
        tips: [
            { title: "Accelerate 一键分布式", content: "accelerate launch train.py 自动处理 DDP、FSDP、DeepSpeed，无需手写 torch.distributed。" },
            { title: "梯度累积模拟大 Batch", content: "gradient_accumulation_steps=4 让 batch_size=8 等效于 32，省显存。" }
        ],
        quizzes: [
            { question: "8 卡训练速度一定是单卡的 8 倍吗？", answer: false, explanation: "不是！通信开销、数据加载瓶颈会降低效率，实际加速比通常是 6-7 倍。" }
        ]
    },
    "13-gradient-checkpointing": {
        tips: [
            { title: "梯度检查点省显存", content: "gradient_checkpointing=True 可节省 30-50% 显存，代价是训练慢 20-30%。" },
            { title: "显存换时间", content: "训练 70B 模型必备技巧！不检查点的话 A100 80G 都不够。" }
        ],
        quizzes: [
            { question: "梯度检查点会影响模型精度吗？", answer: false, explanation: "不会！它只是重计算激活值来节省显存，数学上完全等价。" }
        ]
    },
    "14-deepspeed": {
        tips: [
            { title: "ZeRO-3 最省显存", content: "优化器状态、梯度、参数三者都分片，千亿模型也能在消费级显卡上训练。" },
            { title: "Offload 到 CPU/NVMe", content: "cpu_offload 可再省 2-3 倍显存，但训练速度会降低。" }
        ],
        quizzes: [
            { question: "DeepSpeed 只能用于超大模型吗？", answer: false, explanation: "不！小模型用 ZeRO-2 也能提速，关键是优化器状态分片降低显存峰值。" }
        ]
    },
    "15-model-compression": {
        tips: [
            { title: "知识蒸馏不只是压缩", content: "Student 模型学习 Teacher 的 soft targets，泛化性往往比单独训练更好！" },
            { title: "剪枝要分层", content: "注意力头、FFN 中间层可以大胆剪，LayerNorm 和输出层要谨慎。" }
        ],
        quizzes: [
            { question: "蒸馏后的模型一定比原模型小吗？", answer: false, explanation: "不一定！蒸馏是知识转移，Teacher 可以小，Student 可以大（例如集成蒸馏）。" }
        ]
    },
    "16-deployment": {
        tips: [
            { title: "ONNX 跨框架利器", content: "PyTorch 模型转 ONNX 后可用 TensorRT、OpenVINO 加速，推理速度提升 3-10 倍。" },
            { title: "批处理提升吞吐", content: "推理时 batch_size=8-32 比单条快得多，延迟稍增但吞吐翻倍。" }
        ],
        quizzes: [
            { question: "模型量化一定要在训练时做吗？", answer: false, explanation: "不！Post-Training Quantization (PTQ) 可以在训练后直接量化，适合已有模型。" }
        ]
    },
    "17-custom-models": {
        tips: [
            { title: "继承 PreTrainedModel", content: "自定义模型继承这个基类才能用 from_pretrained()、save_pretrained() 等便捷方法。" },
            { title: "config.json 是关键", content: "配置文件记录模型超参数，加载时自动读取，别忘了定义 MyModelConfig。" }
        ],
        quizzes: [
            { question: "自定义模型能用 Trainer 训练吗？", answer: true, explanation: "可以！只要输出符合 ModelOutput 格式（含 loss、logits），Trainer 全兼容。" }
        ]
    },
    "18-vllm-tgi": {
        tips: [
            { title: "vLLM 专为生成优化", content: "PagedAttention 机制让 KV Cache 显存利用率接近 100%，吞吐提升 10-20 倍！" },
            { title: "TGI 生产首选", content: "Hugging Face 官方推理服务器，支持流式输出、动态批处理、Safetensors 快速加载。" }
        ],
        quizzes: [
            { question: "vLLM 支持所有 Transformer 模型吗？", answer: false, explanation: "不！主要支持 Decoder-only 模型（GPT、LLaMA），Encoder 模型需用 TGI 或 Optimum。" }
        ]
    },
    "19-speculative-decoding": {
        tips: [
            { title: "小模型当草稿", content: "Draft Model 快速生成候选，Target Model 并行验证，加速 2-3 倍且输出完全一致！" },
            { title: "无损加速", content: "Speculative Decoding 是数学等价的，不改变任何概率分布。" }
        ],
        quizzes: [
            { question: "投机解码需要重新训练模型吗？", answer: false, explanation: "不需要！只需一个小模型（草稿）和大模型（目标），都用现成的即可。" }
        ]
    },
    "20-model-export": {
        tips: [
            { title: "Safetensors 更安全", content: "比 pickle 快 10 倍加载，且不执行任意代码，生产环境必备。" },
            { title: "TorchScript 易出错", content: "动态控制流（if/for）容易失败，优先用 torch.export() 或直接 ONNX。" }
        ],
        quizzes: [
            { question: "ONNX 模型比 PyTorch 模型小吗？", answer: false, explanation: "不！大小类似，但 ONNX 是标准格式，跨平台兼容性更好。" }
        ]
    },
    "21-optimum": {
        tips: [
            { title: "Optimum 是加速工具箱", content: "集成了 ONNX Runtime、TensorRT、OpenVINO，一行代码切换后端。" },
            { title: "BetterTransformer 免费", content: "optimum_model = model.to_bettertransformer() 即可提速 20-50%，无需任何修改。" }
        ],
        quizzes: [
            { question: "Optimum 只支持推理加速吗？", answer: false, explanation: "不！也支持训练（如 ONNX Runtime Training），但主要用于推理。" }
        ]
    },
    "22-api-docker": {
        tips: [
            { title: "FastAPI 简单高效", content: "async def + Pydantic 自动生成 API 文档，比 Flask 性能好 3-5 倍。" },
            { title: "Docker 多阶段构建", content: "builder 阶段装依赖，runtime 阶段只复制必要文件，镜像体积减少 50%+。" }
        ],
        quizzes: [
            { question: "Kubernetes 一定比单机 Docker 好吗？", answer: false, explanation: "不！小规模服务（<10 实例）单机 Docker Compose 更简单，K8s 适合大规模集群。" }
        ]
    },
    "23-attention-deep-dive": {
        tips: [
            { title: "Attention 是核心", content: "Q、K、V 三剑客：Query 问'我要什么'，Key 答'我有什么'，Value 提供'具体内容'。" },
            { title: "Multi-Head 多视角", content: "8-12 个头学习不同模式（语法、语义、位置），concat 后全面理解。" }
        ],
        quizzes: [
            { question: "Self-Attention 的复杂度是 O(n²) 吗？", answer: true, explanation: "是的！序列长度 n 的每个 token 都要和所有 token 计算相似度，所以是 n×n。" }
        ]
    },
    "24-custom-model-dev": {
        tips: [
            { title: "先抄再改", content: "从 modeling_bert.py 等官方代码开始改，别从零写，容易漏掉关键细节（如 tie_weights）。" },
            { title: "单元测试必不可少", content: "测试输入输出 shape、梯度流、序列化加载，避免隐藏 bug。" }
        ],
        quizzes: [
            { question: "自定义 Attention 可以用 Flash Attention 吗？", answer: true, explanation: "可以！torch.nn.functional.scaled_dot_product_attention 会自动调用 Flash Attention v2。" }
        ]
    },
    "25-custom-trainer": {
        tips: [
            { title: "Trainer 钩子强大", content: "on_epoch_end、on_log 等 20+ 钩子，无需重写整个训练循环。" },
            { title: "自定义 Loss", content: "重写 compute_loss() 方法即可，支持多任务、对比学习等复杂场景。" }
        ],
        quizzes: [
            { question: "TrainerCallback 会阻塞训练吗？", answer: false, explanation: "不会！Callback 是异步触发的，不影响训练主循环。" }
        ]
    },
    "26-multimodal": {
        tips: [
            { title: "CLIP 对齐图文", content: "Image Encoder + Text Encoder 投影到同一空间，zero-shot 分类、检索都靠它。" },
            { title: "Vision Transformer 吃 Patch", content: "图像切成 16×16 patch，展平后当 token，和文本 Transformer 完全一样！" }
        ],
        quizzes: [
            { question: "CLIP 需要标注数据吗？", answer: false, explanation: "不需要！从互联网爬取图文对（alt-text）即可训练，4 亿对数据零标注。" }
        ]
    },
    "27-rlhf": {
        tips: [
            { title: "RLHF 三步走", content: "1) 监督微调（SFT），2) 训练奖励模型（RM），3) PPO 优化策略。" },
            { title: "DPO 更简单", content: "跳过 RM 和 PPO，直接从偏好数据优化，效果接近 RLHF 但简单 10 倍！" }
        ],
        quizzes: [
            { question: "RLHF 一定比 SFT 好吗？", answer: false, explanation: "不一定！SFT 已经很强，RLHF 主要改善对话对齐和拒绝有害内容。" }
        ]
    },
    "28-frontier-research": {
        tips: [
            { title: "长上下文是趋势", content: "从 2K → 8K → 128K → 1M tokens，RoPE 插值、稀疏注意力、RAG 组合拳。" },
            { title: "MoE 降低成本", content: "Mixtral 8×7B 只激活 2 个专家，推理成本接近 14B 但性能接近 70B！" }
        ],
        quizzes: [
            { question: "所有 MoE 模型都省显存吗？", answer: false, explanation: "不！训练时所有专家都在显存里，只是推理时激活部分专家。" }
        ]
    },
    "appendix": {
        tips: [
            { title: "调试先看报错", content: "90% 的问题都在报错信息里，仔细读 Traceback，别上来就问 ChatGPT。" },
            { title: "性能基准对比", content: "附录 B 的表格是实测数据，可直接引用到你的论文/报告里。" }
        ],
        quizzes: [
            { question: "OOM 只能减小 batch_size 吗？", answer: false, explanation: "不！还有梯度累积、梯度检查点、量化、offload 等 6 种方法，见附录 A.1。" }
        ]
    }
};

interface ModuleRightSidebarProps {
    currentSection?: string;
}

export function ModuleRightSidebar({ currentSection = "" }: ModuleRightSidebarProps) {
    const [content, setContent] = useState<SectionContent>(DEFAULT_CONTENT);
    const [tipIndex, setTipIndex] = useState(0);
    const [quizIndex, setQuizIndex] = useState(0);
    const [showAnswer, setShowAnswer] = useState<boolean | null>(null);
    const [mounted, setMounted] = useState(false);

    // Detect context based on active ID
    useEffect(() => {
        // Simple matching logic: find the first key that is a substring of currentSection
        // e.g. "chapter-2-autograd" matches "chapter-2"
        const matchedKey = Object.keys(CONTENT_DB).find(key => currentSection.includes(key));

        if (matchedKey) {
            setContent(CONTENT_DB[matchedKey]);
            // Reset indices when chapter changes
            setTipIndex(0);
            setQuizIndex(0);
            setShowAnswer(null);
        }
    }, [currentSection]);

    useEffect(() => {
        setMounted(true);
        const timer = setInterval(() => {
            setTipIndex(i => (i + 1) % content.tips.length);
        }, 10000); // Rotate tips every 10s
        return () => clearInterval(timer);
    }, [content.tips.length]);

    if (!mounted) return null;

    const currentTip = content.tips[tipIndex % content.tips.length];
    const currentQuiz = content.quizzes[quizIndex % content.quizzes.length];

    return (
        <aside className="fixed w-64 space-y-6 pl-4 pt-4">
            {/* 1. Learning Streaks / Status */}
            <div className="bg-bg-elevated/80 backdrop-blur border border-border-subtle rounded-xl p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-text-secondary uppercase tracking-wider">当前状态</span>
                    <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                    </span>
                </div>
                <div className="flex items-center gap-3">
                    <div className="text-2xl font-black text-text-primary">Learning</div>
                    <div className="flex flex-col">
                        <div className="text-xs text-text-tertiary">
                            专注模式开启
                        </div>
                        {currentSection && (
                            <div className="text-[10px] text-accent-primary font-mono truncate w-28">
                                #{currentSection}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* 2. Context-Aware Tips */}
            <div className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 border border-indigo-100 dark:border-indigo-800 rounded-xl p-4 shadow-sm relative overflow-hidden group min-h-[140px]">
                <div className="absolute -right-4 -top-4 w-16 h-16 bg-indigo-200/30 rounded-full blur-xl group-hover:scale-150 transition-transform duration-700" />

                <h4 className="text-xs font-bold text-indigo-600 dark:text-indigo-400 mb-2 flex items-center gap-2">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    {currentTip.title}
                </h4>
                <AnimatePresence mode='wait'>
                    <motion.p
                        key={currentTip.content}
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        className="text-sm text-text-secondary leading-relaxed"
                    >
                        {currentTip.content}
                    </motion.p>
                </AnimatePresence>

                {content.tips.length > 1 && (
                    <div className="absolute bottom-4 left-4 flex gap-1">
                        {content.tips.map((_, i) => (
                            <div key={i} className={`h-1 rounded-full transition-all duration-300 ${i === tipIndex % content.tips.length ? 'w-4 bg-indigo-500' : 'w-1 bg-indigo-200'}`} />
                        ))}
                    </div>
                )}
            </div>

            {/* 3. Context-Aware Mini Quiz */}
            <div className="bg-bg-elevated/80 backdrop-blur border border-border-subtle rounded-xl p-4 shadow-sm">
                <h4 className="text-xs font-bold text-text-secondary uppercase tracking-wider mb-3">
                    Daily Quiz
                </h4>

                <AnimatePresence mode="wait">
                    <motion.p
                        key={currentQuiz.question}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-sm font-medium text-text-primary mb-4"
                    >
                        {currentQuiz.question}
                    </motion.p>
                </AnimatePresence>

                {showAnswer === null ? (
                    <div className="flex gap-2">
                        <button
                            onClick={() => setShowAnswer(true)}
                            className="flex-1 py-1.5 px-3 bg-green-50 hover:bg-green-100 text-green-700 text-xs rounded-lg border border-green-200 transition-colors"
                        >
                            Yes
                        </button>
                        <button
                            onClick={() => setShowAnswer(false)}
                            className="flex-1 py-1.5 px-3 bg-red-50 hover:bg-red-100 text-red-700 text-xs rounded-lg border border-red-200 transition-colors"
                        >
                            No
                        </button>
                    </div>
                ) : (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className={`rounded-lg p-3 text-xs ${showAnswer === currentQuiz.answer
                                ? 'bg-green-50 text-green-800 border border-green-200'
                                : 'bg-red-50 text-red-800 border border-red-200'
                            }`}
                    >
                        <div className="font-bold mb-1">
                            {showAnswer === currentQuiz.answer ? "🎉 Correct!" : "❌ Oops!"}
                        </div>
                        {currentQuiz.explanation}

                        <button
                            onClick={() => {
                                setShowAnswer(null);
                                setQuizIndex(i => (i + 1) % content.quizzes.length);
                            }}
                            className="mt-2 w-full py-1 bg-white/50 hover:bg-white/80 rounded text-center"
                        >
                            Next Question →
                        </button>
                    </motion.div>
                )}
            </div>

            <div className="text-[10px] text-text-tertiary text-center">
                Content Context: {content === DEFAULT_CONTENT ? "General" : "Matched"}
            </div>
        </aside>
    );
}
