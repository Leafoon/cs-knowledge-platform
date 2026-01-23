'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type ArchitectureType = 'clip' | 'blip' | 'llava' | 'flamingo';

interface DataFlow {
  from: string;
  to: string;
  label: string;
  color: string;
}

const architectures = {
  clip: {
    name: 'CLIP (Contrastive Learning)',
    description: '双塔架构，通过对比学习对齐图像和文本',
    components: [
      { id: 'image-input', name: 'Image', type: 'input', x: 50, y: 50, color: 'bg-blue-500' },
      { id: 'text-input', name: 'Text', type: 'input', x: 50, y: 350, color: 'bg-green-500' },
      { id: 'vit', name: 'Vision Transformer\n(ViT)', type: 'encoder', x: 250, y: 50, color: 'bg-blue-600' },
      { id: 'text-encoder', name: 'Text Encoder\n(Transformer)', type: 'encoder', x: 250, y: 350, color: 'bg-green-600' },
      { id: 'image-proj', name: 'Linear\nProjection', type: 'projection', x: 500, y: 50, color: 'bg-blue-700' },
      { id: 'text-proj', name: 'Linear\nProjection', type: 'projection', x: 500, y: 350, color: 'bg-green-700' },
      { id: 'contrastive', name: 'Contrastive Loss\n(InfoNCE)', type: 'loss', x: 650, y: 200, color: 'bg-purple-600' },
    ],
    flows: [
      { from: 'image-input', to: 'vit', label: '3×224×224', color: 'rgb(59, 130, 246)' },
      { from: 'text-input', to: 'text-encoder', label: 'tokens', color: 'rgb(34, 197, 94)' },
      { from: 'vit', to: 'image-proj', label: '768-d', color: 'rgb(59, 130, 246)' },
      { from: 'text-encoder', to: 'text-proj', label: '768-d', color: 'rgb(34, 197, 94)' },
      { from: 'image-proj', to: 'contrastive', label: '512-d (L2 norm)', color: 'rgb(59, 130, 246)' },
      { from: 'text-proj', to: 'contrastive', label: '512-d (L2 norm)', color: 'rgb(34, 197, 94)' },
    ],
    features: [
      '对比学习：正样本对相似度最大化',
      '双塔结构：图像和文本编码器独立',
      'Zero-shot 分类：无需微调直接使用',
      '训练数据：4 亿图像-文本对',
    ],
  },
  blip: {
    name: 'BLIP (Bootstrapping)',
    description: '多任务统一框架，支持图像描述、VQA、检索',
    components: [
      { id: 'image', name: 'Image', type: 'input', x: 50, y: 100, color: 'bg-blue-500' },
      { id: 'text', name: 'Text/Question', type: 'input', x: 50, y: 300, color: 'bg-green-500' },
      { id: 'vit', name: 'ViT\nEncoder', type: 'encoder', x: 220, y: 100, color: 'bg-blue-600' },
      { id: 'text-encoder', name: 'Text\nEncoder', type: 'encoder', x: 220, y: 250, color: 'bg-green-600' },
      { id: 'text-decoder', name: 'Text\nDecoder', type: 'decoder', x: 220, y: 400, color: 'bg-green-700' },
      { id: 'cross-attn', name: 'Cross\nAttention', type: 'attention', x: 420, y: 200, color: 'bg-purple-600' },
      { id: 'itc', name: 'ITC Loss\n(Contrastive)', type: 'loss', x: 600, y: 100, color: 'bg-orange-500' },
      { id: 'itm', name: 'ITM Loss\n(Matching)', type: 'loss', x: 600, y: 250, color: 'bg-orange-600' },
      { id: 'lm', name: 'LM Loss\n(Generation)', type: 'loss', x: 600, y: 400, color: 'bg-orange-700' },
    ],
    flows: [
      { from: 'image', to: 'vit', label: 'patches', color: 'rgb(59, 130, 246)' },
      { from: 'text', to: 'text-encoder', label: 'tokens', color: 'rgb(34, 197, 94)' },
      { from: 'text', to: 'text-decoder', label: 'tokens', color: 'rgb(34, 197, 94)' },
      { from: 'vit', to: 'cross-attn', label: 'image features', color: 'rgb(59, 130, 246)' },
      { from: 'text-encoder', to: 'cross-attn', label: 'text features', color: 'rgb(34, 197, 94)' },
      { from: 'vit', to: 'itc', label: '', color: 'rgb(59, 130, 246)' },
      { from: 'text-encoder', to: 'itc', label: '', color: 'rgb(34, 197, 94)' },
      { from: 'cross-attn', to: 'itm', label: 'fused', color: 'rgb(168, 85, 247)' },
      { from: 'text-decoder', to: 'lm', label: 'generation', color: 'rgb(34, 197, 94)' },
    ],
    features: [
      'ITC：图像-文本对比学习',
      'ITM：图像-文本匹配（二分类）',
      'LM：条件语言生成',
      'CapFilt：自举数据质量提升',
    ],
  },
  llava: {
    name: 'LLaVA (Visual Assistant)',
    description: 'ViT + 线性投影 + LLM，简单高效',
    components: [
      { id: 'image', name: 'Image', type: 'input', x: 50, y: 150, color: 'bg-blue-500' },
      { id: 'text', name: 'Instruction', type: 'input', x: 50, y: 350, color: 'bg-green-500' },
      { id: 'clip-vit', name: 'CLIP-ViT\n(frozen)', type: 'encoder', x: 250, y: 150, color: 'bg-blue-600' },
      { id: 'projection', name: 'Linear\nProjection\n(trainable)', type: 'projection', x: 450, y: 150, color: 'bg-purple-600' },
      { id: 'llm', name: 'LLaMA/Vicuna\n(trainable)', type: 'llm', x: 600, y: 250, color: 'bg-green-700' },
      { id: 'output', name: 'Response', type: 'output', x: 750, y: 250, color: 'bg-orange-500' },
    ],
    flows: [
      { from: 'image', to: 'clip-vit', label: '336×336', color: 'rgb(59, 130, 246)' },
      { from: 'clip-vit', to: 'projection', label: 'ViT features\n1024-d', color: 'rgb(59, 130, 246)' },
      { from: 'projection', to: 'llm', label: 'visual tokens\n4096-d', color: 'rgb(168, 85, 247)' },
      { from: 'text', to: 'llm', label: 'text tokens', color: 'rgb(34, 197, 94)' },
      { from: 'llm', to: 'output', label: 'generated text', color: 'rgb(34, 197, 94)' },
    ],
    features: [
      '两阶段训练：预训练投影层 → 指令微调',
      '冻结 ViT：不更新视觉编码器',
      '简单有效：仅一个线性层对齐',
      'GPT-4 生成训练数据：高质量指令',
    ],
  },
  flamingo: {
    name: 'Flamingo / IDEFICS',
    description: '支持交错图像-文本输入，多图像理解',
    components: [
      { id: 'images', name: 'Images\n(multiple)', type: 'input', x: 50, y: 100, color: 'bg-blue-500' },
      { id: 'text', name: 'Interleaved\nText', type: 'input', x: 50, y: 300, color: 'bg-green-500' },
      { id: 'vit', name: 'Vision\nEncoder', type: 'encoder', x: 220, y: 100, color: 'bg-blue-600' },
      { id: 'perceiver', name: 'Perceiver\nResampler', type: 'resampler', x: 400, y: 100, color: 'bg-purple-500' },
      { id: 'llm', name: 'LLM\n(frozen)', type: 'llm', x: 600, y: 200, color: 'bg-green-700' },
      { id: 'gated-xattn', name: 'Gated Cross\nAttention', type: 'attention', x: 400, y: 300, color: 'bg-purple-600' },
      { id: 'output', name: 'Response', type: 'output', x: 750, y: 200, color: 'bg-orange-500' },
    ],
    flows: [
      { from: 'images', to: 'vit', label: 'N images', color: 'rgb(59, 130, 246)' },
      { from: 'vit', to: 'perceiver', label: 'features', color: 'rgb(59, 130, 246)' },
      { from: 'perceiver', to: 'gated-xattn', label: 'fixed-size\ntokens', color: 'rgb(168, 85, 247)' },
      { from: 'text', to: 'gated-xattn', label: 'text tokens', color: 'rgb(34, 197, 94)' },
      { from: 'gated-xattn', to: 'llm', label: 'fused tokens', color: 'rgb(168, 85, 247)' },
      { from: 'llm', to: 'output', label: 'generation', color: 'rgb(34, 197, 94)' },
    ],
    features: [
      'Perceiver Resampler：压缩视觉特征',
      'Gated Cross-Attention：控制视觉信息注入',
      '支持多图像：处理图像序列',
      'Few-shot 学习：上下文示例',
    ],
  },
};

export default function MultimodalArchitecture() {
  const [selectedArch, setSelectedArch] = useState<ArchitectureType>('clip');
  const [hoveredComponent, setHoveredComponent] = useState<string | null>(null);

  const currentArch = architectures[selectedArch];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-2">
          🎨 多模态架构可视化
        </h3>
        <p className="text-gray-600">
          选择不同的架构，了解视觉-语言模型的工作原理
        </p>
      </div>

      {/* Architecture Selector */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {(Object.keys(architectures) as ArchitectureType[]).map((arch) => (
          <button
            key={arch}
            onClick={() => setSelectedArch(arch)}
            className={`p-4 rounded-lg font-semibold transition-all ${
              selectedArch === arch
                ? 'bg-blue-600 text-white shadow-lg scale-105'
                : 'bg-white text-gray-700 hover:bg-blue-50 hover:shadow'
            }`}
          >
            {architectures[arch].name.split(' ')[0]}
          </button>
        ))}
      </div>

      {/* Architecture Info */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h4 className="text-xl font-bold text-gray-800 mb-2">
          {currentArch.name}
        </h4>
        <p className="text-gray-600 mb-4">{currentArch.description}</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {currentArch.features.map((feature, idx) => (
            <div key={idx} className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">✓</span>
              <span className="text-sm text-gray-700">{feature}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture Diagram */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-6 overflow-x-auto">
        <svg width="850" height="500" className="mx-auto">
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 10 3, 0 6" fill="#6b7280" />
            </marker>
          </defs>

          {/* Data Flows */}
          <AnimatePresence>
            {currentArch.flows.map((flow, idx) => {
              const fromComp = currentArch.components.find(c => c.id === flow.from);
              const toComp = currentArch.components.find(c => c.id === flow.to);
              
              if (!fromComp || !toComp) return null;

              const x1 = fromComp.x + 80;
              const y1 = fromComp.y + 40;
              const x2 = toComp.x;
              const y2 = toComp.y + 40;

              return (
                <g key={`${flow.from}-${flow.to}`}>
                  <motion.line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={flow.color}
                    strokeWidth="2"
                    markerEnd="url(#arrowhead)"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.6 }}
                    transition={{ duration: 0.8, delay: idx * 0.1 }}
                  />
                  {flow.label && (
                    <text
                      x={(x1 + x2) / 2}
                      y={(y1 + y2) / 2 - 10}
                      fontSize="11"
                      fill="#4b5563"
                      textAnchor="middle"
                      className="pointer-events-none"
                    >
                      {flow.label.split('\n').map((line, i) => (
                        <tspan key={i} x={(x1 + x2) / 2} dy={i === 0 ? 0 : 12}>
                          {line}
                        </tspan>
                      ))}
                    </text>
                  )}
                </g>
              );
            })}
          </AnimatePresence>

          {/* Components */}
          {currentArch.components.map((comp, idx) => (
            <motion.g
              key={comp.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: idx * 0.05 }}
              onMouseEnter={() => setHoveredComponent(comp.id)}
              onMouseLeave={() => setHoveredComponent(null)}
              className="cursor-pointer"
            >
              <rect
                x={comp.x}
                y={comp.y}
                width="160"
                height="80"
                rx="8"
                className={`${comp.color} ${
                  hoveredComponent === comp.id ? 'opacity-100' : 'opacity-90'
                }`}
                stroke={hoveredComponent === comp.id ? '#1f2937' : 'none'}
                strokeWidth="2"
              />
              <text
                x={comp.x + 80}
                y={comp.y + 35}
                fontSize="13"
                fontWeight="600"
                fill="white"
                textAnchor="middle"
                className="pointer-events-none"
              >
                {comp.name.split('\n').map((line, i) => (
                  <tspan key={i} x={comp.x + 80} dy={i === 0 ? 0 : 16}>
                    {line}
                  </tspan>
                ))}
              </text>
              
              {/* Type Badge */}
              <rect
                x={comp.x + 5}
                y={comp.y + 5}
                width="50"
                height="18"
                rx="4"
                fill="rgba(0,0,0,0.2)"
              />
              <text
                x={comp.x + 30}
                y={comp.y + 17}
                fontSize="9"
                fill="white"
                textAnchor="middle"
                className="pointer-events-none"
              >
                {comp.type}
              </text>
            </motion.g>
          ))}
        </svg>
      </div>

      {/* Code Example */}
      <div className="bg-gray-900 p-4 rounded-lg shadow">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span className="ml-2 text-gray-400 text-sm">Python</span>
        </div>
        <pre className="text-sm text-gray-300 overflow-x-auto">
          <code>
{selectedArch === 'clip' && `from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Zero-shot 图像分类
inputs = processor(text=["a cat", "a dog"], images=image, return_tensors="pt")
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)`}

{selectedArch === 'blip' && `from transformers import BlipForConditionalGeneration, BlipProcessor

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# 图像描述
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)`}

{selectedArch === 'llava' && `from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 多模态对话
conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image"}]}]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt")
outputs = model.generate(**inputs)`}

{selectedArch === 'flamingo' && `from transformers import IdeficsForVisionText2Text, AutoProcessor

model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")

# 多图像输入
prompts = ["User: What's in image 1?", "<image>", "And image 2?", "<image>", "Assistant:"]
inputs = processor(prompts, images=[img1, img2], return_tensors="pt")
outputs = model.generate(**inputs)`}
          </code>
        </pre>
      </div>

      {/* Legend */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-600 rounded"></div>
          <span className="text-sm text-gray-700">图像处理</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-600 rounded"></div>
          <span className="text-sm text-gray-700">文本处理</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-purple-600 rounded"></div>
          <span className="text-sm text-gray-700">跨模态融合</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-orange-600 rounded"></div>
          <span className="text-sm text-gray-700">输出/损失</span>
        </div>
      </div>
    </div>
  );
}
