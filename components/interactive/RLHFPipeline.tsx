'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Stage = 'sft' | 'rm' | 'ppo';

interface StageInfo {
  id: Stage;
  name: string;
  description: string;
  color: string;
  steps: string[];
}

const stages: StageInfo[] = [
  {
    id: 'sft',
    name: '阶段 1: 监督微调 (SFT)',
    description: '使用高质量指令-回复对训练模型，学会遵循指令',
    color: 'blue',
    steps: [
      '收集指令数据集（Prompt + Response）',
      '标准语言模型训练（最大化 log 概率）',
      '模型学会遵循指令格式',
      '输出：SFT 模型',
    ],
  },
  {
    id: 'rm',
    name: '阶段 2: 奖励模型 (RM)',
    description: '训练模型学习人类偏好，为回复打分',
    color: 'purple',
    steps: [
      '人工标注偏好对（Preferred vs Rejected）',
      '训练排序模型（Bradley-Terry Model）',
      '损失函数：-log σ(r_w - r_l)',
      '输出：奖励模型',
    ],
  },
  {
    id: 'ppo',
    name: '阶段 3: 强化学习 (PPO)',
    description: '使用奖励模型和 PPO 算法优化策略',
    color: 'green',
    steps: [
      '采样 Prompt，生成回复',
      '奖励模型打分',
      '计算 KL 散度惩罚（防止过度偏离）',
      'PPO 算法更新策略',
      '输出：RLHF 模型',
    ],
  },
];

export default function RLHFPipeline() {
  const [currentStage, setCurrentStage] = useState<Stage>('sft');
  const [isPlaying, setIsPlaying] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    if (isPlaying) {
      const timer = setInterval(() => {
        setStepIndex((prev) => {
          const currentStageInfo = stages.find(s => s.id === currentStage)!;
          if (prev < currentStageInfo.steps.length - 1) {
            return prev + 1;
          } else {
            // 移动到下一个阶段
            const currentIdx = stages.findIndex(s => s.id === currentStage);
            if (currentIdx < stages.length - 1) {
              setCurrentStage(stages[currentIdx + 1].id);
              return 0;
            } else {
              setIsPlaying(false);
              return prev;
            }
          }
        });
      }, 2000);

      return () => clearInterval(timer);
    }
  }, [isPlaying, currentStage]);

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStage('sft');
    setStepIndex(0);
  };

  const getColorClasses = (color: string, variant: 'bg' | 'border' | 'text') => {
    const colors = {
      blue: { bg: 'bg-blue-500', border: 'border-blue-500', text: 'text-blue-700' },
      purple: { bg: 'bg-purple-500', border: 'border-purple-500', text: 'text-purple-700' },
      green: { bg: 'bg-green-500', border: 'border-green-500', text: 'text-green-700' },
    };
    return colors[color as keyof typeof colors][variant];
  };

  const currentStageInfo = stages.find(s => s.id === currentStage)!;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-gray-50 to-blue-50 rounded-xl shadow-lg">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-2">
          🎯 RLHF 三阶段训练流程
        </h3>
        <p className="text-gray-600">
          InstructGPT / ChatGPT 的核心训练方法
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex gap-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-6 py-2 rounded-lg font-semibold transition ${
              isPlaying
                ? 'bg-orange-500 text-white hover:bg-orange-600'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isPlaying ? '⏸ 暂停' : '▶ 自动播放'}
          </button>
          <button
            onClick={handleReset}
            className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 transition"
          >
            🔄 重置
          </button>
        </div>

        <div className="text-sm text-gray-600">
          当前: <strong className={getColorClasses(currentStageInfo.color, 'text')}>
            {currentStageInfo.name}
          </strong>
        </div>
      </div>

      {/* Stage Selector */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {stages.map((stage, idx) => (
          <button
            key={stage.id}
            onClick={() => {
              setCurrentStage(stage.id);
              setStepIndex(0);
              setIsPlaying(false);
            }}
            className={`p-4 rounded-lg border-2 transition ${
              currentStage === stage.id
                ? `${getColorClasses(stage.color, 'border')} ${getColorClasses(stage.color, 'bg')} bg-opacity-10`
                : 'border-gray-200 bg-white hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-3 mb-2">
              <div className={`w-8 h-8 rounded-full ${getColorClasses(stage.color, 'bg')} text-white flex items-center justify-center font-bold`}>
                {idx + 1}
              </div>
              <h4 className="font-semibold text-gray-800 text-left text-sm">
                {stage.name.split(':')[1]}
              </h4>
            </div>
            <p className="text-xs text-gray-600 text-left">
              {stage.description}
            </p>
          </button>
        ))}
      </div>

      {/* Pipeline Visualization */}
      <div className="bg-white p-8 rounded-lg shadow-lg mb-6">
        <div className="flex items-center justify-between mb-8">
          {stages.map((stage, idx) => (
            <React.Fragment key={stage.id}>
              <motion.div
                className={`flex flex-col items-center ${
                  currentStage === stage.id ? 'scale-110' : 'scale-100'
                }`}
                animate={{
                  scale: currentStage === stage.id ? 1.1 : 1,
                }}
                transition={{ duration: 0.3 }}
              >
                <div
                  className={`w-24 h-24 rounded-full ${getColorClasses(stage.color, 'bg')} flex items-center justify-center text-white font-bold text-3xl shadow-lg ${
                    currentStage === stage.id ? 'ring-4 ring-offset-2 ring-blue-300' : ''
                  }`}
                >
                  {idx + 1}
                </div>
                <div className="mt-3 text-center">
                  <div className="font-semibold text-gray-800 text-sm">
                    {stage.name.split(':')[0]}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {stage.name.split(':')[1].trim()}
                  </div>
                </div>
              </motion.div>

              {idx < stages.length - 1 && (
                <motion.div
                  className="flex-1 h-1 bg-gray-300 mx-4"
                  animate={{
                    backgroundColor: stages.findIndex(s => s.id === currentStage) > idx ? '#3b82f6' : '#d1d5db',
                  }}
                  transition={{ duration: 0.5 }}
                >
                  <motion.div
                    className="h-full bg-blue-600"
                    initial={{ width: 0 }}
                    animate={{
                      width: stages.findIndex(s => s.id === currentStage) > idx ? '100%' : '0%',
                    }}
                    transition={{ duration: 0.5 }}
                  />
                </motion.div>
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Step Details */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStage}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className={`p-6 rounded-lg border-2 ${getColorClasses(currentStageInfo.color, 'border')} bg-opacity-5`}
            style={{ backgroundColor: `${getColorClasses(currentStageInfo.color, 'bg').replace('bg-', '')}10` }}
          >
            <h4 className="text-xl font-bold text-gray-800 mb-4">
              {currentStageInfo.name}
            </h4>
            <p className="text-gray-600 mb-6">{currentStageInfo.description}</p>

            <div className="space-y-3">
              {currentStageInfo.steps.map((step, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0.3, x: -20 }}
                  animate={{
                    opacity: idx <= stepIndex ? 1 : 0.3,
                    x: idx <= stepIndex ? 0 : -20,
                  }}
                  transition={{ duration: 0.3, delay: idx * 0.1 }}
                  className={`flex items-start gap-3 p-3 rounded-lg ${
                    idx <= stepIndex
                      ? `${getColorClasses(currentStageInfo.color, 'bg')} bg-opacity-10 border-2 ${getColorClasses(currentStageInfo.color, 'border')}`
                      : 'bg-gray-100'
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                      idx <= stepIndex
                        ? `${getColorClasses(currentStageInfo.color, 'bg')} text-white`
                        : 'bg-gray-300 text-gray-600'
                    }`}
                  >
                    {idx + 1}
                  </div>
                  <div className="flex-1 pt-1">
                    <p className={`text-sm ${idx <= stepIndex ? 'text-gray-800 font-medium' : 'text-gray-500'}`}>
                      {step}
                    </p>
                  </div>
                  {idx <= stepIndex && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="text-green-500 text-xl"
                    >
                      ✓
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Code Example */}
      <div className="bg-gray-900 p-4 rounded-lg shadow mb-6">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span className="ml-2 text-gray-400 text-sm">Python - {currentStageInfo.name}</span>
        </div>
        <pre className="text-sm text-gray-300 overflow-x-auto">
          <code>
{currentStage === 'sft' && `from trl import SFTTrainer
from transformers import AutoModelForCausalLM, TrainingArguments

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=instruction_dataset,  # 指令数据
    dataset_text_field="text",
    max_seq_length=512,
    packing=True,  # 打包短样本
)

trainer.train()
trainer.save_model("./llama2-sft")`}

{currentStage === 'rm' && `from trl import RewardTrainer
from transformers import AutoModelForSequenceClassification

# 基于 SFT 模型训练奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "llama2-sft",
    num_labels=1  # 输出单个奖励分数
)

# 训练
trainer = RewardTrainer(
    model=reward_model,
    train_dataset=preference_dataset,  # {"chosen": ..., "rejected": ...}
    max_length=512,
)

trainer.train()
trainer.save_model("./reward_model")`}

{currentStage === 'ppo' && `from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 加载策略模型（带 Value Head）
model = AutoModelForCausalLMWithValueHead.from_pretrained("llama2-sft")

# PPO 配置
config = PPOConfig(
    learning_rate=1.4e-5,
    init_kl_coef=0.2,  # KL 惩罚系数
    target_kl=6.0,
)

# 训练
ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

for batch in ppo_trainer.dataloader:
    # 生成回复
    responses = ppo_trainer.generate(batch["input_ids"])
    
    # 奖励模型打分
    rewards = [reward_model(r).item() for r in responses]
    
    # PPO 更新
    ppo_trainer.step(batch["input_ids"], responses, rewards)`}
          </code>
        </pre>
      </div>

      {/* Key Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg border-2 border-blue-200">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-2xl">📚</span>
            <h5 className="font-semibold text-blue-900">SFT 要点</h5>
          </div>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• 数据质量 &gt; 数量</li>
            <li>• 多样化指令覆盖</li>
            <li>• 避免过拟合</li>
          </ul>
        </div>

        <div className="bg-purple-50 p-4 rounded-lg border-2 border-purple-200">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-2xl">🎯</span>
            <h5 className="font-semibold text-purple-900">RM 要点</h5>
          </div>
          <ul className="text-sm text-purple-800 space-y-1">
            <li>• 标注者一致性</li>
            <li>• 偏好数据多样性</li>
            <li>• 防止长度偏好</li>
          </ul>
        </div>

        <div className="bg-green-50 p-4 rounded-lg border-2 border-green-200">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-2xl">⚙️</span>
            <h5 className="font-semibold text-green-900">PPO 要点</h5>
          </div>
          <ul className="text-sm text-green-800 space-y-1">
            <li>• KL 惩罚防崩溃</li>
            <li>• 小学习率</li>
            <li>• 监控奖励漂移</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
