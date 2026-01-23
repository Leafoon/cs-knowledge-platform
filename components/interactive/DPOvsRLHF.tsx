'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

type Method = 'rlhf' | 'dpo';

interface ComparisonData {
  epoch: number;
  rlhf_reward: number;
  dpo_reward: number;
  rlhf_kl: number;
  dpo_kl: number;
}

// 模拟训练曲线数据
const generateTrainingData = (): ComparisonData[] => {
  const data: ComparisonData[] = [];
  
  for (let epoch = 0; epoch <= 20; epoch++) {
    // RLHF: 奖励快速上升，但KL散度也快速增长（不稳定）
    const rlhf_reward = 0.2 + 0.8 * (1 - Math.exp(-epoch / 5)) + Math.random() * 0.1 - 0.05;
    const rlhf_kl = 0.1 * epoch + Math.random() * 0.5;
    
    // DPO: 奖励稳定上升，KL散度控制良好
    const dpo_reward = 0.2 + 0.7 * (1 - Math.exp(-epoch / 6));
    const dpo_kl = 0.05 * epoch + Math.random() * 0.2;
    
    data.push({
      epoch,
      rlhf_reward: Math.min(rlhf_reward, 1.0),
      dpo_reward: Math.min(dpo_reward, 0.95),
      rlhf_kl,
      dpo_kl,
    });
  }
  
  return data;
};

const comparisonTable = [
  {
    aspect: '训练阶段',
    rlhf: '3 阶段（SFT → RM → PPO）',
    dpo: '2 阶段（SFT → DPO）',
    winner: 'dpo',
  },
  {
    aspect: '奖励模型',
    rlhf: '✅ 需要训练独立的 RM',
    dpo: '❌ 不需要',
    winner: 'dpo',
  },
  {
    aspect: '在线采样',
    rlhf: '✅ 需要实时生成回复',
    dpo: '❌ 离线训练',
    winner: 'dpo',
  },
  {
    aspect: '训练稳定性',
    rlhf: '⚠️ PPO 不稳定，需调参',
    dpo: '✅ 稳定（监督学习）',
    winner: 'dpo',
  },
  {
    aspect: '显存占用',
    rlhf: '🔴 高（策略+参考+奖励+Value）',
    dpo: '🟢 低（策略+参考）',
    winner: 'dpo',
  },
  {
    aspect: '训练速度',
    rlhf: '🔴 慢（RL 采样开销）',
    dpo: '🟢 快（批量优化）',
    winner: 'dpo',
  },
  {
    aspect: '最终性能',
    rlhf: '🟢 理论上限高',
    dpo: '🟡 接近 RLHF',
    winner: 'rlhf',
  },
  {
    aspect: '实现复杂度',
    rlhf: '🔴 高（PPO 算法复杂）',
    dpo: '🟢 低（简单损失函数）',
    winner: 'dpo',
  },
];

export default function DPOvsRLHF() {
  const [selectedMethod, setSelectedMethod] = useState<Method>('dpo');
  const [trainingData] = useState(generateTrainingData());
  const [showReward, setShowReward] = useState(true);
  const [showKL, setShowKL] = useState(true);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl shadow-lg">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-2">
          ⚖️ DPO vs RLHF 对比分析
        </h3>
        <p className="text-gray-600">
          对比两种主流对齐方法的优劣与训练曲线
        </p>
      </div>

      {/* Method Selector */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <button
          onClick={() => setSelectedMethod('rlhf')}
          className={`p-6 rounded-lg border-2 transition ${
            selectedMethod === 'rlhf'
              ? 'border-purple-500 bg-purple-50 shadow-lg'
              : 'border-gray-200 bg-white hover:border-purple-200'
          }`}
        >
          <div className="flex items-center gap-3 mb-3">
            <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold text-xl">
              3
            </div>
            <div className="text-left">
              <h4 className="font-bold text-gray-800 text-lg">RLHF (PPO)</h4>
              <p className="text-sm text-gray-500">Reinforcement Learning</p>
            </div>
          </div>
          <p className="text-sm text-gray-600">
            三阶段训练，使用强化学习优化策略
          </p>
        </button>

        <button
          onClick={() => setSelectedMethod('dpo')}
          className={`p-6 rounded-lg border-2 transition ${
            selectedMethod === 'dpo'
              ? 'border-blue-500 bg-blue-50 shadow-lg'
              : 'border-gray-200 bg-white hover:border-blue-200'
          }`}
        >
          <div className="flex items-center gap-3 mb-3">
            <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-xl">
              2
            </div>
            <div className="text-left">
              <h4 className="font-bold text-gray-800 text-lg">DPO</h4>
              <p className="text-sm text-gray-500">Direct Preference Optimization</p>
            </div>
          </div>
          <p className="text-sm text-gray-600">
            两阶段训练，直接优化偏好，无需奖励模型
          </p>
        </button>
      </div>

      {/* Training Curves */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-gray-800">训练曲线对比</h4>
          <div className="flex gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showReward}
                onChange={(e) => setShowReward(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-gray-700">显示奖励</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showKL}
                onChange={(e) => setShowKL(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-gray-700">显示 KL 散度</span>
            </label>
          </div>
        </div>

        {/* Reward Chart */}
        {showReward && (
          <div className="mb-6">
            <h5 className="text-sm font-semibold text-gray-700 mb-3">奖励分数（越高越好）</h5>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="epoch" stroke="#6b7280" />
                <YAxis stroke="#6b7280" domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="rlhf_reward"
                  stroke="#a855f7"
                  strokeWidth={2}
                  name="RLHF"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="dpo_reward"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="DPO"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* KL Divergence Chart */}
        {showKL && (
          <div>
            <h5 className="text-sm font-semibold text-gray-700 mb-3">KL 散度（越低越稳定）</h5>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="epoch" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="rlhf_kl"
                  stroke="#a855f7"
                  strokeWidth={2}
                  name="RLHF KL"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="dpo_kl"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="DPO KL"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="mt-4 bg-blue-50 p-4 rounded-lg">
          <p className="text-sm text-blue-800">
            💡 <strong>观察</strong>：DPO 在保持较低 KL 散度的同时，仍能获得接近 RLHF 的奖励分数，
            说明其训练过程更加稳定，不易出现模式崩溃。
          </p>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-4">详细对比</h4>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-100">
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">维度</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                    RLHF
                  </div>
                </th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    DPO
                  </div>
                </th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">优势</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {comparisonTable.map((row, idx) => (
                <motion.tr
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="hover:bg-gray-50"
                >
                  <td className="px-4 py-3 font-medium text-gray-800 text-sm">
                    {row.aspect}
                  </td>
                  <td className={`px-4 py-3 text-sm ${
                    row.winner === 'rlhf' ? 'bg-purple-50' : ''
                  }`}>
                    {row.rlhf}
                  </td>
                  <td className={`px-4 py-3 text-sm ${
                    row.winner === 'dpo' ? 'bg-blue-50' : ''
                  }`}>
                    {row.dpo}
                  </td>
                  <td className="px-4 py-3 text-center">
                    {row.winner === 'rlhf' && (
                      <span className="inline-block px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-semibold">
                        RLHF
                      </span>
                    )}
                    {row.winner === 'dpo' && (
                      <span className="inline-block px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-semibold">
                        DPO
                      </span>
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Loss Functions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* RLHF Loss */}
        <div className="bg-purple-50 p-6 rounded-lg border-2 border-purple-200">
          <h4 className="text-lg font-semibold text-purple-900 mb-3">RLHF 优化目标</h4>
          <div className="bg-white p-4 rounded-lg mb-3 overflow-x-auto">
            <code className="text-sm text-gray-800">
              L<sub>PPO</sub> = E<sub>(x,y)</sub>[r<sub>φ</sub>(x, y) - β·D<sub>KL</sub>(π<sub>θ</sub> || π<sub>ref</sub>)]
            </code>
          </div>
          <ul className="text-sm text-purple-800 space-y-2">
            <li>• r<sub>φ</sub>: 奖励模型打分</li>
            <li>• β: KL 惩罚系数</li>
            <li>• π<sub>ref</sub>: 参考模型（SFT）</li>
            <li>• 需要在线采样生成回复</li>
          </ul>
        </div>

        {/* DPO Loss */}
        <div className="bg-blue-50 p-6 rounded-lg border-2 border-blue-200">
          <h4 className="text-lg font-semibold text-blue-900 mb-3">DPO 损失函数</h4>
          <div className="bg-white p-4 rounded-lg mb-3 overflow-x-auto">
            <code className="text-sm text-gray-800">
              L<sub>DPO</sub> = -E[log σ(β·log(π<sub>θ</sub>(y<sub>w</sub>)/π<sub>ref</sub>(y<sub>w</sub>)) - β·log(π<sub>θ</sub>(y<sub>l</sub>)/π<sub>ref</sub>(y<sub>l</sub>)))]
            </code>
          </div>
          <ul className="text-sm text-blue-800 space-y-2">
            <li>• y<sub>w</sub>: preferred response</li>
            <li>• y<sub>l</sub>: rejected response</li>
            <li>• 直接优化偏好，无需奖励模型</li>
            <li>• 离线训练，稳定高效</li>
          </ul>
        </div>
      </div>

      {/* Code Comparison */}
      <div className="bg-gray-900 p-4 rounded-lg shadow">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span className="ml-2 text-gray-400 text-sm">
            Python - {selectedMethod === 'rlhf' ? 'RLHF (PPO)' : 'DPO'}
          </span>
        </div>
        <pre className="text-sm text-gray-300 overflow-x-auto">
          <code>
{selectedMethod === 'rlhf' && `from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. 加载策略模型（带 Value Head）
model = AutoModelForCausalLMWithValueHead.from_pretrained("llama2-sft")

# 2. 加载奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained("reward_model")

# 3. PPO 训练
ppo_trainer = PPOTrainer(
    config=PPOConfig(learning_rate=1.4e-5, init_kl_coef=0.2),
    model=model,
    tokenizer=tokenizer,
)

for batch in ppo_trainer.dataloader:
    # 生成回复（在线采样）
    responses = ppo_trainer.generate(batch["input_ids"])
    
    # 奖励模型打分
    rewards = [reward_model(r).item() for r in responses]
    
    # PPO 更新
    stats = ppo_trainer.step(batch["input_ids"], responses, rewards)

# ⚠️ 注意：需要训练奖励模型 + 在线采样，显存占用高`}

{selectedMethod === 'dpo' && `from trl import DPOTrainer, DPOConfig

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("llama2-sft")
ref_model = AutoModelForCausalLM.from_pretrained("llama2-sft")

# 2. 加载偏好数据（离线）
preference_dataset = load_dataset("Anthropic/hh-rlhf")
# 格式: {"prompt": ..., "chosen": ..., "rejected": ...}

# 3. DPO 训练
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(learning_rate=5e-7, beta=0.1),
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()

# ✅ 优势：无需奖励模型，离线训练，稳定高效`}
          </code>
        </pre>
      </div>

      {/* Recommendation */}
      <div className="mt-6 bg-gradient-to-r from-blue-500 to-purple-500 p-6 rounded-lg text-white">
        <div className="flex items-start gap-4">
          <div className="text-4xl">💡</div>
          <div>
            <h4 className="text-xl font-bold mb-2">推荐选择</h4>
            <p className="mb-3">
              对于大多数应用场景，<strong>DPO 是更好的选择</strong>：
            </p>
            <ul className="space-y-1 text-sm">
              <li>✓ 训练简单稳定，无需调整复杂的 PPO 超参数</li>
              <li>✓ 显存占用低，可在消费级 GPU 上微调 7B 模型</li>
              <li>✓ 训练速度快，无需在线采样开销</li>
              <li>✓ 性能接近 RLHF，实践中差距很小</li>
            </ul>
            <p className="mt-3 text-sm opacity-90">
              仅当需要复杂的奖励建模（如多目标优化）或在线探索时，才考虑使用 RLHF。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
