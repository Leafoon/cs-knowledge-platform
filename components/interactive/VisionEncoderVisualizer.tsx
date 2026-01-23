'use client';

import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';

export default function VisionEncoderVisualizer() {
  const [imageSize, setImageSize] = useState(224);
  const [patchSize, setPatchSize] = useState(16);
  const [embedDim, setEmbedDim] = useState(768);
  const [numHeads, setNumHeads] = useState(12);
  const [numLayers, setNumLayers] = useState(12);
  const [showStep, setShowStep] = useState(0);

  const numPatches = useMemo(() => {
    return (imageSize / patchSize) ** 2;
  }, [imageSize, patchSize]);

  const patchesPerRow = imageSize / patchSize;

  const steps = [
    {
      id: 0,
      title: '原始图像',
      description: `输入图像大小为 ${imageSize}×${imageSize}×3（RGB）`,
    },
    {
      id: 1,
      title: 'Patch 切分',
      description: `将图像切分为 ${patchesPerRow}×${patchesPerRow} = ${numPatches} 个 ${patchSize}×${patchSize} 的 patch`,
    },
    {
      id: 2,
      title: '线性投影',
      description: `每个 patch 展平为 ${patchSize * patchSize * 3}-d 向量，通过线性层投影到 ${embedDim}-d`,
    },
    {
      id: 3,
      title: '添加 [CLS] Token',
      description: `添加可学习的 [CLS] token 用于分类，序列长度变为 ${numPatches + 1}`,
    },
    {
      id: 4,
      title: '位置编码',
      description: `添加可学习的位置嵌入（Position Embedding），保留空间信息`,
    },
    {
      id: 5,
      title: 'Transformer Encoder',
      description: `通过 ${numLayers} 层 Transformer，每层包含 ${numHeads} 个注意力头`,
    },
    {
      id: 6,
      title: '输出特征',
      description: `提取 [CLS] token 的输出用于分类，或使用所有 patch 特征进行密集预测`,
    },
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl shadow-lg">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-2">
          🔍 Vision Transformer (ViT) 可视化
        </h3>
        <p className="text-gray-600">
          交互式探索 ViT 如何将图像转换为 Transformer 可处理的 token 序列
        </p>
      </div>

      {/* Parameters */}
      <div className="bg-white p-6 rounded-lg shadow mb-6">
        <h4 className="font-semibold text-gray-800 mb-4">模型参数配置</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Image Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              图像大小: {imageSize}×{imageSize}
            </label>
            <input
              type="range"
              min="112"
              max="384"
              step="56"
              value={imageSize}
              onChange={(e) => setImageSize(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>112</span>
              <span>224</span>
              <span>384</span>
            </div>
          </div>

          {/* Patch Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Patch 大小: {patchSize}×{patchSize}
            </label>
            <input
              type="range"
              min="8"
              max="32"
              step="4"
              value={patchSize}
              onChange={(e) => setPatchSize(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>8</span>
              <span>16</span>
              <span>32</span>
            </div>
          </div>

          {/* Embed Dim */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              嵌入维度: {embedDim}
            </label>
            <input
              type="range"
              min="384"
              max="1024"
              step="128"
              value={embedDim}
              onChange={(e) => setEmbedDim(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>384</span>
              <span>768</span>
              <span>1024</span>
            </div>
          </div>

          {/* Num Heads */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              注意力头数: {numHeads}
            </label>
            <input
              type="range"
              min="4"
              max="16"
              step="4"
              value={numHeads}
              onChange={(e) => setNumHeads(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>4</span>
              <span>12</span>
              <span>16</span>
            </div>
          </div>

          {/* Num Layers */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Transformer 层数: {numLayers}
            </label>
            <input
              type="range"
              min="6"
              max="24"
              step="6"
              value={numLayers}
              onChange={(e) => setNumLayers(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>6</span>
              <span>12</span>
              <span>24</span>
            </div>
          </div>

          {/* Stats */}
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="text-sm font-semibold text-blue-800 mb-2">计算统计</div>
            <div className="space-y-1 text-xs text-blue-700">
              <div>Patch 数量: {numPatches}</div>
              <div>序列长度: {numPatches + 1} (含 [CLS])</div>
              <div>每个 Patch: {patchSize * patchSize * 3}-d</div>
              <div>嵌入维度: {embedDim}-d</div>
            </div>
          </div>
        </div>
      </div>

      {/* Step Navigation */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-gray-800">处理流程</h4>
          <div className="flex gap-2">
            <button
              onClick={() => setShowStep(Math.max(0, showStep - 1))}
              disabled={showStep === 0}
              className="px-3 py-1 bg-gray-200 text-gray-700 rounded disabled:opacity-50 hover:bg-gray-300 transition"
            >
              ← 上一步
            </button>
            <button
              onClick={() => setShowStep(Math.min(steps.length - 1, showStep + 1))}
              disabled={showStep === steps.length - 1}
              className="px-3 py-1 bg-blue-600 text-white rounded disabled:opacity-50 hover:bg-blue-700 transition"
            >
              下一步 →
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="relative h-2 bg-gray-200 rounded-full mb-4">
          <motion.div
            className="absolute h-full bg-blue-600 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${((showStep + 1) / steps.length) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>

        {/* Steps */}
        <div className="flex justify-between mb-4">
          {steps.map((step) => (
            <button
              key={step.id}
              onClick={() => setShowStep(step.id)}
              className={`flex flex-col items-center gap-1 transition ${
                showStep === step.id ? 'opacity-100' : 'opacity-50'
              }`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center font-semibold text-sm ${
                  showStep >= step.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-500'
                }`}
              >
                {step.id + 1}
              </div>
              <span className="text-xs text-gray-600 hidden md:block max-w-[80px] text-center">
                {step.title.split(' ')[0]}
              </span>
            </button>
          ))}
        </div>

        {/* Current Step Info */}
        <div className="bg-blue-50 p-4 rounded-lg">
          <h5 className="font-semibold text-blue-900 mb-2">
            步骤 {showStep + 1}: {steps[showStep].title}
          </h5>
          <p className="text-sm text-blue-800">{steps[showStep].description}</p>
        </div>
      </div>

      {/* Visualization */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-6">
        <div className="flex flex-col items-center gap-6">
          {/* Step 0: Original Image */}
          {showStep === 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative"
            >
              <div
                className="bg-gradient-to-br from-blue-400 to-purple-500 rounded-lg shadow-lg"
                style={{ width: imageSize, height: imageSize }}
              >
                <div className="absolute inset-0 flex items-center justify-center text-white font-bold text-2xl">
                  {imageSize}×{imageSize}×3
                </div>
              </div>
            </motion.div>
          )}

          {/* Step 1: Patch Division */}
          {showStep === 1 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="grid gap-1"
              style={{
                gridTemplateColumns: `repeat(${patchesPerRow}, 1fr)`,
                width: imageSize,
                height: imageSize,
              }}
            >
              {Array.from({ length: numPatches }).map((_, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.01 }}
                  className="bg-gradient-to-br from-blue-400 to-purple-500 rounded flex items-center justify-center"
                  style={{
                    width: patchSize - 2,
                    height: patchSize - 2,
                  }}
                >
                  <span className="text-white text-[8px] font-bold">{idx}</span>
                </motion.div>
              ))}
            </motion.div>
          )}

          {/* Step 2: Linear Projection */}
          {showStep === 2 && (
            <div className="flex items-center gap-8">
              <div className="flex flex-col gap-2">
                <div className="text-sm font-semibold text-gray-700 mb-2">输入 Patch</div>
                {Array.from({ length: Math.min(5, numPatches) }).map((_, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: idx * 0.1 }}
                    className="bg-blue-500 text-white px-4 py-2 rounded font-mono text-xs"
                  >
                    Patch {idx}: {patchSize * patchSize * 3}-d
                  </motion.div>
                ))}
                {numPatches > 5 && (
                  <div className="text-gray-500 text-xs text-center">...</div>
                )}
              </div>

              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.5 }}
                className="bg-purple-600 text-white px-6 py-4 rounded-lg font-semibold"
              >
                Linear
                <div className="text-xs opacity-80">W ∈ ℝ^({patchSize * patchSize * 3}×{embedDim})</div>
              </motion.div>

              <div className="flex flex-col gap-2">
                <div className="text-sm font-semibold text-gray-700 mb-2">嵌入向量</div>
                {Array.from({ length: Math.min(5, numPatches) }).map((_, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ x: 50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: idx * 0.1 + 0.5 }}
                    className="bg-green-500 text-white px-4 py-2 rounded font-mono text-xs"
                  >
                    Embed {idx}: {embedDim}-d
                  </motion.div>
                ))}
                {numPatches > 5 && (
                  <div className="text-gray-500 text-xs text-center">...</div>
                )}
              </div>
            </div>
          )}

          {/* Step 3: Add [CLS] Token */}
          {showStep === 3 && (
            <div className="flex flex-col items-center gap-4">
              <motion.div
                initial={{ y: -30, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="bg-red-500 text-white px-6 py-3 rounded-lg font-bold text-lg"
              >
                [CLS] Token
              </motion.div>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="text-2xl text-gray-400"
              >
                ↓
              </motion.div>
              <div className="flex flex-wrap gap-2 max-w-2xl justify-center">
                <div className="bg-red-500 text-white px-3 py-2 rounded text-xs font-mono">
                  [CLS]
                </div>
                {Array.from({ length: Math.min(10, numPatches) }).map((_, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.5 + idx * 0.05 }}
                    className="bg-green-500 text-white px-3 py-2 rounded text-xs font-mono"
                  >
                    P{idx}
                  </motion.div>
                ))}
                {numPatches > 10 && (
                  <div className="px-3 py-2 text-gray-500 text-xs">
                    ... +{numPatches - 10} patches
                  </div>
                )}
              </div>
              <div className="text-sm text-gray-600 mt-2">
                序列长度: {numPatches + 1}
              </div>
            </div>
          )}

          {/* Step 4: Position Embedding */}
          {showStep === 4 && (
            <div className="flex flex-col items-center gap-6">
              <div className="text-center">
                <div className="text-lg font-semibold text-gray-800 mb-2">
                  Token Embeddings + Position Embeddings
                </div>
                <div className="text-sm text-gray-600">
                  逐元素相加，每个位置有唯一的可学习位置编码
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex flex-col gap-2">
                  <div className="text-xs font-semibold text-gray-600">Token Emb</div>
                  {Array.from({ length: 5 }).map((_, idx) => (
                    <div
                      key={idx}
                      className="bg-green-500 text-white px-4 py-2 rounded text-xs"
                    >
                      {embedDim}-d
                    </div>
                  ))}
                </div>

                <div className="text-3xl text-gray-400">+</div>

                <div className="flex flex-col gap-2">
                  <div className="text-xs font-semibold text-gray-600">Position Emb</div>
                  {Array.from({ length: 5 }).map((_, idx) => (
                    <div
                      key={idx}
                      className="bg-orange-500 text-white px-4 py-2 rounded text-xs"
                    >
                      {embedDim}-d
                    </div>
                  ))}
                </div>

                <div className="text-3xl text-gray-400">=</div>

                <div className="flex flex-col gap-2">
                  <div className="text-xs font-semibold text-gray-600">Final Emb</div>
                  {Array.from({ length: 5 }).map((_, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: idx * 0.1 }}
                      className="bg-blue-600 text-white px-4 py-2 rounded text-xs"
                    >
                      {embedDim}-d
                    </motion.div>
                  ))}
                </div>
              </div>

              <div className="bg-orange-50 p-4 rounded-lg max-w-md">
                <div className="text-sm text-orange-800">
                  💡 <strong>位置编码</strong>：不同于文本的固定位置编码（sin/cos），
                  ViT 使用可学习的位置嵌入，在训练过程中学习空间关系。
                </div>
              </div>
            </div>
          )}

          {/* Step 5: Transformer Encoder */}
          {showStep === 5 && (
            <div className="flex flex-col items-center gap-6">
              <div className="text-center">
                <div className="text-lg font-semibold text-gray-800 mb-2">
                  {numLayers} 层 Transformer Encoder
                </div>
                <div className="text-sm text-gray-600">
                  每层包含 Multi-Head Self-Attention ({numHeads} heads) + Feed-Forward Network
                </div>
              </div>

              <div className="flex flex-col gap-3">
                {Array.from({ length: Math.min(6, numLayers) }).map((_, layerIdx) => (
                  <motion.div
                    key={layerIdx}
                    initial={{ x: -100, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: layerIdx * 0.15 }}
                    className="bg-white border-2 border-purple-300 rounded-lg p-4 w-96"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-semibold text-purple-800">
                        Layer {layerIdx + 1}
                      </span>
                      <span className="text-xs text-gray-500">
                        {numPatches + 1} × {embedDim}
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div className="bg-blue-100 p-2 rounded text-xs">
                        <div className="font-semibold text-blue-800">Multi-Head Attention</div>
                        <div className="text-blue-600">{numHeads} heads × {embedDim / numHeads}-d</div>
                      </div>
                      <div className="bg-green-100 p-2 rounded text-xs">
                        <div className="font-semibold text-green-800">Feed-Forward Network</div>
                        <div className="text-green-600">{embedDim} → {embedDim * 4} → {embedDim}</div>
                      </div>
                    </div>
                  </motion.div>
                ))}
                {numLayers > 6 && (
                  <div className="text-gray-500 text-sm text-center">
                    ... +{numLayers - 6} more layers
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Step 6: Output */}
          {showStep === 6 && (
            <div className="flex flex-col items-center gap-6">
              <div className="text-center">
                <div className="text-lg font-semibold text-gray-800 mb-2">
                  输出特征提取
                </div>
                <div className="text-sm text-gray-600">
                  根据任务选择不同的输出策略
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Classification */}
                <motion.div
                  initial={{ y: 30, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.2 }}
                  className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-lg border-2 border-red-300"
                >
                  <div className="font-semibold text-red-800 mb-3">图像分类</div>
                  <div className="flex flex-col gap-3">
                    <div className="bg-red-500 text-white px-4 py-2 rounded font-mono text-xs text-center">
                      [CLS] Token
                    </div>
                    <div className="text-center text-gray-600">↓</div>
                    <div className="bg-red-600 text-white px-4 py-2 rounded font-mono text-xs text-center">
                      Layer Norm
                    </div>
                    <div className="text-center text-gray-600">↓</div>
                    <div className="bg-red-700 text-white px-4 py-2 rounded font-mono text-xs text-center">
                      Linear({embedDim} → num_classes)
                    </div>
                  </div>
                </motion.div>

                {/* Dense Prediction */}
                <motion.div
                  initial={{ y: 30, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border-2 border-blue-300"
                >
                  <div className="font-semibold text-blue-800 mb-3">密集预测（分割等）</div>
                  <div className="flex flex-col gap-3">
                    <div className="text-xs text-center text-gray-600 mb-2">
                      使用所有 Patch Tokens
                    </div>
                    {Array.from({ length: 4 }).map((_, idx) => (
                      <div
                        key={idx}
                        className="bg-blue-500 text-white px-3 py-1 rounded font-mono text-xs text-center"
                      >
                        Patch {idx}: {embedDim}-d
                      </div>
                    ))}
                    <div className="text-center text-gray-500 text-xs">
                      ... +{numPatches - 4} patches
                    </div>
                  </div>
                </motion.div>
              </div>

              <div className="bg-purple-50 p-4 rounded-lg max-w-2xl">
                <div className="text-sm text-purple-800">
                  💡 <strong>多任务灵活性</strong>：ViT 的输出可用于多种任务。
                  分类任务使用 [CLS] token，而语义分割、目标检测等密集预测任务
                  使用所有 patch tokens 并上采样到原始分辨率。
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Architecture Comparison */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-semibold text-gray-800 mb-4">常见 ViT 变种对比</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-100">
                <th className="px-4 py-2 text-left">模型</th>
                <th className="px-4 py-2 text-left">图像大小</th>
                <th className="px-4 py-2 text-left">Patch 大小</th>
                <th className="px-4 py-2 text-left">嵌入维度</th>
                <th className="px-4 py-2 text-left">层数</th>
                <th className="px-4 py-2 text-left">参数量</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              <tr>
                <td className="px-4 py-2 font-semibold">ViT-B/16</td>
                <td className="px-4 py-2">224×224</td>
                <td className="px-4 py-2">16×16</td>
                <td className="px-4 py-2">768</td>
                <td className="px-4 py-2">12</td>
                <td className="px-4 py-2">86M</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="px-4 py-2 font-semibold">ViT-B/32</td>
                <td className="px-4 py-2">224×224</td>
                <td className="px-4 py-2">32×32</td>
                <td className="px-4 py-2">768</td>
                <td className="px-4 py-2">12</td>
                <td className="px-4 py-2">88M</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-semibold">ViT-L/16</td>
                <td className="px-4 py-2">224×224</td>
                <td className="px-4 py-2">16×16</td>
                <td className="px-4 py-2">1024</td>
                <td className="px-4 py-2">24</td>
                <td className="px-4 py-2">307M</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="px-4 py-2 font-semibold">ViT-H/14</td>
                <td className="px-4 py-2">224×224</td>
                <td className="px-4 py-2">14×14</td>
                <td className="px-4 py-2">1280</td>
                <td className="px-4 py-2">32</td>
                <td className="px-4 py-2">632M</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-4 bg-blue-50 p-4 rounded-lg">
          <div className="text-sm text-blue-800">
            <strong>选择建议</strong>：
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li><strong>ViT-B/32</strong>: 最快，适合快速实验或资源受限场景</li>
              <li><strong>ViT-B/16</strong>: 平衡性能与速度，最常用</li>
              <li><strong>ViT-L/16</strong>: 更高精度，需要更多计算资源</li>
              <li><strong>ViT-H/14</strong>: 最佳性能，适合大规模预训练</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
