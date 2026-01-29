"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

type ScriptMode = 'trace' | 'script'

interface CodeExample {
  title: string
  description: string
  code: string
  traceResult: string
  scriptResult: string
  recommendation: ScriptMode
}

const TorchScriptModeComparison: React.FC = () => {
  const [selectedMode, setSelectedMode] = useState<ScriptMode>('trace')
  const [selectedExample, setSelectedExample] = useState(0)

  const examples: CodeExample[] = [
    {
      title: '简单前向传播',
      description: '无条件分支，固定输入形状',
      code: `class SimpleModel(nn.Module):
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x`,
      traceResult: '✅ 完美支持 - 记录完整计算路径',
      scriptResult: '✅ 支持 - 但不如 trace 优化',
      recommendation: 'trace',
    },
    {
      title: '条件分支',
      description: '包含 if 语句的动态控制流',
      code: `class ConditionalModel(nn.Module):
    def forward(self, x, use_dropout):
        x = self.linear(x)
        if use_dropout:
            x = F.dropout(x, p=0.5)
        return x`,
      traceResult: '❌ 仅记录 use_dropout=True 路径',
      scriptResult: '✅ 完整保留 if/else 逻辑',
      recommendation: 'script',
    },
    {
      title: '循环结构',
      description: '包含 for 循环的迭代计算',
      code: `class LoopModel(nn.Module):
    def forward(self, x, num_layers):
        for i in range(num_layers):
            x = self.layers[i](x)
        return x`,
      traceResult: '❌ 仅记录 num_layers=N 的固定展开',
      scriptResult: '✅ 保留动态循环（需 TorchScript 兼容）',
      recommendation: 'script',
    },
    {
      title: '生成任务',
      description: '自回归生成（动态长度）',
      code: `class GenerativeModel(nn.Module):
    def generate(self, input_ids, max_len):
        for _ in range(max_len):
            logits = self.forward(input_ids)
            next_token = logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids`,
      traceResult: '❌ 无法处理动态序列拼接',
      scriptResult: '⚠️ 需要特殊处理（ONNX 更适合）',
      recommendation: 'script',
    },
  ]

  const currentExample = examples[selectedExample]

  const comparisonTable = [
    { aspect: '使用难度', trace: '简单（一行代码）', script: '中等（需兼容代码）' },
    { aspect: '性能优化', trace: '极致（CUDA Graph）', script: '良好' },
    { aspect: '控制流支持', trace: '❌ 不支持 if/for/while', script: '✅ 完整支持' },
    { aspect: '动态形状', trace: '❌ 固定输入形状', script: '✅ 支持可变形状' },
    { aspect: 'Python 特性', trace: '仅记录张量操作', script: '部分支持（有限制）' },
    { aspect: '调试难度', trace: '容易（错误清晰）', script: '困难（编译错误复杂）' },
  ]

  return (
    <div className="w-full space-y-6 my-8">
      {/* 标题 */}
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">TorchScript: Trace vs Script</h3>
        <p className="text-gray-300">
          两种模式的工作原理与适用场景
        </p>
      </div>

      {/* 模式选择 */}
      <div className="flex gap-4 justify-center">
        <button
          onClick={() => setSelectedMode('trace')}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${
            selectedMode === 'trace'
              ? 'bg-blue-500 text-white shadow-lg scale-105'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-100'
          }`}
        >
          <div className="text-2xl mb-1">🔍</div>
          <div>Trace 模式</div>
          <div className="text-xs opacity-80">记录执行轨迹</div>
        </button>
        <button
          onClick={() => setSelectedMode('script')}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${
            selectedMode === 'script'
              ? 'bg-purple-500 text-white shadow-lg scale-105'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-100'
          }`}
        >
          <div className="text-2xl mb-1">📝</div>
          <div>Script 模式</div>
          <div className="text-xs opacity-80">编译 Python 代码</div>
        </button>
      </div>

      {/* 模式说明 */}
      <motion.div
        key={selectedMode}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg"
      >
        {selectedMode === 'trace' ? (
          <div>
            <h4 className="text-lg font-bold mb-3 flex items-center gap-2">
              <span className="text-2xl">🔍</span>
              Trace 模式工作原理
            </h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg px-3 py-1 text-sm font-mono">
                  Step 1
                </div>
                <p className="text-sm flex-1">
                  提供<strong>示例输入</strong>（example_inputs）
                </p>
              </div>
              <div className="flex items-start gap-3">
                <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg px-3 py-1 text-sm font-mono">
                  Step 2
                </div>
                <p className="text-sm flex-1">
                  执行一次前向传播，<strong>记录所有张量操作</strong>
                </p>
              </div>
              <div className="flex items-start gap-3">
                <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg px-3 py-1 text-sm font-mono">
                  Step 3
                </div>
                <p className="text-sm flex-1">
                  生成<strong>静态计算图</strong>（不包含控制流）
                </p>
              </div>
            </div>
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm font-medium mb-2">代码示例：</p>
              <pre className="text-xs font-mono bg-gray-900 text-gray-100 p-3 rounded overflow-x-auto">
{`traced_model = torch.jit.trace(
    model,
    example_inputs=(input_ids, attention_mask)
)
traced_model.save("model_traced.pt")`}
              </pre>
            </div>
            <div className="mt-4 flex items-start gap-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <span className="text-xl">⚠️</span>
              <p className="text-sm">
                <strong>限制</strong>：无法处理 if、for、while 等控制流，
                仅记录示例输入对应的执行路径
              </p>
            </div>
          </div>
        ) : (
          <div>
            <h4 className="text-lg font-bold mb-3 flex items-center gap-2">
              <span className="text-2xl">📝</span>
              Script 模式工作原理
            </h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg px-3 py-1 text-sm font-mono">
                  Step 1
                </div>
                <p className="text-sm flex-1">
                  分析 Python 源代码（AST）
                </p>
              </div>
              <div className="flex items-start gap-3">
                <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg px-3 py-1 text-sm font-mono">
                  Step 2
                </div>
                <p className="text-sm flex-1">
                  编译为<strong>TorchScript IR</strong>（中间表示）
                </p>
              </div>
              <div className="flex items-start gap-3">
                <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg px-3 py-1 text-sm font-mono">
                  Step 3
                </div>
                <p className="text-sm flex-1">
                  保留控制流逻辑（if、for、while）
                </p>
              </div>
            </div>
            <div className="mt-4 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <p className="text-sm font-medium mb-2">代码示例：</p>
              <pre className="text-xs font-mono bg-gray-900 text-gray-100 p-3 rounded overflow-x-auto">
{`scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")`}
              </pre>
            </div>
            <div className="mt-4 flex items-start gap-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <span className="text-xl">⚠️</span>
              <p className="text-sm">
                <strong>限制</strong>：不支持所有 Python 特性（如列表推导、lambda），
                需要 TorchScript 兼容的代码
              </p>
            </div>
          </div>
        )}
      </motion.div>

      {/* 示例选择 */}
      <div>
        <h4 className="font-semibold mb-3">典型场景对比：</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {examples.map((example, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedExample(idx)}
              className={`p-3 rounded-lg text-left transition-all ${
                selectedExample === idx
                  ? 'bg-indigo-500 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              <div className="text-xs font-medium">{example.title}</div>
            </button>
          ))}
        </div>
      </div>

      {/* 示例详情 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={selectedExample}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg"
        >
          <h4 className="text-lg font-bold mb-2">{currentExample.title}</h4>
          <p className="text-sm text-gray-300 mb-4">
            {currentExample.description}
          </p>

          {/* 代码 */}
          <div className="mb-4">
            <div className="text-sm font-medium mb-2">模型代码：</div>
            <pre className="text-xs font-mono bg-gray-900 text-gray-100 p-4 rounded overflow-x-auto">
              {currentExample.code}
            </pre>
          </div>

          {/* 结果对比 */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-500">
              <div className="font-semibold mb-2 flex items-center gap-2">
                <span>🔍</span>
                Trace 模式
              </div>
              <p className="text-sm">{currentExample.traceResult}</p>
            </div>
            <div className="p-4 rounded-lg bg-purple-50 dark:bg-purple-900/20 border-2 border-purple-500">
              <div className="font-semibold mb-2 flex items-center gap-2">
                <span>📝</span>
                Script 模式
              </div>
              <p className="text-sm">{currentExample.scriptResult}</p>
            </div>
          </div>

          {/* 推荐 */}
          <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg border-2 border-green-500">
            <div className="flex items-center gap-2">
              <span className="text-2xl">💡</span>
              <div>
                <div className="font-semibold">推荐：</div>
                <div className="text-sm">
                  {currentExample.recommendation === 'trace'
                    ? '使用 Trace 模式（性能优异，简单直接）'
                    : '使用 Script 模式（支持动态控制流）'}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* 对比表格 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg overflow-x-auto">
        <h4 className="text-lg font-bold mb-4">详细对比</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-gray-300 dark:border-gray-600">
              <th className="text-left py-3 px-4">对比维度</th>
              <th className="text-left py-3 px-4 bg-blue-50 dark:bg-blue-900/20">
                🔍 Trace 模式
              </th>
              <th className="text-left py-3 px-4 bg-purple-50 dark:bg-purple-900/20">
                📝 Script 模式
              </th>
            </tr>
          </thead>
          <tbody>
            {comparisonTable.map((row, idx) => (
              <tr
                key={idx}
                className="border-b border-gray-200 dark:border-gray-700"
              >
                <td className="py-3 px-4 font-medium">{row.aspect}</td>
                <td className="py-3 px-4 bg-blue-50/50 dark:bg-blue-900/10">
                  {row.trace}
                </td>
                <td className="py-3 px-4 bg-purple-50/50 dark:bg-purple-900/10">
                  {row.script}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 选择建议 */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border-2 border-blue-500">
          <h4 className="font-bold mb-3 flex items-center gap-2">
            <span className="text-2xl">🔍</span>
            选择 Trace 的场景
          </h4>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">✓</span>
              <span>简单前向传播（无 if/for/while）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">✓</span>
              <span>固定输入形状（如分类任务）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">✓</span>
              <span>需要最大性能优化（CUDA Graph）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">✓</span>
              <span>快速原型开发</span>
            </li>
          </ul>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 border-2 border-purple-500">
          <h4 className="font-bold mb-3 flex items-center gap-2">
            <span className="text-2xl">📝</span>
            选择 Script 的场景
          </h4>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">✓</span>
              <span>包含条件分支（if/elif/else）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">✓</span>
              <span>动态循环（for、while）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">✓</span>
              <span>生成任务（可变序列长度）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-1">✓</span>
              <span>需要完整保留模型逻辑</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default TorchScriptModeComparison
