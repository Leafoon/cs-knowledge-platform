'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Code2, Zap, ArrowRight } from 'lucide-react';

const legacyCode = `from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 旧式 LLMChain
prompt = ChatPromptTemplate.from_template(
    "Translate to {language}: {text}"
)
model = ChatOpenAI(model="gpt-4o-mini")

chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True
)

# 调用
result = chain.run(
    language="French",
    text="Hello"
)`;

const lcelCode = `from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LCEL 管道
prompt = ChatPromptTemplate.from_template(
    "Translate to {language}: {text}"
)
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 用管道操作符组合
chain = prompt | model | parser

# 调用
result = chain.invoke({
    "language": "French",
    "text": "Hello"
})`;

const comparisonData = [
  {
    feature: '语法',
    legacy: '类实例化',
    lcel: '管道操作符 |',
    advantage: 'lcel'
  },
  {
    feature: '类型推断',
    legacy: '弱',
    lcel: '强（IDE 支持）',
    advantage: 'lcel'
  },
  {
    feature: '流式支持',
    legacy: '部分支持',
    lcel: '原生支持',
    advantage: 'lcel'
  },
  {
    feature: '并行执行',
    legacy: '需手动实现',
    lcel: 'RunnableParallel',
    advantage: 'lcel'
  },
  {
    feature: '调试',
    legacy: 'verbose=True',
    lcel: 'get_graph()、LangSmith',
    advantage: 'lcel'
  },
  {
    feature: '性能',
    legacy: '一般',
    lcel: '优化的执行引擎',
    advantage: 'lcel'
  }
];

export default function LegacyVsLCELComparison() {
  const [activeTab, setActiveTab] = useState<'legacy' | 'lcel'>('legacy');

  return (
    <div className="w-full bg-gradient-to-br from-slate-900 via-purple-900/20 to-slate-900 rounded-2xl p-8 shadow-2xl">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-white mb-2">
          Legacy Chain vs LCEL 对比
        </h3>
        <p className="text-slate-400">
          理解为什么迁移到 LCEL 是最佳实践
        </p>
      </div>

      {/* 代码对比 */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* Legacy Chain */}
        <div className="bg-slate-800 rounded-xl overflow-hidden border-2 border-red-500/30">
          <div className="bg-gradient-to-r from-red-600 to-orange-600 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Code2 className="w-5 h-5 text-white" />
              <span className="font-semibold text-white">Legacy Chain</span>
            </div>
            <span className="px-3 py-1 bg-red-900/50 text-red-200 text-xs rounded-full">
              已废弃
            </span>
          </div>
          <div className="p-4">
            <pre className="text-sm font-mono text-slate-300 overflow-x-auto whitespace-pre">
              {legacyCode}
            </pre>
          </div>
          <div className="px-4 pb-4">
            <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
              <div className="text-red-400 text-sm font-semibold mb-2">问题：</div>
              <ul className="text-red-300 text-xs space-y-1">
                <li>• 需要记忆不同 Chain 类的 API</li>
                <li>• 组合复杂链时代码冗长</li>
                <li>• 类型推断困难</li>
                <li>• 性能优化受限</li>
              </ul>
            </div>
          </div>
        </div>

        {/* LCEL */}
        <div className="bg-slate-800 rounded-xl overflow-hidden border-2 border-green-500/30">
          <div className="bg-gradient-to-r from-green-600 to-emerald-600 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-white" />
              <span className="font-semibold text-white">LCEL</span>
            </div>
            <span className="px-3 py-1 bg-green-900/50 text-green-200 text-xs rounded-full">
              推荐使用
            </span>
          </div>
          <div className="p-4">
            <pre className="text-sm font-mono text-slate-300 overflow-x-auto whitespace-pre">
              {lcelCode}
            </pre>
          </div>
          <div className="px-4 pb-4">
            <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-3">
              <div className="text-green-400 text-sm font-semibold mb-2">优势：</div>
              <ul className="text-green-300 text-xs space-y-1">
                <li>• 简洁的管道语法</li>
                <li>• 强类型推断与 IDE 支持</li>
                <li>• 原生流式支持</li>
                <li>• 更好的性能</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* 执行流程对比 */}
      <div className="bg-slate-800 rounded-xl p-6 mb-8">
        <h4 className="text-lg font-semibold text-white mb-4">执行流程对比</h4>
        
        <div className="grid md:grid-cols-2 gap-8">
          {/* Legacy 流程 */}
          <div>
            <div className="text-sm font-medium text-red-400 mb-3">Legacy Chain</div>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-red-600 text-white flex items-center justify-center text-sm font-bold">
                  1
                </div>
                <div className="flex-1 p-3 bg-red-900/20 rounded-lg border border-red-500/30">
                  <div className="text-white text-sm">创建 LLMChain 实例</div>
                  <div className="text-red-300 text-xs">需要显式传递参数</div>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-red-600 text-white flex items-center justify-center text-sm font-bold">
                  2
                </div>
                <div className="flex-1 p-3 bg-red-900/20 rounded-lg border border-red-500/30">
                  <div className="text-white text-sm">调用 chain.run()</div>
                  <div className="text-red-300 text-xs">字典参数展开为关键字参数</div>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-red-600 text-white flex items-center justify-center text-sm font-bold">
                  3
                </div>
                <div className="flex-1 p-3 bg-red-900/20 rounded-lg border border-red-500/30">
                  <div className="text-white text-sm">内部处理</div>
                  <div className="text-red-300 text-xs">黑盒执行，难以调试</div>
                </div>
              </div>
            </div>
          </div>

          {/* LCEL 流程 */}
          <div>
            <div className="text-sm font-medium text-green-400 mb-3">LCEL</div>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center text-sm font-bold">
                  1
                </div>
                <div className="flex-1 p-3 bg-green-900/20 rounded-lg border border-green-500/30">
                  <div className="text-white text-sm">prompt | model | parser</div>
                  <div className="text-green-300 text-xs">管道语法，一目了然</div>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center text-sm font-bold">
                  2
                </div>
                <div className="flex-1 p-3 bg-green-900/20 rounded-lg border border-green-500/30">
                  <div className="text-white text-sm">chain.invoke(dict)</div>
                  <div className="text-green-300 text-xs">统一的 Runnable 接口</div>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center text-sm font-bold">
                  3
                </div>
                <div className="flex-1 p-3 bg-green-900/20 rounded-lg border border-green-500/30">
                  <div className="text-white text-sm">透明执行</div>
                  <div className="text-green-300 text-xs">get_graph() 可视化结构</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 功能对比表 */}
      <div className="bg-slate-800 rounded-xl p-6">
        <h4 className="text-lg font-semibold text-white mb-4">详细功能对比</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-4 text-slate-300">特性</th>
                <th className="text-left py-3 px-4 text-slate-300">Legacy Chain</th>
                <th className="text-left py-3 px-4 text-slate-300">LCEL</th>
                <th className="text-center py-3 px-4 text-slate-300">推荐</th>
              </tr>
            </thead>
            <tbody>
              {comparisonData.map((item, index) => (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border-b border-slate-700/50 hover:bg-slate-700/30"
                >
                  <td className="py-3 px-4 font-medium text-white">{item.feature}</td>
                  <td className="py-3 px-4 text-red-300">{item.legacy}</td>
                  <td className="py-3 px-4 text-green-300">{item.lcel}</td>
                  <td className="py-3 px-4 text-center">
                    {item.advantage === 'lcel' ? (
                      <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-500/20 text-green-400 rounded-full text-xs">
                        <Zap className="w-3 h-3" />
                        LCEL
                      </span>
                    ) : (
                      <span className="text-slate-500">-</span>
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 迁移建议 */}
      <div className="mt-6 bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-6 border border-blue-500/30">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <ArrowRight className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h5 className="text-lg font-semibold text-blue-300 mb-2">迁移建议</h5>
            <p className="text-slate-300 text-sm mb-3">
              所有新项目都应使用 LCEL。现有项目可以逐步迁移：
            </p>
            <ul className="text-slate-400 text-sm space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-blue-400 font-bold">1.</span>
                <span>简单链（LLMChain）→ <code className="text-green-400">prompt | model | parser</code></span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 font-bold">2.</span>
                <span>顺序链（SequentialChain）→ 多个管道串联</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 font-bold">3.</span>
                <span>转换链（TransformChain）→ <code className="text-green-400">RunnableLambda</code></span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 font-bold">4.</span>
                <span>启用 LangSmith 追踪，对比性能差异</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
