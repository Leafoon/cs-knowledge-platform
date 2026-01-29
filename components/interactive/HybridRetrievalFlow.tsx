"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { GitBranch, Zap, Filter, Sparkles } from 'lucide-react';

export default function HybridRetrievalFlow() {
  const [step, setStep] = useState(0);

  const steps = [
    {
      id: 0,
      title: '用户查询',
      description: 'User query: "LangChain vector database tutorial"',
      icon: '🔍',
      details: '用户输入自然语言查询'
    },
    {
      id: 1,
      title: 'BM25 检索',
      description: '关键词匹配检索',
      icon: '📝',
      details: 'BM25 算法基于词频和逆文档频率计算相关性'
    },
    {
      id: 2,
      title: 'Vector 检索',
      description: '语义相似度检索',
      icon: '🧠',
      details: '将查询嵌入到向量空间，检索最相似的文档'
    },
    {
      id: 3,
      title: '结果融合',
      description: 'RRF (Reciprocal Rank Fusion)',
      icon: '🔀',
      details: '使用 RRF 算法合并两种检索结果'
    },
    {
      id: 4,
      title: 'Reranking',
      description: 'Cross-Encoder 重排序',
      icon: '⚖️',
      details: '使用更精确的模型对候选文档重新排序'
    },
    {
      id: 5,
      title: '最终结果',
      description: 'Top-K 相关文档',
      icon: '✅',
      details: '返回最相关的 K 个文档'
    }
  ];

  const bm25Results = [
    { id: 1, title: "LangChain Tutorial", score: 0.85, source: 'BM25' },
    { id: 3, title: "Vector Database Guide", score: 0.72, source: 'BM25' },
    { id: 5, title: "Chroma Documentation", score: 0.68, source: 'BM25' }
  ];

  const vectorResults = [
    { id: 2, title: "RAG Architecture", score: 0.91, source: 'Vector' },
    { id: 1, title: "LangChain Tutorial", score: 0.88, source: 'Vector' },
    { id: 4, title: "Embeddings Explained", score: 0.75, source: 'Vector' }
  ];

  const fusedResults = [
    { id: 1, title: "LangChain Tutorial", rrfScore: 0.92, sources: ['BM25', 'Vector'] },
    { id: 2, title: "RAG Architecture", rrfScore: 0.85, sources: ['Vector'] },
    { id: 3, title: "Vector Database Guide", rrfScore: 0.78, sources: ['BM25'] },
    { id: 4, title: "Embeddings Explained", rrfScore: 0.71, sources: ['Vector'] },
    { id: 5, title: "Chroma Documentation", rrfScore: 0.65, sources: ['BM25'] }
  ];

  const rerankedResults = [
    { id: 1, title: "LangChain Tutorial", finalScore: 0.95, badge: '🥇' },
    { id: 2, title: "RAG Architecture", finalScore: 0.89, badge: '🥈' },
    { id: 3, title: "Vector Database Guide", finalScore: 0.84, badge: '🥉' }
  ];

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          Hybrid Retrieval Flow
        </h3>
        <p className="text-slate-600">
          BM25 + Vector + Reranking 混合检索流程可视化
        </p>
      </div>

      {/* Progress Steps */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
        <div className="flex items-center justify-between mb-8">
          {steps.map((s, idx) => (
            <React.Fragment key={s.id}>
              <div
                onClick={() => setStep(s.id)}
                className={`flex flex-col items-center cursor-pointer transition-all ${
                  step >= s.id ? 'opacity-100' : 'opacity-40'
                }`}
              >
                <div className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all ${
                  step === s.id
                    ? 'bg-blue-500 text-white scale-110 shadow-lg'
                    : step > s.id
                    ? 'bg-green-500 text-white'
                    : 'bg-slate-200'
                }`}>
                  {s.icon}
                </div>
                <div className="text-xs font-medium text-slate-700 mt-2 text-center max-w-20">
                  {s.title}
                </div>
              </div>

              {idx < steps.length - 1 && (
                <div className={`flex-1 h-1 mx-2 transition-colors ${
                  step > s.id ? 'bg-green-500' : 'bg-slate-200'
                }`} />
              )}
            </React.Fragment>
          ))}
        </div>

        <div className="text-center">
          <h4 className="text-lg font-semibold text-slate-800 mb-2">
            {steps[step].description}
          </h4>
          <p className="text-sm text-slate-600">{steps[step].details}</p>
        </div>

        <div className="flex gap-3 mt-6 justify-center">
          <button
            onClick={() => setStep(Math.max(0, step - 1))}
            disabled={step === 0}
            className="px-6 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            上一步
          </button>
          <button
            onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
            disabled={step === steps.length - 1}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            下一步
          </button>
        </div>
      </div>

      {/* Step Content */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        {step === 0 && (
          <div className="text-center py-8">
            <div className="text-6xl mb-4">🔍</div>
            <h4 className="text-xl font-bold text-slate-800 mb-2">用户查询</h4>
            <div className="inline-block px-6 py-3 bg-blue-50 border border-blue-200 rounded-lg">
              <code className="text-blue-600 font-mono">
                "LangChain vector database tutorial"
              </code>
            </div>
          </div>
        )}

        {step === 1 && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <GitBranch className="w-5 h-5 text-purple-500" />
              <h4 className="font-semibold text-slate-800">BM25 检索结果</h4>
            </div>
            <div className="space-y-3">
              {bm25Results.map((result, idx) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="p-4 rounded-lg border-l-4 border-purple-500 bg-purple-50"
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <span className="text-sm font-semibold text-purple-700">#{idx + 1}</span>
                      <span className="ml-3 text-slate-800">{result.title}</span>
                    </div>
                    <span className="text-sm font-bold text-purple-600">
                      {(result.score * 100).toFixed(0)}%
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {step === 2 && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Zap className="w-5 h-5 text-blue-500" />
              <h4 className="font-semibold text-slate-800">Vector 检索结果</h4>
            </div>
            <div className="space-y-3">
              {vectorResults.map((result, idx) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="p-4 rounded-lg border-l-4 border-blue-500 bg-blue-50"
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <span className="text-sm font-semibold text-blue-700">#{idx + 1}</span>
                      <span className="ml-3 text-slate-800">{result.title}</span>
                    </div>
                    <span className="text-sm font-bold text-blue-600">
                      {(result.score * 100).toFixed(0)}%
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {step === 3 && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Filter className="w-5 h-5 text-green-500" />
              <h4 className="font-semibold text-slate-800">RRF 融合结果</h4>
            </div>
            <div className="space-y-3">
              {fusedResults.map((result, idx) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="p-4 rounded-lg border border-green-200 bg-green-50"
                >
                  <div className="flex justify-between items-center mb-2">
                    <div>
                      <span className="text-sm font-semibold text-green-700">#{idx + 1}</span>
                      <span className="ml-3 text-slate-800">{result.title}</span>
                    </div>
                    <span className="text-sm font-bold text-green-600">
                      {(result.rrfScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex gap-2">
                    {result.sources.map((source) => (
                      <span
                        key={source}
                        className={`text-xs px-2 py-1 rounded ${
                          source === 'BM25'
                            ? 'bg-purple-100 text-purple-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}
                      >
                        {source}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {(step === 4 || step === 5) && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="w-5 h-5 text-yellow-500" />
              <h4 className="font-semibold text-slate-800">
                {step === 4 ? 'Reranking 重排序' : '最终结果 (Top-3)'}
              </h4>
            </div>
            <div className="space-y-3">
              {rerankedResults.map((result, idx) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.15 }}
                  className="p-6 rounded-lg border-2 border-yellow-300 bg-gradient-to-r from-yellow-50 to-orange-50 shadow-lg"
                >
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <span className="text-3xl">{result.badge}</span>
                      <div>
                        <div className="text-sm font-semibold text-yellow-700">Rank #{idx + 1}</div>
                        <div className="text-lg font-bold text-slate-800">{result.title}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-slate-500">最终分数</div>
                      <div className="text-2xl font-bold text-yellow-600">
                        {(result.finalScore * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">混合检索代码</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 1. BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# 2. Vector Retriever
vectorstore = FAISS.from_documents(documents, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 3. Ensemble（RRF 融合）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # BM25: 40%, Vector: 60%
)

# 4. Reranking
compressor = CohereRerank(model="rerank-english-v2.0", top_n=3)
hybrid_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

# 使用
docs = hybrid_retriever.invoke("LangChain vector database tutorial")
for doc in docs:
    print(doc.page_content)`}
        </pre>
      </div>
    </div>
  );
}
