"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Sparkles } from 'lucide-react';

const sampleDocuments = [
  { id: 1, content: "LangChain is a framework for developing applications powered by language models.", similarity: 0 },
  { id: 2, content: "RAG (Retrieval-Augmented Generation) combines retrieval and generation.", similarity: 0 },
  { id: 3, content: "Vector databases store high-dimensional embeddings for similarity search.", similarity: 0 },
  { id: 4, content: "Chroma is an open-source embedding database for AI applications.", similarity: 0 },
  { id: 5, content: "FAISS is a library for efficient similarity search developed by Facebook.", similarity: 0 },
  { id: 6, content: "Pinecone provides a managed vector database service.", similarity: 0 },
  { id: 7, content: "Embeddings are vector representations of text in high-dimensional space.", similarity: 0 },
  { id: 8, content: "Semantic search finds documents based on meaning rather than exact keywords.", similarity: 0 }
];

export default function SimilaritySearchDemo() {
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState(sampleDocuments);
  const [isSearching, setIsSearching] = useState(false);
  const [k, setK] = useState(3);

  const handleSearch = () => {
    if (!query.trim()) return;

    setIsSearching(true);

    // 模拟相似度计算（实际应使用真实 embedding）
    setTimeout(() => {
      const resultsWithSimilarity = sampleDocuments.map(doc => {
        // 简化的相似度计算：关键词匹配
        const keywords = query.toLowerCase().split(' ');
        const contentLower = doc.content.toLowerCase();
        
        let similarity = 0;
        keywords.forEach(keyword => {
          if (contentLower.includes(keyword)) {
            similarity += 0.3;
          }
        });

        // 添加随机噪声
        similarity = Math.min(1, similarity + Math.random() * 0.2);

        return { ...doc, similarity };
      });

      // 按相似度降序排序
      const sorted = resultsWithSimilarity.sort((a, b) => b.similarity - a.similarity);
      setSearchResults(sorted);
      setIsSearching(false);
    }, 500);
  };

  const topResults = searchResults.slice(0, k);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          Similarity Search Demo
        </h3>
        <p className="text-slate-600">
          实时相似度搜索演示
        </p>
      </div>

      {/* Search Input */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
        <label className="text-sm font-medium text-slate-700 mb-2 block">
          查询文本
        </label>
        <div className="flex gap-3 mb-4">
          <input
            type="text"
            placeholder="例如：vector database for AI"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            className="flex-1 px-4 py-3 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSearch}
            disabled={isSearching || !query.trim()}
            className="px-8 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isSearching ? (
              <>
                <Sparkles className="w-5 h-5 animate-spin" />
                搜索中...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                搜索
              </>
            )}
          </button>
        </div>

        {/* K Parameter */}
        <div>
          <label className="text-sm font-medium text-slate-700 mb-2 block">
            Top-K: {k}
          </label>
          <input
            type="range"
            min="1"
            max="8"
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>1</span>
            <span>8</span>
          </div>
        </div>
      </div>

      {/* Search Results */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h4 className="font-semibold text-slate-800 mb-4">
          搜索结果（Top-{k}）
        </h4>

        {!query ? (
          <div className="text-center py-12 text-slate-400">
            <Search className="w-16 h-16 mx-auto mb-4 opacity-20" />
            <p>输入查询文本开始搜索</p>
          </div>
        ) : (
          <div className="space-y-3">
            {topResults.map((doc, idx) => (
              <motion.div
                key={doc.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="p-4 rounded-lg border-l-4 border-blue-500 bg-slate-50 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between mb-2">
                  <span className="text-sm font-semibold text-blue-600">
                    #{idx + 1} • Doc {doc.id}
                  </span>
                  <div className="flex items-center gap-2">
                    <div className="text-xs text-slate-500">相似度</div>
                    <div className={`text-sm font-bold ${
                      doc.similarity > 0.7 ? 'text-green-600' :
                      doc.similarity > 0.4 ? 'text-yellow-600' :
                      'text-orange-600'
                    }`}>
                      {(doc.similarity * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <p className="text-sm text-slate-700 mb-2">{doc.content}</p>

                {/* Similarity Bar */}
                <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full ${
                      doc.similarity > 0.7 ? 'bg-green-500' :
                      doc.similarity > 0.4 ? 'bg-yellow-500' :
                      'bg-orange-500'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${doc.similarity * 100}%` }}
                    transition={{ duration: 0.5, delay: idx * 0.1 }}
                  />
                </div>
              </motion.div>
            ))}

            {/* Statistics */}
            <div className="mt-6 pt-4 border-t border-slate-200">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-blue-600">{topResults.length}</div>
                  <div className="text-xs text-slate-600">返回文档</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-600">
                    {topResults.length > 0 ? (topResults[0].similarity * 100).toFixed(1) : 0}%
                  </div>
                  <div className="text-xs text-slate-600">最高相似度</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-purple-600">
                    {topResults.length > 0 
                      ? ((topResults.reduce((sum, d) => sum + d.similarity, 0) / topResults.length) * 100).toFixed(1)
                      : 0}%
                  </div>
                  <div className="text-xs text-slate-600">平均相似度</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">相似度搜索代码</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量库
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings()
)

# 相似度搜索
results = vectorstore.similarity_search(
    query="${query || 'vector database'}",
    k=${k}
)

# 带分数的搜索
results_with_scores = vectorstore.similarity_search_with_score(
    query="${query || 'vector database'}",
    k=${k}
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(doc.page_content)
    print("---")`}
        </pre>
      </div>
    </div>
  );
}
