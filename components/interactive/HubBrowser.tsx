"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

type PromptItem = {
  id: string;
  author: string;
  name: string;
  description: string;
  tags: string[];
  downloads: number;
  rating: number;
  category: string;
  code: string;
};

const hubPrompts: PromptItem[] = [
  {
    id: 'rlm/rag-prompt',
    author: 'rlm',
    name: 'RAG Prompt',
    description: '用于检索增强生成(RAG)的高质量提示模板',
    tags: ['rag', 'qa', 'retrieval'],
    downloads: 12500,
    rating: 4.8,
    category: 'QA',
    code: `Answer the question based only on the following context:

{context}

Question: {question}

Answer:`
  },
  {
    id: 'hwchase17/openai-functions-agent',
    author: 'hwchase17',
    name: 'OpenAI Functions Agent',
    description: '使用 OpenAI Function Calling 的 Agent 提示',
    tags: ['agent', 'functions', 'tools'],
    downloads: 8900,
    rating: 4.7,
    category: 'Agent',
    code: `You are a helpful AI assistant with access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Final Answer: the final answer

Question: {input}`
  },
  {
    id: 'langchain-ai/sql-query-system',
    author: 'langchain-ai',
    name: 'SQL Query Generator',
    description: '将自然语言转换为 SQL 查询',
    tags: ['sql', 'database', 'code-generation'],
    downloads: 6700,
    rating: 4.6,
    category: 'Code',
    code: `Given an input question, create a syntactically correct {dialect} query.

Only use the following tables:
{table_info}

Question: {input}

SQL Query:`
  },
  {
    id: 'summarization/map-reduce',
    author: 'summarization',
    name: 'Map-Reduce Summarization',
    description: '用于长文档摘要的 Map-Reduce 提示',
    tags: ['summarization', 'map-reduce'],
    downloads: 5400,
    rating: 4.5,
    category: 'Summarization',
    code: `Write a concise summary of the following:

{text}

CONCISE SUMMARY:`
  },
  {
    id: 'translation/professional',
    author: 'translation',
    name: 'Professional Translation',
    description: '专业级翻译提示，保持语境和风格',
    tags: ['translation', 'multilingual'],
    downloads: 4200,
    rating: 4.7,
    category: 'Translation',
    code: `Translate the following text from {source_lang} to {target_lang}.
Maintain the original tone and style.

Text: {text}

Translation:`
  },
  {
    id: 'creative/story-writer',
    author: 'creative',
    name: 'Story Writer',
    description: '创意写作助手，生成引人入胜的故事',
    tags: ['creative', 'writing', 'story'],
    downloads: 3800,
    rating: 4.4,
    category: 'Creative',
    code: `You are a creative storyteller. Write a compelling story based on:

Theme: {theme}
Genre: {genre}
Style: {style}

Story:`
  }
];

const categories = ['All', 'QA', 'Agent', 'Code', 'Summarization', 'Translation', 'Creative'];

export default function HubBrowser() {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPrompt, setSelectedPrompt] = useState<PromptItem | null>(null);
  const [sortBy, setSortBy] = useState<'downloads' | 'rating'>('downloads');

  // 过滤和排序
  const filteredPrompts = hubPrompts
    .filter(p => {
      const matchCategory = selectedCategory === 'All' || p.category === selectedCategory;
      const matchSearch = searchQuery === '' || 
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchCategory && matchSearch;
    })
    .sort((a, b) => {
      if (sortBy === 'downloads') return b.downloads - a.downloads;
      return b.rating - a.rating;
    });

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
        LangChain Hub 提示浏览器
      </h3>
      
      <p className="text-slate-600 dark:text-slate-400 mb-6">
        浏览、搜索和使用社区共享的高质量提示模板
      </p>

      {/* 搜索和筛选 */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="flex-1">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="搜索提示、标签..."
            className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
          />
        </div>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as 'downloads' | 'rating')}
          className="px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
        >
          <option value="downloads">按下载量</option>
          <option value="rating">按评分</option>
        </select>
      </div>

      {/* 分类标签 */}
      <div className="flex flex-wrap gap-2 mb-6">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
              selectedCategory === cat
                ? 'bg-blue-500 text-white shadow-lg'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* 提示列表 */}
        <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
          {filteredPrompts.length === 0 ? (
            <div className="text-center py-12 text-slate-500 dark:text-slate-400">
              未找到匹配的提示
            </div>
          ) : (
            filteredPrompts.map((prompt) => (
              <motion.button
                key={prompt.id}
                onClick={() => setSelectedPrompt(prompt)}
                className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                  selectedPrompt?.id === prompt.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-600'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h4 className="font-bold text-slate-900 dark:text-white">
                      {prompt.name}
                    </h4>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      by {prompt.author}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-amber-500">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    <span className="text-sm font-semibold">{prompt.rating}</span>
                  </div>
                </div>

                <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                  {prompt.description}
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-1">
                    {prompt.tags.slice(0, 3).map(tag => (
                      <span
                        key={tag}
                        className="text-xs px-2 py-0.5 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">
                    {(prompt.downloads / 1000).toFixed(1)}k 下载
                  </div>
                </div>
              </motion.button>
            ))
          )}
        </div>

        {/* 提示详情 */}
        <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-6 border border-slate-200 dark:border-slate-700">
          {selectedPrompt ? (
            <motion.div
              key={selectedPrompt.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div>
                <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
                  {selectedPrompt.name}
                </h3>
                <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400">
                  <span>by {selectedPrompt.author}</span>
                  <span>•</span>
                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    {selectedPrompt.rating}
                  </span>
                  <span>•</span>
                  <span>{selectedPrompt.downloads.toLocaleString()} 下载</span>
                </div>
              </div>

              <p className="text-slate-700 dark:text-slate-300">
                {selectedPrompt.description}
              </p>

              <div>
                <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                  标签：
                </div>
                <div className="flex flex-wrap gap-2">
                  {selectedPrompt.tags.map(tag => (
                    <span
                      key={tag}
                      className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-sm"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                  提示内容：
                </div>
                <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-sm font-mono overflow-x-auto border border-slate-700">
                  {selectedPrompt.code}
                </pre>
              </div>

              <div>
                <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                  使用代码：
                </div>
                <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-sm font-mono overflow-x-auto border border-slate-700">
                  <code>{`from langchain import hub

# 拉取提示
prompt = hub.pull("${selectedPrompt.id}")

# 使用
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = prompt | ChatOpenAI() | StrOutputParser()
result = chain.invoke({...})`}</code>
                </pre>
              </div>

              <div className="flex gap-3">
                <button className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors">
                  使用此提示
                </button>
                <button className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg font-medium hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors">
                  Fork
                </button>
              </div>
            </motion.div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <svg className="w-16 h-16 text-slate-300 dark:text-slate-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-slate-500 dark:text-slate-400">
                选择一个提示查看详情
              </p>
            </div>
          )}
        </div>
      </div>

      {/* 统计信息 */}
      <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {hubPrompts.length}
          </div>
          <div className="text-sm text-blue-700 dark:text-blue-300 mt-1">
            提示总数
          </div>
        </div>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {categories.length - 1}
          </div>
          <div className="text-sm text-green-700 dark:text-green-300 mt-1">
            分类
          </div>
        </div>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {(hubPrompts.reduce((sum, p) => sum + p.downloads, 0) / 1000).toFixed(0)}k
          </div>
          <div className="text-sm text-purple-700 dark:text-purple-300 mt-1">
            总下载量
          </div>
        </div>
        <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">
            {(hubPrompts.reduce((sum, p) => sum + p.rating, 0) / hubPrompts.length).toFixed(1)}
          </div>
          <div className="text-sm text-amber-700 dark:text-amber-300 mt-1">
            平均评分
          </div>
        </div>
      </div>
    </div>
  );
}
