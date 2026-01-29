"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, CheckCircle, AlertTriangle, Code, FileText, Wrench } from 'lucide-react';

type MigrationStep = {
  id: string;
  title: string;
  description: string;
  code?: string;
  difficulty: 'easy' | 'medium' | 'hard';
  estimatedTime: string;
};

type MigrationPath = {
  id: string;
  from: string;
  to: string;
  title: string;
  description: string;
  steps: MigrationStep[];
  risks: string[];
  benefits: string[];
};

const migrationPaths: MigrationPath[] = [
  {
    id: 'llamaindex-to-langchain',
    from: 'LlamaIndex',
    to: 'LangChain',
    title: 'LlamaIndex → LangChain',
    description: '保留 LlamaIndex 索引，扩展为复杂 Agent 系统',
    steps: [
      {
        id: 's1',
        title: '1. 保留现有 LlamaIndex 索引',
        description: '无需修改现有的索引构建代码',
        code: `from llama_index.core import VectorStoreIndex
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)`,
        difficulty: 'easy',
        estimatedTime: '5 分钟'
      },
      {
        id: 's2',
        title: '2. 包装为 LangChain Retriever',
        description: '使用 LlamaIndexRetriever 桥接',
        code: `from langchain.retrievers import LlamaIndexRetriever
retriever = LlamaIndexRetriever(index=index)`,
        difficulty: 'easy',
        estimatedTime: '10 分钟'
      },
      {
        id: 's3',
        title: '3. 替换查询逻辑为 LangChain Chain',
        description: '从 query_engine 迁移到 RetrievalQA',
        code: `from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)`,
        difficulty: 'medium',
        estimatedTime: '30 分钟'
      },
      {
        id: 's4',
        title: '4. 添加 Agent 和工具',
        description: '扩展为多工具 Agent 系统',
        code: `from langchain.agents import Tool, create_openai_functions_agent

kb_tool = Tool(
    name="knowledge_base",
    func=lambda q: str(index.as_query_engine().query(q)),
    description="搜索内部知识库"
)

agent = create_openai_functions_agent(llm, [kb_tool, ...], prompt)`,
        difficulty: 'hard',
        estimatedTime: '2 小时'
      },
      {
        id: 's5',
        title: '5. 测试与优化',
        description: '对比新旧系统的准确率和性能',
        difficulty: 'medium',
        estimatedTime: '1 小时'
      }
    ],
    risks: ['依赖版本冲突', 'LlamaIndex 索引需定期重建', '性能可能略有下降'],
    benefits: ['保留专业 RAG 能力', '扩展 Agent 功能', '统一 LangChain 生态']
  },
  {
    id: 'haystack-to-langchain',
    from: 'Haystack',
    to: 'LangChain',
    title: 'Haystack → LangChain',
    description: '从传统搜索升级为生成式 QA',
    steps: [
      {
        id: 'h1',
        title: '1. 迁移 DocumentStore 到 VectorStore',
        description: '从 Elasticsearch 迁移到向量存储',
        code: `# 旧代码（Haystack）
from haystack.document_stores import ElasticsearchDocumentStore
doc_store = ElasticsearchDocumentStore()

# 新代码（LangChain）
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())`,
        difficulty: 'medium',
        estimatedTime: '1 小时'
      },
      {
        id: 'h2',
        title: '2. 替换 Retriever',
        description: 'BM25Retriever → VectorStore Retriever',
        code: `# 旧代码
retriever = BM25Retriever(document_store=doc_store)

# 新代码
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})`,
        difficulty: 'easy',
        estimatedTime: '15 分钟'
      },
      {
        id: 'h3',
        title: '3. 替换 Reader 为生成式 LLM',
        description: 'FARM Reader → ChatOpenAI',
        code: `# 旧代码
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# 新代码
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")`,
        difficulty: 'easy',
        estimatedTime: '10 分钟'
      },
      {
        id: 'h4',
        title: '4. 重构 Pipeline 为 Chain',
        description: 'Haystack Pipeline → LangChain RetrievalQA',
        code: `# 旧代码
pipe = Pipeline()
pipe.add_node(retriever, "Retriever", inputs=["Query"])
pipe.add_node(reader, "Reader", inputs=["Retriever"])

# 新代码
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)`,
        difficulty: 'medium',
        estimatedTime: '45 分钟'
      },
      {
        id: 'h5',
        title: '5. 迁移评估指标',
        description: '从 Haystack 评估迁移到 LangSmith',
        difficulty: 'hard',
        estimatedTime: '2 小时'
      }
    ],
    risks: ['BM25 性能优势丢失', '评估体系需重建', 'Elasticsearch 投资浪费'],
    benefits: ['生成式答案更自然', 'LLM 能力更强', '统一现代技术栈']
  },
  {
    id: 'langchain-to-llamaindex',
    from: 'LangChain',
    to: 'LlamaIndex',
    title: 'LangChain → LlamaIndex',
    description: '需要更高级的 RAG 索引能力',
    steps: [
      {
        id: 'l1',
        title: '1. 分析 LangChain RAG 瓶颈',
        description: '确定是否真的需要迁移',
        difficulty: 'easy',
        estimatedTime: '30 分钟'
      },
      {
        id: 'l2',
        title: '2. 构建 LlamaIndex 高级索引',
        description: '使用 Tree Index 或 Hybrid Index',
        code: `from llama_index.core import TreeIndex, VectorStoreIndex
tree_index = TreeIndex.from_documents(documents)
vector_index = VectorStoreIndex.from_documents(documents)`,
        difficulty: 'medium',
        estimatedTime: '1 小时'
      },
      {
        id: 'l3',
        title: '3. 保留 LangChain Agent 逻辑',
        description: '将 LlamaIndex 包装为 LangChain Tool',
        code: `from langchain.tools import Tool

def llamaindex_query(q: str) -> str:
    return str(tree_index.as_query_engine().query(q))

tool = Tool(name="advanced_rag", func=llamaindex_query, description="...")`,
        difficulty: 'medium',
        estimatedTime: '45 分钟'
      },
      {
        id: 'l4',
        title: '4. 混合使用两个框架',
        description: 'LlamaIndex RAG + LangChain Agent',
        difficulty: 'hard',
        estimatedTime: '2 小时'
      }
    ],
    risks: ['维护两套框架复杂度增加', '团队学习成本'],
    benefits: ['RAG 性能提升', '高级索引能力', '保留 Agent 功能']
  }
];

export default function MigrationPathGuide() {
  const [selectedPath, setSelectedPath] = useState<MigrationPath>(migrationPaths[0]);
  const [currentStep, setCurrentStep] = useState(0);

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'bg-green-100 text-green-700';
      case 'medium': return 'bg-yellow-100 text-yellow-700';
      case 'hard': return 'bg-red-100 text-red-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getDifficultyLabel = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return '简单';
      case 'medium': return '中等';
      case 'hard': return '困难';
      default: return '';
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">框架迁移路径指南</h3>
        <p className="text-gray-600">逐步指导如何在不同 LLM 框架间平滑迁移</p>
      </div>

      {/* 迁移路径选择 */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-3">选择迁移路径</label>
        <div className="grid grid-cols-3 gap-3">
          {migrationPaths.map(path => (
            <button
              key={path.id}
              onClick={() => {
                setSelectedPath(path);
                setCurrentStep(0);
              }}
              className={`p-4 rounded-lg border-2 text-left transition-all ${
                selectedPath.id === path.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-blue-300'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-gray-500">{path.from}</span>
                <ArrowRight className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-semibold text-blue-600">{path.to}</span>
              </div>
              <div className="text-xs text-gray-600">{path.description}</div>
              <div className="mt-2 text-xs text-gray-500">
                {path.steps.length} 个步骤
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 迁移概览 */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold mb-3">{selectedPath.title}</h4>
        <p className="text-sm text-gray-700 mb-4">{selectedPath.description}</p>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <h5 className="text-sm font-semibold mb-2 text-green-700 flex items-center gap-1">
              <CheckCircle className="w-4 h-4" />
              迁移收益
            </h5>
            <ul className="space-y-1">
              {selectedPath.benefits.map((benefit, i) => (
                <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>{benefit}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h5 className="text-sm font-semibold mb-2 text-orange-700 flex items-center gap-1">
              <AlertTriangle className="w-4 h-4" />
              潜在风险
            </h5>
            <ul className="space-y-1">
              {selectedPath.risks.map((risk, i) => (
                <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                  <span className="text-orange-600">!</span>
                  <span>{risk}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* 步骤导航 */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold">迁移步骤</h4>
          <div className="text-sm text-gray-600">
            步骤 {currentStep + 1} / {selectedPath.steps.length}
          </div>
        </div>

        <div className="flex gap-2 mb-4">
          {selectedPath.steps.map((step, index) => (
            <button
              key={step.id}
              onClick={() => setCurrentStep(index)}
              className={`flex-1 h-2 rounded-full transition-all ${
                index === currentStep
                  ? 'bg-blue-600'
                  : index < currentStep
                  ? 'bg-green-500'
                  : 'bg-gray-300'
              }`}
            />
          ))}
        </div>
      </div>

      {/* 当前步骤详情 */}
      <motion.div
        key={currentStep}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="mb-6 p-6 border-2 border-blue-300 rounded-lg bg-blue-50"
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <h4 className="text-lg font-semibold mb-2">{selectedPath.steps[currentStep].title}</h4>
            <p className="text-sm text-gray-700 mb-3">{selectedPath.steps[currentStep].description}</p>
            
            <div className="flex items-center gap-3">
              <span className={`px-2 py-1 rounded text-xs font-medium ${getDifficultyColor(selectedPath.steps[currentStep].difficulty)}`}>
                {getDifficultyLabel(selectedPath.steps[currentStep].difficulty)}
              </span>
              <span className="text-xs text-gray-600">
                预计时间: {selectedPath.steps[currentStep].estimatedTime}
              </span>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
              disabled={currentStep === 0}
              className="px-3 py-1 bg-gray-600 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed text-sm"
            >
              上一步
            </button>
            <button
              onClick={() => setCurrentStep(Math.min(selectedPath.steps.length - 1, currentStep + 1))}
              disabled={currentStep === selectedPath.steps.length - 1}
              className="px-3 py-1 bg-blue-600 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed text-sm"
            >
              下一步
            </button>
          </div>
        </div>

        {selectedPath.steps[currentStep].code && (
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <div className="flex items-center gap-2 mb-2 text-xs text-gray-400">
              <Code className="w-3 h-3" />
              <span>代码示例</span>
            </div>
            <pre className="text-sm font-mono">
              {selectedPath.steps[currentStep].code}
            </pre>
          </div>
        )}
      </motion.div>

      {/* 所有步骤概览 */}
      <div>
        <h4 className="font-semibold mb-3">完整步骤概览</h4>
        <div className="space-y-2">
          {selectedPath.steps.map((step, index) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => setCurrentStep(index)}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                index === currentStep
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                  index < currentStep
                    ? 'bg-green-500 text-white'
                    : index === currentStep
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-300 text-gray-600'
                }`}>
                  {index < currentStep ? '✓' : index + 1}
                </div>
                
                <div className="flex-1">
                  <div className="font-medium text-sm">{step.title}</div>
                  <div className="text-xs text-gray-600 mt-1">{step.description}</div>
                </div>

                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-xs ${getDifficultyColor(step.difficulty)}`}>
                    {getDifficultyLabel(step.difficulty)}
                  </span>
                  <span className="text-xs text-gray-500">{step.estimatedTime}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* 提示 */}
      <div className="mt-6 p-4 bg-yellow-50 border-2 border-yellow-200 rounded-lg">
        <div className="flex items-start gap-3">
          <Wrench className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-yellow-900 mb-2">迁移建议</h4>
            <ul className="text-sm text-yellow-800 space-y-1">
              <li>• 采用渐进式迁移，保留旧系统作为回滚方案</li>
              <li>• 在测试环境完整验证后再上生产</li>
              <li>• 对比新旧系统的准确率、延迟、成本等关键指标</li>
              <li>• 使用 Feature Flag 控制流量切换</li>
              <li>• 保持充分的文档记录和团队沟通</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
