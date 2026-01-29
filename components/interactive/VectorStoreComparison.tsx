"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Database, Zap, Cloud, Server, HardDrive, Network } from 'lucide-react';

interface VectorStoreInfo {
  name: string;
  icon: React.ElementType;
  deployment: string;
  performance: number; // 1-5
  cost: number; // 1-5
  scalability: number; // 1-5
  features: string[];
  pros: string[];
  cons: string[];
  useCase: string;
  color: string;
}

const vectorStores: VectorStoreInfo[] = [
  {
    name: 'Chroma',
    icon: HardDrive,
    deployment: '本地',
    performance: 3,
    cost: 1,
    scalability: 2,
    features: ['嵌入式', '持久化', '易于使用'],
    pros: ['免费开源', '快速上手', '本地部署'],
    cons: ['扩展性有限', '无托管服务', '功能较基础'],
    useCase: '开发、小规模应用',
    color: 'blue'
  },
  {
    name: 'Pinecone',
    icon: Cloud,
    deployment: '云服务',
    performance: 5,
    cost: 4,
    scalability: 5,
    features: ['托管服务', 'Namespace', 'Metadata 过滤', '弹性扩展'],
    pros: ['高性能', '自动扩展', '企业级稳定性'],
    cons: ['成本较高', '依赖第三方', '定价复杂'],
    useCase: '生产环境、企业应用',
    color: 'purple'
  },
  {
    name: 'Weaviate',
    icon: Network,
    deployment: '自托管/云',
    performance: 4,
    cost: 3,
    scalability: 4,
    features: ['混合搜索', 'GraphQL API', 'Schema 管理', 'RESTful'],
    pros: ['混合检索强大', 'API 丰富', '灵活部署'],
    cons: ['配置复杂', '学习曲线陡', '资源消耗大'],
    useCase: '混合搜索需求',
    color: 'green'
  },
  {
    name: 'Qdrant',
    icon: Zap,
    deployment: '自托管/云',
    performance: 5,
    cost: 3,
    scalability: 4,
    features: ['Rust 实现', 'Payload 过滤', '分布式', '实时更新'],
    pros: ['极高性能', 'Rust 高效', '过滤强大'],
    cons: ['生态较新', '文档较少', 'API 学习成本'],
    useCase: '高性能需求',
    color: 'orange'
  },
  {
    name: 'FAISS',
    icon: Server,
    deployment: '本地',
    performance: 5,
    cost: 1,
    scalability: 3,
    features: ['高性能索引', '多种算法', 'GPU 加速', '离线优化'],
    pros: ['极速检索', '免费', 'GPU 支持'],
    cons: ['无后端服务', '需手动管理', '无实时更新'],
    useCase: '离线批处理',
    color: 'red'
  },
  {
    name: 'Milvus',
    icon: Database,
    deployment: '自托管/云',
    performance: 5,
    cost: 4,
    scalability: 5,
    features: ['云原生', '分布式', 'GPU 加速', 'PB 级扩展'],
    pros: ['超大规模', '分布式', '高可用'],
    cons: ['部署复杂', '成本高', '运维难度大'],
    useCase: 'PB 级数据',
    color: 'indigo'
  }
];

export default function VectorStoreComparison() {
  const [selectedStore, setSelectedStore] = useState<string>('Chroma');
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards');

  const store = vectorStores.find(s => s.name === selectedStore)!;

  const getRating = (score: number) => {
    return '⭐'.repeat(score) + '☆'.repeat(5 - score);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          Vector Store Comparison
        </h3>
        <p className="text-slate-600">
          主流向量数据库性能与特性对比
        </p>
      </div>

      {/* View Mode Toggle */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={() => setViewMode('cards')}
          className={`px-6 py-2 rounded-lg font-medium ${
            viewMode === 'cards' ? 'bg-blue-500 text-white' : 'bg-white text-slate-600 border border-slate-200'
          }`}
        >
          卡片视图
        </button>
        <button
          onClick={() => setViewMode('table')}
          className={`px-6 py-2 rounded-lg font-medium ${
            viewMode === 'table' ? 'bg-blue-500 text-white' : 'bg-white text-slate-600 border border-slate-200'
          }`}
        >
          对比表
        </button>
      </div>

      {viewMode === 'cards' ? (
        <>
          {/* Store Selection */}
          <div className="grid grid-cols-6 gap-3 mb-6">
            {vectorStores.map((vs) => {
              const Icon = vs.icon;
              return (
                <button
                  key={vs.name}
                  onClick={() => setSelectedStore(vs.name)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    selectedStore === vs.name
                      ? `border-${vs.color}-500 bg-white shadow-lg`
                      : 'border-slate-200 bg-white hover:border-slate-300'
                  }`}
                >
                  <Icon className={`w-6 h-6 mx-auto mb-2 ${selectedStore === vs.name ? `text-${vs.color}-500` : 'text-slate-400'}`} />
                  <div className={`text-xs font-medium text-center ${selectedStore === vs.name ? `text-${vs.color}-600` : 'text-slate-600'}`}>
                    {vs.name}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Store Details */}
          <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
            <div className="flex items-center gap-3 mb-6">
              <div className={`w-16 h-16 rounded-lg bg-${store.color}-100 flex items-center justify-center`}>
                <store.icon className={`w-8 h-8 text-${store.color}-500`} />
              </div>
              <div>
                <h4 className="text-2xl font-bold text-slate-800">{store.name}</h4>
                <p className="text-slate-600">{store.deployment} • {store.useCase}</p>
              </div>
            </div>

            {/* Metrics */}
            <div className="grid md:grid-cols-3 gap-6 mb-6">
              <div>
                <div className="text-sm text-slate-600 mb-1">性能</div>
                <div className="text-2xl font-bold text-slate-800">{getRating(store.performance)}</div>
              </div>
              <div>
                <div className="text-sm text-slate-600 mb-1">成本</div>
                <div className="text-2xl font-bold text-slate-800">{getRating(store.cost)}</div>
              </div>
              <div>
                <div className="text-sm text-slate-600 mb-1">扩展性</div>
                <div className="text-2xl font-bold text-slate-800">{getRating(store.scalability)}</div>
              </div>
            </div>

            {/* Features, Pros, Cons */}
            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <h5 className="font-semibold text-slate-800 mb-3">核心特性</h5>
                <ul className="space-y-2">
                  {store.features.map((feature, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <div className={`w-1.5 h-1.5 rounded-full bg-${store.color}-500 mt-2`} />
                      <span className="text-sm text-slate-600">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h5 className="font-semibold text-green-600 mb-3">优势</h5>
                <ul className="space-y-2">
                  {store.pros.map((pro, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-green-500 mt-2" />
                      <span className="text-sm text-slate-600">{pro}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h5 className="font-semibold text-orange-600 mb-3">局限</h5>
                <ul className="space-y-2">
                  {store.cons.map((con, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-orange-500 mt-2" />
                      <span className="text-sm text-slate-600">{con}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </>
      ) : (
        /* Table View */
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b-2 border-slate-200">
                  <th className="text-left py-3 px-4 text-sm font-semibold text-slate-700">向量库</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-700">部署</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-700">性能</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-700">成本</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-700">扩展性</th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-slate-700">适用场景</th>
                </tr>
              </thead>
              <tbody>
                {vectorStores.map((vs) => {
                  const Icon = vs.icon;
                  return (
                    <tr key={vs.name} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <Icon className={`w-5 h-5 text-${vs.color}-500`} />
                          <span className="font-semibold text-slate-800">{vs.name}</span>
                        </div>
                      </td>
                      <td className="text-center py-3 px-4 text-sm text-slate-600">{vs.deployment}</td>
                      <td className="text-center py-3 px-4 text-sm text-slate-600">{getRating(vs.performance)}</td>
                      <td className="text-center py-3 px-4 text-sm text-slate-600">{getRating(vs.cost)}</td>
                      <td className="text-center py-3 px-4 text-sm text-slate-600">{getRating(vs.scalability)}</td>
                      <td className="py-3 px-4 text-sm text-slate-600">{vs.useCase}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">{store.name} 使用示例</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{store.name === 'Chroma' && `from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

results = vectorstore.similarity_search("query", k=5)`}

{store.name === 'Pinecone' && `from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key="xxx")
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    index_name="langchain-index"
)

results = vectorstore.similarity_search("query", k=5)`}

{store.name === 'Weaviate' && `from langchain_weaviate import WeaviateVectorStore
import weaviate

client = weaviate.Client("http://localhost:8080")
vectorstore = WeaviateVectorStore.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    client=client
)

results = vectorstore.similarity_search("query", k=5)`}

{store.name === 'Qdrant' && `from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    collection_name="docs",
    url="http://localhost:6333"
)

results = vectorstore.similarity_search("query", k=5)`}

{store.name === 'FAISS' && `from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents, OpenAIEmbeddings()
)

vectorstore.save_local("faiss_index")
results = vectorstore.similarity_search("query", k=5)`}

{store.name === 'Milvus' && `from langchain_milvus import Milvus

vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    collection_name="langchain_docs",
    connection_args={"host": "localhost", "port": "19530"}
)

results = vectorstore.similarity_search("query", k=5)`}
        </pre>
      </div>
    </div>
  );
}
