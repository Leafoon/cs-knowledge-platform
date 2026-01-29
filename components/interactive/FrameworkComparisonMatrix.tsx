"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle, MinusCircle, Star, TrendingUp } from 'lucide-react';

type Framework = {
  id: string;
  name: string;
  tagline: string;
  logo: string;
  features: Record<string, number>; // 0: 不支持, 1: 部分支持, 2: 完全支持
  useCases: string[];
  pros: string[];
  cons: string[];
  githubStars: string;
  ecosystem: string;
};

type ComparisonView = 'features' | 'use-cases' | 'recommendations';

const frameworks: Framework[] = [
  {
    id: 'langchain',
    name: 'LangChain',
    tagline: '通用 LLM 应用编排框架',
    logo: '🦜',
    features: {
      'RAG 系统': 2,
      'Agent 系统': 2,
      '记忆管理': 2,
      'LCEL 编排': 2,
      '可观测性': 2,
      '部署工具': 2,
      '多模态': 1,
      '代码执行': 1,
      '群聊 Agent': 1
    },
    useCases: ['通用 LLM 应用', '复杂 Agent 系统', 'RAG + Agent 混合', '生产级部署'],
    pros: ['生态最成熟', 'LangGraph 强大', 'LangSmith 可观测性', '社区活跃'],
    cons: ['抽象层次高', '版本变化快', '性能开销相对较大'],
    githubStars: '88k+',
    ecosystem: 'Python + JS'
  },
  {
    id: 'llamaindex',
    name: 'LlamaIndex',
    tagline: 'RAG 系统专家',
    logo: '🦙',
    features: {
      'RAG 系统': 2,
      'Agent 系统': 1,
      '记忆管理': 1,
      'LCEL 编排': 0,
      '可观测性': 2,
      '部署工具': 1,
      '多模态': 2,
      '代码执行': 0,
      '群聊 Agent': 0
    },
    useCases: ['复杂 RAG 索引', '多模态检索', 'RAG 评估', '知识库构建'],
    pros: ['RAG 功能最强', '高级索引结构', '内置评估工具', '多模态支持好'],
    cons: ['Agent 功能较弱', '部署工具简单', '社区相对较小'],
    githubStars: '33k+',
    ecosystem: 'Python + TS'
  },
  {
    id: 'haystack',
    name: 'Haystack',
    tagline: '企业搜索 + NLP',
    logo: '🌾',
    features: {
      'RAG 系统': 2,
      'Agent 系统': 1,
      '记忆管理': 0,
      'LCEL 编排': 0,
      '可观测性': 1,
      '部署工具': 2,
      '多模态': 1,
      '代码执行': 0,
      '群聊 Agent': 0
    },
    useCases: ['企业级搜索', '传统 NLP 升级', '大规模文档检索', 'QA 系统'],
    pros: ['Pipeline 清晰', 'REST API 内置', 'Elasticsearch 集成好', '企业友好'],
    cons: ['LLM 支持较弱', '社区活跃度下降', '与现代 LLM 框架割裂'],
    githubStars: '15k+',
    ecosystem: 'Python'
  },
  {
    id: 'autogen',
    name: 'AutoGen',
    tagline: '自主多 Agent 对话',
    logo: '🤖',
    features: {
      'RAG 系统': 1,
      'Agent 系统': 2,
      '记忆管理': 0,
      'LCEL 编排': 0,
      '可观测性': 1,
      '部署工具': 0,
      '多模态': 1,
      '代码执行': 2,
      '群聊 Agent': 2
    },
    useCases: ['多 Agent 研究', '代码生成与执行', '自主任务解决', '探索性对话'],
    pros: ['Agent 对话自然', '代码执行安全', '群聊模式强大', '微软支持'],
    cons: ['控制粒度低', '生产化困难', '成本较高', '可观测性弱'],
    githubStars: '28k+',
    ecosystem: 'Python'
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    tagline: '角色化团队协作',
    logo: '👥',
    features: {
      'RAG 系统': 1,
      'Agent 系统': 2,
      '记忆管理': 1,
      'LCEL 编排': 0,
      '可观测性': 1,
      '部署工具': 0,
      '多模态': 0,
      '代码执行': 0,
      '群聊 Agent': 2
    },
    useCases: ['业务流程自动化', '角色分工明确的任务', '内容创作流程', '项目管理'],
    pros: ['角色定义清晰', '业务友好', '流程编排简单', '快速上手'],
    cons: ['功能相对单一', '生态较小', '性能优化少', '扩展性受限'],
    githubStars: '17k+',
    ecosystem: 'Python'
  }
];

export default function FrameworkComparisonMatrix() {
  const [view, setView] = useState<ComparisonView>('features');
  const [selectedFrameworks, setSelectedFrameworks] = useState<string[]>(['langchain', 'llamaindex']);

  const toggleFramework = (id: string) => {
    if (selectedFrameworks.includes(id)) {
      if (selectedFrameworks.length > 1) {
        setSelectedFrameworks(selectedFrameworks.filter(f => f !== id));
      }
    } else {
      setSelectedFrameworks([...selectedFrameworks, id]);
    }
  };

  const getFeatureIcon = (score: number) => {
    if (score === 2) return <CheckCircle className="w-5 h-5 text-green-600" />;
    if (score === 1) return <MinusCircle className="w-5 h-5 text-yellow-600" />;
    return <XCircle className="w-5 h-5 text-gray-400" />;
  };

  const getFeatureLabel = (score: number) => {
    if (score === 2) return '完全支持';
    if (score === 1) return '部分支持';
    return '不支持';
  };

  const selectedFrameworkData = frameworks.filter(f => selectedFrameworks.includes(f.id));
  const featureKeys = Object.keys(frameworks[0].features);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">LLM 框架对比矩阵</h3>
        <p className="text-gray-600">全面对比 LangChain、LlamaIndex、Haystack、AutoGen、CrewAI</p>
      </div>

      {/* 框架选择 */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-3">选择对比框架（至少选择 1 个）</label>
        <div className="grid grid-cols-5 gap-3">
          {frameworks.map(framework => (
            <button
              key={framework.id}
              onClick={() => toggleFramework(framework.id)}
              className={`p-3 rounded-lg border-2 transition-all ${
                selectedFrameworks.includes(framework.id)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <div className="text-3xl mb-2">{framework.logo}</div>
              <div className="font-semibold text-sm">{framework.name}</div>
              <div className="flex items-center justify-center gap-1 mt-1 text-xs text-gray-600">
                <Star className="w-3 h-3 text-yellow-500" />
                {framework.githubStars}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 视图切换 */}
      <div className="flex gap-2 mb-6">
        {[
          { id: 'features' as ComparisonView, label: '功能对比' },
          { id: 'use-cases' as ComparisonView, label: '适用场景' },
          { id: 'recommendations' as ComparisonView, label: '优劣分析' }
        ].map(v => (
          <button
            key={v.id}
            onClick={() => setView(v.id)}
            className={`px-4 py-2 rounded ${
              view === v.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>

      {/* 功能对比视图 */}
      {view === 'features' && (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-gray-100">
                <th className="border p-3 text-left font-semibold">功能</th>
                {selectedFrameworkData.map(framework => (
                  <th key={framework.id} className="border p-3 text-center">
                    <div className="text-2xl mb-1">{framework.logo}</div>
                    <div className="font-semibold">{framework.name}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {featureKeys.map((feature, idx) => (
                <motion.tr
                  key={feature}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="hover:bg-gray-50"
                >
                  <td className="border p-3 font-medium">{feature}</td>
                  {selectedFrameworkData.map(framework => {
                    const score = framework.features[feature];
                    return (
                      <td key={framework.id} className="border p-3 text-center">
                        <div className="flex flex-col items-center gap-1">
                          {getFeatureIcon(score)}
                          <span className="text-xs text-gray-600">{getFeatureLabel(score)}</span>
                        </div>
                      </td>
                    );
                  })}
                </motion.tr>
              ))}
              
              {/* 总分 */}
              <tr className="bg-blue-50 font-semibold">
                <td className="border p-3">功能完整度</td>
                {selectedFrameworkData.map(framework => {
                  const total = Object.values(framework.features).reduce((a, b) => a + b, 0);
                  const maxScore = featureKeys.length * 2;
                  const percentage = Math.round((total / maxScore) * 100);
                  return (
                    <td key={framework.id} className="border p-3 text-center">
                      <div className="flex flex-col items-center gap-2">
                        <div className="text-2xl text-blue-600">{percentage}%</div>
                        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-600"
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                  );
                })}
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* 适用场景视图 */}
      {view === 'use-cases' && (
        <div className="grid gap-4">
          {selectedFrameworkData.map((framework, idx) => (
            <motion.div
              key={framework.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="p-4 border-2 border-gray-300 rounded-lg"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="text-4xl">{framework.logo}</div>
                <div>
                  <h4 className="text-lg font-semibold">{framework.name}</h4>
                  <p className="text-sm text-gray-600">{framework.tagline}</p>
                </div>
              </div>
              
              <div className="mb-3">
                <h5 className="text-sm font-semibold mb-2">💡 最佳适用场景</h5>
                <div className="flex flex-wrap gap-2">
                  {framework.useCases.map((useCase, i) => (
                    <span key={i} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                      {useCase}
                    </span>
                  ))}
                </div>
              </div>

              <div className="text-xs text-gray-500 flex items-center gap-2">
                <span>生态:</span>
                <span className="font-mono bg-gray-100 px-2 py-1 rounded">{framework.ecosystem}</span>
                <span className="ml-auto flex items-center gap-1">
                  <Star className="w-3 h-3 text-yellow-500" />
                  {framework.githubStars}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* 优劣分析视图 */}
      {view === 'recommendations' && (
        <div className="grid gap-4">
          {selectedFrameworkData.map((framework, idx) => (
            <motion.div
              key={framework.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="p-4 border-2 border-gray-300 rounded-lg"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="text-4xl">{framework.logo}</div>
                <div className="flex-1">
                  <h4 className="text-lg font-semibold">{framework.name}</h4>
                  <p className="text-sm text-gray-600">{framework.tagline}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h5 className="text-sm font-semibold mb-2 text-green-700 flex items-center gap-1">
                    <CheckCircle className="w-4 h-4" />
                    优势
                  </h5>
                  <ul className="space-y-1">
                    {framework.pros.map((pro, i) => (
                      <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                        <span className="text-green-600">+</span>
                        <span>{pro}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h5 className="text-sm font-semibold mb-2 text-red-700 flex items-center gap-1">
                    <XCircle className="w-4 h-4" />
                    劣势
                  </h5>
                  <ul className="space-y-1">
                    {framework.cons.map((con, i) => (
                      <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                        <span className="text-red-600">-</span>
                        <span>{con}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* 推荐建议 */}
      <div className="mt-6 p-4 bg-blue-50 border-2 border-blue-200 rounded-lg">
        <div className="flex items-start gap-3">
          <TrendingUp className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-blue-900 mb-2">选择建议</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• <strong>通用 LLM 应用</strong> → LangChain（最成熟生态）</li>
              <li>• <strong>复杂 RAG 系统</strong> → LlamaIndex（专业索引）</li>
              <li>• <strong>企业搜索升级</strong> → Haystack（传统 NLP 友好）</li>
              <li>• <strong>多 Agent 研究</strong> → AutoGen（自主对话）</li>
              <li>• <strong>业务流程自动化</strong> → CrewAI（角色清晰）</li>
              <li>• <strong>混合使用</strong> → LangChain + LlamaIndex（通用编排 + 高级 RAG）</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
