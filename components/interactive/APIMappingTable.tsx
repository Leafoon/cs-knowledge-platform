"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Code, Copy, Check } from 'lucide-react';

type APIMapping = {
  category: string;
  mappings: {
    concept: string;
    langchain: string;
    llamaindex: string;
    haystack: string;
    notes?: string;
  }[];
};

const apiMappings: APIMapping[] = [
  {
    category: '文档加载',
    mappings: [
      {
        concept: '目录加载',
        langchain: 'DirectoryLoader',
        llamaindex: 'SimpleDirectoryReader',
        haystack: 'PDFToTextConverter + FileTypeClassifier',
        notes: 'Haystack 需要多个组件配合'
      },
      {
        concept: 'PDF 加载',
        langchain: 'PyPDFLoader',
        llamaindex: 'SimpleDirectoryReader (自动识别)',
        haystack: 'PDFToTextConverter'
      },
      {
        concept: 'CSV 加载',
        langchain: 'CSVLoader',
        llamaindex: 'SimpleDirectoryReader',
        haystack: 'CSVToDocument'
      }
    ]
  },
  {
    category: '文本分割',
    mappings: [
      {
        concept: '递归分割',
        langchain: 'RecursiveCharacterTextSplitter',
        llamaindex: 'SentenceSplitter',
        haystack: 'TextCleaner + DocumentSplitter'
      },
      {
        concept: '语义分割',
        langchain: 'SemanticChunker',
        llamaindex: 'SemanticSplitterNodeParser',
        haystack: '不支持',
        notes: 'Haystack 无原生语义分割'
      }
    ]
  },
  {
    category: '向量存储',
    mappings: [
      {
        concept: 'FAISS',
        langchain: 'FAISS.from_documents()',
        llamaindex: 'VectorStoreIndex.from_documents() + FaissVectorStore',
        haystack: 'FAISSDocumentStore'
      },
      {
        concept: 'Chroma',
        langchain: 'Chroma.from_documents()',
        llamaindex: 'VectorStoreIndex + ChromaVectorStore',
        haystack: '不支持'
      },
      {
        concept: 'Pinecone',
        langchain: 'Pinecone.from_documents()',
        llamaindex: 'VectorStoreIndex + PineconeVectorStore',
        haystack: 'PineconeDocumentStore'
      }
    ]
  },
  {
    category: '检索器',
    mappings: [
      {
        concept: '向量检索',
        langchain: 'vectorstore.as_retriever()',
        llamaindex: 'index.as_retriever()',
        haystack: 'EmbeddingRetriever'
      },
      {
        concept: 'BM25 检索',
        langchain: 'BM25Retriever (community)',
        llamaindex: 'BM25Retriever',
        haystack: 'BM25Retriever'
      },
      {
        concept: '混合检索',
        langchain: 'EnsembleRetriever',
        llamaindex: 'QueryFusionRetriever',
        haystack: 'Pipeline (多个 Retriever)'
      }
    ]
  },
  {
    category: 'LLM 调用',
    mappings: [
      {
        concept: 'OpenAI',
        langchain: 'ChatOpenAI()',
        llamaindex: 'OpenAI()',
        haystack: 'PromptNode (OpenAI provider)'
      },
      {
        concept: 'Anthropic',
        langchain: 'ChatAnthropic()',
        llamaindex: 'Anthropic()',
        haystack: '社区插件'
      },
      {
        concept: '本地模型',
        langchain: 'Ollama() / HuggingFacePipeline()',
        llamaindex: 'Ollama()',
        haystack: 'HuggingFaceLocalInvocationLayer'
      }
    ]
  },
  {
    category: '链/查询引擎',
    mappings: [
      {
        concept: 'RAG 查询',
        langchain: 'RetrievalQA.from_chain_type()',
        llamaindex: 'index.as_query_engine()',
        haystack: 'Pipeline (Retriever + PromptNode)'
      },
      {
        concept: '对话式 RAG',
        langchain: 'ConversationalRetrievalChain',
        llamaindex: 'CondenseQuestionChatEngine',
        haystack: 'ConversationalAgent + Pipeline'
      },
      {
        concept: 'Agent',
        langchain: 'create_openai_functions_agent()',
        llamaindex: 'OpenAIAgent()',
        haystack: 'Agent (基础实现)'
      }
    ]
  },
  {
    category: '记忆',
    mappings: [
      {
        concept: '缓冲记忆',
        langchain: 'ConversationBufferMemory',
        llamaindex: 'ChatMemoryBuffer',
        haystack: 'ConversationMemory'
      },
      {
        concept: '摘要记忆',
        langchain: 'ConversationSummaryMemory',
        llamaindex: 'ChatSummaryMemoryBuffer',
        haystack: '不支持'
      },
      {
        concept: '向量记忆',
        langchain: 'VectorStoreRetrieverMemory',
        llamaindex: 'VectorMemory',
        haystack: '不支持'
      }
    ]
  },
  {
    category: '评估',
    mappings: [
      {
        concept: '忠实度评估',
        langchain: 'LangSmith (自定义评估器)',
        llamaindex: 'FaithfulnessEvaluator',
        haystack: 'SemanticAnswerSimilarity'
      },
      {
        concept: '相关性评估',
        langchain: 'LangSmith Evaluators',
        llamaindex: 'RelevancyEvaluator',
        haystack: 'Recall / F1 (传统 NLP 指标)'
      },
      {
        concept: '追踪',
        langchain: 'LangSmith Tracing',
        llamaindex: 'LlamaIndex Observability',
        haystack: 'Pipeline Logging'
      }
    ]
  }
];

export default function APIMappingTable() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFramework, setSelectedFramework] = useState<'all' | 'langchain' | 'llamaindex' | 'haystack'>('all');
  const [copiedText, setCopiedText] = useState<string | null>(null);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(text);
    setTimeout(() => setCopiedText(null), 2000);
  };

  const filteredMappings = apiMappings.map(category => ({
    ...category,
    mappings: category.mappings.filter(mapping =>
      mapping.concept.toLowerCase().includes(searchTerm.toLowerCase()) ||
      mapping.langchain.toLowerCase().includes(searchTerm.toLowerCase()) ||
      mapping.llamaindex.toLowerCase().includes(searchTerm.toLowerCase()) ||
      mapping.haystack.toLowerCase().includes(searchTerm.toLowerCase())
    )
  })).filter(category => category.mappings.length > 0);

  const CodeCell = ({ code, framework }: { code: string; framework: string }) => {
    const isNotSupported = code === '不支持' || code.includes('不支持');
    
    return (
      <div className={`relative group ${isNotSupported ? 'text-gray-400 italic' : ''}`}>
        <code className="text-xs font-mono">{code}</code>
        {!isNotSupported && (
          <button
            onClick={() => copyToClipboard(code)}
            className="absolute right-0 top-0 opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-gray-200 rounded"
            title="复制代码"
          >
            {copiedText === code ? (
              <Check className="w-3 h-3 text-green-600" />
            ) : (
              <Copy className="w-3 h-3 text-gray-600" />
            )}
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">API 映射对照表</h3>
        <p className="text-gray-600">快速查找不同框架间的 API 对应关系</p>
      </div>

      {/* 搜索和筛选 */}
      <div className="mb-6 flex gap-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="搜索 API、概念或功能..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="flex gap-2">
          {(['all', 'langchain', 'llamaindex', 'haystack'] as const).map(framework => (
            <button
              key={framework}
              onClick={() => setSelectedFramework(framework)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedFramework === framework
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {framework === 'all' ? '全部' : 
               framework === 'langchain' ? 'LangChain' :
               framework === 'llamaindex' ? 'LlamaIndex' : 'Haystack'}
            </button>
          ))}
        </div>
      </div>

      {/* 映射表格 */}
      <div className="space-y-6">
        {filteredMappings.map((category, categoryIndex) => (
          <motion.div
            key={category.category}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: categoryIndex * 0.1 }}
          >
            <h4 className="font-semibold mb-3 flex items-center gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                {category.category}
              </span>
              <span className="text-xs text-gray-500">
                {category.mappings.length} 项
              </span>
            </h4>

            <div className="overflow-x-auto border rounded-lg">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">概念/功能</th>
                    {(selectedFramework === 'all' || selectedFramework === 'langchain') && (
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                        🦜 LangChain
                      </th>
                    )}
                    {(selectedFramework === 'all' || selectedFramework === 'llamaindex') && (
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                        🦙 LlamaIndex
                      </th>
                    )}
                    {(selectedFramework === 'all' || selectedFramework === 'haystack') && (
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                        🌾 Haystack
                      </th>
                    )}
                    {selectedFramework === 'all' && (
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">备注</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {category.mappings.map((mapping, index) => (
                    <motion.tr
                      key={index}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: categoryIndex * 0.1 + index * 0.05 }}
                      className="border-t hover:bg-gray-50"
                    >
                      <td className="px-4 py-3 font-medium text-sm">{mapping.concept}</td>
                      {(selectedFramework === 'all' || selectedFramework === 'langchain') && (
                        <td className="px-4 py-3">
                          <CodeCell code={mapping.langchain} framework="langchain" />
                        </td>
                      )}
                      {(selectedFramework === 'all' || selectedFramework === 'llamaindex') && (
                        <td className="px-4 py-3">
                          <CodeCell code={mapping.llamaindex} framework="llamaindex" />
                        </td>
                      )}
                      {(selectedFramework === 'all' || selectedFramework === 'haystack') && (
                        <td className="px-4 py-3">
                          <CodeCell code={mapping.haystack} framework="haystack" />
                        </td>
                      )}
                      {selectedFramework === 'all' && mapping.notes && (
                        <td className="px-4 py-3 text-xs text-gray-600">{mapping.notes}</td>
                      )}
                      {selectedFramework === 'all' && !mapping.notes && (
                        <td className="px-4 py-3"></td>
                      )}
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        ))}
      </div>

      {filteredMappings.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <Code className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>未找到匹配的 API 映射</p>
          <p className="text-sm mt-1">尝试搜索其他关键词</p>
        </div>
      )}

      {/* 使用提示 */}
      <div className="mt-6 p-4 bg-blue-50 border-2 border-blue-200 rounded-lg">
        <div className="flex items-start gap-3">
          <Code className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-blue-900 mb-2">使用提示</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• 悬停在代码上可复制到剪贴板</li>
              <li>• "不支持"表示该框架无对应功能，需自定义实现</li>
              <li>• 部分 API 名称相似但参数可能不同，请查阅官方文档</li>
              <li>• 混合使用多框架时，注意版本兼容性</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
