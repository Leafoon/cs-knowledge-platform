"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, ArrowRight, CheckCircle2, AlertCircle, Code2, Play } from 'lucide-react';

type FlowStage = 'idle' | 'tool_definition' | 'binding' | 'llm_decision' | 'param_extraction' | 'execution' | 'result_return' | 'final_response';

const STAGES = [
  { id: 'tool_definition', label: '工具定义', icon: '🔧', description: '使用 @tool 装饰器定义工具函数' },
  { id: 'binding', label: '工具绑定', icon: '🔗', description: '将工具附加到 LLM 模型' },
  { id: 'llm_decision', label: 'LLM 决策', icon: '🤔', description: '模型判断是否需要调用工具' },
  { id: 'param_extraction', label: '参数提取', icon: '📝', description: '生成工具调用的参数' },
  { id: 'execution', label: '工具执行', icon: '⚡', description: '实际执行工具函数' },
  { id: 'result_return', label: '结果返回', icon: '↩️', description: '将执行结果返回给 LLM' },
  { id: 'final_response', label: '最终响应', icon: '💬', description: '基于工具结果生成回复' }
];

const EXAMPLE_SCENARIOS = {
  weather: {
    name: '天气查询',
    userQuery: "What's the weather in Beijing?",
    toolName: 'get_weather',
    toolArgs: { city: 'Beijing' },
    toolResult: 'Sunny, 15°C',
    finalResponse: 'The weather in Beijing is sunny with a temperature of 15°C.'
  },
  calculator: {
    name: '数学计算',
    userQuery: "What is 23 times 47?",
    toolName: 'calculator',
    toolArgs: { expression: '23 * 47' },
    toolResult: '1081',
    finalResponse: '23 times 47 equals 1,081.'
  },
  search: {
    name: '信息搜索',
    userQuery: "Search for LangChain documentation",
    toolName: 'search_web',
    toolArgs: { query: 'LangChain documentation' },
    toolResult: 'Found: https://python.langchain.com/docs/',
    finalResponse: 'I found the LangChain documentation at https://python.langchain.com/docs/'
  }
};

export default function ToolCallingFlow() {
  const [currentStage, setCurrentStage] = useState<FlowStage>('idle');
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<keyof typeof EXAMPLE_SCENARIOS>('weather');
  const [isAnimating, setIsAnimating] = useState(false);

  const scenario = EXAMPLE_SCENARIOS[selectedScenario];

  const runFlow = () => {
    setIsAnimating(true);
    setCompletedStages([]);
    setCurrentStage('tool_definition');

    let stageIndex = 0;
    const interval = setInterval(() => {
      if (stageIndex >= STAGES.length) {
        clearInterval(interval);
        setCurrentStage('idle');
        setIsAnimating(false);
        return;
      }

      const stage = STAGES[stageIndex];
      setCurrentStage(stage.id as FlowStage);
      
      setTimeout(() => {
        setCompletedStages(prev => [...prev, stage.id]);
      }, 800);

      stageIndex++;
    }, 1200);
  };

  const getStageStatus = (stageId: string) => {
    if (completedStages.includes(stageId)) return 'completed';
    if (currentStage === stageId) return 'active';
    return 'pending';
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Tool Calling 完整生命周期</h3>
        <p className="text-slate-600">从工具定义到最终响应的 7 个关键步骤</p>
      </div>

      {/* Scenario Selection */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <label className="block text-sm font-semibold text-slate-700 mb-3">
          选择场景：
        </label>
        <div className="grid grid-cols-3 gap-3">
          {(Object.keys(EXAMPLE_SCENARIOS) as Array<keyof typeof EXAMPLE_SCENARIOS>).map(key => (
            <button
              key={key}
              onClick={() => setSelectedScenario(key)}
              className={`p-3 rounded-lg border-2 transition-all ${
                selectedScenario === key
                  ? 'bg-indigo-50 border-indigo-300 shadow-md'
                  : 'bg-white border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="font-semibold text-sm">{EXAMPLE_SCENARIOS[key].name}</div>
              <div className="text-xs text-slate-600 mt-1">{EXAMPLE_SCENARIOS[key].toolName}</div>
            </button>
          ))}
        </div>
      </div>

      {/* User Query */}
      <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold">
            👤
          </div>
          <span className="font-semibold text-blue-800">用户输入</span>
        </div>
        <p className="text-blue-900 ml-10">{scenario.userQuery}</p>
      </div>

      {/* Flow Stages */}
      <div className="space-y-3 mb-6">
        {STAGES.map((stage, idx) => {
          const status = getStageStatus(stage.id);
          return (
            <motion.div
              key={stage.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`relative p-4 rounded-lg border-2 transition-all ${
                status === 'active'
                  ? 'bg-indigo-50 border-indigo-400 shadow-lg'
                  : status === 'completed'
                  ? 'bg-green-50 border-green-300'
                  : 'bg-white border-slate-200'
              }`}
            >
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center text-2xl ${
                  status === 'completed'
                    ? 'bg-green-100'
                    : status === 'active'
                    ? 'bg-indigo-100 animate-pulse'
                    : 'bg-slate-100'
                }`}>
                  {status === 'completed' ? '✓' : stage.icon}
                </div>

                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-slate-800">{stage.label}</span>
                    {status === 'active' && (
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Zap className="w-4 h-4 text-indigo-500" />
                      </motion.div>
                    )}
                  </div>
                  <p className="text-sm text-slate-600 mt-1">{stage.description}</p>

                  {/* Stage-specific content */}
                  {status === 'active' && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-3 p-3 bg-white rounded border border-indigo-200"
                    >
                      {stage.id === 'tool_definition' && (
                        <pre className="text-xs font-mono">
{`@tool
def ${scenario.toolName}(...) -> str:
    """Tool function"""
    ...`}
                        </pre>
                      )}
                      {stage.id === 'binding' && (
                        <pre className="text-xs font-mono">
{`model_with_tools = model.bind_tools([${scenario.toolName}])`}
                        </pre>
                      )}
                      {stage.id === 'llm_decision' && (
                        <div className="text-xs">
                          <span className="font-semibold">决策：</span>需要调用 <code className="bg-indigo-100 px-1 rounded">{scenario.toolName}</code>
                        </div>
                      )}
                      {stage.id === 'param_extraction' && (
                        <pre className="text-xs font-mono">
{JSON.stringify(scenario.toolArgs, null, 2)}
                        </pre>
                      )}
                      {stage.id === 'execution' && (
                        <div className="text-xs">
                          <span className="font-semibold">执行结果：</span>
                          <code className="bg-green-100 px-1 rounded ml-1">{scenario.toolResult}</code>
                        </div>
                      )}
                      {stage.id === 'result_return' && (
                        <pre className="text-xs font-mono">
{`ToolMessage(
  content="${scenario.toolResult}",
  tool_call_id="call_123"
)`}
                        </pre>
                      )}
                      {stage.id === 'final_response' && (
                        <div className="text-xs text-green-700 font-medium">
                          {scenario.finalResponse}
                        </div>
                      )}
                    </motion.div>
                  )}
                </div>

                {status === 'completed' && (
                  <CheckCircle2 className="w-6 h-6 text-green-500" />
                )}
                {idx < STAGES.length - 1 && (
                  <ArrowRight className={`w-5 h-5 ${
                    status === 'completed' ? 'text-green-500' : 'text-slate-300'
                  }`} />
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Control Button */}
      <button
        onClick={runFlow}
        disabled={isAnimating}
        className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-lg hover:from-indigo-600 hover:to-purple-600 disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed transition-all font-semibold shadow-lg"
      >
        <Play className={`w-5 h-5 ${isAnimating ? 'animate-pulse' : ''}`} />
        {isAnimating ? '执行中...' : '开始执行流程'}
      </button>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <div className="flex items-center gap-2 mb-3">
          <Code2 className="w-5 h-5" />
          <span className="font-semibold">完整代码示例</span>
        </div>
        <pre className="text-xs font-mono overflow-x-auto">
{`from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 1. 定义工具
@tool
def ${scenario.toolName}(${Object.keys(scenario.toolArgs).join(', ')}) -> str:
    """Tool description."""
    return "${scenario.toolResult}"

# 2. 绑定工具
model = ChatOpenAI(model="gpt-4")
model_with_tools = model.bind_tools([${scenario.toolName}])

# 3. 调用
response = model_with_tools.invoke("${scenario.userQuery}")

# 4. 执行工具
tool_call = response.tool_calls[0]
tool_result = ${scenario.toolName}.invoke(tool_call["args"])

# 5. 返回结果给 LLM
messages = [
    HumanMessage(content="${scenario.userQuery}"),
    AIMessage(content="", tool_calls=response.tool_calls),
    ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
]

final_response = model.invoke(messages)
print(final_response.content)
# "${scenario.finalResponse}"`}
        </pre>
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-indigo-600">{STAGES.length}</div>
          <div className="text-sm text-slate-600 mt-1">执行步骤</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-green-600">{completedStages.length}</div>
          <div className="text-sm text-slate-600 mt-1">已完成</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-purple-600">
            {completedStages.length === STAGES.length ? '100' : Math.round((completedStages.length / STAGES.length) * 100)}%
          </div>
          <div className="text-sm text-slate-600 mt-1">进度</div>
        </div>
      </div>
    </div>
  );
}
