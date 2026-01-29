"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, XCircle, AlertCircle, ArrowRight, RefreshCw, Code2 } from 'lucide-react';

type ParserType = 'pydantic' | 'fixing' | 'retry' | 'json';

interface FlowStep {
  id: string;
  label: string;
  status: 'pending' | 'processing' | 'success' | 'error';
  output?: string;
}

const SAMPLE_OUTPUTS = {
  valid: `{
  "name": "John Doe",
  "age": 30,
  "email": "john@example.com",
  "occupation": "Software Engineer"
}`,
  invalid_age: `{
  "name": "Jane Smith",
  "age": "twenty-five",
  "email": "jane@example.com",
  "occupation": "Designer"
}`,
  missing_field: `{
  "name": "Bob Wilson",
  "age": 35,
  "email": "bob@example.com"
}`,
  malformed: `{
  "name": "Alice Brown",
  age: 28
  "email": "alice@example.com"
}`
};

const PARSER_CONFIGS = {
  pydantic: {
    name: 'PydanticOutputParser',
    description: '基础解析器，严格验证 JSON Schema',
    color: 'blue',
    steps: [
      { id: 'receive', label: '接收 LLM 输出' },
      { id: 'validate', label: '验证 JSON 格式' },
      { id: 'parse', label: '解析为 Pydantic 对象' },
      { id: 'typecheck', label: '类型检查' },
      { id: 'result', label: '返回结果' }
    ]
  },
  fixing: {
    name: 'OutputFixingParser',
    description: '自动修复格式错误',
    color: 'green',
    steps: [
      { id: 'receive', label: '接收 LLM 输出' },
      { id: 'try_parse', label: '尝试解析' },
      { id: 'detect_error', label: '检测错误' },
      { id: 'fix', label: 'LLM 修复错误' },
      { id: 'reparse', label: '重新解析' },
      { id: 'result', label: '返回结果' }
    ]
  },
  retry: {
    name: 'RetryWithErrorOutputParser',
    description: '失败时重新调用 LLM',
    color: 'orange',
    steps: [
      { id: 'receive', label: '接收 LLM 输出' },
      { id: 'try_parse', label: '尝试解析' },
      { id: 'detect_error', label: '检测错误' },
      { id: 'retry', label: '发送错误信息给 LLM' },
      { id: 'new_output', label: '获取新输出' },
      { id: 'reparse', label: '解析新输出' },
      { id: 'result', label: '返回结果' }
    ]
  },
  json: {
    name: 'JsonOutputParser',
    description: '简单 JSON 解析',
    color: 'purple',
    steps: [
      { id: 'receive', label: '接收 LLM 输出' },
      { id: 'extract', label: '提取 JSON 内容' },
      { id: 'parse', label: 'JSON.parse()' },
      { id: 'result', label: '返回字典对象' }
    ]
  }
};

export default function OutputParserFlow() {
  const [parserType, setParserType] = useState<ParserType>('pydantic');
  const [outputType, setOutputType] = useState<keyof typeof SAMPLE_OUTPUTS>('valid');
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [flowSteps, setFlowSteps] = useState<FlowStep[]>([]);

  const config = PARSER_CONFIGS[parserType];

  const runParser = () => {
    setIsAnimating(true);
    setCurrentStep(0);
    
    const initialSteps = config.steps.map(step => ({
      ...step,
      status: 'pending' as const
    }));
    setFlowSteps(initialSteps);

    // 模拟解析流程
    let stepIndex = 0;
    const interval = setInterval(() => {
      if (stepIndex >= config.steps.length) {
        clearInterval(interval);
        setIsAnimating(false);
        return;
      }

      setFlowSteps(prev => prev.map((step, idx) => {
        if (idx === stepIndex) {
          // 根据解析器类型和输出类型决定结果
          let status: FlowStep['status'] = 'processing';
          
          if (idx === config.steps.length - 1) {
            // 最后一步
            if (parserType === 'pydantic' && outputType !== 'valid') {
              status = 'error';
            } else {
              status = 'success';
            }
          } else if (parserType === 'pydantic' && outputType !== 'valid' && step.id === 'parse') {
            status = 'error';
          } else if (idx < stepIndex) {
            status = 'success';
          }
          
          return { ...step, status };
        } else if (idx < stepIndex) {
          return { ...step, status: 'success' };
        }
        return step;
      }));

      setCurrentStep(stepIndex);
      stepIndex++;
    }, 800);
  };

  const getStepIcon = (status: FlowStep['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'processing':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-slate-300" />;
    }
  };

  const getColorClasses = (color: string) => {
    const colors = {
      blue: 'bg-blue-500/10 text-blue-700 border-blue-200',
      green: 'bg-green-500/10 text-green-700 border-green-200',
      orange: 'bg-orange-500/10 text-orange-700 border-orange-200',
      purple: 'bg-purple-500/10 text-purple-700 border-purple-200'
    };
    return colors[color as keyof typeof colors] || colors.blue;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Output Parser 工作流程</h3>
        <p className="text-slate-600">对比不同解析器的处理流程与容错机制</p>
      </div>

      {/* Parser Selection */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {(Object.keys(PARSER_CONFIGS) as ParserType[]).map(type => {
          const cfg = PARSER_CONFIGS[type];
          return (
            <button
              key={type}
              onClick={() => setParserType(type)}
              className={`p-4 rounded-lg border-2 transition-all ${
                parserType === type
                  ? `${getColorClasses(cfg.color)} border-current shadow-md`
                  : 'bg-white border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="font-semibold text-sm mb-1">{cfg.name}</div>
              <div className="text-xs opacity-70">{cfg.description}</div>
            </button>
          );
        })}
      </div>

      {/* Output Type Selection */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <label className="block text-sm font-semibold text-slate-700 mb-3">
          选择输入类型：
        </label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {(Object.keys(SAMPLE_OUTPUTS) as Array<keyof typeof SAMPLE_OUTPUTS>).map(type => (
            <button
              key={type}
              onClick={() => setOutputType(type)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                outputType === type
                  ? 'bg-blue-500 text-white shadow-md'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              {type === 'valid' && '✓ 有效输出'}
              {type === 'invalid_age' && '⚠ 类型错误'}
              {type === 'missing_field' && '⚠ 缺少字段'}
              {type === 'malformed' && '✗ 格式错误'}
            </button>
          ))}
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Input */}
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Code2 className="w-5 h-5 text-slate-600" />
            <h4 className="font-semibold text-slate-800">LLM 输出</h4>
          </div>
          <pre className="bg-slate-50 p-3 rounded text-xs font-mono overflow-x-auto border border-slate-200">
            {SAMPLE_OUTPUTS[outputType]}
          </pre>
        </div>

        {/* Expected Schema */}
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Code2 className="w-5 h-5 text-slate-600" />
            <h4 className="font-semibold text-slate-800">预期 Schema</h4>
          </div>
          <pre className="bg-slate-50 p-3 rounded text-xs font-mono overflow-x-auto border border-slate-200">
{`class Person(BaseModel):
    name: str
    age: int
    email: str
    occupation: str`}
          </pre>
        </div>
      </div>

      {/* Flow Visualization */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-slate-800">执行流程</h4>
          <button
            onClick={runParser}
            disabled={isAnimating}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${isAnimating ? 'animate-spin' : ''}`} />
            {isAnimating ? '执行中...' : '开始解析'}
          </button>
        </div>

        <div className="space-y-3">
          {flowSteps.map((step, idx) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`flex items-center gap-4 p-4 rounded-lg border-2 transition-all ${
                step.status === 'processing'
                  ? 'bg-blue-50 border-blue-300 shadow-md'
                  : step.status === 'success'
                  ? 'bg-green-50 border-green-200'
                  : step.status === 'error'
                  ? 'bg-red-50 border-red-200'
                  : 'bg-slate-50 border-slate-200'
              }`}
            >
              <div className="flex-shrink-0">
                {getStepIcon(step.status)}
              </div>
              <div className="flex-1">
                <div className="font-medium text-slate-800">{step.label}</div>
                {step.status === 'error' && (
                  <div className="text-sm text-red-600 mt-1">
                    {outputType === 'invalid_age' && '类型不匹配：age 应为整数'}
                    {outputType === 'missing_field' && '缺少必填字段：occupation'}
                    {outputType === 'malformed' && 'JSON 格式错误'}
                  </div>
                )}
              </div>
              {idx < config.steps.length - 1 && (
                <ArrowRight className="w-4 h-4 text-slate-400" />
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Result */}
      {flowSteps.length > 0 && !isAnimating && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-4 rounded-lg border-2 ${
            flowSteps[flowSteps.length - 1].status === 'success'
              ? 'bg-green-50 border-green-300'
              : 'bg-red-50 border-red-300'
          }`}
        >
          <div className="flex items-start gap-3">
            {flowSteps[flowSteps.length - 1].status === 'success' ? (
              <CheckCircle2 className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
            ) : (
              <XCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1">
              <h4 className="font-semibold text-slate-800 mb-2">
                {flowSteps[flowSteps.length - 1].status === 'success' ? '✓ 解析成功' : '✗ 解析失败'}
              </h4>
              {flowSteps[flowSteps.length - 1].status === 'success' ? (
                <div className="text-sm text-slate-600">
                  <p className="mb-2">已成功解析为 Person 对象：</p>
                  <pre className="bg-white p-3 rounded text-xs font-mono border border-green-200">
{parserType === 'fixing' && outputType !== 'valid'
  ? `Person(
    name="Jane Smith",
    age=25,  # 已自动修复
    email="jane@example.com",
    occupation="Designer"
)`
  : parserType === 'retry' && outputType !== 'valid'
  ? `Person(
    name="Bob Wilson",
    age=35,
    email="bob@example.com",
    occupation="Unknown"  # 重试时补充
)`
  : `Person(
    name="John Doe",
    age=30,
    email="john@example.com",
    occupation="Software Engineer"
)`}
                  </pre>
                </div>
              ) : (
                <div className="text-sm text-red-700">
                  <p className="font-medium mb-1">ValidationError:</p>
                  <p className="font-mono text-xs">
                    {outputType === 'invalid_age' && "Field 'age': Input should be a valid integer"}
                    {outputType === 'missing_field' && "Field 'occupation': Field required"}
                    {outputType === 'malformed' && "JSONDecodeError: Expecting ',' delimiter"}
                  </p>
                  <p className="mt-2 text-red-600">
                    💡 建议使用 OutputFixingParser 或 RetryOutputParser
                  </p>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-50 rounded-lg border border-slate-200">
        <h4 className="font-semibold text-slate-800 mb-3">代码示例</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{parserType === 'pydantic' && `from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Person)
result = parser.parse(llm_output)  # 可能抛出 ValidationError`}

{parserType === 'fixing' && `from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=Person)
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4")
)
result = fixing_parser.parse(llm_output)  # 自动修复错误`}

{parserType === 'retry' && `from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4")
)
result = retry_parser.parse_with_prompt(
    llm_output,
    original_prompt
)  # 带错误信息重试`}

{parserType === 'json' && `from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
result = parser.parse(llm_output)  # 返回 dict`}
        </pre>
      </div>
    </div>
  );
}
