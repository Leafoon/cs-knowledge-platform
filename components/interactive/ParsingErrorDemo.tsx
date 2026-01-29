"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, CheckCircle2, XCircle, Zap, Code2, RefreshCw } from 'lucide-react';

type ErrorType = 'type_mismatch' | 'missing_field' | 'json_invalid' | 'validation_fail';
type Strategy = 'base' | 'fixing' | 'retry' | 'custom_validator';

interface ErrorCase {
  id: ErrorType;
  name: string;
  description: string;
  input: string;
  error: string;
}

const ERROR_CASES: ErrorCase[] = [
  {
    id: 'type_mismatch',
    name: '类型不匹配',
    description: 'age 字段应为整数，但收到字符串',
    input: `{
  "name": "Alice",
  "age": "twenty-five",
  "email": "alice@example.com"
}`,
    error: "ValidationError: 1 validation error for Person\nage\n  Input should be a valid integer, unable to parse string as an integer"
  },
  {
    id: 'missing_field',
    name: '缺少必填字段',
    description: '缺少 email 字段',
    input: `{
  "name": "Bob",
  "age": 30
}`,
    error: "ValidationError: 1 validation error for Person\nemail\n  Field required"
  },
  {
    id: 'json_invalid',
    name: 'JSON 格式错误',
    description: '缺少逗号分隔符',
    input: `{
  "name": "Charlie"
  "age": 35,
  "email": "charlie@example.com"
}`,
    error: "JSONDecodeError: Expecting ',' delimiter: line 3 column 3 (char 25)"
  },
  {
    id: 'validation_fail',
    name: '自定义验证失败',
    description: 'email 格式不正确',
    input: `{
  "name": "David",
  "age": 28,
  "email": "invalid-email"
}`,
    error: "ValidationError: 1 validation error for Person\nemail\n  Value error, Invalid email format"
  }
];

const STRATEGIES = {
  base: {
    name: 'PydanticOutputParser',
    description: '基础解析器，直接抛出异常',
    color: 'red',
    canFix: false
  },
  fixing: {
    name: 'OutputFixingParser',
    description: '使用 LLM 自动修复错误',
    color: 'green',
    canFix: true
  },
  retry: {
    name: 'RetryWithErrorOutputParser',
    description: '将错误发送给 LLM 重新生成',
    color: 'blue',
    canFix: true
  },
  custom_validator: {
    name: 'Custom Validator',
    description: '自定义验证器处理边缘情况',
    color: 'purple',
    canFix: true
  }
};

export default function ParsingErrorDemo() {
  const [selectedError, setSelectedError] = useState<ErrorType>('type_mismatch');
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy>('base');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<{ success: boolean; output?: string; error?: string } | null>(null);

  const errorCase = ERROR_CASES.find(e => e.id === selectedError)!;
  const strategy = STRATEGIES[selectedStrategy];

  const processError = () => {
    setIsProcessing(true);
    setResult(null);

    setTimeout(() => {
      if (selectedStrategy === 'base') {
        // 基础解析器总是失败
        setResult({
          success: false,
          error: errorCase.error
        });
      } else {
        // 其他策略可以修复（模拟）
        let fixedOutput = '';
        
        switch (selectedError) {
          case 'type_mismatch':
            fixedOutput = `{
  "name": "Alice",
  "age": 25,
  "email": "alice@example.com"
}`;
            break;
          case 'missing_field':
            fixedOutput = `{
  "name": "Bob",
  "age": 30,
  "email": "bob@example.com"
}`;
            break;
          case 'json_invalid':
            fixedOutput = `{
  "name": "Charlie",
  "age": 35,
  "email": "charlie@example.com"
}`;
            break;
          case 'validation_fail':
            fixedOutput = `{
  "name": "David",
  "age": 28,
  "email": "david@example.com"
}`;
            break;
        }

        setResult({
          success: true,
          output: fixedOutput
        });
      }

      setIsProcessing(false);
    }, 1500);
  };

  const getStrategyColor = (color: string) => {
    const colors = {
      red: 'bg-red-500/10 text-red-700 border-red-200',
      green: 'bg-green-500/10 text-green-700 border-green-200',
      blue: 'bg-blue-500/10 text-blue-700 border-blue-200',
      purple: 'bg-purple-500/10 text-purple-700 border-purple-200'
    };
    return colors[color as keyof typeof colors];
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Parsing Error 容错演示</h3>
        <p className="text-slate-600">对比不同策略处理解析错误的能力</p>
      </div>

      {/* Error Case Selection */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <label className="block text-sm font-semibold text-slate-700 mb-3">
          选择错误类型：
        </label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {ERROR_CASES.map(errorCase => (
            <button
              key={errorCase.id}
              onClick={() => setSelectedError(errorCase.id)}
              className={`p-3 rounded-lg border-2 transition-all text-left ${
                selectedError === errorCase.id
                  ? 'bg-red-50 border-red-300 shadow-md'
                  : 'bg-white border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="font-semibold text-sm mb-1">{errorCase.name}</div>
              <div className="text-xs text-slate-600">{errorCase.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Strategy Selection */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <label className="block text-sm font-semibold text-slate-700 mb-3">
          选择处理策略：
        </label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {(Object.keys(STRATEGIES) as Strategy[]).map(strategyKey => {
            const strat = STRATEGIES[strategyKey];
            return (
              <button
                key={strategyKey}
                onClick={() => setSelectedStrategy(strategyKey)}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  selectedStrategy === strategyKey
                    ? `${getStrategyColor(strat.color)} border-current shadow-md`
                    : 'bg-white border-slate-200 hover:border-slate-300'
                }`}
              >
                <div className="font-semibold text-sm mb-1">{strat.name}</div>
                <div className="text-xs opacity-70">{strat.description}</div>
                {strat.canFix && (
                  <div className="mt-2 flex items-center gap-1 text-xs">
                    <Zap className="w-3 h-3" />
                    <span>自动修复</span>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Input */}
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-5 h-5 text-orange-500" />
            <h4 className="font-semibold text-slate-800">错误输入</h4>
          </div>
          <pre className="bg-red-50 p-3 rounded text-xs font-mono overflow-x-auto border border-red-200">
            {errorCase.input}
          </pre>
          <div className="mt-3 p-2 bg-yellow-50 rounded border border-yellow-200">
            <div className="text-xs font-semibold text-yellow-800 mb-1">问题：</div>
            <div className="text-xs text-yellow-700">{errorCase.description}</div>
          </div>
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
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError(
                'Invalid email format'
            )
        return v`}
          </pre>
        </div>
      </div>

      {/* Action Button */}
      <div className="mb-6">
        <button
          onClick={processError}
          disabled={isProcessing}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed transition-all font-semibold shadow-lg"
        >
          <RefreshCw className={`w-5 h-5 ${isProcessing ? 'animate-spin' : ''}`} />
          {isProcessing ? '处理中...' : `使用 ${strategy.name} 处理`}
        </button>
      </div>

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className={`p-6 rounded-lg border-2 ${
              result.success
                ? 'bg-green-50 border-green-300'
                : 'bg-red-50 border-red-300'
            }`}
          >
            <div className="flex items-start gap-3">
              {result.success ? (
                <CheckCircle2 className="w-7 h-7 text-green-600 flex-shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-7 h-7 text-red-600 flex-shrink-0 mt-0.5" />
              )}
              
              <div className="flex-1">
                <h4 className="text-lg font-bold text-slate-800 mb-3">
                  {result.success ? '✓ 修复成功' : '✗ 解析失败'}
                </h4>

                {result.success ? (
                  <div>
                    <p className="text-sm text-slate-700 mb-3">
                      {selectedStrategy === 'fixing' && '🔧 OutputFixingParser 自动修复了错误'}
                      {selectedStrategy === 'retry' && '🔄 RetryOutputParser 重新请求 LLM 并成功'}
                      {selectedStrategy === 'custom_validator' && '⚙️ 自定义验证器处理了异常情况'}
                    </p>
                    <div className="bg-white p-4 rounded-lg border border-green-200">
                      <div className="text-sm font-semibold text-slate-700 mb-2">修复后的输出：</div>
                      <pre className="text-xs font-mono overflow-x-auto">
                        {result.output}
                      </pre>
                    </div>

                    <div className="mt-4 p-3 bg-green-100 rounded-lg">
                      <div className="text-sm font-semibold text-green-800 mb-1">
                        修复步骤：
                      </div>
                      <ol className="text-xs text-green-700 space-y-1 list-decimal list-inside">
                        {selectedStrategy === 'fixing' && (
                          <>
                            <li>检测到错误：{errorCase.description}</li>
                            <li>将错误输出发送给 LLM</li>
                            <li>LLM 分析并修复错误</li>
                            <li>重新解析修复后的输出</li>
                          </>
                        )}
                        {selectedStrategy === 'retry' && (
                          <>
                            <li>首次解析失败</li>
                            <li>将错误信息附加到提示词</li>
                            <li>重新调用 LLM 生成输出</li>
                            <li>成功解析新输出</li>
                          </>
                        )}
                        {selectedStrategy === 'custom_validator' && (
                          <>
                            <li>Pydantic 验证器捕获错误</li>
                            <li>应用自定义修复逻辑</li>
                            <li>返回修复后的值</li>
                          </>
                        )}
                      </ol>
                    </div>
                  </div>
                ) : (
                  <div>
                    <p className="text-sm text-red-700 mb-3">
                      基础解析器无法处理此错误，直接抛出异常。
                    </p>
                    <div className="bg-white p-4 rounded-lg border border-red-200">
                      <div className="text-sm font-semibold text-red-700 mb-2">错误信息：</div>
                      <pre className="text-xs font-mono text-red-600 whitespace-pre-wrap">
                        {result.error}
                      </pre>
                    </div>

                    <div className="mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                      <div className="text-sm font-semibold text-yellow-800 mb-1">
                        💡 建议：
                      </div>
                      <p className="text-xs text-yellow-700">
                        使用 OutputFixingParser 或 RetryWithErrorOutputParser 提升容错能力
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Code Examples */}
      <div className="mt-6 grid md:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <h4 className="font-semibold text-slate-800 mb-3">基础解析器（容易失败）</h4>
          <pre className="bg-slate-900 text-slate-100 p-3 rounded text-xs font-mono overflow-x-auto">
{`parser = PydanticOutputParser(
    pydantic_object=Person
)

try:
    result = parser.parse(llm_output)
except ValidationError as e:
    print(f"Error: {e}")
    # 需要手动处理错误`}
          </pre>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <h4 className="font-semibold text-slate-800 mb-3">OutputFixingParser（自动修复）</h4>
          <pre className="bg-slate-900 text-slate-100 p-3 rounded text-xs font-mono overflow-x-auto">
{`fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4")
)

# 自动修复错误
result = fixing_parser.parse(llm_output)
print(result)  # 成功返回对象`}
          </pre>
        </div>
      </div>
    </div>
  );
}
