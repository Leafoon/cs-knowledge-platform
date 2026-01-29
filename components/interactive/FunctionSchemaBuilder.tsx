"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus, Trash2, Code2, Copy, CheckCircle2 } from 'lucide-react';

interface Parameter {
  id: string;
  name: string;
  type: string;
  description: string;
  required: boolean;
  defaultValue?: string;
}

type ParamType = 'string' | 'integer' | 'number' | 'boolean' | 'array' | 'object';

const PARAM_TYPES: { value: ParamType; label: string; example: string }[] = [
  { value: 'string', label: 'String', example: '"text"' },
  { value: 'integer', label: 'Integer', example: '42' },
  { value: 'number', label: 'Number', example: '3.14' },
  { value: 'boolean', label: 'Boolean', example: 'true' },
  { value: 'array', label: 'Array', example: '["a", "b"]' },
  { value: 'object', label: 'Object', example: '{}' }
];

const FUNCTION_TEMPLATES = {
  weather: {
    name: 'get_weather',
    description: 'Get the current weather for a city',
    parameters: [
      { id: '1', name: 'city', type: 'string', description: 'City name', required: true },
      { id: '2', name: 'units', type: 'string', description: 'celsius or fahrenheit', required: false, defaultValue: 'celsius' }
    ]
  },
  search: {
    name: 'search_web',
    description: 'Search the web for information',
    parameters: [
      { id: '1', name: 'query', type: 'string', description: 'Search query', required: true },
      { id: '2', name: 'limit', type: 'integer', description: 'Max results', required: false, defaultValue: '10' }
    ]
  },
  calculator: {
    name: 'calculator',
    description: 'Perform mathematical calculations',
    parameters: [
      { id: '1', name: 'expression', type: 'string', description: 'Math expression to evaluate', required: true }
    ]
  }
};

export default function FunctionSchemaBuilder() {
  const [functionName, setFunctionName] = useState('my_function');
  const [description, setDescription] = useState('Function description');
  const [parameters, setParameters] = useState<Parameter[]>([]);
  const [copied, setCopied] = useState(false);

  const addParameter = () => {
    const newParam: Parameter = {
      id: Date.now().toString(),
      name: 'param_name',
      type: 'string',
      description: 'Parameter description',
      required: true
    };
    setParameters([...parameters, newParam]);
  };

  const removeParameter = (id: string) => {
    setParameters(parameters.filter(p => p.id !== id));
  };

  const updateParameter = (id: string, updates: Partial<Parameter>) => {
    setParameters(parameters.map(p => p.id === id ? { ...p, ...updates } : p));
  };

  const loadTemplate = (templateKey: keyof typeof FUNCTION_TEMPLATES) => {
    const template = FUNCTION_TEMPLATES[templateKey];
    setFunctionName(template.name);
    setDescription(template.description);
    setParameters(template.parameters);
  };

  const generateOpenAISchema = () => {
    const properties: any = {};
    const required: string[] = [];

    parameters.forEach(param => {
      properties[param.name] = {
        type: param.type,
        description: param.description
      };

      if (param.defaultValue) {
        properties[param.name].default = param.type === 'integer' || param.type === 'number'
          ? Number(param.defaultValue)
          : param.type === 'boolean'
          ? param.defaultValue === 'true'
          : param.defaultValue;
      }

      if (param.required) {
        required.push(param.name);
      }
    });

    return {
      name: functionName,
      description: description,
      parameters: {
        type: 'object',
        properties: properties,
        required: required.length > 0 ? required : undefined
      }
    };
  };

  const generatePythonCode = () => {
    const paramsList = parameters.map(p => {
      const typeMap: Record<ParamType, string> = {
        string: 'str',
        integer: 'int',
        number: 'float',
        boolean: 'bool',
        array: 'List',
        object: 'dict'
      };
      const pythonType = typeMap[p.type as ParamType] || 'str';
      
      if (p.defaultValue) {
        return `${p.name}: ${pythonType} = ${p.defaultValue}`;
      }
      return `${p.name}: ${pythonType}`;
    }).join(', ');

    return `from langchain_core.tools import tool

@tool
def ${functionName}(${paramsList}) -> str:
    """${description}
    
    Args:
${parameters.map(p => `        ${p.name}: ${p.description}`).join('\n')}
    """
    # TODO: Implement function logic
    pass`;
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Function Schema Builder</h3>
        <p className="text-slate-600">可视化构建 OpenAI Function Calling Schema</p>
      </div>

      {/* Templates */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <label className="block text-sm font-semibold text-slate-700 mb-3">
          快速模板：
        </label>
        <div className="flex flex-wrap gap-3">
          {(Object.keys(FUNCTION_TEMPLATES) as Array<keyof typeof FUNCTION_TEMPLATES>).map(key => (
            <button
              key={key}
              onClick={() => loadTemplate(key)}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm font-medium"
            >
              {FUNCTION_TEMPLATES[key].name}
            </button>
          ))}
          <button
            onClick={() => { setParameters([]); setFunctionName('my_function'); setDescription('Function description'); }}
            className="px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 transition-colors text-sm font-medium"
          >
            清空
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Builder */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              函数名称：
            </label>
            <input
              type="text"
              value={functionName}
              onChange={(e) => setFunctionName(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
              placeholder="function_name"
            />
          </div>

          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              函数描述：
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
              rows={3}
              placeholder="Describe what this function does"
            />
          </div>

          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-semibold text-slate-800">参数定义</h4>
              <button
                onClick={addParameter}
                className="flex items-center gap-2 px-3 py-1.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
              >
                <Plus className="w-4 h-4" />
                添加参数
              </button>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              <AnimatePresence>
                {parameters.map((param, idx) => (
                  <motion.div
                    key={param.id}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-slate-700">
                        参数 #{idx + 1}
                      </span>
                      <button
                        onClick={() => removeParameter(param.id)}
                        className="text-red-500 hover:text-red-700 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    <input
                      type="text"
                      value={param.name}
                      onChange={(e) => updateParameter(param.id, { name: e.target.value })}
                      placeholder="parameter_name"
                      className="w-full px-3 py-1.5 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono"
                    />

                    <select
                      value={param.type}
                      onChange={(e) => updateParameter(param.id, { type: e.target.value })}
                      className="w-full px-3 py-1.5 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      {PARAM_TYPES.map(pt => (
                        <option key={pt.value} value={pt.value}>
                          {pt.label} - {pt.example}
                        </option>
                      ))}
                    </select>

                    <input
                      type="text"
                      value={param.description}
                      onChange={(e) => updateParameter(param.id, { description: e.target.value })}
                      placeholder="Parameter description"
                      className="w-full px-3 py-1.5 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />

                    <div className="grid grid-cols-2 gap-2">
                      <label className="flex items-center gap-2 text-sm text-slate-700">
                        <input
                          type="checkbox"
                          checked={param.required}
                          onChange={(e) => updateParameter(param.id, { required: e.target.checked })}
                          className="rounded border-slate-300 text-blue-500 focus:ring-blue-500"
                        />
                        必填
                      </label>

                      <input
                        type="text"
                        value={param.defaultValue || ''}
                        onChange={(e) => updateParameter(param.id, { defaultValue: e.target.value })}
                        placeholder="默认值"
                        className="px-2 py-1 text-xs border border-slate-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {parameters.length === 0 && (
                <div className="text-center py-8 text-slate-400">
                  点击"添加参数"开始定义函数参数
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Output */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold text-slate-800 flex items-center gap-2">
                <Code2 className="w-5 h-5" />
                OpenAI Function Schema
              </h4>
              <button
                onClick={() => copyToClipboard(JSON.stringify(generateOpenAISchema(), null, 2))}
                className="flex items-center gap-1 px-3 py-1 text-sm bg-slate-100 hover:bg-slate-200 rounded transition-colors"
              >
                {copied ? <CheckCircle2 className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                {copied ? '已复制' : '复制'}
              </button>
            </div>
            <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg text-xs font-mono overflow-x-auto max-h-80 overflow-y-auto">
              {JSON.stringify(generateOpenAISchema(), null, 2)}
            </pre>
          </div>

          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold text-slate-800">Python @tool 装饰器</h4>
              <button
                onClick={() => copyToClipboard(generatePythonCode())}
                className="flex items-center gap-1 px-3 py-1 text-sm bg-slate-100 hover:bg-slate-200 rounded transition-colors"
              >
                {copied ? <CheckCircle2 className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                {copied ? '已复制' : '复制'}
              </button>
            </div>
            <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg text-xs font-mono overflow-x-auto max-h-80 overflow-y-auto">
              {generatePythonCode()}
            </pre>
          </div>

          <div className="bg-slate-50 rounded-lg border border-slate-200 p-4">
            <h4 className="font-semibold text-slate-800 mb-3">使用示例</h4>
            <pre className="text-xs font-mono overflow-x-auto">
{`# 方式 1: bind_tools
model = ChatOpenAI(model="gpt-4")
model_with_tools = model.bind_tools([${functionName}])

response = model_with_tools.invoke("Your query")

# 方式 2: bind_functions (OpenAI)
from langchain_core.utils.function_calling import convert_to_openai_function

function = convert_to_openai_function(${functionName})
model_with_functions = model.bind_functions([function])`}
            </pre>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-blue-600">{parameters.length}</div>
          <div className="text-sm text-slate-600 mt-1">总参数数</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-green-600">
            {parameters.filter(p => p.required).length}
          </div>
          <div className="text-sm text-slate-600 mt-1">必填参数</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-purple-600">
            {parameters.filter(p => p.defaultValue).length}
          </div>
          <div className="text-sm text-slate-600 mt-1">含默认值</div>
        </div>
      </div>
    </div>
  );
}
