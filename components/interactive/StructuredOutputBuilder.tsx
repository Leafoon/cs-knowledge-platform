"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus, Trash2, Code2, Play, CheckCircle2 } from 'lucide-react';

interface Field {
  id: string;
  name: string;
  type: string;
  description: string;
  required: boolean;
  defaultValue?: string;
}

type FieldType = 'str' | 'int' | 'float' | 'bool' | 'List[str]' | 'Optional[str]' | 'date' | 'Enum';

const FIELD_TYPES: { value: FieldType; label: string; example: string }[] = [
  { value: 'str', label: 'String', example: '"Hello"' },
  { value: 'int', label: 'Integer', example: '42' },
  { value: 'float', label: 'Float', example: '3.14' },
  { value: 'bool', label: 'Boolean', example: 'true' },
  { value: 'List[str]', label: 'List[String]', example: '["a", "b"]' },
  { value: 'Optional[str]', label: 'Optional String', example: 'null or "value"' },
  { value: 'date', label: 'Date', example: '"2024-01-28"' },
  { value: 'Enum', label: 'Enum', example: '"option_a"' }
];

const PRESETS = {
  person: {
    name: 'Person',
    fields: [
      { id: '1', name: 'name', type: 'str', description: 'Full name', required: true },
      { id: '2', name: 'age', type: 'int', description: 'Age in years', required: true },
      { id: '3', name: 'email', type: 'str', description: 'Email address', required: true },
      { id: '4', name: 'phone', type: 'Optional[str]', description: 'Phone number', required: false }
    ]
  },
  product: {
    name: 'Product',
    fields: [
      { id: '1', name: 'name', type: 'str', description: 'Product name', required: true },
      { id: '2', name: 'price', type: 'float', description: 'Price in USD', required: true },
      { id: '3', name: 'in_stock', type: 'bool', description: 'Availability', required: true },
      { id: '4', name: 'tags', type: 'List[str]', description: 'Category tags', required: false }
    ]
  },
  task: {
    name: 'Task',
    fields: [
      { id: '1', name: 'title', type: 'str', description: 'Task title', required: true },
      { id: '2', name: 'priority', type: 'Enum', description: 'low/medium/high', required: true },
      { id: '3', name: 'due_date', type: 'date', description: 'Deadline', required: false },
      { id: '4', name: 'assigned_to', type: 'Optional[str]', description: 'Assignee', required: false }
    ]
  }
};

export default function StructuredOutputBuilder() {
  const [modelName, setModelName] = useState('Person');
  const [fields, setFields] = useState<Field[]>([]);
  const [testInput, setTestInput] = useState('');
  const [testOutput, setTestOutput] = useState<any>(null);
  const [showCode, setShowCode] = useState(true);

  const addField = () => {
    const newField: Field = {
      id: Date.now().toString(),
      name: 'new_field',
      type: 'str',
      description: 'Field description',
      required: true
    };
    setFields([...fields, newField]);
  };

  const removeField = (id: string) => {
    setFields(fields.filter(f => f.id !== id));
  };

  const updateField = (id: string, updates: Partial<Field>) => {
    setFields(fields.map(f => f.id === id ? { ...f, ...updates } : f));
  };

  const loadPreset = (presetKey: keyof typeof PRESETS) => {
    const preset = PRESETS[presetKey];
    setModelName(preset.name);
    setFields(preset.fields);
  };

  const generatePythonCode = () => {
    const imports = ['from pydantic import BaseModel, Field'];
    
    if (fields.some(f => f.type.startsWith('List'))) {
      imports.push('from typing import List');
    }
    if (fields.some(f => f.type.startsWith('Optional'))) {
      imports.push('from typing import Optional');
    }
    if (fields.some(f => f.type === 'date')) {
      imports.push('from datetime import date');
    }
    if (fields.some(f => f.type === 'Enum')) {
      imports.push('from enum import Enum');
    }

    let code = imports.join('\n') + '\n\n';
    
    // Enum definition if needed
    if (fields.some(f => f.type === 'Enum')) {
      const enumField = fields.find(f => f.type === 'Enum');
      code += `class ${enumField?.name.charAt(0).toUpperCase()}${enumField?.name.slice(1)}(str, Enum):\n`;
      code += `    LOW = "low"\n`;
      code += `    MEDIUM = "medium"\n`;
      code += `    HIGH = "high"\n\n`;
    }

    code += `class ${modelName}(BaseModel):\n`;
    
    if (fields.length === 0) {
      code += '    pass\n';
    } else {
      fields.forEach(field => {
        const fieldType = field.type === 'Enum' 
          ? `${field.name.charAt(0).toUpperCase()}${field.name.slice(1)}`
          : field.type;
        
        code += `    ${field.name}: ${fieldType} = Field(\n`;
        code += `        description="${field.description}"`;
        
        if (!field.required && !field.type.startsWith('Optional')) {
          code += `,\n        default=None`;
        } else if (field.defaultValue) {
          code += `,\n        default=${field.defaultValue}`;
        }
        
        code += '\n    )\n';
      });
    }

    return code;
  };

  const generateUsageCode = () => {
    return `# 方式 1: 使用 with_structured_output (推荐)
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
structured_model = model.with_structured_output(${modelName})

result = structured_model.invoke("${testInput || 'Your query here'}")
print(result)

# 方式 2: 使用 PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

parser = PydanticOutputParser(pydantic_object=${modelName})

prompt = PromptTemplate(
    template="Extract information.\\n{format_instructions}\\n{query}\\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser
result = chain.invoke({"query": "${testInput || 'Your query here'}"})`;
  };

  const runTest = () => {
    // 模拟 LLM 输出
    const mockResult: any = {};
    
    fields.forEach(field => {
      if (field.required || Math.random() > 0.3) {
        switch (field.type) {
          case 'str':
            mockResult[field.name] = `Sample ${field.name}`;
            break;
          case 'int':
            mockResult[field.name] = Math.floor(Math.random() * 100);
            break;
          case 'float':
            mockResult[field.name] = (Math.random() * 100).toFixed(2);
            break;
          case 'bool':
            mockResult[field.name] = Math.random() > 0.5;
            break;
          case 'List[str]':
            mockResult[field.name] = ['item1', 'item2'];
            break;
          case 'date':
            mockResult[field.name] = '2024-01-28';
            break;
          case 'Enum':
            mockResult[field.name] = ['low', 'medium', 'high'][Math.floor(Math.random() * 3)];
            break;
          case 'Optional[str]':
            mockResult[field.name] = Math.random() > 0.5 ? `Optional ${field.name}` : null;
            break;
        }
      }
    });

    setTestOutput(mockResult);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Structured Output Builder</h3>
        <p className="text-slate-600">交互式构建 Pydantic 模型并生成代码</p>
      </div>

      {/* Presets */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <label className="block text-sm font-semibold text-slate-700 mb-3">
          快速开始（预设模板）：
        </label>
        <div className="flex flex-wrap gap-3">
          {(Object.keys(PRESETS) as Array<keyof typeof PRESETS>).map(key => (
            <button
              key={key}
              onClick={() => loadPreset(key)}
              className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors text-sm font-medium"
            >
              {PRESETS[key].name}
            </button>
          ))}
          <button
            onClick={() => { setFields([]); setModelName('CustomModel'); }}
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
              模型名称：
            </label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              placeholder="ModelName"
            />
          </div>

          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-semibold text-slate-800">字段定义</h4>
              <button
                onClick={addField}
                className="flex items-center gap-2 px-3 py-1.5 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors text-sm"
              >
                <Plus className="w-4 h-4" />
                添加字段
              </button>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              <AnimatePresence>
                {fields.map((field, idx) => (
                  <motion.div
                    key={field.id}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-slate-700">
                        字段 #{idx + 1}
                      </span>
                      <button
                        onClick={() => removeField(field.id)}
                        className="text-red-500 hover:text-red-700 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    <input
                      type="text"
                      value={field.name}
                      onChange={(e) => updateField(field.id, { name: e.target.value })}
                      placeholder="field_name"
                      className="w-full px-3 py-1.5 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />

                    <select
                      value={field.type}
                      onChange={(e) => updateField(field.id, { type: e.target.value })}
                      className="w-full px-3 py-1.5 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    >
                      {FIELD_TYPES.map(ft => (
                        <option key={ft.value} value={ft.value}>
                          {ft.label} - {ft.example}
                        </option>
                      ))}
                    </select>

                    <input
                      type="text"
                      value={field.description}
                      onChange={(e) => updateField(field.id, { description: e.target.value })}
                      placeholder="Field description"
                      className="w-full px-3 py-1.5 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />

                    <label className="flex items-center gap-2 text-sm text-slate-700">
                      <input
                        type="checkbox"
                        checked={field.required}
                        onChange={(e) => updateField(field.id, { required: e.target.checked })}
                        className="rounded border-slate-300 text-purple-500 focus:ring-purple-500"
                      />
                      必填字段
                    </label>
                  </motion.div>
                ))}
              </AnimatePresence>

              {fields.length === 0 && (
                <div className="text-center py-8 text-slate-400">
                  点击"添加字段"开始构建模型
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Code Output */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold text-slate-800 flex items-center gap-2">
                <Code2 className="w-5 h-5" />
                生成的代码
              </h4>
              <button
                onClick={() => setShowCode(!showCode)}
                className="text-sm text-purple-600 hover:text-purple-700"
              >
                {showCode ? '隐藏' : '显示'}
              </button>
            </div>

            <AnimatePresence>
              {showCode && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg text-xs font-mono overflow-x-auto max-h-96 overflow-y-auto">
                    {generatePythonCode()}
                  </pre>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <h4 className="font-semibold text-slate-800 mb-3">使用示例</h4>
            
            <label className="block text-sm font-medium text-slate-700 mb-2">
              测试输入：
            </label>
            <textarea
              value={testInput}
              onChange={(e) => setTestInput(e.target.value)}
              placeholder="输入要提取的文本..."
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-sm mb-3"
              rows={3}
            />

            <button
              onClick={runTest}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors font-medium"
            >
              <Play className="w-4 h-4" />
              运行测试
            </button>

            {testOutput && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 p-3 bg-green-50 rounded-lg border border-green-200"
              >
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="w-5 h-5 text-green-600" />
                  <span className="font-semibold text-green-800">输出结果</span>
                </div>
                <pre className="bg-white p-3 rounded text-xs font-mono overflow-x-auto border border-green-200">
                  {JSON.stringify(testOutput, null, 2)}
                </pre>
              </motion.div>
            )}
          </div>

          <div className="bg-slate-50 rounded-lg border border-slate-200 p-4">
            <h4 className="font-semibold text-slate-800 mb-3">完整用法</h4>
            <pre className="text-xs font-mono overflow-x-auto">
              {generateUsageCode()}
            </pre>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-purple-600">{fields.length}</div>
          <div className="text-sm text-slate-600 mt-1">总字段数</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-blue-600">
            {fields.filter(f => f.required).length}
          </div>
          <div className="text-sm text-slate-600 mt-1">必填字段</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-green-600">
            {new Set(fields.map(f => f.type)).size}
          </div>
          <div className="text-sm text-slate-600 mt-1">类型种类</div>
        </div>
      </div>
    </div>
  );
}
