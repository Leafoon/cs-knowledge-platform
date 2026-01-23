"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'

type QuantizationMethod = 'dynamic' | 'static' | 'qat'
type Metric = 'model-size' | 'speed' | 'accuracy'

interface BenchmarkData {
  method: string
  modelSize: number // MB
  inferenceSpeed: number // samples/s
  accuracy: number // %
  latency: number // ms
  calibrationRequired: boolean
  trainingRequired: boolean
}

const QuantizationWorkflowVisualizer: React.FC = () => {
  const [selectedMethod, setSelectedMethod] = useState<QuantizationMethod>('dynamic')
  const [selectedMetric, setSelectedMetric] = useState<Metric>('speed')
  const [currentStep, setCurrentStep] = useState(0)

  const benchmarks: BenchmarkData[] = [
    {
      method: 'FP32 (基线)',
      modelSize: 438,
      inferenceSpeed: 42,
      accuracy: 92.3,
      latency: 23.8,
      calibrationRequired: false,
      trainingRequired: false,
    },
    {
      method: '动态 INT8',
      modelSize: 110,
      inferenceSpeed: 178,
      accuracy: 91.8,
      latency: 5.6,
      calibrationRequired: false,
      trainingRequired: false,
    },
    {
      method: '静态 INT8',
      modelSize: 110,
      inferenceSpeed: 212,
      accuracy: 92.1,
      latency: 4.7,
      calibrationRequired: true,
      trainingRequired: false,
    },
    {
      method: 'QAT INT8',
      modelSize: 110,
      inferenceSpeed: 208,
      accuracy: 92.2,
      latency: 4.8,
      calibrationRequired: false,
      trainingRequired: true,
    },
  ]

  const workflows = {
    dynamic: [
      {
        step: 1,
        title: '加载 FP32 模型',
        code: `model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)`,
        description: '导出为 ONNX 格式',
      },
      {
        step: 2,
        title: '配置动态量化',
        code: `quantization_config = AutoQuantizationConfig.avx512_vnni(
    is_static=False,  # 动态量化
    per_channel=True
)`,
        description: '权重量化为 INT8，激活值在推理时动态量化',
      },
      {
        step: 3,
        title: '执行量化',
        code: `quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(
    save_dir="./model_int8",
    quantization_config=quantization_config
)`,
        description: '无需校准数据，直接量化',
      },
      {
        step: 4,
        title: '推理',
        code: `quantized_model = ORTModelForSequenceClassification.from_pretrained(
    "./model_int8"
)
outputs = quantized_model(**inputs)`,
        description: '激活值在运行时动态量化',
      },
    ],
    static: [
      {
        step: 1,
        title: '加载 FP32 模型',
        code: `model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)`,
        description: '导出为 ONNX 格式',
      },
      {
        step: 2,
        title: '准备校准数据',
        code: `dataset = load_dataset("glue", "sst2", split="train[:1000]")
calibration_dataset = dataset.map(
    lambda x: tokenizer(x["sentence"], truncation=True),
    batched=True
)`,
        description: '收集代表性数据用于激活值统计',
      },
      {
        step: 3,
        title: '配置静态量化',
        code: `quantization_config = AutoQuantizationConfig.avx512_vnni(
    is_static=True,  # 静态量化
    per_channel=False
)`,
        description: '权重 + 激活值都预先量化',
      },
      {
        step: 4,
        title: '校准 + 量化',
        code: `quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(
    save_dir="./model_static_int8",
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset
)`,
        description: '运行校准数据收集激活值范围',
      },
      {
        step: 5,
        title: '推理',
        code: `quantized_model = ORTModelForSequenceClassification.from_pretrained(
    "./model_static_int8"
)
outputs = quantized_model(**inputs)`,
        description: '激活值使用预计算的量化参数',
      },
    ],
    qat: [
      {
        step: 1,
        title: '加载 FP32 模型',
        code: `model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased"
)`,
        description: 'PyTorch 模型（训练前）',
      },
      {
        step: 2,
        title: '配置 QAT',
        code: `from optimum.intel import INCConfig, INCTrainer

inc_config = INCConfig(
    quantization_approach="qat",  # 量化感知训练
    accuracy_criterion={"relative": 0.01}
)`,
        description: '在训练中模拟量化效果',
      },
      {
        step: 3,
        title: '插入量化节点',
        code: `trainer = INCTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    quantization_config=inc_config
)`,
        description: '自动在模型中插入 FakeQuantize 节点',
      },
      {
        step: 4,
        title: '训练',
        code: `trainer.train()  # 反向传播包含量化误差
trainer.save_model("./model_qat")`,
        description: '模型学习适应量化约束',
      },
      {
        step: 5,
        title: '导出 INT8',
        code: `quantized_model = ORTModelForSequenceClassification.from_pretrained(
    "./model_qat",
    export=True
)`,
        description: '转换为真正的 INT8 模型',
      },
    ],
  }

  const currentWorkflow = workflows[selectedMethod]
  const maxStep = currentWorkflow.length - 1

  const getMetricValue = (benchmark: BenchmarkData): number => {
    switch (selectedMetric) {
      case 'model-size':
        return benchmark.modelSize
      case 'speed':
        return benchmark.inferenceSpeed
      case 'accuracy':
        return benchmark.accuracy
      default:
        return 0
    }
  }

  const maxMetricValue = Math.max(...benchmarks.map(getMetricValue))

  const getMetricLabel = (value: number): string => {
    switch (selectedMetric) {
      case 'model-size':
        return `${value} MB`
      case 'speed':
        return `${value} samples/s`
      case 'accuracy':
        return `${value.toFixed(1)}%`
      default:
        return `${value}`
    }
  }

  return (
    <div className="w-full space-y-6 my-8">
      {/* 标题 */}
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">量化方法对比与工作流程</h3>
        <p className="text-gray-600 dark:text-gray-400">
          动态量化 vs 静态量化 vs 量化感知训练 (QAT)
        </p>
      </div>

      {/* 方法选择 */}
      <div className="flex gap-3 justify-center flex-wrap">
        <button
          onClick={() => {
            setSelectedMethod('dynamic')
            setCurrentStep(0)
          }}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${
            selectedMethod === 'dynamic'
              ? 'bg-blue-500 text-white shadow-lg scale-105'
              : 'bg-gray-100 dark:bg-gray-800'
          }`}
        >
          <div className="text-2xl mb-1">⚡</div>
          <div>动态量化</div>
          <div className="text-xs opacity-80">运行时量化</div>
        </button>
        <button
          onClick={() => {
            setSelectedMethod('static')
            setCurrentStep(0)
          }}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${
            selectedMethod === 'static'
              ? 'bg-purple-500 text-white shadow-lg scale-105'
              : 'bg-gray-100 dark:bg-gray-800'
          }`}
        >
          <div className="text-2xl mb-1">📊</div>
          <div>静态量化</div>
          <div className="text-xs opacity-80">需校准数据</div>
        </button>
        <button
          onClick={() => {
            setSelectedMethod('qat')
            setCurrentStep(0)
          }}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${
            selectedMethod === 'qat'
              ? 'bg-green-500 text-white shadow-lg scale-105'
              : 'bg-gray-100 dark:bg-gray-800'
          }`}
        >
          <div className="text-2xl mb-1">🎯</div>
          <div>QAT</div>
          <div className="text-xs opacity-80">训练中量化</div>
        </button>
      </div>

      {/* 工作流程 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h4 className="font-semibold mb-4">
          {selectedMethod === 'dynamic' && '动态量化工作流程'}
          {selectedMethod === 'static' && '静态量化工作流程'}
          {selectedMethod === 'qat' && '量化感知训练工作流程'}
        </h4>

        {/* 步骤进度 */}
        <div className="flex items-center justify-between mb-6">
          {currentWorkflow.map((workflow, idx) => (
            <div key={idx} className="flex items-center">
              <button
                onClick={() => setCurrentStep(idx)}
                className={`w-10 h-10 rounded-full font-bold transition-all ${
                  currentStep === idx
                    ? selectedMethod === 'dynamic'
                      ? 'bg-blue-500 text-white scale-110'
                      : selectedMethod === 'static'
                      ? 'bg-purple-500 text-white scale-110'
                      : 'bg-green-500 text-white scale-110'
                    : currentStep > idx
                    ? 'bg-gray-400 text-white'
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                {workflow.step}
              </button>
              {idx < currentWorkflow.length - 1 && (
                <div className="w-8 md:w-16 h-0.5 bg-gray-300 dark:bg-gray-600 mx-1"></div>
              )}
            </div>
          ))}
        </div>

        {/* 当前步骤 */}
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-4"
        >
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h5 className="font-bold text-lg mb-2">
              步骤 {currentWorkflow[currentStep].step}: {currentWorkflow[currentStep].title}
            </h5>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              {currentWorkflow[currentStep].description}
            </p>
          </div>

          <div>
            <div className="text-sm font-semibold mb-2">代码：</div>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded text-xs overflow-x-auto">
              {currentWorkflow[currentStep].code}
            </pre>
          </div>
        </motion.div>

        {/* 导航按钮 */}
        <div className="flex gap-3 mt-6">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 disabled:opacity-50"
          >
            ← 上一步
          </button>
          <button
            onClick={() => setCurrentStep(Math.min(maxStep, currentStep + 1))}
            disabled={currentStep === maxStep}
            className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 disabled:opacity-50"
          >
            下一步 →
          </button>
        </div>
      </div>

      {/* 性能对比 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h4 className="font-semibold mb-4">性能对比</h4>

        {/* 指标选择 */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setSelectedMetric('model-size')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedMetric === 'model-size'
                ? 'bg-orange-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            模型大小
          </button>
          <button
            onClick={() => setSelectedMetric('speed')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedMetric === 'speed'
                ? 'bg-green-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            推理速度
          </button>
          <button
            onClick={() => setSelectedMetric('accuracy')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedMetric === 'accuracy'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            精度
          </button>
        </div>

        {/* 条形图 */}
        <div className="space-y-4">
          {benchmarks.map((benchmark, idx) => {
            const value = getMetricValue(benchmark)
            const percentage = (value / maxMetricValue) * 100
            const isInverse = selectedMetric === 'model-size' // 模型大小越小越好

            return (
              <div key={idx}>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span className="font-medium">{benchmark.method}</span>
                  <span className="text-gray-600 dark:text-gray-400">
                    {getMetricLabel(value)}
                    {idx > 0 && selectedMetric === 'speed' && (
                      <span className="ml-2 text-green-600 dark:text-green-400">
                        ({(value / benchmarks[0].inferenceSpeed).toFixed(1)}x)
                      </span>
                    )}
                  </span>
                </div>
                <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-6 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ duration: 0.8, delay: idx * 0.1 }}
                    className={`h-full flex items-center justify-end pr-2 ${
                      idx === 0
                        ? 'bg-gray-400'
                        : selectedMetric === 'model-size'
                        ? 'bg-orange-500'
                        : selectedMetric === 'speed'
                        ? 'bg-green-500'
                        : 'bg-blue-500'
                    }`}
                  >
                    {percentage > 20 && (
                      <span className="text-xs text-white font-medium">
                        {getMetricLabel(value)}
                      </span>
                    )}
                  </motion.div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 对比表格 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg overflow-x-auto">
        <h4 className="font-semibold mb-4">详细对比</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-gray-300 dark:border-gray-600">
              <th className="text-left py-2 px-4">方法</th>
              <th className="text-left py-2 px-4">模型大小</th>
              <th className="text-left py-2 px-4">推理速度</th>
              <th className="text-left py-2 px-4">精度</th>
              <th className="text-left py-2 px-4">延迟</th>
              <th className="text-left py-2 px-4">校准数据</th>
              <th className="text-left py-2 px-4">重新训练</th>
            </tr>
          </thead>
          <tbody>
            {benchmarks.map((benchmark, idx) => (
              <tr
                key={idx}
                className="border-b border-gray-200 dark:border-gray-700"
              >
                <td className="py-2 px-4 font-medium">{benchmark.method}</td>
                <td className="py-2 px-4">
                  {benchmark.modelSize} MB
                  {idx > 0 && (
                    <span className="text-xs text-green-600 dark:text-green-400 ml-1">
                      (-{(((benchmarks[0].modelSize - benchmark.modelSize) / benchmarks[0].modelSize) * 100).toFixed(0)}%)
                    </span>
                  )}
                </td>
                <td className="py-2 px-4">
                  {benchmark.inferenceSpeed} samples/s
                  {idx > 0 && (
                    <span className="text-xs text-green-600 dark:text-green-400 ml-1">
                      ({(benchmark.inferenceSpeed / benchmarks[0].inferenceSpeed).toFixed(1)}x)
                    </span>
                  )}
                </td>
                <td className="py-2 px-4">
                  {benchmark.accuracy.toFixed(1)}%
                  {idx > 0 && (
                    <span className="text-xs text-red-600 dark:text-red-400 ml-1">
                      ({(benchmark.accuracy - benchmarks[0].accuracy).toFixed(1)}%)
                    </span>
                  )}
                </td>
                <td className="py-2 px-4">{benchmark.latency.toFixed(1)} ms</td>
                <td className="py-2 px-4">
                  {benchmark.calibrationRequired ? '✅ 需要' : '❌ 不需要'}
                </td>
                <td className="py-2 px-4">
                  {benchmark.trainingRequired ? '✅ 需要' : '❌ 不需要'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 选择建议 */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border-2 border-blue-500">
          <div className="text-2xl mb-2">⚡</div>
          <h4 className="font-bold mb-2">动态量化</h4>
          <p className="text-sm mb-3">最简单，无需额外数据</p>
          <ul className="text-xs space-y-1">
            <li className="flex items-start gap-1">
              <span className="text-green-500">✓</span>
              <span>快速部署（一行代码）</span>
            </li>
            <li className="flex items-start gap-1">
              <span className="text-green-500">✓</span>
              <span>4.2x 加速</span>
            </li>
            <li className="flex items-start gap-1">
              <span className="text-yellow-500">⚠</span>
              <span>激活值动态量化（略慢）</span>
            </li>
          </ul>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 border-2 border-purple-500">
          <div className="text-2xl mb-2">📊</div>
          <h4 className="font-bold mb-2">静态量化</h4>
          <p className="text-sm mb-3">最快推理，需校准数据</p>
          <ul className="text-xs space-y-1">
            <li className="flex items-start gap-1">
              <span className="text-green-500">✓</span>
              <span>5.0x 加速（最快）</span>
            </li>
            <li className="flex items-start gap-1">
              <span className="text-green-500">✓</span>
              <span>精度损失小（&lt; 0.5%）</span>
            </li>
            <li className="flex items-start gap-1">
              <span className="text-yellow-500">⚠</span>
              <span>需要 100-1000 条校准数据</span>
            </li>
          </ul>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6 border-2 border-green-500">
          <div className="text-2xl mb-2">🎯</div>
          <h4 className="font-bold mb-2">QAT</h4>
          <p className="text-sm mb-3">精度最高，需重新训练</p>
          <ul className="text-xs space-y-1">
            <li className="flex items-start gap-1">
              <span className="text-green-500">✓</span>
              <span>精度损失最小（&lt; 0.1%）</span>
            </li>
            <li className="flex items-start gap-1">
              <span className="text-green-500">✓</span>
              <span>适合精度敏感任务</span>
            </li>
            <li className="flex items-start gap-1">
              <span className="text-red-500">✗</span>
              <span>需要完整训练流程</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default QuantizationWorkflowVisualizer
