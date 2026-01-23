'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { FileText, MessageSquare, Image, Languages, Search, Code, Tag, Music } from 'lucide-react'

interface TaskType {
  id: string
  name: string
  icon: React.ElementType
  description: string
  examples: string[]
  models: string[]
  color: string
}

const TASK_CATEGORIES = {
  nlp: {
    title: '自然语言处理 (NLP)',
    tasks: [
      {
        id: 'text-classification',
        name: '文本分类',
        icon: FileText,
        description: '将文本分类到预定义类别（情感分析、主题分类等）',
        examples: ['情感分析：正面/负面', '垃圾邮件检测', '新闻分类'],
        models: ['bert-base-uncased', 'roberta-large', 'distilbert'],
        color: 'from-blue-400 to-blue-600'
      },
      {
        id: 'token-classification',
        name: '词元分类',
        icon: Tag,
        description: '为文本中每个词元分配标签（NER、POS等）',
        examples: ['命名实体识别', '词性标注', '分词'],
        models: ['bert-base-NER', 'xlm-roberta-large-finetuned-conll03'],
        color: 'from-green-400 to-green-600'
      },
      {
        id: 'question-answering',
        name: '问答系统',
        icon: MessageSquare,
        description: '从给定上下文中提取答案',
        examples: ['阅读理解', '知识问答', '对话系统'],
        models: ['bert-large-uncased-whole-word-masking-finetuned-squad', 'roberta-base-squad2'],
        color: 'from-purple-400 to-purple-600'
      },
      {
        id: 'text-generation',
        name: '文本生成',
        icon: Code,
        description: '自动生成连贯文本（续写、对话、代码等）',
        examples: ['故事续写', '代码补全', '对话生成'],
        models: ['gpt2', 'gpt-neo-2.7B', 'CodeLlama-7b'],
        color: 'from-pink-400 to-pink-600'
      },
      {
        id: 'translation',
        name: '机器翻译',
        icon: Languages,
        description: '将文本从一种语言翻译为另一种语言',
        examples: ['英译中', '多语言翻译', '同声传译'],
        models: ['Helsinki-NLP/opus-mt-en-zh', 'mBART-large-50'],
        color: 'from-yellow-400 to-yellow-600'
      },
      {
        id: 'summarization',
        name: '文本摘要',
        icon: FileText,
        description: '生成文本的简洁摘要',
        examples: ['新闻摘要', '文档总结', '会议纪要'],
        models: ['facebook/bart-large-cnn', 't5-base'],
        color: 'from-indigo-400 to-indigo-600'
      }
    ]
  },
  vision: {
    title: '计算机视觉 (Vision)',
    tasks: [
      {
        id: 'image-classification',
        name: '图像分类',
        icon: Image,
        description: '识别图像中的主要对象类别',
        examples: ['物体识别', '场景分类', '医疗影像诊断'],
        models: ['google/vit-base-patch16-224', 'microsoft/resnet-50'],
        color: 'from-red-400 to-red-600'
      },
      {
        id: 'object-detection',
        name: '目标检测',
        icon: Search,
        description: '定位并识别图像中的多个对象',
        examples: ['人脸检测', '车辆检测', '缺陷检测'],
        models: ['facebook/detr-resnet-50', 'yolos-tiny'],
        color: 'from-orange-400 to-orange-600'
      },
      {
        id: 'image-segmentation',
        name: '图像分割',
        icon: Image,
        description: '像素级分类（语义/实例/全景分割）',
        examples: ['医学图像分割', '自动驾驶场景理解', '背景移除'],
        models: ['facebook/maskformer-swin-base-ade', 'nvidia/segformer-b0-finetuned-ade-512-512'],
        color: 'from-teal-400 to-teal-600'
      }
    ]
  },
  multimodal: {
    title: '多模态 (Multimodal)',
    tasks: [
      {
        id: 'image-to-text',
        name: '图像描述',
        icon: Image,
        description: '为图像生成文本描述',
        examples: ['图像字幕生成', 'OCR', '视觉问答'],
        models: ['Salesforce/blip-image-captioning-base', 'nlpconnect/vit-gpt2-image-captioning'],
        color: 'from-cyan-400 to-cyan-600'
      },
      {
        id: 'visual-question-answering',
        name: '视觉问答',
        icon: MessageSquare,
        description: '基于图像回答问题',
        examples: ['图片内容问答', '场景理解', '视觉推理'],
        models: ['dandelin/vilt-b32-finetuned-vqa', 'Salesforce/blip-vqa-base'],
        color: 'from-violet-400 to-violet-600'
      }
    ]
  },
  audio: {
    title: '音频处理 (Audio)',
    tasks: [
      {
        id: 'automatic-speech-recognition',
        name: '语音识别',
        icon: Music,
        description: '将语音转换为文本',
        examples: ['语音转文字', '字幕生成', '语音助手'],
        models: ['openai/whisper-base', 'facebook/wav2vec2-base-960h'],
        color: 'from-rose-400 to-rose-600'
      },
      {
        id: 'text-to-speech',
        name: '语音合成',
        icon: Music,
        description: '将文本转换为自然语音',
        examples: ['有声书', '语音导航', '虚拟主播'],
        models: ['facebook/fastspeech2-en-ljspeech', 'microsoft/speecht5_tts'],
        color: 'from-fuchsia-400 to-fuchsia-600'
      }
    ]
  }
}

export default function TaskTypeGallery() {
  const [selectedCategory, setSelectedCategory] = useState<string>('nlp')
  const [selectedTask, setSelectedTask] = useState<TaskType | null>(null)

  const categories = Object.entries(TASK_CATEGORIES)

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        🎯 Transformers 支持的任务类型全览
      </h3>

      {/* 类别标签 */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {categories.map(([key, category]) => (
          <motion.button
            key={key}
            onClick={() => {
              setSelectedCategory(key)
              setSelectedTask(null)
            }}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-all ${
              selectedCategory === key
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg'
                : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
            }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {category.title}
          </motion.button>
        ))}
      </div>

      {/* 任务卡片网格 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {TASK_CATEGORIES[selectedCategory as keyof typeof TASK_CATEGORIES].tasks.map((task) => {
          const Icon = task.icon
          return (
            <motion.div
              key={task.id}
              onClick={() => setSelectedTask(task)}
              className={`p-4 rounded-xl cursor-pointer transition-all border-2 ${
                selectedTask?.id === task.id
                  ? 'border-indigo-500 bg-white shadow-lg'
                  : 'border-slate-200 bg-white/80 hover:border-indigo-300 hover:shadow-md'
              }`}
              whileHover={{ y: -2 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flex items-start gap-3 mb-3">
                <div className={`p-2 rounded-lg bg-gradient-to-br ${task.color}`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1">
                  <h4 className="font-bold text-slate-800 mb-1">{task.name}</h4>
                  <p className="text-xs text-slate-600">{task.description}</p>
                </div>
              </div>

              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span className="bg-slate-100 px-2 py-1 rounded">
                  {task.models.length} 个模型
                </span>
                <span className="bg-slate-100 px-2 py-1 rounded">
                  {task.examples.length} 个场景
                </span>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* 详情面板 */}
      {selectedTask && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-white rounded-xl p-6 border-2 border-indigo-200 shadow-lg"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className={`p-3 rounded-xl bg-gradient-to-br ${selectedTask.color}`}>
              {React.createElement(selectedTask.icon, { className: 'w-6 h-6 text-white' })}
            </div>
            <div>
              <h4 className="text-xl font-bold text-slate-800">{selectedTask.name}</h4>
              <p className="text-sm text-slate-600">{selectedTask.description}</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* 应用场景 */}
            <div>
              <h5 className="font-bold text-slate-700 mb-3 flex items-center gap-2">
                <span className="text-lg">💡</span>
                应用场景
              </h5>
              <ul className="space-y-2">
                {selectedTask.examples.map((example, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-indigo-500 mt-1">▸</span>
                    <span className="text-sm text-slate-700">{example}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* 推荐模型 */}
            <div>
              <h5 className="font-bold text-slate-700 mb-3 flex items-center gap-2">
                <span className="text-lg">🤖</span>
                推荐模型
              </h5>
              <div className="space-y-2">
                {selectedTask.models.map((model, idx) => (
                  <div
                    key={idx}
                    className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs font-mono text-slate-700 hover:bg-slate-100 transition-colors"
                  >
                    {model}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Pipeline 代码示例 */}
          <div className="mt-6 bg-slate-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm text-slate-200">
              <code>{`from transformers import pipeline

# 创建 ${selectedTask.name} Pipeline
pipe = pipeline("${selectedTask.id}")

# 使用示例
result = pipe(${
                selectedTask.id.includes('image') ? '"path/to/image.jpg"' :
                selectedTask.id.includes('audio') ? '"path/to/audio.wav"' :
                '"输入文本示例"'
              })
print(result)`}</code>
            </pre>
          </div>
        </motion.div>
      )}

      {/* 统计信息 */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        {categories.map(([key, category]) => (
          <div key={key} className="bg-white rounded-lg p-4 text-center border border-slate-200">
            <div className="text-2xl font-bold text-indigo-600">{category.tasks.length}</div>
            <div className="text-xs text-slate-600 mt-1">{category.title}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
