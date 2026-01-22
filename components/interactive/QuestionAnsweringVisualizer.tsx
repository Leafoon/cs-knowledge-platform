"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface QAStep {
  stage: number;
  name: string;
  description: string;
  visual: string;
}

const context = `Hugging Face is a company founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf. The company is based in New York City and Paris.`;

const question = "When was Hugging Face founded?";
const answer = "2016";

const steps: QAStep[] = [
  {
    stage: 1,
    name: "输入处理",
    description: "将问题和上下文拼接并编码",
    visual: "[CLS] When was ... [SEP] Hugging Face is ... [SEP]"
  },
  {
    stage: 2,
    name: "模型推理",
    description: "计算每个 token 作为答案起始/结束位置的概率",
    visual: "Start logits: [..., 8.23, ...]\nEnd logits: [..., 7.91, ...]"
  },
  {
    stage: 3,
    name: "答案提取",
    description: "选择最高概率的起始和结束位置",
    visual: "Answer span: [52, 56] → '2016'"
  }
];

export default function QuestionAnsweringVisualizer() {
  const [currentStep, setCurrentStep] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);

  const highlightAnswer = (text: string) => {
    if (!showAnswer) return <span className="text-gray-300">{text}</span>;

    const startIndex = text.indexOf(answer);
    if (startIndex === -1) return <span className="text-gray-300">{text}</span>;

    return (
      <>
        <span className="text-gray-300">{text.slice(0, startIndex)}</span>
        <motion.span
          initial={{ backgroundColor: "transparent" }}
          animate={{ backgroundColor: "rgba(34, 197, 94, 0.3)" }}
          className="px-2 py-1 rounded bg-green-500/30 text-green-300 font-bold"
        >
          {answer}
        </motion.span>
        <span className="text-gray-300">{text.slice(startIndex + answer.length)}</span>
      </>
    );
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl border border-slate-700 shadow-2xl">
      <h3 className="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
        ❓ 抽取式问答可视化
      </h3>

      {/* Question */}
      <div className="mb-4 p-4 bg-blue-900/20 border border-blue-500/50 rounded-lg">
        <h4 className="text-sm font-semibold text-blue-400 mb-2">💬 问题 (Question)</h4>
        <p className="text-white font-medium">{question}</p>
      </div>

      {/* Context */}
      <div className="mb-6 p-4 bg-purple-900/20 border border-purple-500/50 rounded-lg">
        <h4 className="text-sm font-semibold text-purple-400 mb-2">📄 上下文 (Context)</h4>
        <p className="text-sm leading-relaxed">
          {highlightAnswer(context)}
        </p>
      </div>

      {/* Processing Steps */}
      <div className="mb-6">
        <h4 className="text-sm font-semibold text-gray-400 mb-3">🔄 处理流程</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {steps.map((step, index) => (
            <motion.div
              key={step.stage}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 }}
              whileHover={{ scale: 1.05 }}
              onClick={() => setCurrentStep(index)}
              className={`p-4 rounded-lg cursor-pointer transition-all ${
                currentStep === index
                  ? "bg-cyan-600 shadow-xl ring-2 ring-cyan-400"
                  : "bg-slate-800 hover:bg-slate-750"
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                  currentStep === index ? "bg-white text-cyan-600" : "bg-slate-700 text-gray-400"
                }`}>
                  {step.stage}
                </div>
                <h5 className="font-semibold text-white">{step.name}</h5>
              </div>
              <p className="text-xs text-gray-300 mb-3">{step.description}</p>
              <div className="p-2 bg-black/30 rounded text-xs font-mono text-green-300 whitespace-pre-wrap">
                {step.visual}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Answer Reveal */}
      <div className="text-center mb-6">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowAnswer(!showAnswer)}
          className={`px-8 py-3 rounded-lg font-semibold transition-all ${
            showAnswer
              ? "bg-green-600 text-white shadow-lg"
              : "bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-600 hover:to-blue-700"
          }`}
        >
          {showAnswer ? "✓ 答案已显示" : "🔍 显示答案"}
        </motion.button>
      </div>

      {/* Answer Details */}
      {showAnswer && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="p-5 bg-green-900/20 border border-green-500/50 rounded-lg"
        >
          <h4 className="text-lg font-bold text-green-400 mb-3">✅ 提取的答案</h4>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-gray-400 w-20">答案:</span>
              <span className="text-2xl font-bold text-green-300">{answer}</span>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-gray-400 w-20">置信度:</span>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <div className="flex-1 bg-slate-700 rounded-full h-3 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: "98.7%" }}
                      transition={{ duration: 1, delay: 0.3 }}
                      className="h-full bg-gradient-to-r from-green-500 to-emerald-600"
                    />
                  </div>
                  <span className="text-green-300 font-semibold text-sm">98.7%</span>
                </div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-gray-400 w-20">位置:</span>
              <span className="text-white font-mono">[52, 56]</span>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-gray-400 w-20">原文片段:</span>
              <span className="text-gray-300 text-sm">
                "...founded in <span className="text-green-300 font-bold">2016</span> by..."
              </span>
            </div>
          </div>
        </motion.div>
      )}

      {/* How it works */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="mt-6 p-4 bg-yellow-900/20 border border-yellow-500/50 rounded-lg"
      >
        <h5 className="text-sm font-semibold text-yellow-400 mb-2">💡 工作原理</h5>
        <p className="text-xs text-gray-300 leading-relaxed">
          抽取式问答模型通过计算上下文中每个 token 作为答案起始位置和结束位置的概率，
          然后选择最高概率的 span 作为答案。模型不会生成新文本，只会从上下文中提取已有的片段。
        </p>
      </motion.div>
    </div>
  );
}
