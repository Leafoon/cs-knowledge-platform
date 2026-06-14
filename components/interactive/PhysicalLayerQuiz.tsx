"use client";
import { useState } from "react";

interface Question {
  q: string;
  options: string[];
  answer: number;
  explain: string;
}

const questions: Question[] = [
  { q: "物理层的主要功能是什么？", options: ["路由选择", "比特传输", "差错控制", "流量控制"], answer: 1, explain: "物理层负责在物理媒介上传输比特流，不涉及差错控制和流量控制。" },
  { q: "以下哪种是数字调制技术？", options: ["ASK", "CRC", "ARQ", "CSMA"], answer: 0, explain: "ASK（幅移键控）是数字调制技术，将数字信号调制到模拟载波上。" },
  { q: "奈奎斯特准则适用于哪种信道？", options: ["有噪声信道", "无噪声信道", "无线信道", "光纤信道"], answer: 1, explain: "奈奎斯特准则给出无噪声信道的最大数据速率：C = 2W·log₂(V)。" },
  { q: "香农公式中C = W·log₂(1+S/N)，S/N表示？", options: ["信号频率", "信噪比", "带宽", "码元速率"], answer: 1, explain: "S/N是信噪比（Signal-to-Noise Ratio），决定有噪声信道的容量上限。" },
  { q: "以下哪种编码方式每比特都有跳变？", options: ["NRZ", "Manchester", "AMI", "B8ZS"], answer: 1, explain: "Manchester编码在每比特中间都有电平跳变，既表示数据又提供时钟同步。" },
  { q: "单模光纤与多模光纤的主要区别是？", options: ["颜色不同", "纤芯直径不同", "传输方向不同", "协议不同"], answer: 1, explain: "单模光纤纤芯更细（约9μm），只允许一种模式传播，传输距离更远。" },
  { q: "TDM的全称和含义是？", options: ["时分复用", "频分复用", "波分复用", "码分复用"], answer: 0, explain: "TDM（Time Division Multiplexing）时分复用，不同信号轮流使用信道的不同时间片。" },
  { q: "基带传输和宽带传输的区别在于？", options: ["速率不同", "是否调制", "距离不同", "介质不同"], answer: 1, explain: "基带传输直接发送数字信号；宽带传输需要将信号调制到载波上。" },
];

export function PhysicalLayerQuiz() {
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const [finished, setFinished] = useState(false);

  const q = questions[current];

  const handleSelect = (idx: number) => {
    if (showResult) return;
    setSelected(idx);
    setShowResult(true);
    if (idx === q.answer) setScore((s) => s + 1);
  };

  const handleNext = () => {
    if (current + 1 >= questions.length) {
      setFinished(true);
    } else {
      setCurrent((c) => c + 1);
      setSelected(null);
      setShowResult(false);
    }
  };

  const handleRestart = () => {
    setCurrent(0);
    setSelected(null);
    setScore(0);
    setShowResult(false);
    setFinished(false);
  };

  if (finished) {
    return (
      <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
        <h3 className="text-xl font-semibold mb-4 text-text-primary">物理层测验</h3>
        <div className="text-center py-8 space-y-4">
          <div className="text-5xl font-bold text-emerald-500">{score}/{questions.length}</div>
          <p className="text-text-secondary">
            {score === questions.length ? "满分！物理层知识掌握牢固！" : score >= questions.length * 0.7 ? "不错！继续加油！" : "需要复习物理层相关内容。"}
          </p>
          <button onClick={handleRestart} className="px-4 py-2 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-sm font-medium hover:bg-sky-500/25 transition-colors">重新测验</button>
        </div>
      </div>
    );
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">物理层测验</h3>
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs text-text-tertiary">题目 {current + 1}/{questions.length}</span>
        <span className="text-xs text-emerald-600 dark:text-emerald-400">得分: {score}</span>
      </div>
      <div className="w-full bg-bg-tertiary rounded-full h-1.5 mb-4">
        <div className="bg-sky-500 h-1.5 rounded-full transition-all" style={{ width: `${((current + 1) / questions.length) * 100}%` }} />
      </div>
      <p className="text-sm font-medium text-text-primary mb-4">{q.q}</p>
      <div className="space-y-2 mb-4">
        {q.options.map((opt, idx) => {
          let cls = "border-border-subtle bg-bg-tertiary text-text-primary hover:border-sky-400/60";
          if (showResult) {
            if (idx === q.answer) cls = "border-emerald-500 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
            else if (idx === selected) cls = "border-red-500 bg-red-500/10 text-red-700 dark:text-red-300";
            else cls = "border-border-subtle bg-bg-tertiary text-text-tertiary";
          }
          return (
            <button key={idx} onClick={() => handleSelect(idx)} disabled={showResult}
              className={`w-full text-left px-4 py-2.5 rounded-lg border text-sm transition-all ${cls}`}>
              <span className="font-mono mr-2 text-xs opacity-60">{String.fromCharCode(65 + idx)}.</span>{opt}
            </button>
          );
        })}
      </div>
      {showResult && (
        <div className="space-y-3">
          <div className={`text-xs p-3 rounded-lg border ${selected === q.answer ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-300" : "bg-red-500/10 border-red-500/30 text-red-700 dark:text-red-300"}`}>
            {selected === q.answer ? "✓ 正确！" : `✗ 错误，正确答案是 ${String.fromCharCode(65 + q.answer)}`}
            <span className="ml-1 text-text-secondary">{q.explain}</span>
          </div>
          <button onClick={handleNext} className="w-full px-4 py-2 rounded-lg bg-sky-500/15 text-sky-700 dark:text-sky-300 text-sm font-medium hover:bg-sky-500/25 transition-colors">
            {current + 1 >= questions.length ? "查看结果" : "下一题 →"}
          </button>
        </div>
      )}
    </div>
  );
}
export default PhysicalLayerQuiz;
