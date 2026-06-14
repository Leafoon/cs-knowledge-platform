"use client";
import { useState } from "react";

const ENCODINGS = [
  { name: "NRZ-L", rule: "1→高,0→低", example: "10110 → +--+-" },
  { name: "NRZ-I", rule: "遇1翻转,遇0不变", example: "10110 → 翻保持翻翻保持" },
  { name: "Manchester", rule: "1→先低后高,0→先高后低", example: "10110 → ↓↑↑↓↓↑↑↓↑↓" },
  { name: "差分Manchester", rule: "0→起始跳变,1→起始不变,中间必跳", example: "" },
  { name: "AMI", rule: "1→交替±,0→零", example: "10110 → +0-+0" },
  { name: "4B5B", rule: "4位映射5位,确保跳变密度", example: "1010→10110" },
];

export function EncodingExercise() {
  const [input, setInput] = useState("10110");
  const [selectedEnc, setSelectedEnc] = useState(0);
  const [answers, setAnswers] = useState<string[]>(new Array(ENCODINGS.length).fill(""));
  const [showAnswer, setShowAnswer] = useState<boolean[]>(new Array(ENCODINGS.length).fill(false));

  const toggleAnswer = (index: number) => {
    const newShow = [...showAnswer];
    newShow[index] = !newShow[index];
    setShowAnswer(newShow);
  };

  const setAnswer = (index: number, value: string) => {
    const newAnswers = [...answers];
    newAnswers[index] = value;
    setAnswers(newAnswers);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">数字编码交互练习</h3>
      <div className="mb-4">
        <label className="text-sm text-text-secondary mb-1 block">输入比特序列:</label>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value.replace(/[^01]/g, "").slice(0, 12))}
          className="w-full px-3 py-2 rounded border border-border-subtle bg-bg-subtle text-text-primary font-mono"
          placeholder="输入0和1"
        />
      </div>
      <div className="space-y-3 mb-4">
        {ENCODINGS.map((enc, i) => (
          <div key={i} className={`p-3 rounded-lg ${showAnswer[i] ? "bg-green-50 dark:bg-green-900/20" : "bg-bg-muted"}`}>
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="font-semibold text-sm text-text-primary">{enc.name}</span>
                <span className="text-xs text-text-secondary ml-2">规则: {enc.rule}</span>
              </div>
              <button onClick={() => toggleAnswer(i)} className="text-xs px-2 py-1 bg-bg-subtle rounded text-text-secondary hover:bg-bg-muted">
                {showAnswer[i] ? "隐藏" : "查看"}参考
              </button>
            </div>
            {showAnswer[i] && enc.example && (
              <div className="text-xs font-mono text-text-secondary mb-2 p-2 bg-bg-subtle rounded">
                参考: {enc.example}
              </div>
            )}
            <input
              type="text"
              value={answers[i]}
              onChange={(e) => setAnswer(i, e.target.value)}
              className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-subtle text-text-primary text-sm font-mono"
              placeholder="输入你的编码结果..."
            />
          </div>
        ))}
      </div>
      <div className="text-xs text-text-secondary">
        练习: 给定比特序列,尝试用各种编码方案进行编码。点击查看参考答案。
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">编码要点</div>
        <div>• NRZ-L: 电平直接映射比特值，简单但有直流分量</div>
        <div>• Manchester: 每个比特中间都有跳变，自带时钟同步</div>
        <div>• NRZI: 1跳变0不变，USB 2.0使用</div>
        <div>• 差分编码: 依赖相对变化而非绝对电平，抗干扰更强</div>
      </div>
    </div>
  );
}

export default EncodingExercise;
