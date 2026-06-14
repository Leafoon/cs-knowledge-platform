"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { HelpCircle, CheckCircle, XCircle, RotateCcw } from "lucide-react";

interface Question {
  q: string;
  options: string[];
  answer: number;
  explain: string;
}

const questions: Question[] = [
  {
    q: "8位补码能表示的最小整数是？",
    options: ["-127", "-128", "-255", "-256"],
    answer: 1,
    explain: "8位补码范围为 -2^(8-1) ~ 2^(8-1)-1，即 -128 ~ 127。",
  },
  {
    q: "十进制数 -5 的8位补码表示是？",
    options: ["10000101", "11111010", "11111011", "10000100"],
    answer: 2,
    explain: "+5 = 00000101，取反得 11111010，加1得 11111011。",
  },
  {
    q: "补码 11100000 对应的十进制数是？",
    options: ["32", "-32", "96", "-96"],
    answer: 1,
    explain: "符号位为1（负数），取反加1得绝对值：00100000 = 32，所以是 -32。",
  },
  {
    q: "以下哪种编码中零的表示不唯一？",
    options: ["补码", "移码", "原码", "无符号数"],
    answer: 2,
    explain: "原码中 +0 = 00000000，-0 = 10000000，零的表示不唯一。",
  },
  {
    q: "两个正数相加结果为负，说明发生了？",
    options: ["下溢", "上溢（正溢出）", "无溢出", "精度丢失"],
    answer: 1,
    explain: "正+正=负，说明结果超出了正数表示范围，发生正溢出（上溢）。",
  },
];

export function DataRepresentationQuiz() {
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
    if (idx === q.answer) setScore(s => s + 1);
  };

  const next = () => {
    if (current + 1 >= questions.length) {
      setFinished(true);
    } else {
      setCurrent(c => c + 1);
      setSelected(null);
      setShowResult(false);
    }
  };

  const reset = () => {
    setCurrent(0);
    setSelected(null);
    setScore(0);
    setShowResult(false);
    setFinished(false);
  };

  if (finished) {
    return (
      <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated text-center">
        <h3 className="text-lg font-semibold mb-4 flex items-center justify-center gap-2">
          <HelpCircle className="w-5 h-5 text-purple-500" />
          数据表示练习题
        </h3>
        <div className="text-3xl font-bold mb-2">{score} / {questions.length}</div>
        <p className="text-text-secondary mb-4">
          正确率 {Math.round((score / questions.length) * 100)}%
        </p>
        <button onClick={reset} className="px-4 py-2 rounded bg-blue-500 text-white text-sm flex items-center gap-1 mx-auto hover:bg-blue-600">
          <RotateCcw className="w-4 h-4" /> 重新开始
        </button>
      </div>
    );
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
        <HelpCircle className="w-5 h-5 text-purple-500" />
        数据表示练习题
      </h3>
      <div className="text-xs text-text-muted mb-4">第 {current + 1}/{questions.length} 题 · 得分 {score}</div>

      <motion.div key={current} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
        <p className="text-sm mb-4 font-medium">{q.q}</p>
        <div className="space-y-2 mb-4">
          {q.options.map((opt, i) => {
            let cls = "border border-border-subtle hover:border-blue-400";
            if (showResult) {
              if (i === q.answer) cls = "border-green-500 bg-green-500/10";
              else if (i === selected) cls = "border-red-500 bg-red-500/10";
            } else if (selected === i) {
              cls = "border-blue-500 bg-blue-500/10";
            }
            return (
              <button key={i} onClick={() => handleSelect(i)}
                className={`w-full text-left px-4 py-2 rounded text-sm flex items-center gap-2 ${cls}`}>
                {showResult && i === q.answer && <CheckCircle className="w-4 h-4 text-green-400 shrink-0" />}
                {showResult && i === selected && i !== q.answer && <XCircle className="w-4 h-4 text-red-400 shrink-0" />}
                <span className="font-mono mr-2">{String.fromCharCode(65 + i)}.</span>{opt}
              </button>
            );
          })}
        </div>

        {showResult && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="p-3 rounded bg-bg-surface border border-border-subtle text-sm text-text-secondary mb-3">
            {q.explain}
          </motion.div>
        )}

        {showResult && (
          <button onClick={next} className="px-4 py-1.5 rounded bg-blue-500 text-white text-sm hover:bg-blue-600">
            {current + 1 >= questions.length ? "查看结果" : "下一题"}
          </button>
        )}
      </motion.div>
    </div>
  );
}
