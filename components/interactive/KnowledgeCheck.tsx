"use client";
import { useState } from "react";

const questions = [
  {
    q: "OSI模型中，哪一层负责端到端的可靠传输？",
    en: "Which OSI layer handles end-to-end reliable transport?",
    options: ["网络层 (Network)", "传输层 (Transport)", "会话层 (Session)", "数据链路层 (Data Link)"],
    correct: 1,
    explain: "传输层（第4层）负责端到端通信，提供可靠(TCP)或不可靠(UDP)传输服务。",
  },
  {
    q: "TCP三次握手中，第二个报文的标志位组合是什么？",
    en: "What flags are set in the 2nd segment of TCP 3-way handshake?",
    options: ["SYN", "SYN+ACK", "ACK", "FIN+ACK"],
    correct: 1,
    explain: "服务端回复SYN+ACK，同时确认客户端的SYN并发送自己的SYN。",
  },
  {
    q: "IPv4地址192.168.1.0/24中，可用主机地址数量是多少？",
    en: "How many usable host addresses in 192.168.1.0/24?",
    options: ["256", "254", "255", "252"],
    correct: 1,
    explain: "/24有256个地址，减去网络地址(.0)和广播地址(.255)，可用254个。",
  },
  {
    q: "ARP协议的作用是什么？",
    en: "What does ARP do?",
    options: ["IP地址→MAC地址映射", "MAC地址→IP地址映射", "域名→IP地址解析", "IP地址→域名解析"],
    correct: 0,
    explain: "ARP（地址解析协议）将已知的IP地址解析为对应的MAC地址，用于局域网内帧转发。",
  },
];

export function KnowledgeCheck() {
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [scores, setScores] = useState<(boolean | null)[]>(Array(questions.length).fill(null));

  const q = questions[current];
  const answered = selected !== null;
  const isCorrect = selected === q.correct;

  const handleSelect = (idx: number) => {
    if (answered) return;
    setSelected(idx);
    const newScores = [...scores];
    newScores[current] = idx === q.correct;
    setScores(newScores);
  };

  const handleNext = () => {
    setSelected(null);
    setCurrent((prev) => Math.min(prev + 1, questions.length - 1));
  };

  const correctCount = scores.filter((s) => s === true).length;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Knowledge Check <span className="text-text-secondary text-sm">— 知识检查点</span>
      </h3>
      <div className="flex justify-between items-center mb-3">
        <div className="text-sm text-text-secondary">
          问题 {current + 1}/{questions.length}
        </div>
        <div className="text-sm text-text-secondary">
          得分: {correctCount}/{scores.filter((s) => s !== null).length}
        </div>
      </div>
      <div className="flex gap-1 mb-4">
        {scores.map((s, i) => (
          <button
            key={i}
            onClick={() => { setCurrent(i); setSelected(null); }}
            className={`w-8 h-2 rounded ${i === current ? "ring-1 ring-blue-400" : ""} ${s === true ? "bg-green-500" : s === false ? "bg-red-500" : "bg-gray-300 dark:bg-gray-600"}`}
          />
        ))}
      </div>
      <div className="text-text-primary font-medium mb-1">{q.q}</div>
      <div className="text-xs text-text-secondary mb-4">{q.en}</div>
      <div className="space-y-2 mb-4">
        {q.options.map((opt, i) => {
          let cls = "bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700";
          if (answered) {
            if (i === q.correct) cls = "bg-green-100 dark:bg-green-900/40 border border-green-500";
            else if (i === selected && !isCorrect) cls = "bg-red-100 dark:bg-red-900/40 border border-red-500";
          } else if (i === selected) {
            cls = "bg-blue-100 dark:bg-blue-900/40 border border-blue-500";
          }
          return (
            <button
              key={i}
              onClick={() => handleSelect(i)}
              className={`w-full text-left p-3 rounded text-sm ${cls}`}
              disabled={answered}
            >
              {String.fromCharCode(65 + i)}. {opt}
            </button>
          );
        })}
      </div>
      {answered && (
        <div className={`p-3 rounded text-sm mb-3 ${isCorrect ? "bg-green-50 dark:bg-green-900/30" : "bg-red-50 dark:bg-red-900/30"}`}>
          <div className={`font-semibold ${isCorrect ? "text-green-700 dark:text-green-300" : "text-red-700 dark:text-red-300"}`}>
            {isCorrect ? "✓ 正确！" : "✗ 错误"}
          </div>
          <div className="text-text-secondary mt-1">{q.explain}</div>
        </div>
      )}
      {answered && current < questions.length - 1 && (
        <button onClick={handleNext} className="px-4 py-2 rounded bg-blue-600 text-white text-sm">
          下一题
        </button>
      )}
    </div>
  );
}

export default KnowledgeCheck;
