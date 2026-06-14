"use client";
import { useState } from "react";

interface Question {
  question: string;
  options: string[];
  answer: number;
  explanation: string;
}

export function ChapterSummaryQuiz() {
  const [currentQ, setCurrentQ] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [scores, setScores] = useState<(boolean | null)[]>([]);

  const questions: Question[] = [
    {
      question: "在TCP/IP协议栈中，哪一层负责端到端的可靠数据传输？",
      options: ["应用层", "传输层", "网络层", "链路层"],
      answer: 1,
      explanation: "传输层(如TCP)负责端到端通信，提供可靠传输、流量控制和拥塞控制。网络层负责主机间的逻辑寻址和路由。",
    },
    {
      question: "DNS使用哪种传输层协议？",
      options: ["仅TCP", "仅UDP", "TCP和UDP", "SCTP"],
      answer: 2,
      explanation: "DNS主要使用UDP(端口53)进行查询(快速、低开销)，当响应数据超过512字节或进行区域传输时使用TCP。",
    },
    {
      question: "TCP三次握手的第二步中，服务器发送的标志位是？",
      options: ["SYN", "SYN+ACK", "ACK", "FIN+ACK"],
      answer: 1,
      explanation: "三次握手: ①客户端发SYN ②服务器回SYN+ACK(确认客户端SYN并发送自己的SYN) ③客户端发ACK。",
    },
    {
      question: "分组交换相比电路交换的主要优势是？",
      options: ["延迟更低", "带宽保证", "资源共享更高效", "实现更简单"],
      answer: 2,
      explanation: "分组交换允许多个连接共享网络资源(统计多路复用)，适合突发数据流量。电路交换需要专用路径，资源独占导致浪费。",
    },
    {
      question: "HTTP/1.1默认使用的连接方式是？",
      options: ["非持久连接", "持久连接", "并行连接", "WebSocket"],
      answer: 1,
      explanation: "HTTP/1.1默认使用持久连接(Connection: keep-alive)，同一TCP连接上可发送多个请求/响应，减少建立连接的开销。",
    },
    {
      question: "TCP拥塞控制中，慢启动阶段窗口大小如何变化？",
      options: ["线性增长", "指数增长", "不变", "随机变化"],
      answer: 1,
      explanation: "慢启动阶段，每收到一个ACK，窗口大小增加1 MSS，即每个RTT窗口翻倍(指数增长)。当窗口达到阈值后进入拥塞避免阶段(线性增长)。",
    },
    {
      question: "CDN的主要作用是？",
      options: ["加密数据传输", "将内容缓存到离用户更近的位置", "管理DNS记录", "提供负载均衡器"],
      answer: 1,
      explanation: "CDN(Content Delivery Network)通过在全球部署边缘节点缓存内容，使用户从地理上最近的节点获取数据，减少延迟和源站负载。",
    },
    {
      question: "UDP相比TCP缺少的特性是？",
      options: ["复用/分用", "校验和", "可靠数据传输", "端口号"],
      answer: 2,
      explanation: "UDP提供复用/分用(端口号)和校验和，但不提供可靠传输(无确认、无重传)、流量控制和拥塞控制。",
    },
  ];

  const handleSelect = (idx: number) => {
    setSelected(idx);
    setShowExplanation(true);
    const correct = idx === questions[currentQ].answer;
    setScores((prev) => {
      const next = [...prev];
      next[currentQ] = correct;
      return next;
    });
  };

  const nextQ = () => {
    setCurrentQ((c) => Math.min(c + 1, questions.length - 1));
    setSelected(null);
    setShowExplanation(false);
  };

  const prevQ = () => {
    setCurrentQ((c) => Math.max(c - 1, 0));
    setSelected(scores[currentQ - 1] !== null ? questions[currentQ - 1].answer : null);
    setShowExplanation(scores[currentQ - 1] !== null);
  };

  const correctCount = scores.filter((s) => s === true).length;
  const answeredCount = scores.filter((s) => s !== null).length;
  const q = questions[currentQ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">章节测验</h3>
      <div className="flex justify-between items-center mb-4">
        <div className="flex gap-1">
          {questions.map((_, i) => (
            <button key={i} onClick={() => { setCurrentQ(i); setSelected(scores[i] !== null ? questions[i].answer : null); setShowExplanation(scores[i] !== null); }}
              className={`w-7 h-7 rounded text-xs font-bold ${i === currentQ ? "bg-blue-600 text-white" : scores[i] === true ? "bg-green-500 text-white" : scores[i] === false ? "bg-red-500 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
              {i + 1}
            </button>
          ))}
        </div>
        <span className="text-xs text-text-secondary">{correctCount}/{answeredCount} 正确</span>
      </div>
      <div className="p-4 rounded bg-gray-50 dark:bg-gray-800 mb-4">
        <p className="text-xs text-text-secondary mb-1">第 {currentQ + 1}/{questions.length} 题</p>
        <p className="text-sm font-medium text-text-primary">{q.question}</p>
      </div>
      <div className="space-y-2 mb-4">
        {q.options.map((opt, i) => {
          const isCorrect = i === q.answer;
          const isSelected = i === selected;
          let style = "border-border-subtle bg-gray-50 dark:bg-gray-800";
          if (showExplanation) {
            if (isCorrect) style = "border-green-500 bg-green-50 dark:bg-green-900/20";
            else if (isSelected) style = "border-red-500 bg-red-50 dark:bg-red-900/20";
          }
          return (
            <button key={i} onClick={() => !showExplanation && handleSelect(i)} disabled={showExplanation}
              className={`w-full text-left p-3 rounded border text-sm transition-colors ${style}`}>
              <span className="font-medium mr-2">{String.fromCharCode(65 + i)}.</span>
              {opt}
              {showExplanation && isCorrect && <span className="ml-2 text-green-600">✓</span>}
              {showExplanation && isSelected && !isCorrect && <span className="ml-2 text-red-600">✗</span>}
            </button>
          );
        })}
      </div>
      {showExplanation && (
        <div className="p-3 rounded bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 mb-4">
          <p className="text-xs font-medium text-blue-700 dark:text-blue-300 mb-1">解析</p>
          <p className="text-xs text-blue-600 dark:text-blue-400">{q.explanation}</p>
        </div>
      )}
      <div className="flex gap-2">
        <button onClick={prevQ} disabled={currentQ === 0}
          className="flex-1 py-2 bg-gray-500 hover:bg-gray-600 disabled:bg-gray-300 text-white rounded text-sm">上一题</button>
        <button onClick={nextQ} disabled={currentQ === questions.length - 1}
          className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded text-sm">下一题</button>
      </div>
      {answeredCount === questions.length && (
        <div className="mt-3 p-3 rounded bg-green-50 dark:bg-green-900/20 text-center">
          <p className="text-sm font-medium text-green-700 dark:text-green-300">测验完成! 得分: {correctCount}/{questions.length} ({((correctCount / questions.length) * 100).toFixed(0)}%)</p>
        </div>
      )}
    </div>
  );
}
export default ChapterSummaryQuiz;
