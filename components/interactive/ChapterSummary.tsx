"use client";
import { useState } from "react";

interface ChapterContent {
  title: string;
  keywords: { term: string; definition: string; importance: "high" | "medium" | "low" }[];
  summary: string;
  concepts: { name: string; detail: string }[];
}

export function ChapterSummary() {
  const [chapter, setChapter] = useState(0);
  const [expandedKW, setExpandedKW] = useState<number | null>(null);
  const [checklist, setChecklist] = useState<boolean[]>([]);

  const chapters: ChapterContent[] = [
    {
      title: "计算机网络与因特网",
      keywords: [
        { term: "分组交换", definition: "将数据分成小块(分组)独立传输，每经过一个节点进行存储转发", importance: "high" },
        { term: "电路交换", definition: "通信前建立专用端到端路径，资源独占", importance: "medium" },
        { term: "协议栈", definition: "TCP/IP五层模型：应用层→传输层→网络层→链路层→物理层", importance: "high" },
        { term: "时延", definition: "包括处理时延、排队时延、传输时延、传播时延", importance: "high" },
        { term: "吞吐量", definition: "单位时间内传输的数据量，受瓶颈链路限制", importance: "medium" },
      ],
      summary: "互联网是由分组交换网络互连的全球性计算机网络。数据通过分组交换在网络中传输，经过多层协议栈的处理。网络性能由时延、吞吐量和丢包率等指标衡量。",
      concepts: [
        { name: "分组交换 vs 电路交换", detail: "分组交换更灵活高效，适合突发数据；电路交换适合实时性要求高的场景" },
        { name: "五层协议模型", detail: "每层为上层提供服务，对下层隐藏实现细节" },
        { name: "网络时延组成", detail: "总时延 = 处理时延 + 排队时延 + 传输时延 + 传播时延" },
      ],
    },
    {
      title: "应用层",
      keywords: [
        { term: "HTTP", definition: "超文本传输协议，无状态的请求-响应协议", importance: "high" },
        { term: "DNS", definition: "域名系统，将域名解析为IP地址的分布式数据库", importance: "high" },
        { term: "Socket", definition: "应用层与传输层之间的API接口", importance: "medium" },
        { term: "Cookie", definition: "HTTP状态跟踪机制，在客户端存储会话信息", importance: "medium" },
        { term: "SMTP", definition: "简单邮件传输协议，用于发送电子邮件", importance: "low" },
      ],
      summary: "应用层直接为用户应用提供服务。HTTP实现Web通信，DNS实现域名解析，SMTP/POP3/IMAP实现邮件传输。Socket编程接口连接应用层与传输层。",
      concepts: [
        { name: "HTTP特性", detail: "无状态、请求-响应模型、持久/非持久连接" },
        { name: "DNS层级", detail: "根→顶级域→权威域，递归/迭代解析" },
        { name: "P2P架构", detail: "去中心化，每个节点既是客户端又是服务器" },
      ],
    },
    {
      title: "传输层",
      keywords: [
        { term: "TCP", definition: "面向连接、可靠的、基于字节流的传输协议", importance: "high" },
        { term: "UDP", definition: "无连接、不可靠的、基于数据报的传输协议", importance: "high" },
        { term: "三次握手", definition: "TCP建立连接：SYN→SYN-ACK→ACK", importance: "high" },
        { term: "拥塞控制", definition: "TCP通过慢启动、拥塞避免、快速恢复控制发送速率", importance: "high" },
        { term: "滑动窗口", definition: "TCP流量控制机制，接收方通告窗口大小", importance: "medium" },
      ],
      summary: "传输层提供端到端的逻辑通信。TCP提供可靠传输，使用三次握手建立连接，通过序列号、确认号、超时重传保证可靠性，使用拥塞控制避免网络过载。UDP提供轻量级无连接传输。",
      concepts: [
        { name: "TCP可靠性", detail: "序列号+确认号+超时重传+校验和" },
        { name: "拥塞控制四阶段", detail: "慢启动→拥塞避免→快速重传→快速恢复" },
        { name: "TCP vs UDP", detail: "TCP可靠但开销大，UDP高效但不保证交付" },
      ],
    },
  ];

  const ch = chapters[chapter];

  const toggleCheck = (idx: number) => {
    setChecklist((prev) => {
      const next = [...prev];
      next[idx] = !next[idx];
      return next;
    });
  };

  const checkedCount = checklist.filter(Boolean).length;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">章节总结</h3>
      <div className="flex gap-2 mb-4">
        {chapters.map((c, i) => (
          <button key={i} onClick={() => { setChapter(i); setExpandedKW(null); setChecklist([]); }}
            className={`px-3 py-1.5 rounded text-sm ${chapter === i ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
            {c.title}
          </button>
        ))}
      </div>
      <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 mb-4">
        <p className="text-xs font-medium text-text-primary mb-1">核心总结</p>
        <p className="text-xs text-text-secondary">{ch.summary}</p>
      </div>
      <div className="mb-4">
        <p className="text-xs font-medium text-text-primary mb-2">关键术语</p>
        <div className="space-y-1">
          {ch.keywords.map((kw, i) => (
            <button key={i} onClick={() => setExpandedKW(expandedKW === i ? null : i)}
              className="w-full text-left p-2 rounded bg-gray-50 dark:bg-gray-800 border border-border-subtle hover:border-blue-300 transition-colors">
              <div className="flex items-center gap-2">
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${kw.importance === "high" ? "bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400" : kw.importance === "medium" ? "bg-yellow-100 text-yellow-600 dark:bg-yellow-900/30 dark:text-yellow-400" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
                  {kw.importance === "high" ? "重要" : kw.importance === "medium" ? "中等" : "了解"}
                </span>
                <span className="text-sm font-medium text-text-primary">{kw.term}</span>
              </div>
              {expandedKW === i && <p className="text-xs text-text-secondary mt-1 ml-10">{kw.definition}</p>}
            </button>
          ))}
        </div>
      </div>
      <div className="mb-4">
        <p className="text-xs font-medium text-text-primary mb-2">核心概念</p>
        <div className="space-y-2">
          {ch.concepts.map((c, i) => (
            <div key={i} className="flex items-start gap-2">
              <button onClick={() => toggleCheck(i)} className={`mt-0.5 w-4 h-4 rounded border flex items-center justify-center text-[10px] ${checklist[i] ? "bg-green-500 border-green-500 text-white" : "border-gray-300 dark:border-gray-600"}`}>
                {checklist[i] ? "✓" : ""}
              </button>
              <div>
                <span className="text-xs font-medium text-text-primary">{c.name}</span>
                <p className="text-[11px] text-text-secondary">{c.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="p-2 rounded bg-blue-50 dark:bg-blue-900/20 text-xs text-blue-600 dark:text-blue-400 text-center">
        掌握进度: {checkedCount}/{ch.concepts.length}
      </div>
    </div>
  );
}
export default ChapterSummary;
