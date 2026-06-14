"use client";
import { useState } from "react";

interface ProtocolElement {
  name: string;
  en: string;
  desc: string;
  examples: { label: string; detail: string }[];
}

const elements: ProtocolElement[] = [
  {
    name: "语法", en: "Syntax",
    desc: "数据与控制信息的结构或格式，规定数据的组织方式",
    examples: [
      { label: "HTTP报文格式", detail: "请求行 + 首部行 + 空行 + 实体体\nGET /index.html HTTP/1.1\nHost: www.example.com" },
      { label: "以太网帧结构", detail: "目的MAC(6B) | 源MAC(6B) | 类型(2B) | 数据(46-1500B) | CRC(4B)" },
      { label: "IP首部格式", detail: "版本(4b) | 首部长度(4b) | 服务类型(8b) | 总长度(16b) | ..." },
    ],
  },
  {
    name: "语义", en: "Semantics",
    desc: "每一段比特流的含义，规定需要发出何种控制信息及完成何种动作",
    examples: [
      { label: "HTTP状态码", detail: "200 = 成功\n404 = 资源未找到\n500 = 服务器内部错误\n每种码的含义即为语义" },
      { label: "TCP标志位", detail: "SYN=请求建立连接\nACK=确认\nFIN=请求释放连接\nRST=重置连接" },
      { label: "ICMP类型", detail: "Type 8=Echo Request\nType 0=Echo Reply\nType 3=目的不可达" },
    ],
  },
  {
    name: "时序", en: "Timing",
    desc: "事件实现顺序的详细说明，包括速率匹配和排序",
    examples: [
      { label: "TCP三次握手", detail: "客户端→SYN→服务器\n服务器→SYN+ACK→客户端\n客户端→ACK→服务器\n严格按此顺序执行" },
      { label: "HTTP请求/响应", detail: "客户端发送请求 → 服务器处理 → 返回响应\n时序保证请求-响应的对应关系" },
      { label: "ARP解析流程", detail: "广播ARP请求 → 目标主机单播回复\n请求与响应的先后顺序不能颠倒" },
    ],
  },
];

export function ProtocolConceptInteractive() {
  const [active, setActive] = useState(0);
  const [exampleIdx, setExampleIdx] = useState(0);
  const el = elements[active];
  const ex = el.examples[exampleIdx];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">协议三要素交互演示</h3>
      <div className="flex gap-2 mb-4">
        {elements.map((e, i) => (
          <button key={i} onClick={() => { setActive(i); setExampleIdx(0); }}
            className={`flex-1 px-3 py-2 rounded-lg border text-sm font-medium transition-all ${active === i ? "bg-sky-500/20 border-sky-400/60 text-sky-700 dark:text-sky-300" : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}>
            <div>{e.name}</div>
            <div className="text-[10px] opacity-60">{e.en}</div>
          </button>
        ))}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="text-sm font-medium text-text-primary mb-1">{el.name}（{el.en}）</div>
        <div className="text-xs text-text-secondary">{el.desc}</div>
      </div>
      <div className="flex gap-2 mb-3">
        {el.examples.map((e, i) => (
          <button key={i} onClick={() => setExampleIdx(i)}
            className={`px-2.5 py-1 rounded-lg text-xs border transition-all ${exampleIdx === i ? "bg-violet-500/15 border-violet-400/40 text-violet-700 dark:text-violet-300" : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}>
            {e.label}
          </button>
        ))}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-elevated p-4">
        <div className="text-xs font-medium text-text-primary mb-2">{ex.label}</div>
        <pre className="text-xs font-mono text-text-secondary whitespace-pre-wrap leading-relaxed">{ex.detail}</pre>
      </div>
      <div className="mt-3 text-[10px] text-text-tertiary">协议三要素 = 语法 + 语义 + 时序 · 缺一不可</div>
    </div>
  );
}
export default ProtocolConceptInteractive;
