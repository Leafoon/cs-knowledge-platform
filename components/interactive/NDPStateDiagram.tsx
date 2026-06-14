"use client";
import { useState } from "react";

const states = [
  {
    name: "INCOMPLETE",
    zh: "未完成",
    desc: "已发送NS但未收到NA，不知道目标MAC",
    color: "bg-yellow-500",
    transitions: [
      { to: "REACHABLE", trigger: "收到NA", condition: "目标回复NA" },
      { to: "INCOMPLETE", trigger: "重传NS", condition: "超时未回复" },
    ],
  },
  {
    name: "REACHABLE",
    zh: "可达",
    desc: "已确认目标可达，缓存MAC地址",
    color: "bg-green-500",
    transitions: [
      { to: "STALE", trigger: "ReachableTime超时", condition: "30s无通信" },
      { to: "STALE", trigger: "收到非请求NA", condition: "MAC地址可能变更" },
    ],
  },
  {
    name: "STALE",
    zh: "过期",
    desc: "地址可达性未知，下次通信时需验证",
    color: "bg-orange-500",
    transitions: [
      { to: "DELAY", trigger: "发送数据", condition: "有数据要发给目标" },
      { to: "REACHABLE", trigger: "收到非请求NA", condition: "被动更新" },
    ],
  },
  {
    name: "DELAY",
    zh: "延迟",
    desc: "等待上层协议确认可达性（5秒）",
    color: "bg-blue-500",
    transitions: [
      { to: "PROBE", trigger: "Delay超时", condition: "5s内无上层确认" },
      { to: "REACHABLE", trigger: "上层确认", condition: "收到TCP ACK等" },
    ],
  },
  {
    name: "PROBE",
    zh: "探测",
    desc: "发送NS单播探测目标是否仍可达",
    color: "bg-red-500",
    transitions: [
      { to: "REACHABLE", trigger: "收到NA", condition: "目标确认可达" },
      { to: "INCOMPLETE", trigger: "探测失败", condition: "重试耗尽" },
    ],
  },
];

export function NDPStateDiagram() {
  const [selected, setSelected] = useState(0);
  const [simLog, setSimLog] = useState<string[]>([]);
  const [currentState, setCurrentState] = useState(0);

  const state = states[selected];
  const current = states[currentState];

  const simulateTransition = (tIdx: number) => {
    const t = current.transitions[tIdx];
    const toIdx = states.findIndex((s) => s.name === t.to);
    setSimLog([...simLog, `${current.name} → ${t.to} (${t.trigger})`]);
    setCurrentState(toIdx);
    setSelected(toIdx);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        NDP State Diagram <span className="text-text-secondary text-sm">— IPv6邻居状态转换</span>
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {states.map((s, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`px-3 py-1 rounded text-sm text-white ${s.color} ${selected === i ? "ring-2 ring-offset-2 ring-blue-400 dark:ring-offset-gray-900" : "opacity-60"}`}
          >
            {s.name}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">{state.name} ({state.zh})</div>
        <p className="text-sm text-text-secondary mb-3">{state.desc}</p>
        <div className="font-medium text-text-primary text-sm mb-2">状态转换:</div>
        <div className="space-y-2">
          {state.transitions.map((t, i) => (
            <button
              key={i}
              onClick={() => {
                const toIdx = states.findIndex((s) => s.name === t.to);
                setSelected(toIdx);
              }}
              className="w-full text-left bg-white dark:bg-gray-900 p-2 rounded text-sm hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <span className="text-text-primary">→ {t.to}</span>
              <span className="text-text-secondary ml-2">({t.trigger})</span>
              <span className="text-xs text-text-secondary ml-2">条件: {t.condition}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="mb-3">
        <div className="text-sm font-semibold text-text-secondary mb-2">模拟状态转换</div>
        <div className="text-xs text-text-secondary mb-2">当前: {current.name} ({current.zh})</div>
        <div className="flex gap-2 flex-wrap">
          {current.transitions.map((t, i) => (
            <button
              key={i}
              onClick={() => simulateTransition(i)}
              className="px-3 py-1 rounded bg-purple-600 text-white text-sm"
            >
              {t.trigger}
            </button>
          ))}
        </div>
      </div>
      {simLog.length > 0 && (
        <div className="bg-gray-900 p-3 rounded text-xs font-mono max-h-32 overflow-y-auto">
          {simLog.map((l, i) => (
            <div key={i} className="text-green-400">{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export default NDPStateDiagram;
