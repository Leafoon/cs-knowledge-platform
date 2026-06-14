"use client";
import { useState } from "react";

const categories = [
  {
    name: "层次数量",
    items: [
      { osi: "7层", tcpip: "4层", note: "OSI更细粒度，TCP/IP更实用" },
    ],
  },
  {
    name: "设计理念",
    items: [
      { osi: "先标准后实现", tcpip: "先实现后标准", note: "OSI理论优先，TCP/IP实践优先" },
    ],
  },
  {
    name: "应用层",
    items: [
      { osi: "应用+表示+会话", tcpip: "合并为一层", note: "TCP/IP更简洁" },
    ],
  },
  {
    name: "传输层",
    items: [
      { osi: "面向连接+无连接", tcpip: "TCP+UDP", note: "功能基本对等" },
    ],
  },
  {
    name: "网络层",
    items: [
      { osi: "面向连接+无连接", tcpip: "仅无连接(IP)", note: "TCP/IP将可靠性留给端系统" },
    ],
  },
  {
    name: "通用性",
    items: [
      { osi: "协议无关模型", tcpip: "基于TCP/IP协议族", note: "OSI更通用，TCP/IP更具体" },
    ],
  },
  {
    name: "数据单元",
    items: [
      { osi: "PDU逐层命名", tcpip: "统一称segment/datagram", note: "OSI术语更精确" },
    ],
  },
];

export function ModelComparison() {
  const [selected, setSelected] = useState(0);
  const [showAll, setShowAll] = useState(false);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Model Comparison <span className="text-text-secondary text-sm">— OSI vs TCP/IP 交互对比</span>
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {categories.map((c, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`px-3 py-1 rounded text-sm ${selected === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {c.name}
          </button>
        ))}
        <button
          onClick={() => setShowAll(!showAll)}
          className={`px-3 py-1 rounded text-sm ${showAll ? "bg-purple-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
        >
          {showAll ? "单条" : "全部"}
        </button>
      </div>
      {showAll ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="text-left p-2 text-text-secondary">对比项</th>
                <th className="text-left p-2 text-text-secondary">OSI</th>
                <th className="text-left p-2 text-text-secondary">TCP/IP</th>
                <th className="text-left p-2 text-text-secondary">备注</th>
              </tr>
            </thead>
            <tbody>
              {categories.map((c, i) => (
                <tr key={i} className="border-b border-border-subtle">
                  <td className="p-2 text-text-primary font-medium">{c.name}</td>
                  <td className="p-2 text-text-secondary">{c.items[0].osi}</td>
                  <td className="p-2 text-text-secondary">{c.items[0].tcpip}</td>
                  <td className="p-2 text-xs text-text-secondary">{c.items[0].note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
          <div className="font-semibold text-text-primary mb-3">{categories[selected].name}</div>
          {categories[selected].items.map((item, i) => (
            <div key={i} className="grid grid-cols-3 gap-3">
              <div className="bg-blue-50 dark:bg-blue-900/30 p-3 rounded">
                <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">OSI</div>
                <div className="text-text-primary font-medium">{item.osi}</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/30 p-3 rounded">
                <div className="text-xs text-green-600 dark:text-green-400 mb-1">TCP/IP</div>
                <div className="text-text-primary font-medium">{item.tcpip}</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-3 rounded">
                <div className="text-xs text-text-secondary mb-1">备注</div>
                <div className="text-text-primary text-sm">{item.note}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ModelComparison;
