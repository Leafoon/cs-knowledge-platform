"use client";
import { useState } from "react";

interface MacEntry {
  mac: string;
  port: number;
  timestamp: number;
}

interface Frame {
  src: string;
  dst: string;
  fromPort: number;
}

const initialEntries: MacEntry[] = [
  { mac: "AA:BB:CC:DD:EE:01", port: 1, timestamp: Date.now() },
  { mac: "AA:BB:CC:DD:EE:02", port: 2, timestamp: Date.now() },
];

export function LearningBridgeDemo() {
  const [table, setTable] = useState<MacEntry[]>(initialEntries);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [result, setResult] = useState("");

  const sendFrame = (src: string, dst: string, fromPort: number) => {
    const newTable = [...table];
    const srcEntry = newTable.find((e) => e.mac === src);
    if (!srcEntry) {
      newTable.push({ mac: src, port: fromPort, timestamp: Date.now() });
    } else {
      srcEntry.port = fromPort;
      srcEntry.timestamp = Date.now();
    }
    setTable(newTable);

    const dstEntry = newTable.find((e) => e.mac === dst);
    if (dstEntry) {
      if (dstEntry.port === fromPort) {
        setResult(`丢弃: 源和目的在同一端口 (${fromPort})`);
      } else {
        setResult(`转发: 从端口${fromPort} → 端口${dstEntry.port} (已知单播)`);
      }
    } else {
      setResult(`泛洪: 从端口${fromPort} → 所有其他端口 (未知单播)`);
    }
    setFrames([...frames, { src, dst, fromPort }]);
  };

  const reset = () => {
    setTable(initialEntries);
    setFrames([]);
    setResult("");
  };

  const prebuiltScenarios = [
    { label: "已知单播", src: "AA:BB:CC:DD:EE:01", dst: "AA:BB:CC:DD:EE:02", port: 1 },
    { label: "未知泛洪", src: "AA:BB:CC:DD:EE:03", dst: "AA:BB:CC:DD:EE:99", port: 3 },
    { label: "同端口丢弃", src: "AA:BB:CC:DD:EE:04", dst: "AA:BB:CC:DD:EE:01", port: 1 },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Learning Bridge <span className="text-text-secondary text-sm">— 网桥MAC学习与帧转发</span>
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {prebuiltScenarios.map((s, i) => (
          <button
            key={i}
            onClick={() => sendFrame(s.src, s.dst, s.port)}
            className="px-3 py-1 rounded bg-blue-600 text-white text-sm"
          >
            {s.label}
          </button>
        ))}
        <button onClick={reset} className="px-3 py-1 rounded bg-gray-500 text-white text-sm">
          重置
        </button>
      </div>
      {result && (
        <div className="bg-yellow-50 dark:bg-yellow-900/30 p-3 rounded text-sm mb-4 text-text-primary">
          {result}
        </div>
      )}
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
        <div className="font-semibold text-text-primary mb-2">MAC地址表</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left p-1 text-text-secondary">MAC地址</th>
              <th className="text-left p-1 text-text-secondary">端口</th>
              <th className="text-left p-1 text-text-secondary">类型</th>
            </tr>
          </thead>
          <tbody>
            {table.map((e, i) => (
              <tr key={i} className="border-b border-border-subtle">
                <td className="p-1 font-mono text-text-primary text-xs">{e.mac}</td>
                <td className="p-1 text-text-secondary">Port {e.port}</td>
                <td className="p-1 text-text-secondary">
                  {initialEntries.some((ie) => ie.mac === e.mac) ? "静态" : "动态学习"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default LearningBridgeDemo;
