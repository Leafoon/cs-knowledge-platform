"use client";
import { useState } from "react";

const walshCodes: { name: string; code: number[] }[] = [
  { name: "W1", code: [1, 1, 1, 1] },
  { name: "W2", code: [1, -1, 1, -1] },
  { name: "W3", code: [1, 1, -1, -1] },
  { name: "W4", code: [1, -1, -1, 1] },
];

function dotProduct(a: number[], b: number[]): number {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

export function CDMACodeVisualizer() {
  const [sender1, setSender1] = useState(0);
  const [sender2, setSender2] = useState(1);
  const [data1, setData1] = useState(1);
  const [data2, setData2] = useState(-1);
  const [verifyIdx, setVerifyIdx] = useState(0);

  const code1 = walshCodes[sender1].code;
  const code2 = walshCodes[sender2].code;
  const chip1 = code1.map((c) => c * data1);
  const chip2 = code2.map((c) => c * data2);
  const combined = chip1.map((c, i) => c + chip2[i]);

  const recover1 = dotProduct(combined, code1) / code1.length;
  const recover2 = dotProduct(combined, code2) / code2.length;

  const orthogonality = dotProduct(code1, code2);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">CDMA Walsh 码正交性</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 rounded bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
          <label className="text-xs text-blue-600 dark:text-blue-400 mb-1 block">发送者 1</label>
          <div className="flex gap-1 mb-2">
            {walshCodes.map((w, i) => (
              <button key={i} onClick={() => setSender1(i)}
                className={`px-2 py-0.5 rounded text-xs font-mono ${sender1 === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
                {w.name}
              </button>
            ))}
          </div>
          <div className="flex gap-1 mb-2">{code1.map((c, i) => <span key={i} className="w-6 h-6 flex items-center justify-center text-xs font-mono rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">{c > 0 ? "+1" : "-1"}</span>)}</div>
          <div className="flex gap-2">
            <button onClick={() => setData1(1)} className={`px-2 py-0.5 rounded text-xs ${data1 === 1 ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>数据 = +1</button>
            <button onClick={() => setData1(-1)} className={`px-2 py-0.5 rounded text-xs ${data1 === -1 ? "bg-red-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>数据 = -1</button>
          </div>
        </div>
        <div className="p-3 rounded bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800">
          <label className="text-xs text-purple-600 dark:text-purple-400 mb-1 block">发送者 2</label>
          <div className="flex gap-1 mb-2">
            {walshCodes.map((w, i) => (
              <button key={i} onClick={() => setSender2(i)}
                className={`px-2 py-0.5 rounded text-xs font-mono ${sender2 === i ? "bg-purple-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
                {w.name}
              </button>
            ))}
          </div>
          <div className="flex gap-1 mb-2">{code2.map((c, i) => <span key={i} className="w-6 h-6 flex items-center justify-center text-xs font-mono rounded bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">{c > 0 ? "+1" : "-1"}</span>)}</div>
          <div className="flex gap-2">
            <button onClick={() => setData2(1)} className={`px-2 py-0.5 rounded text-xs ${data2 === 1 ? "bg-green-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>数据 = +1</button>
            <button onClick={() => setData2(-1)} className={`px-2 py-0.5 rounded text-xs ${data2 === -1 ? "bg-red-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>数据 = -1</button>
          </div>
        </div>
      </div>
      <div className="mb-4">
        <p className="text-xs text-text-secondary mb-2">叠加信号 (信道中传输)</p>
        <div className="flex gap-1">{combined.map((c, i) => (
          <div key={i} className="flex-1 flex flex-col items-center">
            <div className={`w-full h-8 flex items-center justify-center text-xs font-mono rounded ${c > 0 ? "bg-green-200 dark:bg-green-900/30 text-green-700 dark:text-green-300" : c < 0 ? "bg-red-200 dark:bg-red-900/30 text-red-700 dark:text-red-300" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
              {c > 0 ? `+${c}` : c}
            </div>
          </div>
        ))}</div>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 text-center">
          <div className="text-xs text-text-secondary">W1·W2 正交性</div>
          <div className="text-lg font-bold text-text-primary">{orthogonality}</div>
        </div>
        <div className="p-3 rounded bg-green-50 dark:bg-green-900/20 text-center">
          <div className="text-xs text-green-600 dark:text-green-400">恢复数据1</div>
          <div className="text-lg font-bold text-green-700 dark:text-green-300">{recover1 > 0 ? "+1" : "-1"}</div>
        </div>
        <div className="p-3 rounded bg-green-50 dark:bg-green-900/20 text-center">
          <div className="text-xs text-green-600 dark:text-green-400">恢复数据2</div>
          <div className="text-lg font-bold text-green-700 dark:text-green-300">{recover2 > 0 ? "+1" : "-1"}</div>
        </div>
      </div>
      <p className="text-xs text-text-secondary">Walsh码两两正交(点积=0)，使得多个发送者可以在同一频率同时传输，接收端用对应码片提取目标信号。</p>
    </div>
  );
}
export default CDMACodeVisualizer;
