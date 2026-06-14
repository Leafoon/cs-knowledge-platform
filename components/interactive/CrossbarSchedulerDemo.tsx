"use client";
import { useState } from "react";

const N = 4;
const colors = ["bg-blue-500", "bg-green-500", "bg-purple-500", "bg-orange-500"];

export function CrossbarSchedulerDemo() {
  const [grants, setGrants] = useState<(number | null)[]>(Array(N).fill(null));
  const [round, setRound] = useState(0);

  const requests = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
  ];

  const runStep = () => {
    const newGrants: (number | null)[] = Array(N).fill(null);
    const used = new Set<number>();
    for (let p = 0; p < N; p++) {
      const i = (p + round) % N;
      for (const out of requests[i]) {
        if (!used.has(out)) {
          newGrants[i] = out;
          used.add(out);
          break;
        }
      }
    }
    setGrants(newGrants);
    setRound((r) => r + 1);
  };

  const reset = () => { setGrants(Array(N).fill(null)); setRound(0); };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">交叉开关调度器 (iSLIP算法)</h3>
      <p className="text-sm text-text-secondary mb-4">iSLIP通过轮询(round-robin)实现公平调度，避免饥饿。</p>
      <div className="grid grid-cols-5 gap-1 mb-4 text-center text-xs">
        <div></div>
        {Array.from({ length: N }, (_, i) => <div key={i} className="text-text-secondary font-mono">出{i}</div>)}
        {Array.from({ length: N }, (_, row) => (
          <>
            <div key={`l-${row}`} className="text-text-secondary font-mono flex items-center">入{row}</div>
            {Array.from({ length: N }, (_, col) => (
              <div key={`${row}-${col}`}
                className={`h-10 rounded border flex items-center justify-center font-mono text-xs transition-all duration-300 ${grants[row] === col ? `${colors[row]} text-white border-transparent` : requests[row].includes(col) ? "border-dashed border-gray-400 text-text-secondary" : "border-transparent bg-gray-50 dark:bg-gray-900"}`}>
                {grants[row] === col ? "✓" : requests[row].includes(col) ? "·" : ""}
              </div>
            ))}
          </>
        ))}
      </div>
      <div className="flex gap-2 mb-3">
        <button onClick={runStep} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm">执行一轮调度</button>
        <button onClick={reset} className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded text-sm text-text-secondary">重置</button>
      </div>
      <div className="text-xs text-text-secondary text-center">调度轮次: {round}</div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">iSLIP算法说明</div>
        <div>• 输入端口轮询选择输出端口(Grant阶段)</div>
        <div>• 输出端口从Grant中选择一个接受(Accept阶段)</div>
        <div>• 每轮结束后更新轮询指针，保证公平性</div>
        <div>• 单轮iSLIP即可达到100%吞吐(均匀流量)</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">调度算法比较</div>
        <div>• iSLIP: O(log N)轮迭代，性能好，实现复杂</div>
        <div>• PIM: 随机匹配，简单但不保证公平</div>
        <div>• WRR: 加权轮询，公平但可能有队头阻塞</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">应用场景</div>
        <div>• 高性能路由器/交换机的核心调度机制</div>
        <div>• 数据中心网络中的交换矩阵调度</div>
        <div>• NoC(Network on Chip)片上网络调度</div>
      </div>
    </div>
  );
}
export default CrossbarSchedulerDemo;
