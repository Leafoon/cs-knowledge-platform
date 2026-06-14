"use client";
import { useState } from "react";

const STATES = ["正常发送", "接近拥塞", "拥塞检测", "ECN标记", "降速响应"] as const;
type State = (typeof STATES)[number];

export function ECNVisualizer() {
  const [ecnEnabled, setEcnEnabled] = useState(true);
  const [state, setState] = useState<State>("正常发送");
  const [packetLoss, setPacketLoss] = useState(false);

  const stateIndex = STATES.indexOf(state);

  const advance = () => {
    if (stateIndex < STATES.length - 1) {
      setState(STATES[stateIndex + 1]);
      if (STATES[stateIndex + 1] === "ECN标记" && !ecnEnabled) {
        setPacketLoss(true);
      }
    }
  };

  const reset = () => {
    setState("正常发送");
    setPacketLoss(false);
  };

  const stateColors: Record<State, string> = {
    "正常发送": "bg-green-500", "接近拥塞": "bg-yellow-500", "拥塞检测": "bg-orange-500",
    "ECN标记": "bg-blue-500", "降速响应": "bg-purple-500",
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">ECN显式拥塞标记流程</h3>
      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm text-text-secondary">ECN:</span>
        <button onClick={() => setEcnEnabled(!ecnEnabled)} className={`px-3 py-1.5 rounded text-sm ${ecnEnabled ? "bg-green-500 text-white" : "bg-red-500 text-white"}`}>
          {ecnEnabled ? "已启用" : "已禁用"}
        </button>
      </div>
      <div className="flex items-center gap-1 mb-4">
        {STATES.map((s, i) => (
          <div key={s} className="flex items-center">
            <div className={`px-3 py-2 rounded text-xs text-white transition-all ${i <= stateIndex ? stateColors[s] : "bg-gray-300 dark:bg-gray-700"}`}>
              {s}
            </div>
            {i < STATES.length - 1 && <div className={`w-6 h-0.5 ${i < stateIndex ? "bg-gray-500" : "bg-gray-300 dark:bg-gray-700"}`} />}
          </div>
        ))}
      </div>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        {state === "正常发送" && <p className="text-sm text-text-primary">发送方以正常速率发送数据包。路由器队列未满。</p>}
        {state === "接近拥塞" && <p className="text-sm text-text-primary">路由器队列开始增长,接近阈值。</p>}
        {state === "拥塞检测" && <p className="text-sm text-text-primary">路由器检测到队列即将溢出。</p>}
        {state === "ECN标记" && (
          <div>
            {ecnEnabled ? (
              <p className="text-sm text-text-primary">路由器在IP首部ECN字段设置CE(Congestion Experienced)标记,不丢包。</p>
            ) : (
              <p className="text-sm text-red-500">ECN未启用!路由器被迫丢弃数据包,TCP触发超时重传。</p>
            )}
          </div>
        )}
        {state === "降速响应" && (
          <p className="text-sm text-text-primary">
            {ecnEnabled ? "接收方在ACK中设置ECE标记,发送方收到后降低拥塞窗口。" : "发送方检测到丢包,执行TCP拥塞控制:窗口减半。"}
          </p>
        )}
      </div>
      <div className="flex gap-3 mb-4">
        <button onClick={advance} disabled={stateIndex >= STATES.length - 1} className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 hover:bg-blue-600 text-sm">下一步</button>
        <button onClick={reset} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm">重置</button>
      </div>
      {ecnEnabled && (
        <div className="grid grid-cols-4 gap-2 text-xs text-text-secondary">
          <div className="p-2 bg-bg-muted rounded"><strong>ECT(0)</strong>: ECN-Capable Transport</div>
          <div className="p-2 bg-bg-muted rounded"><strong>ECT(1)</strong>: ECN-Capable Transport</div>
          <div className="p-2 bg-bg-muted rounded"><strong>CE</strong>: Congestion Experienced</div>
          <div className="p-2 bg-bg-muted rounded"><strong>Not-ECT</strong>: 不支持ECN</div>
        </div>
      )}
    </div>
  );
}

export default ECNVisualizer;
