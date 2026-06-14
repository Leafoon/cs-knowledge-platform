"use client";
import { useState } from "react";

const qosLevels = [
  {
    level: 0,
    name: "QoS 0 - At most once",
    zh: "至多一次",
    desc: "发后即忘，不确认不重传，可能丢失",
    delivery: "≤1次",
    ack: false,
    retry: false,
    order: false,
    speed: "最快",
    useCase: "环境传感器数据，偶尔丢失可接受",
  },
  {
    level: 1,
    name: "QoS 1 - At least once",
    zh: "至少一次",
    desc: "确保送达但可能重复，有PUBACK确认",
    delivery: "≥1次",
    ack: true,
    retry: true,
    order: false,
    speed: "中等",
    useCase: "非关键告警，应用层需处理重复",
  },
  {
    level: 2,
    name: "QoS 2 - Exactly once",
    zh: "恰好一次",
    desc: "四步握手确保精确一次送达，最可靠但最慢",
    delivery: "=1次",
    ack: true,
    retry: true,
    order: true,
    speed: "最慢",
    useCase: "计费系统，绝不能丢失或重复",
  },
];

export function MQTTQoSExplorer() {
  const [selected, setSelected] = useState(0);
  const [simulate, setSimulate] = useState(false);
  const [step, setStep] = useState(0);

  const qos = qosLevels[selected];

  const handshakeSteps = [
    [
      ["PUBLISH", "Client → Broker"],
    ],
    [
      ["PUBLISH", "Client → Broker"],
      ["PUBACK", "Broker → Client"],
    ],
    [
      ["PUBLISH", "Client → Broker"],
      ["PUBREC", "Broker → Client"],
      ["PUBREL", "Client → Broker"],
      ["PUBCOMP", "Broker → Client"],
    ],
  ];

  const steps = handshakeSteps[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        MQTT QoS Explorer <span className="text-text-secondary text-sm">— QoS级别对比</span>
      </h3>
      <div className="flex gap-2 mb-4">
        {qosLevels.map((q, i) => (
          <button
            key={i}
            onClick={() => { setSelected(i); setStep(0); setSimulate(false); }}
            className={`flex-1 py-2 rounded text-white text-sm font-semibold ${i === 0 ? "bg-green-600" : i === 1 ? "bg-yellow-600" : "bg-red-600"} ${selected === i ? "ring-2 ring-offset-2 ring-blue-400 dark:ring-offset-gray-900" : "opacity-60"}`}
          >
            QoS {i}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">{qos.name}</div>
        <div className="text-sm text-text-secondary mb-2">{qos.zh} — {qos.desc}</div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
            <div className="text-text-secondary">投递保证</div>
            <div className="font-bold text-text-primary">{qos.delivery}</div>
          </div>
          <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
            <div className="text-text-secondary">速度</div>
            <div className="font-bold text-text-primary">{qos.speed}</div>
          </div>
          <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
            <div className="text-text-secondary">有序</div>
            <div className="font-bold text-text-primary">{qos.order ? "是" : "否"}</div>
          </div>
        </div>
        <div className="text-xs text-text-secondary mt-2">适用场景: {qos.useCase}</div>
      </div>
      <button
        onClick={() => { setSimulate(!simulate); setStep(0); }}
        className="px-3 py-1 rounded bg-purple-600 text-white text-sm mb-3"
      >
        {simulate ? "隐藏" : "显示"}握手流程
      </button>
      {simulate && (
        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
          <div className="flex gap-1 mb-3">
            {steps.map((_, i) => (
              <button
                key={i}
                onClick={() => setStep(i)}
                className={`px-2 py-1 rounded text-xs ${step === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
              >
                步骤 {i + 1}
              </button>
            ))}
          </div>
          <div className="space-y-2">
            {steps.slice(0, step + 1).map((s, i) => (
              <div
                key={i}
                className={`flex items-center gap-3 p-2 rounded text-sm ${i === step ? "bg-blue-100 dark:bg-blue-900/40" : "bg-white dark:bg-gray-900"}`}
              >
                <span className="font-mono font-bold text-text-primary w-20">{s[0]}</span>
                <span className="text-text-secondary">{s[1]}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default MQTTQoSExplorer;
