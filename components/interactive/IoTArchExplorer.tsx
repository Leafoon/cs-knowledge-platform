"use client";
import { useState } from "react";

const layers = [
  {
    name: "感知层",
    en: "Perception Layer",
    color: "bg-green-600",
    devices: ["RFID标签", "温湿度传感器", "摄像头", "GPS模块"],
    protocols: ["ZigBee", "Bluetooth LE", "LoRa"],
    desc: "负责物理世界信息采集，将模拟信号转为数字数据",
  },
  {
    name: "网络层",
    en: "Network Layer",
    color: "bg-blue-600",
    devices: ["网关", "路由器", "基站", "交换机"],
    protocols: ["MQTT", "CoAP", "6LoWPAN", "NB-IoT"],
    desc: "负责数据传输与路由，连接感知层与应用层",
  },
  {
    name: "应用层",
    en: "Application Layer",
    color: "bg-purple-600",
    devices: ["云平台", "数据分析引擎", "用户界面", "AI模型"],
    protocols: ["HTTP/REST", "WebSocket", "AMQP"],
    desc: "数据处理与智能决策，提供具体行业应用服务",
  },
];

export function IoTArchExplorer() {
  const [selected, setSelected] = useState(1);
  const [showDetail, setShowDetail] = useState<"devices" | "protocols" | "desc">("desc");

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        IoT Architecture <span className="text-text-secondary text-sm">— 感知/网络/应用三层架构</span>
      </h3>
      <div className="flex gap-2 mb-4">
        {layers.map((l, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`flex-1 py-2 rounded text-white font-semibold text-sm ${l.color} ${selected === i ? "ring-2 ring-offset-2 ring-blue-400 dark:ring-offset-gray-900" : "opacity-60"}`}
          >
            {l.name}
          </button>
        ))}
      </div>
      <div className="flex gap-2 mb-3">
        {(["desc", "devices", "protocols"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setShowDetail(tab)}
            className={`px-3 py-1 rounded text-xs ${showDetail === tab ? "bg-gray-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {{ desc: "概述", devices: "设备", protocols: "协议" }[tab]}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded">
        <div className="font-semibold text-text-primary mb-2">
          {layers[selected].name} ({layers[selected].en})
        </div>
        {showDetail === "desc" && (
          <p className="text-sm text-text-secondary">{layers[selected].desc}</p>
        )}
        {showDetail === "devices" && (
          <div className="flex flex-wrap gap-2">
            {layers[selected].devices.map((d, i) => (
              <span key={i} className="px-2 py-1 rounded bg-gray-200 dark:bg-gray-700 text-sm text-text-primary">
                {d}
              </span>
            ))}
          </div>
        )}
        {showDetail === "protocols" && (
          <div className="flex flex-wrap gap-2">
            {layers[selected].protocols.map((p, i) => (
              <span key={i} className="px-2 py-1 rounded bg-blue-100 dark:bg-blue-900 text-sm text-blue-800 dark:text-blue-200">
                {p}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="mt-3 text-xs text-text-secondary text-center">
        感知层 → 网络层 → 应用层（数据自下而上流动）
      </div>
    </div>
  );
}

export default IoTArchExplorer;
