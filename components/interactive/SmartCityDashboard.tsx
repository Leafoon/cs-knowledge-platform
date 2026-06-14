"use client";
import { useState } from "react";

interface IoTSensor {
  name: string;
  icon: string;
  value: string;
  status: "normal" | "warning" | "critical";
  desc: string;
}

const sensors: IoTSensor[] = [
  { name: "空气质量指数", icon: "🌬", value: "AQI 42 (优)", status: "normal", desc: "PM2.5: 18μg/m³, PM10: 35μg/m³" },
  { name: "交通流量", icon: "🚗", value: "1,240 辆/小时", status: "normal", desc: "主干道平均车速 45km/h" },
  { name: "垃圾满溢", icon: "🗑", value: "3个满溢点", status: "warning", desc: "人民路、中山路、解放路垃圾箱需清理" },
  { name: "路灯状态", icon: "💡", value: "98.5% 在线", status: "normal", desc: "12盏路灯离线，3盏亮度异常" },
  { name: "水管压力", icon: "💧", value: "0.35 MPa", status: "normal", desc: "供水管网压力正常" },
  { name: "噪音监测", icon: "🔊", value: "68 dB", status: "warning", desc: "施工区域噪音超标（限值65dB）" },
  { name: "环境温度", icon: "🌡", value: "26.5°C", status: "normal", desc: "湿度 62%, 风速 2.3m/s" },
  { name: "能耗监控", icon: "⚡", value: "1,250 kWh", status: "normal", desc: "较昨日同期下降5.2%" },
];

export function SmartCityDashboard() {
  const [selected, setSelected] = useState(0);

  const statusColors = {
    normal: "bg-emerald-500/15 border-emerald-500/30 text-emerald-600 dark:text-emerald-400",
    warning: "bg-amber-500/15 border-amber-500/30 text-amber-600 dark:text-amber-400",
    critical: "bg-red-500/15 border-red-500/30 text-red-600 dark:text-red-400",
  };

  const statusLabels = { normal: "正常", warning: "警告", critical: "异常" };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">智慧城市IoT仪表盘</h3>
      <div className="grid grid-cols-4 gap-2 mb-4">
        {sensors.map((s, i) => (
          <button key={i} onClick={() => setSelected(i)}
            className={`px-2 py-2 rounded-lg border text-center transition-all ${selected === i ? "ring-2 ring-sky-400 bg-sky-500/10" : "bg-bg-tertiary border-border-subtle hover:border-sky-400/20"}`}>
            <div className="text-lg">{s.icon}</div>
            <div className="text-[10px] font-medium text-text-primary truncate">{s.name}</div>
            <div className={`text-[8px] px-1 py-0.5 rounded mt-0.5 ${statusColors[s.status]}`}>{statusLabels[s.status]}</div>
          </button>
        ))}
      </div>
      <div className={`rounded-lg border p-4 mb-4 ${statusColors[sensors[selected].status]}`}>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-2xl">{sensors[selected].icon}</span>
          <span className="text-sm font-semibold">{sensors[selected].name}</span>
          <span className={`ml-auto px-2 py-0.5 rounded text-[10px] ${statusColors[sensors[selected].status]}`}>{statusLabels[sensors[selected].status]}</span>
        </div>
        <div className="text-lg font-bold mb-1">{sensors[selected].value}</div>
        <div className="text-xs opacity-80">{sensors[selected].desc}</div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1 mb-3">
        <div className="font-medium text-text-primary">IoT技术栈</div>
        <div>• 感知层: 传感器（温湿度、PM2.5、摄像头、压力传感器）</div>
        <div>• 网络层: LoRa/NB-IoT/5G 低功耗广域网</div>
        <div>• 平台层: 物联网平台（MQTT/CoAP协议接入）</div>
        <div>• 应用层: 数据分析、智能决策、可视化大屏</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">通信协议对比</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• MQTT: 轻量发布/订阅，QoS保障</li>
            <li>• CoAP: REST风格，适合受限设备</li>
            <li>• LoRaWAN: 低功耗远距离 (&gt;10km)</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">边缘计算</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 本地预处理减少上传数据量</li>
            <li>• 低延迟实时响应 (&lt;10ms)</li>
            <li>• 隐私数据不出域</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default SmartCityDashboard;
