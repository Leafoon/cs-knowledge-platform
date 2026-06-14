"use client";
import { useState } from "react";

const steps = [
  { id: "discover", name: "Discover", from: "客户端", to: "广播", color: "text-blue-600", desc: "客户端广播DHCP Discover报文，寻找DHCP服务器" },
  { id: "offer", name: "Offer", from: "服务器", to: "客户端", color: "text-green-600", desc: "服务器回应Offer报文，提供IP地址等配置" },
  { id: "request", name: "Request", from: "客户端", to: "广播", color: "text-purple-600", desc: "客户端广播Request报文，确认接受某个Offer" },
  { id: "ack", name: "ACK", from: "服务器", to: "客户端", color: "text-orange-600", desc: "服务器发送ACK确认，完成地址分配" },
];

export function DHCPSimulator() {
  const [currentStep, setCurrentStep] = useState(-1);
  const [assignedIP, setAssignedIP] = useState<string | null>(null);
  const [clientMAC] = useState("aa:bb:cc:dd:ee:ff");

  const nextStep = () => {
    const next = currentStep + 1;
    if (next >= steps.length) {
      setCurrentStep(-1);
      setAssignedIP(null);
      return;
    }
    setCurrentStep(next);
    if (next === 1) setAssignedIP("192.168.1.105");
  };

  const step = currentStep >= 0 ? steps[currentStep] : null;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DHCP 四步流程 (DORA)</h3>
      <div className="flex justify-between mb-6">
        <div className="text-center px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
          <div className="text-xs text-text-secondary">客户端</div>
          <div className="font-mono text-sm text-text-primary">{clientMAC}</div>
          {assignedIP && <div className="font-mono text-xs text-green-600 mt-1">{assignedIP}</div>}
        </div>
        <div className="flex-1 flex items-center justify-center">
          {step && (
            <div className={`font-mono text-sm ${step.color} animate-pulse`}>
              {step.from} → {step.to}
            </div>
          )}
        </div>
        <div className="text-center px-4 py-2 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-800">
          <div className="text-xs text-text-secondary">DHCP服务器</div>
          <div className="font-mono text-sm text-text-primary">192.168.1.1</div>
        </div>
      </div>
      <div className="space-y-2 mb-4">
        {steps.map((s, i) => (
          <div key={s.id} className={`flex items-center gap-3 p-3 rounded border transition-all duration-300 ${i === currentStep ? "border-blue-400 bg-blue-50 dark:bg-blue-900/20 scale-[1.02]" : i < currentStep ? "border-green-300 bg-green-50 dark:bg-green-900/10 opacity-70" : "border-border-subtle"}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${i <= currentStep ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-text-secondary"}`}>
              {i < currentStep ? "✓" : i + 1}
            </div>
            <div className="flex-1">
              <span className={`font-mono font-semibold ${s.color}`}>{s.name}</span>
              <span className="text-text-secondary text-sm ml-2">{s.desc}</span>
            </div>
          </div>
        ))}
      </div>
      <button onClick={nextStep}
        className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">
        {currentStep < 0 ? "开始DORA流程" : currentStep < 3 ? `发送 ${steps[currentStep + 1]?.name || ""}` : "重置"}
      </button>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">DHCP关键信息</div>
        <div>• 使用UDP协议，服务器端口67，客户端端口68</div>
        <div>• 租约(Lease)有有效期，到期前客户端需续约(Renew)</div>
        <div>• 支持分配: IP地址、子网掩码、默认网关、DNS服务器</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">DHCP安全</div>
        <div>• DHCP Snooping: 防止伪造DHCP服务器</div>
        <div>• 动态ARP绑定: 防止ARP欺骗</div>
        <div>• IP Source Guard: 防止IP地址伪造</div>
      </div>
    </div>
  );
}
export default DHCPSimulator;
