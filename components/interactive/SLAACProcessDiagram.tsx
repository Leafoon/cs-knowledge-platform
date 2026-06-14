"use client";
import { useState } from "react";

interface SLAACStep {
  title: string;
  en: string;
  detail: string;
  direction?: "→" | "←" | "↕";
}

const steps: SLAACStep[] = [
  { title: "生成链路本地地址", en: "Link-Local Address", detail: "主机用EUI-64从MAC地址生成接口ID，加上前缀FE80::/10，形成链路本地地址", direction: "↕" },
  { title: "DAD检测", en: "Duplicate Address Detection", detail: "发送NS（Neighbor Solicitation）报文检测地址是否冲突，等待一段时间无NA响应则地址唯一", direction: "→" },
  { title: "接收RA报文", en: "Router Advertisement", detail: "路由器周期性或响应性发送RA报文，携带前缀信息（如2001:db8:1::/64）和网络参数", direction: "←" },
  { title: "生成全局地址", en: "Global Address", detail: "主机用RA中的前缀 + EUI-64接口ID生成全局单播地址（如2001:db8:1::xx:xxFF:FExx:xxxx）", direction: "↕" },
  { title: "地址就绪", en: "Address Ready", detail: "全局地址通过DAD验证后即可使用，无需DHCP服务器参与", direction: "↕" },
];

export function SLAACProcessDiagram() {
  const [activeStep, setActiveStep] = useState(0);
  const step = steps[activeStep];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">SLAAC无状态地址自动配置</h3>
      <div className="flex items-center gap-1 mb-4">
        {steps.map((s, i) => (
          <button key={i} onClick={() => setActiveStep(i)}
            className={`flex-1 px-2 py-2 rounded-lg border text-[10px] font-medium text-center transition-all ${activeStep === i ? "bg-sky-500/20 border-sky-400/60 text-sky-700 dark:text-sky-300" : i < activeStep ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400" : "bg-bg-tertiary border-border-subtle text-text-tertiary"}`}>
            {s.title}
          </button>
        ))}
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setActiveStep(Math.max(0, activeStep - 1))} disabled={activeStep === 0}
          className="px-3 py-1 rounded-lg bg-bg-tertiary border border-border-subtle text-xs disabled:opacity-30">←</button>
        <button onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))} disabled={activeStep === steps.length - 1}
          className="px-3 py-1 rounded-lg bg-bg-tertiary border border-border-subtle text-xs disabled:opacity-30">→</button>
        <span className="text-xs text-text-tertiary ml-auto">步骤 {activeStep + 1}/{steps.length}</span>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 mb-4">
        <div className="flex items-center gap-3 mb-2">
          {step.direction && <span className="text-lg text-sky-500 font-mono">{step.direction}</span>}
          <span className="text-sm font-semibold text-text-primary">{step.title}</span>
          <span className="text-xs text-text-tertiary">{step.en}</span>
        </div>
        <div className="text-xs text-text-secondary">{step.detail}</div>
      </div>
      <div className="flex items-center gap-4 mb-4">
        <div className="flex-1 text-center px-3 py-2 rounded-lg bg-sky-500/10 border border-sky-400/30 text-xs">
          <div className="font-medium text-sky-600 dark:text-sky-400">主机</div>
          <div className="font-mono text-text-secondary mt-1">MAC: 00:1A:2B:3C:4D:5E</div>
        </div>
        <div className="text-text-tertiary">↔</div>
        <div className="flex-1 text-center px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-400/30 text-xs">
          <div className="font-medium text-emerald-600 dark:text-emerald-400">路由器</div>
          <div className="font-mono text-text-secondary mt-1">前缀: 2001:db8:1::/64</div>
        </div>
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1 mb-3">
        <div className="font-medium text-text-primary">EUI-64地址生成过程</div>
        <div>1. MAC: 00:1A:2B:FF:FE:3C:4D:5E（插入FF:FE）</div>
        <div>2. 翻转第7位（U/L位）: 02:1A:2B:FF:FE:3C:4D:5E</div>
        <div>3. 全局地址: 2001:db8:1::21A:2BFF:FE3C:4D5E</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">SLAAC vs DHCPv6</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• SLAAC: 无状态，无需服务器</li>
            <li>• DHCPv6: 有状态，可下发DNS等</li>
            <li>• 可组合使用 (无状态+其他信息)</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">隐私扩展 (RFC 4941)</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 随机生成接口ID (非EUI-64)</li>
            <li>• 定期更换地址防追踪</li>
            <li>• 现代OS默认启用</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default SLAACProcessDiagram;
