"use client";
import { useState, useMemo } from "react";

export function SDNPerformanceAnalyzer() {
  const [switches, setSwitches] = useState(20);
  const [rulesPerSwitch, setRulesPerSwitch] = useState(100);
  const [convergenceDelay, setConvergenceDelay] = useState(50);

  const metrics = useMemo(() => {
    const sdnConfigTime = switches * 2 + rulesPerSwitch * 0.1;
    const traditionalConfigTime = switches * 30 + rulesPerSwitch * switches * 0.5;
    const sdnRecovery = convergenceDelay + switches * 1;
    const traditionalRecovery = convergenceDelay * 10 + switches * 15;
    const sdnScalability = Math.log2(switches + 1) * 10;
    const traditionalScalability = switches * 5;
    return { sdnConfigTime, traditionalConfigTime, sdnRecovery, traditionalRecovery, sdnScalability, traditionalScalability };
  }, [switches, rulesPerSwitch, convergenceDelay]);

  const comparisons = [
    { label: "配置下发时间", en: "Config Push", unit: "ms", sdn: metrics.sdnConfigTime, trad: metrics.traditionalConfigTime },
    { label: "故障恢复时间", en: "Failover", unit: "ms", sdn: metrics.sdnRecovery, trad: metrics.traditionalRecovery },
    { label: "可扩展性开销", en: "Scalability", unit: "pts", sdn: metrics.sdnScalability, trad: metrics.traditionalScalability },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">SDN性能分析器</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          交换机数: <span className="text-text-primary font-mono">{switches}</span>
          <input type="range" min={5} max={100} value={switches} onChange={(e) => setSwitches(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          每台规则数: <span className="text-text-primary font-mono">{rulesPerSwitch}</span>
          <input type="range" min={10} max={1000} step={10} value={rulesPerSwitch} onChange={(e) => setRulesPerSwitch(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          收敛延迟: <span className="text-text-primary font-mono">{convergenceDelay}ms</span>
          <input type="range" min={10} max={200} value={convergenceDelay} onChange={(e) => setConvergenceDelay(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="space-y-3 mb-4">
        {comparisons.map((c, i) => {
          const maxVal = Math.max(c.sdn, c.trad, 1);
          return (
            <div key={i} className="rounded-lg border border-border-subtle bg-bg-tertiary p-3">
              <div className="text-xs font-medium text-text-primary mb-2">{c.label} <span className="text-text-tertiary">{c.en}</span></div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-sky-400 w-8">SDN</span>
                  <div className="flex-1 h-3 bg-bg-elevated rounded-full overflow-hidden">
                    <div className="h-full bg-sky-400 rounded-full transition-all" style={{ width: `${(c.sdn / maxVal) * 100}%` }} />
                  </div>
                  <span className="text-[10px] font-mono text-text-secondary w-16 text-right">{c.sdn.toFixed(0)}{c.unit}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-amber-400 w-8">传统</span>
                  <div className="flex-1 h-3 bg-bg-elevated rounded-full overflow-hidden">
                    <div className="h-full bg-amber-400 rounded-full transition-all" style={{ width: `${(c.trad / maxVal) * 100}%` }} />
                  </div>
                  <span className="text-[10px] font-mono text-text-secondary w-16 text-right">{c.trad.toFixed(0)}{c.unit}</span>
                </div>
              </div>
              <div className="text-[10px] text-emerald-500 mt-1">SDN快 {(c.trad / c.sdn).toFixed(1)}x</div>
            </div>
          );
        })}
      </div>
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1 mb-3">
        <div className="font-medium text-text-primary">SDN优势分析</div>
        <div>• 集中控制：配置统一下发，避免逐台手动配置</div>
        <div>• 快速收敛：控制器全局视图，故障秒级切换</div>
        <div>• 可扩展：控制器集群水平扩展，南向接口标准化</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">传统网络局限</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 逐设备 CLI 配置，易出错</li>
            <li>• 分布式协议收敛慢 (OSPF/BGP)</li>
            <li>• 无法快速响应业务需求</li>
          </ul>
        </div>
        <div className="p-3 rounded bg-bg-primary border border-border-subtle">
          <h4 className="text-text-primary text-xs font-medium mb-1">SDN 典型延迟</h4>
          <ul className="text-text-muted text-xs space-y-1">
            <li>• 控制器处理: 1-10ms</li>
            <li>• 南向通信: 5-50ms</li>
            <li>• 流表安装: 0.1-1ms/条</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
export default SDNPerformanceAnalyzer;
