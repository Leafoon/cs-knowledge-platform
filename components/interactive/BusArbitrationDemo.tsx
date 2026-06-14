"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Network, ArrowRight, Zap, RotateCcw } from "lucide-react";

type ArbType = "daisy" | "counter" | "independent";

const ARB_INFO: Record<ArbType, { name: string; desc: string }> = {
  daisy: { name: "链式查询", desc: "总线授权信号串行传递，离控制器最近的设备优先级最高" },
  counter: { name: "计数器查询", desc: "计数器从设备编号开始轮询，优先级可动态调整" },
  independent: { name: "独立请求", desc: "每个设备有独立的请求和授权线，仲裁器直接选择" },
};

export function BusArbitrationDemo() {
  const [type, setType] = useState<ArbType>("daisy");
  const [step, setStep] = useState(0);
  const devices = ["设备0", "设备1", "设备2", "设备3"];
  const requesting = [1, 3];
  const [counter, setCounter] = useState(0);

  const getWinner = (): number => {
    switch (type) {
      case "daisy": return requesting[0];
      case "counter": {
        for (let i = 0; i < devices.length; i++) {
          const idx = (counter + i) % devices.length;
          if (requesting.includes(idx)) return idx;
        }
        return -1;
      }
      case "independent": return Math.min(...requesting);
    }
  };

  const winner = getWinner();
  const maxSteps = type === "daisy" ? 5 : type === "counter" ? 4 : 3;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Network className="w-5 h-5 text-cyan-500" />
        总线仲裁演示
      </h3>
      <div className="flex gap-2 mb-3">
        {(Object.keys(ARB_INFO) as ArbType[]).map(t => (
          <button key={t} onClick={() => { setType(t); setStep(0); setCounter(0); }}
            className={`px-3 py-1.5 rounded text-sm ${type === t ? "bg-cyan-500 text-white" : "bg-bg-subtle"}`}>
            {ARB_INFO[t].name}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-4">{ARB_INFO[type].desc}</p>
      <div className="flex gap-2 mb-4">
        <button onClick={() => { setStep(s => Math.min(s + 1, maxSteps)); if (type === "counter") setCounter(c => (c + 1) % 4); }}
          className="px-3 py-1 bg-cyan-500 text-white rounded text-sm flex items-center gap-1">
          <Zap className="w-3 h-3" /> 下一步
        </button>
        <button onClick={() => { setStep(0); setCounter(0); }}
          className="px-3 py-1 bg-bg-subtle rounded text-sm flex items-center gap-1">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>
      <div className="relative">
        <div className="flex items-center justify-between mb-4">
          <div className="px-3 py-2 bg-cyan-500/20 border border-cyan-500 rounded text-sm font-bold">
            总线控制器
          </div>
          <ArrowRight className="w-5 h-5 text-cyan-500" />
        </div>
        {type === "daisy" && (
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1 bg-cyan-500/30 relative">
              {step >= 1 && <motion.div initial={{ width: 0 }} animate={{ width: "100%" }}
                className="h-full bg-cyan-500" />}
            </div>
            {devices.map((d, i) => {
              const isRequesting = requesting.includes(i);
              const isSelected = step >= (i + 1) && i === winner;
              return (
                <motion.div key={i} animate={{ scale: isSelected ? 1.1 : 1 }}
                  className={`px-3 py-2 rounded border text-sm ${isSelected ? "border-green-500 bg-green-500/20" : isRequesting ? "border-amber-500 bg-amber-500/20" : "border-border-subtle bg-bg-subtle"}`}>
                  {d}
                  {isRequesting && <div className="text-xs text-amber-500">请求中</div>}
                  {isSelected && <div className="text-xs text-green-500">获得总线</div>}
                </motion.div>
              );
            })}
          </div>
        )}
        {type === "counter" && (
          <div>
            <div className="text-sm mb-2">计数器: <span className="font-mono font-bold text-cyan-500">{counter}</span></div>
            <div className="grid grid-cols-4 gap-2">
              {devices.map((d, i) => {
                const isRequesting = requesting.includes(i);
                const isChecking = step >= 1 && i === (counter + step - 1) % 4;
                const isSelected = isChecking && isRequesting;
                return (
                  <motion.div key={i} animate={{ scale: isSelected ? 1.1 : 1 }}
                    className={`px-3 py-3 rounded border text-center text-sm ${isSelected ? "border-green-500 bg-green-500/20" : isChecking ? "border-cyan-500 bg-cyan-500/20" : isRequesting ? "border-amber-500 bg-amber-500/20" : "border-border-subtle bg-bg-subtle"}`}>
                    {d}
                    {isRequesting && <div className="text-xs text-amber-500">请求</div>}
                    {isSelected && <div className="text-xs text-green-500">授权</div>}
                  </motion.div>
                );
              })}
            </div>
          </div>
        )}
        {type === "independent" && (
          <div>
            <div className="grid grid-cols-4 gap-2">
              {devices.map((d, i) => {
                const isRequesting = requesting.includes(i);
                const isSelected = step >= 2 && i === winner;
                return (
                  <motion.div key={i} animate={{ scale: isSelected ? 1.1 : 1 }}
                    className={`px-3 py-3 rounded border text-center text-sm ${isSelected ? "border-green-500 bg-green-500/20" : isRequesting ? "border-amber-500 bg-amber-500/20" : "border-border-subtle bg-bg-subtle"}`}>
                    {d}
                    {isRequesting && (
                      <motion.div initial={{ opacity: 0 }} animate={{ opacity: step >= 1 ? 1 : 0.3 }}
                        className="text-xs text-amber-500">REQ→</motion.div>
                    )}
                    {isSelected && (
                      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                        className="text-xs text-green-500">←GNT</motion.div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
