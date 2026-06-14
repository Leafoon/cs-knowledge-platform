"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { RefreshCw, Play, RotateCcw, CheckCircle, AlertCircle } from "lucide-react";

const steps = [
  { id: 0, label: "CPU 发出查询命令", type: "cpu", desc: "CPU执行指令读取设备状态寄存器" },
  { id: 1, label: "读取状态寄存器", type: "read", desc: "接口返回设备当前状态字" },
  { id: 2, label: "检测 Ready 位", type: "check", desc: "CPU检查状态字中的就绪标志位" },
  { id: 3, label: "设备未就绪，循环等待", type: "loop", desc: "Ready=0，CPU回到步骤0继续查询" },
  { id: 4, label: "设备就绪，传输数据", type: "transfer", desc: "Ready=1，CPU执行数据传输指令" },
  { id: 5, label: "传输完成，处理下一设备", type: "done", desc: "检查是否有其他设备需要服务" },
];

export function ProgramQueryFlow() {
  const [currentStep, setCurrentStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [loopCount, setLoopCount] = useState(0);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!autoPlay) return;
    const timer = setInterval(() => {
      setCurrentStep((s) => {
        if (s === 2 && !ready && loopCount < 2) {
          setLoopCount((c) => c + 1);
          return 3;
        }
        if (s === 3) {
          if (loopCount >= 2) setReady(true);
          return 0;
        }
        if (s === 5) {
          setAutoPlay(false);
          return s;
        }
        return s + 1;
      });
    }, 800);
    return () => clearInterval(timer);
  }, [autoPlay, ready, loopCount]);

  const handleReset = () => {
    setCurrentStep(0);
    setAutoPlay(false);
    setLoopCount(0);
    setReady(false);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <RefreshCw className="w-5 h-5 text-amber-400" />
        <h3 className="text-lg font-semibold">程序查询流程</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setAutoPlay(!autoPlay)}
          className="px-4 py-1.5 bg-amber-600 rounded text-sm text-white hover:bg-amber-500 flex items-center gap-1"
        >
          <Play className="w-3 h-3" /> {autoPlay ? "暂停" : "自动演示"}
        </button>
        <button onClick={handleReset} className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600">
          <RotateCcw className="w-3 h-3 inline mr-1" />重置
        </button>
        <div className="flex items-center gap-2 ml-auto text-xs">
          <span className="text-gray-400">查询次数: <span className="text-amber-300">{loopCount}</span></span>
          <span className={`px-2 py-0.5 rounded ${ready ? "bg-green-600/30 text-green-300" : "bg-red-600/30 text-red-300"}`}>
            {ready ? "Ready=1" : "Ready=0"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
        {steps.map((s) => (
          <motion.div
            key={s.id}
            className={`p-3 rounded-lg border text-sm ${
              currentStep === s.id
                ? "border-amber-400 bg-amber-500/10"
                : "border-gray-700 bg-gray-800/30"
            }`}
            animate={{ scale: currentStep === s.id ? 1.02 : 1 }}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className={`w-5 h-5 flex items-center justify-center rounded-full text-xs ${
                currentStep === s.id ? "bg-amber-500 text-white" : "bg-gray-700 text-gray-400"
              }`}>
                {s.id}
              </span>
              <span className={currentStep === s.id ? "text-amber-300" : "text-gray-400"}>
                {s.label}
              </span>
            </div>
            {currentStep === s.id && (
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-xs text-gray-400 ml-7">
                {s.desc}
              </motion.p>
            )}
          </motion.div>
        ))}
      </div>

      {currentStep === 5 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center gap-2 text-sm text-green-400 p-3 bg-green-500/10 rounded-lg"
        >
          <CheckCircle className="w-4 h-4" /> 数据传输完成
        </motion.div>
      )}
      {currentStep === 3 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center gap-2 text-sm text-red-400 p-3 bg-red-500/10 rounded-lg"
        >
          <AlertCircle className="w-4 h-4" /> CPU 处于忙等待状态，效率低下
        </motion.div>
      )}
    </div>
  );
}
