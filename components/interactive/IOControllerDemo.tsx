"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Settings, ArrowRight, CheckCircle } from "lucide-react";

const steps = [
  { label: "CPU发送命令", desc: "CPU将控制字写入控制寄存器，启动I/O操作", src: "CPU", dst: "CR" },
  { label: "控制器译码", desc: "控制逻辑解析命令，生成设备控制信号", src: "CR", dst: "CL" },
  { label: "设备执行操作", desc: "控制器驱动设备完成指定操作（如磁盘寻道）", src: "CL", dst: "DEV" },
  { label: "状态更新", desc: "设备完成后，控制器更新状态寄存器", src: "DEV", dst: "SR" },
  { label: "数据传输", desc: "数据通过数据寄存器在CPU和设备间传输", src: "DBR", dst: "CPU" },
  { label: "完成/中断", desc: "控制器发出完成信号或中断请求", src: "SR", dst: "CPU" },
];

const blocks = [
  { id: "CR", label: "控制寄存器", x: 100, y: 30 },
  { id: "SR", label: "状态寄存器", x: 250, y: 30 },
  { id: "DBR", label: "数据寄存器", x: 175, y: 100 },
  { id: "CL", label: "控制逻辑", x: 100, y: 170 },
  { id: "AD", label: "地址译码器", x: 250, y: 170 },
];

export function IOControllerDemo() {
  const [step, setStep] = useState(-1);
  const currentStep = step >= 0 ? steps[step] : null;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5 text-teal-400" />
        <h3 className="text-lg font-semibold">I/O 控制器演示</h3>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={() => setStep((s) => (s + 1) % steps.length)}
          className="px-4 py-1.5 bg-teal-600 rounded text-sm text-white hover:bg-teal-500"
        >
          {step < 0 ? "开始演示" : "下一步"}
        </button>
        <button
          onClick={() => setStep(-1)}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600"
        >
          重置
        </button>
        {currentStep && (
          <span className="text-sm text-gray-400">
            步骤 {step + 1}/{steps.length}: <span className="text-teal-300">{currentStep.label}</span>
          </span>
        )}
      </div>

      <div className="relative h-64 mb-4">
        {blocks.map((b) => (
          <motion.div
            key={b.id}
            className={`absolute p-2 rounded-lg border text-center text-xs ${
              currentStep?.src === b.id || currentStep?.dst === b.id
                ? "border-teal-400 bg-teal-500/20 text-teal-200"
                : "border-gray-600 bg-gray-800/50 text-gray-300"
            }`}
            style={{ left: b.x, top: b.y, width: 120 }}
            animate={{
              scale: currentStep?.dst === b.id ? [1, 1.1, 1] : 1,
            }}
            transition={{ duration: 0.5 }}
          >
            {b.label}
            {currentStep?.dst === b.id && (
              <motion.div
                className="absolute -top-1 -right-1 w-3 h-3 bg-teal-400 rounded-full"
                animate={{ scale: [0.8, 1.2, 0.8] }}
                transition={{ repeat: Infinity, duration: 0.8 }}
              />
            )}
          </motion.div>
        ))}

        <div className="absolute left-0 top-1/2 -translate-y-1/2 p-2 bg-blue-600/20 border border-blue-500 rounded-lg text-xs text-blue-300">
          CPU
        </div>
        <div className="absolute right-0 top-1/2 -translate-y-1/2 p-2 bg-green-600/20 border border-green-500 rounded-lg text-xs text-green-300">
          I/O 设备
        </div>

        {step >= 0 && (
          <motion.div
            className="absolute w-2 h-2 bg-teal-400 rounded-full"
            animate={{
              left: step % 2 === 0 ? 50 : 300,
              top: 60 + step * 20,
            }}
            transition={{ duration: 0.4 }}
          />
        )}
      </div>

      {currentStep && (
        <motion.div
          key={step}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 bg-gray-800/30 rounded-lg text-sm text-gray-300 flex items-start gap-2"
        >
          <ArrowRight className="w-4 h-4 text-teal-400 mt-0.5 shrink-0" />
          {currentStep.desc}
        </motion.div>
      )}

      {step === steps.length - 1 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-3 flex items-center gap-2 text-sm text-green-400"
        >
          <CheckCircle className="w-4 h-4" /> I/O 操作完成
        </motion.div>
      )}
    </div>
  );
}
