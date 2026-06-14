"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileCode, Play, RotateCcw, ArrowRight } from "lucide-react";

const channelProgram = [
  { cmd: "01", name: "写入", addr: "0x1000", flags: "CD=1", count: 256, desc: "从内存地址0x1000写出256字节到设备" },
  { cmd: "08", name: "控制", addr: "0x0000", flags: "CD=0", count: 4, desc: "发送4字节控制命令给设备" },
  { cmd: "02", name: "读取", addr: "0x2000", flags: "CD=1", count: 512, desc: "从设备读取512字节到内存0x2000" },
  { cmd: "01", name: "写入", addr: "0x2200", flags: "CD=1,SLI=1", count: 128, desc: "继续写出128字节，遇数据链不中断" },
  { cmd: "04", name: "查询", addr: "0x3000", flags: "CD=0", count: 8, desc: "查询设备状态到内存0x3000" },
];

const ccwFields = [
  { name: "CMD", bits: "31-24", desc: "操作码 (读/写/控制/查询)" },
  { name: "ADDR", bits: "23-8", desc: "数据内存地址" },
  { name: "FLAGS", bits: "7-1", desc: "CD/SLI/CC/Chain标志" },
  { name: "COUNT", bits: "15-0", desc: "传送字节数" },
];

export function ChannelProgramDemo() {
  const [step, setStep] = useState(-1);
  const [showFormat, setShowFormat] = useState(false);
  const current = step >= 0 ? channelProgram[step] : null;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <FileCode className="w-5 h-5 text-lime-400" />
        <h3 className="text-lg font-semibold">通道程序演示</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep((s) => Math.min(s + 1, channelProgram.length - 1))}
          className="px-4 py-1.5 bg-lime-600 rounded text-sm text-white hover:bg-lime-500 flex items-center gap-1">
          <Play className="w-3 h-3" /> {step < 0 ? "开始执行" : "下一步"}
        </button>
        <button onClick={() => { setStep(-1); }}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600">
          <RotateCcw className="w-3 h-3 inline mr-1" />重置
        </button>
        <button onClick={() => setShowFormat(!showFormat)}
          className={`px-3 py-1.5 rounded text-sm ${showFormat ? "bg-lime-600 text-white" : "bg-gray-700 text-gray-300"}`}>
          CCW格式
        </button>
      </div>

      <AnimatePresence>
        {showFormat && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} className="overflow-hidden mb-4">
            <div className="p-3 bg-gray-800/30 rounded-lg">
              <div className="text-xs text-gray-400 mb-2">通道命令字 (CCW) 格式:</div>
              <div className="flex gap-1 mb-2">
                {ccwFields.map((f) => (
                  <div key={f.name} className="flex-1 p-2 bg-gray-700 rounded text-xs text-center">
                    <div className="text-lime-300 font-mono">{f.name}</div>
                    <div className="text-gray-400">{f.bits}</div>
                  </div>
                ))}
              </div>
              <div className="space-y-1">
                {ccwFields.map((f) => (
                  <div key={f.name} className="text-xs text-gray-400">
                    <span className="text-lime-400">{f.name}</span>: {f.desc}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="text-xs text-gray-400 mb-2">通道程序:</div>
      <div className="space-y-1 mb-4">
        {channelProgram.map((ccw, i) => (
          <motion.div key={i}
            className={`flex items-center gap-3 p-2 rounded text-xs ${
              step === i ? "bg-lime-500/10 border border-lime-500" : step > i ? "bg-gray-800/20 opacity-50" : "bg-gray-800/30"
            }`}
            animate={{ x: step === i ? 4 : 0 }}
          >
            <span className="text-gray-500 w-6">#{i}</span>
            <span className="text-lime-400 font-mono w-8">{ccw.cmd}</span>
            <span className="text-gray-200 w-16">{ccw.name}</span>
            <span className="text-blue-300 font-mono w-20">{ccw.addr}</span>
            <span className="text-yellow-300 w-24">{ccw.flags}</span>
            <span className="text-gray-400">{ccw.count}B</span>
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {current && (
          <motion.div key={step}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-3 bg-gray-800/30 rounded-lg flex items-start gap-2"
          >
            <ArrowRight className="w-4 h-4 text-lime-400 mt-0.5 shrink-0" />
            <div>
              <div className="text-sm text-gray-200">
                执行: <span className="text-lime-300">{current.name}</span> → {current.desc}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                地址: {current.addr} | 数量: {current.count}字节 | 标志: {current.flags}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {step === channelProgram.length - 1 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mt-3 p-2 bg-green-500/10 rounded text-xs text-green-400">
          通道程序执行完毕，通道向CPU发送中断
        </motion.div>
      )}
    </div>
  );
}
